/**
 * §0.4.437 — Phase F1: the COMPILER-ROUTE differentiation contract (amended
 * decision 3 of MODEL_LAYER_PLAN.md). The model's forward traces ONCE through
 * the `:autograd` Tracer into a real `DxirFunction` — one param per (input
 * tensor + parameter tensor), arbitrary arity — and `DxirReverseTransform`,
 * the compiler's AD, produces the gradient function. `:nn` writes NO gradient
 * math: the registry rules own every adjoint, and execution is a backend
 * choice ([DxirInterpreter] on host here; the coarsened/StableHLO/PjrtSession
 * GPU lane is F8's integration).
 *
 * REJECTED alternatives, recorded per the house pattern:
 * - The runtime value-tape (`Backward.kt`) as the model route — Pedro's veto,
 *   §0.4.436: host-only, interpreted, and it sidelines the compiler stack that
 *   IS the product. It stays a debugging fallback, untouched.
 * - Packing parameters through the fixed-arity `grad {}` intrinsics — the 1–4
 *   ceiling is a lambda-surface property; flattening a model into it is
 *   unnatural and caps arity for no reason the IR has.
 * - Separate primal and gradient evaluations per step — the transform's
 *   `includeForward = true` mode returns `(loss, *grads)` from ONE function,
 *   so a training step is a single `evalFunction` call. The standalone primal
 *   is still captured and exposed: it is the prediction path, the route pin,
 *   and F8's emission artifact.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform

/** One training-step evaluation: the scalar loss and the per-parameter gradients. */
class StepResult(
    val loss: Float,
    /** Keyed exactly like the model's [Trainable.parameters], in that order. */
    val gradients: Map<String, DTensor<*, F32>>,
    /**
     * Gradients w.r.t. the model INPUTS, positionally. The reverse transform
     * returns one gradient per captured param — inputs included — so these come
     * for free; useful for adversarial/saliency work, ignorable for training.
     */
    val inputGradients: List<DTensor<*, F32>>,
)

/**
 * The captured (primal, gradient) pair for one model STRUCTURE — the cacheable
 * unit. Capture once, [run] per step: `run` only re-binds tensor VALUES
 * positionally (params are `DxirParam`s, not baked constants), so it stays
 * valid while the model's structure — layer list, parameter keys, every dim —
 * is unchanged. Retracing is only needed when the structure changes.
 *
 * Caching contract (recorded for F8): F1 keys nothing automatically — the
 * caller holds the `CapturedStep` across steps, and the [valueAndGradients]
 * convenience honestly retraces per call. F8's GPU integration adds the
 * structure-keyed cache in front of the compiled-executable lane (the §0.4.307
 * amortization), where re-capture actually costs something.
 */
class CapturedStep internal constructor(
    /** The traced forward: params = inputs ++ parameters, one scalar return. */
    val primal: DxirFunction,
    /** `DxirReverseTransform.apply(primal, includeForward = true)`: returns (loss, *grads). */
    val gradient: DxirFunction,
    val parameterKeys: List<String>,
    val inputCount: Int,
) {
    fun run(model: Trainable<*>, inputs: List<DTensor<*, F32>>): StepResult {
        require(inputs.size == inputCount) {
            "CapturedStep.run: captured for $inputCount input(s), got ${inputs.size}"
        }
        val params = model.parameters
        val keys = params.map { it.key }
        require(keys == parameterKeys) {
            "CapturedStep.run: model structure changed — captured keys $parameterKeys, got $keys; re-capture"
        }
        val values = inputs.map { it.hostF32() } + params.map { it.tensor.hostF32() }
        val outs = DxirInterpreter.evalFunction(gradient, values)

        fun tensorAt(outIndex: Int, paramIndex: Int): DTensor<*, F32> {
            val dims = primal.params[paramIndex].type.dims.toIntArray()
            return DTensor<Shape, F32>(HostF32Storage(outs[outIndex]), dims, F32)
        }

        val inputGrads = (0 until inputCount).map { tensorAt(1 + it, it) }
        val gradients = LinkedHashMap<String, DTensor<*, F32>>(parameterKeys.size)
        for (j in parameterKeys.indices) {
            gradients[parameterKeys[j]] = tensorAt(1 + inputCount + j, inputCount + j)
        }
        return StepResult(outs[0][0], gradients, inputGrads)
    }
}

/**
 * The capture step. Runs [model]'s forward in Tracer land with every input and
 * every parameter as a differentiable tape leaf (positional order: inputs
 * first, then parameters in [Trainable.parameters] order), applies [lossFn] to
 * the model output (must reduce to a SCALAR — `.sum()` / `.mean()` are the
 * traced spellings), converts the tape via `Tape.toDxirFunction`, and applies
 * [DxirReverseTransform] once.
 */
fun <M> capture(
    model: M,
    inputs: List<DTensor<*, F32>>,
    name: String = "model",
    lossFn: (Tracer<Shape>) -> Tracer<*>,
): CapturedStep where M : Layer, M : Trainable<M> {
    require(inputs.size == 1) {
        "capture v1: Layer is single-input (DiffKT's LayerSingleInput fold); got ${inputs.size} inputs"
    }
    val params = model.parameters
    val keys = params.map { it.key }
    require(keys.toSet().size == keys.size) { "duplicate parameter keys: $keys" }

    @Suppress("UNCHECKED_CAST")
    val primal = captureN(inputs + params.map { it.tensor }, name) { leaves ->
        val inputTracers = leaves.subList(0, inputs.size)
        val paramTracers: Map<String, Tracer<Shape>> =
            keys.withIndex().associate { (j, key) -> key to leaves[inputs.size + j] }
        val out = model.forward(inputTracers[0], Params { key ->
            paramTracers[key] ?: error("capture: forward asked for unknown parameter key '$key' (known: $keys)")
        })
        val loss = lossFn(out)
        require(loss.dims.isEmpty()) {
            "capture: lossFn must reduce to a scalar (got dims ${loss.dims.toList()}) — end with .sum() or .mean()"
        }
        loss as Tracer<Shape>
    }
    val gradient = DxirReverseTransform.apply(primal, includeForward = true)
    return CapturedStep(primal, gradient, keys, inputs.size)
}

/**
 * The one-shot convenience: capture + run. Honestly retraces per call — hold
 * the [CapturedStep] from [capture] yourself to amortize (see its KDoc for the
 * F8 caching contract).
 */
fun <M> valueAndGradients(
    model: M,
    inputs: List<DTensor<*, F32>>,
    lossFn: (Tracer<Shape>) -> Tracer<*>,
): StepResult where M : Layer, M : Trainable<M> =
    capture(model, inputs, lossFn = lossFn).run(model, inputs)
