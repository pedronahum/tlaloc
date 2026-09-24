/**
 * The COMPILER-ROUTE differentiation contract. The model's forward traces ONCE through
 * the `:autograd` Tracer into a real `DxirFunction` — one param per (input
 * tensor + parameter tensor), arbitrary arity — and `DxirReverseTransform`,
 * the compiler's AD, produces the gradient function. `:nn` writes NO gradient
 * math: the registry rules own every adjoint, and execution is a backend
 * choice ([DxirInterpreter] on host here; the coarsened/StableHLO/PjrtSession
 * GPU lane is a separate backend).
 *
 * REJECTED alternatives:
 * - A runtime value-tape reverse walk as the model route: host-only,
 *   interpreted, and it bypasses the compiler stack.
 * - Packing parameters through the fixed-arity `grad {}` intrinsics — the 1–4
 *   ceiling is a lambda-surface property; flattening a model into it is
 *   unnatural and caps arity for no reason the IR has.
 * - Separate primal and gradient evaluations per step — the transform's
 *   `includeForward = true` mode returns `(loss, *grads)` from ONE function,
 *   so a training step is a single `evalFunction` call. The standalone primal
 *   is still captured and exposed: it is the prediction path and the
 *   function a GPU backend emits.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.hostI32
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.cast
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform

/**
 * The mixed-precision training convention, stated once:
 * **MASTER WEIGHTS IN F32, COMPUTE IN BF16, LOSS AND GRADIENTS IN F32.**
 *
 * [MIXED_BF16] is a CAPTURE-level property, not a model property: the trace
 * injects `Tracer.cast(BF16)` on every F32 input and parameter leaf at the
 * trace boundary (I32 index inputs pass through — an integer has no
 * precision), the model's forward then RECORDS in bf16 (the tape's dtype
 * propagation in `Tape.op`), and the model OUTPUT is cast back to F32
 * before the loss function runs — so the loss computes in f32, and the
 * gradient function's per-parameter outputs are f32 BY CONSTRUCTION (the
 * params are f32-typed; CastRule's straight-through adjoint widens every
 * upstream back through the boundary casts). The model's stored
 * tensors are never touched: they ARE the f32 master weights, the optimizer
 * updates them in f32, and only the traced graph ever sees bf16.
 *
 * NO LOSS SCALING — deliberately, and this is WHY bf16 beats fp16 for
 * training: bf16 keeps f32's full 8-bit exponent (it is the top
 * half of binary32), so gradients cannot underflow the way fp16's 5-bit
 * exponent makes them; the entire GradScaler apparatus fp16 AMP needs
 * (scale, unscale, inf-check, skip-step) has nothing to protect against.
 *
 * REJECTED alternatives:
 * - A `MixedPrecision(model)` WRAPPER layer — precision is a property of one
 *   capture, not of the model structure: the same model must capture both
 *   ways (the f32 capture is the oracle the mixed one certifies against),
 *   and a wrapper would either fake `Trainable` (its "parameters" are the
 *   inner model's) or intercept `Params` lookups per-key at forward time —
 *   both spellings put the cast decision further from the trace boundary
 *   the casts belong to.
 * - CASTING THE LOSS instead of the model output — the loss reduction
 *   (mean/MSE) would then compute in bf16 and round a large-N accumulation
 *   at 8 mantissa bits; casting at the model-output boundary is the
 *   standard AMP shape (torch.autocast computes reductions in f32).
 * - BF16 MASTER WEIGHTS — the optimizer update `w − α·g` with α·g typically
 *   1e-3× smaller than w needs more than 8 mantissa bits or updates round
 *   to zero; f32 masters are the whole point of the mixed convention.
 */
enum class Precision {
    /** The default: everything traces and runs in f32. */
    F32,

    /** f32 master weights, bf16 compute between the boundary casts, f32 loss + gradients. */
    MIXED_BF16,
}

/** One training-step evaluation: the scalar loss and the per-parameter gradients. */
class StepResult(
    val loss: Float,
    /** Keyed exactly like the model's [Trainable.parameters], in that order. */
    val gradients: Map<String, DTensor<*, F32>>,
    /**
     * Gradients w.r.t. the model INPUTS, positionally. The reverse transform
     * returns one gradient per captured param — inputs included — so these come
     * for free; useful for adversarial/saliency work, ignorable for training.
     * An INTEGER input (an embedding-index batch) is
     * non-differentiable by dtype: its slot is the transform's
     * ZEROS_LIKE structural zero, surfaced here as an F32 all-zeros tensor of
     * the input's shape — expected, not a bug.
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
 * Caching contract: nothing is keyed automatically — the
 * caller holds the `CapturedStep` across steps, and the [valueAndGradients]
 * convenience retraces per call. A structure-keyed cache belongs in front of
 * a compiled-executable GPU lane, where re-capture actually costs something.
 */
class CapturedStep internal constructor(
    /** The traced forward: params = inputs ++ parameters, one scalar return. */
    val primal: DxirFunction,
    /** `DxirReverseTransform.apply(primal, includeForward = true)`: returns (loss, *grads). */
    val gradient: DxirFunction,
    val parameterKeys: List<String>,
    val inputCount: Int,
) {
    /**
     * The captured gradient function, printed as compilable Kotlin source over the
     * `:core` host twins. What comes back is the EXACT program [run] executes
     * through [DxirInterpreter] — the same ops, as `val`-per-op Kotlin a user
     * can read, compile and call (certified bit-identical against the
     * interpreter by golden tests).
     */
    fun gradSource(): String = io.tlaloc.ir.render.toKotlinSource(gradient)

    /** [gradSource]'s primal twin: the captured forward, printed. */
    fun primalSource(): String = io.tlaloc.ir.render.toKotlinSource(primal)

    fun run(model: Trainable<*>, inputs: List<DTensor<*, *>>): StepResult {
        require(inputs.size == inputCount) {
            "CapturedStep.run: captured for $inputCount input(s), got ${inputs.size}"
        }
        val params = model.parameters
        val keys = params.map { it.key }
        require(keys == parameterKeys) {
            "CapturedStep.run: model structure changed — captured keys $parameterKeys, got $keys; re-capture"
        }
        val values = inputs.map { hostValues(it) } + params.map { it.tensor.hostF32() }
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
    inputs: List<DTensor<*, *>>,
    name: String = "model",
    precision: Precision = Precision.F32,
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
        // §0.4.458 (G1d) — the MIXED_BF16 trace boundary (see [Precision]):
        // every F32 leaf gets ONE injected cast to bf16 (params eagerly here,
        // so a key looked up twice shares one CAST node); I32 index leaves
        // pass through untouched.
        fun toCompute(t: Tracer<Shape>): Tracer<Shape> =
            if (precision == Precision.MIXED_BF16 && t.dtype == io.tlaloc.core.F32)
                t.cast(io.tlaloc.core.BF16)
            else t
        val inputTracers = leaves.subList(0, inputs.size).map(::toCompute)
        val paramTracers: Map<String, Tracer<Shape>> =
            keys.withIndex().associate { (j, key) -> key to toCompute(leaves[inputs.size + j]) }
        val out = model.forward(inputTracers[0], Params { key ->
            paramTracers[key] ?: error("capture: forward asked for unknown parameter key '$key' (known: $keys)")
        })
        // The output boundary: back to f32 BEFORE the loss, so the loss
        // reduction accumulates in f32 (the standard AMP shape — see the
        // rejected cast-the-loss alternative on [Precision]).
        val lossIn =
            if (precision == Precision.MIXED_BF16 && out.dtype == io.tlaloc.core.BF16)
                out.cast(io.tlaloc.core.F32)
            else out
        val loss = lossFn(lossIn)
        require(loss.dims.isEmpty()) {
            "capture: lossFn must reduce to a scalar (got dims ${loss.dims.toList()}) — end with .sum() or .mean()"
        }
        require(loss.dtype == io.tlaloc.core.F32) {
            "capture: the loss must be F32-typed (got ${loss.dtype.name}) — mixed precision keeps " +
                "the loss in f32; the bf16 region ends at the model-output cast"
        }
        loss as Tracer<Shape>
    }
    val gradient = DxirReverseTransform.apply(primal, includeForward = true)
    return CapturedStep(primal, gradient, keys, inputs.size)
}

/**
 * The one-shot convenience: capture + run. Retraces per call — hold
 * the [CapturedStep] from [capture] yourself to amortize (see its KDoc for the
 * caching contract).
 */
fun <M> valueAndGradients(
    model: M,
    inputs: List<DTensor<*, *>>,
    precision: Precision = Precision.F32,
    lossFn: (Tracer<Shape>) -> Tracer<*>,
): StepResult where M : Layer, M : Trainable<M> =
    capture(model, inputs, precision = precision, lossFn = lossFn).run(model, inputs)

/**
 * The interpreter environment is float-typed for every dtype (its
 * EMBEDDING arms read indices via `toInt()`): an I32 input binds as its
 * float-encoded values, exact below 2²⁴ (the trace leaf asserted the cap; the
 * per-step re-bind asserts it again — new step, new indices).
 */
private fun hostValues(t: DTensor<*, *>): FloatArray = when (t.dtype) {
    F32 -> {
        @Suppress("UNCHECKED_CAST")
        (t as DTensor<*, F32>).hostF32()
    }
    io.tlaloc.core.I32 -> {
        @Suppress("UNCHECKED_CAST")
        val ints = (t as DTensor<*, io.tlaloc.core.I32>).hostI32()
        FloatArray(ints.size) { i ->
            val v = ints[i]
            require(v > -16_777_216 && v < 16_777_216) {
                "CapturedStep.run: I32 input value $v at position $i exceeds the float-encoding " +
                    "exactness cap 2^24"
            }
            v.toFloat()
        }
    }
    else -> error("CapturedStep.run: unsupported input dtype ${t.dtype.name} (F32 and I32 only)")
}
