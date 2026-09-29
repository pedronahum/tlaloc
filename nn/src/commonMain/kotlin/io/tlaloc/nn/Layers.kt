/**
 * The simple layers, in DiffKT's exact semantics: `Dense` (`activation(x matmul W + b)`, bias broadcast over
 * rows, `b` omitted from the trainables when `bias = false`), `Flatten`
 * (`flatten(startDim = 1)` — the batch axis survives), `ReluLayer`, and the
 * `Activation` objects Dense composes post-op.
 *
 * All forwards are TRACE spellings (see the compiler-route contract in
 * `Training.kt`): matmul,
 * broadcast-add and relu record onto the `:autograd` tape, `Tape.toDxirFunction`
 * reproduces them in the captured graph, and `DxirReverseTransform` owns every
 * gradient. Zero adjoint math in this file.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.split
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.sigmoid
import io.tlaloc.autograd.splat
import io.tlaloc.autograd.times
import io.tlaloc.autograd.tanh

/**
 * DiffKT's `Activation` objects (Relu / Identity / Sigmoid / Tanh) plus
 * Silu and GeluTanh, composed by Dense AFTER the affine op. Each is a pure trace spelling over
 * the existing elementwise ops — the registry rules differentiate them.
 */
sealed class Activation {
    abstract fun apply(x: Tracer<Shape>): Tracer<Shape>

    object Identity : Activation() {
        override fun apply(x: Tracer<Shape>): Tracer<Shape> = x
    }

    object Relu : Activation() {
        override fun apply(x: Tracer<Shape>): Tracer<Shape> = x.relu()
    }

    object Sigmoid : Activation() {
        override fun apply(x: Tracer<Shape>): Tracer<Shape> = x.sigmoid()
    }

    object Tanh : Activation() {
        override fun apply(x: Tracer<Shape>): Tracer<Shape> = x.tanh()
    }

    /** `x · sigmoid(x)`, PyTorch's `SiLU` (the gate of Llama's SwiGLU MLP). */
    object Silu : Activation() {
        override fun apply(x: Tracer<Shape>): Tracer<Shape> = x * x.sigmoid()
    }

    /**
     * `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, PyTorch's
     * `GELU(approximate="tanh")` (GPT-2's activation). PyTorch's default
     * `GELU()` uses erf, which has no op here; the two differ by up to about 5e-4.
     */
    object GeluTanh : Activation() {
        override fun apply(x: Tracer<Shape>): Tracer<Shape> {
            val inner = (x + x * x * x * x.splat(0.044715f)) * x.splat(0.7978845608f)
            return x * x.splat(0.5f) * (inner.tanh() + x.splat(1f))
        }
    }
}

/**
 * DiffKT's `Dense(numInputs, numOutputs, random, bias = true, activation)`:
 * `W [numInputs, numOutputs]` and `b [numOutputs]`, forward
 * `activation(x matmul W + b)` with `b` broadcast over the batch rows
 * (`broadcast_dimensions = [1]` — the row broadcast, whose reverse is
 * the axis-0 SUM every bias gradient is). When [b] is null (DiffKT's
 * `bias = false`), the add is skipped entirely and only `w` is trainable —
 * DiffKT keeps a `FloatScalar.ZERO` placeholder out of `trainables`; we keep
 * no placeholder at all, same maths.
 *
 * Input is `[..., numInputs]` at any rank ≥ 2, as in DiffKT and PyTorch: the
 * leading axes are flattened into rows, multiplied, and restored, so a
 * `[batch, seq, numInputs]` activation comes back `[batch, seq, numOutputs]`.
 *
 * The randomly-initialized form is the companion `invoke` — DiffKT's default
 * init for BOTH W and b is `uniform(±sqrt(1/numInputs))`. The key
 * discipline (shared by every `:nn` layer): the layer splits
 * its key ONCE into one child per parameter tensor in declaration order —
 * `split(2)[0]` → w, `split(2)[1]` → b — and `bias = false` still consumes the
 * same split so the drawn W is identical with and without a bias.
 *
 * With a [lora] adapter the forward is
 * `activation(x matmul W + b + scale·(dropout(x) matmul A) matmul B)` and the
 * adapter adds the parameters `lora_A` and `lora_B` after `w` and `b` (see
 * `Lora.kt`).
 */
class Dense(
    val w: DTensor<*, F32>,
    val b: DTensor<*, F32>?,
    val activation: Activation,
    /** A LoRA adapter on this layer, or null. */
    val lora: LoraAdapter?,
) : TrainableLayer<Dense> {

    constructor(
        w: DTensor<*, F32>,
        b: DTensor<*, F32>?,
        activation: Activation = Activation.Identity,
    ) : this(w, b, activation, null)

    init {
        require(w.dims.size == 2) {
            "Dense: w must be rank-2 [numInputs, numOutputs] (got dims ${w.dims.toList()})"
        }
        if (b != null) {
            require(b.dims.size == 1 && b.dims[0] == w.dims[1]) {
                "Dense: b must be rank-1 [numOutputs=${w.dims[1]}] (got dims ${b.dims.toList()})"
            }
        }
        if (lora != null) {
            require(lora.a.dims[0] == w.dims[0] && lora.b.dims[1] == w.dims[1]) {
                "Dense: LoRA A ${lora.a.dims.toList()} and B ${lora.b.dims.toList()} do not fit " +
                    "w ${w.dims.toList()} (A must be [${w.dims[0]}, r], B [r, ${w.dims[1]}])"
            }
        }
    }

    override val parameters: List<NamedParameter> =
        (if (b != null) listOf(NamedParameter("w", w), NamedParameter("b", b))
        else listOf(NamedParameter("w", w))) +
            (if (lora != null) listOf(NamedParameter(LoraAdapter.A_KEY, lora.a), NamedParameter(LoraAdapter.B_KEY, lora.b))
            else emptyList())

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): Dense {
        val known = parameters.mapTo(HashSet()) { it.key }
        val unknown = updated.keys - known
        require(unknown.isEmpty()) { "Dense.withParameters: unknown keys $unknown (known: $known)" }
        val adapter = lora?.let {
            if (LoraAdapter.A_KEY in updated || LoraAdapter.B_KEY in updated) {
                it.withTensors(updated[LoraAdapter.A_KEY] ?: it.a, updated[LoraAdapter.B_KEY] ?: it.b)
            } else it
        }
        return Dense(updated["w"] ?: w, if (b != null) updated["b"] ?: b else null, activation, adapter)
    }

    /** This layer with [adapter] (replacing any it has). */
    fun withLora(adapter: LoraAdapter): Dense = Dense(w, b, activation, adapter)

    /**
     * This layer with its adapter folded into the weight, `W + scale·A·B`,
     * and no adapter; itself when it has none. Dropout is not part of the
     * merged layer, as it is not part of the adapter's inference form.
     */
    fun merged(): Dense {
        val adapter = lora ?: return this
        val base = w.hostF32()
        val delta = adapter.delta()
        val out = FloatArray(base.size) { base[it] + delta[it] }
        return Dense(DTensor<Shape, F32>(HostF32Storage(out), w.dims.copyOf(), F32), b, activation, null)
    }

    @Suppress("UNCHECKED_CAST")
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank >= 2) {
            "Dense: input must be [..., numInputs] with rank >= 2 (got dims ${x.dims.toList()})"
        }
        val inner = x.dims[x.rank - 1]
        require(inner == w.dims[0]) {
            "Dense: input width $inner does not match numInputs ${w.dims[0]}"
        }
        val rows = x.size / inner
        val flat: Tracer<Shape> = if (x.rank == 2) x else x.reshape(intArrayOf(rows, inner))
        val xm = flat as Tracer<Rank2<Sym, Sym>>
        val wm = params["w"] as Tracer<Rank2<Sym, Sym>>
        val z = xm matmul wm
        val zb = if (b != null) z + (params["b"] as Tracer<Rank1<Sym>>) else z
        val za: Tracer<Shape> =
            if (lora != null) (zb as Tracer<Shape>) + lora.forward(xm, params) else zb as Tracer<Shape>
        val y = activation.apply(za)
        return if (x.rank == 2) y else y.reshape(x.dims.copyOf().also { it[it.size - 1] = w.dims[1] })
    }

    companion object {
        /** The DiffKT constructor surface: draw W (and b) from [key] per the class KDoc. */
        operator fun invoke(
            numInputs: Int,
            numOutputs: Int,
            key: RandomKey,
            bias: Boolean = true,
            activation: Activation = Activation.Identity,
        ): Dense {
            require(numInputs > 0 && numOutputs > 0) {
                "Dense: numInputs=$numInputs and numOutputs=$numOutputs must be positive"
            }
            val bound = kotlin.math.sqrt(1f / numInputs)
            val keys = key.split(2)
            val w = uniformInit(keys[0], intArrayOf(numInputs, numOutputs), -bound, bound)
            val b = if (bias) uniformInit(keys[1], intArrayOf(numOutputs), -bound, bound) else null
            return Dense(w, b, activation)
        }
    }
}

/**
 * DiffKT's `Flatten` (an object there too): `input.flatten(startDim = 1)` —
 * the batch axis survives, everything after collapses, `[N, d1, …, dk]` →
 * `[N, d1·…·dk]`. Rank-2 input is already flat and passes through untouched
 * (no RESHAPE recorded — DiffKT's flatten is a no-op view there as well);
 * rank ≥ 3 records the `reshape` trace spelling and `ReshapeRule` hands the
 * gradient back in the input's shape.
 */
object Flatten : Layer {
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank >= 2) {
            "Flatten: input must have a batch axis and at least one more (rank >= 2, got dims ${x.dims.toList()})"
        }
        if (x.rank == 2) return x
        var rest = 1
        for (i in 1 until x.rank) rest *= x.dims[i]
        return x.reshape(intArrayOf(x.dims[0], rest))
    }
}

/** DiffKT's `ReluLayer` object: `relu(input)`, any rank, not trainable. */
object ReluLayer : Layer {
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> = x.relu()
}
