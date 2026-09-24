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
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.split
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.sigmoid
import io.tlaloc.autograd.tanh

/**
 * DiffKT's `Activation` objects: Relu / Identity / Sigmoid / Tanh,
 * composed by Dense AFTER the affine op. Each is a pure trace spelling over
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
 * Input is rank-2 `[batch, numInputs]` in v1. DiffKT accepts rank ≥ 2 (its
 * matmul broadcasts leading axes); the traced matmul spelling is rank-2/rank-3
 * today and the rank-3 Dense input form is not supported (it is refused, not
 * silently computed differently).
 *
 * The randomly-initialized form is the companion `invoke` — DiffKT's default
 * init for BOTH W and b is `uniform(±sqrt(1/numInputs))`. The key
 * discipline (shared by every `:nn` layer): the layer splits
 * its key ONCE into one child per parameter tensor in declaration order —
 * `split(2)[0]` → w, `split(2)[1]` → b — and `bias = false` still consumes the
 * same split so the drawn W is identical with and without a bias.
 */
class Dense(
    val w: DTensor<*, F32>,
    val b: DTensor<*, F32>?,
    val activation: Activation = Activation.Identity,
) : TrainableLayer<Dense> {

    init {
        require(w.dims.size == 2) {
            "Dense: w must be rank-2 [numInputs, numOutputs] (got dims ${w.dims.toList()})"
        }
        if (b != null) {
            require(b.dims.size == 1 && b.dims[0] == w.dims[1]) {
                "Dense: b must be rank-1 [numOutputs=${w.dims[1]}] (got dims ${b.dims.toList()})"
            }
        }
    }

    override val parameters: List<NamedParameter> =
        if (b != null) listOf(NamedParameter("w", w), NamedParameter("b", b))
        else listOf(NamedParameter("w", w))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): Dense {
        val known = if (b != null) setOf("w", "b") else setOf("w")
        val unknown = updated.keys - known
        require(unknown.isEmpty()) { "Dense.withParameters: unknown keys $unknown (known: $known)" }
        return Dense(updated["w"] ?: w, if (b != null) updated["b"] ?: b else null, activation)
    }

    @Suppress("UNCHECKED_CAST")
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank == 2) {
            "Dense v1: input must be rank-2 [batch, numInputs] (got dims ${x.dims.toList()}) — " +
                "rank-3 input is not supported; reshape to rank 2 first"
        }
        require(x.dims[1] == w.dims[0]) {
            "Dense: input width ${x.dims[1]} does not match numInputs ${w.dims[0]}"
        }
        val xm = x as Tracer<Rank2<Sym, Sym>>
        val wm = params["w"] as Tracer<Rank2<Sym, Sym>>
        val z = xm matmul wm
        val zb = if (b != null) z + (params["b"] as Tracer<Rank1<Sym>>) else z
        return activation.apply(zb as Tracer<Shape>)
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
