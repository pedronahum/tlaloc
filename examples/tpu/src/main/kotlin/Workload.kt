/**
 * The graphs this example runs, built by hand out of Tlaloc IR ops.
 *
 * Nothing here knows what a TPU is. A Tlaloc program is a `DxirFunction`: a
 * list of typed ops with no backend in it. The device shows up exactly once,
 * in Main.kt, when that function is handed to a `PjrtSession`. That separation
 * is the reason a TPU example could be written before a TPU existed to run it.
 */
import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.uniformFloats
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.sqrt

/**
 * One scaled-dot-product attention block — the computation a TPU exists for.
 *
 *     scores = Q · Kᵀ          [seq, seq]
 *     probs  = softmax(scores / √dim)
 *     out    = probs · V       [seq, dim]
 *
 * `K` enters already transposed (`[dim, seq]`) so the graph is four ops and the
 * layout question is the caller's, not the graph's. The scale rides in as a
 * splat constant: `const(x, type)` with a scalar value means "this value over
 * that whole shape", which both the interpreter and the emitter understand.
 *
 * The matmuls here are the shape a TPU's systolic array is built for — square
 * `[seq, dim] · [dim, seq]`, then `[seq, seq] · [seq, dim]`. At the default
 * size that is ~17M multiply-accumulates; `--seq 2048 --dim 512` makes it ~4G
 * and starts to be a real unit of TPU work.
 */
object Attention {

    /**
     * The block itself, emitted into whatever function is being built. Declares
     * the three parameters in order (Q, Kᵀ, V) and returns the output node, so
     * [forward] and [loss] are the same graph with and without a reduction on
     * the end — and are therefore guaranteed to agree.
     */
    private fun DxirBuilder.attentionOut(seq: Int, dim: Int) = run {
        val sType = DxirType(F32, listOf(seq, seq))
        val oType = DxirType(F32, listOf(seq, dim))
        val q = param("Q", DxirType(F32, listOf(seq, dim)))
        val kt = param("Kt", DxirType(F32, listOf(dim, seq)))
        val v = param("V", oType)
        val scores = op(OpKind.MATMUL, listOf(q, kt), sType)
        val scale = const(1f / sqrt(dim.toFloat()), sType)
        val scaled = op(OpKind.MUL, listOf(scores, scale), sType)
        val probs = op(OpKind.SOFTMAX, listOf(scaled), sType)
        op(OpKind.MATMUL, listOf(probs, v), oType)
    }

    /** Q·Kᵀ → scale → softmax → ·V. Returns the attention output. */
    fun forward(seq: Int, dim: Int): DxirFunction =
        DxirBuilder.function("attention_f32") { listOf(attentionOut(seq, dim)) }

    /**
     * The same block reduced to a scalar: `loss = Σ attention(Q, Kᵀ, V)`.
     *
     * A scalar is what a gradient needs. Main.kt hands this to
     * `DxirReverseTransform`, which walks the graph backwards and emits the
     * derivative — including the softmax Jacobian-vector product, which nobody
     * in this example (or in Tlaloc's model layer) ever wrote by hand.
     */
    fun loss(seq: Int, dim: Int): DxirFunction =
        DxirBuilder.function("attention_loss_f32") {
            listOf(op(OpKind.SUM, listOf(attentionOut(seq, dim)), DxirType(F32, emptyList())))
        }

    /**
     * Q, Kᵀ and V drawn from the threefry counter-based PRNG, centred on zero.
     *
     * There is no global RNG anywhere in Tlaloc: a draw is a pure function of
     * an explicit key, which is why the numbers below are the same on this
     * machine, on the TPU VM, and inside JAX. Act 3 of this example turns that
     * property into a measurement.
     */
    fun inputs(seq: Int, dim: Int): List<FloatArray> {
        fun centred(key: RandomKey, n: Int) = FloatArray(n).also { out ->
            val u = uniformFloats(key, n)
            for (i in 0 until n) out[i] = u[i] - 0.5f
        }
        return listOf(
            centred(RandomKey(0x5EED_0001, 0x7A11_0C00), seq * dim),  // Q
            centred(RandomKey(0x5EED_0002, 0x7A11_0C00), dim * seq),  // Kᵀ
            centred(RandomKey(0x5EED_0003, 0x7A11_0C00), seq * dim),  // V
        )
    }
}

/**
 * The RNG stream itself, as a one-op program.
 *
 * `RNG_UNIFORM` does not become a device RNG call. Since §0.4.422 it lowers to
 * explicit StableHLO integer ops — threefry's ARX rounds, a shift, a bit-or
 * against the exponent of 1.0, and one exact subtract — so the draw is integer
 * arithmetic plus one exactly-representable float subtraction. Integer
 * arithmetic cannot legally differ between backends. That is the *argument*;
 * Act 3 is the *measurement*, and it is the only claim in this file that would
 * be genuinely new information on a TPU.
 */
object Rng {
    fun graph(dims: List<Int>, key: RandomKey): DxirFunction =
        DxirBuilder.function("rng_uniform") {
            listOf(
                op(
                    OpKind.RNG_UNIFORM, emptyList(), DxirType(F32, dims),
                    attrs = mapOf("key0" to key.k0, "key1" to key.k1, "dims" to dims),
                ),
            )
        }

    /** (shape, key) pairs: an even length, an odd one (the end-pad lane of the
     * two-at-a-time threefry block), a rank-2 shape, and a long run. Keys
     * include one with the sign bit set, because a 32-bit key is not a small
     * non-negative integer. */
    val sweep: List<Pair<List<Int>, RandomKey>> = listOf(
        listOf(6) to RandomKey(42, 7),
        listOf(5) to RandomKey(-1, 12345),
        listOf(2, 3) to RandomKey(0x13198A2E, 0x03707344),
        // 0x85A308D3 has its top bit set: a key is 32 raw bits, not a small
        // non-negative integer, so it is written as a Long and narrowed.
        listOf(4096) to RandomKey(0x243F6A88, 0x85A308D3.toInt()),
    )
}

/**
 * The bf16 narrowing probe: f32 values chosen so that a rounding mistake
 * cannot hide. Ties in both directions, one-ulp neighbours of a tie, signed
 * zeros, infinities, the overflow boundary, and subnormal flushes.
 *
 * Tlaloc's host helper rounds f32→bf16 with round-half-to-even. Act 4 asks the
 * TPU to do the same narrowing and compares the *bit patterns* — not the
 * values — so a device that rounds ties away from zero is caught on lane 1.
 */
object Bf16Probe {
    val values: FloatArray = floatArrayOf(
        Float.fromBits(0x3F808000),          // tie, even mantissa keeps → 1.0
        Float.fromBits(0x3F818000),          // tie, odd mantissa → rounds up to even
        Float.fromBits(0x3F808001),          // just above the tie → up
        Float.fromBits(0x3F817FFF),          // just below the tie → down
        Float.fromBits(0xBF808000.toInt()),  // negative tie → toward zero
        Float.fromBits(0xBF818000.toInt()),  // negative tie → away, to even
        0f, -0f,
        Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
        Float.MAX_VALUE,                     // RNE overflow → +inf
        Float.fromBits(0x7F7F7FFF),          // below the overflow tie → max finite
        Float.fromBits(0x00000001),          // min f32 subnormal → +0
        Float.fromBits(0x80000001.toInt()),  // ...and the sign survives the flush
        1.5f, 256f, 1.0078125f, 3.14159265f, -257f,
    )
}
