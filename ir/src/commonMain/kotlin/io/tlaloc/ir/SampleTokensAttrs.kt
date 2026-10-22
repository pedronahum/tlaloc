package io.tlaloc.ir

import io.tlaloc.core.F32
import io.tlaloc.core.I32

/**
 * [OpKind.SAMPLE_TOKENS]'s operands, one parser for every layer that touches the kind, and the
 * choice both interpreters run (triton/kernels/sample.cu computes the same, bit for bit).
 * ```
 *   0 logits      [N, V]  f32
 *   1 draft       [N]     i32: the token a greedy draft proposes for this row, or -1
 *   2 temperature [N]     f32: <= 0 is greedy
 *   3 topP        [N]     f32: (0, 1]
 *   4 topK        [N]     i32: <= 0 keeps every token
 *   5 seed        [N]     i32
 *   6 position    [N]     i32
 *   -> token      [N]     i32
 * ```
 * Greedy rows take the argmax (the lowest id among equal logits). A sampled row gives each token
 * an integer weight `W = floor(2^20 2^(((x - max) / t) log2 e))` ([exp2Fixed]); orders the tokens
 * by logit (descending) then id; keeps the first `topK`, then the shortest prefix of those whose
 * weights reach `topP` of theirs. A draft (a point mass, so exact speculative sampling needs only
 * the target's weights) is accepted when `u0 M < W(d) 2^24`, M the kept weight; otherwise the
 * token is drawn from the kept tokens other than the draft, by their weights in id order with
 * `u1`. `u0`, `u1` are 24-bit uniforms of (seed, position) ([uniforms]): nothing carries random
 * state from one step to the next, and a row's token does not depend on the other rows.
 */
object SampleTokensAttrs {

    data class Parsed(val rows: Int, val vocab: Int)

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.SAMPLE_TOKENS) { "$layer: SampleTokensAttrs.parse called on ${op.op}" }
        require(op.operands.size == 7) {
            "$layer: SAMPLE_TOKENS takes 7 operands (logits, draft, temperature, topP, topK, seed, position), got ${op.operands.size}"
        }
        val ts = op.operands.map { it.type }
        require(ts[0].rank == 2 && ts[0].dtype == F32) { "$layer: SAMPLE_TOKENS logits must be f32 [N, V], got ${ts[0]}" }
        val (n, v) = ts[0].dims
        val want = listOf(I32, F32, F32, I32, I32, I32)
        for (i in 1..6) {
            require(ts[i] == DxirType(want[i - 1], listOf(n))) {
                "$layer: SAMPLE_TOKENS operand $i must be ${want[i - 1].name} [$n], got ${ts[i]}"
            }
        }
        require(op.attrs.isEmpty()) { "$layer: SAMPLE_TOKENS takes no attributes, got ${op.attrs.keys}" }
        require(op.types.size == 1 && op.type == DxirType(I32, listOf(n))) { "$layer: SAMPLE_TOKENS returns i32 [$n], got ${op.types}" }
        return Parsed(n, v)
    }

    /** `floor(2^20 2^z)` for z in (-21, 0], 0 below: sample.cu's `Exp2Fixed`, the same f32 steps. */
    fun exp2Fixed(z: Float): Long {
        if (!(z > -21f)) return 0
        val fi = kotlin.math.floor(z)
        val f = z - fi
        var p = 1.5403530393381606e-4f
        p = p * f + 1.3333558146428443e-3f
        p = p * f + 9.6181291076284772e-3f
        p = p * f + 5.5504108664821580e-2f
        p = p * f + 2.4022650695910071e-1f
        p = p * f + 6.9314718055994531e-1f
        p = p * f + 1f
        val e = 20 + fi.toInt()
        if (e < 0) return 0
        return (p * Float.fromBits((e + 127) shl 23)).toLong()  // exact: a power of two
    }

    private fun splitMix64(x0: Long): Long {
        var x = x0 + -0x61c8864680b583ebL
        x = (x xor (x ushr 30)) * -0x40a7b892e31b1a47L
        x = (x xor (x ushr 27)) * -0x6b2fb644ecceee15L
        return x xor (x ushr 31)
    }

    /** The two 24-bit uniforms of (seed, position): sample.cu's `Uniforms`. */
    fun uniforms(seed: Int, position: Int): Pair<Long, Long> {
        val key = (seed.toLong() shl 32) or (position.toLong() and 0xFFFFFFFFL)
        return (splitMix64(key) ushr 40) to (splitMix64(key xor 0x5851F42D4C957F2DL) ushr 40)
    }

    /** The token of one row. */
    fun choose(x: FloatArray, from: Int, v: Int, draft: Int, t: Float, topP: Float, topK: Int, seed: Int, position: Int): Int {
        var argmax = 0
        for (i in 1 until v) if (x[from + i] > x[from + argmax]) argmax = i
        if (!(t > 0f)) return argmax
        val m = x[from + argmax]
        val log2e = 1.4426950408889634f
        fun w(i: Int): Long = exp2Fixed(((x[from + i] - m) / t) * log2e)
        val order = (0 until v).sortedWith(compareByDescending<Int> { x[from + it] }.thenBy { it })
        val k = if (topK <= 0 || topK >= v) v else topK
        var sumK = 0L
        for (j in 0 until k) sumK += w(order[j])
        val p24 = if (topP >= 1f) 1L shl 24 else (maxOf(topP, 0f) * 16777216f).toLong()
        var kept = k
        if (p24 < (1L shl 24)) {
            var need = (p24 * sumK + (1L shl 24) - 1) ushr 24
            if (need == 0L) need = 1
            var cum = 0L
            for (j in 0 until k) {
                cum += w(order[j])
                if (cum >= need) { kept = j + 1; break }
            }
        }
        val inKept = BooleanArray(v)
        var mass = 0L
        for (j in 0 until kept) {
            inKept[order[j]] = true
            mass += w(order[j])
        }
        val dKept = draft in 0 until v && inKept[draft]
        val wd = if (dKept) w(draft) else 0L
        val (u0, u1) = uniforms(seed, position)
        if (mass == 0L) return argmax
        if (dKept && u0 * mass < (wd shl 24)) return draft
        val m2 = mass - wd
        if (m2 == 0L) return argmax
        val target = (u1 * m2) ushr 24
        var cum = 0L
        for (i in 0 until v) {
            if (!inKept[i] || i == draft) continue
            cum += w(i)
            if (cum > target) return i
        }
        return argmax
    }

    fun reference(
        p: Parsed, logits: FloatArray, draft: IntArray, temperature: FloatArray, topP: FloatArray, topK: IntArray,
        seed: IntArray, position: IntArray,
    ): IntArray = IntArray(p.rows) { r ->
        choose(logits, r * p.vocab, p.vocab, draft[r], temperature[r], topP[r], topK[r], seed[r], position[r])
    }
}
