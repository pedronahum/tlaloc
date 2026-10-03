package io.tlaloc.ir.passes

import io.tlaloc.ir.MoeExpertsAttrs
import kotlin.math.exp

/**
 * The reference walk of [io.tlaloc.ir.OpKind.MOE_EXPERTS], shared by
 * [DxirInterpreter] and [DxirInterpreterF64]. Per row:
 * ```
 *   p    = softmax(routerLogits)
 *   top  = the top_k largest p (equal values: the lower expert first)
 *   w    = p_top / sum(p_top)
 *   y    = sum_j w_j * down[e_j] (silu(g) * u),   [g | u] = gateUp[e_j] x
 * ```
 * transformers' `Qwen3_5MoeTopKRouter` and `Qwen3_5MoeExperts`. Sums are in
 * double; [round] is applied to every stored value, and [roundWeights] to the
 * activation fed to the down projection (the weights' dtype).
 */
internal object MoeWalk {

    fun experts(
        p: MoeExpertsAttrs.Parsed,
        x: DoubleArray,
        logits: DoubleArray,
        gateUp: DoubleArray,
        down: DoubleArray,
        round: (Double) -> Double,
        roundWeights: (Double) -> Double,
    ): DoubleArray {
        val (r, h, e, inter, k) = p
        val y = DoubleArray(r * h)
        val probs = DoubleArray(e)
        val gu = DoubleArray(2 * inter)
        val act = DoubleArray(inter)
        val acc = DoubleArray(h)
        for (row in 0 until r) {
            var max = Double.NEGATIVE_INFINITY
            for (j in 0 until e) max = maxOf(max, logits[row * e + j])
            var sum = 0.0
            for (j in 0 until e) {
                probs[j] = exp(logits[row * e + j] - max)
                sum += probs[j]
            }
            for (j in 0 until e) probs[j] = round(probs[j] / sum)
            val top = (0 until e).sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it }).take(k)
            var topSum = 0.0
            for (j in top) topSum += probs[j]
            acc.fill(0.0)
            for (ex in top) {
                val w = round(probs[ex] / topSum)
                for (o in 0 until 2 * inter) {
                    var s = 0.0
                    val base = (ex * 2 * inter + o) * h
                    for (c in 0 until h) s += gateUp[base + c] * x[row * h + c]
                    gu[o] = round(s)
                }
                for (o in 0 until inter) {
                    val g = gu[o]
                    act[o] = roundWeights(round(g / (1.0 + exp(-g)) * gu[inter + o]))
                }
                for (o in 0 until h) {
                    var s = 0.0
                    val base = (ex * h + o) * inter
                    for (c in 0 until inter) s += down[base + c] * act[c]
                    acc[o] += w * round(s)
                }
            }
            for (o in 0 until h) y[row * h + o] = round(acc[o])
        }
        return y
    }
}
