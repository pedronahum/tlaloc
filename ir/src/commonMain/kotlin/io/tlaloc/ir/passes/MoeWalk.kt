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
 * activation fed to the down projection (the weights' dtype). With
 * [MoeExpertsAttrs.Parsed.actFp4], x's row and each pair's silu(g) * u are quantized
 * ([io.tlaloc.ir.Nvfp4MatmulAttrs.quantizeActivations]) and their row scales multiply the sums.
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
        gateUpScale: DoubleArray? = null,
        downScale: DoubleArray? = null,
        routerBias: DoubleArray? = null,
        // A shared expert every row takes: its gate/up [2I, H], down [H, I], and each row's weight.
        sharedGateUp: DoubleArray? = null,
        sharedDown: DoubleArray? = null,
        sharedWeight: DoubleArray? = null,
        // Expert parallel: this op holds experts [expertOffset, expertOffset + e) of the router's eR.
        expertOffset: Int = 0,
    ): DoubleArray {
        val (r, h, e, inter, k) = p
        val eR = p.routedExperts
        val y = DoubleArray(r * h)
        val probs = DoubleArray(eR)
        val gu = DoubleArray(2 * inter)
        val act = DoubleArray(inter)
        val acc = DoubleArray(h)
        for (row in 0 until r) {
            val top: List<Int>
            val weight: (Int) -> Double
            if (p.sigmoidBias) {
                // DeepSeek's noaux_tc: choose by sigmoid + bias, weigh by the sigmoid, normalized and scaled.
                for (j in 0 until eR) probs[j] = round(1.0 / (1.0 + exp(-logits[row * eR + j])))
                val bias = routerBias ?: error("MoeWalk: sigmoid routing needs the router bias")
                top = (0 until eR).sortedWith(compareByDescending<Int> { round(probs[it] + bias[it]) }.thenBy { it }).take(k)
                var topSum = 0.0
                for (j in top) topSum += probs[j]
                weight = { ex -> round(round(probs[ex] / (topSum + 1e-20)) * p.routedScale) }
            } else {
                var max = Double.NEGATIVE_INFINITY
                for (j in 0 until eR) max = maxOf(max, logits[row * eR + j])
                var sum = 0.0
                for (j in 0 until eR) {
                    probs[j] = exp(logits[row * eR + j] - max)
                    sum += probs[j]
                }
                for (j in 0 until eR) probs[j] = round(probs[j] / sum)
                top = (0 until eR).sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it }).take(k)
                var topSum = 0.0
                for (j in top) topSum += probs[j]
                weight = { ex -> round(probs[ex] / topSum) }
            }
            acc.fill(0.0)
            // W4A4: the row's codes x scales, and its scale.
            val xRow: DoubleArray
            val xBase: Int
            var gx = 1.0
            if (p.actFp4) {
                xRow = DoubleArray(h)
                gx = io.tlaloc.ir.Nvfp4MatmulAttrs.quantizeActivations(FloatArray(h) { x[row * h + it].toFloat() }, 0, h, xRow, 0).toDouble()
                xBase = 0
            } else {
                xRow = x
                xBase = row * h
            }
            // The chosen experts this op holds, by their local index, with their routing weights.
            val local = top.filter { it - expertOffset in 0 until e }.map { (it - expertOffset) to weight(it) }
            val pairsOf = local + if (sharedWeight != null) listOf(-1 to sharedWeight[row]) else emptyList()
            for ((ex, w) in pairsOf) {
                // ex -1: the shared expert's matrices.
                val gu0 = if (ex < 0) sharedGateUp!! else gateUp
                val dn0 = if (ex < 0) sharedDown!! else down
                val exi = if (ex < 0) 0 else ex
                for (o in 0 until 2 * inter) {
                    var s = 0.0
                    val base = (exi * 2 * inter + o) * h
                    for (c in 0 until h) s += gu0[base + c] * xRow[xBase + c]
                    gu[o] = round(if (gateUpScale == null) s * gx else round(s) * gateUpScale[exi * 2 * inter + o])
                }
                var gh = 1.0
                if (p.actFp4) {
                    val hf = FloatArray(inter) { o -> round(gu[o] / (1.0 + exp(-gu[o])) * gu[inter + o]).toFloat() }
                    gh = io.tlaloc.ir.Nvfp4MatmulAttrs.quantizeActivations(hf, 0, inter, act, 0).toDouble()
                } else {
                    for (o in 0 until inter) {
                        val g = gu[o]
                        act[o] = roundWeights(round(g / (1.0 + exp(-g)) * gu[inter + o]))
                    }
                }
                for (o in 0 until h) {
                    var s = 0.0
                    val base = (exi * h + o) * inter
                    for (c in 0 until inter) s += dn0[base + c] * act[c]
                    acc[o] += w * round(if (downScale == null) s * gh else round(s) * downScale[exi * h + o])
                }
            }
            for (o in 0 until h) y[row * h + o] = round(acc[o])
        }
        return y
    }
}
