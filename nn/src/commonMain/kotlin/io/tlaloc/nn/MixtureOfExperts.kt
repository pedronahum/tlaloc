package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.bmm
import io.tlaloc.autograd.broadcastTo
import io.tlaloc.autograd.cast
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.div
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.neg
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.raggedDot
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.softmax
import io.tlaloc.autograd.splat
import io.tlaloc.autograd.takeAlong
import io.tlaloc.autograd.times
import io.tlaloc.autograd.topK
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.split

/**
 * A mixture of SwiGLU experts, trainable: a router sends each row to its [topK] experts by
 * softmax probability (equal probabilities: the lower expert first), and the row's output is
 * their outputs weighted by the chosen probabilities, renormalized to sum to 1 (Qwen3-MoE and
 * Mixtral routing, the function [io.tlaloc.ir.OpKind.MOE_EXPERTS] serves).
 *
 * The (row, expert) pairs are sorted by expert, so that each expert's rows are consecutive, and
 * the experts run as two [raggedDot]s: `[gate | up] = x · gateUp[e]`, then `down[e]` applied to
 * `silu(gate) · up`. XLA:TPU runs a ragged dot as its grouped matmul. Gradients reach the
 * experts, the router (through the chosen probabilities) and the input; which experts are
 * chosen is piecewise constant in the router's logits and has no gradient.
 *
 * [gateUp] is `[experts, dModel, 2·hidden]` (gate columns first), [down] `[experts, hidden,
 * dModel]`. The sort keys are f32 integers, so `experts · pairs + pairs` must stay below 2²⁴,
 * with pairs the rows times [topK].
 *
 * Parameter keys `router.w`, `gateUp`, `down`.
 */
class MixtureOfExperts(
    val router: Dense,
    val gateUp: DTensor<*, F32>,
    val down: DTensor<*, F32>,
    val topK: Int,
) : TrainableLayer<MixtureOfExperts> {

    val experts: Int get() = gateUp.dims[0]
    val dModel: Int get() = gateUp.dims[1]
    val hidden: Int get() = down.dims[1]

    init {
        require(router.b == null && router.w.dims.size == 2) { "MixtureOfExperts: the router is a [dModel, experts] Dense without bias" }
        require(gateUp.dims.size == 3 && gateUp.dims[2] % 2 == 0) {
            "MixtureOfExperts: gateUp must be [experts, dModel, 2·hidden] (got ${gateUp.dims.toList()})"
        }
        require(down.dims.toList() == listOf(experts, gateUp.dims[2] / 2, dModel)) {
            "MixtureOfExperts: down must be [$experts, ${gateUp.dims[2] / 2}, $dModel] (got ${down.dims.toList()})"
        }
        require(router.w.dims.toList() == listOf(dModel, experts)) {
            "MixtureOfExperts: the router must be [$dModel, $experts] (got ${router.w.dims.toList()})"
        }
        require(topK in 1..experts) { "MixtureOfExperts: topK $topK must be in 1..$experts" }
    }

    override val parameters: List<NamedParameter> =
        childParameters(listOf("router" to router)) + listOf(NamedParameter("gateUp", gateUp), NamedParameter("down", down))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): MixtureOfExperts {
        val own = setOf("gateUp", "down")
        val u = routeUpdates("MixtureOfExperts", updated.filterKeys { it !in own }, listOf("router"))
        return MixtureOfExperts(router.updatedWith(u["router"]), updated["gateUp"] ?: gateUp, updated["down"] ?: down, topK)
    }

    @Suppress("UNCHECKED_CAST")
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank >= 2 && x.dims[x.rank - 1] == dModel) { "MixtureOfExperts: input must be [..., $dModel] (got ${x.dims.toList()})" }
        val h = dModel
        val k = topK
        val e = experts
        val rows = x.size / h
        val pairs = rows * k
        require(e.toLong() * pairs + pairs < (1L shl 24)) {
            "MixtureOfExperts: $rows rows × $k experts over $e experts is past the f32 sort keys' 2^24"
        }
        val flat: Tracer<Shape> = if (x.rank == 2) x else x.reshape(intArrayOf(rows, h))
        fun r2(t: Tracer<*>) = t as Tracer<Rank2<Sym, Sym>>
        fun r3(t: Tracer<*>) = t as Tracer<Rank3<Sym, Sym, Sym>>

        // Routing: the top-k probabilities, renormalized.
        val probs = router.forward(flat, params.scoped("router")).softmax(-1)
        val top = probs.topK(k)
        val chosen = probs.takeAlong(top)
        val rowSums = (r2(chosen) matmul r2(flat.constant<Shape>(FloatArray(k * k) { 1f }, intArrayOf(k, k)))) as Tracer<Shape>
        val weights = chosen / rowSums

        // The pairs (row r, its i-th expert) at r·k + i, sorted by expert, then by pair.
        val ids: Tracer<Shape> = top.cast(F32).reshape(intArrayOf(pairs, 1))
        val key = ids * ids.splat(pairs.toFloat()) + flat.constant(FloatArray(pairs) { it.toFloat() }, intArrayOf(pairs, 1))
        val order = key.neg().reshape<Shape>(intArrayOf(1, pairs)).topK(pairs)
        val back = order.cast(F32).neg().topK(pairs)

        // Each expert's pair count: ones · [id == expert].
        val d: Tracer<Shape> = ids.broadcastTo<Shape>(intArrayOf(pairs, e)) -
            flat.constant<Shape>(FloatArray(e) { it.toFloat() }, intArrayOf(1, e)).broadcastTo(intArrayOf(pairs, e))
        val isExpert = (d.splat(1f) - d).relu() * (d.splat(1f) + d).relu()
        val ones = key.reshape<Shape>(intArrayOf(1, pairs)).splat(1f)
        val sizes = (r2(ones) matmul r2(isExpert)).reshape<Shape>(intArrayOf(e)).cast(I32)

        // The experts on the sorted rows.
        val xs = flat.reshape<Shape>(intArrayOf(rows, 1, h)).broadcastTo<Shape>(intArrayOf(rows, k, h))
            .reshape<Shape>(intArrayOf(1, pairs, h)).takeAlong(order).reshape<Shape>(intArrayOf(pairs, h))
        val gu = xs.raggedDot(params["gateUp"], sizes)
        val a = Activation.Silu.apply(gu.slice(0, hidden, 1)) * gu.slice<Shape>(hidden, 2 * hidden, 1)
        val y = a.raggedDot(params["down"], sizes)

        // Back to pair order, weighted and summed over each row's experts.
        val yPairs = y.reshape<Shape>(intArrayOf(1, pairs, h)).takeAlong(back).reshape<Shape>(intArrayOf(rows, k, h))
        val out = r3(weights.reshape<Shape>(intArrayOf(rows, 1, k))) bmm r3(yPairs)
        return out.reshape(x.dims.copyOf())
    }

    companion object {
        /** [experts] SwiGLU experts of width [hidden], [topK] per row, normal(0, [initStd]) weights. */
        operator fun invoke(dModel: Int, hidden: Int, experts: Int, topK: Int, key: RandomKey, initStd: Float = 0.02f): MixtureOfExperts {
            val keys = key.split(3)
            return MixtureOfExperts(
                normalDense(keys[0], dModel, experts, false, initStd),
                gaussianInit(keys[1], intArrayOf(experts, dModel, 2 * hidden), variance = initStd * initStd),
                gaussianInit(keys[2], intArrayOf(experts, hidden, dModel), variance = initStd * initStd),
                topK,
            )
        }
    }
}
