package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Rank3
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.split
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.argmax
import io.tlaloc.autograd.bmm
import io.tlaloc.autograd.broadcastTo
import io.tlaloc.autograd.concat
import io.tlaloc.autograd.constantMatching
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.softmax
import io.tlaloc.autograd.splat
import io.tlaloc.autograd.takeAlong
import io.tlaloc.autograd.times
import io.tlaloc.autograd.topK
import io.tlaloc.autograd.transpose

/**
 * Mixture of top-k attention (MiTA; Wen et al., arXiv 2602.01219): attention of every
 * query over `m + k` key/value pairs instead of `N`.
 *
 * Per head, with queries `Q`, keys `K` and values `V` `[N, d]`:
 *  - Landmark queries `Q̃` `[m, d]` are average pools of `Q` over [landmarks] windows of
 *    the token axis (PyTorch's adaptive average pooling), or over a grid of
 *    [landmarkGrid] windows when the tokens are an image patch grid [tokenGrid].
 *  - Each landmark `i` routes to the [topK] keys it scores highest, `Top_k(K q̃_i)`, and
 *    its value `ṽ_i = Atten(q̃_i, K, V)` summarises all of them.
 *  - Query `q` goes to the landmark it scores highest, `argmax_i q · q̃_i`, and attends
 *    with one softmax over the `m` landmarks (keys `Q̃`, values `Ṽ`, the shared expert)
 *    and that landmark's `k` keys and values (the routed expert).
 *
 * Bidirectional (no causal mask), as in the paper's vision models. The landmark and
 * routing choices are piecewise constant, so gradients flow through the attention
 * weights and the gathered keys and values, not through the choices. Costs
 * `O(N·(m + k)·d)` per head where full attention costs `O(N²·d)`.
 *
 * Input and output `[batch, seq, dModel]`. Parameter keys `q.w`, `k.w`, `v.w`, `o.w`
 * (and `.b` with biases), as [MultiHeadAttention].
 */
class MixtureOfTopKAttention(
    val q: Dense,
    val k: Dense,
    val v: Dense,
    val o: Dense,
    val numHeads: Int,
    val landmarks: Int,
    val topK: Int,
    val tokenGrid: Pair<Int, Int>? = null,
    val landmarkGrid: Pair<Int, Int>? = null,
) : TrainableLayer<MixtureOfTopKAttention> {

    val headDim: Int = q.w.dims[1] / numHeads

    init {
        require(numHeads > 0 && q.w.dims[1] == numHeads * headDim) {
            "MixtureOfTopKAttention: q projects to ${q.w.dims[1]}, not a multiple of numHeads=$numHeads"
        }
        for ((name, proj) in listOf("k" to k, "v" to v)) {
            require(proj.w.dims[1] == numHeads * headDim) {
                "MixtureOfTopKAttention: $name projects to ${proj.w.dims[1]}, expected ${numHeads * headDim}"
            }
        }
        require(o.w.dims[0] == numHeads * headDim) {
            "MixtureOfTopKAttention: o takes ${o.w.dims[0]} inputs, expected ${numHeads * headDim}"
        }
        require(landmarks > 0 && topK > 0) { "MixtureOfTopKAttention: landmarks and topK must be positive" }
        require((tokenGrid == null) == (landmarkGrid == null)) {
            "MixtureOfTopKAttention: tokenGrid and landmarkGrid come together"
        }
        require(landmarkGrid == null || landmarkGrid.first * landmarkGrid.second == landmarks) {
            "MixtureOfTopKAttention: landmarkGrid $landmarkGrid must hold $landmarks landmarks"
        }
        require(q.activation == Activation.Identity && k.activation == Activation.Identity &&
            v.activation == Activation.Identity && o.activation == Activation.Identity) {
            "MixtureOfTopKAttention: the projections must have no activation"
        }
    }

    private val children: List<Pair<String, Trainable<*>?>>
        get() = listOf("q" to q, "k" to k, "v" to v, "o" to o)

    override val parameters: List<NamedParameter> = childParameters(children)

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): MixtureOfTopKAttention {
        val u = routeUpdates("MixtureOfTopKAttention", updated, children.map { it.first })
        return MixtureOfTopKAttention(
            q.updatedWith(u["q"]), k.updatedWith(u["k"]), v.updatedWith(u["v"]), o.updatedWith(u["o"]),
            numHeads, landmarks, topK, tokenGrid, landmarkGrid,
        )
    }

    /** The `[m, seq]` averaging matrix whose rows make the landmark queries. */
    fun poolingMatrix(seq: Int): FloatArray {
        val m = landmarks
        val out = FloatArray(m * seq)
        if (tokenGrid == null) {
            for (i in 0 until m) {
                val (s, e) = adaptiveWindow(i, m, seq)
                for (t in s until e) out[i * seq + t] = 1f / (e - s)
            }
            return out
        }
        val (h, w) = tokenGrid
        val (mh, mw) = landmarkGrid!!
        require(h * w == seq) { "MixtureOfTopKAttention: tokenGrid $tokenGrid does not hold $seq tokens" }
        for (a in 0 until mh) for (b in 0 until mw) {
            val (rs, re) = adaptiveWindow(a, mh, h)
            val (cs, ce) = adaptiveWindow(b, mw, w)
            val weight = 1f / ((re - rs) * (ce - cs))
            for (r in rs until re) for (c in cs until ce) out[(a * mw + b) * seq + r * w + c] = weight
        }
        return out
    }

    @Suppress("UNCHECKED_CAST")
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank == 3) { "MixtureOfTopKAttention: input must be [batch, seq, dModel] (got ${x.dims.toList()})" }
        val batch = x.dims[0]
        val seq = x.dims[1]
        val m = landmarks
        val kk = topK
        val d = headDim
        require(m <= seq && kk <= seq) {
            "MixtureOfTopKAttention: $m landmarks and top $kk keys need at least that many tokens (got $seq)"
        }
        val g = batch * numHeads

        // [batch, seq, heads·d] → [batch·heads, seq, d]
        fun heads(t: Tracer<Shape>): Tracer<Rank3<Sym, Sym, Sym>> =
            t.reshape<Shape>(intArrayOf(batch, seq, numHeads, d)).transpose<Shape>(0, 2, 1, 3)
                .reshape(intArrayOf(g, seq, d))
        fun r3(t: Tracer<*>): Tracer<Rank3<Sym, Sym, Sym>> = t as Tracer<Rank3<Sym, Sym, Sym>>
        fun tr(t: Tracer<*>): Tracer<Rank3<Sym, Sym, Sym>> = t.transpose(0, 2, 1)

        val q3 = heads(q.forward(x, params.scoped("q")))
        val k3 = heads(k.forward(x, params.scoped("k")))
        val v3 = heads(v.forward(x, params.scoped("v")))
        val scale = 1f / kotlin.math.sqrt(d.toFloat())

        // Landmark queries [g, m, d], their key scores [g, m, seq] and routed keys [g, m, k].
        val pool: Tracer<Shape> = x.constantMatching<Shape>(poolingMatrix(seq), intArrayOf(m, seq))
            .broadcastTo(intArrayOf(g, m, seq))
        val landmarkQ = r3(r3(pool) bmm q3)
        val landmarkScores = r3(landmarkQ bmm tr(k3)) as Tracer<Shape>
        val routedKeys = landmarkScores.topK(kk)
        // The shared expert's values ṽ_i = Atten(q̃_i, K, V).
        val landmarkV = r3(r3((landmarkScores * landmarkScores.splat(scale)).softmax(-1)) bmm v3)

        // Each query's landmark scores [g, seq, m], its landmark, and that landmark's keys.
        val queryScores = r3(q3 bmm tr(landmarkQ)) as Tracer<Shape>
        val route = queryScores.argmax(-1)
        val perQuery = routedKeys.takeAlong(route).reshape<Shape>(intArrayOf(g, seq * kk))
        val kq = k3.takeAlong(perQuery).reshape<Rank3<Sym, Sym, Sym>>(intArrayOf(g * seq, kk, d))
        val vq = v3.takeAlong(perQuery).reshape<Rank3<Sym, Sym, Sym>>(intArrayOf(g * seq, kk, d))
        val routedScores = (q3.reshape<Rank3<Sym, Sym, Sym>>(intArrayOf(g * seq, 1, d)) bmm tr(kq))
            .reshape<Shape>(intArrayOf(g, seq, kk))

        // One softmax over the m landmark keys and the k routed keys.
        val logits = concat<Shape>(listOf(queryScores, routedScores), 2)
        val probs = (logits * logits.splat(scale)).softmax(-1)
        val pShared = probs.slice<Rank3<Sym, Sym, Sym>>(0, m, 2)
        val pRouted = probs.slice<Shape>(m, m + kk, 2).reshape<Rank3<Sym, Sym, Sym>>(intArrayOf(g * seq, 1, kk))
        val attended = (pShared bmm landmarkV) as Tracer<Shape> +
            (pRouted bmm vq).reshape<Shape>(intArrayOf(g, seq, d))
        val merged = attended.reshape<Shape>(intArrayOf(batch, numHeads, seq, d))
            .transpose<Shape>(0, 2, 1, 3)
            .reshape<Shape>(intArrayOf(batch, seq, numHeads * d))
        return o.forward(merged, params.scoped("o"))
    }

    companion object {
        /** Projections drawn from `N(0, initStd²)`, as [MultiHeadAttention]. */
        operator fun invoke(
            dModel: Int,
            numHeads: Int,
            landmarks: Int,
            topK: Int,
            key: RandomKey,
            headDim: Int = dModel / numHeads,
            bias: Boolean = false,
            tokenGrid: Pair<Int, Int>? = null,
            landmarkGrid: Pair<Int, Int>? = null,
            initStd: Float = 0.02f,
        ): MixtureOfTopKAttention {
            val keys = key.split(4)
            fun proj(i: Int, inp: Int, out: Int) = normalDense(keys[i], inp, out, bias, initStd)
            return MixtureOfTopKAttention(
                proj(0, dModel, numHeads * headDim),
                proj(1, dModel, numHeads * headDim),
                proj(2, dModel, numHeads * headDim),
                proj(3, numHeads * headDim, dModel),
                numHeads, landmarks, topK, tokenGrid, landmarkGrid,
            )
        }

        /** PyTorch's adaptive pooling window `i` of `parts` over `n`: `[⌊i·n/parts⌋, ⌈(i+1)·n/parts⌉)`. */
        internal fun adaptiveWindow(i: Int, parts: Int, n: Int): Pair<Int, Int> =
            (i * n) / parts to ((i + 1) * n + parts - 1) / parts
    }
}
