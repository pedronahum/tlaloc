package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.423 — the fused-adjoint family joins forward mode: EMBEDDING_GRAD,
 * the four conv adjoints, AVGPOOL2D_GRAD, MAXPOOL2D_GRAD and SCATTER_ADD get
 * tangent arms, so fwd-over-rev — the hessian composition — now runs THROUGH
 * gradient bodies containing them. Until this slice the transform refused
 * loudly (the §0.4.419 en-passant finding), which made hessians of
 * embedding/conv/pool/gather losses unreachable.
 *
 * Oracles, strongest available per surface:
 * - embedding / gather / avgpool / maxpool: the loss is (piecewise) quadratic
 *   in the param, so the HVP is analytic and the assertions are hand-computed
 *   exact values (maxpool inputs on a shuffled well-separated grid — the
 *   §0.4.363 oracle lesson — so no tangent sits near a tie).
 * - conv (both adjoints in one body, bilinear tangents): the HVP is checked
 *   against CENTRAL DIFFERENCES of the gradient at h = 1e-2 (the loss is
 *   quartic jointly in (x, w); FD error is O(h²) against O(1) curvature) AND
 *   against the symmetry identity ⟨u, Hv⟩ = ⟨v, Hu⟩, which any correct
 *   linearisation of a gradient must satisfy.
 */
class DxirFusedAdjointTangentTest {

    private val scalar = DxirType(F32, emptyList())

    // ---- embedding: loss = Σ emb(T, idx)², idx = [0,2,0,1] over vocab 3 ----

    @Test
    fun hessianThroughEmbeddingGradientBodyIsExact() {
        val idxData = floatArrayOf(0f, 2f, 0f, 1f)
        val fn = DxirBuilder.function("emb_sq") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val idx = const(idxData, DxirType(I32, listOf(4)))
            val emb = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 2)))
            val sq = op(OpKind.MUL, listOf(emb, emb), DxirType(F32, listOf(4, 2)))
            listOf(op(OpKind.SUM, listOf(sq), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val jvpOfGrad = DxirForwardTransform.apply(grad)
        val table = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val v = floatArrayOf(1f, -2f, 0.5f, 3f, -1f, 0.25f)
        val out = DxirInterpreter.evalFunction(jvpOfGrad, listOf(table, v))
        // d_table[r,:] = 2·count(r)·table[r,:] with counts [2,1,1] →
        // HVP[r,:] = 2·count(r)·v[r,:]. Exact: the whole graph is products
        // of the same floats in the same order.
        val counts = floatArrayOf(2f, 1f, 1f)
        for (i in 0 until 6) {
            val want = 2f * counts[i / 2] * v[i]
            assertTrue(
                abs(out[1][i] - want) < 1e-6f,
                "embedding HVP[$i] = ${out[1][i]}, want $want",
            )
        }
    }

    // ---- conv: loss = Σ conv2d(x, w)², both adjoints in the grad body ----

    @Test
    fun hessianThroughConvAdjointsMatchesFiniteDifferencesAndIsSymmetric() {
        val xT = DxirType(F32, listOf(1, 1, 3, 3))
        val wT = DxirType(F32, listOf(1, 1, 2, 2))
        val yT = DxirType(F32, listOf(1, 1, 2, 2))
        val fn = DxirBuilder.function("conv_sq") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(OpKind.CONV2D, listOf(x, w), yT)
            val sq = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(sq), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val jvpOfGrad = DxirForwardTransform.apply(grad)

        val x = floatArrayOf(0.3f, -0.7f, 1.1f, 0.4f, -1.3f, 0.9f, -0.2f, 0.6f, -0.5f)
        val w = floatArrayOf(0.8f, -0.4f, 0.15f, 0.55f)
        fun hvp(vx: FloatArray, vw: FloatArray): Pair<FloatArray, FloatArray> {
            val o = DxirInterpreter.evalFunction(jvpOfGrad, listOf(x, w, vx, vw))
            return o[2] to o[3]
        }
        fun gradAt(xa: FloatArray, wa: FloatArray): Pair<FloatArray, FloatArray> {
            val o = DxirInterpreter.evalFunction(grad, listOf(xa, wa))
            return o[0] to o[1]
        }

        // FD oracle: H(v) ≈ (∇(p + h·v) − ∇(p − h·v)) / 2h.
        val vx = floatArrayOf(0.6f, -0.2f, 0.9f, -1.0f, 0.3f, 0.7f, -0.8f, 0.1f, 0.45f)
        val vw = floatArrayOf(-0.35f, 0.6f, 1.0f, -0.15f)
        val (hx, hw) = hvp(vx, vw)
        val h = 1e-2f
        val (gpx, gpw) = gradAt(
            FloatArray(9) { x[it] + h * vx[it] },
            FloatArray(4) { w[it] + h * vw[it] },
        )
        val (gmx, gmw) = gradAt(
            FloatArray(9) { x[it] - h * vx[it] },
            FloatArray(4) { w[it] - h * vw[it] },
        )
        var worst = 0f
        for (i in 0 until 9) {
            val fd = (gpx[i] - gmx[i]) / (2f * h)
            worst = maxOf(worst, abs(hx[i] - fd) / maxOf(1f, abs(fd)))
        }
        for (i in 0 until 4) {
            val fd = (gpw[i] - gmw[i]) / (2f * h)
            worst = maxOf(worst, abs(hw[i] - fd) / maxOf(1f, abs(fd)))
        }
        assertTrue(worst < 1e-2f, "conv HVP vs central differences: worst rel diff $worst")

        // Symmetry oracle: ⟨u, Hv⟩ == ⟨v, Hu⟩ for an independent direction u.
        val ux = floatArrayOf(-0.5f, 0.8f, 0.2f, 0.35f, -0.9f, -0.1f, 0.75f, -0.6f, 0.25f)
        val uw = floatArrayOf(0.45f, -0.7f, 0.2f, 0.95f)
        val (hux, huw) = hvp(ux, uw)
        var uHv = 0f
        var vHu = 0f
        for (i in 0 until 9) { uHv += ux[i] * hx[i]; vHu += vx[i] * hux[i] }
        for (i in 0 until 4) { uHv += uw[i] * hw[i]; vHu += vw[i] * huw[i] }
        assertTrue(
            abs(uHv - vHu) / maxOf(1f, abs(uHv)) < 1e-4f,
            "conv HVP symmetry broken: ⟨u,Hv⟩=$uHv vs ⟨v,Hu⟩=$vHu",
        )
    }

    // ---- pooling: quadratic losses, hand-exact HVPs ----

    @Test
    fun hessianThroughPoolGradientBodiesIsExact() {
        val xT = DxirType(F32, listOf(1, 1, 4, 4))
        val yT = DxirType(F32, listOf(1, 1, 2, 2))
        val poolAttrs = mapOf("window" to listOf(2, 2), "window_strides" to listOf(2, 2))

        // avgPool: HVP[i] = Σ_{j ∈ window(i)} v[j] / 8 (2·avgᵀ(avg(v)), each
        // avg dividing by 4).
        val avgFn = DxirBuilder.function("avg_sq") {
            val x = param("x", xT)
            val y = op(OpKind.AVGPOOL2D, listOf(x), yT, attrs = poolAttrs)
            val sq = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(sq), scalar))
        }
        val v = FloatArray(16) { (it % 7 - 3) * 0.25f }
        val x = FloatArray(16) { ((it * 5) % 16 - 8) * 0.07f } // shuffled grid, well separated
        val avgJvpOfGrad = DxirForwardTransform.apply(DxirReverseTransform.apply(avgFn))
        val avgOut = DxirInterpreter.evalFunction(avgJvpOfGrad, listOf(x, v))
        for (i in 0 until 16) {
            val r = i / 4
            val c = i % 4
            val wr = (r / 2) * 2
            val wc = (c / 2) * 2
            var windowSum = 0f
            for (dr in 0 until 2) for (dc in 0 until 2) windowSum += v[(wr + dr) * 4 + (wc + dc)]
            val want = windowSum / 8f
            assertTrue(
                abs(avgOut[1][i] - want) < 1e-6f,
                "avgpool HVP[$i] = ${avgOut[1][i]}, want $want",
            )
        }

        // maxPool: HVP[i] = 2·v[i] where i is its window's argmax, else 0
        // (mask locally constant; the shuffled grid keeps every max unique
        // and well separated — the §0.4.363 oracle lesson).
        val maxFn = DxirBuilder.function("max_sq") {
            val xp = param("x", xT)
            val y = op(OpKind.MAXPOOL2D, listOf(xp), yT, attrs = poolAttrs)
            val sq = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(sq), scalar))
        }
        val maxJvpOfGrad = DxirForwardTransform.apply(DxirReverseTransform.apply(maxFn))
        val maxOut = DxirInterpreter.evalFunction(maxJvpOfGrad, listOf(x, v))
        for (i in 0 until 16) {
            val r = i / 4
            val c = i % 4
            val wr = (r / 2) * 2
            val wc = (c / 2) * 2
            var argmax = -1
            var best = Float.NEGATIVE_INFINITY
            for (dr in 0 until 2) for (dc in 0 until 2) {
                val j = (wr + dr) * 4 + (wc + dc)
                if (x[j] > best) { best = x[j]; argmax = j }
            }
            val want = if (i == argmax) 2f * v[i] else 0f
            assertTrue(
                abs(maxOut[1][i] - want) < 1e-6f,
                "maxpool HVP[$i] = ${maxOut[1][i]}, want $want",
            )
        }
    }

    // ---- gather: loss = Σ gather(x, 1)², SCATTER_ADD in the grad body ----

    @Test
    fun hessianThroughScatterAddGradientBodyIsExact() {
        val fn = DxirBuilder.function("gather_sq") {
            val x = param("x", DxirType(F32, listOf(3, 2)))
            val idx = const(1, DxirType(I32, emptyList()))
            val g = op(OpKind.GATHER, listOf(x, idx), DxirType(F32, listOf(2)))
            val sq = op(OpKind.MUL, listOf(g, g), DxirType(F32, listOf(2)))
            listOf(op(OpKind.SUM, listOf(sq), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val jvpOfGrad = DxirForwardTransform.apply(grad)
        val x = floatArrayOf(0.5f, -1.5f, 2.25f, 0.75f, -0.25f, 1.0f)
        val v = floatArrayOf(1f, 2f, -3f, 0.5f, 4f, -1f)
        val out = DxirInterpreter.evalFunction(jvpOfGrad, listOf(x, v))
        // d_x = 2·x on row 1, zero elsewhere → HVP = 2·v on row 1, zero rows
        // 0 and 2 — and those zeros must be EXACT (the scatter pattern is
        // fixed by the integer index).
        val want = floatArrayOf(0f, 0f, 2f * v[2], 2f * v[3], 0f, 0f)
        for (i in 0 until 6) {
            assertTrue(
                abs(out[1][i] - want[i]) < 1e-6f,
                "gather HVP[$i] = ${out[1][i]}, want ${want[i]}",
            )
        }
    }
}
