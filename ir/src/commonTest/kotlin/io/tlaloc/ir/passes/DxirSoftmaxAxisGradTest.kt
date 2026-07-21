package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.368 — Phase A3 (DiffKT parity): softmax over a NON-default axis, and
 * `logSoftmax` as the `LOG(SOFTMAX(x))` composition, differentiate at IR
 * level. The default-axis softmax VJP/JVP were pinned §0.4.359/361; this
 * pins the `axis=0` routing (the attr actually threading through the reduce)
 * and that log∘softmax composes into a correct logSoftmax gradient.
 */
class DxirSoftmaxAxisGradTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun softmaxAxis0GradientMatchesAnalytic() {
        // loss = Σ w ⊙ softmax(x, axis=0) over a [2,3] matrix (columns
        // normalise). dx = y ⊙ (w − Σ_axis0(w⊙y)); dw = y.
        val t = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("softmax_ax0") {
            val x = param("x", t)
            val w = param("w", t)
            val y = op(OpKind.SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 0))
            val weighted = op(OpKind.MUL, listOf(w, y), t)
            listOf(op(OpKind.SUM, listOf(weighted), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.1f, -0.4f, 0.9f, 1.5f, 0f, -2f)
        val w = floatArrayOf(1f, 2f, 3f, -1f, 0.5f, 2f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))

        // Reference in doubles: softmax down each column (axis 0), 2 rows × 3 cols.
        val dx = DoubleArray(6)
        val dw = DoubleArray(6)
        for (col in 0 until 3) {
            val idx = intArrayOf(col, col + 3)
            val mx = idx.maxOf { x[it].toDouble() }
            val e = idx.map { exp(x[it] - mx) }
            val s = e.sum()
            val y = e.map { it / s }
            val dot = idx.indices.sumOf { w[idx[it]] * y[it] }
            for (r in idx.indices) {
                dx[idx[r]] = y[r] * (w[idx[r]] - dot)
                dw[idx[r]] = y[r]
            }
        }
        var maxAbs = 0.0
        for (i in 0 until 6) {
            maxAbs = maxOf(maxAbs, abs(out[0][i] - dx[i]))
            maxAbs = maxOf(maxAbs, abs(out[1][i] - dw[i]))
        }
        assertTrue(maxAbs <= 1e-5, "softmax(axis=0) gradient diverges: max|diff|=$maxAbs")
    }

    @Test
    fun logSoftmaxCompositionGradientMatchesAnalytic() {
        // logSoftmax lowers to LOG(SOFTMAX(x, axis=1)). loss = Σ w ⊙ ls.
        // d ls_i/d x_j = δ_ij − y_j ⇒ dx_j = w_j − y_j·Σ_axis(w); dw = ls.
        val t = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("logsoftmax") {
            val x = param("x", t)
            val w = param("w", t)
            val y = op(OpKind.SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1))
            val ls = op(OpKind.LOG, listOf(y), t)
            val weighted = op(OpKind.MUL, listOf(w, ls), t)
            listOf(op(OpKind.SUM, listOf(weighted), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.1f, -0.4f, 0.9f, 1.5f, 0f, -2f)
        val w = floatArrayOf(1f, 2f, 3f, -1f, 0.5f, 2f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))

        val dx = DoubleArray(6)
        val dw = DoubleArray(6)
        for (row in 0 until 2) {
            val base = row * 3
            val mx = (0 until 3).maxOf { x[base + it].toDouble() }
            val e = DoubleArray(3) { exp(x[base + it] - mx) }
            val s = e.sum()
            val y = DoubleArray(3) { e[it] / s }
            val logS = ln(s)
            val wsum = (0 until 3).sumOf { w[base + it].toDouble() }
            for (j in 0 until 3) {
                dx[base + j] = w[base + j] - y[j] * wsum
                dw[base + j] = x[base + j] - mx - logS
            }
        }
        var maxAbs = 0.0
        for (i in 0 until 6) {
            maxAbs = maxOf(maxAbs, abs(out[0][i] - dx[i]))
            maxAbs = maxOf(maxAbs, abs(out[1][i] - dw[i]))
        }
        assertTrue(maxAbs <= 1e-5, "logSoftmax gradient diverges: max|diff|=$maxAbs")
    }

    @Test
    fun softmaxAxisJvpVjpCrossIdentity() {
        // ⟨∇f, v⟩ == jvp tangent through a softmax(axis=1) chain, no shared
        // path between the two AD modes.
        val t = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("softmax_cross") {
            val x = param("x", t)
            val w = param("w", t)
            val y = op(OpKind.SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1))
            val p = op(OpKind.MUL, listOf(y, w), t)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val x = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 1.1f, 0.9f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.53f, 0.67f)
        val vw = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f, 0.19f, -0.07f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in 0 until 6) dot += grads[0][i].toDouble() * vx[i] + grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through softmax(axis=1): ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
