package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.359 — analytic pins for the DiffKT-comparison rule closures:
 * RESHAPE, MAX/MIN reductions, and raw SOFTMAX now differentiate (each
 * previously failed only at runtime — the §0.4.353 compile-time probe
 * surfaces them as errors no longer). Every gradient is evaluated
 * through [DxirReverseTransform] + [DxirInterpreter] against
 * hand-computed values.
 */
class DxirNewVjpRulesTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun reshapeGradientReshapesUpstreamBack() {
        val fn = DxirBuilder.function("reshape_loss") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val r = op(OpKind.RESHAPE, listOf(x), DxirType(F32, listOf(3, 2)))
            listOf(op(OpKind.SUM, listOf(r), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        assertEquals(listOf(2, 3), grad.returns.single().type.dims, "gradient carries x's shape")
        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)))
        assertEquals(List(6) { 1f }, out.single().toList(), "d sum(reshape(x))/dx = ones at x's shape")
    }

    @Test
    fun maxReduceGradientRoutesToMaximaWithFullTies() {
        // Rows-max keepdims over [2,3]; loss = sum of the maxima.
        val fn = DxirBuilder.function("max_loss") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val m = op(
                OpKind.MAX, listOf(x), DxirType(F32, listOf(2, 1)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(op(OpKind.SUM, listOf(m), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        // Row 0 has a unique max (3); row 1 ties at 5 twice — full upstream
        // to every tie (the JAX select convention, per the rule's KDoc).
        val out = DxirInterpreter.evalFunction(
            grad, listOf(floatArrayOf(1f, 3f, 2f, 5f, 4f, 5f)),
        )
        assertEquals(listOf(0f, 1f, 0f, 1f, 0f, 1f), out.single().toList())
    }

    @Test
    fun minReduceGradientRoutesToMinima() {
        val fn = DxirBuilder.function("min_loss") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            val m = op(
                OpKind.MIN, listOf(x), DxirType(F32, listOf(2, 1)),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(op(OpKind.SUM, listOf(m), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val out = DxirInterpreter.evalFunction(
            grad, listOf(floatArrayOf(1f, 3f, 2f, 5f, 4f, 6f)),
        )
        assertEquals(listOf(1f, 0f, 0f, 0f, 1f, 0f), out.single().toList())
    }

    @Test
    fun softmaxGradientMatchesAnalyticFormula() {
        // loss = Σ w ⊙ softmax(x): dx = y ⊙ (w − Σ_axis(w ⊙ y)).
        val t = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("softmax_loss") {
            val x = param("x", t)
            val w = param("w", t)
            val y = op(OpKind.SOFTMAX, listOf(x), t)
            val weighted = op(OpKind.MUL, listOf(w, y), t)
            listOf(op(OpKind.SUM, listOf(weighted), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        assertEquals(2, grad.returns.size)

        val x = floatArrayOf(0.1f, -0.4f, 0.9f, 1.5f, 0f, -2f)
        val w = floatArrayOf(1f, 2f, 3f, -1f, 0.5f, 2f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))

        // Hand-computed reference in doubles.
        val dx = DoubleArray(6)
        val dw = DoubleArray(6)
        for (row in 0 until 2) {
            val base = row * 3
            val mx = (0 until 3).maxOf { x[base + it].toDouble() }
            val e = DoubleArray(3) { exp(x[base + it] - mx) }
            val sum = e.sum()
            val y = DoubleArray(3) { e[it] / sum }
            val dot = (0 until 3).sumOf { w[base + it] * y[it] }
            for (j in 0 until 3) {
                dx[base + j] = y[j] * (w[base + j] - dot)
                dw[base + j] = y[j]
            }
        }
        var maxAbs = 0.0
        for (i in 0 until 6) {
            maxAbs = maxOf(maxAbs, abs(out[0][i] - dx[i]))
            maxAbs = maxOf(maxAbs, abs(out[1][i] - dw[i]))
        }
        assertTrue(maxAbs <= 1e-5, "softmax gradient diverges from analytic: max|diff|=$maxAbs")
    }
}
