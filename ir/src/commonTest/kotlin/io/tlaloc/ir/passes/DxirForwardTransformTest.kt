package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.361 — forward-mode certification. Three independent oracles pin
 * [DxirForwardTransform]:
 *
 * 1. **Central differences** — `(f(x+εv) − f(x−εv)) / 2ε` computed by the
 *    interpreter on the *primal*, no AD involved.
 * 2. **The reverse transform** — for scalar `f`, `⟨∇f(x), v⟩` must equal
 *    the JVP tangent exactly (both analytic); each transform certifies
 *    the other.
 * 3. **Hand-computed formulas** in doubles (softmax, pow).
 *
 * Plus the composition payoff: `forward(reverse(f))` is a
 * Hessian-vector product, pinned against the analytic Hessian of
 * `sum(x³)`.
 */
class DxirForwardTransformTest {

    private val scalar = DxirType(F32, emptyList())

    /** Scalar-valued mixed graph: sum(tanh(A·B) ⊙ (A·B)). */
    private fun mixedFn(): DxirFunction {
        val aT = DxirType(F32, listOf(2, 3))
        val bT = DxirType(F32, listOf(3, 2))
        val yT = DxirType(F32, listOf(2, 2))
        return DxirBuilder.function("mixed") {
            val a = param("a", aT)
            val b = param("b", bT)
            val m = op(OpKind.MATMUL, listOf(a, b), yT)
            val t = op(OpKind.TANH, listOf(m), yT)
            val w = op(OpKind.MUL, listOf(t, m), yT)
            listOf(op(OpKind.SUM, listOf(w), scalar))
        }
    }

    private val aVals = floatArrayOf(0.3f, -0.5f, 0.8f, 0.1f, 0.9f, -0.2f)
    private val bVals = floatArrayOf(0.4f, -0.7f, 0.2f, 0.6f, -0.3f, 0.5f)
    private val daVals = floatArrayOf(0.11f, -0.07f, 0.05f, 0.13f, -0.02f, 0.09f)
    private val dbVals = floatArrayOf(-0.04f, 0.12f, 0.08f, -0.1f, 0.03f, 0.06f)

    @Test
    fun jvpMatchesCentralDifferencesOnMixedGraph() {
        val fn = mixedFn()
        val jvp = DxirForwardTransform.apply(fn)
        assertEquals(4, jvp.params.size, "params + tangent params")
        assertEquals(2, jvp.returns.size, "primal value + tangent")

        val out = DxirInterpreter.evalFunction(jvp, listOf(aVals, bVals, daVals, dbVals))
        val primal = DxirInterpreter.evalFunction(fn, listOf(aVals, bVals)).single().single()
        assertEquals(primal, out[0].single(), "jvp preserves the primal value stream")

        // Central differences on the primal: no AD in the oracle.
        val eps = 1e-3f
        fun shift(s: Float): Float {
            val aS = FloatArray(6) { aVals[it] + s * daVals[it] }
            val bS = FloatArray(6) { bVals[it] + s * dbVals[it] }
            return DxirInterpreter.evalFunction(fn, listOf(aS, bS)).single().single()
        }
        val fd = (shift(eps) - shift(-eps)) / (2 * eps)
        val tangent = out[1].single()
        assertTrue(
            abs(tangent - fd) <= 2e-3f * maxOf(1f, abs(fd)),
            "jvp tangent $tangent vs central differences $fd",
        )
    }

    @Test
    fun jvpAgreesWithReverseModeDotProduct() {
        // The AD cross-validation identity: ⟨∇f(x), v⟩ == jvp_f(x, v).tangent.
        // Both sides analytic — tight tolerance.
        val fn = mixedFn()
        val tangent = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(aVals, bVals, daVals, dbVals),
        )[1].single()

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(aVals, bVals))
        var dot = 0.0
        for (i in 0 until 6) dot += grads[0][i].toDouble() * daVals[i] + grads[1][i].toDouble() * dbVals[i]

        assertTrue(
            abs(tangent - dot) <= 1e-5,
            "forward and reverse modes disagree: jvp=$tangent, ⟨∇f,v⟩=$dot",
        )
    }

    @Test
    fun softmaxJvpMatchesAnalyticFormula() {
        // dy = y ⊙ (dx − Σ_axis(y ⊙ dx)), hand-computed in doubles.
        val t = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("softmax_fwd") {
            val x = param("x", t)
            listOf(op(OpKind.SOFTMAX, listOf(x), t))
        }
        val x = floatArrayOf(0.1f, -0.4f, 0.9f, 1.5f, 0f, -2f)
        val dx = floatArrayOf(0.3f, -0.2f, 0.5f, -0.1f, 0.4f, 0.2f)
        val out = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(x, dx))

        val want = DoubleArray(6)
        for (row in 0 until 2) {
            val base = row * 3
            val mx = (0 until 3).maxOf { x[base + it].toDouble() }
            val e = DoubleArray(3) { exp(x[base + it] - mx) }
            val sum = e.sum()
            val y = DoubleArray(3) { e[it] / sum }
            val dot = (0 until 3).sumOf { y[it] * dx[base + it] }
            for (j in 0 until 3) want[base + j] = y[j] * (dx[base + j] - dot)
        }
        var maxAbs = 0.0
        for (i in 0 until 6) maxAbs = maxOf(maxAbs, abs(out[1][i] - want[i]))
        assertTrue(maxAbs <= 1e-6, "softmax jvp diverges from analytic: max|diff|=$maxAbs")
    }

    @Test
    fun whereRoutesTangentsByMask() {
        // y = where(a > b, a, b): dy = da where the mask holds, db elsewhere. Exact.
        val t = DxirType(F32, listOf(4))
        val bT = DxirType(Bool, listOf(4))
        val fn = DxirBuilder.function("where_fwd") {
            val a = param("a", t)
            val b = param("b", t)
            val m = op(OpKind.COMPARE, listOf(a, b), bT, attrs = mapOf("direction" to "GT"))
            listOf(op(OpKind.WHERE, listOf(m, a, b), t))
        }
        val a = floatArrayOf(1f, 5f, 2f, 7f)
        val b = floatArrayOf(3f, 4f, 2f, 6f)
        val da = floatArrayOf(10f, 20f, 30f, 40f)
        val db = floatArrayOf(-1f, -2f, -3f, -4f)
        val out = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(a, b, da, db))
        assertEquals(listOf(3f, 5f, 2f, 7f), out[0].toList(), "primal = elementwise max")
        assertEquals(listOf(-1f, 20f, -3f, 40f), out[1].toList(), "tangent routed by a>b mask")
    }

    @Test
    fun maxReduceRoutesTangentThroughArgmax() {
        // Rows-max keepdims: dy_row = dx at the row's argmax (ties sum — the
        // VJP MaxRule convention, forward twin).
        val fn = DxirBuilder.function("max_fwd") {
            val x = param("x", DxirType(F32, listOf(2, 3)))
            listOf(
                op(
                    OpKind.MAX, listOf(x), DxirType(F32, listOf(2, 1)),
                    attrs = mapOf("reduction_dims" to listOf(1)),
                ),
            )
        }
        val x = floatArrayOf(1f, 3f, 2f, 5f, 4f, 6f)
        val dx = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f)
        val out = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(x, dx))
        assertEquals(listOf(3f, 6f), out[0].toList())
        assertEquals(listOf(0.2f, 0.6f), out[1].toList(), "tangent of each row's max element")
    }

    @Test
    fun powAndDivJvpMatchAnalytic() {
        // f = a^c / d elementwise; df = (c·a^(c−1)·da + a^c·ln(a)·dc − f·dd) / d…
        // assembled from the POW and DIV product rules, pinned in doubles.
        val t = DxirType(F32, listOf(3))
        val fn = DxirBuilder.function("powdiv_fwd") {
            val a = param("a", t)
            val c = param("c", t)
            val d = param("d", t)
            val p = op(OpKind.POW, listOf(a, c), t)
            listOf(op(OpKind.DIV, listOf(p, d), t))
        }
        val a = floatArrayOf(1.5f, 2f, 0.7f)
        val c = floatArrayOf(2f, 0.5f, 3f)
        val d = floatArrayOf(2f, 4f, 0.5f)
        val da = floatArrayOf(0.1f, -0.2f, 0.3f)
        val dc = floatArrayOf(0.05f, 0.1f, -0.15f)
        val dd = floatArrayOf(-0.3f, 0.2f, 0.1f)
        val out = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(a, c, d, da, dc, dd),
        )
        var maxAbs = 0.0
        for (i in 0 until 3) {
            val ad = a[i].toDouble(); val cd = c[i].toDouble(); val dd0 = d[i].toDouble()
            val p = ad.pow(cd)
            val dp = cd * ad.pow(cd - 1) * da[i] + p * ln(ad) * dc[i]
            val want = (dp - (p / dd0) * dd[i]) / dd0
            maxAbs = maxOf(maxAbs, abs(out[1][i] - want))
        }
        assertTrue(maxAbs <= 1e-5, "pow/div jvp diverges from analytic: max|diff|=$maxAbs")
    }

    @Test
    fun hessianVectorProductViaForwardOverReverse() {
        // The composition payoff. f(x) = sum(x³): ∇f = 3x², H = diag(6x),
        // so forward(reverse(f))(x, v) = [3x², 6x ⊙ v].
        val t = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("cubic") {
            val x = param("x", t)
            val x2 = op(OpKind.MUL, listOf(x, x), t)
            val x3 = op(OpKind.MUL, listOf(x2, x), t)
            listOf(op(OpKind.SUM, listOf(x3), scalar))
        }
        val hvpFn = DxirForwardTransform.apply(DxirReverseTransform.apply(fn))
        val x = floatArrayOf(1f, -2f, 0.5f, 3f)
        val v = floatArrayOf(1f, 0f, -1f, 2f)
        val out = DxirInterpreter.evalFunction(hvpFn, listOf(x, v))

        val grad = FloatArray(4) { 3f * x[it] * x[it] }
        val hvp = FloatArray(4) { 6f * x[it] * v[it] }
        var maxAbs = 0f
        for (i in 0 until 4) {
            maxAbs = maxOf(maxAbs, abs(out[0][i] - grad[i]))
            maxAbs = maxOf(maxAbs, abs(out[1][i] - hvp[i]))
        }
        assertTrue(maxAbs <= 1e-5f, "HVP diverges from analytic diag(6x)·v: max|diff|=$maxAbs")
    }
}
