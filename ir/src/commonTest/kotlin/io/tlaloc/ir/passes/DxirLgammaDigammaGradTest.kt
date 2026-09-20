package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.digamma
import io.tlaloc.core.lgamma
import io.tlaloc.core.polygamma
import io.tlaloc.core.trigamma
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.402 — Phase C1 special functions (DiffKT parity): LGAMMA and DIGAMMA
 * differentiate through both transforms at IR level. Oracles: hand-computed
 * analytic pins with a NON-uniform upstream (`Σ f(x) ⊙ w`, so a dropped
 * upstream factor cannot pass), the JVP⇄VJP dot-product cross-identity through
 * a body chaining both new rules, and an interpreter-level central-difference
 * cross-check at f32-honest tolerance. TRIGAMMA — the internal op DIGAMMA's
 * adjoint emits — refused both transforms until §0.4.405 landed POLYGAMMA;
 * the flipped pin (both transforms now produce ψ₂) closes this class, and the
 * general-order coverage lives in DxirPolygammaGradTest.
 *
 * Probes stay on the positive axis away from 0 — near a pole the adjoint is
 * astronomically large and a pin there tests rounding, not the rule.
 */
class DxirLgammaDigammaGradTest {

    private val scalar = DxirType(F32, emptyList())
    private val r1 = DxirType(F32, listOf(4))

    @Test
    fun lgammaGradientMatchesAnalytic() {
        // loss = Σ lgamma(x) ⊙ w  ⇒  dx = w ⊙ ψ(x), dw = lgamma(x).
        val fn = DxirBuilder.function("lgamma_loss") {
            val x = param("x", r1)
            val w = param("w", r1)
            val lg = op(OpKind.LGAMMA, listOf(x), r1)
            val p = op(OpKind.MUL, listOf(lg, w), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.5f, 1.3f, 2.7f, 4.1f)
        val w = floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        for (i in 0 until 4) {
            val xd = x[i].toDouble()
            val wantDx = (w[i] * xd.digamma()).toFloat()
            val wantDw = xd.lgamma().toFloat()
            assertTrue(abs(out[0][i] - wantDx) < 1e-4f, "dx[$i] = ${out[0][i]}, want $wantDx")
            assertTrue(abs(out[1][i] - wantDw) < 1e-5f, "dw[$i] = ${out[1][i]}, want $wantDw")
        }
    }

    @Test
    fun digammaGradientMatchesAnalytic() {
        // loss = Σ digamma(x) ⊙ w  ⇒  dx = w ⊙ ψ₁(x), dw = ψ(x). The adjoint
        // contains the internal TRIGAMMA op — this is its interpreter pin too.
        val fn = DxirBuilder.function("digamma_loss") {
            val x = param("x", r1)
            val w = param("w", r1)
            val dg = op(OpKind.DIGAMMA, listOf(x), r1)
            val p = op(OpKind.MUL, listOf(dg, w), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.5f, 1.3f, 2.7f, 4.1f)
        val w = floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        for (i in 0 until 4) {
            val xd = x[i].toDouble()
            val wantDx = (w[i] * xd.trigamma()).toFloat()
            val wantDw = xd.digamma().toFloat()
            assertTrue(abs(out[0][i] - wantDx) < 1e-4f, "dx[$i] = ${out[0][i]}, want $wantDx")
            assertTrue(abs(out[1][i] - wantDw) < 1e-5f, "dw[$i] = ${out[1][i]}, want $wantDw")
        }
    }

    @Test
    fun lgammaDigammaJvpVjpCrossIdentity() {
        // f(x, w) = Σ lgamma(digamma(x ⊙ w)) — both new rules chained in one
        // body (the reverse pass emits DIGAMMA and TRIGAMMA; the forward pass
        // the same pair). Products x⊙w stay ≥ 2 so ψ(x⊙w) > 0.4 keeps lgamma's
        // argument well off its pole. ⟨∇f, v⟩ must equal the forward tangent.
        val fn = DxirBuilder.function("lgamma_digamma_chain") {
            val x = param("x", r1)
            val w = param("w", r1)
            val p = op(OpKind.MUL, listOf(x, w), r1)
            val dg = op(OpKind.DIGAMMA, listOf(p), r1)
            val lg = op(OpKind.LGAMMA, listOf(dg), r1)
            listOf(op(OpKind.SUM, listOf(lg), scalar))
        }
        val x = floatArrayOf(1.7f, 2.3f, 3.1f, 4.0f)
        val w = floatArrayOf(1.2f, 1.0f, 0.9f, 1.5f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f)
        val vw = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in 0 until 4) dot += grads[0][i].toDouble() * vx[i] + grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-3,
            "JVP⇄VJP cross-identity broken through lgamma∘digamma: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }

    @Test
    fun lgammaGradientMatchesCentralDifferences() {
        // ∇ₓ Σ lgamma(x) against interpreter-evaluated central differences —
        // everything in f32 through the same LGAMMA arm, so the tolerance is
        // f32-honest: truncation O(h²·ψ′) + f32 roundoff on values ~O(1) at
        // h = 1e-2 lands well inside 5e-3.
        val fn = DxirBuilder.function("lgamma_sum") {
            val x = param("x", r1)
            val lg = op(OpKind.LGAMMA, listOf(x), r1)
            listOf(op(OpKind.SUM, listOf(lg), scalar))
        }
        val x = floatArrayOf(0.6f, 1.4f, 2.9f, 5.2f)
        val grad = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x)).single()
        val h = 1e-2f
        for (i in 0 until 4) {
            val plus = x.copyOf().also { it[i] += h }
            val minus = x.copyOf().also { it[i] -= h }
            val fPlus = DxirInterpreter.evalFunction(fn, listOf(plus)).single().single()
            val fMinus = DxirInterpreter.evalFunction(fn, listOf(minus)).single().single()
            val want = (fPlus - fMinus) / (2f * h)
            assertTrue(
                abs(grad[i] - want) < 5e-3f,
                "central-diff slot $i: reverse=${grad[i]}, numeric=$want",
            )
        }
    }

    @Test
    fun trigammaDifferentiatesThroughBothTransforms() {
        // §0.4.405 FLIPS the §0.4.402 pinned refusal: TRIGAMMA now carries
        // TrigammaRule (reverse) and a tangent arm (forward), both emitting
        // POLYGAMMA(order = 2) — d ψ₁ = ψ₂ — so second-order derivatives
        // through DIGAMMA compose instead of failing. Reverse pin here; the
        // second-order compositions live in DxirPolygammaGradTest.
        val fn = DxirBuilder.function("trigamma_sum") {
            val x = param("x", r1)
            val tg = op(OpKind.TRIGAMMA, listOf(x), r1)
            listOf(op(OpKind.SUM, listOf(tg), scalar))
        }
        val x = floatArrayOf(0.6f, 1.4f, 2.9f, 5.2f)
        val grad = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x)).single()
        val tangent = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn),
            listOf(x, floatArrayOf(1f, 1f, 1f, 1f)),
        )[1].single()
        var wantTangent = 0.0
        for (i in 0 until 4) {
            val want = x[i].toDouble().polygamma(2)
            wantTangent += want
            assertTrue(
                abs(grad[i] - want) < 1e-3f * maxOf(1.0, abs(want)).toFloat(),
                "d Σψ₁ slot $i: got ${grad[i]}, want ψ₂ = $want",
            )
        }
        assertTrue(
            abs(tangent - wantTangent) < 1e-3 * maxOf(1.0, abs(wantTangent)),
            "forward tangent of Σψ₁ with dx = 1: got $tangent, want Σψ₂ = $wantTangent",
        )
    }
}
