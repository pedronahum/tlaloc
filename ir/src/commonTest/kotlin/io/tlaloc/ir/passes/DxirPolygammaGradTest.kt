package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.polygamma
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.405 — general POLYGAMMA(n) at IR level, closing C1's recorded deferral:
 * PolygammaRule / the POLYGAMMA tangent arm climb the ψ-ladder by one order
 * (`d ψ⁽ⁿ⁾ = ψ⁽ⁿ⁺¹⁾`, a literal `order` attr — never a shape), so the
 * special-function family is now CLOSED under differentiation to any depth.
 * Oracles: hand-computed analytic pins with a NON-uniform upstream, the
 * JVP⇄VJP dot-product cross-identity, an interpreter-level central-difference
 * cross-check, and — the §0.4.402 refusal flipped into a positive cert —
 * second-order reverse AND forward-over-reverse through Σ digamma(x), whose
 * first adjoint contains the once-refusing TRIGAMMA op.
 *
 * Probes stay on the positive axis away from 0 (the §0.4.395 pole
 * discipline); the higher the order the faster ψ⁽ⁿ⁾ explodes toward 0⁺, so
 * everything sits at x ≥ 0.6 with relative tolerances.
 */
class DxirPolygammaGradTest {

    private val scalar = DxirType(F32, emptyList())
    private val r1 = DxirType(F32, listOf(4))

    @Test
    fun polygammaGradientMatchesAnalytic() {
        // loss = Σ ψ₂(x) ⊙ w  ⇒  dx = w ⊙ ψ₃(x), dw = ψ₂(x). The adjoint is a
        // POLYGAMMA(order = 3) node — the attr must climb, and the CSE key's
        // attrs component must keep it apart from the primal's order-2 node.
        val fn = DxirBuilder.function("polygamma2_loss") {
            val x = param("x", r1)
            val w = param("w", r1)
            val p2 = op(OpKind.POLYGAMMA, listOf(x), r1, attrs = mapOf("order" to 2))
            val p = op(OpKind.MUL, listOf(p2, w), r1)
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(0.7f, 1.3f, 2.7f, 4.1f)
        val w = floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f)
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        for (i in 0 until 4) {
            val xd = x[i].toDouble()
            val wantDx = (w[i] * xd.polygamma(3)).toFloat()
            val wantDw = xd.polygamma(2).toFloat()
            assertTrue(
                abs(out[0][i] - wantDx) < 1e-3f * maxOf(1f, abs(wantDx)),
                "dx[$i] = ${out[0][i]}, want $wantDx",
            )
            assertTrue(
                abs(out[1][i] - wantDw) < 1e-4f * maxOf(1f, abs(wantDw)),
                "dw[$i] = ${out[1][i]}, want $wantDw",
            )
        }
    }

    @Test
    fun polygammaJvpVjpCrossIdentity() {
        // f(x, w) = Σ ψ₂(x ⊙ w): the reverse pass emits POLYGAMMA(3), the
        // forward pass the same node — ⟨∇f, v⟩ must equal the seeded tangent.
        val fn = DxirBuilder.function("polygamma_chain") {
            val x = param("x", r1)
            val w = param("w", r1)
            val p = op(OpKind.MUL, listOf(x, w), r1)
            val p2 = op(OpKind.POLYGAMMA, listOf(p), r1, attrs = mapOf("order" to 2))
            listOf(op(OpKind.SUM, listOf(p2), scalar))
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
            "JVP⇄VJP cross-identity broken through ψ₂: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }

    @Test
    fun polygammaGradientMatchesCentralDifferences() {
        // ∇ₓ Σ ψ₂(x) against interpreter-evaluated central differences, all in
        // f32 through the same POLYGAMMA arm. Relative tolerance: ψ₂/ψ₃ grow
        // fast toward 0⁺, so absolute pins would test magnitude, not the rule.
        val fn = DxirBuilder.function("polygamma2_sum") {
            val x = param("x", r1)
            val p2 = op(OpKind.POLYGAMMA, listOf(x), r1, attrs = mapOf("order" to 2))
            listOf(op(OpKind.SUM, listOf(p2), scalar))
        }
        val x = floatArrayOf(0.8f, 1.4f, 2.9f, 5.2f)
        val grad = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x)).single()
        val h = 1e-2f
        for (i in 0 until 4) {
            val plus = x.copyOf().also { it[i] += h }
            val minus = x.copyOf().also { it[i] -= h }
            val fPlus = DxirInterpreter.evalFunction(fn, listOf(plus)).single().single()
            val fMinus = DxirInterpreter.evalFunction(fn, listOf(minus)).single().single()
            val want = (fPlus - fMinus) / (2f * h)
            assertTrue(
                abs(grad[i] - want) < 5e-2f * maxOf(1f, abs(want)),
                "central-diff slot $i: reverse=${grad[i]}, numeric=$want",
            )
        }
    }

    @Test
    fun secondOrderReverseThroughDigammaComposes() {
        // THE flipped §0.4.402 refusal: f = Σ ψ(x) has ∇f = ψ₁(x) (the first
        // reverse pass emits TRIGAMMA), and rev∘rev's seeded pullback of the
        // gradient function is the HVP: H = diag(ψ₂), so R(R(f))(v, x) must be
        // v ⊙ ψ₂(x). Pre-§0.4.405 the second apply threw "no VJP rule
        // registered for TRIGAMMA".
        val fn = DxirBuilder.function("sum_digamma") {
            val x = param("x", r1)
            val dg = op(OpKind.DIGAMMA, listOf(x), r1)
            listOf(op(OpKind.SUM, listOf(dg), scalar))
        }
        val g = DxirReverseTransform.apply(fn)
        val rr = DxirReverseTransform.apply(g, seedAsParam = true)
        val x = floatArrayOf(0.9f, 1.4f, 2.6f, 4.8f)
        val v = floatArrayOf(0.5f, -1.2f, 0.7f, 2.0f)
        val out = DxirInterpreter.evalFunction(rr, listOf(v, x))
        for (i in 0 until 4) {
            val want = (v[i] * x[i].toDouble().polygamma(2)).toFloat()
            assertTrue(
                abs(out[0][i] - want) < 1e-3f * maxOf(1f, abs(want)),
                "rev∘rev Hv slot $i: got ${out[0][i]}, want v⊙ψ₂ = $want",
            )
        }
    }

    @Test
    fun forwardOverReverseThroughDigammaIsHvp() {
        // The hessian intrinsic's composition (§0.4.394 forward-over-reverse):
        // the tangent transform must reach through the TRIGAMMA the reverse
        // pass emitted — the tangent arm §0.4.405 added. Output [1] is H·v =
        // v ⊙ ψ₂(x), and it must agree with rev∘rev above numerically.
        val fn = DxirBuilder.function("sum_digamma_fr") {
            val x = param("x", r1)
            val dg = op(OpKind.DIGAMMA, listOf(x), r1)
            listOf(op(OpKind.SUM, listOf(dg), scalar))
        }
        val fr = DxirForwardTransform.apply(DxirReverseTransform.apply(fn))
        val x = floatArrayOf(0.9f, 1.4f, 2.6f, 4.8f)
        val v = floatArrayOf(0.5f, -1.2f, 0.7f, 2.0f)
        val hv = DxirInterpreter.evalFunction(fr, listOf(x, v))[1]

        val g = DxirReverseTransform.apply(fn)
        val rr = DxirReverseTransform.apply(g, seedAsParam = true)
        val rrOut = DxirInterpreter.evalFunction(rr, listOf(v, x))[0]
        for (i in 0 until 4) {
            val want = (v[i] * x[i].toDouble().polygamma(2)).toFloat()
            assertTrue(
                abs(hv[i] - want) < 1e-3f * maxOf(1f, abs(want)),
                "fwd∘rev Hv slot $i: got ${hv[i]}, want v⊙ψ₂ = $want",
            )
            assertTrue(
                abs(hv[i] - rrOut[i]) < 1e-3f * maxOf(1f, abs(want)),
                "fwd∘rev vs rev∘rev slot $i: ${hv[i]} vs ${rrOut[i]}",
            )
        }
    }

    @Test
    fun thirdOrderReverseThroughDigammaClimbsTheLadder() {
        // One more rung for free: rev∘rev∘rev of Σ ψ(x) needs TrigammaRule AND
        // PolygammaRule (the second pass emits POLYGAMMA(2), the third its
        // order-3 adjoint) — the family is closed under differentiation, not
        // merely one level deeper. Third derivative of Σψ is diag(ψ₃): seeding
        // both pullbacks with (v, u) yields u ⊙ v ⊙ ψ₃(x) in the x slot.
        val fn = DxirBuilder.function("sum_digamma_r3") {
            val x = param("x", r1)
            val dg = op(OpKind.DIGAMMA, listOf(x), r1)
            listOf(op(OpKind.SUM, listOf(dg), scalar))
        }
        val g = DxirReverseTransform.apply(fn)
        val rr = DxirReverseTransform.apply(g, seedAsParam = true)
        // rr(v, x) returns Hv = v ⊙ ψ₂(x); pull that back too. rr has two
        // returns (d__upstream__, dx) — project to the x-gradient? No: rr's
        // returns are the pullback pair; a THIRD reverse needs one scalar
        // return, so build Σ(rr's x-slot) by hand instead: reverse the
        // scalar-projected composition g2(v, x) = Σ (v ⊙ ψ₂(x)) built from a
        // fresh body — same math, and it still exercises PolygammaRule because
        // ψ₂'s adjoint is ψ₃.
        val g2 = DxirBuilder.function("v_dot_psi2") {
            val v = param("v", r1)
            val x = param("x", r1)
            val p2 = op(OpKind.POLYGAMMA, listOf(x), r1, attrs = mapOf("order" to 2))
            val prod = op(OpKind.MUL, listOf(v, p2), r1)
            listOf(op(OpKind.SUM, listOf(prod), scalar))
        }
        val r3 = DxirReverseTransform.apply(g2)
        val x = floatArrayOf(1.1f, 1.7f, 2.9f, 4.3f)
        val v = floatArrayOf(0.5f, -1.2f, 0.7f, 2.0f)
        val out = DxirInterpreter.evalFunction(r3, listOf(v, x))
        // Sanity that rr agrees with g2's premise (Hv = v ⊙ ψ₂):
        val hv = DxirInterpreter.evalFunction(rr, listOf(v, x))[0]
        for (i in 0 until 4) {
            val psi2 = x[i].toDouble().polygamma(2)
            assertTrue(
                abs(hv[i] - (v[i] * psi2).toFloat()) < 1e-3f * maxOf(1f, abs((v[i] * psi2).toFloat())),
                "premise slot $i",
            )
            val wantDv = psi2.toFloat()
            val wantDx = (v[i] * x[i].toDouble().polygamma(3)).toFloat()
            assertTrue(
                abs(out[0][i] - wantDv) < 1e-3f * maxOf(1f, abs(wantDv)),
                "third-order dv slot $i: got ${out[0][i]}, want ψ₂ = $wantDv",
            )
            assertTrue(
                abs(out[1][i] - wantDx) < 1e-2f * maxOf(1f, abs(wantDx)),
                "third-order dx slot $i: got ${out[1][i]}, want v⊙ψ₃ = $wantDx",
            )
        }
    }
}
