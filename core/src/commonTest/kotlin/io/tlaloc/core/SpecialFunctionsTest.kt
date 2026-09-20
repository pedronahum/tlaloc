package io.tlaloc.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.402 — Phase C1 special functions: the shared Double kernels
 * (`Double.lgamma()` Lanczos g=7, `Double.digamma()` / `Double.trigamma()`
 * recurrence-to-asymptotic) validated against hand-pinned reference values,
 * the Γ/ψ/ψ₁ recurrences, the reflection formulas, and — the derivative
 * consistency oracle — central differences of each function against the next
 * one up the derivative chain (lgamma′ = digamma, digamma′ = trigamma), all
 * in Double at tolerances far past the 1e-6 the plan asks.
 */
class SpecialFunctionsTest {

    private val gamma = 0.5772156649015329 // Euler–Mascheroni γ

    private fun assertClose(want: Double, got: Double, tol: Double, label: String) {
        assertTrue(abs(want - got) <= tol, "$label: got $got, want $want (|diff|=${abs(want - got)})")
    }

    @Test
    fun lgammaReferencePins() {
        assertClose(0.5723649429247001, 0.5.lgamma(), 1e-9, "lgamma(0.5) = ln √π")
        assertClose(0.0, 1.0.lgamma(), 1e-12, "lgamma(1)")
        assertClose(0.0, 2.0.lgamma(), 1e-12, "lgamma(2)")
        assertClose(ln(6.0), 4.0.lgamma(), 1e-9, "lgamma(4) = ln 3!")
        assertClose(ln(362880.0), 10.0.lgamma(), 1e-9, "lgamma(10) = ln 9!")
        assertClose(2.252712651734206, 0.1.lgamma(), 1e-9, "lgamma(0.1)")
        // Reflection: Γ(−0.5) = −2√π ⇒ ln|Γ| = ln(2√π).
        assertClose(ln(2.0 * kotlin.math.sqrt(PI)), (-0.5).lgamma(), 1e-9, "lgamma(−0.5)")
        // Poles: +∞ at the non-positive integers.
        assertTrue(0.0.lgamma() == Double.POSITIVE_INFINITY, "lgamma(0) must be +∞")
        assertTrue((-3.0).lgamma() == Double.POSITIVE_INFINITY, "lgamma(−3) must be +∞")
    }

    @Test
    fun digammaReferencePins() {
        assertClose(-gamma, 1.0.digamma(), 1e-11, "ψ(1) = −γ")
        assertClose(-gamma - 2.0 * ln(2.0), 0.5.digamma(), 1e-11, "ψ(0.5) = −γ − 2 ln 2")
        assertClose(1.0 - gamma, 2.0.digamma(), 1e-11, "ψ(2) = 1 − γ")
        assertClose(1.7061176684318003, 6.0.digamma(), 1e-11, "ψ(6) = 137/60 − γ")
        // Reflection: ψ(−0.5) = ψ(0.5) + 2 (recurrence through the negative axis).
        assertClose(2.0 - gamma - 2.0 * ln(2.0), (-0.5).digamma(), 1e-11, "ψ(−0.5)")
    }

    @Test
    fun trigammaReferencePins() {
        assertClose(PI * PI / 6.0, 1.0.trigamma(), 1e-11, "ψ₁(1) = π²/6")
        assertClose(PI * PI / 2.0, 0.5.trigamma(), 1e-11, "ψ₁(0.5) = π²/2")
        assertClose(PI * PI / 6.0 - 1.0, 2.0.trigamma(), 1e-11, "ψ₁(2) = π²/6 − 1")
        // Reflection/recurrence: ψ₁(−0.5) = ψ₁(0.5) + 4.
        assertClose(PI * PI / 2.0 + 4.0, (-0.5).trigamma(), 1e-11, "ψ₁(−0.5)")
    }

    @Test
    fun recurrencesHold() {
        // Γ(x+1) = x·Γ(x) ⇒ lgamma(x+1) − lgamma(x) = ln x; ψ(x+1) − ψ(x) = 1/x;
        // ψ₁(x) − ψ₁(x+1) = 1/x². Probes straddle the reflection (x < 0.5), the
        // recurrence-shift region, and the asymptotic region.
        for (x in listOf(0.2, 0.7, 1.3, 3.9, 7.5, 25.0)) {
            assertClose(ln(x), (x + 1.0).lgamma() - x.lgamma(), 1e-10, "lgamma recurrence at $x")
            assertClose(1.0 / x, (x + 1.0).digamma() - x.digamma(), 1e-10, "digamma recurrence at $x")
            assertClose(1.0 / (x * x), x.trigamma() - (x + 1.0).trigamma(), 1e-10, "trigamma recurrence at $x")
        }
    }

    @Test
    fun derivativeChainMatchesCentralDifferences() {
        // The cross-oracle the gradient rules rest on: lgamma′ = digamma and
        // digamma′ = trigamma, checked numerically. h = 1e-5 ⇒ truncation
        // O(h²·f‴) ≲ 1e-9 on these probes; roundoff ≪ that in Double.
        val h = 1e-5
        for (x in listOf(0.3, 0.8, 1.5, 2.5, 4.2, 9.0)) {
            val dLgamma = ((x + h).lgamma() - (x - h).lgamma()) / (2.0 * h)
            assertClose(x.digamma(), dLgamma, 1e-7, "central-diff lgamma′ vs ψ at $x")
            val dDigamma = ((x + h).digamma() - (x - h).digamma()) / (2.0 * h)
            assertClose(x.trigamma(), dDigamma, 1e-6, "central-diff ψ′ vs ψ₁ at $x")
        }
    }

    // ------------------------------------------------------------- §0.4.405
    // General polygamma(n): the C1 deferral's kernel, validated the same way —
    // hand pins, recurrences, reflection, the derivative chain — PLUS the pin
    // the slice spec demands: the general differentiated-series scheme at
    // n = 1 must independently reproduce the §0.4.402 trigamma kernel.

    @Test
    fun polygammaOrderOneAgreesWithTrigammaKernel() {
        // The general kernel computes n = 1 through the SAME differentiated
        // Bernoulli series / reflection machinery it uses for every order —
        // NOT by delegating to trigamma — so 1e-11 agreement here is the hard
        // validation of the series derivation. Probes cover the asymptotic
        // region, the recurrence-shift region, the reflection region, and
        // near-pole neighbourhoods on the negative axis.
        for (x in listOf(0.1, 0.5, 1.0, 2.5, 7.3, 15.0, 42.0, -0.3, -2.7, -7.6)) {
            val want = x.trigamma()
            val got = x.polygamma(1)
            assertClose(want, got, 1e-11 * maxOf(1.0, abs(want)), "polygamma(1) vs trigamma at $x")
        }
    }

    @Test
    fun polygammaReferencePins() {
        val zeta3 = 1.2020569031595943 // ζ(3)
        val zeta5 = 1.0369277551433699 // ζ(5)
        // ψ⁽ⁿ⁾(1) = (−1)ⁿ⁺¹·n!·ζ(n+1); ψ⁽ⁿ⁾(½) = (−1)ⁿ⁺¹·n!·(2ⁿ⁺¹−1)·ζ(n+1).
        assertClose(-2.0 * zeta3, 1.0.polygamma(2), 1e-11, "ψ₂(1) = −2ζ(3)")
        assertClose(-14.0 * zeta3, 0.5.polygamma(2), 1e-10, "ψ₂(0.5) = −14ζ(3)")
        assertClose(PI.pow(4) / 15.0, 1.0.polygamma(3), 1e-10, "ψ₃(1) = π⁴/15")
        assertClose(PI.pow(4), 0.5.polygamma(3), 1e-9, "ψ₃(0.5) = π⁴")
        assertClose(-24.0 * zeta5, 1.0.polygamma(4), 1e-9, "ψ₄(1) = −24ζ(5)")
        // Reflection probe: ψ₂(−0.5) = ψ₂(0.5) + 16 (one recurrence step).
        assertClose(16.0 - 14.0 * zeta3, (-0.5).polygamma(2), 1e-9, "ψ₂(−0.5)")
        // n = 0 delegates to digamma exactly.
        for (x in listOf(0.7, 3.2, -1.4)) {
            assertTrue(x.polygamma(0) == x.digamma(), "polygamma(0) must delegate to ψ at $x")
        }
        // Poles: even-order pole (odd n) → +∞ both sides; odd-order pole
        // (even n) → sign-indefinite, NaN — the trigamma/digamma conventions.
        assertTrue((-2.0).polygamma(1) == Double.POSITIVE_INFINITY, "ψ₁ pole at −2")
        assertTrue((-2.0).polygamma(3) == Double.POSITIVE_INFINITY, "ψ₃ pole at −2")
        assertTrue(0.0.polygamma(2).isNaN(), "ψ₂ pole at 0 is sign-indefinite")
    }

    @Test
    fun polygammaRecurrencesHold() {
        // ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ·n!/xⁿ⁺¹, probed for n = 1..5 across the
        // reflection (x < 0.5), shift, and asymptotic regions.
        for (n in 1..5) {
            var factN = 1.0
            for (i in 2..n) factN *= i
            val sign = if (n % 2 == 0) 1.0 else -1.0
            for (x in listOf(0.2, 0.7, 1.3, 3.9, 7.5, 25.0)) {
                val step = sign * factN / x.pow(n + 1)
                val lhs = (x + 1.0).polygamma(n) - x.polygamma(n)
                assertClose(step, lhs, 1e-10 * maxOf(1.0, abs(step)), "ψ⁽$n⁾ recurrence at $x")
            }
        }
    }

    @Test
    fun polygammaDerivativeChainMatchesCentralDifferences() {
        // The oracle the §0.4.405 gradient rules rest on: d ψ⁽ⁿ⁾ = ψ⁽ⁿ⁺¹⁾,
        // checked numerically for n = 1..4 (and thereby transitively down to
        // digamma, whose own chain §0.4.402 pinned). Relative tolerance — the
        // magnitudes grow factorially with n at small x.
        val h = 1e-5
        for (n in 1..4) {
            for (x in listOf(0.8, 1.5, 2.5, 4.2, 9.0)) {
                val numeric = ((x + h).polygamma(n) - (x - h).polygamma(n)) / (2.0 * h)
                val want = x.polygamma(n + 1)
                assertClose(want, numeric, 1e-6 * maxOf(1.0, abs(want)), "central-diff ψ⁽$n⁾′ vs ψ⁽${n + 1}⁾ at $x")
            }
        }
    }

    @Test
    fun polygammaRefusesOutOfRangeOrders() {
        assertFailsWith<IllegalArgumentException> { 1.5.polygamma(-1) }
        assertFailsWith<IllegalArgumentException> { 1.5.polygamma(101) }
    }

    @Test
    fun polygammaOverloadSetAgreesWithDoubleKernel() {
        for (p in listOf(0.5f, 2.5f, 7.3f)) {
            val want = p.toDouble().polygamma(2)
            assertClose(want, p.polygamma(2).toDouble(), 1e-5 * maxOf(1.0, abs(want)), "Float.polygamma(2) at $p")
            val fs: DScalar = FloatScalar(p)
            assertClose(
                want, (fs.polygamma(2) as FloatScalar).v.toDouble(),
                1e-5 * maxOf(1.0, abs(want)), "DScalar.polygamma(2) at $p",
            )
            val ds: DScalar = DoubleScalar(p.toDouble())
            assertClose(want, (ds.polygamma(2) as DoubleScalar).v, 1e-12, "DoubleScalar.polygamma(2) at $p")
        }
    }

    @Test
    fun scalarOverloadSetsAgreeWithDoubleKernels() {
        val probes = listOf(0.5f, 2.5f, 7.3f)
        for (p in probes) {
            val want = p.toDouble()
            assertClose(want.lgamma(), p.lgamma().toDouble(), 1e-6, "Float.lgamma at $p")
            assertClose(want.digamma(), p.digamma().toDouble(), 1e-6, "Float.digamma at $p")
            assertClose(want.trigamma(), p.trigamma().toDouble(), 1e-6, "Float.trigamma at $p")
            val fs: DScalar = FloatScalar(p)
            assertClose(want.lgamma(), (fs.lgamma() as FloatScalar).v.toDouble(), 1e-6, "DScalar.lgamma at $p")
            val ds: DScalar = DoubleScalar(want)
            assertClose(want.digamma(), (ds.digamma() as DoubleScalar).v, 1e-12, "DScalar.digamma at $p")
            assertClose(want.trigamma(), (ds.trigamma() as DoubleScalar).v, 1e-12, "DScalar.trigamma at $p")
        }
    }
}
