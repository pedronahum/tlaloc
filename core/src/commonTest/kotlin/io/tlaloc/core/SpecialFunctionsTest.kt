package io.tlaloc.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ln
import kotlin.test.Test
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
