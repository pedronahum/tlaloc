package io.tlaloc.core

import kotlin.math.E
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.411 — Phase C5 `integral`: Romberg quadrature pinned against closed
 * forms, convergence/early-exit behavior pinned through [RombergResult], and
 * the FTC derivative wiring certified against central differences OF THE
 * QUADRATURE ITSELF (the independent oracle: if the analytic `−f(a)`/`f(b)`
 * wiring disagreed with how the computed value actually moves with its
 * bounds, these pins would catch it). The parameter-derivative pin documents
 * the B5-era differentiation contract d/dθ ∫ f(x;θ) dx = ∫ ∂f/∂θ dx at host
 * level, checked three ways (central difference / quadrature of ∂f/∂θ /
 * closed form).
 */
class IntegralTest {

    private fun assertClose(want: Double, got: Double, tol: Double, label: String) {
        assertTrue(abs(want - got) <= tol, "$label: got $got, want $want (|diff|=${abs(want - got)})")
    }

    @Test
    fun closedFormPins() {
        // ∫₀¹ x² dx = 1/3 — Romberg's depth-2 diagonal is already exact.
        assertClose(1.0 / 3.0, integral(0.0, 1.0) { x -> x * x }, 1e-12, "∫₀¹ x²")
        // ∫₀^π sin x dx = 2.
        assertClose(2.0, integral(0.0, PI) { x -> sin(x) }, 1e-9, "∫₀^π sin")
        // ∫₁ᵉ dx/x = 1.
        assertClose(1.0, integral(1.0, E) { x -> 1.0 / x }, 1e-9, "∫₁ᵉ 1/x")
        // Steep boundary layer (the tolerance path — needs real depth):
        // ∫₀¹ e^(−50x) dx = (1 − e^(−50))/50.
        assertClose(
            (1.0 - exp(-50.0)) / 50.0,
            integral(0.0, 1.0) { x -> exp(-50.0 * x) },
            1e-9,
            "∫₀¹ e^(−50x)",
        )
    }

    @Test
    fun floatSurfaceAndOrientation() {
        // Float overload rides the Double kernel — f32-visible tolerance.
        assertClose(
            2.0,
            integral(0.0f, PI.toFloat()) { x -> sin(x.toDouble()).toFloat() }.toDouble(),
            1e-6,
            "Float ∫₀^π sin",
        )
        // Reversed bounds flip the sign — no special case, pinned.
        val fwd = integral(0.3, 1.2) { x -> sin(x) }
        val rev = integral(1.2, 0.3) { x -> sin(x) }
        assertClose(-fwd, rev, 1e-12, "∫ᵇₐ = −∫ₐᵇ")
        // Empty interval is exactly zero.
        assertEquals(0.0, integral(2.5, 2.5) { x -> x * x }, "∫ₐᵃ = 0")
    }

    @Test
    fun convergenceAndEarlyExit() {
        // x² stabilises at depth 2 (Simpson's diagonal is exact; the depth-2
        // diagonal repeats it) — the early exit fires long before maxDepth.
        val quadratic = rombergIntegrate(0.0, 1.0) { x -> x * x }
        assertTrue(quadratic.converged, "x² must converge")
        assertEquals(2, quadratic.depth, "x² early-exits at depth 2")
        // The steep integrand needs the tolerance path: depth 8 at tol 1e-8.
        val steep = rombergIntegrate(0.0, 1.0) { x -> exp(-50.0 * x) }
        assertTrue(steep.converged, "e^(−50x) must converge at maxDepth 16")
        assertTrue(steep.depth >= 5, "e^(−50x) needs real depth, got ${steep.depth}")
        // Starved of depth it reports non-convergence honestly — value is the
        // best diagonal, not a throw.
        val starved = rombergIntegrate(0.0, 1.0, maxDepth = 4) { x -> exp(-50.0 * x) }
        assertTrue(!starved.converged, "maxDepth 4 must not satisfy tol 1e-8")
        assertClose((1.0 - exp(-50.0)) / 50.0, starved.value, 0.01, "starved value still coarse-close")
        // Guard rails.
        assertFailsWith<IllegalArgumentException> { rombergIntegrate(0.0, 1.0, maxDepth = 0) { it } }
        assertFailsWith<IllegalArgumentException> { rombergIntegrate(0.0, 1.0, tol = 0.0) { it } }
    }

    @Test
    fun ftcBoundDerivativesVsCentralDifferences() {
        // The FTC wiring −f(a)/f(b) against central differences of the
        // quadrature itself: tol 1e-12 quadratures keep the FD numerator's
        // quadrature noise (~1e-14) far below the 2h = 2e-4 denominator, and
        // h = 1e-4 keeps FD truncation (h²/6·|f″|) ~ 1.6e-9.
        val a = 0.3
        val b = 1.2
        val f = { x: Double -> sin(x) }
        val g = integralWithBoundGrads(a, b, f)
        assertClose(sin(b), g.dB, 1e-12, "dB = f(b) analytic")
        assertClose(-sin(a), g.dA, 1e-12, "dA = −f(a) analytic")
        val h = 1e-4
        val quad = { lo: Double, hi: Double -> rombergIntegrate(lo, hi, tol = 1e-12, f = f).value }
        val dBFd = (quad(a, b + h) - quad(a, b - h)) / (2.0 * h)
        val dAFd = (quad(a + h, b) - quad(a - h, b)) / (2.0 * h)
        assertClose(g.dB, dBFd, 1e-6, "dB vs central difference of the quadrature")
        assertClose(g.dA, dAFd, 1e-6, "dA vs central difference of the quadrature")
        // And the value itself is the plain integral.
        assertClose(quad(a, b), g.value, 1e-8, "value matches integral")
    }

    @Test
    fun parameterDerivativeUnderTheIntegralSign() {
        // The B5-era differentiation contract at host level, for the θ-family
        // f(x; θ) = e^(−θx) over [0, 1]: I(θ) = (1 − e^(−θ))/θ, so at θ = 2
        // dI/dθ = (3e^(−2) − 1)/4 in closed form. Three-way agreement:
        // central difference of I(θ) ≈ quadrature of ∂f/∂θ = closed form.
        val theta = 2.0
        val integralOf = { t: Double ->
            rombergIntegrate(0.0, 1.0, tol = 1e-12) { x -> exp(-t * x) }.value
        }
        val h = 1e-4
        val fd = (integralOf(theta + h) - integralOf(theta - h)) / (2.0 * h)
        val quadOfPartial =
            rombergIntegrate(0.0, 1.0, tol = 1e-12) { x -> -x * exp(-theta * x) }.value
        val closedForm = (3.0 * exp(-2.0) - 1.0) / 4.0
        assertClose(closedForm, quadOfPartial, 1e-10, "∫ ∂f/∂θ vs closed form")
        assertClose(quadOfPartial, fd, 1e-6, "central difference vs ∫ ∂f/∂θ")
    }

    @Test
    fun leibnizParamGradSugar() {
        // §0.4.426 — integralWithParamGrad, the Leibniz sugar. Linear family
        // f(x; a) = a·x² at a = 2 over [0, 1]: value = 2/3, and the task's
        // own flagship oracle d/da ∫₀¹ a·x² dx = ∫₀¹ x² dx = 1/3.
        val lin = integralWithParamGrad(0.0, 1.0, { x -> 2.0 * x * x }, { x -> x * x })
        assertClose(2.0 / 3.0, lin.value, 1e-9, "∫₀¹ 2x² value")
        assertClose(1.0 / 3.0, lin.dP, 1e-9, "d/da ∫₀¹ a·x² = 1/3")
        // θ-family f(x; θ) = e^(−θx) at θ = 2 — the same closed forms the
        // three-way pin above certifies, now through the sugar itself:
        // value = (1 − e^(−2))/2, dP = ∫₀¹ −x·e^(−2x) dx = (3e^(−2) − 1)/4.
        val theta = integralWithParamGrad(
            0.0,
            1.0,
            { x -> exp(-2.0 * x) },
            { x -> -x * exp(-2.0 * x) },
        )
        assertClose((1.0 - exp(-2.0)) / 2.0, theta.value, 1e-9, "θ-family value")
        assertClose((3.0 * exp(-2.0) - 1.0) / 4.0, theta.dP, 1e-9, "θ-family dP")
    }
}
