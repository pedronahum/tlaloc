package io.tlaloc.core

import kotlin.math.abs

// §0.4.411 — Phase C5 `integral(a, b, f)` (DiffKT `Integral.kt` parity):
// Romberg quadrature — the Richardson-extrapolated trapezoid tableau — with
// its derivatives wired through the fundamental theorem of calculus.
//
// The kernel is pure-Kotlin Double, the §0.4.402 SpecialFunctions convention:
// ONE source of truth that every future consumer (tensor host op, interpreter
// arm) calls verbatim. Scheme: T(i) is the trapezoid rule on 2^i intervals
// (each level reuses the previous level's evaluations — only the odd nodes
// are new), and column j Richardson-extrapolates with weight 1/(4^j − 1),
// cancelling the h^(2j) Euler–Maclaurin error term. Early exit when the
// diagonal stabilises: |R[i][i] − R[i−1][i−1]| ≤ tol·(1 + |R[i][i]|), a
// mixed absolute/relative test so tolerances behave for both ∫≈0 and large
// integrals. Defaults: maxDepth 16 (2^16 + 1 = 65537 evaluations at the
// deepest level, ample for anything smooth), tol 1e-8 in Double.
// `b < a` needs no special case — every formula is linear in (b − a) and
// yields the sign-flipped integral, pinned in IntegralTest.
//
// DERIVATIVES — the FTC contract this file certifies at host level:
//  - Bounds (reverse AND forward — scalar, so the cotangent and tangent
//    coefficients coincide): ∂/∂b ∫ₐᵇ f = f(b), ∂/∂a ∫ₐᵇ f = −f(a).
//    `integralWithBoundGrads` returns (value, dA, dB) wired ANALYTICALLY —
//    two extra evaluations of f, no differentiation of the quadrature loop
//    (differentiating through the tableau would converge to the same numbers
//    the slow way; IntegralTest pins the analytic wiring against central
//    differences of the quadrature itself). A JVP follows as
//    dValue = dB·ḃ + dA·ȧ; a VJP as (ā, b̄) = (dA·v̄, dB·v̄).
//  - Integrand parameters: d/dθ ∫ₐᵇ f(x; θ) dx = ∫ₐᵇ ∂f/∂θ dx
//    (differentiation under the integral sign — fixed bounds, smooth f).
//    At host level that is `integralWithParamGrad` below (§0.4.426), the
//    Leibniz sugar and the parameter-side twin of `integralWithBoundGrads`.
//    The `grad {}`-integrable surface is B5's custom-derivative machinery
//    (CUSTOM_DERIVATIVES_DESIGN.md, landed §0.4.415–416): the user spells
//    the primal as a fixed-node quadrature and attaches the Leibniz adjoint
//    via `customVjp`/`customVjpJvp` — certified E2E in the plugin's
//    IntegralGradientTest. The reusable `integral(a, b) { f }` region-op
//    spelling remains the plan's named C5 deferral.

/**
 * Result of [rombergIntegrate]: the extrapolated value, the deepest
 * trapezoid level `i` evaluated (2^i intervals), and whether the diagonal
 * met the tolerance before `maxDepth` (false means the value is the best
 * available diagonal, returned rather than thrown — the caller sees
 * convergence explicitly).
 */
data class RombergResult(
    val value: Double,
    val depth: Int,
    val converged: Boolean,
)

/**
 * Romberg quadrature of [f] over `[a, b]` with the full convergence report.
 * See the file header for the scheme; [integral] is the plain-value sugar.
 */
fun rombergIntegrate(
    a: Double,
    b: Double,
    maxDepth: Int = 16,
    tol: Double = 1e-8,
    f: (Double) -> Double,
): RombergResult {
    require(maxDepth in 1..30) { "rombergIntegrate: maxDepth must be in 1..30, got $maxDepth" }
    require(tol > 0.0) { "rombergIntegrate: tol must be positive, got $tol" }
    if (a == b) return RombergResult(0.0, 0, true)
    // prev = row i−1 of the tableau; only two rows are ever live.
    var prev = doubleArrayOf(0.5 * (b - a) * (f(a) + f(b)))
    var prevDiag = prev[0]
    for (i in 1..maxDepth) {
        val n = 1 shl i
        val h = (b - a) / n
        var sum = 0.0
        var k = 1
        while (k < n) { // odd nodes only — the even ones live in prev[0]
            sum += f(a + k * h)
            k += 2
        }
        val cur = DoubleArray(i + 1)
        cur[0] = 0.5 * prev[0] + h * sum
        var pow4 = 1.0
        for (j in 1..i) {
            pow4 *= 4.0
            cur[j] = cur[j - 1] + (cur[j - 1] - prev[j - 1]) / (pow4 - 1.0)
        }
        val diag = cur[i]
        if (abs(diag - prevDiag) <= tol * (1.0 + abs(diag))) {
            return RombergResult(diag, i, true)
        }
        prevDiag = diag
        prev = cur
    }
    return RombergResult(prevDiag, maxDepth, false)
}

/** Romberg quadrature of [f] over `[a, b]` at the default depth/tolerance. */
fun integral(a: Double, b: Double, f: (Double) -> Double): Double =
    rombergIntegrate(a, b, f = f).value

/** Float surface over the Double kernel (the scalar-overload convention). */
fun integral(a: Float, b: Float, f: (Float) -> Float): Float =
    rombergIntegrate(a.toDouble(), b.toDouble()) { x -> f(x.toFloat()).toDouble() }
        .value.toFloat()

/**
 * `∫ₐᵇ f` with its bound derivatives wired by the fundamental theorem of
 * calculus: `dA = −f(a)`, `dB = f(b)` — analytic, not differentiated through
 * the quadrature loop. Scalar, so the same two numbers serve forward mode
 * (`dValue = dB·ḃ + dA·ȧ`) and reverse mode (`ā = dA·v̄`, `b̄ = dB·v̄`).
 */
data class IntegralWithBoundGrads(
    val value: Double,
    val dA: Double,
    val dB: Double,
)

/** FTC-wired bound derivatives alongside the Romberg value; see the header. */
fun integralWithBoundGrads(
    a: Double,
    b: Double,
    f: (Double) -> Double,
): IntegralWithBoundGrads =
    IntegralWithBoundGrads(integral(a, b, f), -f(a), f(b))

/**
 * §0.4.426 — `∫ₐᵇ f(x; p) dx` with its parameter derivative wired by
 * differentiation under the integral sign (Leibniz, fixed bounds):
 * `dP = ∫ₐᵇ ∂f/∂p dx`, quadratured over the caller-supplied [dfdp] with the
 * same Romberg kernel — analytic in the integrand, never differentiated
 * through the tableau, the parameter-side twin of [integralWithBoundGrads].
 * Scalar, so the one number serves forward mode (`dValue = dP·ṗ`) and
 * reverse mode (`p̄ = dP·v̄`). The two quadratures run independently: the
 * derivative integrand earns its own adaptive depth.
 */
data class IntegralWithParamGrad(
    val value: Double,
    val dP: Double,
)

/** Leibniz-wired parameter derivative alongside the Romberg value. */
fun integralWithParamGrad(
    a: Double,
    b: Double,
    f: (Double) -> Double,
    dfdp: (Double) -> Double,
): IntegralWithParamGrad =
    IntegralWithParamGrad(integral(a, b, f), integral(a, b, dfdp))
