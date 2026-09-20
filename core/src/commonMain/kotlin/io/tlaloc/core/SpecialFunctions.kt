package io.tlaloc.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.tan

// §0.4.402 — Phase C1 special functions (DiffKT parity: its Dirichlet example
// depends on lgamma/digamma/polygamma). Pure-Kotlin Double implementations,
// shared verbatim by every engine: the scalar five-overload sets below, the
// tensor host ops in `:core/ops/HostOps.kt`, and the `:ir` interpreter arms
// (`:ir` depends on `:core`, so LGAMMA/DIGAMMA/TRIGAMMA evaluate through these
// exact functions — bit-for-bit agreement between host and interpreter by
// construction rather than by duplicated code).
//
// Algorithms (the standard schemes the slice spec names):
//  - `lgamma`: Lanczos approximation, g = 7 with the standard 9-coefficient
//    set, computed on x ≥ 0.5 and extended below by the reflection formula
//    Γ(x)Γ(1−x) = π/sin(πx), i.e. lgamma(x) = ln(π/|sin(πx)|) − lgamma(1−x).
//    Returns ln|Γ(x)| (the C `lgamma` convention); +∞ at the poles 0, −1, −2, …
//  - `digamma`: recurrence ψ(x) = ψ(x+1) − 1/x to shift the argument up past 8,
//    then the asymptotic series ψ(x) ≈ ln x − 1/(2x) − Σ B₂ₙ/(2n·x²ⁿ) with
//    Bernoulli terms through x⁻¹⁰; reflection ψ(x) = ψ(1−x) − π/tan(πx) for
//    x < 0.5 (covers the whole negative axis away from the poles).
//  - `trigamma` (ψ₁ = polygamma(1) — the INTERNAL op DIGAMMA's derivative
//    needs): recurrence ψ₁(x) = ψ₁(x+1) + 1/x², asymptotic ψ₁(x) ≈ 1/x +
//    1/(2x²) + Σ B₂ₙ/x²ⁿ⁺¹, reflection ψ₁(x) + ψ₁(1−x) = π²/sin²(πx).
//  - `polygamma` (§0.4.405 — ψ⁽ⁿ⁾ for general n ≥ 0, C1's recorded deferral):
//    n = 0 delegates to `digamma`; n ≥ 1 runs the same scheme one derivative
//    up per order — recurrence ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+1) − (−1)ⁿ·n!/xⁿ⁺¹ shifting
//    past 10 + n (the truncated Bernoulli series' error term grows with n, so
//    the shift threshold must too), then the n-times-differentiated digamma
//    asymptotic ψ⁽ⁿ⁾(x) ≈ (−1)ⁿ⁻¹[(n−1)!/xⁿ + n!/(2xⁿ⁺¹) +
//    Σₖ B₂ₖ·(2k+n−1)!/((2k)!·x²ᵏ⁺ⁿ)] through B₁₂, and on the negative axis
//    the differentiated reflection (−1)ⁿ·ψ⁽ⁿ⁾(1−x) = ψ⁽ⁿ⁾(x) + π·dⁿcot(πx)/dxⁿ
//    with dⁿcot(πx)/dxⁿ = πⁿ·Pₙ(cot πx), P₀(t) = t, Pₖ₊₁ = −(1+t²)·Pₖ′ (a
//    small polynomial-coefficient recurrence). The n = 1 arm computes through
//    this GENERAL scheme, not by delegation — SpecialFunctionsTest pins its
//    agreement with the independent `trigamma` kernel to 1e-11, which is the
//    hard validation of the differentiated-series derivation.
//
// Accuracy: ≤ ~1e-13 relative against reference values on the positive axis
// (pinned in SpecialFunctionsTest to 1e-9, far past the 1e-6 the plan asks),
// which is exact to the last bit once narrowed to F32 tensor storage.

// Lanczos g = 7, n = 9 — the standard coefficient set (Godfrey/Pugh lineage,
// the same numbers Boost and the GSL document for (g=7, n=9)).
private val LANCZOS_G7: DoubleArray = doubleArrayOf(
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

private const val LN_SQRT_TWO_PI = 0.9189385332046727 // ln √(2π)

/** ln|Γ(x)| via Lanczos (g = 7, 9 coefficients); reflection below 0.5. */
fun Double.lgamma(): Double {
    val x = this
    if (x.isNaN()) return Double.NaN
    if (x < 0.5) {
        // Poles at 0, −1, −2, …: +∞, the C convention. Detected exactly
        // (floor test) — sin(πx) at a large negative integer is a rounding
        // residue like 3.7e-16, never exactly 0.0.
        if (x == floor(x)) return Double.POSITIVE_INFINITY
        return ln(PI / abs(sin(PI * x))) - (1.0 - x).lgamma()
    }
    val z = x - 1.0
    var acc = LANCZOS_G7[0]
    for (i in 1 until LANCZOS_G7.size) acc += LANCZOS_G7[i] / (z + i)
    val t = z + 7.5 // z + g + 0.5
    return LN_SQRT_TWO_PI + (z + 0.5) * ln(t) - t + ln(acc)
}

/** ψ(x) = d/dx ln Γ(x): recurrence past 8, then the Bernoulli asymptotic series. */
fun Double.digamma(): Double {
    var x = this
    if (x.isNaN()) return Double.NaN
    if (x < 0.5) {
        // Reflection: ψ(x) = ψ(1−x) − π·cot(πx). Poles at 0, −1, −2, … (exact
        // floor test — tan(πx) at a negative integer is a rounding residue).
        if (x == floor(x)) return Double.NaN
        return (1.0 - x).digamma() - PI / tan(PI * x)
    }
    var acc = 0.0
    while (x < 8.0) {
        acc -= 1.0 / x
        x += 1.0
    }
    val inv = 1.0 / x
    val inv2 = inv * inv
    // ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸) − 1/(132x¹⁰)
    val series = inv2 * (
        1.0 / 12.0 - inv2 * (
            1.0 / 120.0 - inv2 * (
                1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0)
                )
            )
        )
    return acc + ln(x) - 0.5 * inv - series
}

/** ψ₁(x) = ψ′(x) (trigamma): recurrence past 8, then the Bernoulli asymptotic series. */
fun Double.trigamma(): Double {
    var x = this
    if (x.isNaN()) return Double.NaN
    if (x < 0.5) {
        // Reflection: ψ₁(x) + ψ₁(1−x) = π²/sin²(πx). Poles at 0, −1, −2, … (+∞,
        // exact floor test).
        if (x == floor(x)) return Double.POSITIVE_INFINITY
        val sinPiX = sin(PI * x)
        return PI * PI / (sinPiX * sinPiX) - (1.0 - x).trigamma()
    }
    var acc = 0.0
    while (x < 8.0) {
        acc += 1.0 / (x * x)
        x += 1.0
    }
    val inv = 1.0 / x
    val inv2 = inv * inv
    // 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − 1/(30x⁹) + 5/(66x¹¹)
    val series = inv * inv2 * (
        1.0 / 6.0 - inv2 * (
            1.0 / 30.0 - inv2 * (
                1.0 / 42.0 - inv2 * (1.0 / 30.0 - inv2 * 5.0 / 66.0)
                )
            )
        )
    return acc + inv + 0.5 * inv2 + series
}

// §0.4.405 — B₂ through B₁₂, the Bernoulli coefficients of the polygamma
// asymptotic series (one more term than digamma/trigamma carry: the series'
// truncation error grows combinatorially with the order n, and B₁₂ plus the
// order-dependent shift threshold keeps it ≤ ~1e-11 relative through n = 50).
private val POLYGAMMA_BERNOULLI: DoubleArray = doubleArrayOf(
    1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0, -691.0 / 2730.0,
)

/**
 * ψ⁽ⁿ⁾(x) = dⁿ⁺¹/dxⁿ⁺¹ ln Γ(x) (polygamma of order [n]). §0.4.405 — C1's
 * recorded deferral. n = 0 delegates to [digamma]; n ≥ 1 runs the
 * recurrence-shift + differentiated Bernoulli asymptotic series, with the
 * differentiated reflection formula (cot-derivative polynomial recurrence)
 * covering the negative axis. Poles at 0, −1, −2, … return +∞ for odd n (the
 * pole is even-order, both sides diverge up — the trigamma convention) and NaN
 * for even n (odd-order pole, sign-indefinite — the digamma convention).
 * Orders above 100 are refused: the series' factorial coefficients and the
 * reflection polynomial's coefficients leave Double's precision regime there,
 * and no honest value can be returned.
 */
fun Double.polygamma(n: Int): Double {
    require(n >= 0) { "polygamma order must be ≥ 0; got $n" }
    require(n <= 100) { "polygamma order must be ≤ 100 (Double-precision factorial regime); got $n" }
    if (n == 0) return digamma()
    var x = this
    if (x.isNaN()) return Double.NaN
    val signN = if (n % 2 == 0) 1.0 else -1.0 // (−1)ⁿ
    if (x < 0.5) {
        // Poles by the exact floor test (§0.4.402 lesson: sin/tan of π·integer
        // is a rounding residue, never 0.0).
        if (x == floor(x)) return if (n % 2 == 1) Double.POSITIVE_INFINITY else Double.NaN
        // Reflection, differentiated n times from ψ(1−x) = ψ(x) + π·cot(πx):
        //   (−1)ⁿ·ψ⁽ⁿ⁾(1−x) = ψ⁽ⁿ⁾(x) + π·dⁿcot(πx)/dxⁿ,
        // with dⁿcot(πx)/dxⁿ = πⁿ·Pₙ(cot πx) and the polynomial recurrence
        // P₀(t) = t, Pₖ₊₁ = −(1+t²)·Pₖ′ (dt/dx = −π(1+t²) chain rule).
        var p = doubleArrayOf(0.0, 1.0) // P₀(t) = t
        repeat(n) {
            val q = DoubleArray(p.size + 1)
            for (i in 1 until p.size) {
                val d = i * p[i] // Pₖ′ coefficient of t^{i−1}
                q[i - 1] -= d // −Pₖ′
                q[i + 1] -= d // −t²·Pₖ′
            }
            p = q
        }
        val t = 1.0 / tan(PI * x)
        var poly = 0.0
        for (i in p.indices.reversed()) poly = poly * t + p[i]
        return signN * (1.0 - x).polygamma(n) - PI.pow(n + 1) * poly
    }
    var factN = 1.0
    for (i in 2..n) factN *= i // n!
    val factNm1 = factN / n // (n−1)!
    // Recurrence ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+1) − (−1)ⁿ·n!/xⁿ⁺¹, shifted past 10 + n:
    // the first omitted series term is ~B₁₄·C(n+13,14)/T¹⁴ relative to the
    // leading term, which the order-dependent threshold pins ≤ ~1e-11.
    val threshold = 10.0 + n
    var acc = 0.0
    while (x < threshold) {
        acc -= signN * factN / x.pow(n + 1)
        x += 1.0
    }
    // ψ⁽ⁿ⁾(x) = (−1)ⁿ⁻¹[(n−1)!/xⁿ + n!/(2xⁿ⁺¹) + Σₖ B₂ₖ·(2k+n−1)!/((2k)!·x²ᵏ⁺ⁿ)]
    val u = 1.0 / x
    val u2 = u * u
    var series = factNm1 * u.pow(n) + factN * 0.5 * u.pow(n + 1)
    var c = factN * (n + 1) / 2.0 // c₁ = (n+1)!/2!
    var upow = u.pow(n + 2) // u^{2k+n} at k = 1
    for (k in 1..POLYGAMMA_BERNOULLI.size) {
        series += POLYGAMMA_BERNOULLI[k - 1] * c * upow
        val tk = 2 * k
        c *= (tk + n + 1).toDouble() * (tk + n) / ((tk + 2).toDouble() * (tk + 1))
        upow *= u2
    }
    return acc - signN * series
}

// --- Scalar lgamma / digamma five-overload sets (§0.4.402, Phase C1) ---
//
// The §0.4.377 pattern: `Float.lgamma()` / `Double.lgamma()` resolve at FQN
// `io.tlaloc.core.lgamma` / `.digamma`, mapped by the FIR lowering to
// `OpKind.LGAMMA` / `OpKind.DIGAMMA`; LgammaRule (`d lgamma = digamma`) and
// DigammaRule (`d digamma = trigamma`) cover the gradient side. There is no
// `kotlin.math` equivalent for any of these, so synthesis resolves the
// `io.tlaloc.core` extensions themselves (the sigmoid `irCoreScalarCall` path).
fun Float.lgamma(): Float = this.toDouble().lgamma().toFloat()
fun FloatScalar.lgamma(): FloatScalar = FloatScalar(v.lgamma())
fun DoubleScalar.lgamma(): DoubleScalar = DoubleScalar(v.lgamma())
fun DScalar.lgamma(): DScalar = when (this) {
    is FloatScalar -> lgamma()
    is DoubleScalar -> lgamma()
}

fun Float.digamma(): Float = this.toDouble().digamma().toFloat()
fun FloatScalar.digamma(): FloatScalar = FloatScalar(v.digamma())
fun DoubleScalar.digamma(): DoubleScalar = DoubleScalar(v.digamma())
fun DScalar.digamma(): DScalar = when (this) {
    is FloatScalar -> digamma()
    is DoubleScalar -> digamma()
}

// Trigamma is GRADIENT MACHINERY, not user parity surface: it exists so that
// DIGAMMA's adjoint/tangent synthesise (∇ digamma bodies contain TRIGAMMA
// nodes, and the generated IR calls these public symbols from user modules).
// It has NO FIR map entry of its own — the user spelling for ψ₁ is
// `polygamma(1)`, which the FIR normalises to the TRIGAMMA op (§0.4.405).
// Since §0.4.405 TRIGAMMA also HAS a VjpRule/tangent arm (d ψ₁ = ψ₂ =
// polygamma(2)) — second-order reverse through DIGAMMA composes.
fun Float.trigamma(): Float = this.toDouble().trigamma().toFloat()
fun FloatScalar.trigamma(): FloatScalar = FloatScalar(v.trigamma())
fun DoubleScalar.trigamma(): DoubleScalar = DoubleScalar(v.trigamma())
fun DScalar.trigamma(): DScalar = when (this) {
    is FloatScalar -> trigamma()
    is DoubleScalar -> trigamma()
}

// §0.4.405 — the polygamma(n) five-overload set. The order is a value
// parameter host-side but a COMPILE-TIME Int literal inside `grad {}` (the FIR
// folds it into the op: n = 0 → DIGAMMA, n = 1 → TRIGAMMA, n ≥ 2 →
// POLYGAMMA with an `order` attr). Exactly one positional parameter and no
// defaults — the K2 named-arg landmine: attr-bearing host ops must stay
// positional-arity-disambiguated.
fun Float.polygamma(n: Int): Float = this.toDouble().polygamma(n).toFloat()
fun FloatScalar.polygamma(n: Int): FloatScalar = FloatScalar(v.polygamma(n))
fun DoubleScalar.polygamma(n: Int): DoubleScalar = DoubleScalar(v.polygamma(n))
fun DScalar.polygamma(n: Int): DScalar = when (this) {
    is FloatScalar -> polygamma(n)
    is DoubleScalar -> polygamma(n)
}
