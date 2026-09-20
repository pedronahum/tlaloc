package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random

/**
 * §0.4.415 — Phase B5: user-defined custom derivatives (the ratified
 * Candidate A call-form of docs/CUSTOM_DERIVATIVES_DESIGN.md). `customVjp(f,
 * vjpFn)` attaches a USER-written reverse-mode adjoint to `f`: inside a
 * differentiated lambda (`grad {}` / `vjp {}`), applying the returned function
 * runs `f` in the value stream while reverse mode splices `vjpFn` — verbatim,
 * with NO fallback to composing `f`'s own rules — as the gradient.
 * `vjpFn(upstream, x)` receives the cotangent flowing into `f`'s result and
 * the primal argument, and must return `d_x` shaped like `x` (asserted at
 * runtime — a wrong-shaped return fails loudly, never silently corrupts the
 * gradient). Use cases: a cheaper or numerically stabler analytical adjoint,
 * and derivatives that DIFFER from the body's math by intent —
 * straight-through estimators, gradient stopping (`customVjp(f = { it },
 * vjpFn = { u, _ -> u * 0f })` is `stopGradient`).
 *
 * The K2 plugin's FIR lowering recognises the call-form APPLIED inside the
 * lambda being lowered (directly, or through a local `val` applied later in
 * the same body) and emits one `OpKind.COARSENED` node with the two lowered
 * lambda bodies as `primal_body` / `gradient_body` and `user_gradient = true`.
 * v1 scope: both arguments must be lambda literals at the call site; the
 * result must not ESCAPE the lambda (passing it to another function or
 * re-binding it refuses loudly at compile time); captured locals must be
 * literal-initialised.
 *
 * **Forward mode refuses**: `jvp {}` over a body containing a `customVjp`
 * application errors loudly naming `user_gradient` — auto-differentiating
 * `f`'s primal would silently disagree with a deliberately divergent user
 * adjoint (the ratified refuse-unless-jvpFn policy; `customVjpJvp` is the
 * recorded tail that lifts it).
 *
 * **Host-stub asymmetry vs [grad]**: without the plugin (or outside any
 * differentiated context) `customVjp` does NOT throw [pluginMissing] — it
 * returns `f` itself, because applying the primal is exactly the right
 * plain-Kotlin meaning of the call; only the derivative ATTACHMENT needs the
 * plugin, and it simply doesn't exist here. `grad`'s stub throws because a
 * gradient computed as "identity" would be silently wrong; `customVjp`'s
 * value-stream behaviour is `f` under every interpretation.
 */
fun <A, R> customVjp(f: (A) -> R, vjpFn: (R, A) -> A): (A) -> R = f

/**
 * §0.4.415 — the two-argument [customVjp]: `vjpFn(upstream, a, b)` returns
 * `Pair(d_a, d_b)`, which the FIR lowering unboxes to the COARSENED
 * `gradient_body`'s 2-return convention. The `Pair` must be constructed
 * directly at the return position (`Pair(da, db)` or `da to db`).
 */
fun <A, B, R> customVjp2(f: (A, B) -> R, vjpFn: (R, A, B) -> Pair<A, B>): (A, B) -> R = f

/**
 * §0.4.415 — the result of [checkCustomVjp]: both sides of the JVP⇄VJP
 * inner-product identity `⟨ȳ, J·v⟩ = ⟨vjpFn(ȳ, x), v⟩` at one random probe,
 * with `J·v` estimated by central differences of `f`.
 */
data class CustomVjpCheck(
    /** `⟨ȳ, J·v⟩` — the forward side, from central differences of `f`. */
    val forwardInner: Float,
    /** `⟨vjpFn(ȳ, at), v⟩` — the reverse side, from the user's adjoint. */
    val vjpInner: Float,
    /** `|forwardInner − vjpInner| / max(1, |forwardInner|, |vjpInner|)`. */
    val relativeError: Float,
    /** `relativeError <= tolerance`. */
    val passed: Boolean,
)

/**
 * §0.4.415 — the OPT-IN debug oracle for a [customVjp] pair (the ratified
 * third decision of docs/CUSTOM_DERIVATIVES_DESIGN.md §6): numerically checks
 * the JVP⇄VJP cross-identity `⟨ȳ, J_f(at)·v⟩ = ⟨vjpFn(ȳ, at), v⟩` at random
 * directions `v`, `ȳ` drawn from a seeded [Random], with the directional
 * derivative `J·v` estimated by CENTRAL DIFFERENCES of `f` — pure host math on
 * `DTensor`s, requiring neither the compiler plugin nor the IR.
 *
 * A mathematically correct `vjpFn` (use case 1 — a stabler or cheaper
 * spelling of the true adjoint) passes within [tolerance]. A DELIBERATELY
 * divergent one — straight-through estimators, stopGradient, any use-case-3
 * adjoint — **fails this check BY DESIGN**: divergence from the primal's math
 * is the feature. That is why this is a test-time helper you call on the
 * pairs you intend to be exact, never a default gate the pipeline runs.
 *
 * [eps] trades truncation against cancellation in the central difference;
 * the default suits inputs and gradients of order ~1.
 */
fun <SA : Shape, SR : Shape> checkCustomVjp(
    f: (DTensor<SA, F32>) -> DTensor<SR, F32>,
    vjpFn: (DTensor<SR, F32>, DTensor<SA, F32>) -> DTensor<SA, F32>,
    at: DTensor<SA, F32>,
    seed: Int = 0,
    eps: Float = 1e-3f,
    tolerance: Float = 1e-2f,
): CustomVjpCheck {
    val rnd = Random(seed)
    val x = at.hostF32()
    val v = FloatArray(x.size) { rnd.nextFloat() * 2f - 1f }

    fun perturbed(sign: Float): DTensor<SA, F32> =
        DTensor(HostF32Storage(FloatArray(x.size) { x[it] + sign * eps * v[it] }), at.dims.copyOf(), F32)

    val yPlus = f(perturbed(+1f))
    val yMinus = f(perturbed(-1f))
    val yp = yPlus.hostF32()
    val ym = yMinus.hostF32()
    require(yp.size == ym.size) {
        "checkCustomVjp: f returned differing sizes (${yp.size} vs ${ym.size}) at x ± eps·v"
    }
    // J·v ≈ (f(x + eps·v) − f(x − eps·v)) / (2·eps), elementwise over y.
    val jv = FloatArray(yp.size) { (yp[it] - ym[it]) / (2f * eps) }
    val yBarData = FloatArray(yp.size) { rnd.nextFloat() * 2f - 1f }
    var forwardInner = 0f
    for (i in jv.indices) forwardInner += yBarData[i] * jv[i]

    val yBar = DTensor<SR, F32>(HostF32Storage(yBarData.copyOf()), yPlus.dims.copyOf(), F32)
    val xBar = vjpFn(yBar, at).hostF32()
    require(xBar.size == x.size) {
        "checkCustomVjp: vjpFn returned size ${xBar.size} for an input of size ${x.size} — " +
            "the VJP shape contract (d_x must match x's shape) is violated"
    }
    var vjpInner = 0f
    for (i in xBar.indices) vjpInner += xBar[i] * v[i]

    val relativeError =
        abs(forwardInner - vjpInner) / max(1f, max(abs(forwardInner), abs(vjpInner)))
    return CustomVjpCheck(forwardInner, vjpInner, relativeError, relativeError <= tolerance)
}
