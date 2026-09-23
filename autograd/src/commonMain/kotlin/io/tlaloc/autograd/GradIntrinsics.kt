package io.tlaloc.autograd

/**
 * The user-facing gradient intrinsics: the API surface the Tlaloc K2
 * compiler plugin recognizes (by these exact FQNs) and rewrites at compile
 * time into synthesized gradient code
 * (`TlalocIntrinsicCallChecker` + `DxirToIrSynthesis`).
 *
 * **These bodies are the no-plugin fallback.** With the plugin applied
 * (see docs/GETTING_STARTED.md), a call whose lambda lowers is replaced
 * wholesale — the body below never runs. Without the plugin (or when
 * you invoke the returned function after a lowering fallback that the
 * plugin's tape path doesn't cover), you get an exception that says
 * exactly what to do, instead of silently-wrong identity results.
 *
 * For plugin-free work, the [Tracer]-capture API ([gradWithScalars],
 * [valueAndGradWithScalars], and the `Grad.kt` family) remains available —
 * it runs the same engine as the plugin route: the traced
 * lambda is captured via `Tape.toDxirFunction`, differentiated by
 * `DxirReverseTransform`, and evaluated by `DxirInterpreter`. The only
 * difference is trace-at-runtime vs rewrite-at-compile-time.
 */
internal fun pluginMissing(name: String): Nothing = throw IllegalStateException(
    "Tlaloc: `$name { }` was not rewritten at compile time, so this fallback body ran and " +
        "there is no gradient to return. One of these is the cause:\n" +
        "  (1) the Tlaloc compiler plugin is not applied to the module that contains this call — " +
        "apply the Gradle plugin `id(\"io.github.pedronahum.tlaloc\")`, or add " +
        "`io.github.pedronahum:tlaloc-compiler-plugin` to kotlinCompilerPluginClasspath " +
        "(docs/GETTING_STARTED.md);\n" +
        "  (2) the plugin is applied and could not compile this call, and the build sets " +
        "strictLowering to false (`tlaloc { strictLowering.set(false) }`, or " +
        "-P plugin:io.tlaloc.plugin:strictLowering=false), which turns that compile error into a " +
        "warning — the compile log has a `w: Tlaloc ...` warning at this call site that names the " +
        "reason; remove the option to get it as an error;\n" +
        "  (3) `$name` was not called as `$name { ... }` with the lambda written at the call site " +
        "(for example through a reference to `$name` itself), so the plugin had no call to rewrite.\n" +
        "The Tracer-capture API (io.tlaloc.autograd.gradWithScalars) computes gradients without the plugin.",
)

/** Gradient of a scalar-valued function of one tensor/value argument. */
fun <A, R> grad(f: (A) -> R): (A) -> A = { _ -> pluginMissing("grad") }

/** Gradient of a scalar-valued function of two arguments. */
fun <A, B, R> grad2(f: (A, B) -> R): (A, B) -> Pair<A, B> = { _, _ -> pluginMissing("grad2") }

/** Value and gradient of a scalar-valued function of one argument. */
fun <A, R> valueAndGrad(f: (A) -> R): (A) -> Pair<R, A> = { _ -> pluginMissing("valueAndGrad") }

/** Value and gradient of a scalar-valued function of two arguments. */
fun <A, B, R> valueAndGrad2(f: (A, B) -> R): (A, B) -> Triple<R, A, B> =
    { _, _ -> pluginMissing("valueAndGrad2") }

/**
 * Gradient of a scalar-valued function of three arguments. The plugin's
 * reverse path is arity-agnostic (`DxirReverseTransform` emits
 * `(*params) → (*grads)` for any arity; synthesis boxes 3 returns as
 * [Triple] and 4 as [Quadruple]). The [Tracer]-tape `grad3` /
 * `valueAndGrad3` overloads in `Grad.kt` coexist with these the same way
 * the tape `grad2` does with the intrinsic `grad2`: lambda parameter types
 * disambiguate.
 */
fun <A, B, C, R> grad3(f: (A, B, C) -> R): (A, B, C) -> Triple<A, B, C> =
    { _, _, _ -> pluginMissing("grad3") }

/** Value and all three gradients of a scalar-valued function of three arguments. */
fun <A, B, C, R> valueAndGrad3(f: (A, B, C) -> R): (A, B, C) -> Quadruple<R, A, B, C> =
    { _, _, _ -> pluginMissing("valueAndGrad3") }

/**
 * Forward-mode AD, backed by [io.tlaloc.ir.passes.DxirForwardTransform].
 * DiffKT's `forwardDerivative` / `primalAndForwardDerivative`.
 *
 * The Tlaloc idiom curries like [grad]: `jvp(f)` returns a function of
 * `(x, dx)` — the primal input and a tangent (perturbation direction) of the
 * same type — producing the directional derivative `dy = J_f(x)·dx`, computed
 * in ONE forward pass (dual-number semantics), NOT by finite differences.
 * DiffKT spells the same thing as `jvp(x, v, f)` (all args at the call site);
 * currying `f` first mirrors `grad` and lets the plugin lower the lambda body
 * exactly as it does for reverse mode.
 *
 * [valueAndJvp] additionally returns the primal output `y` alongside `dy` —
 * both fall out of the forward transform's `(x, dx) → (y, dy)` output for free.
 *
 * These are the no-plugin fallbacks (see [pluginMissing]); with the plugin the
 * call is replaced wholesale by [io.tlaloc.ir.passes.DxirForwardTransform] +
 * synthesis. Scope: single argument, straight-line bodies (the forward
 * transform's scope; region-bearing bodies fall back to the tape).
 */
fun <A, R> jvp(f: (A) -> R): (A, A) -> R = { _, _ -> pluginMissing("jvp") }

/** Primal value and directional derivative in one pass: `(x, dx) -> (y, dy)`. */
fun <A, R> valueAndJvp(f: (A) -> R): (A, A) -> Pair<R, R> =
    { _, _ -> pluginMissing("valueAndJvp") }

/**
 * The seeded-cotangent surface (DiffKT's `vjp` /
 * `primalAndPullback`): reverse mode generalised to
 * TENSOR-valued `f`. `vjp(f)` returns `(x, ȳ) → x̄` — the pullback of a
 * user-supplied cotangent `ȳ` (of `f`'s OUTPUT type) through `f` at `x`,
 * computed in ONE reverse pass. `grad(f)` is exactly `vjp(f)` with `ȳ` fixed
 * to the unit seed of a scalar `f`; conversely a full Jacobian is `n` calls
 * of `vjp` over the output basis (which is what [jacobian] loops for you).
 *
 * Unlike [jacobian] there is no runtime assembly helper: the synthesised
 * seeded pass IS the replacement — the plugin runs
 * `DxirReverseTransform(seedAsParam = true)` over the lambda, whose output
 * signature `(upstream, x) → x̄` is reordered to the declared `(x, ȳ)`.
 *
 * These are the no-plugin fallbacks (see [pluginMissing]); like `jacobian`
 * there is no runtime-tape path — a failed synthesis is a loud error at
 * first call. Scope: single argument, straight-line bodies, single return.
 */
fun <A, R> vjp(f: (A) -> R): (A, R) -> A = { _, _ -> pluginMissing("vjp") }

/** Primal value and seeded pullback in one pass: `(x, ȳ) -> (y, x̄)`. */
fun <A, R> valueAndVjp(f: (A) -> R): (A, R) -> Pair<R, A> =
    { _, _ -> pluginMissing("valueAndVjp") }

/**
 * The two-argument seeded-cotangent surface. It follows the
 * primals-then-seeds order of `grad2`/`jvp2`: `vjp2(f)` returns
 * `(x, w, ȳ) → (x̄, w̄)` — the pullback of a user-supplied cotangent `ȳ`
 * (of `f`'s OUTPUT type, cotangent LAST) through `f` at `(x, w)`, both
 * gradients from ONE reverse pass.
 * `DxirReverseTransform(seedAsParam = true)` emits
 * `(upstream, *params) → (*grads)` for any arity (the COARSENED
 * `gradient_body` signature); the plugin rotates the parameters into the
 * declared order.
 */
fun <A, B, R> vjp2(f: (A, B) -> R): (A, B, R) -> Pair<A, B> =
    { _, _, _ -> pluginMissing("vjp2") }

/** Primal value and both seeded pullbacks in one pass: `(x, w, ȳ) -> (y, x̄, w̄)`. */
fun <A, B, R> valueAndVjp2(f: (A, B) -> R): (A, B, R) -> Triple<R, A, B> =
    { _, _, _ -> pluginMissing("valueAndVjp2") }

/**
 * Forward mode for a two-argument function, the `grad2` of the forward
 * pair. The curried result takes the
 * primals then the tangents — `(x, w, dx, dw) -> dy` — which is the parameter
 * order [io.tlaloc.ir.passes.DxirForwardTransform] itself emits (all primals,
 * then `d_`-prefixed tangents), so nothing has to be permuted on the way in.
 *
 * `dy` is the directional derivative along the tangent PAIR: for a bilinear
 * `f` it is the product rule's sum, `∂f/∂x·dx + ∂f/∂w·dw`, in one forward pass.
 * This is what makes forward-mode conv differentiable with a real `(x, w)`
 * kernel rather than the self-convolution a single-argument `jvp` forces.
 */
fun <A, B, R> jvp2(f: (A, B) -> R): (A, B, A, B) -> R = { _, _, _, _ -> pluginMissing("jvp2") }

/** Primal value and directional derivative for a two-argument function. */
fun <A, B, R> valueAndJvp2(f: (A, B) -> R): (A, B, A, B) -> Pair<R, R> =
    { _, _, _, _ -> pluginMissing("valueAndJvp2") }
