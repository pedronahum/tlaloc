package io.tlaloc.autograd

/**
 * §0.4.355 — the user-facing gradient intrinsics: the API surface the
 * Tlaloc K2 compiler plugin recognizes (by these exact FQNs) and
 * rewrites at compile time into synthesized gradient code
 * (`TlalocIntrinsicCallChecker` + `DxirToIrSynthesis`). Until this
 * commit these existed only as per-test stubs — a consumer of the
 * published artifacts had no `grad {}` to call. Now they ship.
 *
 * **These bodies are the no-plugin fallback.** With the plugin applied
 * (see docs/GETTING_STARTED.md), a call whose lambda lowers is replaced
 * wholesale — the body below never runs. Without the plugin (or when
 * you invoke the returned function after a lowering fallback that the
 * plugin's tape path doesn't cover), you get an exception that says
 * exactly what to do, instead of silently-wrong identity results.
 *
 * For plugin-free scalar work, the [Tracer]-tape API
 * ([gradWithScalars], [valueAndGradWithScalars]) remains available.
 */
private fun pluginMissing(name: String): Nothing = throw IllegalStateException(
    "Tlaloc: `$name { }` requires the Tlaloc K2 compiler plugin, which rewrites this call at " +
        "compile time. Add `io.tlaloc:compiler-plugin` to kotlinCompilerPluginClasspath " +
        "(docs/GETTING_STARTED.md), or use the Tracer-tape API (io.tlaloc.autograd.gradWithScalars) " +
        "for plugin-free scalar gradients.",
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
