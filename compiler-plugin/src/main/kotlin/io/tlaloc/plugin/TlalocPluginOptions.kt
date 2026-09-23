package io.tlaloc.plugin

/**
 * §0.4.499 — the plugin's per-compilation user knobs, resolved ONCE in
 * [TlalocCompilerPluginRegistrar] from the [TlalocCommandLineProcessor] options and
 * then handed to both halves of the plugin: the FIR checker (through
 * [TlalocFirExtensionRegistrar] → [TlalocCheckersExtension] →
 * [TlalocIntrinsicCallChecker]) and the IR extension.
 *
 * It is a VALUE passed down the extension-construction chain, deliberately not a
 * process-global mutable (the [TlalocLoweringHandoff] table travels the same way since
 * §0.4.514): `strictLowering` decides
 * whether a diagnostic is an error, and two modules compiled concurrently in one
 * Kotlin daemon must not be able to flip each other's severity.
 *
 * @property dumpLoweredIr developer introspection: dump the lowered dxir for every
 *   recognised intrinsic lambda. OFF by default — until §0.4.499 the plugin emitted
 *   two WARNINGs per `grad {}` into every consumer's build log, which also made the
 *   plugin unusable in any project compiling with `-Werror`
 *   (`allWarningsAsErrors = true`).
 * @property strictLowering an intrinsic lambda the plugin cannot lower is a
 *   compile-time ERROR (default). The pre-§0.4.499 behaviour — a WARNING, then an
 *   `IllegalStateException` from `io.tlaloc.autograd.pluginMissing` at the first
 *   call — is still reachable by setting this to false.
 */
data class TlalocPluginOptions(
    val dumpLoweredIr: Boolean = false,
    val strictLowering: Boolean = true,
)
