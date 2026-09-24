package io.tlaloc.plugin

/**
 * The plugin's per-compilation user knobs, resolved ONCE in
 * [TlalocCompilerPluginRegistrar] from the [TlalocCommandLineProcessor] options and
 * then handed to both halves of the plugin: the FIR checker (through
 * [TlalocFirExtensionRegistrar] → [TlalocCheckersExtension] →
 * [TlalocIntrinsicCallChecker]) and the IR extension.
 *
 * It is a VALUE passed down the extension-construction chain, deliberately not a
 * process-global mutable (the [TlalocLoweringHandoff] table travels the same way):
 * `strictLowering` decides
 * whether a diagnostic is an error, and two modules compiled concurrently in one
 * Kotlin daemon must not be able to flip each other's severity.
 *
 * @property dumpLoweredIr developer introspection: dump the lowered dxir for every
 *   recognised intrinsic lambda. OFF by default. When on, it emits
 *   two WARNINGs per `grad {}`, which fails any project compiling with `-Werror`
 *   (`allWarningsAsErrors = true`).
 * @property strictLowering an intrinsic lambda the plugin cannot lower is a
 *   compile-time ERROR (default). Setting this to false downgrades it to a WARNING;
 *   the call is then left unrewritten and `io.tlaloc.autograd.pluginMissing` throws
 *   an `IllegalStateException` at the first call.
 */
data class TlalocPluginOptions(
    val dumpLoweredIr: Boolean = false,
    val strictLowering: Boolean = true,
)
