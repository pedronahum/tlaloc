package io.tlaloc.plugin

import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.compiler.plugin.CompilerPluginRegistrar
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.fir.extensions.FirExtensionRegistrarAdapter

@OptIn(ExperimentalCompilerApi::class)
class TlalocCompilerPluginRegistrar : CompilerPluginRegistrar() {
    override val pluginId: String = "io.tlaloc.plugin"

    override val supportsK2: Boolean = true

    override fun ExtensionStorage.registerExtensions(configuration: CompilerConfiguration) {
        // §0.4.503 (Tier 3, item 2) — THE VERSION GUARD RUNS FIRST, before a single
        // extension is registered, and that ordering is the whole design. Every K2
        // internal this plugin touches is reached from inside an extension: the FIR
        // checker, the FIR→dxir lowering, the IR generation extension. If the running
        // compiler's internals have moved, the failure happens once one of those is
        // constructed or invoked — deep inside `compileKotlin`, as a NoSuchMethodError
        // naming JetBrains classes. Refusing here, in the one method whose signature is
        // a published plugin API, is the difference between a sentence the user can act
        // on and a stack trace they cannot.
        //
        // Reported to the MessageCollector as an ERROR (which fails the compilation)
        // rather than thrown, per the house rule: a clean refusal, not a crash. The
        // collector is absent only in a host that configured none, and then there is no
        // channel to be clean on, so the guard throws with the same text.
        val allowUnsupportedKotlin =
            configuration.get(TlalocCommandLineProcessor.UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_KEY) ?: false
        val messageCollector = configuration.get(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY)
        val mayRegister = KotlinVersionGuard.check(
            found = KotlinVersionGuard.detectRunningCompilerVersion(),
            allowUnsupported = allowUnsupportedKotlin,
        ) { severity, text ->
            if (messageCollector != null) {
                messageCollector.report(severity, text, null)
            } else if (severity == CompilerMessageSeverity.ERROR) {
                error(text)
            }
        }
        if (!mayRegister) return

        // §0.4.499 — the per-compilation knobs, resolved once and handed to BOTH
        // halves of the plugin (FIR checker + IR extension) as a value.
        val options = TlalocPluginOptions(
            dumpLoweredIr = configuration.get(TlalocCommandLineProcessor.DUMP_LOWERED_IR_KEY) ?: false,
            strictLowering = configuration.get(TlalocCommandLineProcessor.STRICT_LOWERING_KEY) ?: true,
        )
        FirExtensionRegistrarAdapter.registerExtension(TlalocFirExtensionRegistrar(options))
        // §0.4.450 — the readable-reverse dump options (see TlalocCommandLineProcessor):
        // the dir form implies the message form.
        val dumpDir = configuration.get(TlalocCommandLineProcessor.DUMP_GRAD_SOURCE_DIR_KEY)
        val dump = (configuration.get(TlalocCommandLineProcessor.DUMP_GRAD_SOURCE_KEY) ?: false) ||
            dumpDir != null
        IrGenerationExtension.registerExtension(
            TlalocIrGenerationExtension(
                dumpGradSource = dump,
                dumpGradSourceDir = dumpDir,
                options = options,
            ),
        )
    }
}
