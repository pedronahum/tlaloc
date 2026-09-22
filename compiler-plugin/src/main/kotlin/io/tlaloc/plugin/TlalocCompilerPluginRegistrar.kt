package io.tlaloc.plugin

import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.compiler.plugin.CompilerPluginRegistrar
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.fir.extensions.FirExtensionRegistrarAdapter

@OptIn(ExperimentalCompilerApi::class)
class TlalocCompilerPluginRegistrar : CompilerPluginRegistrar() {
    override val pluginId: String = "io.tlaloc.plugin"

    override val supportsK2: Boolean = true

    override fun ExtensionStorage.registerExtensions(configuration: CompilerConfiguration) {
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
