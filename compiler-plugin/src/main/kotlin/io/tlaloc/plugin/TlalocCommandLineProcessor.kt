package io.tlaloc.plugin

import org.jetbrains.kotlin.compiler.plugin.AbstractCliOption
import org.jetbrains.kotlin.compiler.plugin.CliOption
import org.jetbrains.kotlin.compiler.plugin.CliOptionProcessingException
import org.jetbrains.kotlin.compiler.plugin.CommandLineProcessor
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.config.CompilerConfigurationKey

/**
 * §0.4.450 — the plugin's first CLI options: the readable-reverse dump (the
 * north star's compile-time half — docs/AD_SINGLE_ENGINE_AUDIT.md, surface 2).
 * Until now every plugin knob rode a system property (`tlaloc.cache.dir`,
 * `tlaloc.soi.enabled`, …) because they tune the COMPILER PROCESS; the dump is
 * a per-compilation USER request, so it takes the front door:
 *
 *   -P plugin:io.tlaloc.plugin:dumpGradSource=true
 *   -P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>
 *
 * `dumpGradSource=true` emits, for every `grad {}` / `grad2 {}` /
 * `valueAndGrad {}` (…the reverse-gradient family) lambda the plugin
 * successfully synthesises, a compiler INFO message headed by the lambda's
 * source location and carrying the REVERSE-TRANSFORMED gradient rendered as
 * Kotlin source by `DxirFunction.toKotlinSource()` (§0.4.449). The dump
 * happens at the dxir level BEFORE synthesis: the gradient the user reads is
 * the SAME function the synthesis then compiles to bytecode.
 * `dumpGradSourceDir=<dir>` implies the message form and ADDITIONALLY writes
 * each rendering as a `.kt` file named after the lambda's source location.
 */
@OptIn(ExperimentalCompilerApi::class)
class TlalocCommandLineProcessor : CommandLineProcessor {

    override val pluginId: String = PLUGIN_ID

    override val pluginOptions: Collection<AbstractCliOption> = listOf(
        DUMP_GRAD_SOURCE_OPTION,
        DUMP_GRAD_SOURCE_DIR_OPTION,
    )

    override fun processOption(
        option: AbstractCliOption,
        value: String,
        configuration: CompilerConfiguration,
    ) = when (option.optionName) {
        DUMP_GRAD_SOURCE_OPTION.optionName ->
            configuration.put(DUMP_GRAD_SOURCE_KEY, value.toBooleanStrictOrNull() ?: (value == "true"))
        DUMP_GRAD_SOURCE_DIR_OPTION.optionName ->
            configuration.put(DUMP_GRAD_SOURCE_DIR_KEY, value)
        else -> throw CliOptionProcessingException("Unknown option: ${option.optionName}")
    }

    companion object {
        const val PLUGIN_ID: String = "io.tlaloc.plugin"

        val DUMP_GRAD_SOURCE_OPTION = CliOption(
            optionName = "dumpGradSource",
            valueDescription = "true|false",
            description = "Emit each successfully synthesised grad {} gradient as readable " +
                "Kotlin source (a compiler INFO message headed by the lambda's location)",
            required = false,
            allowMultipleOccurrences = false,
        )

        val DUMP_GRAD_SOURCE_DIR_OPTION = CliOption(
            optionName = "dumpGradSourceDir",
            valueDescription = "<directory>",
            description = "Like dumpGradSource, and additionally write each rendered gradient " +
                "as a .kt file (named after the lambda's source location) under this directory",
            required = false,
            allowMultipleOccurrences = false,
        )

        val DUMP_GRAD_SOURCE_KEY: CompilerConfigurationKey<Boolean> =
            CompilerConfigurationKey.create("dump synthesised gradients as Kotlin source")

        val DUMP_GRAD_SOURCE_DIR_KEY: CompilerConfigurationKey<String> =
            CompilerConfigurationKey.create("directory for dumped gradient .kt files")
    }
}
