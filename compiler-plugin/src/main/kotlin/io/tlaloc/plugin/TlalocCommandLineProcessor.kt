package io.tlaloc.plugin

import org.jetbrains.kotlin.compiler.plugin.AbstractCliOption
import org.jetbrains.kotlin.compiler.plugin.CliOption
import org.jetbrains.kotlin.compiler.plugin.CliOptionProcessingException
import org.jetbrains.kotlin.compiler.plugin.CommandLineProcessor
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.config.CompilerConfigurationKey

/**
 * The plugin's CLI options. The first of them is the readable-reverse dump.
 * Knobs that tune the COMPILER PROCESS ride system properties
 * (`tlaloc.cache.dir`, `tlaloc.soi.enabled`, …); per-compilation USER requests
 * such as the dump are CLI options:
 *
 *   -P plugin:io.tlaloc.plugin:dumpGradSource=true
 *   -P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>
 *
 * `dumpGradSource=true` emits, for every `grad {}` / `grad2 {}` /
 * `valueAndGrad {}` (…the reverse-gradient family) lambda the plugin
 * successfully synthesises, a compiler INFO message headed by the lambda's
 * source location and carrying the REVERSE-TRANSFORMED gradient rendered as
 * Kotlin source by `DxirFunction.toKotlinSource()`. The dump
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
        DUMP_LOWERED_IR_OPTION,
        STRICT_LOWERING_OPTION,
        UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_OPTION,
    )

    override fun processOption(
        option: AbstractCliOption,
        value: String,
        configuration: CompilerConfiguration,
    ) = when (option.optionName) {
        DUMP_GRAD_SOURCE_OPTION.optionName ->
            configuration.put(DUMP_GRAD_SOURCE_KEY, parseBoolean(DUMP_GRAD_SOURCE_OPTION.optionName, value))
        DUMP_GRAD_SOURCE_DIR_OPTION.optionName ->
            configuration.put(DUMP_GRAD_SOURCE_DIR_KEY, value)
        DUMP_LOWERED_IR_OPTION.optionName ->
            configuration.put(DUMP_LOWERED_IR_KEY, parseBoolean(DUMP_LOWERED_IR_OPTION.optionName, value))
        STRICT_LOWERING_OPTION.optionName ->
            configuration.put(STRICT_LOWERING_KEY, parseBoolean(STRICT_LOWERING_OPTION.optionName, value))
        UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_OPTION.optionName ->
            configuration.put(
                UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_KEY,
                parseBoolean(UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_OPTION.optionName, value),
            )
        else -> throw CliOptionProcessingException("Unknown option: ${option.optionName}")
    }

    /**
     * A boolean option REFUSES a value it does not understand, by name,
     * instead of silently reading it as `false`: `dumpLoweredIr=ture` fails the
     * compilation and says which option and which value.
     */
    private fun parseBoolean(optionName: String, value: String): Boolean =
        value.toBooleanStrictOrNull() ?: throw CliOptionProcessingException(
            "Tlaloc plugin option '$optionName' expects true or false, got '$value'",
        )

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

        /**
         * Developer introspection: the lowered dxir for every recognised
         * intrinsic lambda. Off by default; see [TlalocPluginOptions.dumpLoweredIr].
         */
        val DUMP_LOWERED_IR_OPTION = CliOption(
            optionName = "dumpLoweredIr",
            valueDescription = "true|false",
            description = "Dump the lowered Tlaloc IR for every recognised grad {} / jvp {} / " +
                "vjp {} lambda (developer introspection; off by default so a working build is " +
                "silent). The IR-phase half is an INFO message; the FIR-phase half is a " +
                "WARNING, because K2's diagnostic DSL has no INFO severity — do not combine " +
                "this option with -Werror",
            required = false,
            allowMultipleOccurrences = false,
        )

        /**
         * The refusal severity for a lambda the plugin cannot lower. On by
         * default; see [TlalocPluginOptions.strictLowering].
         */
        val STRICT_LOWERING_OPTION = CliOption(
            optionName = "strictLowering",
            valueDescription = "true|false",
            description = "Refuse an unlowerable grad {} / jvp {} / vjp {} lambda at COMPILE " +
                "time, naming the lowering's own reason (default: true). Set false to get the " +
                "pre-0.1.0-alpha01 behaviour instead: a warning at compile time and an " +
                "IllegalStateException from the io.tlaloc.autograd fallback body at the first call",
            required = false,
            allowMultipleOccurrences = false,
        )

        /**
         * The escape hatch on [KotlinVersionGuard]. Off by
         * default: a Kotlin version outside the guard's range is a compile-time ERROR
         * and the plugin registers nothing. Set true to downgrade that to a WARNING and
         * run anyway.
         */
        val UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_OPTION = CliOption(
            optionName = "unsafeAllowUnsupportedKotlin",
            valueDescription = "true|false",
            description = "Register the plugin even when the running Kotlin compiler is " +
                "outside the version range it was built against (default: false, which is a " +
                "compile ERROR naming both versions). The plugin reads K2 FIR/IR internals " +
                "that are not a stable API, so 'unsafe' is literal: a NoSuchMethodError from " +
                "inside the compiler is the expected outcome, not a bug",
            required = false,
            allowMultipleOccurrences = false,
        )

        val DUMP_GRAD_SOURCE_KEY: CompilerConfigurationKey<Boolean> =
            CompilerConfigurationKey.create("dump synthesised gradients as Kotlin source")

        val DUMP_GRAD_SOURCE_DIR_KEY: CompilerConfigurationKey<String> =
            CompilerConfigurationKey.create("directory for dumped gradient .kt files")

        val DUMP_LOWERED_IR_KEY: CompilerConfigurationKey<Boolean> =
            CompilerConfigurationKey.create("dump the lowered dxir for every recognised intrinsic lambda")

        val STRICT_LOWERING_KEY: CompilerConfigurationKey<Boolean> =
            CompilerConfigurationKey.create("refuse an unlowerable intrinsic lambda at compile time")

        val UNSAFE_ALLOW_UNSUPPORTED_KOTLIN_KEY: CompilerConfigurationKey<Boolean> =
            CompilerConfigurationKey.create("register the plugin on an unsupported Kotlin version")
    }
}
