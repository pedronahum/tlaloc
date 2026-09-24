package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.CliDiagnostics
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.report
import org.jetbrains.kotlin.config.CommonConfigurationKeys
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.config.MessageCollectorAccess

/**
 * Where [KotlinVersionGuard]'s refusal (or, with the opt-out, its warning) goes.
 *
 * The guard runs inside compilers this plugin was not built against — that is its
 * whole job — so the reporting call cannot assume the compiler API it was compiled
 * against exists at run time. Two channels, tried in order:
 *
 * 1. [CliDiagnostics.COMPILER_PLUGIN_INITIALIZATION_ERROR] / `_WARNING` through
 *    `CompilerConfiguration.report`, the Kotlin 2.4 API for sourceless compiler
 *    diagnostics. Kotlin 2.4 marks direct `MessageCollector` access with the
 *    `@MessageCollectorAccess` opt-in and names this function as the replacement.
 * 2. The configuration's `MessageCollector`, when (1) does not link. Kotlin 2.3
 *    has neither `CliDiagnosticReportingKt` nor those two factories, so a 2.3
 *    compiler loading this plugin throws a [LinkageError] from (1). The collector key
 *    is present in every Kotlin 2.x compiler; the opt-in is compile-time only and
 *    leaves nothing in the bytecode that a 2.3 compiler could fail to resolve.
 *
 * With neither available an ERROR is thrown with the same text, so a refusal can
 * never be silent.
 */
internal object GuardReporter {

    /** Which channel carried the message. Returned so tests can pin the choice. */
    enum class Channel { CLI_DIAGNOSTIC, MESSAGE_COLLECTOR, DROPPED }

    fun report(
        configuration: CompilerConfiguration,
        severity: CompilerMessageSeverity,
        text: String,
    ): Channel = try {
        val factory = if (severity == CompilerMessageSeverity.ERROR) {
            CliDiagnostics.COMPILER_PLUGIN_INITIALIZATION_ERROR
        } else {
            CliDiagnostics.COMPILER_PLUGIN_INITIALIZATION_WARNING
        }
        configuration.report(factory, text)
        Channel.CLI_DIAGNOSTIC
    } catch (_: LinkageError) {
        reportToMessageCollector(configuration, severity, text)
    } catch (_: IllegalStateException) {
        // A host that did not configure the diagnostics collector the CLI pipeline sets.
        reportToMessageCollector(configuration, severity, text)
    }

    @OptIn(MessageCollectorAccess::class)
    internal fun reportToMessageCollector(
        configuration: CompilerConfiguration,
        severity: CompilerMessageSeverity,
        text: String,
    ): Channel {
        val collector = configuration.get(CommonConfigurationKeys.MESSAGE_COLLECTOR_KEY)
        return when {
            collector != null -> {
                collector.report(severity, text, null)
                Channel.MESSAGE_COLLECTOR
            }
            severity == CompilerMessageSeverity.ERROR -> error(text)
            else -> Channel.DROPPED
        }
    }
}
