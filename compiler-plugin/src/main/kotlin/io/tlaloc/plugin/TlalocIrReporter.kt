package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.messages.CompilerMessageLocation
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.reportInfo
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactory1
import org.jetbrains.kotlin.ir.IrDiagnosticReporter
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.IrFile

/**
 * Every message the IR phase emits goes through here.
 *
 * Errors and warnings are compiler diagnostics reported through the IR plugin
 * context's [IrDiagnosticReporter], at the file, line and column of the intrinsic
 * call being rewritten. Errors fail the build before code generation and warnings
 * fail a `-Werror` build.
 *
 * Warnings are reported on the call itself (the factories are in [TlalocIrErrors]),
 * so `@Suppress` naming the factory silences them. Errors are reported without a
 * source element (the factory is in [TlalocIrSourcelessErrors]) so that `@Suppress`
 * cannot reach them: a refused call is left as written and throws when it runs, and
 * the only way to compile it anyway is `strictLowering=false`, which turns the error
 * into a warning. A warning with no call to point at (a whole-file failure, a call
 * the IR phase never found) is reported the same sourceless way, with the best
 * location there is.
 *
 * Informational output (the `dumpLoweredIr` and `dumpGradSource` dumps) is not a
 * diagnostic: it goes to the compiler's message output at INFO severity, so a
 * working program still compiles silently under `-Werror`.
 *
 * The severity rule for refusals lives here and nowhere else: an ERROR under
 * [TlalocPluginOptions.strictLowering] (the default), a WARNING when the build opted
 * out.
 *
 * Only reporter calls present in every supported Kotlin release are used:
 * `at(IrElement, IrFile)` and the sourceless `report(factory, message, location)`.
 */
internal class TlalocIrReporter(
    private val reporter: IrDiagnosticReporter,
    private val configuration: CompilerConfiguration,
    private val strictLowering: Boolean,
) {
    private var file: IrFile? = null
    private var call: IrElement? = null

    /** Start reporting for [irFile]; no call is current until [atCall]. */
    fun enterFile(irFile: IrFile?) {
        file = irFile
        call = null
    }

    /** Point subsequent diagnostics at [element], the call being rewritten. */
    fun atCall(element: IrElement) {
        call = element
    }

    /** Forget the current file and call, for messages that belong to neither. */
    fun leaveFiles() {
        file = null
        call = null
    }

    /** The source location of the current call, or null when there is none. */
    val callLocation: CompilerMessageSourceLocation?
        get() = locationOf(file, call?.startOffset ?: -1)

    /**
     * The call cannot be rewritten and is left as written: an error under
     * `strictLowering`, a warning otherwise.
     */
    fun refuse(text: String) {
        if (strictLowering) {
            error(text, callLocation)
        } else {
            warn(TlalocIrErrors.IR_LOWERING_REFUSED_WARNING, text, fallbackLocation = callLocation)
        }
    }

    /**
     * An unexpected failure inside the plugin. Reported at the current call when there
     * is one, otherwise at [location].
     */
    fun internalError(text: String, location: CompilerMessageSourceLocation? = callLocation) {
        if (strictLowering) {
            error(text, location)
        } else {
            warn(TlalocIrErrors.IR_INTERNAL_ERROR_WARNING, text, fallbackLocation = location)
        }
    }

    /** A step failed but compilation carries on with a less processed input. */
    fun degraded(text: String) {
        warn(TlalocIrErrors.IR_DEGRADED, text, fallbackLocation = callLocation)
    }

    /** Developer introspection: printed as compiler INFO output, never a diagnostic. */
    fun info(text: String) {
        configuration.reportInfo(text)
    }

    /** An error, sourceless so that `@Suppress` on an enclosing declaration cannot
     * silence it; [location] still carries the call's file, line and column. */
    private fun error(text: String, location: CompilerMessageSourceLocation?) {
        reporter.report(TlalocIrSourcelessErrors.IR_ERROR_NO_SOURCE, text, location)
    }

    private fun warn(
        factory: KtDiagnosticFactory1<String>,
        text: String,
        fallbackLocation: CompilerMessageSourceLocation?,
    ) {
        val element = call
        val irFile = file
        // A diagnostic factory needs a source element, and the reporter derives it from
        // the element's offsets: a report on an element without them would be dropped.
        if (element != null && irFile != null && element.startOffset >= 0) {
            reporter.at(element, irFile).report(factory, text)
        } else {
            reporter.report(TlalocIrSourcelessErrors.IR_WARNING_NO_SOURCE, text, fallbackLocation)
        }
    }

    internal companion object {
        /** The file, line and column of [startOffset] in [file]; the file alone when the
         * offset is unknown. */
        fun locationOf(file: IrFile?, startOffset: Int): CompilerMessageSourceLocation? {
            val entry = file?.fileEntry ?: return null
            if (startOffset < 0) return CompilerMessageLocation.create(entry.name)
            return CompilerMessageLocation.create(
                entry.name,
                entry.getLineNumber(startOffset) + 1,
                entry.getColumnNumber(startOffset) + 1,
                null,
            )
        }
    }
}
