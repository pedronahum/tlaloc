package io.tlaloc.plugin

import org.jetbrains.kotlin.cli.common.messages.CompilerMessageLocation
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.reportInfo
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactory1
import org.jetbrains.kotlin.diagnostics.KtSourcelessDiagnosticFactory
import org.jetbrains.kotlin.ir.IrDiagnosticReporter
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.IrFile

/**
 * Every message the IR phase emits goes through here.
 *
 * Errors and warnings are compiler diagnostics reported through the IR plugin
 * context's [IrDiagnosticReporter] (the factories are in [TlalocIrErrors]). They
 * point at the intrinsic call being rewritten, so they carry its file, line and
 * column, fail the build before code generation when they are errors, fail a
 * `-Werror` build when they are warnings, and can be silenced with `@Suppress`
 * naming the factory. A message with no call to point at (a whole-file failure, a
 * call the IR phase never found) is reported without a source element, with the
 * best location there is.
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
        report(
            if (strictLowering) TlalocIrErrors.IR_LOWERING_REFUSED else TlalocIrErrors.IR_LOWERING_REFUSED_WARNING,
            text,
            fallbackLocation = callLocation,
        )
    }

    /**
     * An unexpected failure inside the plugin. Reported at the current call when there
     * is one, otherwise at [location].
     */
    fun internalError(text: String, location: CompilerMessageSourceLocation? = callLocation) {
        report(
            if (strictLowering) TlalocIrErrors.IR_INTERNAL_ERROR else TlalocIrErrors.IR_INTERNAL_ERROR_WARNING,
            text,
            fallbackLocation = location,
        )
    }

    /** A step failed but compilation carries on with a less processed input. */
    fun degraded(text: String) {
        report(TlalocIrErrors.IR_DEGRADED, text, fallbackLocation = callLocation)
    }

    /** Developer introspection: printed as compiler INFO output, never a diagnostic. */
    fun info(text: String) {
        configuration.reportInfo(text)
    }

    private fun report(
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
            reporter.report(sourcelessTwin(factory), text, fallbackLocation)
        }
    }

    private fun sourcelessTwin(factory: KtDiagnosticFactory1<String>): KtSourcelessDiagnosticFactory =
        if (factory == TlalocIrErrors.IR_LOWERING_REFUSED || factory == TlalocIrErrors.IR_INTERNAL_ERROR) {
            TlalocIrSourcelessErrors.IR_ERROR_NO_SOURCE
        } else {
            TlalocIrSourcelessErrors.IR_WARNING_NO_SOURCE
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
