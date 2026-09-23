package io.tlaloc.plugin

/**
 * §0.4.514 — the plugin's handling of exceptions it did not expect.
 *
 * Both halves of the plugin (the FIR checker and the IR extension) wrap their work on
 * a recognised intrinsic call in a guard that turns an unexpected exception into a
 * diagnostic at that call site, carrying [describe]'s text. Without the guard the
 * exception escapes into the compiler, which reports an internal compiler error with
 * a stack trace of compiler classes and no sign of which plugin, or which line of the
 * user's code, was involved.
 */
object TlalocInternalErrors {
    const val ISSUES_URL: String = "https://github.com/pedronahum/tlaloc/issues"

    /**
     * Test-only fault injection: when the JVM system property of this name equals a
     * [Phase]'s [Phase.propertyValue], the guarded code in that phase throws an
     * [IllegalStateException] before doing any work. It is how the guards are tested
     * without depending on a real plugin bug.
     */
    const val INJECT_FAULT_PROPERTY: String = "tlaloc.internal.injectFault"

    enum class Phase(val propertyValue: String) {
        FIR("fir"),
        IR("ir"),

        /** The IR phase skips the handoff lookup, as if the two phases disagreed on
         * a call's file or offsets, so the lowered entry is never claimed. */
        IR_HANDOFF_MISS("ir-handoff-miss"),
    }

    fun faultInjected(phase: Phase): Boolean = System.getProperty(INJECT_FAULT_PROPERTY) == phase.propertyValue

    fun maybeInjectFault(phase: Phase) {
        if (faultInjected(phase)) {
            throw IllegalStateException("injected fault ($INJECT_FAULT_PROPERTY=${phase.propertyValue})")
        }
    }

    /**
     * True for the throwables a guard must NOT swallow: JVM errors (out of memory,
     * stack overflow, class-linkage failures — the last is also how an unsupported
     * compiler version shows up), thread interruption, and the compiler's own
     * cancellation / control-flow exceptions, which it uses to abort work and which
     * must keep propagating. Those are matched by name because the compiler ships them
     * under relocated packages.
     */
    fun mustRethrow(t: Throwable): Boolean {
        if (t is VirtualMachineError || t is LinkageError || t is InterruptedException) return true
        return isControlFlow(t.javaClass)
    }

    private fun isControlFlow(c: Class<*>?): Boolean {
        var k: Class<*>? = c
        while (k != null) {
            val simple = k.simpleName
            if (simple == "ProcessCanceledException" || simple == "ControlFlowException" ||
                simple == "CancellationException"
            ) return true
            if (k.interfaces.any { isControlFlow(it) }) return true
            k = k.superclass
        }
        return false
    }

    /**
     * The diagnostic text for a call whose lambda the FIR phase lowered but whose call
     * the IR phase never reached: the call is left as written, so its fallback body
     * throws at the first call.
     */
    fun describeUnclaimed(callableName: String, strictLowering: Boolean = true): String {
        val consequence = if (strictLowering) {
            ""
        } else {
            " The call is left as written, so its fallback body throws at the first call."
        }
        return "Tlaloc internal error: the lambda of this `$callableName` call was lowered, but the " +
            "IR phase found no call at this position to rewrite. This is a bug in the Tlaloc " +
            "compiler plugin, not in your code. Please report it at $ISSUES_URL with the code " +
            "of this call.$consequence"
    }

    /** The diagnostic text for an unexpected [t]. */
    fun describe(t: Throwable, strictLowering: Boolean = true): String {
        val frame = t.stackTrace.firstOrNull()?.let { " at ${it.className}.${it.methodName}(${it.fileName}:${it.lineNumber})" }
            .orEmpty()
        val consequence = if (strictLowering) {
            ""
        } else {
            " The call is left as written, so its fallback body throws at the first call."
        }
        return "Tlaloc internal error while compiling this call: ${t.javaClass.name}: ${t.message}$frame. " +
            "This is a bug in the Tlaloc compiler plugin, not in your code. Please report it at " +
            "$ISSUES_URL with the code of this call.$consequence"
    }
}
