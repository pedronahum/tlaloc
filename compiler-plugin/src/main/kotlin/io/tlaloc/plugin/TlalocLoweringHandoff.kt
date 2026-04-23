package io.tlaloc.plugin

import io.tlaloc.ir.DxirFunction

/**
 * In-process side table used to hand off the [DxirFunction] produced by the FIR-phase
 * [FirLambdaToDxirLowering] to the IR-phase [TlalocIrGenerationExtension]. Keyed by the
 * source text range of the recognised `io.tlaloc.autograd` intrinsic call — FIR writes
 * the lowered dxir when the lambda succeeds; the IR extension reads + clears.
 *
 * v0 intentionally uses just `(startOffset, endOffset)` as the key. This is sufficient
 * for single-file compile tests; for multi-file compilations we'd key by file path too.
 * The IR extension clears the table after its pass so stale entries don't leak across
 * `K2JVMCompiler.exec(...)` invocations that share this JVM's classloader (matters for
 * the in-process test harness).
 */
object TlalocLoweringHandoff {
    private val byRange: MutableMap<LongRange, DxirFunction> = HashMap()

    @Synchronized
    fun record(startOffset: Int, endOffset: Int, fn: DxirFunction) {
        byRange[startOffset.toLong()..endOffset.toLong()] = fn
    }

    @Synchronized
    fun take(startOffset: Int, endOffset: Int): DxirFunction? =
        byRange.remove(startOffset.toLong()..endOffset.toLong())

    @Synchronized
    fun clear() { byRange.clear() }
}
