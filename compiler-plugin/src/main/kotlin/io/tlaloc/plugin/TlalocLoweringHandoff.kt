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
 *
 * §0.4.501 — the entry is a [LoweredLambda] rather than a bare [DxirFunction]: a
 * captured runtime value is a trailing param of the lowered function whose BINDING
 * lives at the call site, so the list of captures has to cross this boundary with it.
 */
object TlalocLoweringHandoff {
    /**
     * @property fn the lowered lambda body.
     * @property captures §0.4.501 — the captured runtime values, index-aligned with
     *   `fn.params.takeLast(captures.size)`. Empty for a lambda that captures nothing
     *   (every lambda in the repo before §0.4.501), in which case this record is
     *   behaviourally identical to the pre-§0.4.501 bare function.
     */
    data class LoweredLambda(
        val fn: DxirFunction,
        val captures: List<FirLambdaToDxirLowering.CapturedRuntimeValue> = emptyList(),
    )

    private val byRange: MutableMap<LongRange, LoweredLambda> = HashMap()

    @Synchronized
    fun record(
        startOffset: Int,
        endOffset: Int,
        fn: DxirFunction,
        captures: List<FirLambdaToDxirLowering.CapturedRuntimeValue> = emptyList(),
    ) {
        byRange[startOffset.toLong()..endOffset.toLong()] = LoweredLambda(fn, captures)
    }

    @Synchronized
    fun take(startOffset: Int, endOffset: Int): LoweredLambda? =
        byRange.remove(startOffset.toLong()..endOffset.toLong())

    @Synchronized
    fun clear() { byRange.clear() }
}
