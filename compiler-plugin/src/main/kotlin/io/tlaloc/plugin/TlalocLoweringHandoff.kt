package io.tlaloc.plugin

import io.tlaloc.ir.DxirFunction

/**
 * Side table used to hand off the [DxirFunction] produced by the FIR-phase
 * [FirLambdaToDxirLowering] to the IR-phase [TlalocIrGenerationExtension]. FIR writes
 * the lowered dxir when the lambda succeeds; the IR extension reads and removes it.
 *
 * §0.4.514 — two properties the v0 table did not have, both load-bearing:
 *
 *  - **Keyed by file AND source range.** The v0 key was `(startOffset, endOffset)`
 *    alone, and offsets are per FILE. Two `grad {}` calls at the same offsets in two
 *    files of one module (routine: two files with the same imports and one function
 *    each) shared a key, the second FIR write overwrote the first, and the IR phase
 *    then compiled file B's gradient into file A's call site — a wrong gradient with
 *    no diagnostic. The key is now `(file path, start, end)`, with the path taken
 *    from the FIR checker's `containingFilePath` and the IR file's
 *    `fileEntry.name` (the same string in both phases; see [normalisePath]).
 *  - **One instance per compilation.** v0 was a process-global `object`, cleared
 *    by the IR extension at the end of its pass. In a Kotlin daemon compiling two
 *    modules at once, one compilation's clear() discarded the other's pending
 *    entries (leaving its calls unrewritten), and entries could collide across
 *    compilations. [TlalocCompilerPluginRegistrar] now creates one instance per
 *    `registerExtensions` call — one per compilation — and hands the same instance
 *    to the FIR checker and the IR extension.
 *
 * §0.4.501 — the entry is a [LoweredLambda] rather than a bare [DxirFunction]: a
 * captured runtime value is a trailing param of the lowered function whose BINDING
 * lives at the call site, so the list of captures has to cross this boundary with it.
 */
class TlalocLoweringHandoff {
    /**
     * @property fn the lowered lambda body.
     * @property captures §0.4.501 — the captured runtime values, index-aligned with
     *   `fn.params.takeLast(captures.size)`. Empty for a lambda that captures nothing.
     */
    data class LoweredLambda(
        val fn: DxirFunction,
        val captures: List<FirLambdaToDxirLowering.CapturedRuntimeValue> = emptyList(),
    )

    private data class Key(val file: String, val startOffset: Int, val endOffset: Int)

    private val entries: MutableMap<Key, LoweredLambda> = HashMap()

    @Synchronized
    fun record(
        filePath: String,
        startOffset: Int,
        endOffset: Int,
        fn: DxirFunction,
        captures: List<FirLambdaToDxirLowering.CapturedRuntimeValue> = emptyList(),
    ) {
        entries[Key(normalisePath(filePath), startOffset, endOffset)] = LoweredLambda(fn, captures)
    }

    @Synchronized
    fun take(filePath: String, startOffset: Int, endOffset: Int): LoweredLambda? =
        entries.remove(Key(normalisePath(filePath), startOffset, endOffset))

    /** Number of entries not yet taken. */
    @Synchronized
    fun size(): Int = entries.size

    /** Drops every entry of THIS compilation's table. */
    @Synchronized
    fun clear() { entries.clear() }

    companion object {
        /** The FIR and IR phases both report the path the compiler was given; this
         * only folds Windows separators so the two spellings cannot drift apart. */
        fun normalisePath(path: String): String = path.replace('\\', '/')
    }
}
