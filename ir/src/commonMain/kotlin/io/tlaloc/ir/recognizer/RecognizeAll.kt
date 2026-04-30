package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction

/**
 * Layer 3 §0.4.250+ — aggregate recognizer entry point.
 *
 * Runs every per-pattern recognizer over [fn] and returns the union of
 * matches, with overlapping smaller matches stripped (the resolver
 * prefers the largest containing pattern).
 *
 * # Adding a new recognizer
 *
 * 1. Define a new sealed subtype of [RecognitionMatch] in
 *    `RecognitionMatch.kt`.
 * 2. Write a recognizer function `recognizeXxx(fn, diagnostics): List<RecognitionMatch.Xxx>`
 *    in its own file (mirror `FlashAttentionRecognizer.kt`'s shape).
 * 3. Add one line to [recognizeAll] below.
 *
 * No build-system changes; no central registry to thread; no service-
 * loader. The aggregator is the registry.
 *
 * @param fn the DXIR function to scan.
 * @param diagnostics optional collector for near-miss diagnostics. When
 *   provided, recognizers append a [RecognitionDiagnostic] for each
 *   pre-filter trigger that didn't fully match. Useful for IDE tooling
 *   that wants to surface "you almost wrote attention here" hints.
 */
fun recognizeAll(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch> {
    val all = mutableListOf<RecognitionMatch>()
    all += recognizeFlashAttention(fn, diagnostics)
    // L3.1 plumbs RMS norm / RoPE / cross-entropy here.
    return resolveLargestMatch(all)
}

/**
 * Strip overlapping smaller matches when a larger match contains them.
 *
 * v1 implementation: for each matched op id, the recognizer with the
 * largest [RecognitionMatch.ops] list wins. Ties broken by stable order
 * (FlashAttention before RmsNorm before Rope before CrossEntropy —
 * matches `recognizeAll`'s call order).
 *
 * Layer 3 v1 has no compound patterns that contain other recognized
 * patterns (FlashAttention's MATMUL/SOFTMAX/MATMUL ops aren't separately
 * recognized by RmsNorm/RoPE/CrossEntropy), so this resolver is mostly
 * a no-op for v1. It's plumbed because v2 patterns (e.g.
 * "transformer-block" containing both FlashAttention and RmsNorm) will
 * need it.
 */
internal fun resolveLargestMatch(matches: List<RecognitionMatch>): List<RecognitionMatch> {
    if (matches.size < 2) return matches
    // Sort by descending op-count so larger matches win at insertion.
    val sorted = matches.sortedByDescending { it.ops.size }
    val claimedOpIds = HashSet<Int>()
    val out = mutableListOf<RecognitionMatch>()
    for (match in sorted) {
        val matchOpIds = match.ops.map { it.id }.toSet()
        if (matchOpIds.any { it in claimedOpIds }) continue
        claimedOpIds += matchOpIds
        out += match
    }
    return out
}
