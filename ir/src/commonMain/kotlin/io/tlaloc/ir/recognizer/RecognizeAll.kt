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
    all += recognizeRmsNorm(fn, diagnostics)
    all += recognizeRope(fn, diagnostics)
    all += recognizeCrossEntropy(fn, diagnostics)
    // Compound patterns (larger than their primitive supersets) come
    // before the primitives they contain, so the resolver's stable-sort
    // tie-break picks them when sizes happen to match. By size alone the
    // §0.4.282 resolver already prefers them — listing first is doc.
    all += recognizeTransformerMLP(fn, diagnostics)
    all += recognizeSwiGLU(fn, diagnostics)
    // §0.4.318 — SQRT+DIV-form LayerNorm. v1 doesn't overlap with
    // RmsNorm (different anchor op); a future RSQRT+MUL form would.
    all += recognizeLayerNorm(fn, diagnostics)
    // §0.4.320 — GroupedQueryAttention is a strict superset of
    // FlashAttention (same MATMUL/SOFTMAX/MATMUL anchor, plus the K and V
    // BROADCAST expansion chains). When both fire on the same softmax,
    // [resolveLargestMatch] picks GQA by op count.
    all += recognizeGroupedQueryAttention(fn, diagnostics)
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
 * Layer 4 §0.4.314 — the first v2 compound (TransformerMLP, a strict
 * superset of SwiGLU) lights up this resolver on production code: every
 * Llama-style decoder MLP matches both TransformerMLP (5 ops) and the
 * bare SwiGLU (4 ops) on the same SILU anchor, and the resolver picks
 * the compound. Pre-§0.4.314, this function ran but no production
 * patterns ever overlapped — the resolver was tested but dormant.
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
