package io.tlaloc.ir.passes

import io.tlaloc.ir.OpKind

/**
 * §0.4.448 — audit finding C (docs/AD_SINGLE_ENGINE_AUDIT.md): the demoted
 * OpKinds and their loud, named refusal messages.
 *
 * These kinds are DELIBERATELY partial. Recognition (cost-model pricing,
 * recognizer acceptance) and StableHLO emission are their sanctioned layers;
 * the interpreter and both AD transforms refuse them BY NAME so a hand-built
 * graph using one fails with the sanctioned alternative in the message instead
 * of a generic "unsupported op" — or worse, a silent skip (pre-§0.4.448, a
 * multi-result op whose consumers sat at index > 0 slipped past the reverse
 * walk's index-0 upstream lookup and silently dropped its gradient).
 *
 * REJECTED alternative (the audit's demote decision): completing the kinds —
 * interpreter arms, VjpRules, forward tangents — would duplicate work the
 * coarseners already own with certification (layernorm/attention fused
 * semantics).
 *
 * Departures from the §0.4.448 table:
 * - SPLIT was DELETED outright in §0.4.454 (Phase G slice 1): unreachable —
 *   per-piece SLICE is the sanctioned spelling.
 * - ALL_REDUCE and SHARD_CONSTRAINT were UN-DEMOTED in §0.4.460 (Phase G3a),
 *   deliberately: ALL_REDUCE-sum is a linear, self-adjoint op (interpreter
 *   arm with single-process semantics, AllReduceRule, forward tangent,
 *   StableHLO region emission — see [io.tlaloc.ir.AllReduceAttrs]), and
 *   reading showed SHARD_CONSTRAINT is a value-identity with layout metadata
 *   (identity adjoint/tangent are honest and cheap). GradShardingVerify keeps
 *   its sharded-pipeline collective-duality role alongside.
 */
internal val DEMOTED_OP_KINDS: Set<OpKind> = setOf(
    OpKind.LAYERNORM,
    OpKind.SCALED_DOT_PRODUCT_ATTENTION,
)

/**
 * The refusal message for a demoted kind, prefixed with the refusing [layer]
 * (e.g. `"DxirInterpreter"`), or null when [kind] is not demoted. Every
 * message names the kind AND the sanctioned alternative.
 */
internal fun demotedKindRefusal(kind: OpKind, layer: String): String? = when (kind) {
    OpKind.LAYERNORM ->
        "$layer: LAYERNORM is a coarsener-recognized/emission-only kind (cost model + " +
            "StableHLO emitter are its sanctioned layers) — spell layernorm via ops " +
            "(SUB/DIV/SQRT over MEAN, the §0.4.390 batchNorm desugaring precedent) " +
            "and let the coarsener own the fused semantics"
    OpKind.SCALED_DOT_PRODUCT_ATTENTION ->
        "$layer: SCALED_DOT_PRODUCT_ATTENTION is a coarsener-recognized/emission-only " +
            "kind (FlashAttentionRecognizer + cost model + StableHLO emitter are its " +
            "sanctioned layers) — spell attention via the FlashAttention composition " +
            "(MATMUL/softmax/MATMUL) and let the coarsener own the fused semantics"
    else -> null
}

/**
 * §0.4.465 — Phase H1a: the INFERENCE-ONLY kinds. A SIBLING of
 * [DEMOTED_OP_KINDS], not a member of it, and the difference is the whole
 * point:
 *
 * - a DEMOTED kind is partial in the *execution* layers too — no interpreter
 *   arm, because a coarsener already owns its semantics with certification;
 * - an INFERENCE-ONLY kind is COMPLETE where it runs (interpreter arm +
 *   StableHLO emission are mandatory — serving has to actually execute) and
 *   deliberately absent only in the two AD transforms.
 *
 * The rationale is not "we didn't get to it": there is no training graph in
 * which these are differentiable intermediates. Paged attention reads a KV
 * page pool that previous decode steps mutated, indexed by an integer block
 * table the allocator produced — state and bookkeeping, not a differentiable
 * value. Writing a VJP for it would invent math nobody can check against a
 * reference, which the house forbids (see the no-adjoint-shortcuts rule).
 *
 * So both transforms refuse BY NAME, with the TRAINING spelling in the
 * message — never a silent gap, never a zero. The arc rule that produced this
 * file's predecessor applies unchanged: any new op either renders/handles or
 * refuses by name.
 */
internal val INFERENCE_ONLY_OP_KINDS: Set<OpKind> = setOf(
    OpKind.PAGED_ATTENTION,
    // §0.4.466 — Phase H1b: the other half of the paged decode step. Same
    // rationale, one step earlier in the loop: the write DEPOSITS into the
    // pool that paged attention then reads.
    OpKind.KV_CACHE_WRITE,
)

/**
 * The refusal message for an inference-only kind, prefixed with the refusing
 * [layer], or null when [kind] is not inference-only. Every message names the
 * kind, the inference-only RATIONALE, and the differentiable alternative.
 */
internal fun inferenceOnlyKindRefusal(kind: OpKind, layer: String): String? = when (kind) {
    OpKind.PAGED_ATTENTION ->
        "$layer: PAGED_ATTENTION is INFERENCE-ONLY BY DESIGN (Phase H1a, " +
            "docs/INFERENCE_SERVING_AUDIT.md) and carries no adjoint and no tangent — " +
            "not a gap: its key/value operands are a block-table-indexed KV PAGE POOL " +
            "mutated across decode steps and addressed by integer allocator bookkeeping, " +
            "so it is not a differentiable intermediate in any training graph. It DOES " +
            "have an interpreter arm and StableHLO emission (serving executes it). To " +
            "DIFFERENTIATE attention, use the training spelling: the FlashAttention " +
            "composition (MATMUL/softmax/MATMUL) or the GQA recognizer's coarsened form, " +
            "which the coarseners own with certified gradients"
    OpKind.KV_CACHE_WRITE ->
        "$layer: KV_CACHE_WRITE is INFERENCE-ONLY BY DESIGN (Phase H1b, " +
            "docs/INFERENCE_SERVING_AUDIT.md) and carries no adjoint and no tangent — " +
            "not a gap: it deposits a decode step's new keys/values into a KV PAGE POOL " +
            "at flat slots an allocator named, so its cache operand is serving-runtime " +
            "state threaded across steps and its slotMapping is integer bookkeeping — " +
            "neither is a differentiable intermediate in any training graph. It DOES " +
            "have an interpreter arm and StableHLO emission (serving executes it). To " +
            "DIFFERENTIATE a placement of values into a tensor, use the differentiable " +
            "spelling: the SCATTER / SCATTER_ADD family, which carries certified rules"
    else -> null
}
