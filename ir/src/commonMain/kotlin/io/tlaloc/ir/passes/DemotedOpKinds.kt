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
