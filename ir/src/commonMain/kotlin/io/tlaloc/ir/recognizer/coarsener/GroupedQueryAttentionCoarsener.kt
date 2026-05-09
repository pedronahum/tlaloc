package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * §0.4.321 — GroupedQueryAttention analytical-backward coarsener.
 *
 * # What this is
 *
 * Consumes a [RecognitionMatch.GroupedQueryAttention] (the
 * `BROADCAST(K_raw) → MATMUL → SOFTMAX → MATMUL ← BROADCAST(V_raw)`
 * shape) and produces a [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — `(Q, K_raw, V_raw) → O` recapitulating the
 *   matched forward op-by-op: expand K and V via the same BROADCAST
 *   attrs the user wrote, then attention.
 * - **`gradient_body`** — analytical VJP. Signature
 *   `(dO, Q, K_raw, V_raw) → (dQ, dK_raw, dV_raw)`. The expanded-side
 *   gradients (`dK_expanded`, `dV_expanded`) come from the same
 *   FlashAttention chain rule; the BROADCAST adjoint (sum-reduce over
 *   the expansion axis with keepdims) collapses them back to the raw
 *   K/V shape.
 *
 * # Scope (v1)
 *
 * Only the **MQA-canonical form** is coarsened: a single BROADCAST per
 * side, with the BROADCAST's input == the raw leaf (no inner RESHAPE)
 * and no outer RESHAPE/TRANSPOSE wrappers. Concretely, this fires when
 * `match.ops.size == 5` (qk + softmax + pv + kBroadcast + vBroadcast).
 *
 * The GQA-canonical `RESHAPE → BROADCAST → RESHAPE` form (the PyTorch
 * `repeat_kv` shape) is **declined** — the coarsener returns `null`,
 * leaving the recognizer's match untouched. Adding GQA-canonical needs
 * RESHAPE-pair inversion in the gradient body; deferred until a real
 * model in the bench harness exercises it.
 *
 * Other v1 limitations (mirroring [coarsenFlashAttention]):
 * - rank ≥ 2 with consistent ranks across Q / K_expanded / V_expanded.
 * - no causal mask, no scale factor (the user-written attention chain
 *   absorbed neither — same v1 boundary as FlashAttention).
 * - same broadcast attrs are reused unchanged when emitting the primal
 *   and recomputing the forward in the gradient body.
 */
internal fun coarsenGroupedQueryAttention(
    match: RecognitionMatch.GroupedQueryAttention,
): CoarsenedBundle? {
    val qInput = match.qInput
    val kRaw = match.kRawInput
    val vRaw = match.vRawInput
    val qType = qInput.type
    val kRawType = kRaw.type
    val vRawType = vRaw.type
    val kExpandedType = match.kBroadcast.type
    val vExpandedType = match.vBroadcast.type
    val sType = match.scoreType
    val oType = match.outputType

    // v1: MQA-canonical shape only. Recognizer's ops list contains all
    // absorbed ops; for MQA the count is exactly 5 (the attention
    // triple plus the two BROADCASTs). GQA-canonical adds 4 RESHAPEs
    // for size-1 insert + flatten on each side, hitting 9.
    if (match.ops.size != 5) return null

    // Sanity: each BROADCAST's input must be the raw leaf (no inner
    // RESHAPE between leaf and broadcast).
    val kBcastInput = match.kBroadcast.operands.singleOrNull() ?: return null
    val vBcastInput = match.vBroadcast.operands.singleOrNull() ?: return null
    if (kBcastInput.id != kRaw.id || vBcastInput.id != vRaw.id) return null

    // Rank consistency (mirrors FlashAttentionCoarsener).
    val rank = qType.rank
    if (rank < 2) return null
    if (kExpandedType.rank != rank || vExpandedType.rank != rank) return null
    if (sType.rank != rank || oType.rank != rank) return null

    // Find the size-1 → size-N expansion axis on each BROADCAST. v1
    // requires same-rank broadcasts with exactly one expansion axis.
    val kExpansionAxis = findSingleExpansionAxis(match.kBroadcast) ?: return null
    val vExpansionAxis = findSingleExpansionAxis(match.vBroadcast) ?: return null

    val primalBody = buildGqaPrimal(
        qType = qType,
        kRawType = kRawType,
        vRawType = vRawType,
        kExpandedType = kExpandedType,
        vExpandedType = vExpandedType,
        sType = sType,
        oType = oType,
        kBcastAttrs = match.kBroadcast.attrs,
        vBcastAttrs = match.vBroadcast.attrs,
    )
    val gradientBody = buildGqaGradient(
        qType = qType,
        kRawType = kRawType,
        vRawType = vRawType,
        kExpandedType = kExpandedType,
        vExpandedType = vExpandedType,
        sType = sType,
        oType = oType,
        kBcastAttrs = match.kBroadcast.attrs,
        vBcastAttrs = match.vBroadcast.attrs,
        kExpansionAxis = kExpansionAxis,
        vExpansionAxis = vExpansionAxis,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = setOf(
            match.qkMatmul.id, match.softmax.id, match.pvMatmul.id,
            match.kBroadcast.id, match.vBroadcast.id,
        ),
        anchorOpId = match.pvMatmul.id,
        outerOperands = listOf(qInput, kRaw, vRaw),
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

/**
 * Find the single output axis where the BROADCAST expands a size-1
 * input dim. Returns `null` if v1 assumptions don't hold:
 * `broadcast_dimensions` length != input rank (rank-changing broadcast),
 * a non-1 input dim doesn't match its output dim, or multiple expansion
 * axes (different shape pattern, not single-head expansion).
 */
@Suppress("UNCHECKED_CAST")
private fun findSingleExpansionAxis(broadcast: DxirOp): Int? {
    val input = broadcast.operands.singleOrNull() ?: return null
    val inputDims = input.type.dims
    val outputDims = broadcast.type.dims
    val broadcastDims = (broadcast.attrs["broadcast_dimensions"] as? List<Int>) ?: return null
    if (broadcastDims.size != inputDims.size) return null

    var expansionAxis: Int? = null
    for (i in inputDims.indices) {
        val outAxis = broadcastDims[i]
        if (outAxis < 0 || outAxis >= outputDims.size) return null
        if (inputDims[i] != outputDims[outAxis]) {
            if (inputDims[i] != 1) return null
            if (expansionAxis != null) return null
            expansionAxis = outAxis
        }
    }
    return expansionAxis
}

private fun buildGqaPrimal(
    qType: DxirType,
    kRawType: DxirType,
    vRawType: DxirType,
    kExpandedType: DxirType,
    vExpandedType: DxirType,
    sType: DxirType,
    oType: DxirType,
    kBcastAttrs: Map<String, Any>,
    vBcastAttrs: Map<String, Any>,
): DxirFunction = DxirBuilder.function("gqa_primal") {
    val q = param("Q", qType)
    val kRaw = param("K_raw", kRawType)
    val vRaw = param("V_raw", vRawType)
    val kExpanded = op(OpKind.BROADCAST, listOf(kRaw), kExpandedType, attrs = kBcastAttrs)
    val vExpanded = op(OpKind.BROADCAST, listOf(vRaw), vExpandedType, attrs = vBcastAttrs)
    val s = op(OpKind.MATMUL, listOf(q, kExpanded), sType)
    val p = op(OpKind.SOFTMAX, listOf(s), sType)
    val o = op(OpKind.MATMUL, listOf(p, vExpanded), oType)
    listOf(o)
}

private fun buildGqaGradient(
    qType: DxirType,
    kRawType: DxirType,
    vRawType: DxirType,
    kExpandedType: DxirType,
    vExpandedType: DxirType,
    sType: DxirType,
    oType: DxirType,
    kBcastAttrs: Map<String, Any>,
    vBcastAttrs: Map<String, Any>,
    kExpansionAxis: Int,
    vExpansionAxis: Int,
): DxirFunction = DxirBuilder.function("gqa_grad") {
    val dO = param("dO", oType)
    val q = param("Q", qType)
    val kRaw = param("K_raw", kRawType)
    val vRaw = param("V_raw", vRawType)

    // Recompute forward: expand K, V, then attention.
    val kExpanded = op(OpKind.BROADCAST, listOf(kRaw), kExpandedType, attrs = kBcastAttrs)
    val vExpanded = op(OpKind.BROADCAST, listOf(vRaw), vExpandedType, attrs = vBcastAttrs)
    val sRecomputed = op(OpKind.MATMUL, listOf(q, kExpanded), sType)
    val pRecomputed = op(OpKind.SOFTMAX, listOf(sRecomputed), sType)

    // FlashAttention chain rule for gradients on the expanded operands.
    val rank = sType.rank
    val lastAxis = rank - 1
    val swapLastTwoPerm: List<Int> = (0 until rank - 2).toList() + listOf(rank - 1, rank - 2)

    fun transposeLastTwo(operand: DxirNode, opType: DxirType): DxirNode {
        val swapped = opType.dims.toMutableList().also {
            val a = it[rank - 2]; it[rank - 2] = it[rank - 1]; it[rank - 1] = a
        }
        val outType = DxirType(opType.dtype, swapped)
        return op(OpKind.TRANSPOSE, listOf(operand), outType, attrs = mapOf("permutation" to swapLastTwoPerm))
    }

    // dV_expanded = P^T · dO
    val pT = transposeLastTwo(pRecomputed, sType)
    val dVExpanded = op(OpKind.MATMUL, listOf(pT, dO), vExpandedType)

    // dP = dO · V_expanded^T
    val vT = transposeLastTwo(vExpanded, vExpandedType)
    val dP = op(OpKind.MATMUL, listOf(dO, vT), sType)

    // dS = (dP - sum(P · dP, last_axis, keepdims=False) · BCAST_back) · P
    val pdp = op(OpKind.MUL, listOf(pRecomputed, dP), sType)
    val reducedDims = sType.dims.toMutableList().also { it.removeAt(lastAxis) }
    val reducedType = DxirType(sType.dtype, reducedDims)
    val sumPdp = op(
        OpKind.SUM, listOf(pdp), reducedType,
        attrs = mapOf("reduction_dims" to listOf(lastAxis)),
    )
    val keptAxes = (0 until rank).filter { it != lastAxis }
    val sumPdpBroadcast = op(
        OpKind.BROADCAST, listOf(sumPdp), sType,
        attrs = mapOf("broadcast_dimensions" to keptAxes),
    )
    val dPMinusSum = op(OpKind.SUB, listOf(dP, sumPdpBroadcast), sType)
    val dS = op(OpKind.MUL, listOf(dPMinusSum, pRecomputed), sType)

    // dQ = dS · K_expanded^T
    val kT = transposeLastTwo(kExpanded, kExpandedType)
    val dQ = op(OpKind.MATMUL, listOf(dS, kT), qType)

    // dK_expanded = Q^T · dS
    val qT = transposeLastTwo(q, qType)
    val dKExpanded = op(OpKind.MATMUL, listOf(qT, dS), kExpandedType)

    // BROADCAST adjoint: sum-reduce with keepdims=true over the
    // expansion axis. The output type matches K_raw / V_raw exactly
    // because v1 only handles the MQA shape where the expansion axis
    // already had size 1 in the raw input.
    val dKRaw = op(
        OpKind.SUM, listOf(dKExpanded), kRawType,
        attrs = mapOf("reduction_dims" to listOf(kExpansionAxis)),
    )
    val dVRaw = op(
        OpKind.SUM, listOf(dVExpanded), vRawType,
        attrs = mapOf("reduction_dims" to listOf(vExpansionAxis)),
    )
    listOf(dQ, dKRaw, dVRaw)
}
