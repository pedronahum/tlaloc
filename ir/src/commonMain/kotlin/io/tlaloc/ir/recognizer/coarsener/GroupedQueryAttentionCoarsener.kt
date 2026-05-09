package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * §0.4.321 / §0.4.322 — GroupedQueryAttention analytical-backward coarsener.
 *
 * # What this is
 *
 * Consumes a [RecognitionMatch.GroupedQueryAttention] (an attention chain
 * whose K and V operands trace back through an explicit BROADCAST head-
 * expansion step) and produces a [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — `(Q, K_raw, V_raw) → O` recapitulating the
 *   matched forward op-by-op: replay the K/V expansion chain (one
 *   optional inner RESHAPE → BROADCAST → optional outer RESHAPE per
 *   side), then attention.
 * - **`gradient_body`** — analytical VJP. Signature
 *   `(dO, Q, K_raw, V_raw) → (dQ, dK_raw, dV_raw)`. The expanded-side
 *   gradients (`dK_expanded`, `dV_expanded`) come from the same
 *   FlashAttention chain rule; the expansion chain is then inverted
 *   step-by-step — outer RESHAPE inverse → BROADCAST adjoint
 *   (`SUM` with keepdims over the expansion axis) → inner RESHAPE
 *   inverse — to deliver gradients at the raw K / V shape.
 *
 * # Scope
 *
 * Two production shapes are coarsened, sharing the same logic:
 *
 * - **MQA-canonical** (§0.4.321 v1): exactly one BROADCAST per side, no
 *   surrounding RESHAPE. `match.ops.size == 5`.
 * - **GQA-canonical** (§0.4.322 v2): one inner RESHAPE → BROADCAST →
 *   one outer RESHAPE per side, the PyTorch `repeat_kv` form used by
 *   Llama-3 / Mistral / Qwen-2. `match.ops.size == 9`.
 *
 * Anything else — TRANSPOSE in the chain (the `Q · K^T` shape), more
 * than one outer/inner RESHAPE, asymmetric K vs V chains — is declined
 * (`return null`). The recognizer's match is left untouched so
 * downstream lowering decomposes back to primitives.
 *
 * # Other v1+v2 limitations (mirroring [coarsenFlashAttention])
 *
 * - rank ≥ 2 with consistent ranks across Q / K_expanded / V_expanded.
 * - no causal mask, no scale factor.
 * - same broadcast attrs are reused unchanged when emitting primal and
 *   recomputing the forward in the gradient body.
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
    val sType = match.scoreType
    val oType = match.outputType

    // Identify the K and V operands of the matmuls. For MQA-canonical
    // these equal the BROADCAST output shape; for GQA-canonical they're
    // the post-flatten shape (after the outer RESHAPE), which has the
    // same rank as Q.
    val qkOperands = match.qkMatmul.operands
    if (qkOperands.size != 2) return null
    val kForMatmul = qkOperands.firstOrNull { it.id != qInput.id } ?: return null

    val pvOperands = match.pvMatmul.operands
    if (pvOperands.size != 2) return null
    val vForMatmul = pvOperands.firstOrNull { it.id != match.softmax.id } ?: return null

    val kExpandedType = kForMatmul.type
    val vExpandedType = vForMatmul.type

    // Walk each chain explicitly so we know the per-step types.
    val kChain = extractExpansionChain(kForMatmul, match.kBroadcast, kRaw) ?: return null
    val vChain = extractExpansionChain(vForMatmul, match.vBroadcast, vRaw) ?: return null

    // Scope: ≤ 1 outer RESHAPE and ≤ 1 inner RESHAPE per side. No
    // TRANSPOSE in v2 — the `Q · K^T` form needs the transpose inverted
    // alongside, deferred until a real model exercises it.
    if (!isCoarsenableChain(kChain) || !isCoarsenableChain(vChain)) return null

    // Rank consistency (mirrors FlashAttentionCoarsener).
    val rank = qType.rank
    if (rank < 2) return null
    if (kExpandedType.rank != rank || vExpandedType.rank != rank) return null
    if (sType.rank != rank || oType.rank != rank) return null

    val kExpansionAxis = findSingleExpansionAxis(match.kBroadcast) ?: return null
    val vExpansionAxis = findSingleExpansionAxis(match.vBroadcast) ?: return null

    val absorbedOps = buildSet {
        add(match.qkMatmul.id); add(match.softmax.id); add(match.pvMatmul.id)
        add(match.kBroadcast.id); add(match.vBroadcast.id)
        kChain.outerOps.forEach { add(it.id) }
        kChain.innerOps.forEach { add(it.id) }
        vChain.outerOps.forEach { add(it.id) }
        vChain.innerOps.forEach { add(it.id) }
    }

    val primalBody = buildGqaPrimal(
        qType = qType,
        kRawType = kRawType,
        vRawType = vRawType,
        sType = sType,
        oType = oType,
        kBroadcast = match.kBroadcast,
        vBroadcast = match.vBroadcast,
        kChain = kChain,
        vChain = vChain,
    )
    val gradientBody = buildGqaGradient(
        qType = qType,
        kRawType = kRawType,
        vRawType = vRawType,
        kExpandedType = kExpandedType,
        vExpandedType = vExpandedType,
        sType = sType,
        oType = oType,
        kBroadcast = match.kBroadcast,
        vBroadcast = match.vBroadcast,
        kChain = kChain,
        vChain = vChain,
        kExpansionAxis = kExpansionAxis,
        vExpansionAxis = vExpansionAxis,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = absorbedOps,
        anchorOpId = match.pvMatmul.id,
        outerOperands = listOf(qInput, kRaw, vRaw),
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

/**
 * The recognized chain structure between a raw K / V leaf and its
 * matmul operand position. Walked in graph-reverse order:
 *
 * - [outerOps] — ops between the matmul operand and the BROADCAST,
 *   first element closest to the matmul, last closest to the BROADCAST.
 *   For MQA-canonical: empty. For GQA-canonical: one outer RESHAPE.
 * - [innerOps] — ops between the BROADCAST and the raw leaf, first
 *   element closest to the BROADCAST, last closest to the leaf.
 *   For MQA-canonical: empty. For GQA-canonical: one inner RESHAPE.
 */
private data class ExpansionChain(val outerOps: List<DxirOp>, val innerOps: List<DxirOp>)

private fun extractExpansionChain(
    forMatmul: DxirNode,
    broadcast: DxirOp,
    rawInput: DxirNode,
): ExpansionChain? {
    val outerOps = mutableListOf<DxirOp>()
    var cursor: DxirNode = forMatmul
    while (cursor is DxirOp && cursor.id != broadcast.id) {
        if (cursor.op != OpKind.RESHAPE && cursor.op != OpKind.TRANSPOSE) return null
        outerOps.add(cursor)
        cursor = cursor.operands.singleOrNull() ?: return null
    }
    if (cursor !is DxirOp || cursor.id != broadcast.id) return null

    val innerOps = mutableListOf<DxirOp>()
    var inner: DxirNode = broadcast.operands.singleOrNull() ?: return null
    while (inner is DxirOp && inner.id != rawInput.id) {
        if (inner.op != OpKind.RESHAPE) return null
        innerOps.add(inner)
        inner = inner.operands.singleOrNull() ?: return null
    }
    if (inner.id != rawInput.id) return null

    return ExpansionChain(outerOps = outerOps, innerOps = innerOps)
}

private fun isCoarsenableChain(chain: ExpansionChain): Boolean {
    if (chain.outerOps.size > 1 || chain.innerOps.size > 1) return false
    if (chain.outerOps.any { it.op != OpKind.RESHAPE }) return false
    if (chain.innerOps.any { it.op != OpKind.RESHAPE }) return false
    return true
}

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

/**
 * Apply the K/V forward expansion chain inside a [DxirBuilder.function]
 * scope: optional inner RESHAPE, then BROADCAST, then optional outer
 * RESHAPE. Each emitted op gets the SAME output type as the matched
 * original (we read the type off the chain ops directly).
 */
private fun DxirBuilder.applyExpansion(
    raw: DxirNode,
    chain: ExpansionChain,
    broadcast: DxirOp,
): DxirNode {
    var cursor: DxirNode = raw
    // innerOps walked broadcast→leaf, so emitting them in walked-reverse
    // order replays the forward graph order (leaf → broadcast).
    for (innerOp in chain.innerOps.reversed()) {
        cursor = op(OpKind.RESHAPE, listOf(cursor), innerOp.type)
    }
    cursor = op(OpKind.BROADCAST, listOf(cursor), broadcast.type, attrs = broadcast.attrs)
    // outerOps walked matmul→broadcast, emitting walked-reverse replays
    // forward graph order (broadcast → matmul).
    for (outerOp in chain.outerOps.reversed()) {
        cursor = op(OpKind.RESHAPE, listOf(cursor), outerOp.type)
    }
    return cursor
}

/**
 * Invert the expansion chain on the backward side: outer RESHAPE
 * inverses (output type = each outer op's input type), then SUM with
 * keepdims over the expansion axis (the BROADCAST adjoint), then inner
 * RESHAPE inverses. Walking outerOps and innerOps in their RECORDED
 * order (closer-to-matmul / closer-to-broadcast first) gets the inverse
 * sequence right.
 */
private fun DxirBuilder.invertExpansion(
    dExpanded: DxirNode,
    chain: ExpansionChain,
    broadcast: DxirOp,
    expansionAxis: Int,
    rawType: DxirType,
): DxirNode {
    var cursor: DxirNode = dExpanded
    for (outerOp in chain.outerOps) {
        val targetType = outerOp.operands.single().type
        cursor = op(OpKind.RESHAPE, listOf(cursor), targetType)
    }
    val bcInputType = broadcast.operands.single().type
    cursor = op(
        OpKind.SUM, listOf(cursor), bcInputType,
        attrs = mapOf("reduction_dims" to listOf(expansionAxis)),
    )
    for (innerOp in chain.innerOps) {
        val targetType = innerOp.operands.single().type
        cursor = op(OpKind.RESHAPE, listOf(cursor), targetType)
    }
    // Sanity: after walking through all inner inverses we should be at
    // raw shape. If the chain is empty on both sides, SUM-keepdim's
    // output already matches rawType (MQA case: kBroadcast input == kRaw).
    require(cursor.type == rawType) {
        "GroupedQueryAttentionCoarsener: gradient inversion landed at " +
            "${cursor.type} instead of expected $rawType"
    }
    return cursor
}

private fun buildGqaPrimal(
    qType: DxirType,
    kRawType: DxirType,
    vRawType: DxirType,
    sType: DxirType,
    oType: DxirType,
    kBroadcast: DxirOp,
    vBroadcast: DxirOp,
    kChain: ExpansionChain,
    vChain: ExpansionChain,
): DxirFunction = DxirBuilder.function("gqa_primal") {
    val q = param("Q", qType)
    val kRaw = param("K_raw", kRawType)
    val vRaw = param("V_raw", vRawType)
    val kForMatmul = applyExpansion(kRaw, kChain, kBroadcast)
    val vForMatmul = applyExpansion(vRaw, vChain, vBroadcast)
    val s = op(OpKind.MATMUL, listOf(q, kForMatmul), sType)
    val p = op(OpKind.SOFTMAX, listOf(s), sType)
    val o = op(OpKind.MATMUL, listOf(p, vForMatmul), oType)
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
    kBroadcast: DxirOp,
    vBroadcast: DxirOp,
    kChain: ExpansionChain,
    vChain: ExpansionChain,
    kExpansionAxis: Int,
    vExpansionAxis: Int,
): DxirFunction = DxirBuilder.function("gqa_grad") {
    val dO = param("dO", oType)
    val q = param("Q", qType)
    val kRaw = param("K_raw", kRawType)
    val vRaw = param("V_raw", vRawType)

    // Recompute forward expansions + attention.
    val kForMatmul = applyExpansion(kRaw, kChain, kBroadcast)
    val vForMatmul = applyExpansion(vRaw, vChain, vBroadcast)
    val sRecomputed = op(OpKind.MATMUL, listOf(q, kForMatmul), sType)
    val pRecomputed = op(OpKind.SOFTMAX, listOf(sRecomputed), sType)

    // FlashAttention chain rule (mirror of coarsenFlashAttention's grad).
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

    val pT = transposeLastTwo(pRecomputed, sType)
    val dVForMatmul = op(OpKind.MATMUL, listOf(pT, dO), vExpandedType)

    val vT = transposeLastTwo(vForMatmul, vExpandedType)
    val dP = op(OpKind.MATMUL, listOf(dO, vT), sType)

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

    val kT = transposeLastTwo(kForMatmul, kExpandedType)
    val dQ = op(OpKind.MATMUL, listOf(dS, kT), qType)

    val qT = transposeLastTwo(q, qType)
    val dKForMatmul = op(OpKind.MATMUL, listOf(qT, dS), kExpandedType)

    // Invert the K/V expansion chain to land at K_raw / V_raw shape.
    val dKRaw = invertExpansion(dKForMatmul, kChain, kBroadcast, kExpansionAxis, kRawType)
    val dVRaw = invertExpansion(dVForMatmul, vChain, vBroadcast, vExpansionAxis, vRawType)

    listOf(dQ, dKRaw, dVRaw)
}
