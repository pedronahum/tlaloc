package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * Layer 3 §0.4.252+ — FlashAttention analytical-backward coarsener.
 *
 * # What this is
 *
 * Consumes a [RecognitionMatch.FlashAttention] (the canonical
 * `MATMUL → SOFTMAX → MATMUL` shape) and produces a [CoarsenedBundle]
 * carrying:
 *
 * - **`primal_body`** — a free-standing `DxirFunction` that recapitulates
 *   the matched forward: `S = MATMUL(Q, K); P = SOFTMAX(S); O = MATMUL(P, V)`.
 *   The COARSENED op's three operands map positionally to `(Q, K, V)`.
 * - **`gradient_body`** — the analytical VJP. Signature
 *   `(dO, Q, K, V) → (dQ, dK, dV)`. Recomputes `S` and `P` internally
 *   (cheaper than threading them through the COARSENED's payload, given
 *   our VJP doesn't have softmax-stat re-use in v1) and emits the
 *   standard chain-rule formulas.
 * - **`reads_primal_indices`** = `{0, 1, 2}` — gradient body
 *   dereferences Q, K, V (computed via [computeGradientReads]).
 *
 * # The math
 *
 * Forward (with the matmul operand convention the recognizer captures —
 * `Q` is `qkMatmul.operands[0]`, `K` is `qkMatmul.operands[1]`, so
 * MATMUL semantics is `S = Q · K`; users who want `Q · K^T` are expected
 * to have pre-transposed `K`):
 *
 * ```
 * S = MATMUL(Q, K)              # scores      [..., m, n]
 * P = SOFTMAX(S, last_axis)     # probs       [..., m, n]
 * O = MATMUL(P, V)              # output      [..., m, d_v]
 * ```
 *
 * Backward, given `dO`:
 *
 * ```
 * # Through O = P · V:
 * dV = MATMUL(P^T, dO)          [..., n, d_v]
 * dP = MATMUL(dO, V^T)          [..., m, n]
 *
 * # Through P = softmax(S, last_axis):
 * #   dS = (dP - sum(P · dP, last_axis, keepdims=True)) · P
 * pdp     = MUL(P, dP)          [..., m, n]
 * sum_pdp = SUM(pdp, axis=last) # reduces last axis           [..., m]
 * sum_pdp_b = BROADCAST_back    # bring back to [..., m, n]   (last-axis broadcast)
 * dS = MUL(SUB(dP, sum_pdp_b), P)
 *
 * # Through S = Q · K:
 * dQ = MATMUL(dS, K^T)          [..., m, k]
 * dK = MATMUL(Q^T, dS)          [..., k, n]
 * ```
 *
 * # Scope notes
 *
 * - **Recompute P, S inside the gradient.** v1 doesn't try to thread
 *   softmax statistics out of the COARSENED — it's simpler to recompute.
 *   A future variant could expose `(P, log_sum_exp)` as additional
 *   primal_body returns and reference them inside the gradient_body
 *   (saves one MATMUL + one SOFTMAX per backward call).
 *
 * - **No tile-streaming yet.** This is the "fused, but not tiled"
 *   FlashAttention. L3.4's tile-fusion pass + KV-quant + cost model
 *   build on top of this envelope to emit the actually-fused kernel.
 *   The COARSENED op gives downstream lowering one node to substitute.
 *
 * - **BROADCAST shape note.** The interpreter's `BROADCAST` only
 *   supports scalar→rank-N (DxirInterpreter.kt:436). The gradient body
 *   here uses rank-(r-1)→rank-r, which the interpreter doesn't yet
 *   handle. Coarsened FlashAttention won't roundtrip through
 *   DxirInterpreter; downstream lowering (StableHLO emit / IREE) is
 *   what consumes the gradient body. Tests verify *structural*
 *   correctness only.
 *
 * - **Operand-convention agnostic.** The recognizer doesn't enforce
 *   `Q · K^T` vs `Q · K`. Whatever the user wrote as `qkMatmul`'s
 *   operands becomes `(Q, K)` in our primal; the gradient is consistent
 *   with that choice (the `K^T` we emit in `dQ` matches whatever the
 *   user fed in).
 */
internal fun coarsenFlashAttention(
    match: RecognitionMatch.FlashAttention,
): CoarsenedBundle? {
    val qType = match.qInput.type
    val kType = match.kInput.type
    val vType = match.vInput.type
    val sType = match.scoreType
    val oType = match.outputType

    // v1 only handles rank ≥ 2 with consistent ranks. Higher-rank batched
    // attention works via the generic `[..., last_two]` convention.
    val rank = qType.rank
    if (rank < 2 || kType.rank != rank || vType.rank != rank) return null
    if (sType.rank != rank || oType.rank != rank) return null

    val primalBody = buildFlashAttentionPrimal(qType, kType, vType, sType, oType)
    val gradientBody = buildFlashAttentionGradient(qType, kType, vType, sType, oType)
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = setOf(match.qkMatmul.id, match.softmax.id, match.pvMatmul.id),
        anchorOpId = match.pvMatmul.id,
        outerOperands = listOf(match.qInput, match.kInput, match.vInput),
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

private fun buildFlashAttentionPrimal(
    qType: DxirType,
    kType: DxirType,
    vType: DxirType,
    sType: DxirType,
    oType: DxirType,
): DxirFunction = DxirBuilder.function("flash_attention_primal") {
    val q = param("Q", qType)
    val k = param("K", kType)
    val v = param("V", vType)
    val s = op(OpKind.MATMUL, listOf(q, k), sType)
    val p = op(OpKind.SOFTMAX, listOf(s), sType)
    val o = op(OpKind.MATMUL, listOf(p, v), oType)
    listOf(o)
}

private fun buildFlashAttentionGradient(
    qType: DxirType,
    kType: DxirType,
    vType: DxirType,
    sType: DxirType,
    oType: DxirType,
): DxirFunction = DxirBuilder.function("flash_attention_grad") {
    val dO = param("dO", oType)
    val q = param("Q", qType)
    val k = param("K", kType)
    val v = param("V", vType)

    val rank = sType.rank
    val lastAxis = rank - 1

    // Permutation that swaps the last two axes: [0..r-3, r-1, r-2].
    val swapLastTwoPerm: List<Int> = (0 until rank - 2).toList() + listOf(rank - 1, rank - 2)

    // For Q · K → S, we need K^T (swap last two of K) and Q^T (swap last
    // two of Q) for the dQ/dK formulas. Same for V.
    fun transposeLastTwo(operand: io.tlaloc.ir.DxirNode, opType: DxirType): io.tlaloc.ir.DxirNode {
        val swapped = opType.dims.toMutableList().also {
            val a = it[rank - 2]; it[rank - 2] = it[rank - 1]; it[rank - 1] = a
        }
        val outType = DxirType(opType.dtype, swapped)
        return op(OpKind.TRANSPOSE, listOf(operand), outType, attrs = mapOf("permutation" to swapLastTwoPerm))
    }

    // Recompute S and P (cheaper to recompute than thread through).
    val sRecomputed = op(OpKind.MATMUL, listOf(q, k), sType)
    val pRecomputed = op(OpKind.SOFTMAX, listOf(sRecomputed), sType)

    // dV = P^T · dO  →  shape [..., n, d_v]
    val pT = transposeLastTwo(pRecomputed, sType)
    val dV = op(OpKind.MATMUL, listOf(pT, dO), vType)

    // dP = dO · V^T  →  shape [..., m, n]
    val vT = transposeLastTwo(v, vType)
    val dP = op(OpKind.MATMUL, listOf(dO, vT), sType)

    // dS = (dP - sum(P · dP, last_axis, keepdims=True)) · P
    val pdp = op(OpKind.MUL, listOf(pRecomputed, dP), sType)
    // SUM with reduction_dims = [lastAxis] drops the last axis.
    val reducedDims = sType.dims.toMutableList().also { it.removeAt(lastAxis) }
    val reducedType = DxirType(sType.dtype, reducedDims)
    val sumPdp = op(
        OpKind.SUM,
        listOf(pdp),
        reducedType,
        attrs = mapOf("reduction_dims" to listOf(lastAxis)),
    )
    // Broadcast back to sType. broadcast_dimensions specifies which axes
    // of the input map to which axes of the output; for last-axis reduce,
    // every kept axis stays at the same position.
    val keptAxes = (0 until rank).filter { it != lastAxis }
    val sumPdpBroadcast = op(
        OpKind.BROADCAST,
        listOf(sumPdp),
        sType,
        attrs = mapOf("broadcast_dimensions" to keptAxes),
    )
    val dPMinusSum = op(OpKind.SUB, listOf(dP, sumPdpBroadcast), sType)
    val dS = op(OpKind.MUL, listOf(dPMinusSum, pRecomputed), sType)

    // dQ = dS · K^T
    val kT = transposeLastTwo(k, kType)
    val dQ = op(OpKind.MATMUL, listOf(dS, kT), qType)

    // dK = Q^T · dS
    val qT = transposeLastTwo(q, qType)
    val dK = op(OpKind.MATMUL, listOf(qT, dS), kType)

    listOf(dQ, dK, dV)
}
