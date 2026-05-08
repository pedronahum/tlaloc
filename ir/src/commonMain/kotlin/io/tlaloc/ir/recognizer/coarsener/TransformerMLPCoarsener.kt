package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * Layer 4 §0.4.314 — TransformerMLP analytical-backward coarsener.
 *
 * # What this is
 *
 * Strict superset of [coarsenSwiGLU] for the SwiGLU + down-projection
 * fused-MLP block. Consumes a [RecognitionMatch.TransformerMLP] and
 * produces a [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — recapitulates `MATMUL(x, W_gate)`,
 *   `MATMUL(x, W_up)`, `SILU(gate)`, `silu_g · up`, and
 *   `MATMUL(silu_g, W_down)`, preserving each operand's left/right
 *   position from the user's original graph. The COARSENED's four
 *   operands map positionally to `(x, W_gate, W_up, W_down)`.
 * - **`gradient_body`** — analytical VJP. Signature
 *   `(dy, x, W_gate, W_up, W_down) → (d_x, d_W_gate, d_W_up, d_W_down)`.
 *   The down-proj VJP fires first; its outputs feed the SwiGLU VJP for
 *   the chain back to `(x, W_gate, W_up)`.
 * - **`reads_primal_indices`** — `{0, 1, 2, 3}`: all four primal operands
 *   are dereferenced (recompute matmuls + the four matmul-VJP transposes).
 *
 * # The math
 *
 * Forward (canonical Llama: x-left in projections, silu-left in gating):
 *
 * ```
 * gate    = MATMUL(x, W_gate)              [..., D_ff]
 * up      = MATMUL(x, W_up)                [..., D_ff]
 * silu_g  = SILU(gate)                     [..., D_ff]
 * silu_up = silu_g · up                    [..., D_ff]
 * out     = MATMUL(silu_up, W_down)        [..., D_model]
 * ```
 *
 * Backward, given `dy = ∂L/∂out`. First the down-proj VJP:
 *
 * ```
 * d_silu_up = MATMUL(dy, W_down^T)
 * d_W_down  = MATMUL(silu_up^T, dy)
 * ```
 *
 * Then the SwiGLU VJP using `d_silu_up` as the upstream gradient:
 *
 * ```
 * d_silu_g    = d_silu_up · up
 * d_up        = d_silu_up · silu_g
 * silu_prime  = sig + silu_g − silu_g · sig    (no const-1 broadcast)
 * d_gate      = d_silu_g · silu_prime
 *
 * d_x_gate = MATMUL(d_gate, W_gate^T)
 * d_x_up   = MATMUL(d_up,   W_up^T)
 * d_W_gate = MATMUL(x^T,    d_gate)
 * d_W_up   = MATMUL(x^T,    d_up)
 *
 * d_x = d_x_gate + d_x_up
 * ```
 *
 * Identical math to the SwiGLU coarsener for the `(x, W_gate, W_up)`
 * gradients — just with `d_silu_up` driven by the down-proj VJP rather
 * than supplied directly as the upstream gradient.
 *
 * # Recompute strategy
 *
 * Mirrors the SwiGLU coarsener: recompute everything (gate, up, silu_g,
 * silu_up, sig) from the four primal operands. The recompute matmuls
 * (gate and up) dominate the gradient body's cost, so v2 may want to
 * hoist them into the COARSENED's payload as additional primal
 * outputs — matches the SwiGLU coarsener's identical TODO.
 *
 * # Scope notes
 *
 * - **Type uniformity for the SwiGLU half.** Same as
 *   [coarsenSwiGLU]: `gateMatmul.type == upMatmul.type` and
 *   `wGate.type == wUp.type`.
 * - **Down-proj shape.** The down MATMUL must be a standard two-operand
 *   matmul. `silu_up.type` and `wDown.type` can differ (different
 *   widths) — that's the whole point of the down-proj. Result type is
 *   the COARSENED's output type.
 * - **Rank ≥ 2** for transposes. Same as SwiGLU.
 */
internal fun coarsenTransformerMLP(
    match: RecognitionMatch.TransformerMLP,
): CoarsenedBundle? {
    val opsList = match.ops
    if (opsList.size != 5) return null
    val silu = opsList[0]
    val gateMatmul = opsList[1]
    val upMatmul = opsList[2]
    val gatingMul = opsList[3]
    val downMatmul = opsList[4]

    if (silu.op != OpKind.SILU || gateMatmul.op != OpKind.MATMUL ||
        upMatmul.op != OpKind.MATMUL || gatingMul.op != OpKind.MUL ||
        downMatmul.op != OpKind.MATMUL
    ) return null

    val x = match.xInput
    val wGate = match.wGate
    val wUp = match.wUp
    val wDown = match.wDown

    // SwiGLU-half type checks (same as coarsenSwiGLU).
    if (gateMatmul.type != upMatmul.type) return null
    if (wGate.type != wUp.type) return null

    // Rank-2 minimum for swap-last-two transposes on every matmul-VJP.
    if (x.type.rank < 2 || wGate.type.rank < 2 || gateMatmul.type.rank < 2 ||
        wDown.type.rank < 2 || downMatmul.type.rank < 2
    ) return null

    // Preserve user's MATMUL operand order in the primal so emit fidelity
    // matches the original graph.
    if (gateMatmul.operands.size != 2 || upMatmul.operands.size != 2 ||
        gatingMul.operands.size != 2 || downMatmul.operands.size != 2
    ) return null
    val xIsLeftInGateMatmul = gateMatmul.operands[0].id == x.id
    val xIsLeftInUpMatmul = upMatmul.operands[0].id == x.id
    val siluIsLeftInGatingMul = gatingMul.operands[0].id == silu.id
    val siluUpIsLeftInDownMatmul = downMatmul.operands[0].id == gatingMul.id

    val absorbedOps = setOf(silu.id, gateMatmul.id, upMatmul.id, gatingMul.id, downMatmul.id)
    val outerOperands = listOf(x, wGate, wUp, wDown)

    val primalBody = buildTransformerMLPPrimal(
        xType = x.type,
        wGateType = wGate.type,
        wUpType = wUp.type,
        wDownType = wDown.type,
        gateProjType = gateMatmul.type,
        upProjType = upMatmul.type,
        siluType = silu.type,
        siluUpType = gatingMul.type,
        outputType = downMatmul.type,
        xIsLeftInGateMatmul = xIsLeftInGateMatmul,
        xIsLeftInUpMatmul = xIsLeftInUpMatmul,
        siluIsLeftInGatingMul = siluIsLeftInGatingMul,
        siluUpIsLeftInDownMatmul = siluUpIsLeftInDownMatmul,
    )
    val gradientBody = buildTransformerMLPGradient(
        xType = x.type,
        wGateType = wGate.type,
        wUpType = wUp.type,
        wDownType = wDown.type,
        gateProjType = gateMatmul.type,
        siluUpType = gatingMul.type,
        outputType = downMatmul.type,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = absorbedOps,
        anchorOpId = downMatmul.id,
        outerOperands = outerOperands,
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

private fun buildTransformerMLPPrimal(
    xType: DxirType,
    wGateType: DxirType,
    wUpType: DxirType,
    wDownType: DxirType,
    gateProjType: DxirType,
    upProjType: DxirType,
    siluType: DxirType,
    siluUpType: DxirType,
    outputType: DxirType,
    xIsLeftInGateMatmul: Boolean,
    xIsLeftInUpMatmul: Boolean,
    siluIsLeftInGatingMul: Boolean,
    siluUpIsLeftInDownMatmul: Boolean,
): DxirFunction = DxirBuilder.function("transformer_mlp_primal") {
    val x = param("x", xType)
    val wGate = param("w_gate", wGateType)
    val wUp = param("w_up", wUpType)
    val wDown = param("w_down", wDownType)

    val gateOperands = if (xIsLeftInGateMatmul) listOf(x, wGate) else listOf(wGate, x)
    val upOperands = if (xIsLeftInUpMatmul) listOf(x, wUp) else listOf(wUp, x)
    val gate = op(OpKind.MATMUL, gateOperands, gateProjType)
    val up = op(OpKind.MATMUL, upOperands, upProjType)
    val siluGate = op(OpKind.SILU, listOf(gate), siluType)

    val gatingOperands = if (siluIsLeftInGatingMul) listOf(siluGate, up) else listOf(up, siluGate)
    val siluUp = op(OpKind.MUL, gatingOperands, siluUpType)

    val downOperands = if (siluUpIsLeftInDownMatmul) listOf(siluUp, wDown) else listOf(wDown, siluUp)
    val out = op(OpKind.MATMUL, downOperands, outputType)
    listOf(out)
}

private fun buildTransformerMLPGradient(
    xType: DxirType,
    wGateType: DxirType,
    wUpType: DxirType,
    wDownType: DxirType,
    gateProjType: DxirType,
    siluUpType: DxirType,
    outputType: DxirType,
): DxirFunction = DxirBuilder.function("transformer_mlp_grad") {
    val dy = param("dy", outputType)
    val x = param("x", xType)
    val wGate = param("w_gate", wGateType)
    val wUp = param("w_up", wUpType)
    val wDown = param("w_down", wDownType)

    // Recompute forward intermediates needed by the gradient.
    val gate = op(OpKind.MATMUL, listOf(x, wGate), gateProjType)
    val up = op(OpKind.MATMUL, listOf(x, wUp), gateProjType)
    val siluG = op(OpKind.SILU, listOf(gate), gateProjType)
    val sig = op(OpKind.SIGMOID, listOf(gate), gateProjType)
    val siluUp = op(OpKind.MUL, listOf(siluG, up), siluUpType)

    // Down-proj VJP: rank ≥ 2 swap-last-two transpose convention.
    fun transposeLastTwo(operand: DxirNode, opType: DxirType): DxirNode {
        val rank = opType.rank
        val perm: List<Int> = (0 until rank - 2).toList() + listOf(rank - 1, rank - 2)
        val swapped = opType.dims.toMutableList().also {
            val a = it[rank - 2]; it[rank - 2] = it[rank - 1]; it[rank - 1] = a
        }
        val outType = DxirType(opType.dtype, swapped)
        return op(OpKind.TRANSPOSE, listOf(operand), outType, attrs = mapOf("permutation" to perm))
    }
    val wDownT = transposeLastTwo(wDown, wDown.type)
    val siluUpT = transposeLastTwo(siluUp, siluUp.type)
    val xT = transposeLastTwo(x, x.type)
    val wGateT = transposeLastTwo(wGate, wGate.type)
    val wUpT = transposeLastTwo(wUp, wUp.type)

    // d_silu_up = MATMUL(dy, W_down^T) ; d_W_down = MATMUL(silu_up^T, dy).
    val dSiluUp = op(OpKind.MATMUL, listOf(dy, wDownT), siluUpType)
    val dWDown = op(OpKind.MATMUL, listOf(siluUpT, dy), wDownType)

    // SwiGLU VJP, identical to coarsenSwiGLU but driven by d_silu_up
    // rather than the COARSENED's direct upstream.
    val dSiluG = op(OpKind.MUL, listOf(dSiluUp, up), gateProjType)
    val dUp = op(OpKind.MUL, listOf(dSiluUp, siluG), gateProjType)

    val siluSig = op(OpKind.MUL, listOf(siluG, sig), gateProjType)
    val sigPlusSilu = op(OpKind.ADD, listOf(sig, siluG), gateProjType)
    val siluPrime = op(OpKind.SUB, listOf(sigPlusSilu, siluSig), gateProjType)
    val dGate = op(OpKind.MUL, listOf(dSiluG, siluPrime), gateProjType)

    val dXGate = op(OpKind.MATMUL, listOf(dGate, wGateT), xType)
    val dXUp = op(OpKind.MATMUL, listOf(dUp, wUpT), xType)
    val dX = op(OpKind.ADD, listOf(dXGate, dXUp), xType)

    val dWGate = op(OpKind.MATMUL, listOf(xT, dGate), wGateType)
    val dWUp = op(OpKind.MATMUL, listOf(xT, dUp), wUpType)

    listOf(dX, dWGate, dWUp, dWDown)
}
