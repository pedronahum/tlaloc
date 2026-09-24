package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * SwiGLU analytical-backward coarsener.
 *
 * # What this is
 *
 * Consumes a [RecognitionMatch.SwiGLU] (the
 * `MATMUL(x, W_gate) → SILU → MUL ← MATMUL(x, W_up)` Llama-style MLP
 * gate) and produces a [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — a free-standing [DxirFunction] that recapitulates
 *   the matched activation, preserving each MATMUL's operand order
 *   (x-left vs x-right) and the gating MUL's operand order
 *   (silu-left vs up-left). The COARSENED's three operands map
 *   positionally to `(x, W_gate, W_up)`.
 * - **`gradient_body`** — the analytical VJP. Signature
 *   `(dy, x, W_gate, W_up) → (d_x, d_W_gate, d_W_up)`. Recomputes
 *   `gate`, `up`, `silu(gate)`, `sigmoid(gate)` internally and emits the
 *   matmul-VJPs through both projections, summing the two gradient
 *   flows into `d_x`.
 * - **`reads_primal_indices`** — `{0, 1, 2}`: the gradient body
 *   dereferences `x` (for the recompute matmuls + the `x^T` weight
 *   gradients), `W_gate` (for `W_gate^T` in `d_x_gate`), and `W_up`
 *   (for `W_up^T` in `d_x_up`).
 *
 * # The math
 *
 * Forward (canonical, silu-left in gating MUL, x-left in both MATMULs):
 *
 * ```
 * gate   = MATMUL(x, W_gate)              [..., D_ff]
 * up     = MATMUL(x, W_up)                [..., D_ff]
 * silu_g = SILU(gate)                     [..., D_ff]
 * out    = silu_g · up                    [..., D_ff]
 * ```
 *
 * SILU's derivative (used in the gradient body):
 *
 * ```
 * SILU(z)   = z · sigmoid(z)
 * SILU'(z)  = sigmoid(z) + z · sigmoid(z) · (1 − sigmoid(z))
 *           = sigmoid(z) · (1 + z · (1 − sigmoid(z)))
 *           = sigmoid(z) + SILU(z) · (1 − sigmoid(z))
 *           = sigmoid(z) + SILU(z) − SILU(z) · sigmoid(z)
 * ```
 *
 * The "no-const-1" form on the last line lets us emit the derivative
 * with only ADD/SUB/MUL plus the recomputed `silu_g` and `sig` —
 * avoiding a const-1-broadcast op.
 *
 * Backward, given `dy = ∂L/∂out`:
 *
 * ```
 * d_silu_g    = dy · up
 * d_up        = dy · silu_g
 * silu_prime  = sig + silu_g − silu_g · sig
 * d_gate      = d_silu_g · silu_prime
 *
 * # Through MATMUL(x, W_gate) → gate:
 * d_x_gate = MATMUL(d_gate, W_gate^T)
 * d_W_gate = MATMUL(x^T, d_gate)
 *
 * # Through MATMUL(x, W_up) → up:
 * d_x_up   = MATMUL(d_up, W_up^T)
 * d_W_up   = MATMUL(x^T, d_up)
 *
 * d_x = d_x_gate + d_x_up
 * ```
 *
 * Transposes use the FlashAttention coarsener's
 * "swap-last-two-axes" convention, treating the last two axes as the
 * matrix axes and leaving any leading batch axes untouched. This
 * mirrors the StableHLO matmul-VJP convention.
 *
 * # Scope notes
 *
 * - **Type uniformity required.** `gateMatmul.type == upMatmul.type` and
 *   `wGate.type == wUp.type` — the canonical Llama form. Mismatched
 *   shapes between the two projections (the rare case where gate and up
 *   have different output widths) decline (`return null`) for v1.
 *
 * - **Rank ≥ 2.** The matmul-VJP transposes need at least two axes.
 *   Higher-rank batched matmuls (rank 3+ with batch in front) work via
 *   the swap-last-two convention. The `d_W_gate = MATMUL(x^T, d_gate)`
 *   contraction reduces over the batch axes implicitly — this matches
 *   StableHLO's matmul semantics, which sums over all leading axes.
 *
 * - **No bias.** The recognizer matches the bias-free form (no ADD
 *   between MATMUL and SILU), which is what Llama-2/3, Mistral, and
 *   PaLM ship. A bias-bearing variant would land as a separate
 *   recognizer + coarsener.
 *
 * - **Recompute everything.** Mirrors RmsNorm/RoPE/CrossEntropy/FA's
 *   "recompute is cheaper than thread" choice. A later version could hoist `gate`,
 *   `up`, `silu_g`, `sig` into the COARSENED's payload as additional
 *   primal returns to skip the recompute matmuls — a meaningful win
 *   for the SwiGLU case, since the recompute matmuls are the heaviest
 *   part of the gradient body.
 */
internal fun coarsenSwiGLU(
    match: RecognitionMatch.SwiGLU,
): CoarsenedBundle? {
    val opsList = match.ops
    if (opsList.size != 4) return null
    val silu = opsList[0]
    val gateMatmul = opsList[1]
    val upMatmul = opsList[2]
    val gatingMul = opsList[3]

    if (silu.op != OpKind.SILU || gateMatmul.op != OpKind.MATMUL ||
        upMatmul.op != OpKind.MATMUL || gatingMul.op != OpKind.MUL
    ) return null

    val x = match.xInput
    val wGate = match.wGate
    val wUp = match.wUp

    // v1: gate and up projections must share the same output type and
    // the same weight type — the canonical Llama form. Mismatched shapes
    // would need broader matmul-VJP handling than v1 emits.
    if (gateMatmul.type != upMatmul.type) return null
    if (wGate.type != wUp.type) return null

    // Rank-2 minimum for the swap-last-two transposes.
    if (x.type.rank < 2 || wGate.type.rank < 2 || gateMatmul.type.rank < 2) return null

    // Preserve the user's MATMUL operand order in the primal so emit
    // fidelity matches the original graph.
    if (gateMatmul.operands.size != 2 || upMatmul.operands.size != 2) return null
    val xIsLeftInGateMatmul = gateMatmul.operands[0].id == x.id
    val xIsLeftInUpMatmul = upMatmul.operands[0].id == x.id

    if (gatingMul.operands.size != 2) return null
    val siluIsLeftInGatingMul = gatingMul.operands[0].id == silu.id

    val absorbedOps = setOf(silu.id, gateMatmul.id, upMatmul.id, gatingMul.id)
    val outerOperands = listOf(x, wGate, wUp)

    val primalBody = buildSwiGLUPrimal(
        xType = x.type,
        wGateType = wGate.type,
        wUpType = wUp.type,
        gateProjType = gateMatmul.type,
        upProjType = upMatmul.type,
        siluType = silu.type,
        outputType = gatingMul.type,
        xIsLeftInGateMatmul = xIsLeftInGateMatmul,
        xIsLeftInUpMatmul = xIsLeftInUpMatmul,
        siluIsLeftInGatingMul = siluIsLeftInGatingMul,
    )
    val gradientBody = buildSwiGLUGradient(
        xType = x.type,
        wGateType = wGate.type,
        wUpType = wUp.type,
        gateProjType = gateMatmul.type,
        outputType = gatingMul.type,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = absorbedOps,
        anchorOpId = gatingMul.id,
        outerOperands = outerOperands,
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

private fun buildSwiGLUPrimal(
    xType: DxirType,
    wGateType: DxirType,
    wUpType: DxirType,
    gateProjType: DxirType,
    upProjType: DxirType,
    siluType: DxirType,
    outputType: DxirType,
    xIsLeftInGateMatmul: Boolean,
    xIsLeftInUpMatmul: Boolean,
    siluIsLeftInGatingMul: Boolean,
): DxirFunction = DxirBuilder.function("swiglu_primal") {
    val x = param("x", xType)
    val wGate = param("w_gate", wGateType)
    val wUp = param("w_up", wUpType)

    val gateOperands = if (xIsLeftInGateMatmul) listOf(x, wGate) else listOf(wGate, x)
    val upOperands = if (xIsLeftInUpMatmul) listOf(x, wUp) else listOf(wUp, x)
    val gate = op(OpKind.MATMUL, gateOperands, gateProjType)
    val up = op(OpKind.MATMUL, upOperands, upProjType)
    val siluGate = op(OpKind.SILU, listOf(gate), siluType)

    val gatingOperands = if (siluIsLeftInGatingMul) listOf(siluGate, up) else listOf(up, siluGate)
    val out = op(OpKind.MUL, gatingOperands, outputType)
    listOf(out)
}

private fun buildSwiGLUGradient(
    xType: DxirType,
    wGateType: DxirType,
    wUpType: DxirType,
    gateProjType: DxirType,
    outputType: DxirType,
): DxirFunction = DxirBuilder.function("swiglu_grad") {
    val dy = param("dy", outputType)
    val x = param("x", xType)
    val wGate = param("w_gate", wGateType)
    val wUp = param("w_up", wUpType)

    // Recompute forward intermediates needed by the gradient.
    val gate = op(OpKind.MATMUL, listOf(x, wGate), gateProjType)
    val up = op(OpKind.MATMUL, listOf(x, wUp), gateProjType)
    val siluG = op(OpKind.SILU, listOf(gate), gateProjType)
    val sig = op(OpKind.SIGMOID, listOf(gate), gateProjType)

    // d_silu_g = dy · up; d_up = dy · silu_g.
    val dSiluG = op(OpKind.MUL, listOf(dy, up), gateProjType)
    val dUp = op(OpKind.MUL, listOf(dy, siluG), gateProjType)

    // SILU'(gate) = sig + silu_g − silu_g · sig (no const-1 broadcast).
    val siluSig = op(OpKind.MUL, listOf(siluG, sig), gateProjType)
    val sigPlusSilu = op(OpKind.ADD, listOf(sig, siluG), gateProjType)
    val siluPrime = op(OpKind.SUB, listOf(sigPlusSilu, siluSig), gateProjType)

    // d_gate = d_silu_g · SILU'(gate).
    val dGate = op(OpKind.MUL, listOf(dSiluG, siluPrime), gateProjType)

    // Matmul-VJPs through both projections. Rank ≥ 2; treat the last two
    // axes as the matrix axes and leave any leading batch axes untouched
    // (mirrors FlashAttention's transposeLastTwo helper, inlined here to
    // keep each coarsener self-contained).
    fun transposeLastTwo(operand: DxirNode, opType: DxirType): DxirNode {
        val rank = opType.rank
        val perm: List<Int> = (0 until rank - 2).toList() + listOf(rank - 1, rank - 2)
        val swapped = opType.dims.toMutableList().also {
            val a = it[rank - 2]; it[rank - 2] = it[rank - 1]; it[rank - 1] = a
        }
        val outType = DxirType(opType.dtype, swapped)
        return op(OpKind.TRANSPOSE, listOf(operand), outType, attrs = mapOf("permutation" to perm))
    }
    val xT = transposeLastTwo(x, x.type)
    val wGateT = transposeLastTwo(wGate, wGate.type)
    val wUpT = transposeLastTwo(wUp, wUp.type)

    // d_x_gate = MATMUL(d_gate, W_gate^T) ; d_x_up = MATMUL(d_up, W_up^T).
    val dXGate = op(OpKind.MATMUL, listOf(dGate, wGateT), xType)
    val dXUp = op(OpKind.MATMUL, listOf(dUp, wUpT), xType)
    val dX = op(OpKind.ADD, listOf(dXGate, dXUp), xType)

    // d_W_gate = MATMUL(x^T, d_gate) ; d_W_up = MATMUL(x^T, d_up).
    val dWGate = op(OpKind.MATMUL, listOf(xT, dGate), wGateType)
    val dWUp = op(OpKind.MATMUL, listOf(xT, dUp), wUpType)

    listOf(dX, dWGate, dWUp)
}

