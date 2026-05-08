package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Layer 4 §0.4.314 — recognize the SwiGLU + down-projection fused MLP block.
 *
 * # Match shape
 *
 * Strict superset of [recognizeSwiGLU] — same SwiGLU activation plus the
 * trailing down-projection MATMUL:
 *
 * ```
 * gate_proj = MATMUL(x, W_gate)
 * up_proj   = MATMUL(x, W_up)
 * silu_g    = MUL(SILU(gate_proj), up_proj)        // or commuted
 * out       = MATMUL(silu_g, W_down)
 * ```
 *
 * Anchored on `OpKind.SILU` like [recognizeSwiGLU]. The recognizer reuses
 * the SwiGLU four-op detection (SILU + 2 MATMULs + gating MUL) and then
 * walks forward from the gating MUL to confirm exactly one MATMUL
 * consumer whose other operand is a non-SwiGLU node (the down-projection
 * weight).
 *
 * # Why this is a v2 compound
 *
 * `recognizeAll` runs both this recognizer and [recognizeSwiGLU] over the
 * same SILU anchor. When the down-proj is present, both recognizers
 * match — but their op-id sets overlap (the four SwiGLU ops). The §0.4.282
 * `resolveLargestMatch` resolver picks the larger match (this one,
 * 5 ops) and discards the bare SwiGLU. When the down-proj is absent
 * (e.g. the LM-head matmul that bypasses SwiGLU, or a SwiGLU whose
 * output feeds a non-MATMUL consumer), only the bare SwiGLU recognizer
 * matches and the resolver leaves it intact. Production has none of
 * the latter — every Llama-style decoder MLP has a down-proj — so this
 * compound subsumes the bare SwiGLU on every real model.
 *
 * # What's NOT matched
 *
 * - **Gating MUL has multiple consumers.** Llama's silu_g feeds only the
 *   down-proj, but a hypothetical residual or skip-connection sourced
 *   from silu_g would also need that op to survive. v1 declines if the
 *   gating MUL has any consumer other than the candidate down-proj
 *   MATMUL.
 * - **Bias-bearing variants.** Same as [recognizeSwiGLU]: bias-free only.
 * - **Down-proj MATMUL with extra operands.** The recognizer accepts
 *   the standard two-operand `MATMUL(silu_g, W_down)` shape; any
 *   batched-matmul attrs the user attached propagate through unchanged.
 *
 * # Near-miss diagnostics
 *
 * Inherits SwiGLU's near-misses (when the inner SwiGLU shape doesn't
 * match) plus three additional records when the SwiGLU matches but the
 * down-proj envelope doesn't:
 *
 * - The gating MUL's consumer isn't a MATMUL.
 * - The gating MUL has multiple consumers (so absorbing it into a
 *   COARSENED would lose values).
 * - The candidate down-proj MATMUL's other operand is itself a SwiGLU
 *   ingredient (gate/up matmul or the SILU) rather than a fresh weight.
 */
fun recognizeTransformerMLP(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.TransformerMLP> {
    val uses = buildUseListLocalTransformerMLP(fn)
    val out = mutableListOf<RecognitionMatch.TransformerMLP>()
    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.SILU) continue
        val match = tryMatchTransformerMLP(node, uses, diagnostics)
        if (match != null) out += match
    }
    return out
}

private fun tryMatchTransformerMLP(
    silu: DxirOp,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.TransformerMLP? {
    // 1-3. Identify the SwiGLU four-op core. Mirror recognizeSwiGLU's
    //      checks; emit diagnostics under the "TransformerMLP" pattern
    //      name so a near-miss reads as "you almost wrote a fused MLP".
    val siluOperand = silu.operands.singleOrNull()
    if (siluOperand !is DxirOp || siluOperand.op != OpKind.MATMUL) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "SILU operand is ${(siluOperand as? DxirOp)?.op ?: "non-op"}; " +
                    "expected MATMUL(x, W_gate)",
                opId = silu.id,
            ),
        )
        return null
    }
    val gateMatmul = siluOperand

    val gatingMul = uses[silu.id].orEmpty().firstOrNull { it.op == OpKind.MUL }
    if (gatingMul == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "SILU has no MUL consumer; gate computed but not gated",
                opId = silu.id,
            ),
        )
        return null
    }

    val gatingOther = gatingMul.operands.firstOrNull { it.id != silu.id }
    if (gatingOther !is DxirOp || gatingOther.op != OpKind.MATMUL || gatingOther.id == gateMatmul.id) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "gating MUL's other operand is " +
                    "${(gatingOther as? DxirOp)?.op ?: "non-op"}; expected a distinct MATMUL " +
                    "(the up-projection)",
                opId = gatingMul.id,
            ),
        )
        return null
    }
    val upMatmul = gatingOther

    val gateOperandIds = gateMatmul.operands.map { it.id }.toSet()
    val upOperandIds = upMatmul.operands.map { it.id }.toSet()
    val sharedIds = gateOperandIds.intersect(upOperandIds)
    if (sharedIds.size != 1) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "gate and up matmuls share ${sharedIds.size} operand(s); expected one (the input x)",
                opId = silu.id,
            ),
        )
        return null
    }
    val xId = sharedIds.single()
    val xInput: DxirNode = gateMatmul.operands.first { it.id == xId }
    val wGate: DxirNode = gateMatmul.operands.first { it.id != xId }
    val wUp: DxirNode = upMatmul.operands.first { it.id != xId }

    // 4. Walk forward from the gating MUL: it must have exactly one
    //    consumer, and that consumer must be a MATMUL.
    val gatingConsumers = uses[gatingMul.id].orEmpty()
    if (gatingConsumers.size != 1) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "gating MUL has ${gatingConsumers.size} consumers; " +
                    "expected exactly one (the down-projection MATMUL). " +
                    "Multi-consumer silu_g would make the fused-MLP envelope unsafe to absorb",
                opId = gatingMul.id,
            ),
        )
        return null
    }
    val downMatmul = gatingConsumers.single()
    if (downMatmul.op != OpKind.MATMUL) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "gating MUL's consumer is ${downMatmul.op}; expected MATMUL (the down-projection)",
                opId = downMatmul.id,
            ),
        )
        return null
    }

    // 5. The down-proj MATMUL must have exactly two operands, one of
    //    which is the gating MUL. The other becomes wDown — it must not
    //    be one of the SwiGLU ingredients (else the user wrote a
    //    self-referential graph that doesn't fit the fused-MLP shape).
    if (downMatmul.operands.size != 2) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "down-proj MATMUL has ${downMatmul.operands.size} operands; expected 2",
                opId = downMatmul.id,
            ),
        )
        return null
    }
    val wDown: DxirNode = downMatmul.operands.first { it.id != gatingMul.id }
    val swiGLUInternalIds = setOf(silu.id, gateMatmul.id, upMatmul.id)
    if (wDown.id in swiGLUInternalIds) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "TransformerMLP",
                reason = "down-proj weight is one of the SwiGLU's internal ops " +
                    "(id=${wDown.id}); expected a fresh weight tensor",
                opId = downMatmul.id,
            ),
        )
        return null
    }

    return RecognitionMatch.TransformerMLP(
        ops = listOf(silu, gateMatmul, upMatmul, gatingMul, downMatmul),
        xInput = xInput,
        wGate = wGate,
        wUp = wUp,
        wDown = wDown,
        output = downMatmul,
        xType = xInput.type,
        outputType = downMatmul.type,
    )
}

private fun buildUseListLocalTransformerMLP(fn: DxirFunction): Map<Int, List<DxirOp>> {
    val uses = HashMap<Int, MutableList<DxirOp>>()
    for (node in fn.body) {
        if (node !is DxirOp) continue
        for (operand in node.operands) {
            uses.getOrPut(operand.id) { mutableListOf() } += node
        }
    }
    return uses
}
