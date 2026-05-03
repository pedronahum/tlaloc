package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Layer 4 §0.4.267 — recognize the SwiGLU gated-MLP activation.
 *
 * # Match shape
 *
 * Llama / Mistral / PaLM-style MLP gate (no biases — the standard
 * Llama-2/3 form):
 *
 * ```
 * gate_proj = MATMUL(x, W_gate)
 * up_proj   = MATMUL(x, W_up)
 * out       = MUL(SILU(gate_proj), up_proj)        // or MUL(up_proj, SILU(gate_proj))
 * ```
 *
 * The recognizer anchors on `OpKind.SILU` (rare outside MLP gates) and
 * walks back to confirm the SILU's operand is a MATMUL, then forward to
 * confirm a MUL consumer whose other operand is *another* MATMUL sharing
 * exactly one operand-id with the gate matmul (the input `x`).
 *
 * # What's NOT matched
 *
 * - **Bias-bearing variants.** `SILU(MATMUL(x, W_gate) + b_gate)` adds an
 *   ADD between the MATMUL and SILU; v1 declines. Llama-style MLPs are
 *   bias-free, so this covers the dominant case.
 * - **Down projection.** v1 matches just the SwiGLU activation (two
 *   parallel matmuls + SILU + gating), not the surrounding
 *   `MATMUL(out, W_down)`. A future "TransformerMLP" recognizer can
 *   absorb the down-proj for cuBLASLt-style fused-MLP kernels — this
 *   is a strict superset and gets its own commit.
 * - **GeGLU / ReGLU.** Same structural shape but with GELU/RELU instead
 *   of SILU. Anchoring on a different OpKind would land them as separate
 *   recognizers; they share no code with this one beyond the shape.
 *
 * # Near-miss diagnostics
 *
 * - SILU operand isn't an op or isn't MATMUL — likely a generic SILU on
 *   an activation, not a gate.
 * - SILU has no MUL consumer — the gate is computed but not gated against
 *   anything.
 * - The gating MUL's other operand isn't a MATMUL — likely a hand-written
 *   `silu(x) · scalar` pattern, not SwiGLU.
 * - The two MATMULs share no input operand — the chain exists but isn't
 *   a co-rooted gate.
 */
fun recognizeSwiGLU(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.SwiGLU> {
    val uses = buildUseListLocalSwiGLU(fn)
    val out = mutableListOf<RecognitionMatch.SwiGLU>()
    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.SILU) continue
        val match = tryMatchSwiGLU(node, uses, diagnostics)
        if (match != null) out += match
    }
    return out
}

private fun tryMatchSwiGLU(
    silu: DxirOp,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.SwiGLU? {
    // 1. SILU operand should be MATMUL (the gate projection).
    val siluOperand = silu.operands.singleOrNull()
    if (siluOperand !is DxirOp || siluOperand.op != OpKind.MATMUL) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "SwiGLU",
                reason = "SILU operand is ${(siluOperand as? DxirOp)?.op ?: "non-op"}; " +
                    "expected MATMUL(x, W_gate) for SwiGLU gate projection",
                opId = silu.id,
            ),
        )
        return null
    }
    val gateMatmul = siluOperand

    // 2. SILU must have a MUL consumer (the gating step).
    val gatingMul = uses[silu.id].orEmpty().firstOrNull { it.op == OpKind.MUL }
    if (gatingMul == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "SwiGLU",
                reason = "SILU has no MUL consumer; gate computed but not gated against an up-projection",
                opId = silu.id,
            ),
        )
        return null
    }

    // 3. The gating MUL's other operand must be another MATMUL (the up
    //    projection), distinct from gateMatmul.
    val gatingOther = gatingMul.operands.firstOrNull { it.id != silu.id }
    if (gatingOther !is DxirOp || gatingOther.op != OpKind.MATMUL || gatingOther.id == gateMatmul.id) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "SwiGLU",
                reason = "gating MUL's other operand is " +
                    "${(gatingOther as? DxirOp)?.op ?: "non-op"}; expected a distinct MATMUL " +
                    "(the up-projection)",
                opId = gatingMul.id,
            ),
        )
        return null
    }
    val upMatmul = gatingOther

    // 4. The two MATMULs must share exactly one operand-id (the input x).
    //    Both matmul's other operand becomes the per-projection weight.
    val gateOperandIds = gateMatmul.operands.map { it.id }.toSet()
    val upOperandIds = upMatmul.operands.map { it.id }.toSet()
    val sharedIds = gateOperandIds.intersect(upOperandIds)
    if (sharedIds.size != 1) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "SwiGLU",
                reason = "gate and up matmuls share ${sharedIds.size} operand(s); " +
                    "expected exactly one (the input x)",
                opId = silu.id,
            ),
        )
        return null
    }
    val xId = sharedIds.single()
    val xInput: DxirNode = gateMatmul.operands.first { it.id == xId }
    val wGate: DxirNode = gateMatmul.operands.first { it.id != xId }
    val wUp: DxirNode = upMatmul.operands.first { it.id != xId }

    return RecognitionMatch.SwiGLU(
        ops = listOf(silu, gateMatmul, upMatmul, gatingMul),
        xInput = xInput,
        wGate = wGate,
        wUp = wUp,
        output = gatingMul,
        xType = xInput.type,
        outputType = gatingMul.type,
    )
}

private fun buildUseListLocalSwiGLU(fn: DxirFunction): Map<Int, List<DxirOp>> {
    val uses = HashMap<Int, MutableList<DxirOp>>()
    for (node in fn.body) {
        if (node !is DxirOp) continue
        for (operand in node.operands) {
            uses.getOrPut(operand.id) { mutableListOf() } += node
        }
    }
    return uses
}
