package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Recognize `MATMUL → SOFTMAX → MATMUL` compound
 * forms typical of unfused attention forward passes.
 *
 * # Match shape
 *
 * The recognizer walks the function body, finds each `OpKind.SOFTMAX`
 * node, and checks whether:
 *
 * 1. Its sole operand is a `OpKind.MATMUL` op (the QK matmul producing
 *    raw scores).
 * 2. The softmax's result has at least one consumer that's another
 *    `OpKind.MATMUL` op (the PV matmul producing the attention output).
 *
 * If both hold, emit a [RecognitionMatch.FlashAttention] carrying the
 * three matched ops + the Q / K / V leaf inputs. The recognizer is
 * structural — it doesn't depend on a specific source-language form.
 *
 * # What's NOT matched (by design)
 *
 * - Pre-fused `OpKind.SCALED_DOT_PRODUCT_ATTENTION` ops. Users who opt
 *   into the pre-fused op already have the structure the downstream
 *   coarsener wants; matching them again would duplicate work.
 *   A separate `recognizeFusedAttention` could surface those for
 *   uniform handling later.
 * - Causal / variable-length predication.
 * - Multi-head attention with explicit head-index projection ops between
 *   the matmul and softmax. Only the simplest compound form is matched.
 *
 * # Near-miss diagnostics
 *
 * When `OpKind.SOFTMAX` appears but the surrounding pattern doesn't
 * match, append a [RecognitionDiagnostic] describing why. Examples:
 *
 * - "softmax operand is not a MATMUL"
 * - "softmax has no MATMUL consumer (final loss term?)"
 * - "softmax → matmul exists but the matmul's other operand is the same
 *   tensor as the matmul before softmax (looks like A·A^T not Q·K^T·V)"
 */
fun recognizeFlashAttention(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.FlashAttention> {
    // Build a one-pass use list: for each op, the ops that consume it.
    val uses: Map<Int, List<DxirOp>> = buildUseList(fn)

    val out = mutableListOf<RecognitionMatch.FlashAttention>()
    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.SOFTMAX) continue
        val match = tryMatchAroundSoftmax(node, uses, diagnostics)
        if (match != null) out += match
    }
    return out
}

/** Walk the function body once, build node-id → consuming-ops map. */
private fun buildUseList(fn: DxirFunction): Map<Int, List<DxirOp>> {
    val uses = HashMap<Int, MutableList<DxirOp>>()
    for (node in fn.body) {
        if (node !is DxirOp) continue
        for (operand in node.operands) {
            uses.getOrPut(operand.id) { mutableListOf() } += node
        }
    }
    return uses
}

private fun tryMatchAroundSoftmax(
    softmax: DxirOp,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.FlashAttention? {
    // 1. Softmax's operand must be the QK matmul.
    val softmaxOperand = softmax.operands.singleOrNull()
    if (softmaxOperand !is DxirOp) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "FlashAttention",
                reason = "softmax operand is not an op (likely a param/const) — softmax(QK) chain expected",
                opId = softmax.id,
            ),
        )
        return null
    }
    if (softmaxOperand.op != OpKind.MATMUL) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "FlashAttention",
                reason = "softmax operand is ${softmaxOperand.op}, expected MATMUL (Q · K^T)",
                opId = softmax.id,
            ),
        )
        return null
    }
    val qkMatmul = softmaxOperand

    // 2. Softmax must have a MATMUL consumer (the PV matmul).
    val softmaxConsumers = uses[softmax.id].orEmpty()
    val pvMatmul = softmaxConsumers.firstOrNull { it.op == OpKind.MATMUL }
    if (pvMatmul == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "FlashAttention",
                reason = "softmax has no MATMUL consumer; downstream is " +
                    softmaxConsumers.joinToString(",") { it.op.name } +
                    " (final-loss term, not attention)",
                opId = softmax.id,
            ),
        )
        return null
    }

    // 3. The PV matmul's operand list must include the softmax result + a
    //    distinct V tensor. Reject self-contractions (A · A^T).
    val pvOperands = pvMatmul.operands
    if (pvOperands.size != 2) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "FlashAttention",
                reason = "PV matmul has ${pvOperands.size} operands; expected 2",
                opId = pvMatmul.id,
            ),
        )
        return null
    }
    // The PV matmul should consume the softmax (directly or via OpResult);
    // the *other* operand is V. Identify which side is which.
    val pvLhsIsSoftmax = pvOperands[0].id == softmax.id
    val pvRhsIsSoftmax = pvOperands[1].id == softmax.id
    if (!pvLhsIsSoftmax && !pvRhsIsSoftmax) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "FlashAttention",
                reason = "PV matmul operands don't include the softmax result " +
                    "(uses[] returned ${pvMatmul.id} as a consumer but operand check disagrees)",
                opId = pvMatmul.id,
            ),
        )
        return null
    }
    val vInput = if (pvLhsIsSoftmax) pvOperands[1] else pvOperands[0]

    // 4. The QK matmul's operands are Q and K (in some order).
    val qkOperands = qkMatmul.operands
    if (qkOperands.size != 2) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "FlashAttention",
                reason = "QK matmul has ${qkOperands.size} operands; expected 2",
                opId = qkMatmul.id,
            ),
        )
        return null
    }
    val qInput = qkOperands[0]
    val kInput = qkOperands[1]

    // 5. Reject self-contractions: if Q ≡ K and Q ≡ V (the A·A^T·A
    //    look-alike), this isn't attention.
    if (qInput.id == kInput.id && kInput.id == vInput.id) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "FlashAttention",
                reason = "Q ≡ K ≡ V (self-contraction); likely a stylized matmul chain, not attention",
                opId = qkMatmul.id,
            ),
        )
        return null
    }

    return RecognitionMatch.FlashAttention(
        qkMatmul = qkMatmul,
        softmax = softmax,
        pvMatmul = pvMatmul,
        qInput = qInput,
        kInput = kInput,
        vInput = vInput,
        scoreType = qkMatmul.types.first(),
        outputType = pvMatmul.types.first(),
    )
}
