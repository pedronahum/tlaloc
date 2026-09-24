package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Recognize RMS-norm compound forms.
 *
 * # Match shape
 *
 * Canonical RMS norm: `x * rsqrt(mean(x²) + eps)` (or `mean(x²)` without
 * the epsilon stabiliser). The DXIR sequence:
 *
 * ```
 * sq    = MUL(x, x)
 * mean  = MEAN(sq)             // optionally followed by ADD(mean, eps)
 * rsq   = RSQRT(mean_or_sum)
 * out   = MUL(x, rsq)          // same x as in the squared product
 * ```
 *
 * The recognizer anchors on `OpKind.RSQRT` (rare in non-norm graphs) and
 * walks back to confirm the `MEAN(MUL(x, x))` structure, then forward to
 * confirm the final multiply uses the *same* x tensor.
 *
 * Near-miss diagnostics fire when:
 * - RSQRT operand isn't a MEAN or `ADD(MEAN, ...)` — likely a different
 *   normalisation form (e.g. `1/sqrt(variance)` from a batch-norm).
 * - MEAN operand isn't a MUL — likely a plain reduction, not RMS.
 * - MUL(x, x) operands aren't the *same* tensor — likely a generic
 *   pairwise product, not a square.
 * - RSQRT result isn't multiplied back by the original x — the chain
 *   exists but isn't wired up as RMS.
 */
fun recognizeRmsNorm(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.RmsNorm> {
    val uses = buildUseListLocal(fn)
    val out = mutableListOf<RecognitionMatch.RmsNorm>()
    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.RSQRT) continue
        val match = tryMatchRmsNorm(node, uses, diagnostics)
        if (match != null) out += match
    }
    return out
}

private fun tryMatchRmsNorm(
    rsqrt: DxirOp,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.RmsNorm? {
    // 1. RSQRT operand should be MEAN, or ADD(MEAN, scalar-eps).
    val rsqrtOperand = rsqrt.operands.singleOrNull()
    if (rsqrtOperand !is DxirOp) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "RmsNorm",
                reason = "RSQRT operand is not an op (param/const); expected MEAN(x²) for RMS norm",
                opId = rsqrt.id,
            ),
        )
        return null
    }
    val meanOp: DxirOp = when (rsqrtOperand.op) {
        OpKind.MEAN -> rsqrtOperand
        OpKind.ADD -> {
            val nestedMean = rsqrtOperand.operands.firstOrNull { it is DxirOp && it.op == OpKind.MEAN } as? DxirOp
            if (nestedMean == null) {
                diagnostics?.add(
                    RecognitionDiagnostic(
                        pattern = "RmsNorm",
                        reason = "RSQRT operand is ADD but no MEAN among its operands — likely a different norm form",
                        opId = rsqrt.id,
                    ),
                )
                return null
            }
            nestedMean
        }
        else -> {
            diagnostics?.add(
                RecognitionDiagnostic(
                    pattern = "RmsNorm",
                    reason = "RSQRT operand is ${rsqrtOperand.op}; expected MEAN or ADD(MEAN, eps)",
                    opId = rsqrt.id,
                ),
            )
            return null
        }
    }

    // 2. MEAN operand should be MUL(x, x).
    val meanOperand = meanOp.operands.singleOrNull()
    if (meanOperand !is DxirOp || meanOperand.op != OpKind.MUL) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "RmsNorm",
                reason = "MEAN operand is ${(meanOperand as? DxirOp)?.op ?: "non-op"}; " +
                    "expected MUL(x, x) (the squared step)",
                opId = meanOp.id,
            ),
        )
        return null
    }
    val sqMul: DxirOp = meanOperand
    val sqA = sqMul.operands.getOrNull(0)
    val sqB = sqMul.operands.getOrNull(1)
    if (sqA == null || sqB == null || sqA.id != sqB.id) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "RmsNorm",
                reason = "MUL operands aren't the same tensor (id=${sqA?.id} vs id=${sqB?.id}); " +
                    "expected x · x (squared)",
                opId = sqMul.id,
            ),
        )
        return null
    }
    val xInput: DxirNode = sqA

    // 3. RSQRT result should be multiplied back by the original x.
    val rsqrtConsumers = uses[rsqrt.id].orEmpty()
    val finalMul = rsqrtConsumers.firstOrNull { consumer ->
        consumer.op == OpKind.MUL && consumer.operands.any { it.id == xInput.id }
    }
    if (finalMul == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "RmsNorm",
                reason = "RSQRT has no MUL consumer that includes the original x (id=${xInput.id}); " +
                    "chain exists but isn't wired up as RMS norm",
                opId = rsqrt.id,
            ),
        )
        return null
    }

    // Build the matched ops list. Includes ADD only if it was the
    // intermediate; otherwise just MUL→MEAN→RSQRT→MUL.
    val matchedOps = buildList {
        add(sqMul)
        add(meanOp)
        if (rsqrtOperand.op == OpKind.ADD) add(rsqrtOperand)
        add(rsqrt)
        add(finalMul)
    }
    return RecognitionMatch.RmsNorm(
        ops = matchedOps,
        input = xInput,
        output = finalMul,
    )
}

/**
 * File-private helper: same shape as the FlashAttention recognizer's
 * use-list builder. Kotlin doesn't share `internal` helpers across files
 * cleanly without exposing them at the package level; we duplicate ~10
 * lines here to keep each recognizer self-contained.
 */
private fun buildUseListLocal(fn: DxirFunction): Map<Int, List<DxirOp>> {
    val uses = HashMap<Int, MutableList<DxirOp>>()
    for (node in fn.body) {
        if (node !is DxirOp) continue
        for (operand in node.operands) {
            uses.getOrPut(operand.id) { mutableListOf() } += node
        }
    }
    return uses
}
