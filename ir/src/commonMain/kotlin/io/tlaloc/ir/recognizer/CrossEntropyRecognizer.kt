package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Recognize cross-entropy compound forms.
 *
 * # Match shape
 *
 * Canonical (negative log-likelihood) cross entropy: `-Σ labels·log(softmax(logits))`.
 * The DXIR sequence:
 *
 * ```
 * probs = SOFTMAX(logits)
 * logp  = LOG(probs)
 * prod  = MUL(labels, logp)
 * loss  = SUM(prod)               // optionally followed by NEG and DIV(N) for mean
 * ```
 *
 * The recognizer anchors on `OpKind.LOG` (rare immediately after SOFTMAX
 * outside of NLL). Walks back to confirm the LOG operand is a SOFTMAX,
 * forward to confirm a downstream MUL+SUM (the loss accumulation).
 *
 * # What's NOT matched
 *
 * - Stable `LogSumExp` form (`LOGSUMEXP(logits) - logits[label]`) is not
 *   matched; a separate `recognizeLogSumExpCrossEntropy` could cover
 *   it later.
 * - Sparse / one-hot label optimisations that bypass the explicit MUL
 *   with labels.
 *
 * # Near-miss diagnostics
 *
 * - LOG operand isn't SOFTMAX — likely a generic logarithm, not NLL.
 * - LOG result isn't multiplied by labels — likely a log-prob output
 *   without the loss accumulation.
 * - MUL result isn't reduced by SUM — likely an unreduced log-likelihood
 *   tensor, not a scalar loss.
 */
fun recognizeCrossEntropy(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.CrossEntropy> {
    val uses = buildUseListLocalCe(fn)
    val out = mutableListOf<RecognitionMatch.CrossEntropy>()
    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.LOG) continue
        val match = tryMatchCrossEntropy(node, uses, diagnostics)
        if (match != null) out += match
    }
    return out
}

private fun tryMatchCrossEntropy(
    log: DxirOp,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.CrossEntropy? {
    // 1. LOG operand should be SOFTMAX.
    val logOperand = log.operands.singleOrNull()
    if (logOperand !is DxirOp || logOperand.op != OpKind.SOFTMAX) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "CrossEntropy",
                reason = "LOG operand is ${(logOperand as? DxirOp)?.op ?: "non-op"}; expected SOFTMAX (NLL form)",
                opId = log.id,
            ),
        )
        return null
    }
    val softmax = logOperand
    val logits = softmax.operands.firstOrNull()
        ?: return null

    // 2. LOG must have a MUL consumer (the labels · log_probs step).
    val mul = uses[log.id].orEmpty().firstOrNull { it.op == OpKind.MUL }
    if (mul == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "CrossEntropy",
                reason = "LOG(SOFTMAX(...)) has no MUL consumer; likely a log-prob output without the labels multiply",
                opId = log.id,
            ),
        )
        return null
    }

    // 3. The MUL's other operand is the labels tensor.
    val labels = mul.operands.firstOrNull { it.id != log.id }
        ?: return null

    // 4. MUL must be reduced by SUM (the loss accumulation).
    val sum = uses[mul.id].orEmpty().firstOrNull { it.op == OpKind.SUM }
    if (sum == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "CrossEntropy",
                reason = "MUL(labels, log_probs) isn't reduced by SUM; likely an unreduced log-likelihood tensor, not a scalar loss",
                opId = mul.id,
            ),
        )
        return null
    }

    return RecognitionMatch.CrossEntropy(
        ops = listOf(softmax, log, mul, sum),
        logits = logits,
        labels = labels,
        output = sum,
    )
}

private fun buildUseListLocalCe(fn: DxirFunction): Map<Int, List<DxirOp>> {
    val uses = HashMap<Int, MutableList<DxirOp>>()
    for (node in fn.body) {
        if (node !is DxirOp) continue
        for (operand in node.operands) {
            uses.getOrPut(operand.id) { mutableListOf() } += node
        }
    }
    return uses
}
