package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.RecognitionMatch

/**
 * Layer 4 §0.4.266 — Cross-entropy analytical-backward coarsener.
 *
 * # What this is
 *
 * Consumes a [RecognitionMatch.CrossEntropy] (the canonical
 * `SOFTMAX → LOG → MUL(labels, ·) → SUM` chain) and produces a
 * [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — a free-standing [DxirFunction] that recapitulates
 *   the matched forward, preserving the MUL's operand order
 *   (labels-first vs log-first) and the SUM's reduction attrs from the
 *   user's chain. The COARSENED op's two operands map positionally to
 *   `(logits, labels)`.
 * - **`gradient_body`** — the analytical VJP that *replaces* the
 *   decomposed softmax-then-log gradient with the textbook fused form.
 *   Signature `(d_loss, logits, labels) → (d_logits, d_labels)`.
 *   The point of coarsening cross-entropy: forward + decomposed backward
 *   would compute SOFTMAX three times and propagate through LOG (which
 *   has its own numerical hazards near `softmax → 0`); the analytical
 *   form `d_logits = d_loss · (labels − softmax(logits))` is one SUB +
 *   one MUL on top of a single SOFTMAX recompute, and dodges the
 *   `1/softmax(logits)` term entirely.
 * - **`reads_primal_indices`** — `{0, 1}`: the gradient body
 *   dereferences both `logits` (to recompute SOFTMAX) and `labels` (the
 *   `labels − softmax(logits)` SUB).
 *
 * # The math
 *
 * The recognizer matches the un-negated log-likelihood form:
 *
 * ```
 * probs = SOFTMAX(logits)         [..., V]
 * logp  = LOG(probs)              [..., V]
 * pw    = labels · logp           [..., V]
 * loss  = SUM(pw)                 []           (scalar)
 * ```
 *
 * Backward, given `d_loss = ∂L/∂loss` (a scalar):
 *
 * ```
 * # Working through the chain rule and using ∑_j labels[..., j] = 1
 * # (true for one-hot or any normalised label distribution),
 * # the j-summation collapses to:
 * #   ∂loss/∂logits[..., k] = labels[..., k] − softmax(logits)[..., k]
 * # (positive sign because the recognizer matches the un-negated form;
 * #  user code that wants NLL applies a NEG outside this envelope.)
 * d_logits = d_loss · (labels − softmax(logits))
 * d_labels = d_loss · log(softmax(logits))    (§0.4.292; was 0 pre-fix)
 * ```
 *
 * The scalar `d_loss` is broadcast to `logits`'s shape via an explicit
 * `BROADCAST` (the interpreter's BROADCAST handles scalar→rank-N per
 * `DxirInterpreter.kt:436`; this matches the FlashAttention coarsener's
 * convention).
 *
 * # Scope notes
 *
 * - **Sum must be scalar.** v1 requires `sum.type.rank == 0`. A
 *   per-row-sum form (where the SUM only reduces the class axis,
 *   producing one loss per row) would need a different upstream-broadcast
 *   shape; deferred until a caller actually needs it.
 *
 * - **Labels share logits' shape + dtype.** The `labels − softmax(logits)`
 *   SUB requires identical types. Sparse / index-form labels (where
 *   `labels` carries class indices into a reduced shape) are out of v1
 *   scope — they'd need a GATHER step in the gradient body. Coarsener
 *   declines on type mismatch.
 *
 * - **Recompute softmax inside the gradient.** Mirrors RmsNorm /
 *   FlashAttention's "recompute is cheaper than thread" choice. v2 can
 *   hoist `probs` into the COARSENED's payload as an additional primal
 *   return.
 *
 * - **Labels gradient.** §0.4.292 closed the prior shortcut (`d_labels = 0`).
 *   Tlaloc honours the math: `d_labels = d_loss · log(softmax(logits))`.
 *   Callers who treat labels as observed data (the common case) can stop
 *   gradient propagation themselves; the coarsener doesn't privilege that
 *   choice. Mirrors the same fix in RmsNorm + RoPE.
 */
internal fun coarsenCrossEntropy(
    match: RecognitionMatch.CrossEntropy,
): CoarsenedBundle? {
    // Ops layout from CrossEntropyRecognizer: [softmax, log, mul, sum].
    val opsList = match.ops
    if (opsList.size != 4) return null
    val softmax = opsList[0]
    val log = opsList[1]
    val mul = opsList[2]
    val sum = opsList[3]

    if (softmax.op != OpKind.SOFTMAX || log.op != OpKind.LOG ||
        mul.op != OpKind.MUL || sum.op != OpKind.SUM
    ) return null

    val logits = match.logits
    val labels = match.labels

    // v1: sum must be a scalar (the loss). Per-row losses (where SUM only
    // reduces the class axis) need a different upstream broadcast.
    if (sum.type.rank != 0) return null

    // v1: labels must share logits' type for the labels − softmax SUB.
    if (labels.type != logits.type) return null
    if (logits.type.dtype != F32 && logits.type.dtype != F64) return null

    // Preserve the user's MUL operand order in the primal so emit
    // fidelity matches the original graph.
    if (mul.operands.size != 2) return null
    val labelsIsLeftInMul = mul.operands[0].id == labels.id

    val absorbedOps = setOf(softmax.id, log.id, mul.id, sum.id)
    val outerOperands = listOf(logits, labels)

    val primalBody = buildCrossEntropyPrimal(
        logitsType = logits.type,
        labelsType = labels.type,
        softmaxType = softmax.type,
        softmaxAttrs = softmax.attrs,
        logType = log.type,
        mulType = mul.type,
        sumType = sum.type,
        sumAttrs = sum.attrs,
        labelsIsLeftInMul = labelsIsLeftInMul,
    )
    val gradientBody = buildCrossEntropyGradient(
        logitsType = logits.type,
        labelsType = labels.type,
        softmaxType = softmax.type,
        softmaxAttrs = softmax.attrs,
        sumType = sum.type,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = absorbedOps,
        anchorOpId = sum.id,
        outerOperands = outerOperands,
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

private fun buildCrossEntropyPrimal(
    logitsType: DxirType,
    labelsType: DxirType,
    softmaxType: DxirType,
    softmaxAttrs: Map<String, Any>,
    logType: DxirType,
    mulType: DxirType,
    sumType: DxirType,
    sumAttrs: Map<String, Any>,
    labelsIsLeftInMul: Boolean,
): DxirFunction = DxirBuilder.function("cross_entropy_primal") {
    val logits = param("logits", logitsType)
    val labels = param("labels", labelsType)
    val probs = op(OpKind.SOFTMAX, listOf(logits), softmaxType, attrs = softmaxAttrs)
    val logp = op(OpKind.LOG, listOf(probs), logType)
    val mulOperands = if (labelsIsLeftInMul) listOf(labels, logp) else listOf(logp, labels)
    val pw = op(OpKind.MUL, mulOperands, mulType)
    val loss = op(OpKind.SUM, listOf(pw), sumType, attrs = sumAttrs)
    listOf(loss)
}

private fun buildCrossEntropyGradient(
    logitsType: DxirType,
    labelsType: DxirType,
    softmaxType: DxirType,
    softmaxAttrs: Map<String, Any>,
    sumType: DxirType,
): DxirFunction = DxirBuilder.function("cross_entropy_grad") {
    val dLoss = param("d_loss", sumType)
    val logits = param("logits", logitsType)
    val labels = param("labels", labelsType)

    // Recompute softmax(logits).
    val probs = op(OpKind.SOFTMAX, listOf(logits), softmaxType, attrs = softmaxAttrs)

    // diff = labels − softmax(logits). Same shape as logits.
    val diff = op(OpKind.SUB, listOf(labels, probs), logitsType)

    // Broadcast scalar d_loss across logits' shape. Scalar→rank-N broadcast
    // uses an empty broadcast_dimensions (no input axes to map).
    val dLossBroadcast = op(
        OpKind.BROADCAST,
        listOf(dLoss),
        logitsType,
        attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
    )

    // d_logits = d_loss · (labels − softmax(logits)).
    val dLogits = op(OpKind.MUL, listOf(dLossBroadcast, diff), logitsType)

    // d_labels = d_loss · log(softmax(logits)). The forward computes
    // `loss = SUM(labels · log(softmax(logits)))`, so ∂loss/∂labels[i] is
    // log(softmax(logits))[i] (= logp). §0.4.292 closed the prior shortcut
    // (d_labels = 0) — emitting the structural zero produced bit-exact
    // disagreement with PyTorch's torch.autograd.grad on tests that
    // request labels' gradient. Tlaloc honours the math without
    // privileging a "labels are observed data" assumption that callers
    // can enforce themselves by stop_gradient'ing labels.
    val logp = op(OpKind.LOG, listOf(probs), logitsType)
    val dLabels = op(OpKind.MUL, listOf(dLossBroadcast, logp), labelsType)

    listOf(dLogits, dLabels)
}

private fun zeroValueFor(t: DxirType): Any = when (t.dtype) {
    F32 -> 0.0f
    F64 -> 0.0
    else -> error("CrossEntropyCoarsener: unsupported labels dtype ${t.dtype}")
}
