package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Recognise grouped-query / multi-query attention shapes.
 *
 * Anchors on [OpKind.SOFTMAX] like [recognizeFlashAttention], then walks
 * the K and V operands of the surrounding matmuls back through a chain
 * of `{TRANSPOSE, RESHAPE}` ops until hitting a `BROADCAST`. Both K and V
 * must reach a BROADCAST whose expansion factor (output element count /
 * input element count) is the same and ≥ 2.
 *
 * The match's [RecognitionMatch.GroupedQueryAttention.ops] list contains
 * the FlashAttention triple plus all walked TRANSPOSE/RESHAPE/BROADCAST
 * ops, so [resolveLargestMatch] picks this over the bare
 * FlashAttention match every time (as it picks TransformerMLP over SwiGLU).
 *
 * # Near-miss diagnostics
 *
 * - K's chain has no BROADCAST → standard MHA, not GQA.
 * - V's chain has no BROADCAST → only K is expanded (incomplete GQA).
 * - K and V ratios differ → suspicious; reject rather than guess.
 * - BROADCAST has ratio == 1 → degenerate (size identity, not an expansion).
 */
fun recognizeGroupedQueryAttention(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.GroupedQueryAttention> {
    val uses = buildUseListLocal(fn)
    val out = mutableListOf<RecognitionMatch.GroupedQueryAttention>()
    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.SOFTMAX) continue
        val match = tryMatchGqa(node, uses, diagnostics)
        if (match != null) out += match
    }
    return out
}

private data class ExpansionTrace(
    val rawInput: DxirNode,
    val broadcast: DxirOp,
    /** Every op walked through, including the BROADCAST and any RESHAPEs/TRANSPOSEs. */
    val ops: List<DxirOp>,
    val groupRatio: Int,
)

/**
 * Walk back from [start] through `{TRANSPOSE, RESHAPE}` ops until hitting
 * a [OpKind.BROADCAST] whose expansion factor is ≥ 2. If found, continue
 * walking back through any leading `RESHAPE` ops (the GQA canonical form
 * has a `RESHAPE` inside the BROADCAST that inserts the size-1 axis) and
 * report the resulting raw leaf input. Returns `null` if no expansion
 * BROADCAST is found.
 */
private fun walkBackToExpansion(start: DxirNode): ExpansionTrace? {
    val traversed = mutableListOf<DxirOp>()
    var cursor: DxirNode = start
    while (cursor is DxirOp && (cursor.op == OpKind.TRANSPOSE || cursor.op == OpKind.RESHAPE)) {
        traversed.add(cursor)
        cursor = cursor.operands.singleOrNull() ?: return null
    }
    if (cursor !is DxirOp || cursor.op != OpKind.BROADCAST) return null
    val broadcast = cursor
    traversed.add(broadcast)

    val bcInput = broadcast.operands.singleOrNull() ?: return null
    val outProduct = broadcast.type.dims.fold(1L) { acc, d -> acc * d }
    val inProduct = bcInput.type.dims.fold(1L) { acc, d -> acc * d }
    if (inProduct == 0L || outProduct % inProduct != 0L) return null
    val ratio = (outProduct / inProduct).toInt()
    if (ratio < 2) return null

    var raw: DxirNode = bcInput
    while (raw is DxirOp && raw.op == OpKind.RESHAPE) {
        traversed.add(raw)
        raw = raw.operands.singleOrNull() ?: return null
    }

    return ExpansionTrace(rawInput = raw, broadcast = broadcast, ops = traversed, groupRatio = ratio)
}

private fun tryMatchGqa(
    softmax: DxirOp,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.GroupedQueryAttention? {
    // 1. SOFTMAX operand should be the QK matmul (same prerequisite as FlashAttention).
    val softmaxOperand = softmax.operands.singleOrNull()
    if (softmaxOperand !is DxirOp || softmaxOperand.op != OpKind.MATMUL) return null
    val qkMatmul = softmaxOperand
    val qkOperands = qkMatmul.operands
    if (qkOperands.size != 2) return null

    // 2. SOFTMAX must have a MATMUL consumer (the PV matmul).
    val pvMatmul = uses[softmax.id].orEmpty().firstOrNull { it.op == OpKind.MATMUL } ?: return null
    val pvOperands = pvMatmul.operands
    if (pvOperands.size != 2) return null
    val pvLhsIsSoftmax = pvOperands[0].id == softmax.id
    val pvRhsIsSoftmax = pvOperands[1].id == softmax.id
    if (!pvLhsIsSoftmax && !pvRhsIsSoftmax) return null

    // 3. The K operand of the QK matmul should walk back to a BROADCAST.
    //    The convention in this codebase (verified by §0.4.286) is that
    //    `MATMUL(Q, K^T)` puts K at index 1; we still try both for
    //    robustness against alternative QK layouts.
    val kCandidate0 = qkOperands[1]
    val kCandidate1 = qkOperands[0]
    val qInput: DxirNode
    val kTrace: ExpansionTrace
    val w0 = walkBackToExpansion(kCandidate0)
    val w1 = walkBackToExpansion(kCandidate1)
    when {
        w0 != null -> {
            kTrace = w0
            qInput = kCandidate1
        }
        w1 != null -> {
            kTrace = w1
            qInput = kCandidate0
        }
        else -> {
            diagnostics?.add(
                RecognitionDiagnostic(
                    pattern = "GroupedQueryAttention",
                    reason = "neither QK matmul operand walks back to a BROADCAST — looks like standard MHA",
                    opId = qkMatmul.id,
                ),
            )
            return null
        }
    }

    // 4. The V operand of the PV matmul should walk back to a BROADCAST.
    val vOperand = if (pvLhsIsSoftmax) pvOperands[1] else pvOperands[0]
    val vTrace = walkBackToExpansion(vOperand)
    if (vTrace == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "GroupedQueryAttention",
                reason = "V operand has no BROADCAST in its lineage; K is expanded but V isn't — incomplete GQA",
                opId = pvMatmul.id,
            ),
        )
        return null
    }

    // 5. K and V expansion ratios must match — otherwise the head
    //    structure is inconsistent and this isn't a sound GQA shape.
    if (kTrace.groupRatio != vTrace.groupRatio) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "GroupedQueryAttention",
                reason = "K group ratio ${kTrace.groupRatio} != V group ratio ${vTrace.groupRatio} — " +
                    "inconsistent head structure",
                opId = pvMatmul.id,
            ),
        )
        return null
    }

    // 6. Reject the self-contraction degenerate.
    if (qInput.id == kTrace.rawInput.id && kTrace.rawInput.id == vTrace.rawInput.id) {
        return null
    }

    val matchedOps = buildList {
        add(qkMatmul)
        add(softmax)
        add(pvMatmul)
        addAll(kTrace.ops)
        addAll(vTrace.ops)
    }.distinctBy { it.id }

    return RecognitionMatch.GroupedQueryAttention(
        ops = matchedOps,
        qkMatmul = qkMatmul,
        softmax = softmax,
        pvMatmul = pvMatmul,
        qInput = qInput,
        kRawInput = kTrace.rawInput,
        vRawInput = vTrace.rawInput,
        kBroadcast = kTrace.broadcast,
        vBroadcast = vTrace.broadcast,
        groupRatio = kTrace.groupRatio,
        scoreType = qkMatmul.types.first(),
        outputType = pvMatmul.types.first(),
    )
}

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
