package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Recognise the canonical LayerNorm shape.
 *
 * # Match shape
 *
 * Anchors on [OpKind.SQRT] and walks back through MEAN(centered²) /
 * SUB(x, MEAN(x)) / MEAN(x) to the input parameter, then forward through
 * the divisive step.
 *
 * ```
 * mean1 = MEAN(x)                  keepdims, [..., 1]
 * sub   = SUB(x, mean1)            centered, [..., D] (broadcast)
 * sq    = MUL(sub, sub)            squared deviations
 * mean2 = MEAN(sq)                 variance, keepdims
 * [add  = ADD(mean2, eps)]         optional stabiliser
 * std   = SQRT(mean2 or add)
 * out   = DIV(sub, std)            normalised
 * ```
 *
 * # Disambiguation from [RecognitionMatch.RmsNorm]
 *
 * [RmsNormRecognizer] anchors on RSQRT, this one on SQRT — different
 * ops, so no overlap. The squared step in LayerNorm is
 * `MUL(centered, centered)` rather than `MUL(x, x)`, which is the
 * disambiguator if a future RSQRT-form variant is added (the resolver
 * would then prefer LayerNorm because LayerNorm's op set strictly
 * contains RmsNorm's plus the {mean1, sub} centring pair).
 *
 * # Near-miss diagnostics
 *
 * - SQRT operand isn't MEAN or `ADD(MEAN, eps)` — likely a different norm.
 * - mean2 operand isn't `MUL(centered, centered)` — likely a plain reduction.
 * - The squared MUL operands aren't the *same* tensor — pairwise product, not square.
 * - The squared tensor isn't a SUB op — looks like RmsNorm-shaped, not LayerNorm.
 * - SUB right operand isn't a MEAN — centring source unclear.
 * - MEAN(x) input doesn't match SUB's left operand — centring uses a different mean.
 * - SQRT result isn't consumed by a `DIV(centered, sqrt)` — the chain isn't wired up.
 */
fun recognizeLayerNorm(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.LayerNorm> {
    val uses = buildUseListLocal(fn)
    val out = mutableListOf<RecognitionMatch.LayerNorm>()
    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.SQRT) continue
        val match = tryMatchLayerNorm(node, uses, diagnostics)
        if (match != null) out += match
    }
    return out
}

private fun tryMatchLayerNorm(
    sqrt: DxirOp,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.LayerNorm? {
    // 1. SQRT operand should be MEAN, or ADD(MEAN, scalar-eps).
    val sqrtOperand = sqrt.operands.singleOrNull()
    if (sqrtOperand !is DxirOp) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "LayerNorm",
                reason = "SQRT operand is not an op (param/const); expected MEAN(centered²) for LayerNorm",
                opId = sqrt.id,
            ),
        )
        return null
    }
    val mean2: DxirOp = when (sqrtOperand.op) {
        OpKind.MEAN -> sqrtOperand
        OpKind.ADD -> {
            val nestedMean = sqrtOperand.operands.firstOrNull { it is DxirOp && it.op == OpKind.MEAN } as? DxirOp
            if (nestedMean == null) {
                diagnostics?.add(
                    RecognitionDiagnostic(
                        pattern = "LayerNorm",
                        reason = "SQRT operand is ADD but no MEAN among its operands — likely a different norm form",
                        opId = sqrt.id,
                    ),
                )
                return null
            }
            nestedMean
        }
        else -> {
            diagnostics?.add(
                RecognitionDiagnostic(
                    pattern = "LayerNorm",
                    reason = "SQRT operand is ${sqrtOperand.op}; expected MEAN or ADD(MEAN, eps)",
                    opId = sqrt.id,
                ),
            )
            return null
        }
    }
    val addOp: DxirOp? = if (sqrtOperand.op == OpKind.ADD) sqrtOperand else null

    // 2. mean2 operand should be MUL(centered, centered).
    val mean2Operand = mean2.operands.singleOrNull()
    if (mean2Operand !is DxirOp || mean2Operand.op != OpKind.MUL) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "LayerNorm",
                reason = "MEAN-of-variance operand is ${(mean2Operand as? DxirOp)?.op ?: "non-op"}; " +
                    "expected MUL(centered, centered) (the squared-deviation step)",
                opId = mean2.id,
            ),
        )
        return null
    }
    val sqMul: DxirOp = mean2Operand
    val sqA = sqMul.operands.getOrNull(0)
    val sqB = sqMul.operands.getOrNull(1)
    if (sqA == null || sqB == null || sqA.id != sqB.id) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "LayerNorm",
                reason = "Squared-deviation MUL operands aren't the same tensor (id=${sqA?.id} vs id=${sqB?.id}); " +
                    "expected centered · centered",
                opId = sqMul.id,
            ),
        )
        return null
    }

    // 3. The "centered" tensor should be SUB(x, mean1), where mean1 = MEAN(x).
    if (sqA !is DxirOp || sqA.op != OpKind.SUB) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "LayerNorm",
                reason = "Squared tensor isn't SUB(x, mean(x)); op is ${(sqA as? DxirOp)?.op ?: "non-op"} " +
                    "— shape looks like RmsNorm, not LayerNorm",
                opId = sqMul.id,
            ),
        )
        return null
    }
    val sub: DxirOp = sqA
    val xCandidate = sub.operands.getOrNull(0)
    val meanLhs = sub.operands.getOrNull(1)
    if (xCandidate == null || meanLhs !is DxirOp || meanLhs.op != OpKind.MEAN) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "LayerNorm",
                reason = "SUB right operand isn't a MEAN op; got ${(meanLhs as? DxirOp)?.op ?: "non-op"} " +
                    "— centring source unclear",
                opId = sub.id,
            ),
        )
        return null
    }
    val mean1: DxirOp = meanLhs
    val mean1Operand = mean1.operands.singleOrNull()
    if (mean1Operand == null || mean1Operand.id != xCandidate.id) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "LayerNorm",
                reason = "MEAN(x) input (id=${mean1Operand?.id}) doesn't match the SUB's left operand " +
                    "(id=${xCandidate.id}); centring uses a different mean source than the input",
                opId = mean1.id,
            ),
        )
        return null
    }
    val xInput: DxirNode = xCandidate

    // 4. SQRT result should be consumed by DIV(centered, sqrt).
    val sqrtConsumers = uses[sqrt.id].orEmpty()
    val finalDiv = sqrtConsumers.firstOrNull { consumer ->
        consumer.op == OpKind.DIV &&
            consumer.operands.size == 2 &&
            consumer.operands[0].id == sub.id &&
            consumer.operands[1].id == sqrt.id
    }
    if (finalDiv == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "LayerNorm",
                reason = "SQRT has no DIV(centered, sqrt) consumer (id=${sub.id} as numerator); " +
                    "chain exists but isn't wired up as LayerNorm",
                opId = sqrt.id,
            ),
        )
        return null
    }

    val matchedOps = buildList {
        add(mean1)
        add(sub)
        add(sqMul)
        add(mean2)
        if (addOp != null) add(addOp)
        add(sqrt)
        add(finalDiv)
    }
    return RecognitionMatch.LayerNorm(
        ops = matchedOps,
        input = xInput,
        output = finalDiv,
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
