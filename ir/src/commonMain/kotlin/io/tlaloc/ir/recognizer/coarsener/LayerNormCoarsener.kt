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
 * LayerNorm analytical-backward coarsener.
 *
 * Consumes a [RecognitionMatch.LayerNorm] (the canonical
 * `MEAN(x) → SUB(x, mean1) → MUL(centered²) → MEAN → [+eps] → SQRT → DIV`
 * shape) and produces a [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — `(x[, eps]) → y` recapitulating the matched
 *   forward op-by-op so downstream lowering can substitute a single
 *   `OpKind.COARSENED` for the chain.
 * - **`gradient_body`** — `(dy, x[, eps]) → (dx[, d_eps])`, the analytical
 *   VJP. No const-zero shortcuts: every operand gets its analytical gradient.
 *
 * # Math
 *
 * Forward (with `c = x − mean(x)`, `s = sqrt(mean(c²) + eps)`):
 *
 * ```
 * y = c / s
 * ```
 *
 * Backward derivation. Using `Σ_k c_k = 0` (the centring identity), the
 * variance gradient simplifies dramatically:
 *
 * ```
 * ∂var/∂x_j = (2/D) · c_j      (cross-terms cancel)
 * ∂s/∂x_j   = c_j / (D · s)
 * ∂y_i/∂x_j = (δ_ij − 1/D) / s − c_i · c_j / (D · s³)
 * ```
 *
 * Reducing the j-sum:
 *
 * ```
 * dx_j = (dy_j − mean(dy) − c_j · mean(dy · c) / s²) / s
 * ```
 *
 * For the eps gradient (eps treated as a fixed hyperparameter unless
 * present in [outerOperands]):
 *
 * ```
 * ∂y_i/∂eps  = −c_i / (2 s³)
 * d_eps      = −(1/(2 s³)) · sum(dy · c)
 * ```
 *
 * # Scope notes
 *
 * - **Keepdims-only.** Requires both MEAN ops to preserve rank
 *   (`mean.type.rank == x.type.rank`, with size-1 reduced axes). The
 *   coarsener declines (`return null`) on rank mismatch — same scope as
 *   [RmsNormCoarsener].
 * - **Eps as outer operand when present.** When the matched chain has
 *   `ADD(mean2, eps)`, eps becomes the second outer operand and the
 *   gradient body returns `(dx, d_eps)`. v1 only handles
 *   `eps.type == sqrtType` (eps broadcasts trivially across the reduced
 *   axis); arbitrary eps shapes need an additional reduction step.
 * - **No affine.** The optional `* gamma + beta` post-scale is not part
 *   of the v1 LayerNorm recognizer or coarsener.
 * - **Recompute everything from x in the gradient body.** Same trade-off
 *   as RmsNorm — recompute is cheaper than threading saved tensors when
 *   the forward intermediates are small.
 */
internal fun coarsenLayerNorm(
    match: RecognitionMatch.LayerNorm,
): CoarsenedBundle? {
    // Ops layout from LayerNormRecognizer:
    //   no eps:  [mean1, sub, sqMul, mean2, sqrt, finalDiv]              (6)
    //   eps:     [mean1, sub, sqMul, mean2, addOp, sqrt, finalDiv]       (7)
    val opsList = match.ops
    if (opsList.size !in 6..7) return null
    val mean1 = opsList[0]
    val sub = opsList[1]
    val sqMul = opsList[2]
    val mean2 = opsList[3]
    val hasEps = opsList.size == 7
    val addOp = if (hasEps) opsList[4] else null
    val sqrt = opsList[opsList.size - 2]
    val finalDiv = opsList.last()

    val xNode = match.input
    val xType = xNode.type

    // v1: keepdims-only on both reductions.
    if (mean1.type.rank != xType.rank) return null
    if (mean2.type.rank != xType.rank) return null

    // Eps node = whichever ADD operand isn't mean2.
    val epsNode = addOp?.operands?.firstOrNull { it.id != mean2.id }
    if (addOp != null && epsNode == null) return null
    val epsType = epsNode?.type
    if (epsType != null && epsType.dtype != F32 && epsType.dtype != F64) {
        // The d_eps `-0.5` constant only knows F32/F64.
        return null
    }

    val absorbedOps = buildSet {
        add(mean1.id); add(sub.id); add(sqMul.id); add(mean2.id)
        add(sqrt.id); add(finalDiv.id)
        if (addOp != null) add(addOp.id)
    }
    val outerOperands: List<DxirNode> =
        if (epsNode != null) listOf(xNode, epsNode) else listOf(xNode)

    val primalBody = buildLayerNormPrimal(
        xType = xType,
        meanType = mean1.type,
        meanAttrs = mean1.attrs,
        subType = sub.type,
        sqType = sqMul.type,
        addType = addOp?.type,
        addAttrs = addOp?.attrs ?: emptyMap(),
        sqrtType = sqrt.type,
        outputType = finalDiv.type,
        epsType = epsType,
    )
    val gradientBody = buildLayerNormGradient(
        xType = xType,
        meanType = mean1.type,
        meanAttrs = mean1.attrs,
        subType = sub.type,
        sqType = sqMul.type,
        addType = addOp?.type,
        addAttrs = addOp?.attrs ?: emptyMap(),
        sqrtType = sqrt.type,
        outputType = finalDiv.type,
        epsType = epsType,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = absorbedOps,
        anchorOpId = finalDiv.id,
        outerOperands = outerOperands,
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

private fun buildLayerNormPrimal(
    xType: DxirType,
    meanType: DxirType,
    meanAttrs: Map<String, Any>,
    subType: DxirType,
    sqType: DxirType,
    addType: DxirType?,
    addAttrs: Map<String, Any>,
    sqrtType: DxirType,
    outputType: DxirType,
    epsType: DxirType?,
): DxirFunction = DxirBuilder.function("layer_norm_primal") {
    val x = param("x", xType)
    val mean1 = op(OpKind.MEAN, listOf(x), meanType, attrs = meanAttrs)
    val sub = op(OpKind.SUB, listOf(x, mean1), subType)
    val sq = op(OpKind.MUL, listOf(sub, sub), sqType)
    val mean2 = op(OpKind.MEAN, listOf(sq), meanType, attrs = meanAttrs)
    val sqrtIn = if (epsType != null) {
        val eps = param("eps", epsType)
        op(OpKind.ADD, listOf(mean2, eps), addType!!, attrs = addAttrs)
    } else {
        mean2
    }
    val s = op(OpKind.SQRT, listOf(sqrtIn), sqrtType)
    val y = op(OpKind.DIV, listOf(sub, s), outputType)
    listOf(y)
}

private fun buildLayerNormGradient(
    xType: DxirType,
    meanType: DxirType,
    meanAttrs: Map<String, Any>,
    subType: DxirType,
    sqType: DxirType,
    addType: DxirType?,
    addAttrs: Map<String, Any>,
    sqrtType: DxirType,
    outputType: DxirType,
    epsType: DxirType?,
): DxirFunction = DxirBuilder.function("layer_norm_grad") {
    val dy = param("dy", outputType)
    val x = param("x", xType)
    val eps = if (epsType != null) param("eps", epsType) else null

    // Recompute c = x − mean(x), s = sqrt(mean(c²) [+ eps]).
    val mean1Recomputed = op(OpKind.MEAN, listOf(x), meanType, attrs = meanAttrs)
    val subRecomputed = op(OpKind.SUB, listOf(x, mean1Recomputed), subType)
    val sqRecomputed = op(OpKind.MUL, listOf(subRecomputed, subRecomputed), sqType)
    val mean2Recomputed = op(OpKind.MEAN, listOf(sqRecomputed), meanType, attrs = meanAttrs)
    val sqrtIn = if (eps != null) {
        op(OpKind.ADD, listOf(mean2Recomputed, eps), addType!!, attrs = addAttrs)
    } else {
        mean2Recomputed
    }
    val s = op(OpKind.SQRT, listOf(sqrtIn), sqrtType)

    // dx = (dy − mean(dy) − c · mean(dy · c) / s²) / s
    val sSquared = op(OpKind.MUL, listOf(s, s), sqrtType)
    val meanDy = op(OpKind.MEAN, listOf(dy), meanType, attrs = meanAttrs)
    val dyTimesC = op(OpKind.MUL, listOf(dy, subRecomputed), xType)
    val meanDyC = op(OpKind.MEAN, listOf(dyTimesC), meanType, attrs = meanAttrs)
    val ratio = op(OpKind.DIV, listOf(meanDyC, sSquared), sqrtType)
    val cScaled = op(OpKind.MUL, listOf(subRecomputed, ratio), xType)
    val dyMinusMean = op(OpKind.SUB, listOf(dy, meanDy), xType)
    val inner = op(OpKind.SUB, listOf(dyMinusMean, cScaled), xType)
    val dx = op(OpKind.DIV, listOf(inner, s), xType)

    if (eps != null) {
        // d_eps = -0.5 · sum(dy · c) / s³. Mirrors RmsNormCoarsener's
        // structure: SUM with the same reduction attrs as the forward
        // MEAN; -0.5 broadcast as a scalar across the reduced shape.
        require(eps.type == sqrtType) {
            "LayerNormCoarsener: eps gradient requires eps.type (${eps.type}) to match sqrt " +
                "output type ($sqrtType); arbitrary eps shapes need a follow-up reduction step"
        }
        val sCubed = op(OpKind.MUL, listOf(sSquared, s), sqrtType)
        val sumDyC = op(OpKind.SUM, listOf(dyTimesC), sqrtType, attrs = meanAttrs)
        val sumDyCOverSCubed = op(OpKind.DIV, listOf(sumDyC, sCubed), sqrtType)
        val scalarType = DxirType(eps.type.dtype, emptyList())
        val negHalfValue: Any = when (eps.type.dtype) {
            F32 -> -0.5f
            F64 -> -0.5
            else -> error("LayerNormCoarsener: unsupported eps dtype ${eps.type.dtype}")
        }
        val negHalf = const(negHalfValue, scalarType)
        val negHalfBroadcast = op(
            OpKind.BROADCAST,
            listOf(negHalf),
            sqrtType,
            attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
        )
        val dEps = op(OpKind.MUL, listOf(negHalfBroadcast, sumDyCOverSCubed), sqrtType)
        listOf(dx, dEps)
    } else {
        listOf(dx)
    }
}
