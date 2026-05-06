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
 * Layer 4 §0.4.264 — RMS-norm analytical-backward coarsener.
 *
 * # What this is
 *
 * Consumes a [RecognitionMatch.RmsNorm] (the canonical
 * `MUL(x, x) → MEAN → [ADD eps] → RSQRT → MUL(x, rsq)` shape) and
 * produces a [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — a free-standing [DxirFunction] that recapitulates
 *   the matched forward, preserving each op's type and attrs from the
 *   user's original chain. The COARSENED op's operands map positionally:
 *   `[x]` (no eps) or `[x, eps]` (with eps).
 * - **`gradient_body`** — the analytical VJP. Signature
 *   `(dy, x) → (dx)` (no eps) or `(dy, x, eps) → (dx, deps_zero)`
 *   (with eps).
 * - **`reads_primal_indices`** — computed via [computeGradientReads];
 *   `{0}` for the no-eps case, `{0, 1}` with eps if the gradient body
 *   dereferences eps (it does, while recomputing `r`).
 *
 * # The math
 *
 * Forward (`r = (mean(x²) + eps)^(-1/2)`, broadcast across the reduced
 * axis since the recognizer matches the keepdims-style chain — mean's
 * output has the same rank as `x`, with size-1 reduced axes):
 *
 * ```
 * sq    = x · x                           [..., D]
 * m     = mean(sq)                        [..., 1]   (last axis, keepdims)
 * m_eps = m + eps                         [..., 1]   (eps optional)
 * r     = rsqrt(m_or_meps)                [..., 1]
 * y     = x · r                           [..., D]
 * ```
 *
 * Backward (chain-rule expansion through square / mean / rsqrt / multiply):
 *
 * ```
 * #   ∂y/∂x_d = r + x_d · ∂r/∂x_d
 * #   ∂r/∂x_d = (∂r/∂m) · (∂m/∂x_d) = (-r³/2) · (2x_d / D) = -r³ x_d / D
 * # ⇒ dx_d   = dy_d · r − x_d · r³ · (1/D) Σ_e (dy_e · x_e)
 * #          = r · (dy_d − x_d · r² · mean(dy · x))
 * ```
 *
 * Emitted as:
 *
 * ```
 * r²        = r · r                       [..., 1]   (recomputed)
 * dy_x_mean = mean(dy · x)                [..., 1]
 * inner     = dy − x · r² · dy_x_mean     [..., D]
 * dx        = r · inner                   [..., D]
 * ```
 *
 * The size-1 reduced-axis broadcast in the user's original chain handles
 * the rank pun in `x · r²` and `x · r² · dy_x_mean` — no explicit
 * BROADCAST step is needed here.
 *
 * # Scope notes
 *
 * - **Keepdims-only.** Requires the matched MEAN to preserve rank
 *   (`mean.type.rank == x.type.rank`, with size-1 reduced axes). This is
 *   the form the §0.4.251 recognizer's positive-match test produces and
 *   what real LLM code (Llama, Mistral) emits. A no-keepdims variant
 *   would need an explicit BROADCAST in the gradient body, mirroring
 *   FlashAttentionCoarsener's pattern; out of v1 scope. The coarsener
 *   declines (`return null`) on rank mismatch.
 *
 * - **Eps surfaced as outer operand when present.** When the matched
 *   chain has `ADD(MEAN, eps)`, eps becomes the second outer operand and
 *   the gradient body returns `(dx, zero_deps)` — eps is treated as a
 *   fixed hyperparameter. A learnable-eps VJP would require reducing
 *   the per-position contribution across all batch axes, which depends
 *   on eps's actual rank (typically scalar `[]`, but could be `[..., 1]`
 *   per-row); deferred until a downstream caller actually needs it.
 *
 * - **Recompute r inside the gradient.** Mirrors FlashAttention's
 *   "recompute is cheaper than thread" choice; v2 can hoist `r` (and
 *   intermediate `m`) into the COARSENED's payload as additional primal
 *   returns to skip the second `MEAN+RSQRT`.
 */
internal fun coarsenRmsNorm(
    match: RecognitionMatch.RmsNorm,
): CoarsenedBundle? {
    // Ops layout from RmsNormRecognizer:
    //   no eps:  [sqMul, mean, rsqrt, finalMul]            (4 entries)
    //   eps:     [sqMul, mean, addOp, rsqrt, finalMul]     (5 entries)
    val opsList = match.ops
    if (opsList.size !in 4..5) return null
    val sqMul = opsList[0]
    val meanOp = opsList[1]
    val hasEps = opsList.size == 5
    val addOp = if (hasEps) opsList[2] else null
    val rsqrt = opsList[opsList.size - 2]
    val finalMul = opsList.last()

    val xNode = match.input
    val xType = xNode.type

    // v1: keepdims-only. mean.type.rank == x.type.rank means the reduced
    // axis is preserved as size 1. No-keepdims would need an explicit
    // BROADCAST in the gradient body — out of v1 scope.
    if (meanOp.type.rank != xType.rank) return null

    // Eps node = whichever ADD operand isn't the MEAN.
    val epsNode = addOp?.operands?.firstOrNull { it.id != meanOp.id }
    if (addOp != null && epsNode == null) return null
    val epsType = epsNode?.type
    if (epsType != null && epsType.dtype != F32 && epsType.dtype != F64) {
        // The d_eps `-0.5` constant only knows F32/F64; other dtypes (I32/I64/Bool)
        // wouldn't make sense as an RmsNorm epsilon anyway.
        return null
    }

    val absorbedOps = buildSet {
        add(sqMul.id); add(meanOp.id); add(rsqrt.id); add(finalMul.id)
        if (addOp != null) add(addOp.id)
    }
    val outerOperands: List<DxirNode> =
        if (epsNode != null) listOf(xNode, epsNode) else listOf(xNode)

    val primalBody = buildRmsNormPrimal(
        xType = xType,
        sqType = sqMul.type,
        meanType = meanOp.type,
        meanAttrs = meanOp.attrs,
        addType = addOp?.type,
        addAttrs = addOp?.attrs ?: emptyMap(),
        rsqrtType = rsqrt.type,
        outputType = finalMul.type,
        epsType = epsType,
    )
    val gradientBody = buildRmsNormGradient(
        xType = xType,
        sqType = sqMul.type,
        meanType = meanOp.type,
        meanAttrs = meanOp.attrs,
        addType = addOp?.type,
        addAttrs = addOp?.attrs ?: emptyMap(),
        rsqrtType = rsqrt.type,
        outputType = finalMul.type,
        epsType = epsType,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = absorbedOps,
        anchorOpId = finalMul.id,
        outerOperands = outerOperands,
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

private fun buildRmsNormPrimal(
    xType: DxirType,
    sqType: DxirType,
    meanType: DxirType,
    meanAttrs: Map<String, Any>,
    addType: DxirType?,
    addAttrs: Map<String, Any>,
    rsqrtType: DxirType,
    outputType: DxirType,
    epsType: DxirType?,
): DxirFunction = DxirBuilder.function("rms_norm_primal") {
    val x = param("x", xType)
    val sq = op(OpKind.MUL, listOf(x, x), sqType)
    val mean = op(OpKind.MEAN, listOf(sq), meanType, attrs = meanAttrs)
    val rsqrtIn = if (epsType != null) {
        val eps = param("eps", epsType)
        op(OpKind.ADD, listOf(mean, eps), addType!!, attrs = addAttrs)
    } else mean
    val r = op(OpKind.RSQRT, listOf(rsqrtIn), rsqrtType)
    val y = op(OpKind.MUL, listOf(x, r), outputType)
    listOf(y)
}

private fun buildRmsNormGradient(
    xType: DxirType,
    sqType: DxirType,
    meanType: DxirType,
    meanAttrs: Map<String, Any>,
    addType: DxirType?,
    addAttrs: Map<String, Any>,
    rsqrtType: DxirType,
    outputType: DxirType,
    epsType: DxirType?,
): DxirFunction = DxirBuilder.function("rms_norm_grad") {
    val dy = param("dy", outputType)
    val x = param("x", xType)
    val eps = if (epsType != null) param("eps", epsType) else null

    // Recompute r = rsqrt(mean(x²) [+ eps]).
    val sqRecomputed = op(OpKind.MUL, listOf(x, x), sqType)
    val meanRecomputed = op(OpKind.MEAN, listOf(sqRecomputed), meanType, attrs = meanAttrs)
    val rsqrtIn = if (eps != null) {
        op(OpKind.ADD, listOf(meanRecomputed, eps), addType!!, attrs = addAttrs)
    } else meanRecomputed
    val r = op(OpKind.RSQRT, listOf(rsqrtIn), rsqrtType)

    // dx = r · (dy − x · r² · mean(dy · x))
    val rSquared = op(OpKind.MUL, listOf(r, r), rsqrtType)
    val dyx = op(OpKind.MUL, listOf(dy, x), xType)
    val dyxMean = op(OpKind.MEAN, listOf(dyx), meanType, attrs = meanAttrs)
    val xR2 = op(OpKind.MUL, listOf(x, rSquared), xType)
    val xR2DyxMean = op(OpKind.MUL, listOf(xR2, dyxMean), xType)
    val dyMinusInner = op(OpKind.SUB, listOf(dy, xR2DyxMean), xType)
    val dx = op(OpKind.MUL, listOf(dyMinusInner, r), xType)

    if (eps != null) {
        // d_eps = -0.5 · r³ · SUM(dy · x, last-axis, keep-dims). §0.4.292
        // closed the prior shortcut (d_eps = const(0)) — emitting structural
        // zero produced disagreement with PyTorch's torch.autograd.grad on any
        // gradient consumer that asks for d_eps. Derivation:
        //   y[k,j] = x[k,j] · r[k] where r[k] = rsqrt(mean(x²)[k] + eps[k])
        //   ∂r[k]/∂eps[k] = -0.5 · r[k]³
        //   ∂loss/∂eps[k] = sum_j dy[k,j] · x[k,j] · (-0.5 · r[k]³).
        // v1 only handles eps.type == rsqrtType (eps broadcasts trivially across
        // the dim axis); arbitrary eps shapes need an additional SUM-reduce
        // and are deferred until a caller hits one.
        require(eps.type == rsqrtType) {
            "RmsNormCoarsener: eps gradient requires eps.type (${eps.type}) to match rsqrt " +
                "output type ($rsqrtType); arbitrary eps shapes need a follow-up reduction step"
        }
        val rCubed = op(OpKind.MUL, listOf(rSquared, r), rsqrtType)
        // SUM(dy · x) with the same reduction_dims as the forward MEAN —
        // the dimension axes the forward MEAN collapses are the axes whose
        // contribution to eps is summed in reverse.
        val sumDyx = op(OpKind.SUM, listOf(dyx), rsqrtType, attrs = meanAttrs)
        val scalarType = DxirType(eps.type.dtype, emptyList())
        val negHalfValue: Any = when (eps.type.dtype) {
            F32 -> -0.5f
            F64 -> -0.5
            else -> error("RmsNormCoarsener: unsupported eps dtype ${eps.type.dtype}")
        }
        val negHalf = const(negHalfValue, scalarType)
        val negHalfBroadcast = op(
            OpKind.BROADCAST,
            listOf(negHalf),
            rsqrtType,
            attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
        )
        val rCubedSum = op(OpKind.MUL, listOf(rCubed, sumDyx), rsqrtType)
        val dEps = op(OpKind.MUL, listOf(negHalfBroadcast, rCubedSum), rsqrtType)
        listOf(dx, dEps)
    } else {
        listOf(dx)
    }
}
