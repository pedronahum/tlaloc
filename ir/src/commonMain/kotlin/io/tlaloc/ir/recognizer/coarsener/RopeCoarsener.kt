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
 * Layer 4 §0.4.265 — RoPE analytical-backward coarsener.
 *
 * # What this is
 *
 * Consumes a [RecognitionMatch.Rope] (the canonical
 * `SIN/COS → MUL/MUL → ADD-or-SUB` rotation) and produces a
 * [CoarsenedBundle] carrying:
 *
 * - **`primal_body`** — a free-standing [DxirFunction] that recapitulates
 *   the matched rotation, preserving each MUL's operand order and the
 *   recombination's op kind (ADD vs SUB) from the user's chain. The
 *   COARSENED op's three operands map positionally: `(x_real, x_imag,
 *   theta)` — `theta` is the angle, `cos`/`sin` are recomputed inside.
 * - **`gradient_body`** — the analytical VJP. Signature
 *   `(dy, x_real, x_imag, theta) → (d_x_real, d_x_imag, d_theta_zero)`.
 *   The rotation is orthogonal, so the Jacobian is the transposed
 *   rotation: `d_x_real = dy · cos`, `d_x_imag = ±dy · sin`. The sign on
 *   the imag branch flips with the recombination (SUB ⇒ negate the imag
 *   gradient when cosMul is the left operand of the SUB; flip again when
 *   sinMul is the left operand). `theta` is treated as a fixed positional
 *   embedding — the gradient returns const-zero for it, mirroring the
 *   eps-treatment in [coarsenRmsNorm].
 * - **`reads_primal_indices`** — `{2}` (theta only). The gradient
 *   recomputes cos/sin from theta but never dereferences x_real or
 *   x_imag — they flow into the gradient solely through the upstream
 *   `dy`.
 *
 * # The math
 *
 * Forward (canonical positive-match form: `out = x_real · cos − x_imag · sin`):
 *
 * ```
 * cos = COS(theta)              [..., D]
 * sin = SIN(theta)              [..., D]
 * a   = x_real · cos            [..., D]
 * b   = x_imag · sin            [..., D]
 * out = a − b                   [..., D]      (or a + b for the ADD form)
 * ```
 *
 * Backward, given `dy = ∂L/∂out`:
 *
 * ```
 * # ADD form (out = x_real·cos + x_imag·sin):
 * d_x_real = dy · cos
 * d_x_imag = dy · sin
 *
 * # SUB form, cosMul-first (out = x_real·cos − x_imag·sin):
 * d_x_real = dy · cos
 * d_x_imag = −(dy · sin)
 *
 * # SUB form, sinMul-first (out = x_imag·sin − x_real·cos):
 * d_x_real = −(dy · cos)
 * d_x_imag = dy · sin
 * ```
 *
 * In all three cases `d_theta` is structurally zero — theta is a
 * positional argument, not a learnable parameter.
 *
 * # Scope notes
 *
 * - **Shared theta.** `SIN(theta_a)` and `COS(theta_b)` with
 *   `theta_a.id ≠ theta_b.id` declines (`return null`) — v1 only handles
 *   the canonical "single rotation table" form. A future variant could
 *   surface theta_sin and theta_cos as separate outer operands.
 *
 * - **Recompute cos/sin inside the gradient.** Mirrors RmsNorm's
 *   "recompute is cheaper than thread" choice. v2 can hoist them into
 *   the COARSENED's payload as additional primal returns to skip the
 *   second `COS`/`SIN`.
 *
 * - **Type uniformity assumed.** v1 expects x_real, x_imag, cos, sin,
 *   and the recombination's result to share a single dtype + shape
 *   (the only form the recognizer's positive-match test exercises). The
 *   coarsener does not enforce this — types are threaded through from
 *   the matched ops directly. Mismatched shapes would surface as a
 *   downstream type error at lowering, not here.
 *
 * - **Theta dtype.** `d_theta`'s const-zero only knows F32/F64. Other
 *   dtypes would need explicit zero-value constructors; deferred until a
 *   caller actually uses non-floating theta (unusual — positional
 *   embeddings are typically F32).
 */
internal fun coarsenRope(
    match: RecognitionMatch.Rope,
): CoarsenedBundle? {
    // Ops layout from RopeRecognizer: [sin, cos, sinMul, cosMul, recombine].
    val opsList = match.ops
    if (opsList.size != 5) return null
    val sinOp = opsList[0]
    val cosOp = opsList[1]
    val sinMul = opsList[2]
    val cosMul = opsList[3]
    val recombine = opsList[4]

    if (sinOp.op != OpKind.SIN || cosOp.op != OpKind.COS) return null
    if (sinMul.op != OpKind.MUL || cosMul.op != OpKind.MUL) return null
    if (recombine.op != OpKind.ADD && recombine.op != OpKind.SUB) return null

    // Both SIN and COS must share a single theta operand for v1.
    val thetaSin = sinOp.operands.firstOrNull() ?: return null
    val thetaCos = cosOp.operands.firstOrNull() ?: return null
    if (thetaSin.id != thetaCos.id) return null
    val theta = thetaSin

    // Identify the non-trig leaf operands of each MUL.
    if (sinMul.operands.size != 2 || cosMul.operands.size != 2) return null
    val xImag = sinMul.operands.firstOrNull { it.id != sinOp.id } ?: return null
    val xReal = cosMul.operands.firstOrNull { it.id != cosOp.id } ?: return null

    // Preserve the user's operand-order in the primal so emit fidelity
    // matches the original graph.
    val sinIsLeftInSinMul = sinMul.operands[0].id == sinOp.id
    val cosIsLeftInCosMul = cosMul.operands[0].id == cosOp.id

    if (recombine.operands.size != 2) return null
    val recombSet = recombine.operands.map { it.id }.toSet()
    if (recombSet != setOf(sinMul.id, cosMul.id)) return null
    val cosMulFirstInRecombine = recombine.operands[0].id == cosMul.id

    // Theta dtype constraint — d_theta is emitted as const-zero matching
    // theta's type; only F32/F64 are wired up.
    if (theta.type.dtype != F32 && theta.type.dtype != F64) return null

    val absorbedOps = setOf(sinOp.id, cosOp.id, sinMul.id, cosMul.id, recombine.id)
    val outerOperands = listOf(xReal, xImag, theta)

    val primalBody = buildRopePrimal(
        xRealType = xReal.type,
        xImagType = xImag.type,
        thetaType = theta.type,
        sinType = sinOp.type,
        cosType = cosOp.type,
        sinMulType = sinMul.type,
        cosMulType = cosMul.type,
        recombineType = recombine.type,
        sinIsLeftInSinMul = sinIsLeftInSinMul,
        cosIsLeftInCosMul = cosIsLeftInCosMul,
        cosMulFirstInRecombine = cosMulFirstInRecombine,
        recombineKind = recombine.op,
    )
    val gradientBody = buildRopeGradient(
        xRealType = xReal.type,
        xImagType = xImag.type,
        thetaType = theta.type,
        sinType = sinOp.type,
        cosType = cosOp.type,
        recombineType = recombine.type,
        cosMulFirstInRecombine = cosMulFirstInRecombine,
        recombineKind = recombine.op,
    )
    val reads = computeGradientReads(gradientBody, numUpstreamParams = 1)

    return CoarsenedBundle(
        absorbedOpIds = absorbedOps,
        anchorOpId = recombine.id,
        outerOperands = outerOperands,
        primalBody = primalBody,
        gradientBody = gradientBody,
        readsPrimalIndices = reads,
    )
}

private fun buildRopePrimal(
    xRealType: DxirType,
    xImagType: DxirType,
    thetaType: DxirType,
    sinType: DxirType,
    cosType: DxirType,
    sinMulType: DxirType,
    cosMulType: DxirType,
    recombineType: DxirType,
    sinIsLeftInSinMul: Boolean,
    cosIsLeftInCosMul: Boolean,
    cosMulFirstInRecombine: Boolean,
    recombineKind: OpKind,
): DxirFunction = DxirBuilder.function("rope_primal") {
    val xReal = param("x_real", xRealType)
    val xImag = param("x_imag", xImagType)
    val theta = param("theta", thetaType)
    val cosT = op(OpKind.COS, listOf(theta), cosType)
    val sinT = op(OpKind.SIN, listOf(theta), sinType)

    val sinMulOperands = if (sinIsLeftInSinMul) listOf(sinT, xImag) else listOf(xImag, sinT)
    val cosMulOperands = if (cosIsLeftInCosMul) listOf(cosT, xReal) else listOf(xReal, cosT)
    val sinMulPrimal = op(OpKind.MUL, sinMulOperands, sinMulType)
    val cosMulPrimal = op(OpKind.MUL, cosMulOperands, cosMulType)

    val recombineOperands = if (cosMulFirstInRecombine) {
        listOf(cosMulPrimal, sinMulPrimal)
    } else {
        listOf(sinMulPrimal, cosMulPrimal)
    }
    val out = op(recombineKind, recombineOperands, recombineType)
    listOf(out)
}

private fun buildRopeGradient(
    xRealType: DxirType,
    xImagType: DxirType,
    thetaType: DxirType,
    sinType: DxirType,
    cosType: DxirType,
    recombineType: DxirType,
    cosMulFirstInRecombine: Boolean,
    recombineKind: OpKind,
): DxirFunction = DxirBuilder.function("rope_grad") {
    val dy = param("dy", recombineType)
    @Suppress("UNUSED_VARIABLE") val xReal = param("x_real", xRealType)
    @Suppress("UNUSED_VARIABLE") val xImag = param("x_imag", xImagType)
    val theta = param("theta", thetaType)

    // Recompute cos and sin from theta.
    val cosR = op(OpKind.COS, listOf(theta), cosType)
    val sinR = op(OpKind.SIN, listOf(theta), sinType)

    val dyCos = op(OpKind.MUL, listOf(dy, cosR), xRealType)
    val dySin = op(OpKind.MUL, listOf(dy, sinR), xImagType)

    val (dXReal, dXImag) = when (recombineKind) {
        OpKind.ADD -> dyCos to dySin
        OpKind.SUB -> if (cosMulFirstInRecombine) {
            // out = cosMul - sinMul ⇒ d_x_imag picks up the negation.
            dyCos to op(OpKind.NEG, listOf(dySin), xImagType)
        } else {
            // out = sinMul - cosMul ⇒ d_x_real picks up the negation.
            op(OpKind.NEG, listOf(dyCos), xRealType) to dySin
        }
        else -> error("RopeCoarsener: unsupported recombine kind $recombineKind")
    }

    // Theta treated as a fixed positional embedding; gradient is structurally
    // zero matching theta's type. A learnable-theta VJP would compute
    // d_theta = dy · ∂out/∂theta, which depends on the rotation form;
    // deferred until a caller actually wants trainable positional embeddings.
    val dTheta = const(zeroValueFor(theta.type), theta.type)

    listOf(dXReal, dXImag, dTheta)
}

private fun zeroValueFor(t: DxirType): Any = when (t.dtype) {
    F32 -> 0.0f
    F64 -> 0.0
    else -> error("RopeCoarsener: unsupported theta dtype ${t.dtype}")
}
