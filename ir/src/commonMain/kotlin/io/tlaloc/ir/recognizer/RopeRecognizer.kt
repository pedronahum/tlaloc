package io.tlaloc.ir.recognizer

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * Recognize rotary positional embedding (RoPE)
 * compound forms.
 *
 * # Match shape
 *
 * Canonical RoPE applies a rotation by `(cos θ, sin θ)` to a tensor of
 * pairs (real, imag). Simplified DXIR sequence:
 *
 * ```
 * cosθ = COS(theta)
 * sinθ = SIN(theta)
 * a    = MUL(x_real, cosθ)        // first half × cos
 * b    = MUL(x_imag, sinθ)        // second half × sin
 * out  = SUB(a, b)                // or ADD, depending on rotation direction
 * ```
 *
 * The recognizer anchors on `OpKind.SIN` (rare in non-RoPE graphs) and looks for a
 * paired `COS` consumed by a different `MUL` that combines with the
 * SIN-MUL via `ADD` or `SUB`.
 *
 * # What's NOT matched
 *
 * - Per-element bit-tricks (some implementations re-shape input to (real,
 *   imag) pairs explicitly via slicing + broadcasting). Only the
 *   simplest in-place form is matched.
 * - Multi-block frequency tables (different θ per dim). Only a
 *   single (cosθ, sinθ) pair is matched.
 *
 * # Near-miss diagnostics
 *
 * - SIN present but no paired COS in the function — likely a generic sine
 *   primitive, not RoPE.
 * - SIN + COS present but neither feeds a MUL — likely pre-tabulated
 *   embeddings used elsewhere.
 * - SIN-MUL and COS-MUL present but no ADD/SUB recombines them — chain
 *   exists but isn't wired up as a rotation.
 */
fun recognizeRope(
    fn: DxirFunction,
    diagnostics: MutableList<RecognitionDiagnostic>? = null,
): List<RecognitionMatch.Rope> {
    val out = mutableListOf<RecognitionMatch.Rope>()
    val seenSinIds = HashSet<Int>()

    val uses = buildUseListLocalRope(fn)

    for (node in fn.body) {
        if (node !is DxirOp || node.op != OpKind.SIN) continue
        if (node.id in seenSinIds) continue

        val match = tryMatchRope(node, fn, uses, diagnostics)
        if (match != null) {
            out += match
            // Mark all SIN ops in this match as seen so we don't double-fire.
            for (op in match.ops) if (op.op == OpKind.SIN) seenSinIds += op.id
        }
    }
    return out
}

private fun tryMatchRope(
    sin: DxirOp,
    fn: DxirFunction,
    uses: Map<Int, List<DxirOp>>,
    diagnostics: MutableList<RecognitionDiagnostic>?,
): RecognitionMatch.Rope? {
    // 1. Find a COS op in the same function (anywhere — they're typically
    //    co-located in user code, but we don't enforce ordering).
    val cos = fn.body.firstOrNull { it is DxirOp && it.op == OpKind.COS } as? DxirOp
    if (cos == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "Rope",
                reason = "SIN present but no COS in the same function — likely a standalone sine primitive",
                opId = sin.id,
            ),
        )
        return null
    }

    // 2. SIN must have a MUL consumer (the "x_imag * sinθ" step).
    val sinMul = uses[sin.id].orEmpty().firstOrNull { it.op == OpKind.MUL }
    if (sinMul == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "Rope",
                reason = "SIN has no MUL consumer; expected x_imag · sinθ step",
                opId = sin.id,
            ),
        )
        return null
    }

    // 3. COS must have a MUL consumer (the "x_real * cosθ" step), and
    //    that MUL must be distinct from the SIN-MUL.
    val cosMul = uses[cos.id].orEmpty().firstOrNull { it.op == OpKind.MUL && it.id != sinMul.id }
    if (cosMul == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "Rope",
                reason = "COS has no MUL consumer distinct from SIN's MUL; rotation pair not formed",
                opId = cos.id,
            ),
        )
        return null
    }

    // 4. Both MULs must feed a common ADD or SUB consumer (the rotation
    //    recombination).
    val sinMulConsumers = uses[sinMul.id].orEmpty().toSet()
    val cosMulConsumers = uses[cosMul.id].orEmpty().toSet()
    val recombine = sinMulConsumers.intersect(cosMulConsumers)
        .firstOrNull { it.op == OpKind.ADD || it.op == OpKind.SUB }
    if (recombine == null) {
        diagnostics?.add(
            RecognitionDiagnostic(
                pattern = "Rope",
                reason = "SIN-MUL and COS-MUL don't share an ADD/SUB consumer; chain exists but isn't wired up as rotation",
                opId = sin.id,
            ),
        )
        return null
    }

    // 5. Identify the input tensor: the operand of either MUL that isn't
    //    the SIN/COS result. Both sides should originate from the same
    //    upstream input (Q or K), but for v1 we just record one.
    val xInput = sinMul.operands.firstOrNull { it.id != sin.id }
        ?: return null

    return RecognitionMatch.Rope(
        ops = listOf(sin, cos, sinMul, cosMul, recombine),
        input = xInput,
        output = recombine,
    )
}

private fun buildUseListLocalRope(fn: DxirFunction): Map<Int, List<DxirOp>> {
    val uses = HashMap<Int, MutableList<DxirOp>>()
    for (node in fn.body) {
        if (node !is DxirOp) continue
        for (operand in node.operands) {
            uses.getOrPut(operand.id) { mutableListOf() } += node
        }
    }
    return uses
}
