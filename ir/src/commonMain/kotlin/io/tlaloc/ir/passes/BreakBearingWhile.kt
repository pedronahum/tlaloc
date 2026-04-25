package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * §0.4.123 — D.3i Phase 1. Structural detector for the LAND-composed break-bearing
 * WHILE pattern. The FIR lowering hoists `while (cond) { ... if (break_cond) break }`
 * into a `WHILE` whose cond region is `LAND(cond, NOT(break_cond))` (per [OpKind.LAND]'s
 * source comment). This detector recognises that shape so future phases can apply
 * paper §6's closed-form closure to break-bearing loops without re-parsing the
 * structure each time.
 *
 * Phase 1 ships only the detector — no rewrite, no gradient handling. Subsequent
 * phases will plug into [PhiCalculus.coarsenFunction]'s WHILE path or
 * [DxirReverseTransform]'s reverse walk to apply the closure rule.
 *
 * Returned [Pattern] carries:
 *  - [origCond]: the first operand of the top-level LAND — the loop's "natural"
 *    termination predicate (e.g., `STEP(SUB(n, counter))`).
 *  - [breakCond]: the operand of the NOT — the predicate that, when true, causes
 *    the loop to break (e.g., `STEP(SUB(threshold, accumulator))`).
 *  - [whileOp]: the original [OpKind.WHILE] op for downstream consumers.
 *
 * @see OpKind.LAND for the FIR-side hoist that produces this shape.
 * @see PhiCalculus.detectAffineRecurrence for the analogous pattern detector
 *      applied to the C6 affine-recurrence form.
 */
object BreakBearingWhile {

    /**
     * The recognised structural pattern. All three fields are non-null on a successful
     * match. Future phases extend this with the closed-form coefficients (counter
     * mapping, trip-count bound, etc.).
     */
    data class Pattern(
        val whileOp: DxirOp,
        val origCond: DxirNode,
        val breakCond: DxirNode,
    )

    /**
     * Recognise the LAND-composed break-bearing shape on [op]. Returns null if [op]
     * isn't a [OpKind.WHILE], if its cond region's terminator isn't the canonical
     * `LAND(origCond, NOT(breakCond))` shape, or if the structural invariants of
     * single-block / single-yield cond region fail.
     *
     * The match is purely structural: this detector doesn't validate that
     * [Pattern.origCond] is well-formed (e.g., `STEP(SUB(n, counter))` shape) or
     * that [Pattern.breakCond] is a clean predicate. Those tighter checks belong
     * to phases that consume the pattern; this detector's contract is "is the
     * top-level cond an LAND with a NOT on its second slot?".
     */
    fun detect(op: DxirOp): Pattern? {
        if (op.op != OpKind.WHILE) return null
        if (op.regions.size != 2) return null
        val condRegion = op.regions[0]
        if (condRegion.blocks.size != 1) return null
        val condBlock = condRegion.blocks.single()
        if (condBlock.terminator.size != 1) return null
        val terminator = condBlock.terminator.single() as? DxirOp ?: return null
        if (terminator.op != OpKind.LAND) return null
        if (terminator.operands.size != 2) return null
        val notNode = terminator.operands[1] as? DxirOp ?: return null
        if (notNode.op != OpKind.NOT) return null
        if (notNode.operands.size != 1) return null
        return Pattern(
            whileOp = op,
            origCond = terminator.operands[0],
            breakCond = notNode.operands[0],
        )
    }
}
