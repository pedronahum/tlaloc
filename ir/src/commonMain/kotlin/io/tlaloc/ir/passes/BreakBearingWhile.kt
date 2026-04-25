package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirBlockArg
import io.tlaloc.ir.DxirCall
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
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
     * The recognised structural pattern. The first three fields ([whileOp], [origCond],
     * [breakCond]) are always non-null on a successful match. The trip-count fields
     * ([counterArgIdx], [tripCountConst], [tripCountParam]) are populated when
     * [origCond] matches the canonical `STEP(SUB(n, args[counterArgIdx]))` C5/C6 shape;
     * otherwise they're null and consumers fall back to other detection paths.
     *
     * Trip count is mutually exclusive: at most one of [tripCountConst] / [tripCountParam]
     * is non-null when [counterArgIdx] is set. A concrete-int bound flows through
     * [tripCountConst]; a loop-invariant scalar param bound flows through [tripCountParam].
     */
    data class Pattern(
        val whileOp: DxirOp,
        val origCond: DxirNode,
        val breakCond: DxirNode,
        val counterArgIdx: Int? = null,
        val tripCountConst: Int? = null,
        val tripCountParam: DxirParam? = null,
    )

    /**
     * §0.4.126 — D.3i Phase 3a. Dependency classification of the break predicate.
     * Future closure phases dispatch on this to pick the right break-iteration
     * computation strategy:
     *  - [Constant] — folded at compile time; the closure is fully resolved (skip
     *    the loop entirely or treat as a vanilla [OpKind.WHILE]).
     *  - [LoopInvariant] — predicate is constant across iterations but only known
     *    at runtime; the closure can evaluate it once before the loop and branch.
     *  - [CounterOnly] — depends only on the counter block-arg (and loop-invariant
     *    operands); break iteration is structurally derivable (Phase 3b: Symja).
     *  - [CarriedDependent] — depends on at least one non-counter carried block-arg;
     *    the break must stay per-iteration (Phase 3c: runtime fallback).
     */
    sealed class BreakCondClass {
        /**
         * The predicate folded down to a Boolean constant. [breakIteration] is `0`
         * when the loop breaks unconditionally and `null` when it never breaks.
         */
        data class Constant(val alwaysBreaks: Boolean) : BreakCondClass() {
            val breakIteration: Int? get() = if (alwaysBreaks) 0 else null
        }

        /**
         * Predicate depends only on values stable across the loop body (function
         * params, region-external constants, ops outside the [OpKind.WHILE]). The
         * runtime arm can evaluate it once before the loop runs.
         */
        data object LoopInvariant : BreakCondClass()

        /**
         * Predicate depends only on the counter block-arg (plus loop-invariant
         * values). The closure can solve symbolically for the smallest `k` where
         * the predicate becomes true.
         */
        data object CounterOnly : BreakCondClass()

        /**
         * Predicate depends on at least one carried block-arg other than the
         * counter — i.e., the break-iteration depends on accumulated state. The
         * closure must keep per-iteration evaluation of the predicate.
         */
        data object CarriedDependent : BreakCondClass()
    }

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

        val origCond = terminator.operands[0]
        val breakCond = notNode.operands[0]

        // §0.4.124 — when origCond matches the canonical C5/C6 shape
        // `STEP(SUB(n, args[counterArgIdx]))`, extract the counter info.
        val counter = extractStepCounter(origCond, condBlock.args)
        if (counter == null) {
            return Pattern(whileOp = op, origCond = origCond, breakCond = breakCond)
        }
        // §0.4.125 — Phase 2: also validate the counter's initial value (`const(0)`)
        // and the body's back-edge (`ADD(bodyArgs[counterArgIdx], const(1))`). If
        // either check fails, downgrade to "structural LAND-NOT match only" by
        // leaving the counter fields null. Closure consumers can rely on populated
        // counter fields meaning all three Phase-1/2 invariants hold.
        if (!validateCounterInitAndBackEdge(op, counter.argIdx)) {
            return Pattern(whileOp = op, origCond = origCond, breakCond = breakCond)
        }
        return Pattern(
            whileOp = op,
            origCond = origCond,
            breakCond = breakCond,
            counterArgIdx = counter.argIdx,
            tripCountConst = counter.tripCountConst,
            tripCountParam = counter.tripCountParam,
        )
    }

    /**
     * §0.4.126 — D.3i Phase 3a. Classify [pattern]'s [Pattern.breakCond] by what
     * it depends on within the cond region, so future closure phases can dispatch
     * to the right break-iteration computation strategy.
     *
     * Returns null when [Pattern.counterArgIdx] is null — the classifier requires
     * the Phase-1/1.5/2 invariants (see [Pattern]) to be satisfied so it can
     * distinguish counter references from generic carried-arg references.
     *
     * The walk descends through [DxirOp.operands], [DxirOpResult.source], and
     * [DxirCall.args], collecting the cond block-arg ids reached. It does NOT
     * enter nested regions: [DxirOp.regions] are scoped, so any block args
     * declared inside them are out of scope of the cond region's analysis.
     */
    fun classifyBreakCond(pattern: Pattern): BreakCondClass? {
        val counterArgIdx = pattern.counterArgIdx ?: return null
        val condBlock = pattern.whileOp.regions[0].blocks.single()
        val condArgIds = condBlock.args.map { it.id }.toHashSet()
        val counterArgId = condBlock.args[counterArgIdx].id

        val reached = HashSet<Int>()
        collectCondArgsReached(pattern.breakCond, condArgIds, reached, HashSet())

        return when {
            reached.isEmpty() -> {
                val constBool = (pattern.breakCond as? DxirConst)?.value as? Boolean
                if (constBool != null) BreakCondClass.Constant(alwaysBreaks = constBool)
                else BreakCondClass.LoopInvariant
            }
            reached.all { it == counterArgId } -> BreakCondClass.CounterOnly
            else -> BreakCondClass.CarriedDependent
        }
    }

    private fun collectCondArgsReached(
        node: DxirNode,
        condArgIds: Set<Int>,
        reached: MutableSet<Int>,
        visited: MutableSet<Int>,
    ) {
        if (!visited.add(node.id)) return
        when (node) {
            is DxirBlockArg -> if (node.id in condArgIds) reached += node.id
            is DxirOp -> for (operand in node.operands) collectCondArgsReached(operand, condArgIds, reached, visited)
            is DxirOpResult -> for (operand in node.source.operands) collectCondArgsReached(operand, condArgIds, reached, visited)
            is DxirCall -> for (arg in node.args) collectCondArgsReached(arg, condArgIds, reached, visited)
            is DxirConst, is DxirParam -> Unit
        }
    }

    /**
     * §0.4.125 — Phase 2 validation. Confirm that:
     *  - `whileOp.operands[counterArgIdx]` is `const(0)` (integer-valued).
     *  - The body block's terminator at slot [counterArgIdx] is
     *    `ADD(bodyArgs[counterArgIdx], const(1))` (the standard +=1 increment).
     *
     * Returns true when both invariants hold. Mirrors the same checks
     * [PhiCalculus.detectSimpleLoop] and [PhiCalculus.detectAffineRecurrence]
     * apply for the C5/C6 paths — keeping them in lockstep means the trip-count
     * surface this detector produces is interoperable with the existing
     * closure-pipeline expectations.
     */
    private fun validateCounterInitAndBackEdge(op: DxirOp, counterArgIdx: Int): Boolean {
        val initNode = op.operands.getOrNull(counterArgIdx) as? DxirConst ?: return false
        val initValue = (initNode.value as? Number)?.toDouble() ?: return false
        if (initValue != 0.0) return false

        val bodyBlock = op.regions[1].blocks.single()
        val bodyArgs = bodyBlock.args
        if (counterArgIdx !in bodyArgs.indices) return false
        val backEdge = bodyBlock.terminator.getOrNull(counterArgIdx) as? DxirOp ?: return false
        if (backEdge.op != OpKind.ADD) return false
        if (backEdge.operands.size != 2) return false
        if (backEdge.operands[0].id != bodyArgs[counterArgIdx].id) return false
        val incrConst = backEdge.operands[1] as? DxirConst ?: return false
        val incrValue = (incrConst.value as? Number)?.toDouble() ?: return false
        return incrValue == 1.0
    }

    /**
     * §0.4.124 — extract counter index + trip count bound from a node matching
     * `STEP(SUB(n, args[counterArgIdx]))`, the canonical C5 / C6 cond shape used
     * across [PhiCalculus.detectSimpleLoop] and [PhiCalculus.detectAffineRecurrence].
     * Returns null if [node] doesn't match.
     *
     * The bound `n` is recognised in two forms:
     *  - [DxirConst] with a non-negative integer-valued numeric → `tripCountConst`.
     *  - [DxirParam] of scalar type → `tripCountParam` (loop-invariant symbolic bound).
     *
     * This helper is intentionally NOT shared with the existing PhiCalculus.kt
     * detectors — those run earlier in the pipeline and target slightly different
     * structural shapes (concrete-only for C5; broader for C6). Keeping the D.3i
     * extraction in its own surface lets it evolve independently.
     */
    private data class CounterMatch(
        val argIdx: Int,
        val tripCountConst: Int? = null,
        val tripCountParam: DxirParam? = null,
    )

    private fun extractStepCounter(
        node: DxirNode,
        condArgs: List<DxirBlockArg>,
    ): CounterMatch? {
        if (node !is DxirOp) return null
        if (node.op != OpKind.STEP) return null
        if (node.operands.size != 1) return null
        val sub = node.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        if (sub.operands.size != 2) return null
        val nNode = sub.operands[0]
        val counterArgRef = sub.operands[1]
        val argIdx = condArgs.indexOfFirst { it.id == counterArgRef.id }
        if (argIdx < 0) return null
        return when (nNode) {
            is DxirConst -> {
                val v = (nNode.value as? Number)?.toDouble() ?: return null
                if (v < 0.0 || v != v.toInt().toDouble()) return null
                CounterMatch(argIdx = argIdx, tripCountConst = v.toInt())
            }
            is DxirParam -> {
                if (!nNode.type.isScalar) return null
                CounterMatch(argIdx = argIdx, tripCountParam = nNode)
            }
            else -> null
        }
    }
}
