package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirBlockArg
import io.tlaloc.ir.DxirCall
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam

/**
 * §0.4.27 — Stage C.1 def-use analysis on a [DxirFunction]. Records both **forward**
 * (def → list of consumer op ids) and **reverse** (backward reachability from any
 * specified sink) information. The forward map is used by the SOI identification
 * algorithm's `splitOnReuses` step (picks the free variable with the most consumers
 * to split a too-large leaf); the backward-reachable set is used to restrict the
 * region tree to ops actually contributing to a specific active sink `s ∈ S`.
 *
 * Scope (C.1 first cut):
 *  - Records uses from op-operand edges at every nesting depth (top-level body, IF/
 *    WHILE regions, nested regions). Region terminators (yields) also count — they
 *    carry values out of the region.
 *  - Records [DxirFunction.returns] as uses too, so sinks show up in [usersOf].
 *  - [DxirOpResult] references are attributed to the underlying source op's id (mirrors
 *    the SSA convention that multi-result ops share one id — the result-index only
 *    disambiguates which output is read, not the producer).
 *  - [DxirCall] is included for completeness but the FIR-emitted surface doesn't use
 *    it today; every op-operand in a call is an ordinary SSA edge.
 *
 * O(body size) to build. Immutable after construction — safe to share across passes.
 */
class DefUseChain internal constructor(
    /** Forward map: def-id → list of consumer op ids. Consumer ids are the OP's id
     *  (not the operand-within-the-op); if op X references V twice, X appears once. */
    private val usesByDef: Map<Int, List<Int>>,
    /** Every declared SSA id in the function (params + body nodes + region block args
     *  + ops nested in regions). Useful for sanity checks and reachability scope. */
    private val declaredIds: Set<Int>,
) {

    /** Consumer op ids of the value produced at [defId]. Empty list if unused. */
    fun usersOf(defId: Int): List<Int> = usesByDef[defId] ?: emptyList()

    /** Number of consumers of [defId]. Convenience for `splitOnReuses` heuristic. */
    fun useCount(defId: Int): Int = usersOf(defId).size

    /** True if [id] is declared somewhere in the function's scope. */
    fun isDeclared(id: Int): Boolean = id in declaredIds

    /** All ids with at least one recorded use (i.e., appearing in an operand edge). */
    fun allDefsWithUses(): Set<Int> = usesByDef.keys

    /**
     * Ids transitively reachable backward from [sinkId] through operand edges. The
     * traversal:
     *  - starts from `sinkId` (which is itself included in the result),
     *  - for each [DxirOp] encountered, recurses into its operand ids and also walks
     *    its regions' block bodies (every op-operand + every yield reference),
     *  - stops at [DxirParam] / [DxirConst] / [DxirBlockArg] leaves (they have no
     *    operands to follow).
     *
     * Returns `Set<Int>` of ids (canonical — DxirOpResult indices are collapsed).
     */
    fun backwardReachable(sinkId: Int, fn: DxirFunction): Set<Int> {
        val byId: Map<Int, DxirNode> = buildIdMap(fn)
        val reached = HashSet<Int>()
        val worklist = ArrayDeque<Int>()
        fun enqueue(id: Int) { if (reached.add(id)) worklist.addLast(id) }
        enqueue(sinkId)
        while (worklist.isNotEmpty()) {
            val id = worklist.removeFirst()
            val node = byId[id] ?: continue
            if (node !is DxirOp) continue
            for (operand in node.operands) enqueue(operand.id)
            for (region in node.regions) for (block in region.blocks) {
                for (bodyNode in block.body) {
                    // Walking region body: only reach into ops that are ALREADY reached
                    // via an enclosing chain. Pre-emptively including every region op
                    // would pollute the backward set with unreachable leaf computation.
                    // But since we haven't proven reachability for region body nodes
                    // individually, safely enumerate: if the region's parent op is in
                    // the reached set (it is — we got here by following from sinkId),
                    // every region body node is part of the region's semantics and may
                    // be consumed via the terminator's yield chain — so enqueue.
                    enqueue(bodyNode.id)
                    if (bodyNode is DxirOp) {
                        for (operand in bodyNode.operands) enqueue(operand.id)
                    }
                }
                for (term in block.terminator) enqueue(term.id)
                for (arg in block.args) enqueue(arg.id)
            }
        }
        return reached
    }

    companion object {
        /**
         * Build the def-use chain for [fn]. One pass over every node at every nesting
         * depth; O(body size).
         */
        fun build(fn: DxirFunction): DefUseChain {
            val uses = HashMap<Int, MutableList<Int>>()
            val declared = HashSet<Int>()

            // Walk the function, recording every id and every operand-use edge.
            for (p in fn.params) declared += p.id
            fun recordUse(def: Int, consumer: Int) {
                val list = uses.getOrPut(def) { mutableListOf() }
                if (consumer !in list) list += consumer
            }
            fun walkNode(node: DxirNode, parentId: Int?) {
                declared += node.id
                when (node) {
                    is DxirOp -> {
                        for (operand in node.operands) recordUse(operand.id, node.id)
                        for (region in node.regions) for (block in region.blocks) {
                            for (arg in block.args) declared += arg.id
                            for (bodyNode in block.body) walkNode(bodyNode, node.id)
                            // Block terminator references use the enclosing op id as
                            // the "consumer" — yielding a value out of a block is a
                            // use scoped to the op whose region contains the block.
                            for (term in block.terminator) recordUse(term.id, node.id)
                        }
                    }
                    is DxirCall -> for (arg in node.args) recordUse(arg.id, node.id)
                    else -> {} // params / consts / block args / op results — no outgoing edges
                }
            }
            for (node in fn.body) walkNode(node, null)
            // Function returns are sinks: attribute them with the sentinel consumer -1.
            for (ret in fn.returns) recordUse(ret.id, -1)

            return DefUseChain(
                usesByDef = uses.mapValues { it.value.toList() },
                declaredIds = declared,
            )
        }

        /** Flatten the function into `id → node` for reachability's operand lookups. */
        private fun buildIdMap(fn: DxirFunction): Map<Int, DxirNode> {
            val map = HashMap<Int, DxirNode>()
            for (p in fn.params) map[p.id] = p
            fun walk(node: DxirNode) {
                if (node is DxirOpResult) return // shares id with source; don't shadow
                map[node.id] = node
                if (node is DxirOp) for (region in node.regions) for (block in region.blocks) {
                    for (arg in block.args) map[arg.id] = arg
                    for (b in block.body) walk(b)
                }
            }
            for (n in fn.body) walk(n)
            return map
        }
    }
}
