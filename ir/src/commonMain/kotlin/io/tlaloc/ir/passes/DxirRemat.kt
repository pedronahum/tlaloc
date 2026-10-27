package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.OpKind

/**
 * Rematerialization (activation checkpointing) of a gradient program: the backward pass
 * recomputes the forward values it reads instead of keeping them alive from the forward pass.
 *
 * The input is a gradient built with its forward pass included
 * (`DxirReverseTransform.apply(primal, includeForward = true)`): return 0 is the loss, and the
 * forward part is everything the loss depends on. The other body nodes are the backward part.
 *
 * Every forward op the backward part reads is either saved ([Policy.saves]) and read as it is,
 * or recomputed from saved values, parameters and constants. A saved value or parameter that
 * feeds a recomputation passes through an [OpKind.OPTIMIZATION_BARRIER] anchored on the backward
 * value computed just before the backward node that first needs the recomputation. XLA can then neither run the
 * recomputation during the forward pass nor merge it with the forward copy, so the unsaved
 * forward values die when the forward pass ends.
 *
 * The program computes the same values: every recomputed op is a copy of a forward op with the
 * same operands, so results are equal bit for bit in the interpreter. Ops with regions and
 * multi-result ops are always saved.
 */
object DxirRemat {

    /** Which forward values the gradient keeps. Parameters and constants are always available. */
    fun interface Policy {
        fun saves(op: DxirOp): Boolean
    }

    /** Keep nothing: recompute every forward value the backward pass reads. The first backward
     * step then recomputes the whole forward pass, and those copies stay alive through the
     * backward pass, so this saves memory only for a short forward. */
    val SAVE_NOTHING: Policy = Policy { false }

    /** Keep the results of matrix products and convolutions; recompute the elementwise work
     * between them (JAX's `dots_saveable`). */
    val SAVE_DOTS: Policy = Policy { it.op in DOTS }

    private val DOTS = setOf(OpKind.MATMUL, OpKind.DOT, OpKind.CONV2D, OpKind.CONV_TRANSPOSE2D)

    /** Keep the forward values with these node ids. */
    fun saveIds(ids: Set<Int>): Policy = Policy { it.id in ids }

    /** The attribute that marks an op's result as a checkpoint for [SAVE_MARKED]: set
     * `SAVE_ATTR to true` on the op in the primal (for example each layer's output). The
     * reverse transform copies attributes into the gradient program. */
    const val SAVE_ATTR: String = "remat_save"

    /** Keep the results of ops marked with [SAVE_ATTR]; recompute everything between them
     * (JAX's `save_only_these_names` over `checkpoint_name`). */
    val SAVE_MARKED: Policy = Policy { it.attrs[SAVE_ATTR] == true }

    /** The attribute that makes an op's result recomputed under every policy: set
     * `RECOMPUTE_ATTR to true` on an op whose result is large and cheap to recompute (a chunk of
     * an LM head's logits). */
    const val RECOMPUTE_ATTR: String = "remat_recompute"

    /** [SAVE_MARKED] and [SAVE_DOTS] together. */
    val SAVE_MARKED_AND_DOTS: Policy = Policy { SAVE_MARKED.saves(it) || SAVE_DOTS.saves(it) }

    fun apply(gradient: DxirFunction, policy: Policy): DxirFunction {
        require(gradient.returns.isNotEmpty()) { "DxirRemat: '${gradient.name}' returns nothing" }
        val forward = ancestors(gradient.returns[0])
        var nextId = DxirInline.maxId(gradient) + 1

        fun keeps(n: DxirNode): Boolean = when (n) {
            is DxirParam, is DxirConst -> true
            is DxirOp -> n.id !in forward || n.hasRegions || n.isMultiResult ||
                (n.attrs[RECOMPUTE_ATTR] != true && policy.saves(n))
            else -> true // DxirOpResult and region arguments are never copied
        }

        val body = ArrayList<DxirNode>(gradient.body.size)
        val recomputed = HashMap<Int, DxirNode>()
        val barriered = HashMap<Int, DxirNode>()
        var anchor: DxirNode = gradient.returns[0]

        fun barrier(saved: DxirNode): DxirNode = barriered.getOrPut(saved.id) {
            // Parameters pass through the barrier as well: a recomputation fed only by
            // parameters would otherwise be merged with its forward copy.
            if (saved is DxirConst) return saved
            DxirOp(nextId++, OpKind.OPTIMIZATION_BARRIER, listOf(saved, anchor), types = listOf(saved.type)).also { body += it }
        }

        var group = 0
        val groupOf = HashMap<Int, Int>()

        fun recompute(n: DxirNode): DxirNode {
            if (keeps(n)) return n
            recomputed[n.id]?.let { earlier ->
                // Recomputed for an earlier backward node: read it through this node's barrier, so
                // what is recomputed from it now is not scheduled at that earlier node.
                return if (groupOf[earlier.id] == group) earlier else barrier(earlier)
            }
            val op = n as DxirOp
            val operands = op.operands.map { u -> if (keeps(u)) barrier(u) else recompute(u) }
            val copy = DxirOp(nextId++, op.op, operands, op.attrs, op.types, op.sharding)
            body += copy
            recomputed[n.id] = copy
            groupOf[copy.id] = group
            return copy
        }

        val replaced = HashMap<Int, DxirNode>()
        for (node in gradient.body) {
            if (node.id in forward || node !is DxirOp || node.hasRegions) {
                body += node
                continue
            }
            // Recomputations started for this backward node read their saved inputs through
            // barriers anchored here, not through barriers made for an earlier node: values
            // recomputed for different backward steps (an LM head's chunks) then cannot all be
            // scheduled at the first one.
            barriered.clear()
            group++
            val operands = node.operands.map { u ->
                when {
                    replaced[u.id] != null && u is DxirOp -> replaced.getValue(u.id)
                    u.id in forward && !keeps(u) -> recompute(u)
                    else -> u
                }
            }
            val rewritten = if (operands.indices.all { operands[it] === node.operands[it] }) node
            else DxirOp(node.id, node.op, operands, node.attrs, node.types, node.sharding, node.regions)
            if (rewritten !== node) replaced[node.id] = rewritten
            body += rewritten
            anchor = rewritten
        }
        val returns = gradient.returns.map { r -> if (r is DxirOp) replaced[r.id] ?: r else r }
        return deadCodeEliminated(DxirFunction(gradient.name, gradient.params, body, returns, gradient.meshes))
    }

    /** The ids of [root] and every op it depends on. */
    private fun ancestors(root: DxirNode): Set<Int> {
        val seen = HashSet<Int>()
        val stack = ArrayDeque(listOf(root))
        while (stack.isNotEmpty()) {
            val n = stack.removeLast()
            if (!seen.add(n.id)) continue
            if (n is DxirOp) stack.addAll(n.operands)
        }
        return seen
    }

    /** [fn] without the forward ops nothing reads any more. Left as it is when an op has regions,
     * whose bodies may read outer values. */
    private fun deadCodeEliminated(fn: DxirFunction): DxirFunction {
        if (fn.body.any { it is DxirOp && it.hasRegions }) return fn
        val live = HashSet<Int>()
        fn.returns.forEach { live += it.id }
        for (n in fn.body.asReversed()) {
            if (n.id in live && n is DxirOp) n.operands.forEach { live += it.id }
        }
        val body = fn.body.filter { it.id in live }
        return if (body.size == fn.body.size) fn else DxirFunction(fn.name, fn.params, body, fn.returns, fn.meshes)
    }
}
