package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp

/**
 * §0.4.27 — Stage C.1 def-use region tree. Structures a [DxirFunction]'s ops by the
 * natural code-region hierarchy: the root is the function body; each [DxirOp] with
 * non-empty [DxirOp.regions] (IF, WHILE, MANUAL_COMPUTATION) introduces a child node
 * per region-block. Leaves are straight-line sequences of ops with no nested regions.
 *
 * Per the paper (§5, two paragraphs above Fig. 7): "Def-use region tree has two major
 * differences from code region hierarchy: (i) only relevant variable definitions are
 * considered; (ii) every loop-exit 𝜙 function is put as part of the region of the
 * associated loop." Property (i) requires intersection with the backward-reachable
 * set for a specific sink, handled by [SoiIdentification] — [RegionTree.build] emits
 * the full structural tree (which [SoiIdentification] filters per sink). Property
 * (ii) is implicit in dxir because loop-exit φs don't exist as standalone ops — a
 * WHILE's yielded loop-carried values are DxirOpResults of the WHILE itself, which
 * sit in the same scope as the WHILE (the parent region).
 *
 * Scope (C.1 first cut):
 *  - Every [DxirOp] with regions creates children: one [RegionTreeNode] per region
 *    block (IF has 2 regions × 1 block each → 2 children; WHILE is the same).
 *  - The node's [directOps] includes the region-bearing op itself (it's declared at
 *    this scope) — children only cover the region's INTERIOR.
 *  - Bottom-up traversal yields deepest-children-first, parents-last.
 *  - Multi-block regions (not yet emitted by DxirBuilder surface) would produce one
 *    child per block; deferred scope until needed.
 */
class RegionTreeNode internal constructor(
    val parent: RegionTreeNode?,
    /**
     * All nodes directly declared in this scope, in program order. Includes
     * region-bearing ops (IF/WHILE) whose bodies live in child nodes. Does NOT include
     * nodes from child scopes.
     */
    val directOps: List<DxirNode>,
    /**
     * The [DxirOp] whose region body this node represents, or null if this is the
     * function root. Useful for walking parentward or querying `region.isBody ? 1 : 0`.
     */
    val regionOp: DxirOp?,
    /** Which of [regionOp]'s regions this node represents. 0 for root. */
    val regionIndex: Int,
    /** Which block of the region this node covers (v1: always 0 — single-block regions). */
    val blockIndex: Int,
) {
    internal val mutableChildren: MutableList<RegionTreeNode> = mutableListOf()
    val children: List<RegionTreeNode> get() = mutableChildren

    val isRoot: Boolean get() = parent == null
    val isLeaf: Boolean get() = children.isEmpty()

    /**
     * Depth from the root. Root is 0; direct children of root are 1; etc. O(depth).
     */
    val depth: Int
        get() {
            var d = 0
            var n = parent
            while (n != null) { d++; n = n.parent }
            return d
        }

    /**
     * All ids declared at this scope (directOps + their block args if any). Does NOT
     * include ids from child scopes — use [collectIds] for transitive closure.
     */
    fun localIds(): Set<Int> {
        val ids = HashSet<Int>()
        for (n in directOps) ids += n.id
        return ids
    }

    /**
     * §0.4.28 — Total op count in this node's subtree (directOps + all children
     * transitively). Used as the size proxy for the C.2a size-check heuristic:
     * a subtree bigger than `L` (plan §8.2; default 50) signals that symbolic
     * coarsening would blow past the engine's budget. Raw pre-coarsening count is
     * a conservative upper bound — the true post-`PhiCalculus.apply` symbolic
     * expression is typically smaller because F1/F2/C1/C3 collapse φs and C5-C9
     * close loops. C.3's engine-backed `PhiCalculus.coarsen(subtree, engine)` will
     * replace this with the exact count when the integration lands.
     */
    fun subtreeSize(): Int {
        var total = directOps.size
        for (child in children) total += child.subtreeSize()
        return total
    }
}

class RegionTree internal constructor(
    val root: RegionTreeNode,
) {
    /**
     * Depth-first post-order traversal: each child is listed before its parent. The
     * root appears last. Among siblings, order follows program order (regionIndex
     * ascending). This matches the paper's "W: a worklist with all nodes in T added
     * in bottom-up order" — sinks first, root last.
     */
    fun bottomUp(): List<RegionTreeNode> {
        val out = mutableListOf<RegionTreeNode>()
        fun visit(n: RegionTreeNode) {
            for (child in n.children) visit(child)
            out += n
        }
        visit(root)
        return out
    }

    /** All nodes in the tree, in depth-first pre-order (root first). Useful for tests. */
    fun preOrder(): List<RegionTreeNode> {
        val out = mutableListOf<RegionTreeNode>()
        fun visit(n: RegionTreeNode) {
            out += n
            for (child in n.children) visit(child)
        }
        visit(root)
        return out
    }

    /** Count of tree nodes. */
    val size: Int get() = preOrder().size

    companion object {
        /**
         * Build the region tree for [fn]. O(body size) — each op is visited once; each
         * region-bearing op spawns child nodes that recurse into their blocks.
         */
        fun build(fn: DxirFunction): RegionTree {
            val root = RegionTreeNode(
                parent = null,
                directOps = fn.body.toList(),
                regionOp = null,
                regionIndex = 0,
                blockIndex = 0,
            )
            for (node in fn.body) attachChildren(root, node)
            return RegionTree(root)
        }

        private fun attachChildren(parent: RegionTreeNode, node: DxirNode) {
            if (node !is DxirOp) return
            if (!node.hasRegions) return
            for ((rIdx, region) in node.regions.withIndex()) {
                for ((bIdx, block) in region.blocks.withIndex()) {
                    val child = RegionTreeNode(
                        parent = parent,
                        directOps = block.body.toList(),
                        regionOp = node,
                        regionIndex = rIdx,
                        blockIndex = bIdx,
                    )
                    parent.mutableChildren += child
                    for (inner in block.body) attachChildren(child, inner)
                }
            }
        }
    }
}
