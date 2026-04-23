package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp

/**
 * §0.4.27 / §0.4.28 — Stage C SOI-identification. Ports paper Fig. 7(a)'s algorithm
 * in two slices:
 *
 *  - **C.1 (§0.4.27)** — the scaffolding: def-use chain, region tree, per-sink
 *    bottom-up traversal. [identify] runs the worklist but doesn't size-check; every
 *    reachable region emits a candidate.
 *  - **C.2a (§0.4.28)** — size-limit marking. [identifyWithSizeLimit] adds the `L`
 *    cap, the `markedLarge` propagation (children-large → parent-large; own-size > L
 *    → self-large), and the small-children-as-SOI promotion when a parent is marked
 *    large. Uses **raw op count** as the size proxy (a conservative upper bound on
 *    the post-coarsening symbolic-expression size); C.3 will replace the size source
 *    with an engine-backed `PhiCalculus.coarsen(subtree, engine)` call.
 *
 * Still deferred to C.2b / C.3:
 *  - `splitOnReuses()` — when a large leaf can't be reduced by its parent, pick the
 *    most-reused SSA variable + partition the leaf's ops into two new nodes. Needs
 *    [DefUseChain.useCount] (already in place) plus a leaf-partitioning rewriter.
 *  - `mergeSomeChildren()` — greedy merge of consecutive small children when their
 *    combined size stays under `L`. C.2b.
 *  - `PhiCalculus.coarsen(subtree, engine)` — produce the simplified primal +
 *    gradient for each final SOI. C.3.
 *
 * The "active sinks" `S` map to [DxirFunction.returns] — each function return is a
 * sink whose gradient a caller wants. Multi-return functions (e.g., `grad2`'s
 * `Pair<dA, dB>`) drive independent traversals; output concatenates per-sink.
 */
data class SoiCandidate(
    /** Tree node identifying which region this candidate covers. */
    val node: RegionTreeNode,
    /** The sink id that drove this candidate's discovery. */
    val sinkId: Int,
    /** Intersection of the node's local ids with the sink's backward-reachable set. */
    val reachableOps: Set<Int>,
    /**
     * §0.4.28 — raw op count of the node's subtree. Present only when [identify] was
     * called with a size-limit threshold; otherwise null (C.1 bare-scaffold mode).
     */
    val subtreeSize: Int? = null,
    /**
     * §0.4.28 — `true` when the node was marked large by the C.2a pass (either its
     * subtree size exceeded `L` OR one of its children was marked large first). When
     * present alongside a list of same-sink candidates, a `markedLarge` parent
     * emits its **small children** (not itself) as SOIs per paper Fig. 7(a). When
     * [identify] was called without a size-limit, this is always `false`.
     */
    val markedLarge: Boolean = false,
)

object SoiIdentification {

    /**
     * C.1 entry point — enumerate every tree node that intersects each sink's
     * backward-reachable set, in bottom-up order. Does NOT size-check or mark
     * nodes. Every emitted candidate is a "small, unmarked" node. Useful when the
     * caller wants to drive its own sizing pass.
     */
    fun identify(fn: DxirFunction): List<SoiCandidate> {
        val chain = DefUseChain.build(fn)
        val tree = RegionTree.build(fn)
        val bottomUp = tree.bottomUp()
        val candidates = mutableListOf<SoiCandidate>()
        for (sinkNode in fn.returns) {
            val reached = chain.backwardReachable(sinkNode.id, fn)
            for (tn in bottomUp) {
                val localIds = tn.directOps.mapNotNull { it.id.takeIf { id -> id in reached } }
                if (localIds.isEmpty()) continue
                candidates += SoiCandidate(
                    node = tn,
                    sinkId = sinkNode.id,
                    reachableOps = localIds.toSet(),
                )
            }
        }
        return candidates
    }

    /**
     * §0.4.28 / §0.4.29 — C.2 entry point. For each active sink, walk the region
     * tree bottom-up and mark nodes `markedLarge`:
     *
     *  - A node is large if any child is large.
     *  - Else if the node's raw subtree op count exceeds [sizeLimit], it's large.
     *  - Else it's small.
     *
     * §0.4.29 — when a marked-large node is a leaf, attempt `splitOnReuses` per
     * paper Fig. 7(a) line 18: partition the leaf's ops around the free-variable
     * SSA id with the highest use count in `fn`. If the split succeeds, the two
     * new sub-leaves become SOIs instead of the original oversized leaf. If the
     * split fails (no free variables, all ops depend on pivot, single-op leaf),
     * the large leaf falls through as an SOI-as-is (C.2a behaviour).
     *
     * Returns `SoiResult` holding:
     *  - `candidates`: all reachable tree nodes for each sink, with `markedLarge` + `subtreeSize` set.
     *  - `sois`: the final SOI set — small root, or small-children-of-large-parents, or
     *    split-sub-leaves when a large leaf was partitioned.
     */
    fun identifyWithSizeLimit(
        fn: DxirFunction,
        sizeLimit: Int,
        engine: SymbolicEngine? = null,
    ): SoiResult {
        require(sizeLimit > 0) { "sizeLimit must be positive (got $sizeLimit)" }
        val chain = DefUseChain.build(fn)
        val tree = RegionTree.build(fn)
        val bottomUp = tree.bottomUp()
        // §0.4.30 — compute per-leaf sizes. With an engine, synthesize a mini-function
        // per leaf, run PhiCalculus.apply, and use the resulting body op count. Without
        // an engine, fall back to raw op count (subtreeSize). Leaves that fail to
        // coarsen (region-bearing op, multi-result, engine crash) also fall back.
        val leafSizes = HashMap<RegionTreeNode, Int>()
        for (n in bottomUp) {
            if (!n.isLeaf) continue
            leafSizes[n] = if (engine == null) n.subtreeSize() else {
                when (val cr = PhiCalculus.coarsenLeaf(n, fn, engine)) {
                    is CoarsenResult.Success -> cr.size
                    is CoarsenResult.Failure -> n.subtreeSize()
                }
            }
        }
        val marked = markedLargeSet(bottomUp, sizeLimit, leafSizes)
        val allCandidates = mutableListOf<SoiCandidate>()
        val finalSois = mutableListOf<SoiCandidate>()
        for (sinkNode in fn.returns) {
            val reached = chain.backwardReachable(sinkNode.id, fn)
            val perSinkCandidates = mutableListOf<SoiCandidate>()
            for (tn in bottomUp) {
                val localIds = tn.directOps.mapNotNull { it.id.takeIf { id -> id in reached } }
                if (localIds.isEmpty()) continue
                perSinkCandidates += SoiCandidate(
                    node = tn,
                    sinkId = sinkNode.id,
                    reachableOps = localIds.toSet(),
                    subtreeSize = sizeFor(tn, leafSizes),
                    markedLarge = tn in marked,
                )
            }
            allCandidates += perSinkCandidates
            finalSois += chooseSois(perSinkCandidates, tree.root, marked, fn, chain, sizeLimit)
        }
        return SoiResult(
            candidates = allCandidates,
            sois = finalSois,
            sizeLimit = sizeLimit,
            tree = tree,
        )
    }

    /**
     * §0.4.30 — resolve a node's size from the precomputed leaf-size map. Leaves use
     * the engine-backed size when available (else raw count); non-leaves aggregate
     * their children's sizes plus their own directOps count minus the region-bearing
     * ops whose contribution lives in the children (to avoid double-counting).
     */
    private fun sizeFor(
        node: RegionTreeNode,
        leafSizes: Map<RegionTreeNode, Int>,
    ): Int {
        if (node.isLeaf) return leafSizes[node] ?: node.subtreeSize()
        // Non-leaf: count directOps (including region-bearing op itself — it's an op)
        // plus each child's size. Children cover the INTERIOR of region-bearing ops,
        // which don't have their own ops to count beyond the op itself — so directOps
        // already contains the region-bearing op without double-count.
        var total = node.directOps.size
        for (c in node.children) total += sizeFor(c, leafSizes)
        return total
    }

    /**
     * Bottom-up marking pass. A node is marked large iff:
     *  - any of its children is marked large, OR
     *  - its size (engine-backed for leaves via [leafSizes], raw count for non-leaves
     *    via [sizeFor] aggregation) exceeds [sizeLimit].
     * Since [bottomUp] lists children before parents, a single pass suffices.
     */
    private fun markedLargeSet(
        bottomUp: List<RegionTreeNode>,
        sizeLimit: Int,
        leafSizes: Map<RegionTreeNode, Int>,
    ): Set<RegionTreeNode> {
        val marked = HashSet<RegionTreeNode>()
        for (n in bottomUp) {
            val anyChildLarge = n.children.any { it in marked }
            if (anyChildLarge || sizeFor(n, leafSizes) > sizeLimit) marked += n
        }
        return marked
    }

    /**
     * Pick the final SOI set for a single sink. Paper Fig. 7(a)'s convention: when a
     * parent is marked large, its small children become SOIs; a marked-large parent
     * does NOT itself become an SOI (it's too big to coarsen in one shot). When the
     * root is NOT marked large, the root is the single SOI.
     *
     * §0.4.29: large leaves are further refined by `splitOnReuses` — partition the
     * leaf's ops around its highest-use-count free variable. Split fragments become
     * SOIs in place of the original oversized leaf. If the split can't reduce size
     * (single op, no free variables, or all ops depend on pivot), the large leaf
     * falls through as-is (C.2a fallback).
     *
     * Edge cases:
     *  - A node filtered by per-sink reachability (empty `directOps` intersection)
     *    isn't considered for SOI emission — matches C.1's property (i) filter.
     */
    private fun chooseSois(
        perSinkCandidates: List<SoiCandidate>,
        root: RegionTreeNode,
        marked: Set<RegionTreeNode>,
        fn: DxirFunction,
        chain: DefUseChain,
        sizeLimit: Int,
    ): List<SoiCandidate> {
        val byTreeNode = perSinkCandidates.associateBy { it.node }
        // If the root survived marking, it's the SOI for this sink.
        val rootCand = byTreeNode[root]
        if (rootCand != null && !rootCand.markedLarge) return listOf(rootCand)
        // Otherwise walk marked-large parents top-down and emit their small children.
        val sois = mutableListOf<SoiCandidate>()
        fun visit(n: RegionTreeNode) {
            val cand = byTreeNode[n]
            if (cand == null) {
                // Not in this sink's reachable set; recurse to children.
                for (c in n.children) visit(c)
                return
            }
            if (!cand.markedLarge) {
                sois += cand
                return
            }
            // Marked large — promote small children; recurse through large children.
            for (c in n.children) visit(c)
            if (n.isLeaf && cand.markedLarge) {
                // §0.4.29: try splitting a large leaf around its most-reused free var.
                val split = splitOnReuses(n, fn, chain, sizeLimit, cand.sinkId)
                if (split != null) sois += split else sois += cand
            }
        }
        visit(root)
        return sois
    }

    // ------------------------------------------------------------------------
    // §0.4.29 — C.2b: splitOnReuses
    // ------------------------------------------------------------------------

    /**
     * Paper Fig. 7(a) line 18 — partition a large leaf around its most-reused free
     * variable. Returns two [SoiCandidate]s covering disjoint subsets of the leaf's
     * directOps, or null if the split can't make progress.
     *
     * Algorithm:
     *  1. Collect the leaf's "free variables" — operand ids referenced from inside
     *     the leaf but NOT declared by it. For a leaf region, this is every outer-
     *     scope operand referenced by any op in `directOps`.
     *  2. Pick the pivot: free variable with the maximum `DefUseChain.useCount` in
     *     `fn` (breaks ties by smallest id for determinism). Rationale from paper
     *     §5: "splitting point is chosen to be the variable that... has the largest
     *     number of references (and hence reuses) in `f`." Most-reused variables
     *     participate in more chain-rule simplifications when the splitter respects
     *     them, so the gradient pass can factor through them.
     *  3. Compute `dependsOnPivot`: ops that transitively reference the pivot.
     *  4. Partition: `pre` = ops NOT depending on pivot; `post` = ops depending on
     *     pivot. Both keep the original program order.
     *  5. Refuse the split if either partition is empty (one-sided split = no
     *     reduction) or the leaf has no free variables at all (pivot candidates
     *     empty).
     *
     * Returns null when split can't reduce the leaf. Caller falls back to emitting
     * the oversized leaf as-is (C.2a fallback).
     */
    private fun splitOnReuses(
        leaf: RegionTreeNode,
        fn: DxirFunction,
        chain: DefUseChain,
        sizeLimit: Int,
        sinkId: Int,
    ): List<SoiCandidate>? {
        if (!leaf.isLeaf) return null
        if (leaf.directOps.size < 2) return null

        // Step 1: leaf's free variables (ids referenced from inside but not declared here).
        val localIds = leaf.directOps.map { it.id }.toHashSet()
        val freeVars = LinkedHashSet<Int>()
        for (n in leaf.directOps) {
            if (n !is DxirOp) continue
            for (operand in n.operands) {
                val opId = operand.id
                if (opId !in localIds) freeVars += opId
            }
        }
        if (freeVars.isEmpty()) return null

        // Step 2: pivot = most-reused free var in `fn` (ties → smallest id for determinism).
        val pivot = freeVars.maxWithOrNull(
            compareBy<Int> { chain.useCount(it) }.thenBy { -it },
        ) ?: return null

        // Step 3: transitive closure of ops depending on pivot.
        val dependsOnPivot = HashSet<Int>()
        for (n in leaf.directOps) {
            if (n !is DxirOp) continue
            val direct = n.operands.any { it.id == pivot || it.id in dependsOnPivot }
            if (direct) dependsOnPivot += n.id
        }

        // Step 4: partition preserving program order.
        val pre = leaf.directOps.filter { it.id !in dependsOnPivot }
        val post = leaf.directOps.filter { it.id in dependsOnPivot }
        if (pre.isEmpty() || post.isEmpty()) return null

        // Build two new RegionTreeNodes (same parent/regionOp as the original leaf).
        val preNode = RegionTreeNode(
            parent = leaf.parent,
            directOps = pre,
            regionOp = leaf.regionOp,
            regionIndex = leaf.regionIndex,
            blockIndex = leaf.blockIndex,
        )
        val postNode = RegionTreeNode(
            parent = leaf.parent,
            directOps = post,
            regionOp = leaf.regionOp,
            regionIndex = leaf.regionIndex,
            blockIndex = leaf.blockIndex,
        )

        // Emit as SOI candidates. We DON'T re-run size checking on the fragments —
        // the split is a one-shot attempt; if a fragment is still > sizeLimit, it
        // flows through as a large-leaf fallback (same behavior as pre-split). C.2b's
        // termination is guaranteed (each split strictly reduces op count; we don't
        // split single-op nodes).
        fun buildCand(n: RegionTreeNode): SoiCandidate = SoiCandidate(
            node = n,
            sinkId = sinkId,
            reachableOps = n.directOps.map { it.id }.toSet(),
            subtreeSize = n.subtreeSize(),
            markedLarge = n.subtreeSize() > sizeLimit,
        )
        return listOf(buildCand(preNode), buildCand(postNode))
    }

    /**
     * Test-only helper: return the raw [RegionTree] for [fn] without running the full
     * identification pipeline. Useful when tests want to assert tree shape before
     * checking which subset ends up as [SoiCandidate]s.
     */
    internal fun regionTreeOf(fn: DxirFunction): RegionTree = RegionTree.build(fn)
}

/**
 * §0.4.28 — result of [SoiIdentification.identifyWithSizeLimit]. Separates:
 *  - [candidates]: every reachable tree node for every sink (for debugging /
 *    introspection). Same content a C.1 [SoiIdentification.identify] would emit,
 *    enriched with `subtreeSize` + `markedLarge`.
 *  - [sois]: the final SOI set — what C.3's coarsening + splice pipeline should act on.
 *  - [tree] (§0.4.35): the [RegionTree] used for identification. Exposed so downstream
 *    passes (e.g., [PhiCalculus.coarsenFunction]) can distinguish original tree nodes
 *    from split fragments produced by [splitOnReuses] via reference-identity lookup.
 */
data class SoiResult(
    val candidates: List<SoiCandidate>,
    val sois: List<SoiCandidate>,
    val sizeLimit: Int,
    val tree: RegionTree,
)
