package io.tlaloc.ir.recognizer.fusion

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * Layer 3 §0.4.256+ — tile-fusion candidate identification + annotation.
 *
 * # What
 *
 * Walks a `DxirFunction` and identifies *tile groups* — maximal connected
 * sets of elementwise ops in the body that share an output shape. Each
 * group describes a region downstream codegen can emit as a single tiled
 * loop (one fused kernel reading inputs, computing the chain in
 * registers, writing outputs once).
 *
 * # Two entry points
 *
 * - [identifyTileGroups] — analysis only. Returns a `List<TileGroup>`.
 *   Doesn't mutate the IR.
 * - [annotateTileGroups] — analysis + IR rewrite. Returns a new
 *   `DxirFunction` where each fusion candidate carries `tile_group:
 *   <groupId>` in its attrs. Stable across passes; codegen reads the
 *   attr to decide tile-loop boundaries.
 *
 * # What's NOT fused (by design, v1)
 *
 * - **Reduction → elementwise** chains (the "softmax bias" pattern).
 *   Reductions break the elementwise-shape invariant; tile-loop
 *   reduction needs a two-pass schedule that v1 doesn't synthesise.
 *   L3.4 closer / IREE backend can extend.
 * - **Cross-COARSENED fusion**. The fusion pass walks the outer
 *   function's straight-line body; ops *inside* a `COARSENED` op's
 *   `primal_body` are not visited. Future-cost: recurse into bodies.
 * - **Multi-result + region-bearing ops** are skipped wholesale.
 *
 * # Why a separate pass, not inline-with-coarsening
 *
 * The L3.2 coarsener focuses on *recognized compound forms* (FlashAttention,
 * RMS norm, etc.) — patterns with vendor-fused kernel equivalents.
 * Tile fusion handles the *unrecognized residue* — long chains of plain
 * elementwise ops that don't match any compound. Both passes target
 * the same goal (reducing HBM traffic) but at different granularities.
 *
 * Running coarsening first means recognized regions become a single
 * COARSENED op (treated as opaque by the fusion pass). Tile fusion then
 * groups whatever elementwise residue remains around those coarsened
 * regions.
 */
data class TileGroup(
    val groupId: Int,
    val opIds: List<Int>,
    val sharedShape: DxirType,
)

/**
 * Op kinds that are eligible for tile fusion. Pure elementwise unary or
 * binary, single-result, no nested regions. Reductions, matmul, casts,
 * shape-only ops (TRANSPOSE/RESHAPE/BROADCAST) are excluded — they don't
 * fit the "one-op-per-output-element" tile-loop body shape.
 */
internal val ELEMENTWISE_OP_KINDS: Set<OpKind> = setOf(
    OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW,
    OpKind.NEG, OpKind.ABS, OpKind.EXP, OpKind.LOG, OpKind.SQRT, OpKind.RSQRT,
    OpKind.TANH, OpKind.SIGMOID, OpKind.RELU, OpKind.GELU, OpKind.SILU,
    OpKind.SIN, OpKind.COS, OpKind.SIGN, OpKind.STEP,
    // Comparison + logical: also elementwise.
    OpKind.LAND, OpKind.NOT,
)

/**
 * Identify maximal connected sets of elementwise ops sharing an output
 * shape. Returns groups of size ≥ 2 (singleton ops gain nothing from
 * tiling on their own).
 *
 * Algorithm: union-find. Mark each elementwise op as a singleton; for
 * each operand → consumer pair where both are elementwise and have the
 * same shape, union them. Components of size ≥ 2 become tile groups.
 */
fun identifyTileGroups(fn: DxirFunction): List<TileGroup> {
    val opsById = fn.body.filterIsInstance<DxirOp>().associateBy { it.id }
    val candidates = opsById.values
        .filter {
            it.op in ELEMENTWISE_OP_KINDS &&
                !it.isMultiResult &&
                it.regions.isEmpty()
        }
        .map { it.id }
        .toSet()
    if (candidates.size < 2) return emptyList()

    // Union-find. Initially every candidate is its own component.
    val parent = HashMap<Int, Int>().apply {
        for (id in candidates) put(id, id)
    }
    fun find(x: Int): Int {
        var cur = x
        while (parent[cur] != cur) cur = parent[cur]!!
        // Path compression.
        var node = x
        while (parent[node] != cur) {
            val next = parent[node]!!
            parent[node] = cur
            node = next
        }
        return cur
    }
    fun union(a: Int, b: Int) {
        val ra = find(a)
        val rb = find(b)
        if (ra != rb) parent[ra] = rb
    }

    for (op in opsById.values) {
        if (op.id !in candidates) continue
        for (operand in op.operands) {
            val producerId = operand.id
            if (producerId in candidates) {
                val producer = opsById[producerId] ?: continue
                if (producer.type == op.type) {
                    union(producerId, op.id)
                }
            }
        }
    }

    // Group by root; emit tile groups for components of size ≥ 2.
    val byRoot: MutableMap<Int, MutableList<Int>> = HashMap()
    for (id in candidates) {
        byRoot.getOrPut(find(id)) { mutableListOf() } += id
    }
    val bodyOrder = fn.body.withIndex().associate { (i, n) -> n.id to i }
    var nextGroupId = 0
    return byRoot.values
        .filter { it.size >= 2 }
        .map { ids ->
            val sorted = ids.sortedBy { bodyOrder[it] ?: Int.MAX_VALUE }
            TileGroup(
                groupId = nextGroupId++,
                opIds = sorted,
                sharedShape = opsById[sorted.first()]!!.type,
            )
        }
        // Stable order: by first body-position of the group (avoids
        // dependency on HashMap iteration order, which is impl-defined
        // even within Kotlin/JVM).
        .sortedBy { bodyOrder[it.opIds.first()] ?: Int.MAX_VALUE }
        .mapIndexed { i, g -> g.copy(groupId = i) }
}

/**
 * Run [identifyTileGroups] and annotate each candidate op with
 * `tile_group: <Int>` in its attrs. Returns a new function — the
 * original is unchanged.
 *
 * Annotation is a small, additive change (one extra map entry per
 * group-member op); other passes ignore the attr unless they explicitly
 * opt in. StableHLO emit / IREE codegen consume it to choose tile-loop
 * boundaries.
 */
fun annotateTileGroups(fn: DxirFunction): DxirFunction {
    val groups = identifyTileGroups(fn)
    if (groups.isEmpty()) return fn
    val opIdToGroup = HashMap<Int, Int>()
    for (g in groups) for (id in g.opIds) opIdToGroup[id] = g.groupId

    return DxirBuilder.function(fn.name) {
        val nodeMap = HashMap<Int, DxirNode>()
        for (p in fn.params) {
            nodeMap[p.id] = param(p.name, p.type, p.sharding)
        }
        for (node in fn.body) {
            when (node) {
                is DxirConst -> {
                    nodeMap[node.id] = const(node.value, node.type, node.sharding)
                }
                is DxirOp -> {
                    require(!node.isMultiResult) {
                        "annotateTileGroups: multi-result op (id=${node.id}) out of v1 scope"
                    }
                    require(node.regions.isEmpty()) {
                        "annotateTileGroups: region-bearing op (id=${node.id}) out of v1 scope"
                    }
                    val operands = node.operands.map {
                        require(it !is DxirOpResult) {
                            "annotateTileGroups: DxirOpResult operand out of v1 scope"
                        }
                        nodeMap[it.id]
                            ?: error("annotateTileGroups: operand id=${it.id} not in nodeMap")
                    }
                    val groupId = opIdToGroup[node.id]
                    val attrs = if (groupId != null) {
                        node.attrs + mapOf("tile_group" to groupId)
                    } else {
                        node.attrs
                    }
                    val newOp = op(node.op, operands, node.type, attrs, node.sharding)
                    nodeMap[node.id] = newOp
                }
                else -> error("annotateTileGroups: unsupported body node $node")
            }
        }
        fn.returns.map {
            nodeMap[it.id] ?: error("annotateTileGroups: return ref id=${it.id} not in nodeMap")
        }
    }
}
