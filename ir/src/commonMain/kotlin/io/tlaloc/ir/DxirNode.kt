package io.tlaloc.ir

sealed class DxirNode {
    abstract val type: DxirType
    abstract val id: Int
    abstract val sharding: DxirSharding?
}

class DxirParam(
    override val id: Int,
    val name: String,
    override val type: DxirType,
    override val sharding: DxirSharding? = null,
) : DxirNode()

class DxirConst(
    override val id: Int,
    val value: Any,
    override val type: DxirType,
    override val sharding: DxirSharding? = null,
) : DxirNode()

class DxirOp(
    override val id: Int,
    val op: OpKind,
    val operands: List<DxirNode>,
    val attrs: Map<String, Any> = emptyMap(),
    val types: List<DxirType>,
    override val sharding: DxirSharding? = null,
    /** Nested regions (empty for most ops; used by MANUAL_COMPUTATION, future If/While/Scan). */
    val regions: List<DxirRegion> = emptyList(),
) : DxirNode() {

    init {
        require(types.isNotEmpty()) { "DxirOp must have at least one result type" }
    }

    /**
     * First result type. Multi-result ops can still access this (it returns types[0]),
     * but should prefer [result] + [DxirOpResult] references for non-zero indices.
     */
    override val type: DxirType get() = types.first()

    val isMultiResult: Boolean get() = types.size > 1
    val numResults: Int get() = types.size
    val hasRegions: Boolean get() = regions.isNotEmpty()

    /** Reference to the k-th result. For k=0 on single-result ops, returns `this` directly. */
    fun result(k: Int): DxirNode {
        require(k in types.indices) {
            "result index $k out of bounds for op with $numResults outputs"
        }
        return if (!isMultiResult && k == 0) this else DxirOpResult(this, k)
    }

    /** Backward-compat single-result constructor: `DxirOp(id, op, operands, attrs, type)`. */
    constructor(
        id: Int,
        op: OpKind,
        operands: List<DxirNode>,
        attrs: Map<String, Any> = emptyMap(),
        type: DxirType,
        sharding: DxirSharding? = null,
    ) : this(id, op, operands, attrs, listOf(type), sharding, emptyList())
}

/**
 * Block argument — a typed SSA value introduced by a region's entry block. Behaves as a
 * DxirNode so it can be referenced by body ops within the region.
 */
class DxirBlockArg(
    override val id: Int,
    override val type: DxirType,
    override val sharding: DxirSharding? = null,
) : DxirNode()

/**
 * A block in a region. Has typed [args] (SSA values local to the block), a [body] of ops,
 * and a [terminator] list specifying the values yielded by the block — analogous to
 * `DxirFunction.returns` but scoped to this block.
 */
class DxirBlock(
    val args: List<DxirBlockArg>,
    val body: List<DxirNode>,
    val terminator: List<DxirNode>,
) {
    /** Returns the set of ids declared by this block (args + ops + nested declarations). */
    fun declaredIds(): Set<Int> {
        val ids = HashSet<Int>(args.size + body.size)
        for (a in args) ids += a.id
        for (node in body) {
            ids += node.id
            if (node is DxirOp) for (r in node.regions) for (b in r.blocks) ids += b.declaredIds()
        }
        return ids
    }
}

/** A region containing one or more blocks. v0: single-block regions (structured control flow). */
class DxirRegion(val blocks: List<DxirBlock>) {
    init { require(blocks.isNotEmpty()) { "region must have at least one block" } }
    val entryBlock: DxirBlock get() = blocks.first()

    companion object {
        /** Convenience: build a single-block region. */
        fun ofBlock(
            args: List<DxirBlockArg>,
            body: List<DxirNode>,
            terminator: List<DxirNode>,
        ): DxirRegion = DxirRegion(listOf(DxirBlock(args, body, terminator)))
    }
}

/**
 * A handle to a specific result of a multi-output op — equivalent to MLIR's `%op#k`.
 * Shares the source op's [id]; not independently declared in [DxirFunction.body].
 */
class DxirOpResult(
    val source: DxirOp,
    val index: Int,
) : DxirNode() {

    init {
        require(index in source.types.indices) {
            "result index $index out of bounds for op with ${source.numResults} outputs"
        }
    }

    override val id: Int get() = source.id
    override val type: DxirType get() = source.types[index]
    override val sharding: DxirSharding? get() = source.sharding
}

class DxirCall(
    override val id: Int,
    val callee: DxirFunction,
    val args: List<DxirNode>,
    override val type: DxirType,
    override val sharding: DxirSharding? = null,
) : DxirNode()
