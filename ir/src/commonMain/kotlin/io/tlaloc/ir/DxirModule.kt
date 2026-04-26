package io.tlaloc.ir

class DxirFunction(
    val name: String,
    val params: List<DxirParam>,
    val body: List<DxirNode>,
    val returns: List<DxirNode>,
    val meshes: List<DxirMesh> = emptyList(),
) {
    init {
        // Validate structured-control-flow op shapes (IF / WHILE) before SSA ref-checking,
        // so a malformed region surfaces a precise diagnostic instead of cascading into
        // an unknown-id error.
        validateControlFlowShapes(body)

        // Collect all declared ids: params + body ops + anything inside nested regions.
        val declared = HashSet<Int>(params.size + body.size)
        for (p in params) declared += p.id
        for (node in body) {
            declared += node.id
            if (node is DxirOp) for (r in node.regions) for (b in r.blocks) declared += b.declaredIds()
        }

        // References include top-level body op operands + returns + nested-region references.
        val referenced = returns.map { it.id } +
            collectReferencedIds(body)

        val missing = referenced.filter { it !in declared }
        if (missing.isNotEmpty()) {
            // §0.4.172 — augment the validation error with a partial-state dump
            // (full body listing) so downstream diagnostics can find which op
            // references the unknown id without re-running the build path.
            // Multiple call sites construct DxirFunction directly (the post-passes
            // dropUnreachableBody / applyCSE / applyConstFold in DxirReverseTransform);
            // augmenting at the DxirFunction.init level catches them all uniformly.
            throw IllegalArgumentException(
                "function $name references unknown node ids: $missing\n" +
                    renderBodyDump(name, params, body, returns),
            )
        }

        val meshNames = meshes.map { it.name }.toSet()
        val shardings = (params + body).mapNotNull { it.sharding }
        val unknownMeshes = shardings.map { it.meshName }.filter { it !in meshNames }.toSet()
        require(unknownMeshes.isEmpty()) {
            "function $name references undeclared meshes: $unknownMeshes"
        }
    }

    private fun validateControlFlowShapes(nodes: List<DxirNode>) {
        for (node in nodes) {
            if (node !is DxirOp) continue
            when (node.op) {
                OpKind.IF -> validateIfShape(node)
                OpKind.WHILE -> validateWhileShape(node)
                OpKind.COARSENED -> validateCoarsenedShape(node)
                else -> {}
            }
            // Recurse into nested regions of any op (covers IF/WHILE inside MANUAL_COMPUTATION etc.)
            for (r in node.regions) for (b in r.blocks) validateControlFlowShapes(b.body)
        }
    }

    /**
     * §0.4.31 — shape check for the SOI splice op. The COARSENED op carries a
     * nested [DxirFunction] as `attrs["primal_body"]`; its operands must align
     * positionally with that function's params, and its result types must align
     * with the primal's returns. [attrs["gradient_body"]] + [attrs["reads_primal_indices"]]
     * are required at construction time — C.3b.2 consumes both during reverse-mode.
     */
    private fun validateCoarsenedShape(op: DxirOp) {
        val primal = op.attrs["primal_body"]
        require(primal is DxirFunction) {
            "function $name: COARSENED op id=${op.id} requires `primal_body` attr of type DxirFunction; got ${primal?.let { it::class.simpleName }}"
        }
        val gradient = op.attrs["gradient_body"]
        require(gradient is DxirFunction) {
            "function $name: COARSENED op id=${op.id} requires `gradient_body` attr of type DxirFunction; got ${gradient?.let { it::class.simpleName }}"
        }
        val reads = op.attrs["reads_primal_indices"]
        require(reads is Set<*> && reads.all { it is Int }) {
            "function $name: COARSENED op id=${op.id} requires `reads_primal_indices` attr of type Set<Int>; got ${reads?.let { it::class.simpleName }}"
        }
        @Suppress("UNCHECKED_CAST")
        val readsInt = reads as Set<Int>
        require(op.operands.size == primal.params.size) {
            "function $name: COARSENED op id=${op.id} operand count ${op.operands.size} ≠ primal_body.params count ${primal.params.size}"
        }
        require(op.types.size == primal.returns.size) {
            "function $name: COARSENED op id=${op.id} result count ${op.types.size} ≠ primal_body.returns count ${primal.returns.size}"
        }
        for ((i, p) in primal.params.withIndex()) {
            require(op.operands[i].type == p.type) {
                "function $name: COARSENED op id=${op.id} operand[$i].type=${op.operands[i].type} ≠ primal_body.params[$i].type=${p.type}"
            }
        }
        for ((i, r) in primal.returns.withIndex()) {
            require(op.types[i] == r.type) {
                "function $name: COARSENED op id=${op.id} types[$i]=${op.types[i]} ≠ primal_body.returns[$i].type=${r.type}"
            }
        }
        // §0.4.179 — Phase 5c: Gradient body signature widened to multi-result.
        //   single-result (K=1): `(upstream, *primal_operands) → (d_operand_0, ...)` (= 1 + N params).
        //   multi-result (K>1):  `(upstream_0, …, upstream_K-1, *primal_operands) → (d_operand_0, ...)` (= K + N params).
        // The first K params are upstreams (one per result type, in order); the next N
        // params are primal operands (positionally aligned with op.operands). N returns
        // give per-operand gradient contributions. K==1 stays bit-exact equivalent to
        // the §0.4.31 contract — single-result COARSENED tests don't change.
        val k = op.types.size
        require(gradient.params.size == k + op.operands.size) {
            "function $name: COARSENED op id=${op.id} gradient_body.params count ${gradient.params.size} ≠ K + N (= $k + ${op.operands.size}) — K upstreams + N primal operands"
        }
        for ((i, t) in op.types.withIndex()) {
            require(gradient.params[i].type == t) {
                "function $name: COARSENED op id=${op.id} gradient_body.params[$i] (upstream for result $i) type ${gradient.params[i].type} ≠ op.types[$i]=$t"
            }
        }
        for ((i, operand) in op.operands.withIndex()) {
            require(gradient.params[k + i].type == operand.type) {
                "function $name: COARSENED op id=${op.id} gradient_body.params[${k + i}] (primal operand $i) type=${gradient.params[k + i].type} ≠ operand[$i].type=${operand.type}"
            }
        }
        require(gradient.returns.size == op.operands.size) {
            "function $name: COARSENED op id=${op.id} gradient_body.returns count ${gradient.returns.size} ≠ operand count ${op.operands.size}"
        }
        // Every index in reads_primal_indices must be a valid operand index.
        for (i in readsInt) require(i in op.operands.indices) {
            "function $name: COARSENED op id=${op.id} reads_primal_indices contains out-of-range index $i (operand count=${op.operands.size})"
        }
    }

    private fun validateIfShape(op: DxirOp) {
        require(op.operands.size == 1) {
            "function $name: IF op id=${op.id} requires exactly 1 operand (the predicate); got ${op.operands.size}"
        }
        val predType = op.operands[0].type
        require(predType.dtype == io.tlaloc.core.Bool && predType.isScalar) {
            "function $name: IF op id=${op.id} predicate must be a scalar Bool; got $predType"
        }
        require(op.regions.size == 2) {
            "function $name: IF op id=${op.id} requires exactly 2 regions [then, else]; got ${op.regions.size}"
        }
        for ((idx, r) in op.regions.withIndex()) {
            val branch = if (idx == 0) "then" else "else"
            require(r.blocks.size == 1) {
                "function $name: IF op id=${op.id} $branch-region must be single-block; got ${r.blocks.size}"
            }
            val block = r.blocks.single()
            require(block.args.isEmpty()) {
                "function $name: IF op id=${op.id} $branch-region block must have no args; got ${block.args.size}"
            }
            require(block.terminator.size == op.types.size) {
                "function $name: IF op id=${op.id} $branch-region terminator arity ${block.terminator.size} ≠ op.types arity ${op.types.size}"
            }
            for ((tIdx, term) in block.terminator.withIndex()) {
                require(term.type == op.types[tIdx]) {
                    "function $name: IF op id=${op.id} $branch-region terminator[$tIdx] type ${term.type} ≠ op.types[$tIdx] ${op.types[tIdx]}"
                }
            }
        }
    }

    private fun validateWhileShape(op: DxirOp) {
        require(op.regions.size == 2) {
            "function $name: WHILE op id=${op.id} requires exactly 2 regions [cond, body]; got ${op.regions.size}"
        }
        require(op.types.size == op.operands.size) {
            "function $name: WHILE op id=${op.id} types arity ${op.types.size} ≠ operands arity ${op.operands.size} (every loop-carried value must have a result type)"
        }
        for ((i, t) in op.types.withIndex()) {
            require(t == op.operands[i].type) {
                "function $name: WHILE op id=${op.id} types[$i]=$t ≠ operands[$i].type=${op.operands[i].type} (loop-carried types must match init types)"
            }
        }
        // Cond region: N args matching operand types, terminator is one Bool scalar.
        val condRegion = op.regions[0]
        require(condRegion.blocks.size == 1) {
            "function $name: WHILE op id=${op.id} cond-region must be single-block; got ${condRegion.blocks.size}"
        }
        val condBlock = condRegion.blocks.single()
        require(condBlock.args.size == op.operands.size) {
            "function $name: WHILE op id=${op.id} cond-region block must have ${op.operands.size} args (one per loop-carried value); got ${condBlock.args.size}"
        }
        for ((i, a) in condBlock.args.withIndex()) {
            require(a.type == op.operands[i].type) {
                "function $name: WHILE op id=${op.id} cond-region args[$i].type=${a.type} ≠ operands[$i].type=${op.operands[i].type}"
            }
        }
        require(condBlock.terminator.size == 1) {
            "function $name: WHILE op id=${op.id} cond-region terminator must yield exactly 1 value (the predicate); got ${condBlock.terminator.size}"
        }
        val predTermType = condBlock.terminator.single().type
        require(predTermType.dtype == io.tlaloc.core.Bool && predTermType.isScalar) {
            "function $name: WHILE op id=${op.id} cond-region terminator must be a scalar Bool predicate; got $predTermType"
        }
        // Body region: N args matching operand types, terminator yields N values matching op.types.
        val bodyRegion = op.regions[1]
        require(bodyRegion.blocks.size == 1) {
            "function $name: WHILE op id=${op.id} body-region must be single-block; got ${bodyRegion.blocks.size}"
        }
        val bodyBlock = bodyRegion.blocks.single()
        require(bodyBlock.args.size == op.operands.size) {
            "function $name: WHILE op id=${op.id} body-region block must have ${op.operands.size} args (one per loop-carried value); got ${bodyBlock.args.size}"
        }
        for ((i, a) in bodyBlock.args.withIndex()) {
            require(a.type == op.operands[i].type) {
                "function $name: WHILE op id=${op.id} body-region args[$i].type=${a.type} ≠ operands[$i].type=${op.operands[i].type}"
            }
        }
        require(bodyBlock.terminator.size == op.types.size) {
            "function $name: WHILE op id=${op.id} body-region terminator arity ${bodyBlock.terminator.size} ≠ op.types arity ${op.types.size}"
        }
        for ((i, term) in bodyBlock.terminator.withIndex()) {
            require(term.type == op.types[i]) {
                "function $name: WHILE op id=${op.id} body-region terminator[$i] type ${term.type} ≠ op.types[$i] ${op.types[i]}"
            }
        }
    }

    private fun collectReferencedIds(nodes: List<DxirNode>): List<Int> {
        val refs = mutableListOf<Int>()
        for (node in nodes) {
            when (node) {
                is DxirOp -> {
                    refs += node.operands.map { it.id }
                    for (r in node.regions) for (b in r.blocks) {
                        refs += collectReferencedIds(b.body)
                        refs += b.terminator.map { it.id }
                    }
                }
                is DxirCall -> refs += node.args.map { it.id }
                else -> {}
            }
        }
        return refs
    }
}

/**
 * §0.4.172 — pretty-print a [DxirFunction]'s in-progress params + body + returns
 * as a flat listing for diagnostic dumps when validation fails. Used by
 * [DxirFunction]'s init when the SSA-id check fails — letting the caller see
 * which op references the unknown id without re-running the build path. Top-level
 * private (rather than a member) so it can render the partial state without
 * needing a fully-constructed [DxirFunction] instance.
 */
private fun renderBodyDump(
    name: String,
    params: List<DxirParam>,
    body: List<DxirNode>,
    returns: List<DxirNode>,
): String = buildString {
    appendLine("partial function dump: name=$name")
    appendLine("params: ${params.map { "%${it.id}=${it.name}:${it.type}" }}")
    appendLine("body (${body.size} ops):")
    for (n in body) {
        when (n) {
            is DxirOp -> appendLine(
                "  %${n.id} = ${n.op} ops=${n.operands.map { "%${it.id}" }} types=${n.types}",
            )
            is DxirConst -> appendLine("  %${n.id} = const ${n.value} : ${n.type}")
            is DxirCall -> appendLine(
                "  %${n.id} = call ${n.callee.name} args=${n.args.map { "%${it.id}" }}",
            )
            else -> appendLine("  %${n.id} = ${n::class.simpleName}")
        }
    }
    appendLine("returns: ${returns.map { "%${it.id}" }}")
}

class DxirModule(
    val functions: List<DxirFunction>,
    val meshes: List<DxirMesh> = emptyList(),
) {
    fun function(name: String): DxirFunction? = functions.firstOrNull { it.name == name }
    fun mesh(name: String): DxirMesh? = meshes.firstOrNull { it.name == name }
}

/**
 * Common emit surface shared by [DxirBuilder] (function body) and [DxirRegionBuilder]
 * (nested region body). Lets pass-writers (e.g., [io.tlaloc.ir.passes.PhiCalculus])
 * write builder-agnostic helper functions that emit ops into either context. Stage B.1
 * needs this for clone-and-rewrite walkers that descend into IF/WHILE regions.
 *
 * Contract: every method has the same signature on both implementations and produces a
 * node that lands in the appropriate scope's body list.
 */
interface DxirEmitter {
    fun op(
        kind: OpKind,
        operands: List<DxirNode>,
        type: DxirType,
        attrs: Map<String, Any> = emptyMap(),
        sharding: DxirSharding? = null,
        regions: List<DxirRegion> = emptyList(),
    ): DxirOp

    fun opMulti(
        kind: OpKind,
        operands: List<DxirNode>,
        types: List<DxirType>,
        attrs: Map<String, Any> = emptyMap(),
        sharding: DxirSharding? = null,
        regions: List<DxirRegion> = emptyList(),
    ): DxirOp

    fun const(value: Any, type: DxirType, sharding: DxirSharding? = null): DxirConst

    /**
     * §0.4.50 — lifted from [DxirBuilder] to support nested control flow (e.g.,
     * `for (k) { for (i) { ... } }` or `for in if`). Both builders support the same
     * signature; previously `whileOp` was only on [DxirBuilder], forcing FIR lowering
     * to cast emitters and fail on nested for-loops.
     */
    fun whileOp(
        inits: List<DxirNode>,
        cond: DxirRegionBuilder.(args: List<DxirBlockArg>) -> Unit,
        body: DxirRegionBuilder.(args: List<DxirBlockArg>) -> Unit,
    ): DxirOp
}

class DxirBuilder private constructor() : DxirEmitter {
    private var nextId = 0
    private val params = mutableListOf<DxirParam>()
    private val body = mutableListOf<DxirNode>()
    private val meshes = mutableListOf<DxirMesh>()

    /** Allocates a fresh SSA id. Exposed for [DxirRegionBuilder] which shares our id space. */
    internal fun allocateId(): Int = nextId++

    fun declareMesh(mesh: DxirMesh): DxirMesh {
        require(meshes.none { it.name == mesh.name }) { "mesh '${mesh.name}' already declared" }
        meshes += mesh
        return mesh
    }

    fun param(name: String, type: DxirType, sharding: DxirSharding? = null): DxirParam =
        DxirParam(allocateId(), name, type, sharding).also { params += it }

    override fun const(value: Any, type: DxirType, sharding: DxirSharding?): DxirConst =
        DxirConst(allocateId(), value, type, sharding).also { body += it }

    override fun op(
        kind: OpKind,
        operands: List<DxirNode>,
        type: DxirType,
        attrs: Map<String, Any>,
        sharding: DxirSharding?,
        regions: List<DxirRegion>,
    ): DxirOp = DxirOp(allocateId(), kind, operands, attrs, listOf(type), sharding, regions)
        .also { body += it }

    /** Multi-result op. Access results via [DxirOp.result]. */
    override fun opMulti(
        kind: OpKind,
        operands: List<DxirNode>,
        types: List<DxirType>,
        attrs: Map<String, Any>,
        sharding: DxirSharding?,
        regions: List<DxirRegion>,
    ): DxirOp {
        require(types.isNotEmpty()) { "opMulti requires at least one result type" }
        return DxirOp(allocateId(), kind, operands, attrs, types, sharding, regions)
            .also { body += it }
    }

    fun call(
        callee: DxirFunction,
        args: List<DxirNode>,
        type: DxirType,
        sharding: DxirSharding? = null,
    ): DxirCall = DxirCall(allocateId(), callee, args, type, sharding).also { body += it }

    /** Build a single-block region whose ids allocate from this builder's id space. */
    fun region(block: DxirRegionBuilder.() -> Unit): DxirRegion {
        val rb = DxirRegionBuilder(this)
        rb.block()
        return rb.finish()
    }

    /**
     * [OpKind.IF] convenience builder — boolean-scalar predicate operand + two regions
     * `[then, else]` whose single blocks each have **no args** and yield `types.size`-many
     * values. Single-result IFs return a [DxirOp]; multi-result IFs are also valid (use
     * [DxirOp.result] to extract individual outputs). See plan §3.1.1 for the full shape.
     */
    fun ifOp(
        cond: DxirNode,
        types: List<DxirType>,
        thenRegion: DxirRegion,
        elseRegion: DxirRegion,
    ): DxirOp = DxirOp(
        id = allocateId(),
        op = OpKind.IF,
        operands = listOf(cond),
        attrs = emptyMap(),
        types = types,
        sharding = null,
        regions = listOf(thenRegion, elseRegion),
    ).also { body += it }

    /**
     * [OpKind.WHILE] convenience builder — N loop-carried operands + two regions
     * `[cond, body]` whose single blocks each take N args matching operand types.
     * The condRegion's terminator yields one boolean scalar (loop predicate); the
     * bodyRegion's terminator yields N values matching the operand types.
     *
     * The cond / body lambdas receive the freshly-allocated block-arg list, so users
     * can wire arg references into the region body without manually calling [arg].
     *
     * Single back-edge only — no `break` / multi-back-edge form (deferred per plan §4.6).
     * The op is multi-result with `types == operands.map { it.type }`; access individual
     * loop-carried results via [DxirOp.result].
     */
    override fun whileOp(
        inits: List<DxirNode>,
        cond: DxirRegionBuilder.(args: List<DxirBlockArg>) -> Unit,
        body: DxirRegionBuilder.(args: List<DxirBlockArg>) -> Unit,
    ): DxirOp {
        val condRegion = region {
            val args = inits.map { arg(it.type) }
            cond(args)
        }
        val bodyRegion = region {
            val args = inits.map { arg(it.type) }
            body(args)
        }
        return DxirOp(
            id = allocateId(),
            op = OpKind.WHILE,
            operands = inits,
            attrs = emptyMap(),
            types = inits.map { it.type },
            sharding = null,
            regions = listOf(condRegion, bodyRegion),
        ).also { this.body += it }
    }

    /**
     * §0.4.31 — convenience builder for [OpKind.COARSENED]. The SOI splice op
     * carries a nested [DxirFunction] as `primal_body` + gradient-body for
     * pre-computed VJP splicing (C.3b.2). Operand types must match `primal_body.params`
     * positionally; result types are derived from `primal_body.returns`. See
     * docs/STAGE_B_PLAN.md §3.5 for the splice contract.
     */
    fun coarsened(
        operands: List<DxirNode>,
        primalBody: DxirFunction,
        gradientBody: DxirFunction,
        readsPrimalIndices: Set<Int>,
    ): DxirOp {
        val types = primalBody.returns.map { it.type }
        require(types.isNotEmpty()) { "COARSENED op requires primal_body with at least one return" }
        val attrs: Map<String, Any> = mapOf(
            "primal_body" to primalBody,
            "gradient_body" to gradientBody,
            "reads_primal_indices" to readsPrimalIndices,
        )
        return DxirOp(
            id = allocateId(),
            op = OpKind.COARSENED,
            operands = operands,
            attrs = attrs,
            types = types,
            sharding = null,
            regions = emptyList(),
        ).also { body += it }
    }

    fun build(name: String, returns: List<DxirNode>): DxirFunction =
        DxirFunction(name, params.toList(), body.toList(), returns, meshes.toList())

    companion object {
        fun function(name: String, block: DxirBuilder.() -> List<DxirNode>): DxirFunction {
            val b = DxirBuilder()
            val returns = b.block()
            return b.build(name, returns)
        }
    }
}

/**
 * Builder for a single-block region. Shares the outer [DxirBuilder]'s id counter so all SSA
 * values (outer + nested) are unique. Ops declared here land in the region's body, not outer.
 */
class DxirRegionBuilder internal constructor(private val outer: DxirBuilder) : DxirEmitter {
    private val args = mutableListOf<DxirBlockArg>()
    private val body = mutableListOf<DxirNode>()
    private var yields: List<DxirNode> = emptyList()

    fun arg(type: DxirType, sharding: DxirSharding? = null): DxirBlockArg =
        DxirBlockArg(outer.allocateId(), type, sharding).also { args += it }

    /** Build a nested single-block region inside this region. Shares the outer id-space. */
    fun region(block: DxirRegionBuilder.() -> Unit): DxirRegion {
        val rb = DxirRegionBuilder(outer)
        rb.block()
        return rb.finish()
    }

    override fun op(
        kind: OpKind,
        operands: List<DxirNode>,
        type: DxirType,
        attrs: Map<String, Any>,
        sharding: DxirSharding?,
        regions: List<DxirRegion>,
    ): DxirOp = DxirOp(outer.allocateId(), kind, operands, attrs, listOf(type), sharding, regions)
        .also { body += it }

    override fun opMulti(
        kind: OpKind,
        operands: List<DxirNode>,
        types: List<DxirType>,
        attrs: Map<String, Any>,
        sharding: DxirSharding?,
        regions: List<DxirRegion>,
    ): DxirOp {
        require(types.isNotEmpty()) { "opMulti requires at least one result type" }
        return DxirOp(outer.allocateId(), kind, operands, attrs, types, sharding, regions)
            .also { body += it }
    }

    override fun const(value: Any, type: DxirType, sharding: DxirSharding?): DxirConst =
        DxirConst(outer.allocateId(), value, type, sharding).also { body += it }

    /** [io.tlaloc.ir.passes.PhiCalculus]'s C3 helper builds nested IF inside a region. */
    fun ifOp(
        cond: DxirNode,
        types: List<DxirType>,
        thenRegion: DxirRegion,
        elseRegion: DxirRegion,
    ): DxirOp = DxirOp(
        id = outer.allocateId(),
        op = OpKind.IF,
        operands = listOf(cond),
        attrs = emptyMap(),
        types = types,
        sharding = null,
        regions = listOf(thenRegion, elseRegion),
    ).also { body += it }

    /**
     * §0.4.50 — nested WHILE inside a region (enables FIR lowering of nested
     * for-loops like BGDHyperOpt's Fig. 6 `for (k) { for (i) { … } }`). Same
     * semantics as [DxirBuilder.whileOp]; uses the shared id space via [outer]
     * and writes the constructed op into this region's body list.
     */
    override fun whileOp(
        inits: List<DxirNode>,
        cond: DxirRegionBuilder.(args: List<DxirBlockArg>) -> Unit,
        body: DxirRegionBuilder.(args: List<DxirBlockArg>) -> Unit,
    ): DxirOp {
        val condRegion = region {
            val args = inits.map { arg(it.type) }
            cond(args)
        }
        val bodyRegion = region {
            val args = inits.map { arg(it.type) }
            body(args)
        }
        return DxirOp(
            id = outer.allocateId(),
            op = OpKind.WHILE,
            operands = inits,
            attrs = emptyMap(),
            types = inits.map { it.type },
            sharding = null,
            regions = listOf(condRegion, bodyRegion),
        ).also { this.body += it }
    }

    /** Mark which SSA values this block yields (equivalent to `sdy.return` / `stablehlo.return`). */
    fun yields(vararg results: DxirNode) {
        yields = results.toList()
    }

    internal fun finish(): DxirRegion =
        DxirRegion.ofBlock(args.toList(), body.toList(), yields)
}
