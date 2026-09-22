package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirBlock
import io.tlaloc.ir.DxirBlockArg
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirCall
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirEmitter
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirRegionBuilder
import io.tlaloc.ir.DxirRegion
import io.tlaloc.ir.OpKind

/**
 * Stage B.1 — φ-calculus rewrite pass on dxir per docs/STAGE_B_PLAN.md §4 + §7.2.
 *
 * Implements F1 (identity), F3 (commutative canonicalisation), F2 / C1 (distributive
 * push of an op into an IF's branches), and C3 (nested-IF flattening). F4 is implicit
 * in [SymbolicEngine.nest] (used by C5–C9 in Stage B.2/B.3); F5 is trivial for
 * single-back-edge WHILE and is handled by Stage B.2's WHILE-targeted rewrites.
 * C2 (`d/dx φ(a,b) = φ(da/dx, db/dx)`) is realised by Stage A SCT running on the
 * post-PhiCalculus dxir; C4 (`f(φ(a,a)) = f(a)`) is the pipeline composition of
 * C1 then F1 (no standalone Kotlin function).
 *
 * ### Apply pipeline (per plan §4.13)
 *
 * 1. F1: identity collapse — `IF(p, x, x) ⇒ x`.
 * 2. F3: canonicalisation — swap branches + wrap predicate in NOT to put a deterministic
 *    branch order; this exposes more F1 collapse opportunities downstream.
 * 3. F2 / C1: distributive — push an outer op into both branches of an IF operand. F2
 *    is the unary/binary case; C1 is the k-ary generalisation (paper §4.7). Same
 *    Kotlin function handles both. Anti-swell gate per plan §4.3 — only fire when the
 *    IF has a single use OR both branch bodies are < 5 ops.
 * 4. F1 again — collapse identity branches that distribution may have exposed.
 * 5. C3: nested-IF flattening — `outerIF(c, innerIF(c2, a, b), z) ⇒ innerIF(c2, outerIF(c, a, z), outerIF(c, b, z))`.
 *    Distributes inner IF outward, swapping nesting. Useful when downstream consumers
 *    can fuse outerIFs sharing the same predicate (deferred CSE-style optimisation).
 * 6. F1 again.
 *
 * Iterates the whole pipeline to fixpoint (cap: 50 iterations) so chained rewrites
 * eventually converge. Most realistic primals fixpoint in 1-3 iterations.
 *
 * ### Implementation strategy
 *
 * Each pass clones the entire [DxirFunction] into a fresh [DxirBuilder], rewriting
 * matching ops in flight. Param + non-rewritten ops + nested regions are cloned
 * verbatim via a `nodeMap: Map<oldId, newNode>`. When a rewrite fires (e.g., F1
 * eliminates an IF), `nodeMap[oldIfId]` maps to the new node that replaces it
 * (e.g., the cloned then-branch yield). Subsequent ops that referenced the old IF
 * resolve through `nodeMap` to the replacement.
 *
 * Dead ops (e.g., the original elementwise op consumed by F2's distributive push)
 * are emitted but never referenced; a future Stage B.3 DCE pass strips them. The
 * [DxirInterpreter] tolerates dead nodes — it walks the body in program order,
 * caching results that aren't read.
 *
 * ### Scope (B.1)
 *
 * - **No SymbolicEngine calls.** All five rewrites are pure dxir-level structural
 *   transformations. The engine wires in for B.2 (C5) and B.3 (C6–C9).
 * - **No WHILE-targeted rewrites.** Stage B.2 lands C5; Stage B.3 lands C6–C9.
 * - **Single-back-edge IF only.** Multi-result IFs (per plan §3.1.1's region shape)
 *   are accepted by the rewrites but the F2 / C1 distribution path requires single-
 *   result; multi-result distribution is deferred.
 */
object PhiCalculus {

    /**
     * Hard cap on the apply-to-fixpoint loop. Catches a runaway rewrite (e.g., a
     * ping-pong where two rewrites undo each other) before it hangs the test suite.
     * Realistic primals converge in 1-3 iterations; 50 is overkill-safe.
     */
    const val FIXPOINT_CAP: Int = 50

    /**
     * Op kinds that are "elementwise enough" to participate in F2 / C1 distribution
     * into IF branches. Each op produces a result of the same dtype as its input(s)
     * (or, for binary ops, of the same dtype as both operands which must agree).
     * STEP / NOT are excluded because they take a numeric / Bool input and produce
     * a Bool output — pushing them into a branch yielding numeric values would
     * change the IF's result type, requiring the IF op to be re-typed; out of B.1 scope.
     */
    private val DISTRIBUTABLE_OPS: Set<OpKind> = setOf(
        OpKind.NEG, OpKind.ABS, OpKind.EXP, OpKind.LOG, OpKind.SQRT, OpKind.RSQRT,
        OpKind.TANH, OpKind.SIGMOID, OpKind.RELU, OpKind.GELU, OpKind.SILU,
        OpKind.TAN, OpKind.ATAN,
        // §0.4.405 — POLYGAMMA distributes too: the F2/C1 push-into-branch
        // clone carries `op.attrs` verbatim, so the `order` attr survives.
        OpKind.LGAMMA, OpKind.DIGAMMA, OpKind.TRIGAMMA, OpKind.POLYGAMMA,
        OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW,
    )

    /**
     * Apply the φ-calculus rewrite pipeline to [fn] until fixpoint. Returns the
     * rewritten function (numerically equivalent to [fn]; can be wider OR narrower in
     * op count depending on which rewrites fire).
     *
     * @param engine optional [SymbolicEngine] enabling C6/C7/C8/C9 (engine-backed
     *   closed-form rewrites for affine + power loop recurrences). If null, only the
     *   structural rewrites (F1/F3/F2/C1/C3) and the engine-free C5 (direct unroll for
     *   concrete-trip-count loops) fire — preserves backward-compatible behaviour for
     *   callers from `commonTest` that have no engine impl available.
     */
    /**
     * §0.4.103 — **D.1i Phase 1**. Apply Symja's `Simplify` to each return expression
     * of [fn], producing an equivalent function whose body has been algebraically
     * simplified by the symbolic engine. This is the paper's §6.1 mechanism (ii)
     * ("computation simplification thanks to the large-scoped symbolic
     * differentiation"), applied to whole gradient expressions output by
     * `DxirReverseTransform`.
     *
     * **Phase 1 scope** (this session): minimal lift→simplify→lower scaffolding for
     * arithmetic-only return expressions. Uses [SymbolicEngine.liftNode] (currently
     * supports DxirParam / DxirConst with integer values / arithmetic ops at the
     * scalar level), [SymbolicEngine.simplify] (Symja's `Simplify`), and
     * [SymbolicEngine.lowerToDxir] (with a per-param symbol map so free variables
     * in the simplified expression resolve back to the new function's params).
     *
     * **Bail-out semantics**: if any return fails to lift OR any simplified expression
     * fails to lower (e.g. the body contains a non-arithmetic op the engine doesn't
     * recognise, or a fractional Float constant the current `liftNode` lifts as the
     * truncated Long), this returns [fn] unchanged. Any future broadening of
     * `liftNode`'s coverage automatically widens what this pass can simplify, with
     * no callsite changes here.
     *
     * **Not yet wired** into `apply`'s pipeline. Callers must invoke this pass
     * explicitly. Phase 2 will integrate it into `TlalocIrGenerationExtension`
     * behind an opt-in system property; Phase 3+ will widen `liftNode` to handle
     * fractional constants and the non-arithmetic ops that gradient bodies for
     * tensor surfaces emit.
     */
    fun simplifyReturns(fn: DxirFunction, engine: SymbolicEngine): DxirFunction {
        // §0.4.107 — D.1i Phase 4. Lift each return through `liftReturnWithLeaves`,
        // which creates fresh Symja sentinel symbols for any DxirNode the engine's
        // base `liftNode` doesn't recognise (SUM, MEAN, MATMUL, GATHER, EXP, LOG,
        // ... and any DxirOpResult). Symja sees those sentinels as opaque variables
        // and can still simplify the surrounding arithmetic — `MUL(SUM(x), 1)` lifts
        // as `_leaf_<id> * 1`, simplifies to `_leaf_<id>`, and lowers back to the
        // cloned SUM op.
        val opaqueLeaves: MutableMap<String, DxirNode> = LinkedHashMap()
        val simplifiedExprs: List<SymExpr> = try {
            fn.returns.map { ret ->
                val lifted = liftReturnWithLeaves(ret, engine, opaqueLeaves)
                engine.simplify(lifted)
            }
        } catch (e: Throwable) {
            return fn
        }
        // Lower each simplified expression back to dxir under a freshly-built
        // function with new DxirParams (one per original param). The symbol map
        // routes lifted free variables back to the new params by name AND to
        // newly-cloned opaque-leaf subtrees by sentinel name. Only leaves the
        // simplified output actually references get cloned — unused ones (e.g.,
        // an opaque `MUL(SUM, 0)` collapsed to 0 by Simplify) leave no trace.
        return try {
            DxirBuilder.function(fn.name) {
                val symbolMap = HashMap<String, DxirNode>()
                val paramByName = HashMap<String, DxirNode>()
                for (origParam in fn.params) {
                    val newParam = param(origParam.name, origParam.type)
                    symbolMap[origParam.name] = newParam
                    paramByName[origParam.name] = newParam
                }
                val cloneCache = HashMap<Int, DxirNode>()
                for ((sentinel, origNode) in opaqueLeaves) {
                    val sentinelVar = engine.variable(sentinel)
                    val referenced = simplifiedExprs.any { engine.containsVariable(it, sentinelVar) }
                    if (!referenced) continue
                    symbolMap[sentinel] = cloneOpaqueSubtree(origNode, this, paramByName, cloneCache)
                }
                simplifiedExprs.zip(fn.returns).map { (sym, origReturn) ->
                    engine.lowerToDxir(sym, origReturn.type, this, symbolMap)
                }
            }
        } catch (e: Throwable) {
            // Lowering failure (Symja produced an op the lower-half can't emit, or
            // an opaque-leaf subtree fell outside the cloner's scope) → unchanged.
            fn
        }
    }

    /**
     * Phase-4 lifter for `simplifyReturns`. Recurses through arithmetic ops
     * (ADD/SUB/MUL/DIV/NEG/POW) using [SymbolicEngine] primitives; for everything
     * else, registers an opaque sentinel symbol in [leafMap] and returns
     * `engine.variable(sentinel)`. The sentinel name is `_simplify_leaf_<id>`,
     * unique per node.id and unlikely to collide with any real param name.
     *
     * This is intentionally a private helper rather than a [SymbolicEngine] method
     * because the opaque-leaf protocol is specific to `simplifyReturns`'s
     * lift-and-clone-back pipeline; engines used elsewhere (PhiCalculus.apply's
     * C5–C9 paths) carry their own [AffineRecurrencePattern.symOpaqueLeaves]
     * machinery and shouldn't share implementation with this surface.
     */
    private fun liftReturnWithLeaves(
        node: DxirNode,
        engine: SymbolicEngine,
        leafMap: MutableMap<String, DxirNode>,
    ): SymExpr = when (node) {
        is DxirParam -> engine.liftNode(node)
        is DxirConst -> engine.liftNode(node)
        is DxirOp -> when (node.op) {
            OpKind.ADD -> engine.add(
                liftReturnWithLeaves(node.operands[0], engine, leafMap),
                liftReturnWithLeaves(node.operands[1], engine, leafMap),
            )
            OpKind.SUB -> engine.sub(
                liftReturnWithLeaves(node.operands[0], engine, leafMap),
                liftReturnWithLeaves(node.operands[1], engine, leafMap),
            )
            OpKind.MUL -> engine.mul(
                liftReturnWithLeaves(node.operands[0], engine, leafMap),
                liftReturnWithLeaves(node.operands[1], engine, leafMap),
            )
            OpKind.DIV -> engine.div(
                liftReturnWithLeaves(node.operands[0], engine, leafMap),
                liftReturnWithLeaves(node.operands[1], engine, leafMap),
            )
            OpKind.NEG -> engine.neg(liftReturnWithLeaves(node.operands[0], engine, leafMap))
            OpKind.POW -> engine.pow(
                liftReturnWithLeaves(node.operands[0], engine, leafMap),
                liftReturnWithLeaves(node.operands[1], engine, leafMap),
            )
            else -> registerOpaqueLeaf(node, engine, leafMap)
        }
        else -> registerOpaqueLeaf(node, engine, leafMap)
    }

    private fun registerOpaqueLeaf(
        node: DxirNode,
        engine: SymbolicEngine,
        leafMap: MutableMap<String, DxirNode>,
    ): SymExpr {
        val sentinel = "_simplify_leaf_${node.id}"
        leafMap.putIfAbsent(sentinel, node)
        return engine.variable(sentinel)
    }

    /**
     * Clone an opaque-leaf subtree from the input function into [builder]'s scope.
     * Handles single-result `DxirOp` (no regions), `DxirConst`, `DxirParam` (resolved
     * via [paramByName]), and `DxirOpResult` (clone source + reindex). Multi-result
     * ops, region-bearing ops (IF / WHILE), and unrecognised node kinds throw, which
     * the outer [simplifyReturns] catches to bail out unchanged. This is deliberately
     * narrower than [cloneNode] — the existing C5/C6 paths handle the more elaborate
     * cases via their own machinery, while `simplifyReturns` only ever sees scalar
     * gradient bodies whose opaque leaves are typically straight-line tensor ops.
     */
    private fun cloneOpaqueSubtree(
        src: DxirNode,
        builder: DxirBuilder,
        paramByName: Map<String, DxirNode>,
        cache: MutableMap<Int, DxirNode>,
    ): DxirNode {
        cache[src.id]?.let { return it }
        val cloned: DxirNode = when (src) {
            is DxirParam -> paramByName[src.name]
                ?: error("simplifyReturns clone: param ${src.name} not present in new function's params")
            is DxirConst -> builder.const(src.value, src.type, src.sharding)
            is DxirOp -> {
                require(src.regions.isEmpty()) {
                    "simplifyReturns clone: opaque-leaf op ${src.op} carries regions; " +
                        "outer simplifyReturns will catch this and bail out unchanged"
                }
                require(!src.isMultiResult) {
                    "simplifyReturns clone: opaque-leaf op ${src.op} is multi-result; " +
                        "outer simplifyReturns will catch this and bail out unchanged"
                }
                val clonedOperands = src.operands.map { cloneOpaqueSubtree(it, builder, paramByName, cache) }
                builder.op(src.op, clonedOperands, src.type, src.attrs, src.sharding)
            }
            is DxirOpResult -> {
                val clonedSource = cloneOpaqueSubtree(src.source, builder, paramByName, cache)
                require(clonedSource is DxirOp) {
                    "simplifyReturns clone: DxirOpResult source did not clone to a DxirOp"
                }
                clonedSource.result(src.index)
            }
            else -> error("simplifyReturns clone: cannot clone node kind ${src::class.simpleName}")
        }
        cache[src.id] = cloned
        return cloned
    }

    /**
     * §0.4.174 — pre-SCT region-body lift pass. For each top-level [OpKind.IF] whose
     * regions have non-empty branch bodies, lifts the region-internal ops to the
     * function's top level just before the IF; the IF's regions are replaced with
     * empty-body regions yielding the lifted version of each terminator.
     *
     * **Why**: §0.4.173 traced a CartPole-shape SSA leak — coarsening's `distribute`
     * rule produces region-internal ops referencing OUTER-scope IFs as forward
     * operands (`%59 = MUL(%58, %57-OUTER-IF)` inside a sibling IF's branch).
     * [DxirReverseTransform.apply]'s cloning loop maps `nodeMap[primal-IF] = primal-IF`
     * (skipping the deep clone), so [walkBranchReverse]'s step-1 emits a top-level
     * cloned MUL whose `operand[1]` is the primal IF — leaking that primal id into
     * the grad body's SSA. Lifting the body ops to top level routes them through the
     * normal cloning path (which DOES rebuild operands through nodeMap), so the leak
     * doesn't fire. Plus the post-lift IF has empty regions, satisfying
     * [DxirToIrSynthesis.irIfOp]'s "branches yield outer-scope values only" gate.
     *
     * **Safety contract**: lifting is sound only when region body ops are total
     * functions — no DIV, SQRT, LOG, EXP that could fault when evaluated in a branch
     * the predicate doesn't select. This pass bails out (returns [fn] unchanged) if
     * the function contains:
     *
     *  - Any non-IF region-bearing op (WHILE, MANUAL_COMPUTATION, COARSENED with
     *    region attrs). Coarsening doesn't introduce these in the post-`distribute`
     *    shape, so this is rare in practice.
     *  - Any IF whose region body contains an op outside [SAFE_LIFT_OPS], a
     *    multi-result op, or a nested region-bearing op. Conservative: a single
     *    unsafe op disables the lift for the whole function rather than tracking
     *    per-IF safety, since partial transforms can leave the function in an
     *    inconsistent state.
     *
     * **Idempotent**: re-running the pass on an already-lifted function is a no-op
     * (no IF has non-empty body to lift).
     */
    fun liftIfRegionBodies(fn: DxirFunction): DxirFunction {
        if (!isLiftSafe(fn)) return fn
        val anyLift = fn.body.any { n ->
            n is DxirOp && n.op == OpKind.IF &&
                n.regions.any { r -> r.blocks.any { it.body.isNotEmpty() } }
        }
        if (!anyLift) return fn
        return DxirBuilder.function(fn.name) {
            val nodeMap = HashMap<Int, DxirNode>()
            for (p in fn.params) nodeMap[p.id] = param(p.name, p.type, p.sharding)
            for (n in fn.body) {
                val cloned = cloneTopLevelForLift(n, nodeMap, this)
                nodeMap[n.id] = cloned
            }
            fn.returns.map { resolveLiftedRef(it, nodeMap) }
        }
    }

    private val SAFE_LIFT_OPS: Set<OpKind> = setOf(
        OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.NEG,
        OpKind.STEP, OpKind.SIN, OpKind.COS, OpKind.ABS, OpKind.RELU,
        OpKind.NOT, OpKind.LAND, OpKind.CAST,
    )

    private fun isLiftSafe(fn: DxirFunction): Boolean {
        for (n in fn.body) {
            if (n !is DxirOp) continue
            if (n.op == OpKind.IF) {
                if (!isIfLiftSafe(n)) return false
            } else if (n.regions.isNotEmpty()) {
                // Non-IF region-bearing op (WHILE, MANUAL_COMPUTATION, etc.). The
                // lift pass doesn't touch these — bail out for the whole function.
                return false
            }
        }
        return true
    }

    private fun isIfLiftSafe(ifOp: DxirOp): Boolean {
        if (ifOp.regions.size != 2) return false
        for (region in ifOp.regions) {
            if (region.blocks.size != 1) return false
            val block = region.blocks.single()
            for (n in block.body) {
                when (n) {
                    is DxirConst -> {}
                    is DxirOp -> {
                        if (n.op !in SAFE_LIFT_OPS) return false
                        if (n.hasRegions) return false
                        if (n.isMultiResult) return false
                    }
                    else -> return false
                }
            }
        }
        return true
    }

    private fun resolveLiftedRef(node: DxirNode, nodeMap: Map<Int, DxirNode>): DxirNode {
        val mapped = nodeMap[node.id]
            ?: error("liftIfRegionBodies: id ${node.id} not in nodeMap")
        if (node !is DxirOpResult) return mapped
        return if (mapped is DxirOp) mapped.result(node.index) else mapped
    }

    private fun cloneTopLevelForLift(
        n: DxirNode,
        nodeMap: MutableMap<Int, DxirNode>,
        builder: DxirBuilder,
    ): DxirNode {
        return when (n) {
            is DxirConst -> builder.const(n.value, n.type, n.sharding)
            is DxirOp -> {
                val resolvedOperands = n.operands.map { resolveLiftedRef(it, nodeMap) }
                val hasNonEmptyRegion = n.regions.any { r -> r.blocks.any { it.body.isNotEmpty() } }
                if (n.op == OpKind.IF && hasNonEmptyRegion) {
                    liftIfOpBody(n, resolvedOperands, nodeMap, builder)
                } else if (n.regions.isNotEmpty()) {
                    // Empty-body IF (or other region-bearing op already filtered by
                    // isLiftSafe). Rebuild regions so terminators that reference outer
                    // scope route through the new nodeMap.
                    val rewrittenRegions = n.regions.map { region ->
                        rewriteIfRegionForLift(region, nodeMap, builder)
                    }
                    if (n.types.size == 1) {
                        builder.op(n.op, resolvedOperands, n.type, n.attrs, n.sharding, rewrittenRegions)
                    } else {
                        builder.opMulti(n.op, resolvedOperands, n.types, n.attrs, n.sharding, rewrittenRegions)
                    }
                } else {
                    if (n.types.size == 1) {
                        builder.op(n.op, resolvedOperands, n.type, n.attrs, n.sharding, emptyList())
                    } else {
                        builder.opMulti(n.op, resolvedOperands, n.types, n.attrs, n.sharding, emptyList())
                    }
                }
            }
            else -> error("liftIfRegionBodies: unsupported top-level node ${n::class.simpleName}")
        }
    }

    private fun liftIfOpBody(
        ifOp: DxirOp,
        resolvedOperands: List<DxirNode>,
        outerNodeMap: MutableMap<Int, DxirNode>,
        builder: DxirBuilder,
    ): DxirOp {
        val newRegions = ifOp.regions.map { region ->
            val block = region.blocks.single()
            val regionMap = HashMap<Int, DxirNode>(outerNodeMap)
            for (n in block.body) {
                val lifted: DxirNode = when (n) {
                    is DxirConst -> builder.const(n.value, n.type, n.sharding)
                    is DxirOp -> {
                        val operands = n.operands.map { resolveLiftedRef(it, regionMap) }
                        builder.op(n.op, operands, n.type, n.attrs, n.sharding)
                    }
                    else -> error("liftIfOpBody: unexpected ${n::class.simpleName}")
                }
                regionMap[n.id] = lifted
            }
            builder.region {
                val terms = block.terminator.map { resolveLiftedRef(it, regionMap) }
                yields(*terms.toTypedArray())
            }
        }
        return if (ifOp.types.size == 1) {
            builder.op(OpKind.IF, resolvedOperands, ifOp.type, ifOp.attrs, ifOp.sharding, newRegions)
        } else {
            builder.opMulti(OpKind.IF, resolvedOperands, ifOp.types, ifOp.attrs, ifOp.sharding, newRegions)
        }
    }

    /**
     * Rebuild a region whose body is empty (single terminator using outer-scope refs).
     * Block args are not supported — IFs don't take any (per dxir contract).
     */
    private fun rewriteIfRegionForLift(
        region: DxirRegion,
        outerNodeMap: Map<Int, DxirNode>,
        builder: DxirBuilder,
    ): DxirRegion {
        require(region.blocks.size == 1) {
            "liftIfRegionBodies: rewriteIfRegionForLift expects single-block region"
        }
        val block = region.blocks.single()
        require(block.args.isEmpty()) {
            "liftIfRegionBodies: IF regions don't take block args; got ${block.args.size}"
        }
        require(block.body.isEmpty()) {
            "liftIfRegionBodies: this rewriter is for empty-body regions only " +
                "(${block.body.size} body ops); use liftIfOpBody for non-empty bodies"
        }
        return builder.region {
            val terms = block.terminator.map { resolveLiftedRef(it, outerNodeMap) }
            yields(*terms.toTypedArray())
        }
    }

    /**
     * §0.4.503 (Tier 3, item 3) — **does this function still contain a loop?**
     *
     * Engine-free, side-effect-free, and the whole reason it exists: with Symja
     * turned into an optional dependency, a caller needs to be able to tell the
     * difference between "coarsening finished" and "coarsening finished as far as it
     * could without a computer algebra system". The engine-backed corollaries
     * C6–C9 are the only rewrites that close a WHILE the engine-free C5 unroll
     * cannot, so a WHILE surviving [apply] is exactly the condition under which
     * the absence of the CAS became visible.
     *
     * A `for` loop with a concrete trip count is NOT this condition: C5 unrolls it
     * engine-free and no WHILE remains. That asymmetry is the point — it is what
     * keeps `examples/differentiable-physics`, whose loop bound is a `const val`,
     * working with no Symja on the classpath at all.
     *
     * Recurses into every region, and into the nested primal carried by a
     * `COARSENED` op's `primal_body` attribute, because the SOI path puts the
     * user's body there.
     */
    fun containsLoop(fn: DxirFunction): Boolean = containsLoop(fn.body)

    private fun containsLoop(nodes: List<DxirNode>): Boolean {
        for (node in nodes) {
            if (node !is DxirOp) continue
            if (node.op == OpKind.WHILE) return true
            for (r in node.regions) for (b in r.blocks) if (containsLoop(b.body)) return true
            val nested = node.attrs["primal_body"]
            if (nested is DxirFunction && containsLoop(nested.body)) return true
        }
        return false
    }

    fun apply(fn: DxirFunction, engine: SymbolicEngine? = null): DxirFunction {
        var current = fn
        for (iter in 0 until FIXPOINT_CAP) {
            val next = singlePass(current, engine)
            if (structurallyEqual(current, next)) return next
            current = next
        }
        error(
            "PhiCalculus.apply exceeded fixpoint cap of $FIXPOINT_CAP iterations on " +
                "function '${fn.name}' — likely a ping-pong between two rewrites.",
        )
    }

    /** One pass through the rewrite pipeline. Each step is a full clone+rewrite of [fn]. */
    private fun singlePass(fn: DxirFunction, engine: SymbolicEngine?): DxirFunction {
        var work = fn
        work = applyF1Pass(work)
        work = applyF3Pass(work)
        work = applyDistributePass(work) // F2 + C1 (unified)
        work = applyF1Pass(work)
        work = applyC3Pass(work)
        work = applyF1Pass(work)
        // §0.4.127 — D.3i Phase 3b (Constant arms). §0.4.128 — D.3i Phase 3c
        // (LoopInvariant arm). See [applyBreakBearingClosurePass] for the full
        // dispatch on [BreakBearingWhile.classifyBreakCond]. Runs before C5–C9 so
        // any newly-vanilla WHILEs produced by the closure flow through the
        // standard corollaries in the same [singlePass] iteration.
        work = applyBreakBearingClosurePass(work)
        // Engine-backed corollaries first (most specific first per plan §4.13). When
        // fired, they replace WHILEs with closed-form expressions — including for
        // SYMBOLIC trip counts, which C5's direct unroll cannot handle. C5 is the
        // fallback for concrete-n loops that the engine-backed rules don't match.
        //
        // Order: C9 → C8 → C7 → C6. C9 is most specific (power-form `MUL(a, POW(d,b))`);
        // C8 next (variable affine a[i]); C7 next (constant a, counter-indexed b[i]);
        // C6 most general (constant a + constant b). The patterns have distinct root-op
        // shapes so they're structurally disjoint — ordering is for documentation rather
        // than fall-through priority.
        if (engine != null) {
            work = applyC9Pass(work, engine)
            work = applyC8Pass(work, engine)
            work = applyC7Pass(work, engine)
            work = applyC6Pass(work, engine)
        }
        work = applyC5Pass(work) // Stage B.2: simple-loop direct unroll, engine-free
        return work
    }

    // ------------------------------------------------------------------------
    // F1 — identity collapse: IF(p, x, x) ⇒ x (paper §4.3.1 + plan §4.2)
    // ------------------------------------------------------------------------

    private fun applyF1Pass(fn: DxirFunction): DxirFunction =
        rewriteFunction(fn) { op, nodeMap, _, _ ->
            if (op.op != OpKind.IF) return@rewriteFunction null
            // Single-result only — multi-result IFs would need per-result equality check.
            if (op.types.size != 1) return@rewriteFunction null
            val thenYield = op.regions[0].blocks.single().terminator.single()
            val elseYield = op.regions[1].blocks.single().terminator.single()
            // F1 fires only when both branches yield the SAME outer-scope SSA value.
            // Block-arg yields are out of scope for B.1 (IF has no block args; WHILE
            // block args are scoped to their own region and never appear here).
            if (thenYield.id != elseYield.id) return@rewriteFunction null
            // The yielded value is in outer scope; resolve through nodeMap.
            nodeMap[thenYield.id]
                ?: error(
                    "F1: IF op id=${op.id} yields outer-scope id=${thenYield.id} " +
                        "but it is missing from nodeMap (broken SSA dominance)",
                )
        }

    // ------------------------------------------------------------------------
    // F3 — canonicalisation: IF(p, a, b) ⇒ IF(NOT(p), b, a) when else-yield-id < then-yield-id
    // (paper §4.3.1 + plan §4.4). Ensures branches are in deterministic order so F1
    // collapse and downstream CSE-like rewrites see one canonical form.
    // ------------------------------------------------------------------------

    private fun applyF3Pass(fn: DxirFunction): DxirFunction =
        rewriteFunction(fn) { op, nodeMap, _, builder ->
            if (op.op != OpKind.IF) return@rewriteFunction null
            if (op.types.size != 1) return@rewriteFunction null
            val thenYield = op.regions[0].blocks.single().terminator.single()
            val elseYield = op.regions[1].blocks.single().terminator.single()
            // Already canonical: then-yield-id <= else-yield-id (by SSA id total order).
            if (thenYield.id <= elseYield.id) return@rewriteFunction null
            // Swap: emit IF(NOT(p), <cloned else region>, <cloned then region>).
            val clonedPred = nodeMap[op.operands[0].id]!!
            val notPred = builder.op(OpKind.NOT, listOf(clonedPred), clonedPred.type)
            val newThenRegion = cloneRegion(op.regions[1], nodeMap, builder)
            val newElseRegion = cloneRegion(op.regions[0], nodeMap, builder)
            builder.ifOp(notPred, op.types, newThenRegion, newElseRegion)
        }

    // ------------------------------------------------------------------------
    // F2 / C1 — distributive: f(IF(p, a, b)) ⇒ IF(p, f(a), f(b)). Unifies F2 (unary /
    // binary) and C1 (k-ary). Anti-swell gate per plan §4.3: only fire when the IF
    // has a single use OR both branch bodies are < 5 ops.
    // ------------------------------------------------------------------------

    private fun applyDistributePass(fn: DxirFunction): DxirFunction {
        val useCounts = computeUseCounts(fn)
        return rewriteFunction(fn) { op, nodeMap, _, builder ->
            if (op.op !in DISTRIBUTABLE_OPS) return@rewriteFunction null
            // Single-result distributable op only. Required to keep the new IF's types
            // well-defined.
            if (op.types.size != 1) return@rewriteFunction null
            // Find the first IF operand. If multiple operands are IFs, pull the FIRST
            // out this pass; subsequent passes pull the others (fixpoint loop).
            val ifIdx = op.operands.indexOfFirst { it is DxirOp && it.op == OpKind.IF }
            if (ifIdx < 0) return@rewriteFunction null
            val ifOp = op.operands[ifIdx] as DxirOp
            if (ifOp.types.size != 1) return@rewriteFunction null
            // Anti-swell gate.
            val singleUse = (useCounts[ifOp.id] ?: 0) <= 1
            val smallBranches = ifOp.regions.all { it.blocks.single().body.size < 5 }
            if (!singleUse && !smallBranches) return@rewriteFunction null
            // Result type sanity: the distributed op must produce the same dtype as the
            // IF it's replacing (otherwise pushing it in changes the IF's result type).
            // For a same-dtype distributable op (the DISTRIBUTABLE_OPS set), this holds.
            if (op.type != ifOp.type) return@rewriteFunction null
            // Build the new IF: each branch contains the cloned outer-op-body with the
            // IF's branch yield substituted into operand position [ifIdx].
            val clonedPred = nodeMap[ifOp.operands[0].id]!!
            val newThenRegion = builder.region {
                // Clone any body ops in the original IF's then-region into this new region.
                val branchNodeMap = HashMap(nodeMap)
                for (n in ifOp.regions[0].blocks.single().body) {
                    branchNodeMap[n.id] = cloneNode(n, branchNodeMap, this as DxirEmitter)
                }
                val branchYield = ifOp.regions[0].blocks.single().terminator.single()
                val branchYieldClone = branchNodeMap[branchYield.id]
                    ?: error("F2/C1: branch yield id=${branchYield.id} missing from branchNodeMap")
                // Build the new operand list with branchYieldClone at ifIdx.
                val newOperands = op.operands.mapIndexed { i, o ->
                    if (i == ifIdx) branchYieldClone
                    else nodeMap[o.id]
                        ?: error("F2/C1: outer-scope operand id=${o.id} missing from nodeMap")
                }
                val pushedOp = op(op.op, newOperands, op.type, op.attrs)
                yields(pushedOp)
            }
            val newElseRegion = builder.region {
                val branchNodeMap = HashMap(nodeMap)
                for (n in ifOp.regions[1].blocks.single().body) {
                    branchNodeMap[n.id] = cloneNode(n, branchNodeMap, this as DxirEmitter)
                }
                val branchYield = ifOp.regions[1].blocks.single().terminator.single()
                val branchYieldClone = branchNodeMap[branchYield.id]
                    ?: error("F2/C1: branch yield id=${branchYield.id} missing from branchNodeMap")
                val newOperands = op.operands.mapIndexed { i, o ->
                    if (i == ifIdx) branchYieldClone
                    else nodeMap[o.id]
                        ?: error("F2/C1: outer-scope operand id=${o.id} missing from nodeMap")
                }
                val pushedOp = op(op.op, newOperands, op.type, op.attrs)
                yields(pushedOp)
            }
            builder.ifOp(clonedPred, op.types, newThenRegion, newElseRegion)
        }
    }

    // ------------------------------------------------------------------------
    // C3 — nested-IF flattening: outerIF(c, innerIF(c', p, q), z) ⇒
    //                            innerIF(c', outerIF(c, p, z), outerIF(c, q, z))
    // (paper §4.7 + plan §4.7). Distributes the inner IF outward, swapping which
    // predicate is outermost. Symmetric for the inner-IF-on-else-branch case.
    // ------------------------------------------------------------------------

    private fun applyC3Pass(fn: DxirFunction): DxirFunction =
        rewriteFunction(fn) { op, nodeMap, _, builder ->
            if (op.op != OpKind.IF) return@rewriteFunction null
            if (op.types.size != 1) return@rewriteFunction null
            val thenBlock = op.regions[0].blocks.single()
            val elseBlock = op.regions[1].blocks.single()
            // Inner IF must be the branch's single yielded value (no other body ops),
            // else C3 would over-generate. Tighter heuristic for B.1; loosen later.
            val thenYield = thenBlock.terminator.single()
            val elseYield = elseBlock.terminator.single()
            // Detect inner-IF in then-branch only for B.1 (mirror case is symmetric).
            val innerIf = thenYield as? DxirOp ?: return@rewriteFunction null
            if (innerIf.op != OpKind.IF) return@rewriteFunction null
            if (innerIf.types.size != 1) return@rewriteFunction null
            if (thenBlock.body.singleOrNull() !== innerIf) return@rewriteFunction null
            // Pull the outer-scope SSA values: outer pred, inner pred, inner-then-yield p,
            // inner-else-yield q, outer-else-yield z. All are outer-scope to the new C3
            // form (block-arg-free IF structure, B.0a region shape).
            val outerPred = nodeMap[op.operands[0].id]!!
            val innerPred = nodeMap[innerIf.operands[0].id]!!
            val p = innerIf.regions[0].blocks.single().terminator.single()
            val q = innerIf.regions[1].blocks.single().terminator.single()
            val pClone = nodeMap[p.id]
                ?: error("C3: inner-then yield id=${p.id} missing from nodeMap")
            val qClone = nodeMap[q.id]
                ?: error("C3: inner-else yield id=${q.id} missing from nodeMap")
            val zClone = nodeMap[elseYield.id]
                ?: error("C3: outer-else yield id=${elseYield.id} missing from nodeMap")
            // Build innerIF(innerPred, outerIF(outerPred, p, z), outerIF(outerPred, q, z)).
            // Each outerIF-clone is a new region containing a single IF op.
            val pBranchOuter = builder.region {
                val outerInPThen = region { yields(pClone) }
                val outerInPElse = region { yields(zClone) }
                val nestedOuterIf = ifOp(outerPred, op.types, outerInPThen, outerInPElse)
                yields(nestedOuterIf)
            }
            val qBranchOuter = builder.region {
                val outerInQThen = region { yields(qClone) }
                val outerInQElse = region { yields(zClone) }
                val nestedOuterIf = ifOp(outerPred, op.types, outerInQThen, outerInQElse)
                yields(nestedOuterIf)
            }
            builder.ifOp(innerPred, op.types, pBranchOuter, qBranchOuter)
        }

    // ------------------------------------------------------------------------
    // §0.4.127 — D.3i Phase 3b (Constant arms) + §0.4.128 — D.3i Phase 3c
    // (LoopInvariant arm). Closure rewrites for break-bearing WHILEs.
    //
    // The FIR-side hoist for `while (cond) { ...; if (break_cond) break }` produces a
    // WHILE whose cond region is `LAND(origCond, NOT(breakCond))` (per [OpKind.LAND]'s
    // source comment). [BreakBearingWhile.classifyBreakCond] returns a typed enum that
    // tells this pass how to rewrite the WHILE — three arms today, with CounterOnly
    // and CarriedDependent left for later D.3i phases:
    //  - Constant(alwaysBreaks=true)  → loop runs zero times; results = inits.
    //  - Constant(alwaysBreaks=false) → drop the LAND-NOT wrapper; cond region
    //    terminator becomes origCond. The resulting vanilla bounded WHILE flows
    //    into the standard C5–C9 closures inside the same [singlePass] iteration.
    //  - LoopInvariant → lift breakCond into outer scope and emit
    //    `IF(breakCond, then=inits, else=vanillaWhile)`. The runtime evaluation of
    //    breakCond happens once before the loop instead of every iteration.
    // ------------------------------------------------------------------------

    private fun applyBreakBearingClosurePass(fn: DxirFunction): DxirFunction {
        // Pre-scan: collect break-bearing WHILEs whose breakCond classifies as
        // Constant or LoopInvariant. Other classifications fall through to later
        // D.3i phases.
        val toRewrite = HashMap<Int, BreakBearingWhile.BreakCondClass>()
        for (n in fn.body) {
            if (n !is DxirOp) continue
            if (n.op != OpKind.WHILE) continue
            val pattern = BreakBearingWhile.detect(n) ?: continue
            val klass = BreakBearingWhile.classifyBreakCond(pattern) ?: continue
            when (klass) {
                is BreakBearingWhile.BreakCondClass.Constant,
                BreakBearingWhile.BreakCondClass.LoopInvariant -> toRewrite[n.id] = klass
                BreakBearingWhile.BreakCondClass.CounterOnly -> {
                    // §0.4.131 (Phase 3e) handles the both-concrete shape
                    // `STEP(SUB(args[counter], thresholdConst))` with concrete origCond
                    // bound; §0.4.141 (Phase 3f) extends to the symbolic case where at
                    // least one of n / threshold is a region-external [DxirParam]. Other
                    // CounterOnly shapes (region-internal symbolic, non-canonical operand
                    // order) still fall through to later D.3i phases.
                    if (computeCounterOnlyEffectiveTripCount(pattern) != null
                        || computeCounterOnlySymbolicShape(pattern) != null
                    ) {
                        toRewrite[n.id] = klass
                    }
                }
                BreakBearingWhile.BreakCondClass.CarriedDependent -> Unit
            }
        }
        if (toRewrite.isEmpty()) return fn

        return rewriteFunction(fn) { op, nodeMap, multiOut, builder ->
            when (val klass = toRewrite[op.id] ?: return@rewriteFunction null) {
                is BreakBearingWhile.BreakCondClass.Constant -> rewriteConstantBreak(
                    op, klass, nodeMap, multiOut, builder,
                )
                BreakBearingWhile.BreakCondClass.LoopInvariant -> rewriteLoopInvariantBreak(
                    op, nodeMap, multiOut, builder,
                )
                BreakBearingWhile.BreakCondClass.CounterOnly -> {
                    val pattern = BreakBearingWhile.detect(op)!!
                    // Phase 3e gets first shot on the both-concrete shape (its
                    // const-fold composes with C5 to unroll the WHILE downstream
                    // in the same singlePass iteration); Phase 3f handles the
                    // residual symbolic-bound cases that emit a runtime min via
                    // an outer IF chain.
                    if (computeCounterOnlyEffectiveTripCount(pattern) != null) {
                        rewriteCounterOnlyBreak(op, nodeMap, multiOut, builder)
                    } else {
                        rewriteCounterOnlySymbolicBreak(op, nodeMap, multiOut, builder)
                    }
                }
                else -> error(
                    "applyBreakBearingClosurePass: unexpected class ${klass::class.simpleName} " +
                        "for WHILE id=${op.id} — pre-scan and rewrite must agree on which " +
                        "classes are handled",
                )
            }
        }
    }

    /**
     * §0.4.131 — D.3i Phase 3e helper. Recognise the canonical CounterOnly shape
     * `breakCond = STEP(SUB(args[counterArgIdx], thresholdConst))` (i.e., "break
     * when counter > threshold") and compute the resulting effective trip count.
     *
     * With counter starting at 0 and incrementing by 1, `breakCond` first becomes
     * true at iteration `threshold + 1`. Combined with the original natural bound
     * `n` (already validated as a concrete int by §0.4.124's [Pattern.tripCountConst]),
     * the effective trip count is `min(n, threshold + 1)`.
     *
     * Returns null when (a) the breakCond doesn't match the canonical shape, (b)
     * the threshold isn't a concrete non-negative int, or (c) the original bound
     * `n` is symbolic ([Pattern.tripCountParam] is non-null) — symbolic-bound
     * support requires a runtime `MIN` op which dxir doesn't carry today.
     */
    private fun computeCounterOnlyEffectiveTripCount(
        pattern: BreakBearingWhile.Pattern,
    ): Int? {
        val origN = pattern.tripCountConst ?: return null
        val counterArgIdx = pattern.counterArgIdx ?: return null
        val condBlock = pattern.whileOp.regions[0].blocks.single()
        val counterArgId = condBlock.args[counterArgIdx].id

        val step = pattern.breakCond as? DxirOp ?: return null
        if (step.op != OpKind.STEP) return null
        if (step.operands.size != 1) return null
        val sub = step.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        if (sub.operands.size != 2) return null
        if (sub.operands[0].id != counterArgId) return null
        val thresholdConst = sub.operands[1] as? DxirConst ?: return null
        val thresholdValue = (thresholdConst.value as? Number)?.toDouble() ?: return null
        if (thresholdValue < 0.0 || thresholdValue != thresholdValue.toInt().toDouble()) return null
        val threshold = thresholdValue.toInt()
        return minOf(origN, threshold + 1)
    }

    /**
     * §0.4.131 — CounterOnly arm. The break predicate has the canonical shape
     * `STEP(SUB(args[counter], thresholdConst))` AND the natural bound `n` is a
     * concrete integer. Both bounds compose into a single effective trip count
     * `min(n, threshold + 1)`; this rewrite emits a vanilla bounded WHILE whose
     * cond region terminator is `STEP(SUB(const(effectiveTrip), counterArg))` and
     * leaves the body region unchanged. The C5–C9 corollaries close it downstream
     * in the same [singlePass] iteration.
     */
    private fun rewriteCounterOnlyBreak(
        op: DxirOp,
        nodeMap: MutableMap<Int, DxirNode>,
        multiOut: MutableMap<Int, List<DxirNode>>,
        builder: DxirBuilder,
    ): DxirNode {
        val pattern = BreakBearingWhile.detect(op)!!
        val effectiveTrip = computeCounterOnlyEffectiveTripCount(pattern)
            ?: error(
                "applyBreakBearingClosurePass: rewriteCounterOnlyBreak called on WHILE " +
                    "id=${op.id} but the canonical CounterOnly shape no longer matches — " +
                    "pre-scan and rewrite must agree",
            )
        val counterArgIdx = pattern.counterArgIdx!!
        val counterType = op.operands[counterArgIdx].type
        val condBlock = op.regions[0].blocks.single()
        val condTerminator = condBlock.terminator.single() as DxirOp
        val boolType = condTerminator.type

        val newCondRegion = builder.region {
            val newArgs = condBlock.args.map { arg(it.type, it.sharding) }
            val counterArg = newArgs[counterArgIdx]
            val newN = const(effectiveTrip, counterType)
            val newSub = op(OpKind.SUB, listOf(newN, counterArg), counterType)
            val newStep = op(OpKind.STEP, listOf(newSub), boolType)
            yields(newStep)
        }
        val newBodyRegion = cloneRegion(op.regions[1], nodeMap, builder, multiOut)
        val clonedInits = op.operands.map { init ->
            nodeMap[init.id]
                ?: error(
                    "applyBreakBearingClosurePass: init id=${init.id} for WHILE id=${op.id} " +
                        "missing from nodeMap",
                )
        }
        return builder.opMulti(
            OpKind.WHILE,
            clonedInits,
            op.types,
            regions = listOf(newCondRegion, newBodyRegion),
        )
    }

    /**
     * §0.4.141 — D.3i Phase 3f. Captures the canonical CounterOnly shape
     * `breakCond = STEP(SUB(args[counterArgIdx], threshold))` when at least one of
     * `n` / `threshold` is a region-external [DxirParam] — i.e., the cases where
     * the effective trip count `min(n, threshold + 1)` can't be folded at compile
     * time but CAN be emitted as a runtime IF chain in outer scope. Phase 3e's
     * concrete-int shortcut handles the both-concrete case via const folding.
     *
     * Returns null when:
     *  - The breakCond doesn't match the canonical `STEP(SUB(args[counter], threshold))`
     *    shape, or [Pattern.counterArgIdx] is missing.
     *  - `threshold` is a [DxirOpResult] / [DxirCall] / unrecognised node kind.
     *  - `threshold` is a region-internal [DxirOp] whose subtree transitively
     *    references a cond-region block-arg ([DxirBlockArg]) — i.e., the threshold
     *    isn't actually loop-invariant (the classifier should have routed this to
     *    [BreakBearingWhile.BreakCondClass.CarriedDependent], but a defensive check
     *    here keeps the lift sound).
     *  - `n` is a region-internal [DxirOp] whose subtree transitively references a
     *    cond-region block-arg (defensive check, mirrors the threshold case).
     *  - BOTH `n` and `threshold` are concrete [DxirConst] — that's Phase 3e's case.
     *  - The threshold const is negative or non-integer (mirrors Phase 3e's checks).
     *
     * §0.4.142 (Phase 3g) widened the accepted threshold shape from
     * `DxirConst | DxirParam` to also include `DxirOp` — outer-scope [DxirOp]s
     * resolve through `nodeMap` directly, region-internal [DxirOp]s get their
     * dependency tree lifted into outer scope by [rewriteCounterOnlySymbolicBreak]
     * via [liftRegionInternalSubtree], mirroring §0.4.128's `rewriteLoopInvariantBreak`.
     *
     * §0.4.143 (Phase 3h) extends the same widening to `n` via the new
     * [BreakBearingWhile.Pattern.tripCountOp] field — the dispatch is identical
     * to the threshold path, just applied to the n operand.
     */
    private data class CounterOnlySymbolicShape(
        val nNode: DxirNode,
        val thresholdNode: DxirNode,
        val counterArgIdx: Int,
        val counterType: io.tlaloc.ir.DxirType,
        val boolType: io.tlaloc.ir.DxirType,
    )

    private fun computeCounterOnlySymbolicShape(
        pattern: BreakBearingWhile.Pattern,
    ): CounterOnlySymbolicShape? {
        val counterArgIdx = pattern.counterArgIdx ?: return null
        val condBlock = pattern.whileOp.regions[0].blocks.single()
        val counterArgId = condBlock.args[counterArgIdx].id
        val condBodyIds = condBlock.body.map { it.id }.toSet()

        val step = pattern.breakCond as? DxirOp ?: return null
        if (step.op != OpKind.STEP) return null
        if (step.operands.size != 1) return null
        val sub = step.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        if (sub.operands.size != 2) return null
        if (sub.operands[0].id != counterArgId) return null
        val thresholdNode = sub.operands[1]
        when (thresholdNode) {
            is DxirConst -> {
                val v = (thresholdNode.value as? Number)?.toDouble() ?: return null
                if (v < 0.0 || v != v.toInt().toDouble()) return null
            }
            is DxirParam -> Unit
            is DxirOp -> {
                // Region-internal threshold must be liftable; outer-scope thresholds
                // resolve through nodeMap at rewrite time. Either way, the threshold's
                // subtree must not transitively reference any cond-region block-arg.
                if (thresholdNode.id in condBodyIds &&
                    !isRegionInternalSubtreeLiftable(thresholdNode, condBodyIds)
                ) return null
            }
            is DxirOpResult -> {
                // §0.4.149 — DxirOpResult threshold (e.g., a multi-result op's
                // result(k) used directly as the break threshold). When the source
                // op is region-internal, check the source's subtree for liftability;
                // when it's outer-scope, accept directly (resolved via nodeMap at
                // rewrite time). Same scope discipline as the DxirOp case above.
                if (thresholdNode.source.id in condBodyIds &&
                    !isRegionInternalSubtreeLiftable(thresholdNode, condBodyIds)
                ) return null
            }
            else -> return null
        }

        // §0.4.143 — n can be DxirParam (existing), DxirOp (new), or DxirConst
        // (when neither tripCountParam nor tripCountOp is populated). For DxirConst
        // we walk back through origCond to recover the actual node.
        // §0.4.150 — n can also be DxirOpResult (a multi-result op's result(k)).
        val nNode: DxirNode = pattern.tripCountParam
            ?: pattern.tripCountOp
            ?: pattern.tripCountOpResult
            ?: run {
                val origCondOp = pattern.origCond as? DxirOp ?: return null
                if (origCondOp.op != OpKind.STEP) return null
                val origSub = origCondOp.operands[0] as? DxirOp ?: return null
                if (origSub.op != OpKind.SUB) return null
                origSub.operands[0]
            }
        when (nNode) {
            is DxirConst, is DxirParam -> Unit
            is DxirOp -> {
                // §0.4.143 — region-internal DxirOp n must be liftable; outer-scope
                // resolves through nodeMap at rewrite time. Same scope discipline as
                // §0.4.142's DxirOp threshold path.
                if (nNode.id in condBodyIds &&
                    !isRegionInternalSubtreeLiftable(nNode, condBodyIds)
                ) return null
            }
            is DxirOpResult -> {
                // §0.4.150 — region-internal DxirOpResult n requires the source op
                // (multi-result) to be liftable. Outer-scope sources resolve through
                // nodeMap at rewrite time. Mirrors §0.4.149's DxirOpResult threshold
                // dispatch.
                if (nNode.source.id in condBodyIds &&
                    !isRegionInternalSubtreeLiftable(nNode, condBodyIds)
                ) return null
            }
            else -> return null
        }

        // Phase 3e handles both-concrete; Phase 3f/3g/3h's contract is "at least one symbolic".
        if (nNode is DxirConst && thresholdNode is DxirConst) return null

        return CounterOnlySymbolicShape(
            nNode = nNode,
            thresholdNode = thresholdNode,
            counterArgIdx = counterArgIdx,
            counterType = pattern.whileOp.operands[counterArgIdx].type,
            boolType = step.type,
        )
    }

    /**
     * §0.4.142 — D.3i Phase 3g check. Returns true when [root] is a region-internal
     * subtree whose dependency walk reaches only outer-scope leaves ([DxirParam],
     * outer-scope [DxirOp]s/[DxirConst]s) and other region-internal [DxirOp]/[DxirConst]
     * nodes — never a [DxirBlockArg]. A block-arg dep means the threshold is
     * iteration-dependent (carried-arg derived), which the classifier should have
     * routed to [BreakBearingWhile.BreakCondClass.CarriedDependent]; this check
     * defensively rejects such shapes regardless of how they were classified.
     *
     * §0.4.149 — [DxirOpResult] is now accepted: the walker recurses into the
     * source op's operands, treating the multi-result op as if it were a regular
     * [DxirOp]. [cloneNode] handles the multi-result reconstruction via `opMulti`
     * based on `node.types.size > 1`. [DxirCall] still rejects (cross-function
     * call boundaries need separate handling).
     */
    private fun isRegionInternalSubtreeLiftable(
        root: DxirNode,
        condBodyIds: Set<Int>,
    ): Boolean {
        val visited = HashSet<Int>()
        val stack = ArrayDeque<DxirNode>()
        stack.addLast(root)
        while (stack.isNotEmpty()) {
            val n = stack.removeLast()
            if (!visited.add(n.id)) continue
            if (n is DxirBlockArg) return false
            if (n.id !in condBodyIds) continue  // outer-scope leaf — already in nodeMap
            when (n) {
                is DxirOp -> n.operands.forEach { stack.addLast(it) }
                is DxirConst -> Unit
                is DxirOpResult -> n.source.operands.forEach { stack.addLast(it) }
                is DxirCall -> return false
                else -> Unit
            }
        }
        return true
    }

    /**
     * §0.4.141 — D.3i Phase 3f. Symbolic-bound CounterOnly arm. The closed-form
     * effective trip count `min(n, threshold + 1)` becomes a runtime `IF` chain
     * lifted into outer scope:
     *
     * ```
     * thresholdPlus1 = ADD(threshold, 1)
     * cmp            = STEP(SUB(thresholdPlus1, n))   // 1 iff n < thresholdPlus1
     * effectiveN     = IF(cmp, n, thresholdPlus1)     // picks min
     * ```
     *
     * The rewritten WHILE's cond region terminator is `STEP(SUB(effectiveN, counter))`
     * — the LAND-NOT closure wrapper is gone. C5 cannot unroll this WHILE downstream
     * (its bound is no longer a compile-time int), so the WHILE remains for the
     * interpreter to evaluate at runtime; the gain is structural — the closure is
     * resolved into a vanilla bounded WHILE with a hoisted runtime bound.
     */
    private fun rewriteCounterOnlySymbolicBreak(
        op: DxirOp,
        nodeMap: MutableMap<Int, DxirNode>,
        multiOut: MutableMap<Int, List<DxirNode>>,
        builder: DxirBuilder,
    ): DxirNode {
        val pattern = BreakBearingWhile.detect(op)!!
        val shape = computeCounterOnlySymbolicShape(pattern)
            ?: error(
                "applyBreakBearingClosurePass: rewriteCounterOnlySymbolicBreak called on " +
                    "WHILE id=${op.id} but the symbolic CounterOnly shape no longer matches — " +
                    "pre-scan and rewrite must agree",
            )

        val condBlock = op.regions[0].blocks.single()
        val condBodyIds = condBlock.body.map { it.id }.toSet()

        // [DxirConst] nodes are reconstructed (no SSA-id dependency); [DxirParam]
        // nodes resolve through nodeMap to the cloned param in the new function;
        // outer-scope [DxirOp]s also resolve through nodeMap; §0.4.142 — region-internal
        // [DxirOp] thresholds get their dependency tree lifted via clone, mirroring
        // §0.4.128's `rewriteLoopInvariantBreak`. §0.4.143 — same dispatch applies
        // to `n`, which now also accepts [DxirOp] (region-internal liftable or
        // outer-scope) per [BreakBearingWhile.Pattern.tripCountOp].
        val nClone: DxirNode = when (val n = shape.nNode) {
            is DxirConst -> builder.const(n.value, n.type, n.sharding)
            is DxirOp -> if (n.id in condBodyIds) {
                liftRegionInternalSubtree(n, condBlock, nodeMap, multiOut, builder, op.id)
            } else {
                nodeMap[n.id]
                    ?: error(
                        "applyBreakBearingClosurePass: outer-scope n op id=${n.id} for " +
                            "WHILE id=${op.id} missing from nodeMap",
                    )
            }
            is DxirOpResult -> {
                // §0.4.150 — DxirOpResult n: lift the source op (multi-result), then
                // wrap as `clonedSource.result(n.index)`. Mirrors §0.4.149's
                // DxirOpResult threshold dispatch.
                val sourceClone: DxirNode = if (n.source.id in condBodyIds) {
                    liftRegionInternalSubtree(n.source, condBlock, nodeMap, multiOut, builder, op.id)
                } else {
                    nodeMap[n.source.id]
                        ?: error(
                            "applyBreakBearingClosurePass: outer-scope n DxirOpResult source " +
                                "id=${n.source.id} for WHILE id=${op.id} missing from nodeMap",
                        )
                }
                require(sourceClone is DxirOp) {
                    "applyBreakBearingClosurePass: n DxirOpResult's cloned source must be a " +
                        "DxirOp (got ${sourceClone::class.simpleName}) for WHILE id=${op.id}"
                }
                sourceClone.result(n.index)
            }
            else -> nodeMap[n.id]
                ?: error(
                    "applyBreakBearingClosurePass: n node id=${n.id} for WHILE id=${op.id} " +
                        "missing from nodeMap",
                )
        }
        val thresholdClone: DxirNode = when (val t = shape.thresholdNode) {
            is DxirConst -> builder.const(t.value, t.type, t.sharding)
            is DxirOp -> if (t.id in condBodyIds) {
                liftRegionInternalSubtree(t, condBlock, nodeMap, multiOut, builder, op.id)
            } else {
                nodeMap[t.id]
                    ?: error(
                        "applyBreakBearingClosurePass: outer-scope threshold op id=${t.id} for " +
                            "WHILE id=${op.id} missing from nodeMap",
                    )
            }
            is DxirOpResult -> {
                // §0.4.149 — lift the source op (multi-result), then wrap the
                // cloned source as `result(t.index)` to recover the indexed ref.
                val sourceClone: DxirNode = if (t.source.id in condBodyIds) {
                    liftRegionInternalSubtree(t.source, condBlock, nodeMap, multiOut, builder, op.id)
                } else {
                    nodeMap[t.source.id]
                        ?: error(
                            "applyBreakBearingClosurePass: outer-scope threshold DxirOpResult " +
                                "source id=${t.source.id} for WHILE id=${op.id} missing from nodeMap",
                        )
                }
                require(sourceClone is DxirOp) {
                    "applyBreakBearingClosurePass: threshold DxirOpResult's cloned source " +
                        "must be a DxirOp (got ${sourceClone::class.simpleName}) for WHILE id=${op.id}"
                }
                sourceClone.result(t.index)
            }
            else -> nodeMap[t.id]
                ?: error(
                    "applyBreakBearingClosurePass: threshold node id=${t.id} for WHILE id=${op.id} " +
                        "missing from nodeMap",
                )
        }

        val one = builder.const(1, shape.counterType)
        val thresholdPlus1 = builder.op(OpKind.ADD, listOf(thresholdClone, one), shape.counterType)
        val cmpSub = builder.op(OpKind.SUB, listOf(thresholdPlus1, nClone), shape.counterType)
        val cmp = builder.op(OpKind.STEP, listOf(cmpSub), shape.boolType)
        val effectiveN = builder.ifOp(
            cond = cmp,
            types = listOf(shape.counterType),
            thenRegion = builder.region { yields(nClone) },
            elseRegion = builder.region { yields(thresholdPlus1) },
        )

        val newCondRegion = builder.region {
            val newArgs = condBlock.args.map { arg(it.type, it.sharding) }
            val counterArg = newArgs[shape.counterArgIdx]
            val newSub = op(OpKind.SUB, listOf(effectiveN, counterArg), shape.counterType)
            val newStep = op(OpKind.STEP, listOf(newSub), shape.boolType)
            yields(newStep)
        }
        val newBodyRegion = cloneRegion(op.regions[1], nodeMap, builder, multiOut)
        val clonedInits = op.operands.map { init ->
            nodeMap[init.id]
                ?: error(
                    "applyBreakBearingClosurePass: init id=${init.id} for WHILE id=${op.id} " +
                        "missing from nodeMap",
                )
        }
        return builder.opMulti(
            OpKind.WHILE,
            clonedInits,
            op.types,
            regions = listOf(newCondRegion, newBodyRegion),
        )
    }

    /**
     * §0.4.142 — D.3i Phase 3g lift helper. Walks [root] post-order over the cond
     * region's body, collects region-internal ids that need cloning into outer
     * scope, and emits the clones via [cloneNode] in topological order (operands
     * before users). Returns the lifted root — i.e., `nodeMap[root.id]` after the
     * clones land — so the caller can splice it into outer-scope arithmetic.
     *
     * Mirrors the inline walk in §0.4.128's [rewriteLoopInvariantBreak]: same
     * post-order traversal, same `LinkedHashSet` for ordering, same `cloneNode`
     * dispatch. Errors on [DxirBlockArg] because [computeCounterOnlySymbolicShape]'s
     * pre-check ([isRegionInternalSubtreeLiftable]) should have rejected any
     * subtree with carried-arg deps before reaching here.
     *
     * §0.4.149 — [DxirOpResult] is now walked through to its source's operands;
     * the source op (multi-result) is added to `toClone` and `cloneNode` rebuilds
     * it via `opMulti`. The DxirOpResult ref itself doesn't need a separate
     * clone — `nodeMap[source.id]` after the lift holds the cloned multi-result
     * op, and the caller wraps it as `clonedSource.result(index)` to recover the
     * indexed reference.
     */
    private fun liftRegionInternalSubtree(
        root: DxirNode,
        condBlock: DxirBlock,
        nodeMap: MutableMap<Int, DxirNode>,
        multiOut: MutableMap<Int, List<DxirNode>>,
        builder: DxirBuilder,
        contextOpId: Int,
    ): DxirNode {
        val condBodyById = condBlock.body.associateBy { it.id }
        val condBodyIds = condBodyById.keys
        val toClone = LinkedHashSet<Int>()
        val visited = HashSet<Int>()
        fun walk(n: DxirNode) {
            if (!visited.add(n.id)) return
            if (n is DxirBlockArg) {
                error(
                    "applyBreakBearingClosurePass: liftRegionInternalSubtree reached block-arg " +
                        "id=${n.id} for WHILE id=$contextOpId — pre-check should have rejected " +
                        "this subtree as not loop-invariant",
                )
            }
            if (n.id !in condBodyIds) return
            when (n) {
                is DxirOp -> {
                    for (operand in n.operands) walk(operand)
                    toClone.add(n.id)
                }
                is DxirConst -> toClone.add(n.id)
                is DxirOpResult -> {
                    // §0.4.149 — walk through to the source's operands; the
                    // source's id (== n.id) gets added to toClone, and cloneNode
                    // reconstructs it via opMulti for multi-result.
                    for (operand in n.source.operands) walk(operand)
                    toClone.add(n.id)
                }
                is DxirCall -> error(
                    "applyBreakBearingClosurePass: liftRegionInternalSubtree reached " +
                        "${n::class.simpleName} id=${n.id} for WHILE id=$contextOpId — " +
                        "pre-check should have rejected this subtree shape",
                )
                else -> Unit
            }
        }
        walk(root)
        for (id in toClone) {
            val node = condBodyById[id]
                ?: error(
                    "applyBreakBearingClosurePass: id=$id was collected as region-internal " +
                        "but not found in condBlock.body for WHILE id=$contextOpId",
                )
            nodeMap[id] = cloneNode(node, nodeMap, builder as DxirEmitter, multiOut)
        }
        return nodeMap[root.id]
            ?: error(
                "applyBreakBearingClosurePass: root id=${root.id} not in nodeMap after lift " +
                    "for WHILE id=$contextOpId",
            )
    }

    /**
     * §0.4.127 — Constant arm. `alwaysBreaks=true` collapses the WHILE to its inits
     * via `multiOut` (loop runs zero times); `alwaysBreaks=false` rebuilds the WHILE
     * with the LAND-NOT wrapper dropped from the cond region, leaving a vanilla
     * bounded WHILE for the C5–C9 corollaries to close downstream in the same pass.
     */
    private fun rewriteConstantBreak(
        op: DxirOp,
        klass: BreakBearingWhile.BreakCondClass.Constant,
        nodeMap: MutableMap<Int, DxirNode>,
        multiOut: MutableMap<Int, List<DxirNode>>,
        builder: DxirBuilder,
    ): DxirNode {
        if (klass.alwaysBreaks) {
            val perIndex = op.operands.map { init ->
                nodeMap[init.id]
                    ?: error(
                        "applyBreakBearingClosurePass: init id=${init.id} for WHILE id=${op.id} " +
                            "missing from nodeMap (broken SSA before fold)",
                    )
            }
            multiOut[op.id] = perIndex
            // Mirror the [applyC5Pass] convention: nodeMap[op.id] gets the index-0
            // value (multiOut takes precedence for DxirOpResult refs at higher indices).
            return perIndex[0]
        }
        val origCondId = BreakBearingWhile.detect(op)!!.origCond.id
        val condBlock = op.regions[0].blocks.single()
        val newCondRegion = builder.region {
            val regionNodeMap = HashMap(nodeMap)
            for (a in condBlock.args) {
                regionNodeMap[a.id] = arg(a.type, a.sharding)
            }
            for (n in condBlock.body) {
                regionNodeMap[n.id] = cloneNode(n, regionNodeMap, this as DxirEmitter, multiOut)
            }
            val newTerm = regionNodeMap[origCondId]
                ?: error(
                    "applyBreakBearingClosurePass: origCond id=$origCondId not found in cond " +
                        "region body for WHILE id=${op.id}",
                )
            yields(newTerm)
        }
        val newBodyRegion = cloneRegion(op.regions[1], nodeMap, builder, multiOut)
        val clonedInits = op.operands.map { init ->
            nodeMap[init.id]
                ?: error(
                    "applyBreakBearingClosurePass: init id=${init.id} for WHILE id=${op.id} " +
                        "missing from nodeMap",
                )
        }
        return builder.opMulti(
            OpKind.WHILE,
            clonedInits,
            op.types,
            regions = listOf(newCondRegion, newBodyRegion),
        )
    }

    /**
     * §0.4.128 — LoopInvariant arm. The breakCond predicate doesn't depend on any
     * cond block-arg, so it evaluates to the same value every iteration. Lift it
     * into outer scope and emit `IF(breakCond, then=inits, else=vanillaWhile)` —
     * the predicate runs once before the loop, then either short-circuits the loop
     * (yields the inits) or runs the loop with the LAND-NOT wrapper dropped.
     *
     * Lifting walks `Pattern.breakCond` post-order and clones any cond-region
     * intermediate ops into outer scope. Leaf references that are already
     * outer-scope ([DxirParam], outer [DxirConst], outer [DxirOp]) resolve through
     * the outer [nodeMap]; region-internal ops get fresh outer-scope ids via
     * [cloneNode]. Block-arg references inside breakCond would violate the
     * LoopInvariant classification — the walker errors out as a defensive sanity
     * check on the classifier's contract.
     */
    private fun rewriteLoopInvariantBreak(
        op: DxirOp,
        nodeMap: MutableMap<Int, DxirNode>,
        multiOut: MutableMap<Int, List<DxirNode>>,
        builder: DxirBuilder,
    ): DxirNode {
        val pattern = BreakBearingWhile.detect(op)!!
        val origCondId = pattern.origCond.id
        val condBlock = op.regions[0].blocks.single()
        val condBodyById = condBlock.body.associateBy { it.id }
        val condBodyIds = condBodyById.keys

        // Post-order walk over breakCond, collecting cond-region-internal ids that
        // need cloning into outer scope. LinkedHashSet preserves insertion order so
        // the subsequent clone loop emits ops bottom-up (operands before users).
        val toClone = LinkedHashSet<Int>()
        val visited = HashSet<Int>()
        fun walk(n: DxirNode) {
            if (!visited.add(n.id)) return
            if (n is DxirBlockArg) {
                error(
                    "applyBreakBearingClosurePass: LoopInvariant breakCond reaches block-arg " +
                        "id=${n.id} for WHILE id=${op.id} — classifier contract violated",
                )
            }
            if (n.id !in condBodyIds) return  // outer-scope leaf, in nodeMap already
            when (n) {
                is DxirOp -> {
                    for (operand in n.operands) walk(operand)
                    toClone.add(n.id)
                }
                is DxirOpResult -> walk(n.source)
                is DxirCall -> {
                    for (arg in n.args) walk(arg)
                    toClone.add(n.id)
                }
                is DxirConst -> toClone.add(n.id)
                is DxirParam -> Unit
                is DxirBlockArg -> Unit  // unreachable; handled above
            }
        }
        walk(pattern.breakCond)

        // Clone collected region-internal ops into outer scope in topological order.
        for (id in toClone) {
            val node = condBodyById[id]
                ?: error(
                    "applyBreakBearingClosurePass: id=$id was collected as region-internal " +
                        "but not found in condBlock.body for WHILE id=${op.id}",
                )
            nodeMap[id] = cloneNode(node, nodeMap, builder as DxirEmitter, multiOut)
        }

        val liftedBreakCond = resolveClonedOperand(pattern.breakCond, nodeMap, op.id, multiOut)
        val clonedInits = op.operands.map { init ->
            resolveClonedOperand(init, nodeMap, op.id, multiOut)
        }

        // then-arm: zero iterations, yield the inits as-is.
        val thenRegion = builder.region {
            yields(*clonedInits.toTypedArray())
        }
        // else-arm: vanilla bounded WHILE (LAND-NOT wrapper dropped), then yield its
        // results. Same shape as [rewriteConstantBreak]'s alwaysBreaks=false case.
        val elseRegion = builder.region {
            val newCondRegion = region {
                val regionNodeMap = HashMap(nodeMap)
                for (a in condBlock.args) {
                    regionNodeMap[a.id] = arg(a.type, a.sharding)
                }
                for (n in condBlock.body) {
                    regionNodeMap[n.id] = cloneNode(n, regionNodeMap, this as DxirEmitter, multiOut)
                }
                val newTerm = regionNodeMap[origCondId]
                    ?: error(
                        "applyBreakBearingClosurePass: origCond id=$origCondId not found in " +
                            "cond region body for WHILE id=${op.id}",
                    )
                yields(newTerm)
            }
            val newBodyRegion = cloneRegion(op.regions[1], nodeMap, this as DxirEmitter, multiOut)
            val newWhile = (this as DxirEmitter).opMulti(
                OpKind.WHILE,
                clonedInits,
                op.types,
                regions = listOf(newCondRegion, newBodyRegion),
            )
            val whileResults = (0 until op.types.size).map { newWhile.result(it) }
            yields(*whileResults.toTypedArray())
        }
        return builder.ifOp(liftedBreakCond, op.types, thenRegion, elseRegion)
    }

    // ------------------------------------------------------------------------
    // C5 — simple-loop closed-form: 𝔏^n_L d = f(φ_L(p, d)) ⇒ d_exit = f^[n](p)
    // (paper §4.3.2 + plan §4.8). Direct-unroll implementation: detect a 2-loop-carried
    // WHILE with one counter and one carried value, extract the concrete trip count
    // from the cond predicate, and unroll the body's back-edge subtree N times in
    // dxir. F4 (loop-entry-φ recurrence) is implicit in the unroll — each iteration
    // substitutes the previous result for the carried block-arg, exactly what F4
    // formalises.
    //
    // Symbolic trip counts (n is a runtime param, not a const) and engine-backed
    // simplification (Symja's Nest + simplify) are deferred to Stage B.3. For B.2
    // first cut, only concrete `n = const(integer)` fires.
    // ------------------------------------------------------------------------

    /** Pattern detected by [detectSimpleLoop]. */
    private data class SimpleLoopPattern(
        /**
         * Indices of every non-counter result referenced downstream (§0.4.51 widened C5
         * from single to multi). The unroll tracks ALL carrieds per iteration and, after
         * `tripCount` iterations, publishes each referenced index's final value into
         * `rewriteFunction`'s multiOut map. If the set is empty, C5 skips (the loop is
         * dead — a separate DCE concern).
         */
        val referencedIndices: Set<Int>,
        /** Index of the counter (loop-carried iteration variable, dropped after C5 fires). */
        val counterIdx: Int,
        /** Concrete trip count extracted from the cond predicate's bound. */
        val tripCount: Int,
    )

    /**
     * Recognise an N-loop-carried WHILE matching the C5 simple-loop pattern. Widened
     * in §0.4.39 from the original 2-carried case to support multi-var for-loop bodies
     * (the FIR lowering now emits one carried per mutated outer `var` plus the counter):
     *  - N ≥ 1 loop-carried user values + 1 counter (any positions; counter detected
     *    structurally via the cond-STEP-of-SUB shape)
     *  - Counter init is `const(0)` of integer type
     *  - Counter back-edge is `ADD(args[counterIdx], const(1))`
     *  - Cond predicate is `STEP(SUB(n, args[counterIdx]))` where `n` is a `DxirConst`
     *    with concrete integer value
     *  - NO carried back-edge references `args[counterIdx]` (the paper's C5 hypothesis:
     *    every carried update is a function of carrieds + loop-invariants, not the
     *    iteration index)
     *  - Exactly one non-counter result is referenced downstream (caller-checked via
     *    [findSingleReferencedCarried]); multi-reference is deferred until the
     *    `rewriteFunction` framework grows multi-result replacement support.
     *
     * Returns null if any check fails.
     */
    private fun detectSimpleLoop(op: DxirOp, referencedIndices: Set<Int>): SimpleLoopPattern? {
        if (op.op != OpKind.WHILE) return null
        if (op.operands.size < 2) return null
        val condBlock = op.regions[0].blocks.single()
        val bodyBlock = op.regions[1].blocks.single()
        val condArgs = condBlock.args
        val bodyArgs = bodyBlock.args

        // Cond shape: STEP(SUB(n, args[counterIdx]))
        val pred = condBlock.terminator.single() as? DxirOp ?: return null
        if (pred.op != OpKind.STEP) return null
        val sub = pred.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        val nBound = sub.operands[0]
        val counterArgRef = sub.operands[1]
        // counterArgRef must be one of the cond region's block args.
        val condCounterIdx = condArgs.indexOfFirst { it.id == counterArgRef.id }
        if (condCounterIdx < 0) return null
        // Trip-count bound must be a concrete integer DxirConst (B.2 first cut).
        val nConst = nBound as? DxirConst ?: return null
        val nValue = (nConst.value as? Number)?.toInt() ?: return null
        if (nValue < 0) return null

        // Counter init: const(0) of same dtype as the counter arg.
        val counterInit = op.operands[condCounterIdx] as? DxirConst ?: return null
        val initValue = (counterInit.value as? Number)?.toInt() ?: return null
        if (initValue != 0) return null

        // Counter back-edge: ADD(bodyArgs[condCounterIdx], const(1))
        val counterBackEdge = bodyBlock.terminator[condCounterIdx] as? DxirOp ?: return null
        if (counterBackEdge.op != OpKind.ADD) return null
        if (counterBackEdge.operands[0].id != bodyArgs[condCounterIdx].id) return null
        val incrConst = counterBackEdge.operands[1] as? DxirConst ?: return null
        if ((incrConst.value as? Number)?.toInt() != 1) return null

        // At least one non-counter index must be referenced downstream (empty =
        // dead loop, out of scope for C5). Counter refs are disqualifying.
        if (referencedIndices.isEmpty()) return null
        if (condCounterIdx in referencedIndices) return null
        for (idx in referencedIndices) {
            if (idx < 0 || idx >= op.operands.size) return null
        }

        // §0.4.40 — the pre-§0.4.40 check rejected carried back-edges that referenced
        // `args[condCounterIdx]` (the "C5 hypothesis: f doesn't depend on i"). That was
        // overly conservative for a concrete-N direct unroll: counter references become
        // per-iteration concrete-value substitutions naturally through [applyC5Pass]'s
        // `perIterMap[counterArgId] = counterValue` binding. Loop-index-dependent body
        // expressions like `d = d + i.toFloat()` (CAST of the counter arg) are unrolled
        // into `d += CAST(0); d += CAST(1); d += CAST(2); …` — semantically correct,
        // and the downstream [DxirReverseTransform] sees only straight-line ops.
        //
        // The old check still matters for a hypothetical symbolic-N C5 (where counter
        // references can't be concretised). That's not today's path.

        return SimpleLoopPattern(
            referencedIndices = referencedIndices,
            counterIdx = condCounterIdx,
            tripCount = nValue,
        )
    }

    /**
     * Walk [root]'s operand-closure within [scope] (the body region's body ops) and
     * report whether any node references SSA id [target]. Returns true if found.
     */
    private fun referencesId(root: DxirNode, target: Int, scope: List<DxirNode>): Boolean {
        if (root.id == target) return true
        val scopeIds = scope.map { it.id }.toHashSet()
        val visited = HashSet<Int>()
        fun visit(n: DxirNode): Boolean {
            if (!visited.add(n.id)) return false
            if (n.id == target) return true
            if (n is DxirOp) {
                for (o in n.operands) {
                    // Recurse only into ops within scope OR direct arg/const matches.
                    if (o.id == target) return true
                    if (o.id in scopeIds && visit(o)) return true
                }
                for (r in n.regions) for (b in r.blocks) {
                    for (sub in b.body) if (visit(sub)) return true
                    for (term in b.terminator) if (visit(term)) return true
                }
            }
            return false
        }
        return visit(root)
    }

    /**
     * C5 rewrite pass — direct unroll of detected simple loops.
     *
     * For each WHILE matching [detectSimpleLoop], build the closed-form value of the
     * single downstream-referenced carried result by cloning the body region's body N
     * times into the outer builder. Each iteration's block args are bound to the
     * previous iteration's back-edge values — ALL carried args, not just the referenced
     * one, because body ops may read them (e.g., a multi-var brachistochrone body
     * computes `v_new` then reads `v_old` + `v_new` when updating `t`). The referenced
     * carried's final value replaces the WHILE in the rewritten function.
     *
     * **Restriction (§0.4.39):** only fires when exactly one non-counter result is
     * referenced downstream. Zero references means the loop is dead (a separate DCE
     * concern); multiple references would need multi-result replacement in the
     * [rewriteFunction] framework (deferred). The counter result is always dropped.
     */
    private fun applyC5Pass(fn: DxirFunction): DxirFunction {
        // §0.4.152 — region-recursive pre-scan: detect C5-eligible WHILEs at top level
        // AND inside IF region bodies. §0.4.161 — Phase 4b: the recursion now walks
        // into ALL region-bearing ops' regions (IF and WHILE alike), so a safeC5
        // WHILE nested inside another WHILE's body is discovered. The rewrite is
        // similarly generalised: any region-bearing op with safeC5 descendants gets
        // its regions rewritten, not just IF.
        val safeC5: Map<Int, SimpleLoopPattern> = HashMap<Int, SimpleLoopPattern>().apply {
            fun scan(nodes: List<DxirNode>) {
                for (n in nodes) {
                    if (n !is DxirOp) continue
                    if (n.op == OpKind.WHILE && n.operands.size >= 2) {
                        val referencedIndices = findReferencedCarried(fn, n)
                        if (referencedIndices.isNotEmpty()) {
                            detectSimpleLoop(n, referencedIndices)?.let { this[n.id] = it }
                        }
                    }
                    // §0.4.161 — recurse into ALL region bodies, not just IF's. This
                    // discovers a safeC5 WHILE inside an outer WHILE's body region.
                    if (n.regions.isNotEmpty()) {
                        for (r in n.regions) for (b in r.blocks) scan(b.body)
                    }
                }
            }
            scan(fn.body)
        }
        if (safeC5.isEmpty()) return fn

        return rewriteFunction(fn) { op, nodeMap, multiOut, builder ->
            // (a) Top-level safeC5 WHILE — unroll in place.
            safeC5[op.id]?.let { pat ->
                return@rewriteFunction unrollC5InEmitter(
                    op, pat, nodeMap, multiOut, builder as DxirEmitter,
                )
            }
            // (b) §0.4.152 / §0.4.161 — any region-bearing op whose region descendants
            // include a safeC5 WHILE. Build a replacement op whose regions clone-or-
            // rewrite each body op (recursing into nested region-bearing ops as needed).
            // §0.4.161 widens this from IF-only (§0.4.152) to any region-bearing op
            // (most importantly: WHILE, for the WHILE-inside-WHILE case).
            if (!op.hasRegions) return@rewriteFunction null
            if (!regionsContainSafeC5(op, safeC5)) return@rewriteFunction null
            val operandClones = op.operands.map { o ->
                resolveClonedOperand(o, nodeMap, op.id, multiOut)
            }
            val newRegions = op.regions.map { region ->
                rewriteRegionForC5(region, nodeMap, multiOut, builder, safeC5)
            }
            if (op.types.size == 1) {
                (builder as DxirEmitter).op(
                    op.op, operandClones, op.type, op.attrs, op.sharding, newRegions,
                )
            } else {
                (builder as DxirEmitter).opMulti(
                    op.op, operandClones, op.types, op.attrs, op.sharding, newRegions,
                )
            }
        }
    }

    /**
     * §0.4.152 — extract the per-WHILE C5 unroll body so it can run from either a
     * top-level [DxirBuilder] context or a nested [DxirRegionBuilder] context. The
     * unrolled iter clones land in [emitter]'s body; [multiOut] is populated with
     * the per-result-index replacement list and the smallest referenced index is
     * returned as the nominal node (multiOut takes precedence for `DxirOpResult`
     * resolution).
     */
    private fun unrollC5InEmitter(
        op: DxirOp,
        pattern: SimpleLoopPattern,
        nodeMap: MutableMap<Int, DxirNode>,
        multiOut: MutableMap<Int, List<DxirNode>>,
        emitter: DxirEmitter,
    ): DxirNode {
        val bodyBlock = op.regions[1].blocks.single()
        val bodyArgs = bodyBlock.args
        val argValues: MutableMap<Int, DxirNode> = HashMap()
        for (i in op.operands.indices) {
            argValues[bodyArgs[i].id] = nodeMap[op.operands[i].id]
                ?: error("C5: init id=${op.operands[i].id} (arg $i) missing from nodeMap")
        }
        for (k in 0 until pattern.tripCount) {
            val perIterMap = HashMap(nodeMap)
            for ((argId, value) in argValues) perIterMap[argId] = value
            for (n in bodyBlock.body) {
                perIterMap[n.id] = cloneNode(n, perIterMap, emitter, multiOut)
            }
            for (i in op.operands.indices) {
                argValues[bodyArgs[i].id] = perIterMap[bodyBlock.terminator[i].id]
                    ?: error("C5: back-edge arg=$i id missing after iter $k")
            }
        }
        val perIndex = List(op.operands.size) { i ->
            argValues[bodyArgs[i].id]
                ?: error("C5: back-edge arg=$i id missing after unroll")
        }
        multiOut[op.id] = perIndex
        return perIndex[pattern.referencedIndices.min()]
    }

    /**
     * §0.4.152 / §0.4.161 — true iff any descendant op (in any region of [op], at any
     * nesting depth) is a [safeC5]-keyed WHILE. Used to decide whether [op] needs
     * region-recursive C5 rewriting (else it's cloned verbatim). §0.4.152 named this
     * `ifRegionsContainSafeC5` and applied only to IF; §0.4.161's Phase 4b widens
     * to all region-bearing ops (WHILE, IF, future region-bearing kinds).
     */
    private fun regionsContainSafeC5(
        op: DxirOp,
        safeC5: Map<Int, SimpleLoopPattern>,
    ): Boolean {
        fun walk(nodes: List<DxirNode>): Boolean {
            for (n in nodes) {
                if (n !is DxirOp) continue
                if (n.id in safeC5) return true
                for (r in n.regions) for (b in r.blocks) {
                    if (walk(b.body)) return true
                }
            }
            return false
        }
        for (r in op.regions) for (b in r.blocks) {
            if (walk(b.body)) return true
        }
        return false
    }

    /**
     * §0.4.152 — clone [region] into [parentEmitter], rewriting safeC5 WHILEs to
     * unrolled chains and recursing into nested IFs that themselves contain safeC5
     * WHILEs. Mirrors [cloneRegion]'s terminator handling for `DxirOpResult` /
     * multi-result clones.
     */
    private fun rewriteRegionForC5(
        region: DxirRegion,
        outerNodeMap: MutableMap<Int, DxirNode>,
        multiOut: MutableMap<Int, List<DxirNode>>,
        parentEmitter: DxirEmitter,
        safeC5: Map<Int, SimpleLoopPattern>,
    ): DxirRegion {
        require(region.blocks.size == 1) {
            "rewriteRegionForC5: single-block regions only; got ${region.blocks.size}"
        }
        val origBlock = region.blocks.single()
        val regionLambda: DxirRegionBuilder.() -> Unit = {
            val regionNodeMap = HashMap(outerNodeMap)
            for (a in origBlock.args) {
                val newArg = arg(a.type, a.sharding)
                regionNodeMap[a.id] = newArg
            }
            for (n in origBlock.body) {
                val cloned: DxirNode = when {
                    n is DxirOp && safeC5[n.id] != null ->
                        unrollC5InEmitter(
                            n, safeC5[n.id]!!, regionNodeMap, multiOut, this as DxirEmitter,
                        )
                    // §0.4.161 — Phase 4b: any region-bearing op (IF or WHILE) with
                    // safeC5 descendants gets its regions rewritten in place. Previously
                    // (§0.4.152) this branch was IF-only. The unified arm uses
                    // `resolveClonedOperand` for each operand so DxirOpResult inits
                    // (e.g., a WHILE init that's `prevOp.result(k)`) preserve their
                    // index correctly.
                    n is DxirOp && n.hasRegions && regionsContainSafeC5(n, safeC5) -> {
                        val operandClones = n.operands.map { o ->
                            resolveClonedOperand(o, regionNodeMap, n.id, multiOut)
                        }
                        val newRegions = n.regions.map { r ->
                            rewriteRegionForC5(
                                r, regionNodeMap, multiOut, this as DxirEmitter, safeC5,
                            )
                        }
                        if (n.types.size == 1) {
                            (this as DxirEmitter).op(
                                n.op, operandClones, n.type, n.attrs, n.sharding, newRegions,
                            )
                        } else {
                            (this as DxirEmitter).opMulti(
                                n.op, operandClones, n.types, n.attrs, n.sharding, newRegions,
                            )
                        }
                    }
                    else -> cloneNode(n, regionNodeMap, this as DxirEmitter, multiOut)
                }
                regionNodeMap[n.id] = cloned
            }
            val terms = origBlock.terminator.map {
                if (it is DxirOpResult) {
                    multiOut[it.source.id]?.let { repl -> return@map repl[it.index] }
                }
                val mapped = regionNodeMap[it.id]
                    ?: error(
                        "rewriteRegionForC5: terminator id=${it.id} not in regionNodeMap",
                    )
                if (it !is DxirOpResult) return@map mapped
                when {
                    mapped is DxirOp -> mapped.result(it.index)
                    it.index == 0 -> mapped
                    else -> error(
                        "rewriteRegionForC5: terminator id=${it.id} index=${it.index} on " +
                            "non-Op clone (mapped=${mapped::class.simpleName})",
                    )
                }
            }
            yields(*terms.toTypedArray())
        }
        return when (parentEmitter) {
            is DxirBuilder -> parentEmitter.region(regionLambda)
            is DxirRegionBuilder -> parentEmitter.region(regionLambda)
            else -> error(
                "rewriteRegionForC5: unsupported emitter type ${parentEmitter::class}",
            )
        }
    }

    /**
     * Walk [fn]'s body + returns for references to [whileOp]'s results. Return the
     * index of the single non-counter result that is referenced, or null if zero / more
     * than one non-counter result is read. Counter-result references are disqualifying
     * (C5 drops the counter).
     *
     * Widened in §0.4.39 from the 2-carried-only `onlyCarriedResultReferenced` check:
     * previously the caller pre-computed `carriedIdx` via the "OTHER index" rule and
     * this function binary-validated it; now we discover the referenced index from the
     * use-sites directly, which generalises to N-carried WHILEs.
     */
    private fun findSingleReferencedCarried(
        fn: DxirFunction,
        whileOp: DxirOp,
    ): Int? = findReferencedCarried(fn, whileOp).singleOrNull()

    /**
     * §0.4.51 — return every result index of [whileOp] that is referenced downstream
     * (including the counter slot; the caller filters that out via the counter-idx
     * check). Used by C5's multi-result widening and by the C6/C7/C8/C9 single-result
     * guards (via [findSingleReferencedCarried]).
     */
    private fun findReferencedCarried(
        fn: DxirFunction,
        whileOp: DxirOp,
    ): Set<Int> {
        val whileId = whileOp.id
        val referenced = HashSet<Int>()
        fun checkRef(node: DxirNode) {
            if (node is DxirOpResult && node.source.id == whileId) {
                referenced += node.index
            } else if (node is DxirOp && node.id == whileId) {
                // Direct reference to a multi-result op implicitly targets index 0.
                referenced += 0
            }
        }
        fun walk(nodes: List<DxirNode>) {
            for (n in nodes) {
                if (n !is DxirOp) continue
                if (n === whileOp) continue
                for (o in n.operands) checkRef(o)
                for (r in n.regions) for (b in r.blocks) {
                    walk(b.body)
                    for (term in b.terminator) checkRef(term)
                }
            }
        }
        walk(fn.body)
        for (r in fn.returns) checkRef(r)
        return referenced
    }

    /**
     * Backwards-compatible wrapper for C6/C7/C8/C9: returns true when [whileOp]'s only
     * downstream reference is to result [carriedIdx]. Those corollaries still target
     * the 2-carried shape (one counter + one user-carried, §0.4.17's `detectAffineRecurrence`
     * hard-codes `op.operands.size == 2`), so this is equivalent to the §0.4.39 extension's
     * `findSingleReferencedCarried() == carriedIdx` check. When C6/C7/C8/C9 grow N-carried
     * support themselves, they should call [findSingleReferencedCarried] directly.
     */
    private fun onlyCarriedResultReferenced(
        fn: DxirFunction,
        whileOp: DxirOp,
        carriedIdx: Int,
    ): Boolean = findSingleReferencedCarried(fn, whileOp) == carriedIdx

    // ------------------------------------------------------------------------
    // C6 — affine constant-coefficient recurrence (Stage B.3, engine-backed):
    //   𝔏^n d = a · φ_L(p, d) + b  ⟹  d_exit = a^n · p + b · Σ_{i=0}^{n-1} a^i
    // (paper §4.3.2 + plan §4.9). Detects WHILEs whose carried back-edge is
    // `ADD(MUL(const_a, args[carried]), const_b)` (with optional simpler shapes:
    // pure `MUL` when b=0, pure `ADD` when a=1), lifts to symbolic via the engine,
    // closes the geometric series via `engine.sum`, and lowers back to dxir. Unlike
    // C5 (direct unroll), C6 produces a compact closed form regardless of trip count
    // — including for SYMBOLIC `n` (a runtime parameter), which C5 cannot handle.
    // ------------------------------------------------------------------------

    /** Concrete-or-symbolic trip count. Discriminated for the C6 lifting path. */
    private sealed interface TripCount {
        data class Concrete(val value: Int) : TripCount
        data class Symbolic(val node: DxirParam) : TripCount
    }

    /** Pattern detected by [detectAffineRecurrence]. */
    private data class AffineRecurrencePattern(
        val carriedIdx: Int,
        val counterIdx: Int,
        val tripCount: TripCount,
        /** The cloneable initial value of the carried — must be a `DxirParam` or `DxirConst`. */
        val carriedInit: DxirNode,
        /**
         * Multiplicative coefficient `a` — a loop-invariant subtree (doesn't depend on
         * counter or carried). Null when the pattern matched the `ADD(args[carried], b)`
         * shape (a=1 implicit). Can be a `DxirConst`, a `DxirParam`, or any arithmetic
         * op tree composed of loop-invariants (including function-param references).
         * Widened in §0.4.20 for BGDHyperOpt e2e — previously required `DxirConst`.
         *
         * §0.4.52 — null when the symbolic Symja-backed path matched (see [aSym]).
         */
        val aRoot: DxirNode?,
        /**
         * Additive coefficient `b` — loop-invariant subtree (same constraints as [aRoot]).
         * Null when the pattern matched `MUL(a, args[carried])` (b=0 implicit), or when
         * the symbolic path matched (see [bSym]).
         */
        val bRoot: DxirNode?,
        /**
         * §0.4.52 — symbolic multiplicative coefficient produced by the Symja-backed
         * detection path for user code that doesn't fit one of the three syntactic
         * back-edge shapes (e.g., `w = w - r·(2·(Sx2·w - Sxy))/M`). When non-null,
         * [aRoot] is null and [applyC6Pass] uses this directly as the `a_sym` symbolic
         * coefficient in the closed-form construction, skipping [liftOffsetSubtree].
         */
        val aSym: SymExpr? = null,
        /** §0.4.52 — symbolic additive coefficient; companion to [aSym]. */
        val bSym: SymExpr? = null,
        /**
         * §0.4.52 — params referenced by the lifted back-edge expression, collected
         * during Symja lifting. Used by [applyC6Pass] to build the `symbolMap` for
         * `lowerToDxir`. Empty set when the syntactic path matched.
         */
        val symLiftedParams: Set<DxirParam> = emptySet(),
        /**
         * §0.4.52 — opaque-leaf map for the Symja path: variable name → original dxir
         * node. Non-arithmetic ops (GATHER, SQRT, SCATTER, …) and unreduced
         * DxirOpResult refs that appear in the back-edge get assigned fresh Symja
         * symbols during lifting; this map routes them back to the original dxir
         * nodes during the `lowerToDxir` symbolMap construction.
         */
        val symOpaqueLeaves: Map<String, DxirNode> = emptyMap(),
    )

    /**
     * Recognise the C6 affine-recurrence pattern. Looser than C5's matcher in two ways:
     *  - Trip count `n` may be either a `DxirConst` (concrete) OR a `DxirParam`
     *    (symbolic, loop-invariant). Symbolic `n` lets C6 close loops whose trip count
     *    is unknown at compile time — the unique value-add over C5's direct unroll.
     *  - Carried back-edge must match `ADD(MUL(const_a, args[carried]), const_b)`,
     *    `MUL(const_a, args[carried])` (b=0), or `ADD(args[carried], const_b)` (a=1).
     *    `const_a` and `const_b` must be `DxirConst` with concrete float values.
     *
     * Counter pattern is the same as C5 (init=0, increment=1, `STEP(SUB(n, args[counter]))`).
     */
    private fun detectAffineRecurrence(op: DxirOp): AffineRecurrencePattern? {
        if (op.op != OpKind.WHILE) return null
        if (op.operands.size != 2) return null
        val condBlock = op.regions[0].blocks.single()
        val bodyBlock = op.regions[1].blocks.single()
        val condArgs = condBlock.args
        val bodyArgs = bodyBlock.args

        // Cond shape: STEP(SUB(n, args[counter])) where n is const OR loop-invariant param.
        val pred = condBlock.terminator.single() as? DxirOp ?: return null
        if (pred.op != OpKind.STEP) return null
        val sub = pred.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        val nNode = sub.operands[0]
        val counterArgRef = sub.operands[1]
        val counterIdx = condArgs.indexOfFirst { it.id == counterArgRef.id }
        if (counterIdx < 0) return null
        // Trip-count: const numeric OR param of any scalar dtype. Accept Float / Double
        // counters too — when the carried is F32, a uniformly-typed F32 counter avoids
        // the need for an i32→f32 CAST op (dxir doesn't have one yet, so heterogeneous-
        // type loops can't be constructed at all today).
        val tripCount: TripCount = extractTripCount(nNode) ?: return null

        // Counter init: const(0) of same dtype as the counter arg.
        val counterInit = op.operands[counterIdx] as? DxirConst ?: return null
        if ((counterInit.value as? Number)?.toDouble() != 0.0) return null

        // Counter back-edge: ADD(bodyArgs[counterIdx], const(1))
        val counterBackEdge = bodyBlock.terminator[counterIdx] as? DxirOp ?: return null
        if (counterBackEdge.op != OpKind.ADD) return null
        if (counterBackEdge.operands[0].id != bodyArgs[counterIdx].id) return null
        val incrConst = counterBackEdge.operands[1] as? DxirConst ?: return null
        if ((incrConst.value as? Number)?.toDouble() != 1.0) return null

        // Carried index is the OTHER one.
        val carriedIdx = if (counterIdx == 0) 1 else 0
        val carriedArg = bodyArgs[carriedIdx]
        val carriedBackEdgeRoot = bodyBlock.terminator[carriedIdx]

        // The carried back-edge MUST NOT reference args[counterIdx] (C6 is C5's hypothesis
        // restricted to affine f).
        val counterArgId = bodyArgs[counterIdx].id
        if (referencesId(carriedBackEdgeRoot, counterArgId, bodyBlock.body)) return null

        // Match one of three shapes for the carried back-edge:
        //   (1) ADD(MUL(a_subtree, args[carriedArg]), b_subtree)  — full affine
        //   (2) MUL(a_subtree, args[carriedArg])                  — b=0 implicit
        //   (3) ADD(args[carriedArg], b_subtree)                  — a=1 implicit
        // a_subtree / b_subtree may be any loop-invariant subtree (const, param, or
        // arithmetic op tree over those), not just `DxirConst`. The loop-invariance
        // check (no reference to carried or counter) is verified post-extraction.
        val (aRoot, bRoot) = extractAffineSubtrees(carriedBackEdgeRoot, carriedArg.id) ?: return null

        // a and b must not depend on the carried arg (otherwise the recurrence isn't
        // affine in the way C6 assumes). The outer back-edge check already excluded
        // counter-dependence, so we only re-check for carried here.
        if (aRoot != null && referencesId(aRoot, carriedArg.id, bodyBlock.body)) return null
        if (bRoot != null && referencesId(bRoot, carriedArg.id, bodyBlock.body)) return null

        // Carried init must be a clone-friendly node: a function param OR a const.
        val carriedInit = op.operands[carriedIdx]
        if (carriedInit !is DxirParam && carriedInit !is DxirConst) return null

        return AffineRecurrencePattern(
            carriedIdx = carriedIdx,
            counterIdx = counterIdx,
            tripCount = tripCount,
            carriedInit = carriedInit,
            aRoot = aRoot,
            bRoot = bRoot,
        )
    }

    /**
     * §0.4.52 — Symja-backed fallback detector for C6. Same counter pattern as
     * [detectAffineRecurrence]; differs in how the carried back-edge is recognized.
     *
     * Lifts the back-edge expression into [SymExpr] with the carried arg replaced by a
     * sentinel variable `_c6_w`, then uses `simplify(diff(expr, _c6_w))` to extract the
     * linear coefficient `a` and `simplify(substitute(expr, _c6_w → 0))` to extract the
     * constant `b`. The affine check is `containsVariable(a, _c6_w) == false`: if the
     * linear coefficient itself references `_c6_w`, the expression is quadratic or
     * higher in the carried and isn't a valid C6 recurrence.
     *
     * Matches user code like `w = w - r·(2·(Sx2·w - Sxy))/M` whose syntactic form
     * `SUB(w, expr(w))` isn't one of the three shapes [extractAffineSubtrees] handles,
     * but is algebraically affine (w · (1 - 2r·Sx2/M) + 2r·Sxy/M) after Symja normalizes.
     *
     * Returns null if:
     *   - The back-edge contains a dxir op [liftOffsetSubtree] doesn't know how to lift
     *     (e.g., GATHER, SCATTER inside the loop body).
     *   - The expression is non-affine in the carried (first derivative still depends
     *     on the carried).
     *   - The counter pattern doesn't match the standard C5/C6 shape.
     */
    private fun detectAffineRecurrenceViaSymja(
        op: DxirOp,
        engine: SymbolicEngine,
    ): AffineRecurrencePattern? {
        if (op.op != OpKind.WHILE) return null
        if (op.operands.size != 2) return null
        val condBlock = op.regions[0].blocks.single()
        val bodyBlock = op.regions[1].blocks.single()
        val condArgs = condBlock.args
        val bodyArgs = bodyBlock.args

        // Cond shape: STEP(SUB(n, args[counter])) — same as detectAffineRecurrence.
        val pred = condBlock.terminator.single() as? DxirOp ?: return null
        if (pred.op != OpKind.STEP) return null
        val sub = pred.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        val nNode = sub.operands[0]
        val counterArgRef = sub.operands[1]
        val counterIdx = condArgs.indexOfFirst { it.id == counterArgRef.id }
        if (counterIdx < 0) return null
        val tripCount = extractTripCount(nNode) ?: return null

        // Counter init: const(0).
        val counterInit = op.operands[counterIdx] as? DxirConst ?: return null
        if ((counterInit.value as? Number)?.toDouble() != 0.0) return null

        // Counter back-edge: ADD(bodyArgs[counterIdx], const(1)).
        val counterBackEdge = bodyBlock.terminator[counterIdx] as? DxirOp ?: return null
        if (counterBackEdge.op != OpKind.ADD) return null
        if (counterBackEdge.operands[0].id != bodyArgs[counterIdx].id) return null
        val incrConst = counterBackEdge.operands[1] as? DxirConst ?: return null
        if ((incrConst.value as? Number)?.toDouble() != 1.0) return null

        val carriedIdx = if (counterIdx == 0) 1 else 0
        val carriedArg = bodyArgs[carriedIdx]
        val carriedBackEdgeRoot = bodyBlock.terminator[carriedIdx]
        val counterArgId = bodyArgs[counterIdx].id
        if (referencesId(carriedBackEdgeRoot, counterArgId, bodyBlock.body)) return null

        val carriedInit = op.operands[carriedIdx]
        if (carriedInit !is DxirParam && carriedInit !is DxirConst) return null

        // Lift the back-edge into SymExpr. Carried arg → sentinel variable; counter is
        // blocked above so the sentinel counter symbol won't be reached.
        val wVar = engine.variable("_c6_w")
        val liftedParams = HashSet<DxirParam>()
        val opaqueLeaves = HashMap<String, DxirNode>()
        val sentinelCounterSym = engine.variable("__c6_unused_counter__")
        val backEdgeSym: SymExpr = try {
            liftOffsetSubtreeWithCarried(
                carriedBackEdgeRoot, carriedArg.id, wVar, counterArgId, sentinelCounterSym,
                engine, liftedParams, opaqueLeaves,
            )
        } catch (e: IllegalStateException) {
            return null
        }
        // §0.4.52 — the lifted opaque leaves must all be TOP-LEVEL (fn.body-scope)
        // nodes, not loop-varying ops inside the WHILE's own body region. If any leaf
        // refers to a node inside bodyBlock (including nested WHILEs whose results
        // change each iteration), the recurrence isn't truly affine in the carried.
        val bodyInternalIds = HashSet<Int>()
        for (n in bodyBlock.body) bodyInternalIds += n.id
        for (arg in bodyArgs) bodyInternalIds += arg.id
        for ((_, node) in opaqueLeaves) {
            val srcId = when (node) {
                is DxirOpResult -> node.source.id
                else -> node.id
            }
            if (srcId in bodyInternalIds) return null
        }

        val aSym = engine.simplify(engine.diff(backEdgeSym, wVar))
        if (engine.containsVariable(aSym, wVar)) return null
        val bSym = engine.simplify(
            engine.substitute(backEdgeSym, mapOf(wVar to engine.realLiteral(0.0))),
        )
        if (engine.containsVariable(bSym, wVar)) return null

        return AffineRecurrencePattern(
            carriedIdx = carriedIdx,
            counterIdx = counterIdx,
            tripCount = tripCount,
            carriedInit = carriedInit,
            aRoot = null,
            bRoot = null,
            aSym = aSym,
            bSym = bSym,
            symLiftedParams = liftedParams,
            symOpaqueLeaves = opaqueLeaves,
        )
    }

    /**
     * §0.4.52 — variant of [liftOffsetSubtree] that handles the carried-arg mapping
     * in addition to the counter. When `root.id == carriedArgId`, returns [carriedSym]
     * (the sentinel variable representing the carried in the symbolic lift).
     *
     * Non-arithmetic ops (GATHER, SCATTER, SQRT, etc.) are treated as OPAQUE LEAVES —
     * they're loop-invariant from the outer while's perspective (otherwise the
     * `referencesId(carriedBackEdgeRoot, counterArgId, ...)` check would have already
     * rejected the loop), so we assign each one a fresh Symja variable keyed by its
     * SSA id and add it to [opaqueLeaves] so the eventual `lowerToDxir` step can route
     * the symbolic variable back to the original dxir node.
     */
    private fun liftOffsetSubtreeWithCarried(
        root: DxirNode,
        carriedArgId: Int,
        carriedSym: SymExpr,
        counterArgId: Int,
        counterSymbol: SymExpr,
        engine: SymbolicEngine,
        paramsAccumulator: MutableSet<DxirParam>,
        opaqueLeaves: MutableMap<String, DxirNode>,
    ): SymExpr {
        if (root.id == carriedArgId) return carriedSym
        if (root.id == counterArgId) return counterSymbol
        return when (root) {
            is DxirConst -> {
                val v = (root.value as? Number)?.toDouble()
                    ?: error("liftOffsetSubtreeWithCarried: non-numeric const value ${root.value}")
                engine.realLiteral(v)
            }
            is DxirParam -> {
                paramsAccumulator.add(root)
                engine.variable(root.name)
            }
            is DxirOp -> {
                fun rec(child: DxirNode) = liftOffsetSubtreeWithCarried(
                    child, carriedArgId, carriedSym, counterArgId, counterSymbol,
                    engine, paramsAccumulator, opaqueLeaves,
                )
                when (root.op) {
                    OpKind.ADD -> engine.add(rec(root.operands[0]), rec(root.operands[1]))
                    OpKind.SUB -> engine.sub(rec(root.operands[0]), rec(root.operands[1]))
                    OpKind.MUL -> engine.mul(rec(root.operands[0]), rec(root.operands[1]))
                    OpKind.DIV -> engine.div(rec(root.operands[0]), rec(root.operands[1]))
                    OpKind.NEG -> engine.neg(rec(root.operands[0]))
                    OpKind.POW -> engine.pow(rec(root.operands[0]), rec(root.operands[1]))
                    else -> {
                        // Opaque leaf — treat as a fresh symbolic variable tied to the
                        // op's SSA id. Must be loop-invariant (no counter / carried
                        // refs); the caller's `referencesId` guard establishes that.
                        val name = "_c6_leaf_${root.id}"
                        opaqueLeaves[name] = root
                        engine.variable(name)
                    }
                }
            }
            is DxirOpResult -> {
                // A multi-result op that wasn't closed yet. Treat as opaque leaf keyed
                // by `id#index` so different indices of the same op get distinct syms.
                val name = "_c6_leaf_${root.source.id}_r${root.index}"
                opaqueLeaves[name] = root
                engine.variable(name)
            }
            else -> error(
                "liftOffsetSubtreeWithCarried: unsupported node ${root::class.simpleName}",
            )
        }
    }

    /**
     * Shared trip-count extractor used by both C6 (`detectAffineRecurrence`) and C7
     * (`detectIndexedAffineRecurrence`). Accepts any numeric scalar — int OR float —
     * so that C6/C7 fire on uniformly-typed F32-counter loops too (which is needed
     * until dxir grows an i32→f32 CAST op).
     */
    private fun extractTripCount(node: DxirNode): TripCount? {
        return when (node) {
            is DxirConst -> {
                val v = (node.value as? Number)?.toDouble() ?: return null
                if (v < 0.0 || v != v.toInt().toDouble()) return null
                TripCount.Concrete(v.toInt())
            }
            is DxirParam -> {
                if (!node.type.isScalar) return null
                TripCount.Symbolic(node)
            }
            else -> null
        }
    }

    /**
     * Extract the `(a_subtree, b_subtree)` coefficients from a carried back-edge root,
     * matching one of the three shapes documented in [detectAffineRecurrence]. Returns
     * null if the root doesn't fit any of the three patterns. A null element in the
     * returned pair means "implicit value": `aRoot = null` ⇒ a = 1; `bRoot = null` ⇒ b = 0.
     *
     * Subtrees may be any dxir node — `DxirConst`, `DxirParam`, or arithmetic `DxirOp`
     * trees over loop-invariants. Widened from the pre-§0.4.20 const-only version to
     * support BGDHyperOpt-shaped loops where coefficients are runtime-parameter
     * expressions (e.g., `a = 1 + 2r·Sx2/M`).
     */
    private fun extractAffineSubtrees(root: DxirNode, carriedArgId: Int): Pair<DxirNode?, DxirNode?>? {
        if (root !is DxirOp) return null
        return when (root.op) {
            OpKind.ADD -> {
                val lhs = root.operands[0]
                val rhs = root.operands[1]
                // Shape (1): ADD(MUL(a_subtree, args[carriedArg]), b_subtree)
                if (lhs is DxirOp && lhs.op == OpKind.MUL) {
                    val a = extractMulCoefficientSubtree(lhs, carriedArgId)
                    if (a != null) return a to rhs
                }
                // Shape (3): ADD(args[carriedArg], b_subtree) — a=1 implicit
                if (lhs.id == carriedArgId) return null to rhs
                null
            }
            OpKind.MUL -> {
                // Shape (2): MUL(a_subtree, args[carriedArg]) — b=0 implicit
                val a = extractMulCoefficientSubtree(root, carriedArgId)
                if (a != null) return a to null
                null
            }
            else -> null
        }
    }

    /**
     * Extract the `a_subtree` coefficient from a `MUL(a_subtree, args[carriedArgId])` op
     * (or its commutative twin). Returns null if neither operand is the carried arg.
     * Unlike the pre-§0.4.20 version which required a `DxirConst` coefficient, this
     * accepts any dxir node — the loop-invariance check happens at the pattern-match
     * call site.
     */
    private fun extractMulCoefficientSubtree(mul: DxirOp, carriedArgId: Int): DxirNode? {
        require(mul.op == OpKind.MUL)
        val lhs = mul.operands[0]
        val rhs = mul.operands[1]
        return when {
            rhs.id == carriedArgId -> lhs
            lhs.id == carriedArgId -> rhs
            else -> null
        }
    }

    /**
     * C6 rewrite pass — engine-backed closed-form construction. For each WHILE matching
     * [detectAffineRecurrence] and passing [onlyCarriedResultReferenced], lifts the
     * recurrence to symbolic form, computes `a^n * p + b * Σ_{i=0}^{n-1} a^i` via the
     * engine, simplifies, and lowers back to dxir.
     */
    private fun applyC6Pass(fn: DxirFunction, engine: SymbolicEngine): DxirFunction {
        val safeC6: Map<Int, AffineRecurrencePattern> = HashMap<Int, AffineRecurrencePattern>().apply {
            for (n in fn.body) {
                if (n !is DxirOp) continue
                if (n.op != OpKind.WHILE) continue
                // §0.4.52 — try syntactic detector first (cheap); fall back to the
                // Symja-backed path for kernels whose natural source form doesn't hit
                // the three pinned shapes (e.g., `w = w - r·(...)`).
                // §0.4.52 — try syntactic detector first (cheap); fall back to the
                // Symja-backed path for kernels whose natural source form doesn't hit
                // the three pinned shapes (e.g., `w = w - r·(...)`).
                val pattern = detectAffineRecurrence(n)
                    ?: detectAffineRecurrenceViaSymja(n, engine)
                    ?: continue
                if (!onlyCarriedResultReferenced(fn, n, pattern.carriedIdx)) continue
                this[n.id] = pattern
            }
        }
        if (safeC6.isEmpty()) return fn

        return rewriteFunction(fn) { op, nodeMap, _, builder ->
            val pattern = safeC6[op.id] ?: return@rewriteFunction null
            val resultType = op.types[pattern.carriedIdx]
            // Build the closed form symbolically.
            //   p_sym = lift(carriedInit) — variable when init is a param, rational when const
            //   n_sym = lift(tripCount)
            //   a_sym, b_sym = lift(aRoot/bRoot) via liftOffsetSubtree (handles const,
            //                  param, and arithmetic trees). Null subtrees → implicit 1 / 0.
            //   geom = engine.sum(λi. a^i, 0, n-1)
            //   closed = (a^n * p) + (b * geom)
            //   simplified = engine.simplify(closed)
            // Lower back to dxir with symbolMap = {paramName → clonedParam} including
            // any params encountered during a/b lifting.
            val (pSym, pSymbolName, pClone) = liftCarriedInit(pattern.carriedInit, nodeMap, engine)
            val (nSym, nSymbolName, nClone) = liftTripCount(pattern.tripCount, nodeMap, engine)
            val collectedParams = HashSet<DxirParam>()
            // The sentinel counter isn't used by the lift (we've verified a/b don't
            // reference the counter), so any unique placeholder id works.
            val sentinelCounterId = -1
            val sentinelCounterSym = engine.variable("__c6_unused_counter__")
            val aSym: SymExpr = when {
                // §0.4.52 — Symja-backed detection already produced aSym; reuse it and
                // merge its collected params into the main accumulator.
                pattern.aSym != null -> {
                    collectedParams += pattern.symLiftedParams
                    pattern.aSym
                }
                pattern.aRoot == null -> engine.realLiteral(1.0)
                else -> try {
                    liftOffsetSubtree(pattern.aRoot, sentinelCounterId, sentinelCounterSym, engine, collectedParams)
                } catch (e: IllegalStateException) {
                    return@rewriteFunction null
                }
            }
            val bSym: SymExpr = when {
                pattern.bSym != null -> pattern.bSym
                pattern.bRoot == null -> engine.realLiteral(0.0)
                else -> try {
                    liftOffsetSubtree(pattern.bRoot, sentinelCounterId, sentinelCounterSym, engine, collectedParams)
                } catch (e: IllegalStateException) {
                    return@rewriteFunction null
                }
            }
            val iVar = engine.variable("__c6_idx__")
            val powExpr = engine.pow(aSym, iVar)
            val geomFn = engine.closure("__c6_idx__", powExpr)
            val zero = engine.rational(0)
            val nMinusOne = engine.sub(nSym, engine.rational(1))
            val geomSum = engine.sum(geomFn, zero, nMinusOne)
            val closed = engine.add(
                engine.mul(engine.pow(aSym, nSym), pSym),
                engine.mul(bSym, geomSum),
            )
            // §0.4.52 — Symja's `Simplify` can produce the literal `Indeterminate` when
            // the closed-form geometric sum `(a^n - 1)/(a - 1)` has a division-by-zero
            // at a=1. For concrete a (param-subtree that's numerically != 1), the raw
            // closed form is still well-formed and lowers cleanly. Try simplified first;
            // if it stringifies as `Indeterminate`, fall back to the un-simplified form.
            val simplifiedMaybeInd = engine.simplify(closed)
            val simplified = if (simplifiedMaybeInd.toString() == "Indeterminate") closed else simplifiedMaybeInd
            val symbolMap = buildMap {
                if (pSymbolName != null && pClone != null) put(pSymbolName, pClone)
                if (nSymbolName != null && nClone != null) put(nSymbolName, nClone)
                for (param in collectedParams) {
                    val cloned = nodeMap[param.id]
                        ?: error("applyC6Pass: param '${param.name}' (id=${param.id}) not in nodeMap")
                    putIfAbsent(param.name, cloned)
                }
                // §0.4.52 — route opaque-leaf variables (GATHER, SQRT, DxirOpResult refs
                // to inner WHILE results) back to their original dxir nodes. Without this,
                // `lowerToDxir` would throw on the sentinel `_c6_leaf_<id>` symbols.
                for ((symName, originalNode) in pattern.symOpaqueLeaves) {
                    val cloned = resolveClonedOperand(originalNode, nodeMap as MutableMap<Int, DxirNode>, op.id)
                    putIfAbsent(symName, cloned)
                }
            }
            try {
                engine.lowerToDxir(simplified, resultType, builder as DxirBuilder, symbolMap)
            } catch (e: SymbolicEngine.LoweringException) {
                // Fall back to leaving the WHILE in place (C5 may still pick it up).
                null
            }
        }
    }

    /** Lift a carried-init dxir node ([DxirParam] or [DxirConst]) to a [SymExpr]. */
    private fun liftCarriedInit(
        init: DxirNode,
        nodeMap: Map<Int, DxirNode>,
        engine: SymbolicEngine,
    ): Triple<SymExpr, String?, DxirNode?> {
        return when (init) {
            is DxirParam -> {
                val clone = nodeMap[init.id]
                    ?: error("liftCarriedInit: param id=${init.id} not in nodeMap")
                Triple(engine.variable(init.name), init.name, clone)
            }
            is DxirConst -> {
                val v = (init.value as? Number)?.toDouble()
                    ?: error("liftCarriedInit: non-numeric const value ${init.value}")
                Triple(engine.realLiteral(v), null, null)
            }
            else -> error("liftCarriedInit: unsupported init kind ${init::class.simpleName}")
        }
    }

    /** Lift a trip-count [TripCount] to a [SymExpr] (concrete → rational; symbolic → variable). */
    private fun liftTripCount(
        tc: TripCount,
        nodeMap: Map<Int, DxirNode>,
        engine: SymbolicEngine,
    ): Triple<SymExpr, String?, DxirNode?> {
        return when (tc) {
            is TripCount.Concrete -> Triple(engine.rational(tc.value.toLong()), null, null)
            is TripCount.Symbolic -> {
                val clone = nodeMap[tc.node.id]
                    ?: error("liftTripCount: param id=${tc.node.id} not in nodeMap")
                Triple(engine.variable(tc.node.name), tc.node.name, clone)
            }
        }
    }

    // ------------------------------------------------------------------------
    // C7 — affine recurrence with counter-indexed offset (Stage B.3, engine-backed):
    //   𝔏^n d = a · φ_L(p, d) + b[i]  ⟹  d_exit = a^n · p + Σ_{i=0}^{n-1} a^i · b[n-1-i]
    // (paper §4.3.2 + plan §4.10). Generalises C6 by allowing the additive term to
    // depend on the loop counter `i`. Load-bearing for BGDHyperOpt's inner for-loop
    // (paper Fig. 6c lines 1-2): `d_3 = φ_i(0, d_3) + 2·x[i]·y[i] - 2·x[i]²·w_2`
    // closes via C7 with a=1 and b[i] = (the counter-indexed expression).
    //
    // Detection mirrors C6 but: (a) the additive term is NOT a const — it's a subtree
    // that references args[counter]; (b) the subtree must NOT reference args[carried];
    // (c) the lift requires mapping the counter block-arg to a fresh symbol so the
    // engine sees b as a function of a free variable.
    // ------------------------------------------------------------------------

    /** Pattern detected by [detectIndexedAffineRecurrence]. */
    private data class IndexedAffineRecurrencePattern(
        val carriedIdx: Int,
        val counterIdx: Int,
        val tripCount: TripCount,
        /** The cloneable initial value of the carried — must be a `DxirParam` or `DxirConst`. */
        val carriedInit: DxirNode,
        /** Multiplicative coefficient `a` (concrete float). */
        val constA: Float,
        /** The counter-dependent offset subtree (root). Lifted via [liftOffsetSubtree]. */
        val offsetRoot: DxirNode,
        /** The counter block-arg id; rewritten to a free symbol during the lift. */
        val counterArgId: Int,
    )

    /**
     * Recognise the C7 indexed-affine-recurrence pattern. Identical to [detectAffineRecurrence]
     * (same counter/trip/init detection) but the additive term in the back-edge must:
     *  - depend on `args[counter]` (otherwise C6 fires);
     *  - NOT depend on `args[carried]` (the C5/C6 hypothesis: the "f" applied to the
     *    carried value is purely affine in the carried, with the offset being a
     *    function of the iteration index alone).
     *
     * Carried back-edge shapes:
     *  - `ADD(MUL(const_a, args[carried]), offset_subtree)` — full affine with indexed offset
     *  - `ADD(args[carried], offset_subtree)` — a=1
     *
     * Note: the `MUL(const_a, args[carried])` (b=0) shape does NOT apply to C7 — there's
     * no offset to depend on the counter — so C7 doesn't match it; C6 would.
     */
    private fun detectIndexedAffineRecurrence(op: DxirOp): IndexedAffineRecurrencePattern? {
        if (op.op != OpKind.WHILE) return null
        if (op.operands.size != 2) return null
        val condBlock = op.regions[0].blocks.single()
        val bodyBlock = op.regions[1].blocks.single()
        val condArgs = condBlock.args
        val bodyArgs = bodyBlock.args

        // Counter+trip-count detection — uses the shared [extractTripCount] helper that
        // accepts any scalar numeric counter type (int or float).
        val pred = condBlock.terminator.single() as? DxirOp ?: return null
        if (pred.op != OpKind.STEP) return null
        val sub = pred.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        val nNode = sub.operands[0]
        val counterArgRef = sub.operands[1]
        val counterIdx = condArgs.indexOfFirst { it.id == counterArgRef.id }
        if (counterIdx < 0) return null
        val tripCount: TripCount = extractTripCount(nNode) ?: return null
        val counterInit = op.operands[counterIdx] as? DxirConst ?: return null
        if ((counterInit.value as? Number)?.toDouble() != 0.0) return null
        val counterBackEdge = bodyBlock.terminator[counterIdx] as? DxirOp ?: return null
        if (counterBackEdge.op != OpKind.ADD) return null
        if (counterBackEdge.operands[0].id != bodyArgs[counterIdx].id) return null
        val incrConst = counterBackEdge.operands[1] as? DxirConst ?: return null
        if ((incrConst.value as? Number)?.toDouble() != 1.0) return null

        val carriedIdx = if (counterIdx == 0) 1 else 0
        val carriedArg = bodyArgs[carriedIdx]
        val carriedBackEdgeRoot = bodyBlock.terminator[carriedIdx]
        val counterArgId = bodyArgs[counterIdx].id

        // Match the C7 back-edge shape and extract (a, offsetRoot).
        val (a, offsetRoot) = extractIndexedAffineParts(carriedBackEdgeRoot, carriedArg.id)
            ?: return null

        // Offset MUST depend on counter (otherwise C6 should have fired).
        if (!referencesId(offsetRoot, counterArgId, bodyBlock.body)) return null

        // Offset MUST NOT depend on carried (else this isn't an affine recurrence).
        if (referencesId(offsetRoot, carriedArg.id, bodyBlock.body)) return null

        val carriedInit = op.operands[carriedIdx]
        if (carriedInit !is DxirParam && carriedInit !is DxirConst) return null

        return IndexedAffineRecurrencePattern(
            carriedIdx = carriedIdx,
            counterIdx = counterIdx,
            tripCount = tripCount,
            carriedInit = carriedInit,
            constA = a,
            offsetRoot = offsetRoot,
            counterArgId = counterArgId,
        )
    }

    /**
     * Match the C7 back-edge shape and extract `(a, offset_subtree)`. Two shapes:
     *  - `ADD(MUL(const_a, args[carried]), offset_subtree)` → returns `(const_a, offset)`
     *  - `ADD(args[carried], offset_subtree)` → returns `(1f, offset)`
     *
     * C7's first-cut narrower than C6: requires the multiplicative coefficient `a` to
     * be a concrete `DxirConst`. Widening to subtree (like C6's §0.4.20 widening) would
     * collide with C8's pattern; deferred until a benchmark needs it.
     */
    private fun extractIndexedAffineParts(root: DxirNode, carriedArgId: Int): Pair<Float, DxirNode>? {
        if (root !is DxirOp) return null
        if (root.op != OpKind.ADD) return null
        val lhs = root.operands[0]
        val rhs = root.operands[1]
        // Shape: ADD(MUL(const_a, args[carriedArg]), offset_subtree)
        if (lhs is DxirOp && lhs.op == OpKind.MUL) {
            val a = extractMulConstCoefficient(lhs, carriedArgId)
            if (a != null) return a to rhs
        }
        // Shape: ADD(args[carriedArg], offset_subtree) — a=1
        if (lhs.id == carriedArgId) return 1f to rhs
        return null
    }

    /**
     * Narrow variant of the MUL-coefficient extractor that requires the coefficient to
     * be a concrete `DxirConst`. Used by C7's pattern matcher which (for first cut)
     * doesn't accept runtime-param `a` subtrees. Distinct from C6's widened
     * `extractMulCoefficientSubtree` which accepts any node.
     */
    private fun extractMulConstCoefficient(mul: DxirOp, carriedArgId: Int): Float? {
        require(mul.op == OpKind.MUL)
        val lhs = mul.operands[0]
        val rhs = mul.operands[1]
        return when {
            lhs is DxirConst && rhs.id == carriedArgId -> (lhs.value as? Number)?.toFloat()
            rhs is DxirConst && lhs.id == carriedArgId -> (rhs.value as? Number)?.toFloat()
            else -> null
        }
    }

    /**
     * Lift a counter-indexed offset subtree to a [SymExpr], mapping `args[counterArgId]`
     * to `counterSymbol`. Recurses through arithmetic ops and gathers any [DxirParam]
     * references encountered into [paramsAccumulator] for symbolMap reconstruction at
     * lowering time.
     *
     * Supported op kinds: ADD, SUB, MUL, DIV, NEG, POW. Other ops (SQRT/EXP/LOG/etc.)
     * error explicitly — Stage B.3 first-cut focus is the BGDHyperOpt inner-loop shape
     * (polynomial in the counter); other shapes wait for a benchmark that needs them.
     */
    private fun liftOffsetSubtree(
        root: DxirNode,
        counterArgId: Int,
        counterSymbol: SymExpr,
        engine: SymbolicEngine,
        paramsAccumulator: MutableSet<DxirParam>,
    ): SymExpr {
        if (root.id == counterArgId) return counterSymbol
        return when (root) {
            is DxirConst -> {
                val v = (root.value as? Number)?.toDouble()
                    ?: error("liftOffsetSubtree: non-numeric const value ${root.value}")
                engine.realLiteral(v)
            }
            is DxirParam -> {
                paramsAccumulator.add(root)
                engine.variable(root.name)
            }
            is DxirOp -> when (root.op) {
                OpKind.ADD -> engine.add(
                    liftOffsetSubtree(root.operands[0], counterArgId, counterSymbol, engine, paramsAccumulator),
                    liftOffsetSubtree(root.operands[1], counterArgId, counterSymbol, engine, paramsAccumulator),
                )
                OpKind.SUB -> engine.sub(
                    liftOffsetSubtree(root.operands[0], counterArgId, counterSymbol, engine, paramsAccumulator),
                    liftOffsetSubtree(root.operands[1], counterArgId, counterSymbol, engine, paramsAccumulator),
                )
                OpKind.MUL -> engine.mul(
                    liftOffsetSubtree(root.operands[0], counterArgId, counterSymbol, engine, paramsAccumulator),
                    liftOffsetSubtree(root.operands[1], counterArgId, counterSymbol, engine, paramsAccumulator),
                )
                OpKind.DIV -> engine.div(
                    liftOffsetSubtree(root.operands[0], counterArgId, counterSymbol, engine, paramsAccumulator),
                    liftOffsetSubtree(root.operands[1], counterArgId, counterSymbol, engine, paramsAccumulator),
                )
                OpKind.NEG -> engine.neg(
                    liftOffsetSubtree(root.operands[0], counterArgId, counterSymbol, engine, paramsAccumulator),
                )
                OpKind.POW -> engine.pow(
                    liftOffsetSubtree(root.operands[0], counterArgId, counterSymbol, engine, paramsAccumulator),
                    liftOffsetSubtree(root.operands[1], counterArgId, counterSymbol, engine, paramsAccumulator),
                )
                else -> error(
                    "liftOffsetSubtree: op kind ${root.op} not yet supported for C7 lift " +
                        "(B.3 first-cut handles arithmetic only)",
                )
            }
            else -> error(
                "liftOffsetSubtree: unsupported node kind ${root::class.simpleName} (id=${root.id})",
            )
        }
    }

    /**
     * C7 rewrite pass — engine-backed closed-form construction for indexed-affine
     * recurrences. For each WHILE matching [detectIndexedAffineRecurrence] and passing
     * [onlyCarriedResultReferenced]: lift the offset subtree to symbolic with the counter
     * mapped to a fresh `k` variable; build `Σ_{i=0}^{n-1} a^i · b[n-1-i]` where b[j] is
     * the offset evaluated at counter=j; add the homogeneous term `a^n · p`; simplify;
     * lower back to dxir.
     */
    private fun applyC7Pass(fn: DxirFunction, engine: SymbolicEngine): DxirFunction {
        val safeC7: Map<Int, IndexedAffineRecurrencePattern> = HashMap<Int, IndexedAffineRecurrencePattern>().apply {
            for (n in fn.body) {
                if (n !is DxirOp) continue
                if (n.op != OpKind.WHILE) continue
                val pattern = detectIndexedAffineRecurrence(n) ?: continue
                if (!onlyCarriedResultReferenced(fn, n, pattern.carriedIdx)) continue
                this[n.id] = pattern
            }
        }
        if (safeC7.isEmpty()) return fn

        return rewriteFunction(fn) { op, nodeMap, _, builder ->
            val pattern = safeC7[op.id] ?: return@rewriteFunction null
            val resultType = op.types[pattern.carriedIdx]
            val (pSym, pSymbolName, pClone) = liftCarriedInit(pattern.carriedInit, nodeMap, engine)
            val (nSym, nSymbolName, nClone) = liftTripCount(pattern.tripCount, nodeMap, engine)
            val aSym = engine.realLiteral(pattern.constA.toDouble())
            // Lift the offset subtree with the counter mapped to a fresh symbol `k`.
            val kSym = engine.variable("__c7_k__")
            val collectedParams = HashSet<DxirParam>()
            val offsetLifted: SymExpr = try {
                liftOffsetSubtree(pattern.offsetRoot, pattern.counterArgId, kSym, engine, collectedParams)
            } catch (e: IllegalStateException) {
                // Unsupported op kind in the offset subtree — fall back, leave WHILE.
                return@rewriteFunction null
            }
            // Build the summand `a^i · offset[n-1-i]`. The offset is `b(k)` lifted; we
            // need `b(n-1-i)` so substitute `k → (n-1-i)`.
            val iSym = engine.variable("__c7_i__")
            val nMinus1 = engine.sub(nSym, engine.rational(1))
            val nMinus1MinusI = engine.sub(nMinus1, iSym)
            val offsetAtIndex = engine.substitute(offsetLifted, mapOf(kSym to nMinus1MinusI))
            val summand = engine.mul(engine.pow(aSym, iSym), offsetAtIndex)
            val summandFn = engine.closure("__c7_i__", summand)
            val sumExpr = engine.sum(summandFn, engine.rational(0), nMinus1)
            val homogeneous = engine.mul(engine.pow(aSym, nSym), pSym)
            val closed = engine.add(homogeneous, sumExpr)
            val simplified = engine.simplify(closed)
            // SymbolMap: p, n (if symbolic) + every param the offset lift encountered.
            val symbolMap = buildMap {
                if (pSymbolName != null && pClone != null) put(pSymbolName, pClone)
                if (nSymbolName != null && nClone != null) put(nSymbolName, nClone)
                for (param in collectedParams) {
                    val cloned = nodeMap[param.id]
                        ?: error("applyC7Pass: param '${param.name}' (id=${param.id}) not in nodeMap")
                    // Avoid stomping the p / n entries with a same-named param (shouldn't
                    // happen in well-formed dxir but defensive).
                    putIfAbsent(param.name, cloned)
                }
            }
            try {
                engine.lowerToDxir(simplified, resultType, builder as DxirBuilder, symbolMap)
            } catch (e: SymbolicEngine.LoweringException) {
                // Symja couldn't close the sum to a form lowerToDxir handles — fall back.
                null
            }
        }
    }

    // ------------------------------------------------------------------------
    // C8 — variable-coefficient affine recurrence (Stage B.3, engine-backed):
    //   𝔏^n d = a[i] · φ_L(p, d) + b[i]
    //   ⟹ d_exit = p · ∏_{i=0}^{n-1} a[i] + Σ_{k=0}^{n-1} b[k] · ∏_{j=k+1}^{n-1} a[j]
    // (paper §4.3.2 + plan §4.11; closed form verified by hand-iteration — the paper
    // Fig. 5 transcription had ambiguous indexing, so we implement the canonical
    // Horner-like closed form that matches step-by-step iteration on test cases.)
    //
    // Generalises C7 by allowing the multiplicative coefficient `a` to also depend on
    // the counter. The pattern-matcher discriminates: if `a` is a `DxirConst`, C7 fires;
    // if `a` is a counter-dependent subtree, C8 fires. The constructions are closely
    // related — both lift counter-dependent subtrees, both use engine.sum with a
    // substitute-based re-binding — but C8 additionally uses engine.product for the
    // ∏ a[i] terms.
    // ------------------------------------------------------------------------

    /** Pattern detected by [detectVariableCoefficientRecurrence]. */
    private data class VariableCoefficientPattern(
        val carriedIdx: Int,
        val counterIdx: Int,
        val tripCount: TripCount,
        /** The cloneable initial value of the carried — must be a `DxirParam` or `DxirConst`. */
        val carriedInit: DxirNode,
        /** The counter-dependent multiplicative coefficient subtree (root). */
        val aRoot: DxirNode,
        /** The additive offset subtree (root). May or may not depend on counter. */
        val bRoot: DxirNode,
        /** The counter block-arg id; rewritten to a free symbol during the lift. */
        val counterArgId: Int,
    )

    /**
     * Recognise the C8 variable-coefficient pattern. Same counter/trip-count detection
     * as C6/C7. The carried back-edge must match `ADD(MUL(a_subtree, args[carried]), b_subtree)`
     * with the following distinguishing constraints (vs C6/C7):
     *  - `a_subtree` MUST depend on `args[counter]` (otherwise C6 or C7 fires: C6 when
     *    a_subtree is a `DxirConst`, C7 when a_subtree is a const but b_subtree depends
     *    on counter — both pre-empt C8 in the pipeline).
     *  - Neither subtree may reference `args[carried]` (standard C5/C6/C7/C8 hypothesis).
     *
     * Only the full affine form `ADD(MUL(a, carried), b)` matches — the degenerate
     * shapes (a=1 implicit, b=0 implicit) are already handled by C6/C7/C8-with-simpler-
     * constructions; no reason to re-match them here.
     */
    private fun detectVariableCoefficientRecurrence(op: DxirOp): VariableCoefficientPattern? {
        if (op.op != OpKind.WHILE) return null
        if (op.operands.size != 2) return null
        val condBlock = op.regions[0].blocks.single()
        val bodyBlock = op.regions[1].blocks.single()
        val condArgs = condBlock.args
        val bodyArgs = bodyBlock.args

        // Counter+trip-count detection — shared helper.
        val pred = condBlock.terminator.single() as? DxirOp ?: return null
        if (pred.op != OpKind.STEP) return null
        val sub = pred.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        val nNode = sub.operands[0]
        val counterArgRef = sub.operands[1]
        val counterIdx = condArgs.indexOfFirst { it.id == counterArgRef.id }
        if (counterIdx < 0) return null
        val tripCount: TripCount = extractTripCount(nNode) ?: return null
        val counterInit = op.operands[counterIdx] as? DxirConst ?: return null
        if ((counterInit.value as? Number)?.toDouble() != 0.0) return null
        val counterBackEdge = bodyBlock.terminator[counterIdx] as? DxirOp ?: return null
        if (counterBackEdge.op != OpKind.ADD) return null
        if (counterBackEdge.operands[0].id != bodyArgs[counterIdx].id) return null
        val incrConst = counterBackEdge.operands[1] as? DxirConst ?: return null
        if ((incrConst.value as? Number)?.toDouble() != 1.0) return null

        val carriedIdx = if (counterIdx == 0) 1 else 0
        val carriedArg = bodyArgs[carriedIdx]
        val carriedBackEdgeRoot = bodyBlock.terminator[carriedIdx]
        val counterArgId = bodyArgs[counterIdx].id

        // Match `ADD(MUL(a_subtree, args[carried]), b_subtree)` and extract (a, b).
        val (aRoot, bRoot) = extractVariableCoefficientParts(carriedBackEdgeRoot, carriedArg.id)
            ?: return null

        // C8's discriminating constraint: a must depend on the counter (otherwise C6/C7).
        if (!referencesId(aRoot, counterArgId, bodyBlock.body)) return null

        // Neither a nor b may depend on the carried arg (standard affine hypothesis).
        if (referencesId(aRoot, carriedArg.id, bodyBlock.body)) return null
        if (referencesId(bRoot, carriedArg.id, bodyBlock.body)) return null

        val carriedInit = op.operands[carriedIdx]
        if (carriedInit !is DxirParam && carriedInit !is DxirConst) return null

        return VariableCoefficientPattern(
            carriedIdx = carriedIdx,
            counterIdx = counterIdx,
            tripCount = tripCount,
            carriedInit = carriedInit,
            aRoot = aRoot,
            bRoot = bRoot,
            counterArgId = counterArgId,
        )
    }

    /**
     * Match the C8 back-edge shape `ADD(MUL(a_subtree, args[carried]), b_subtree)` and
     * return `(a_subtree, b_subtree)`. The MUL's carried-arg position can be either
     * operand (commutative); both operands are tried.
     */
    private fun extractVariableCoefficientParts(
        root: DxirNode,
        carriedArgId: Int,
    ): Pair<DxirNode, DxirNode>? {
        if (root !is DxirOp) return null
        if (root.op != OpKind.ADD) return null
        val lhs = root.operands[0]
        val rhs = root.operands[1]
        if (lhs !is DxirOp || lhs.op != OpKind.MUL) return null
        val mulLhs = lhs.operands[0]
        val mulRhs = lhs.operands[1]
        val aRoot = when {
            mulRhs.id == carriedArgId -> mulLhs
            mulLhs.id == carriedArgId -> mulRhs
            else -> return null
        }
        return aRoot to rhs
    }

    /**
     * C8 rewrite pass — engine-backed closed-form construction for variable-coefficient
     * affine recurrences. Closed form (derivation-verified, see §0.4.18):
     *   `d_exit = p · ∏_{i=0}^{n-1} a[i] + Σ_{k=0}^{n-1} b[k] · ∏_{j=k+1}^{n-1} a[j]`
     */
    private fun applyC8Pass(fn: DxirFunction, engine: SymbolicEngine): DxirFunction {
        val safeC8: Map<Int, VariableCoefficientPattern> = HashMap<Int, VariableCoefficientPattern>().apply {
            for (n in fn.body) {
                if (n !is DxirOp) continue
                if (n.op != OpKind.WHILE) continue
                val pattern = detectVariableCoefficientRecurrence(n) ?: continue
                if (!onlyCarriedResultReferenced(fn, n, pattern.carriedIdx)) continue
                this[n.id] = pattern
            }
        }
        if (safeC8.isEmpty()) return fn

        return rewriteFunction(fn) { op, nodeMap, _, builder ->
            val pattern = safeC8[op.id] ?: return@rewriteFunction null
            val resultType = op.types[pattern.carriedIdx]
            val (pSym, pSymbolName, pClone) = liftCarriedInit(pattern.carriedInit, nodeMap, engine)
            val (nSym, nSymbolName, nClone) = liftTripCount(pattern.tripCount, nodeMap, engine)

            // Lift a and b with the counter mapped to fresh placeholder symbols (distinct
            // so substitutions into each don't collide).
            val aPlaceholder = engine.variable("__c8_aidx__")
            val bPlaceholder = engine.variable("__c8_bidx__")
            val collectedParams = HashSet<DxirParam>()
            val aLifted: SymExpr = try {
                liftOffsetSubtree(pattern.aRoot, pattern.counterArgId, aPlaceholder, engine, collectedParams)
            } catch (e: IllegalStateException) {
                return@rewriteFunction null
            }
            val bLifted: SymExpr = try {
                liftOffsetSubtree(pattern.bRoot, pattern.counterArgId, bPlaceholder, engine, collectedParams)
            } catch (e: IllegalStateException) {
                return@rewriteFunction null
            }

            val nMinus1 = engine.sub(nSym, engine.rational(1))
            // Full product Π_{i=0}^{n-1} a[i]. Bound variable: __c8_k__.
            val kForProduct = engine.variable("__c8_k__")
            val aAtK = engine.substitute(aLifted, mapOf(aPlaceholder to kForProduct))
            val fullProductClosure = engine.closure("__c8_k__", aAtK)
            val fullProduct = engine.product(fullProductClosure, engine.rational(0), nMinus1)

            // Outer sum Σ_{k=0}^{n-1} b[k] · ∏_{j=k+1}^{n-1} a[j]
            val jForInnerProduct = engine.variable("__c8_j__")
            val aAtJ = engine.substitute(aLifted, mapOf(aPlaceholder to jForInnerProduct))
            val innerProductClosure = engine.closure("__c8_j__", aAtJ)
            val kForSum = engine.variable("__c8_k__")
            val innerProduct = engine.product(
                innerProductClosure,
                engine.add(kForSum, engine.rational(1)),
                nMinus1,
            )
            val bAtK = engine.substitute(bLifted, mapOf(bPlaceholder to kForSum))
            val sumBody = engine.mul(bAtK, innerProduct)
            val outerSumClosure = engine.closure("__c8_k__", sumBody)
            val outerSum = engine.sum(outerSumClosure, engine.rational(0), nMinus1)

            val pTerm = engine.mul(pSym, fullProduct)
            val closed = engine.add(pTerm, outerSum)
            val simplified = engine.simplify(closed)

            val symbolMap = buildMap {
                if (pSymbolName != null && pClone != null) put(pSymbolName, pClone)
                if (nSymbolName != null && nClone != null) put(nSymbolName, nClone)
                for (param in collectedParams) {
                    val cloned = nodeMap[param.id]
                        ?: error("applyC8Pass: param '${param.name}' (id=${param.id}) not in nodeMap")
                    putIfAbsent(param.name, cloned)
                }
            }
            try {
                engine.lowerToDxir(simplified, resultType, builder as DxirBuilder, symbolMap)
            } catch (e: SymbolicEngine.LoweringException) {
                // Symja couldn't close the product/sum — fall back, leaving WHILE for C5.
                null
            }
        }
    }

    // ------------------------------------------------------------------------
    // C9 — power-form recurrence (Stage B.3, engine-backed):
    //   𝔏^n d = a · (φ_L(p, d))^b
    //   ⟹ d_exit = a^(Σ_{i=0}^{n-1} b^i) · p^(b^n)
    //          = a^((b^n - 1)/(b - 1)) · p^(b^n)  for b ≠ 1
    //          = a^n · p                          for b = 1
    // (paper §4.3.2 + plan §4.12; closed form derivation-verified by hand at multiple
    // (a, b, n, p) configurations. The §0.4.11 paper transcription `a^{b+n-1} · p^{b^n}`
    // is wrong — fails at n=3, b=2 (gives a^4 vs the correct a^7); see §0.4.19 inline
    // derivation. Implemented form uses `engine.sum` for the geometric exponent on `a`,
    // which Symja closes for concrete b regardless of n's concreteness.)
    // ------------------------------------------------------------------------

    /** Pattern detected by [detectPowerFormRecurrence]. */
    private data class PowerFormPattern(
        val carriedIdx: Int,
        val counterIdx: Int,
        val tripCount: TripCount,
        val carriedInit: DxirNode,
        /** Multiplicative coefficient `a` (concrete float). */
        val constA: Float,
        /** Power exponent `b` (concrete float). */
        val constB: Float,
    )

    /**
     * Recognise the C9 power-form pattern. Same counter/trip-count detection as
     * C6/C7/C8. The carried back-edge must match `MUL(const_a, POW(args[carried], const_b))`
     * (or its commutative twin) or `POW(args[carried], const_b)` (a=1 implicit).
     * Both `const_a` and `const_b` must be `DxirConst` with concrete float values.
     *
     * No discriminating constraint vs C6/C7/C8 — those matchers all expect ADD or MUL
     * at the root with the carried arg as a direct MUL operand, not nested through POW.
     * So C9's `MUL(_, POW(carried, _))` shape is structurally distinct.
     */
    private fun detectPowerFormRecurrence(op: DxirOp): PowerFormPattern? {
        if (op.op != OpKind.WHILE) return null
        if (op.operands.size != 2) return null
        val condBlock = op.regions[0].blocks.single()
        val bodyBlock = op.regions[1].blocks.single()
        val condArgs = condBlock.args
        val bodyArgs = bodyBlock.args

        // Counter+trip-count detection — shared helper.
        val pred = condBlock.terminator.single() as? DxirOp ?: return null
        if (pred.op != OpKind.STEP) return null
        val sub = pred.operands[0] as? DxirOp ?: return null
        if (sub.op != OpKind.SUB) return null
        val nNode = sub.operands[0]
        val counterArgRef = sub.operands[1]
        val counterIdx = condArgs.indexOfFirst { it.id == counterArgRef.id }
        if (counterIdx < 0) return null
        val tripCount: TripCount = extractTripCount(nNode) ?: return null
        val counterInit = op.operands[counterIdx] as? DxirConst ?: return null
        if ((counterInit.value as? Number)?.toDouble() != 0.0) return null
        val counterBackEdge = bodyBlock.terminator[counterIdx] as? DxirOp ?: return null
        if (counterBackEdge.op != OpKind.ADD) return null
        if (counterBackEdge.operands[0].id != bodyArgs[counterIdx].id) return null
        val incrConst = counterBackEdge.operands[1] as? DxirConst ?: return null
        if ((incrConst.value as? Number)?.toDouble() != 1.0) return null

        val carriedIdx = if (counterIdx == 0) 1 else 0
        val carriedArg = bodyArgs[carriedIdx]
        val carriedBackEdgeRoot = bodyBlock.terminator[carriedIdx]
        val counterArgId = bodyArgs[counterIdx].id

        // The carried back-edge MUST NOT reference args[counterIdx] (C9 hypothesis: `f`
        // is purely a power-form function of the carried, not of the iteration index).
        if (referencesId(carriedBackEdgeRoot, counterArgId, bodyBlock.body)) return null

        val (a, b) = extractPowerFormCoefficients(carriedBackEdgeRoot, carriedArg.id) ?: return null

        val carriedInit = op.operands[carriedIdx]
        if (carriedInit !is DxirParam && carriedInit !is DxirConst) return null

        return PowerFormPattern(
            carriedIdx = carriedIdx,
            counterIdx = counterIdx,
            tripCount = tripCount,
            carriedInit = carriedInit,
            constA = a,
            constB = b,
        )
    }

    /**
     * Match the C9 back-edge shape and extract `(a, b)`. Three shapes:
     *  - `MUL(const_a, POW(args[carried], const_b))`         — full power form
     *  - `MUL(POW(args[carried], const_b), const_a)`         — commutative twin
     *  - `POW(args[carried], const_b)`                       — a=1 implicit
     */
    private fun extractPowerFormCoefficients(
        root: DxirNode,
        carriedArgId: Int,
    ): Pair<Float, Float>? {
        if (root !is DxirOp) return null
        return when (root.op) {
            OpKind.MUL -> {
                val lhs = root.operands[0]
                val rhs = root.operands[1]
                val (constSide, powSide) = when {
                    lhs is DxirConst && rhs is DxirOp && rhs.op == OpKind.POW -> lhs to rhs
                    rhs is DxirConst && lhs is DxirOp && lhs.op == OpKind.POW -> rhs to lhs
                    else -> return null
                }
                val a = (constSide.value as? Number)?.toFloat() ?: return null
                val b = extractPowExponent(powSide, carriedArgId) ?: return null
                a to b
            }
            OpKind.POW -> {
                val b = extractPowExponent(root, carriedArgId) ?: return null
                1f to b
            }
            else -> null
        }
    }

    /**
     * Extract the exponent from a `POW(args[carriedArgId], const_b)` op. Returns null
     * if the base isn't the carried arg or the exponent isn't a concrete float const.
     */
    private fun extractPowExponent(pow: DxirOp, carriedArgId: Int): Float? {
        require(pow.op == OpKind.POW)
        if (pow.operands[0].id != carriedArgId) return null
        val expConst = pow.operands[1] as? DxirConst ?: return null
        return (expConst.value as? Number)?.toFloat()
    }

    /**
     * C9 rewrite pass — engine-backed closed-form construction for power-form
     * recurrences. Closed form: `d_exit = a^(Σ_{i=0}^{n-1} b^i) · p^(b^n)`.
     */
    private fun applyC9Pass(fn: DxirFunction, engine: SymbolicEngine): DxirFunction {
        val safeC9: Map<Int, PowerFormPattern> = HashMap<Int, PowerFormPattern>().apply {
            for (n in fn.body) {
                if (n !is DxirOp) continue
                if (n.op != OpKind.WHILE) continue
                val pattern = detectPowerFormRecurrence(n) ?: continue
                if (!onlyCarriedResultReferenced(fn, n, pattern.carriedIdx)) continue
                this[n.id] = pattern
            }
        }
        if (safeC9.isEmpty()) return fn

        return rewriteFunction(fn) { op, nodeMap, _, builder ->
            val pattern = safeC9[op.id] ?: return@rewriteFunction null
            val resultType = op.types[pattern.carriedIdx]
            val (pSym, pSymbolName, pClone) = liftCarriedInit(pattern.carriedInit, nodeMap, engine)
            val (nSym, nSymbolName, nClone) = liftTripCount(pattern.tripCount, nodeMap, engine)
            val aSym = engine.realLiteral(pattern.constA.toDouble())
            val bSym = engine.realLiteral(pattern.constB.toDouble())

            val nMinus1 = engine.sub(nSym, engine.rational(1))
            // Exponent on `a`: Σ_{i=0}^{n-1} b^i (the geometric series).
            val iSym = engine.variable("__c9_i__")
            val bPowI = engine.pow(bSym, iSym)
            val expAClosure = engine.closure("__c9_i__", bPowI)
            val expA = engine.sum(expAClosure, engine.rational(0), nMinus1)
            // Exponent on `p`: b^n.
            val expP = engine.pow(bSym, nSym)
            val closed = engine.mul(engine.pow(aSym, expA), engine.pow(pSym, expP))
            val simplified = engine.simplify(closed)

            val symbolMap = buildMap {
                if (pSymbolName != null && pClone != null) put(pSymbolName, pClone)
                if (nSymbolName != null && nClone != null) put(nSymbolName, nClone)
            }
            try {
                engine.lowerToDxir(simplified, resultType, builder as DxirBuilder, symbolMap)
            } catch (e: SymbolicEngine.LoweringException) {
                null
            }
        }
    }

    // ------------------------------------------------------------------------
    // Framework: clone-and-rewrite walker
    // ------------------------------------------------------------------------

    /**
     * One-shot rewrite walker. Clones [fn] into a fresh [DxirBuilder]; for each old
     * body op, calls [rewrite] to optionally replace it with a custom node. If
     * [rewrite] returns null, the op is cloned verbatim. The replacement node (if any)
     * is recorded in `nodeMap` under the old op's id so subsequent references resolve
     * to the new node.
     *
     * Dead ops (cloned verbatim but never referenced by downstream) are tolerated —
     * the [DxirInterpreter] walks them but their results are never read. Stage B.3 DCE
     * pass will strip them.
     */
    private fun rewriteFunction(
        fn: DxirFunction,
        rewrite: DxirBuilder.(
            op: DxirOp,
            nodeMap: MutableMap<Int, DxirNode>,
            multiOut: MutableMap<Int, List<DxirNode>>,
            builder: DxirBuilder,
        ) -> DxirNode?,
    ): DxirFunction {
        return DxirBuilder.function(fn.name) {
            val nodeMap = HashMap<Int, DxirNode>()
            // §0.4.51 — when a rewrite closes a multi-result op (e.g., C5 unrolling a
            // multi-carried WHILE with ≥2 non-counter downstream refs), it populates
            // this map with one replacement node per result index. cloneNode consults
            // [multiOut] ahead of [nodeMap] so DxirOpResult operands resolve to the
            // correct per-index value rather than collapsing to index 0.
            val multiOut = HashMap<Int, List<DxirNode>>()
            for (p in fn.params) {
                nodeMap[p.id] = param(p.name, p.type, p.sharding)
            }
            for (oldNode in fn.body) {
                val cloned: DxirNode = when (oldNode) {
                    is DxirOp -> {
                        val rewritten = rewrite(oldNode, nodeMap, multiOut, this)
                        rewritten ?: cloneNode(oldNode, nodeMap, this as DxirEmitter, multiOut)
                    }
                    else -> cloneNode(oldNode, nodeMap, this as DxirEmitter, multiOut)
                }
                nodeMap[oldNode.id] = cloned
            }
            fn.returns.map { resolveReturn(it, nodeMap, multiOut) }
        }
    }

    private fun resolveReturn(
        ret: DxirNode,
        nodeMap: Map<Int, DxirNode>,
        multiOut: Map<Int, List<DxirNode>>,
    ): DxirNode {
        if (ret is DxirOpResult) {
            multiOut[ret.source.id]?.let { return it[ret.index] }
        }
        val mapped = nodeMap[ret.id]
            ?: error(
                "PhiCalculus.rewriteFunction: return id=${ret.id} not in nodeMap " +
                    "(broken SSA after rewrite)",
            )
        if (ret !is DxirOpResult) return mapped
        // §0.4.128 — when a multi-result op is cloned verbatim (no multiOut) the
        // mapped node IS the cloned multi-result DxirOp; route through .result(k)
        // so DxirOpResult returns at index k>=1 yield the correct per-index result.
        return when {
            mapped is DxirOp -> mapped.result(ret.index)
            ret.index == 0 -> mapped
            else -> error(
                "PhiCalculus.rewriteFunction: return id=${ret.id} index=${ret.index} on " +
                    "non-Op clone (mapped=${mapped::class.simpleName}); only index 0 is " +
                    "tolerated for non-Op clones (e.g., C5 collapse)",
            )
        }
    }

    /**
     * Clone a node verbatim into [emitter]'s scope, with operand references resolved
     * through [nodeMap]. Returns the cloned node (also added to the emitter's body).
     * [emitter] is either a [DxirBuilder] (function-body context) or a
     * [DxirRegionBuilder] (nested-region context); both implement [DxirEmitter].
     */
    private fun cloneNode(
        node: DxirNode,
        nodeMap: MutableMap<Int, DxirNode>,
        emitter: DxirEmitter,
        multiOut: Map<Int, List<DxirNode>> = emptyMap(),
    ): DxirNode {
        return when (node) {
            is DxirParam -> nodeMap[node.id]
                ?: error("cloneNode: param id=${node.id} not in nodeMap (param decl missing?)")
            is DxirConst -> emitter.const(node.value, node.type, node.sharding)
            is DxirOp -> {
                val clonedOperands = node.operands.map {
                    resolveClonedOperand(it, nodeMap, node.id, multiOut)
                }
                val clonedRegions = node.regions.map { cloneRegion(it, nodeMap, emitter, multiOut) }
                if (node.types.size == 1) {
                    emitter.op(
                        node.op, clonedOperands, node.type, node.attrs, node.sharding, clonedRegions,
                    )
                } else {
                    emitter.opMulti(
                        node.op, clonedOperands, node.types, node.attrs, node.sharding, clonedRegions,
                    )
                }
            }
            is DxirOpResult -> {
                // §0.4.51 — check multiOut first (multi-result C5 unroll case).
                multiOut[node.source.id]?.let { return it[node.index] }
                // The source op should already be in nodeMap (cloned earlier in body order).
                val clonedSource = nodeMap[node.source.id]
                    ?: error("cloneNode: DxirOpResult source id=${node.source.id} not in nodeMap")
                when {
                    // Common case: source cloned to a DxirOp; route through .result(idx).
                    clonedSource is DxirOp -> clonedSource.result(node.index)
                    // C5 collapse case: a multi-result WHILE got rewritten to a single value
                    // (the carried result), and the source's clone is now a non-Op (param /
                    // const / collapsed scalar). Tolerate index==0 — the only result that
                    // could be referenced after C5's "only-carried-result" precondition.
                    node.index == 0 -> clonedSource
                    else -> error(
                        "cloneNode: DxirOpResult index=${node.index} on non-Op clone " +
                            "(source.id=${node.source.id}); only index 0 is tolerated for " +
                            "non-Op clones (C5 collapse). Did C5 fire on a WHILE whose " +
                            "non-carried results are also referenced?",
                    )
                }
            }
            is DxirCall -> error(
                "PhiCalculus.cloneNode: DxirCall not supported in B.1 (id=${node.id})",
            )
            is io.tlaloc.ir.DxirBlockArg -> error(
                "PhiCalculus.cloneNode: DxirBlockArg outside region context (id=${node.id})",
            )
        }
    }

    /**
     * §0.4.51 — resolve a cloned operand. For plain nodes the `nodeMap[id]` lookup is
     * sufficient. For [DxirOpResult] operands the lookup strips the multi-result index
     * (because `DxirOpResult.id == source.id`), so we re-wrap with the right index when
     * the cloned source is still a multi-result [DxirOp]. If C5 collapsed the source to
     * a single scalar, index 0 tolerates the collapse; higher indices error (same rule
     * as the [DxirOpResult] branch in [cloneNode]).
     */
    private fun resolveClonedOperand(
        operand: DxirNode,
        nodeMap: MutableMap<Int, DxirNode>,
        ownerOpId: Int,
        multiOut: Map<Int, List<DxirNode>> = emptyMap(),
    ): DxirNode {
        // §0.4.51 — for DxirOpResult operands whose source WHILE was unrolled to
        // multiple per-index replacements, the multiOut lookup returns the correct
        // index-specific value directly (bypassing the id-only nodeMap).
        if (operand is DxirOpResult) {
            multiOut[operand.source.id]?.let { return it[operand.index] }
        }
        val mapped = nodeMap[operand.id]
            ?: error("cloneNode: operand id=${operand.id} of op id=$ownerOpId not in nodeMap")
        if (operand !is DxirOpResult) return mapped
        return when {
            mapped is DxirOp -> mapped.result(operand.index)
            operand.index == 0 -> mapped
            else -> error(
                "cloneNode: DxirOpResult index=${operand.index} on non-Op clone " +
                    "(source.id=${operand.source.id}); only index 0 is tolerated for " +
                    "non-Op clones (C5 collapse). Did C5 fire on a WHILE whose " +
                    "non-carried results are also referenced?",
            )
        }
    }

    /**
     * Clone a [DxirRegion] into [emitter]'s id-space (uses the underlying [DxirBuilder]
     * shared id allocator). Block args are reallocated; body ops are cloned with operand
     * references resolved through a per-region nodeMap that shadows the outer one with
     * new block-arg ids. Works whether [emitter] is a function-body [DxirBuilder] or a
     * nested-region [io.tlaloc.ir.DxirRegionBuilder].
     */
    private fun cloneRegion(
        region: DxirRegion,
        outerNodeMap: MutableMap<Int, DxirNode>,
        emitter: DxirEmitter,
        multiOut: Map<Int, List<DxirNode>> = emptyMap(),
    ): DxirRegion {
        require(region.blocks.size == 1) {
            "PhiCalculus.cloneRegion: B.1 supports single-block regions only; got ${region.blocks.size}"
        }
        val origBlock = region.blocks.single()
        // Both DxirBuilder and DxirRegionBuilder expose a `region { ... }` block builder.
        val regionLambda: io.tlaloc.ir.DxirRegionBuilder.() -> Unit = {
            val regionNodeMap = HashMap(outerNodeMap)
            for (a in origBlock.args) {
                val newArg = arg(a.type, a.sharding)
                regionNodeMap[a.id] = newArg
            }
            for (n in origBlock.body) {
                regionNodeMap[n.id] = cloneNode(n, regionNodeMap, this as DxirEmitter, multiOut)
            }
            val terms = origBlock.terminator.map {
                if (it is DxirOpResult) {
                    multiOut[it.source.id]?.let { repl -> return@map repl[it.index] }
                }
                val mapped = regionNodeMap[it.id]
                    ?: error(
                        "PhiCalculus.cloneRegion: terminator id=${it.id} not in regionNodeMap",
                    )
                if (it !is DxirOpResult) return@map mapped
                // §0.4.128 — multi-result op cloned verbatim into the region: route
                // through .result(k) so DxirOpResult yields the correct per-index
                // value rather than collapsing to the source op (whose .type is
                // types[0]). Mirrors [resolveClonedOperand]'s tail logic.
                when {
                    mapped is DxirOp -> mapped.result(it.index)
                    it.index == 0 -> mapped
                    else -> error(
                        "PhiCalculus.cloneRegion: terminator id=${it.id} index=${it.index} on " +
                            "non-Op clone (mapped=${mapped::class.simpleName}); only index 0 " +
                            "is tolerated for non-Op clones",
                    )
                }
            }
            yields(*terms.toTypedArray())
        }
        return when (emitter) {
            is DxirBuilder -> emitter.region(regionLambda)
            is io.tlaloc.ir.DxirRegionBuilder -> emitter.region(regionLambda)
            else -> error("PhiCalculus.cloneRegion: unsupported emitter type ${emitter::class}")
        }
    }

    // ------------------------------------------------------------------------
    // Use-count analysis (for F2 / C1 anti-swell gate)
    // ------------------------------------------------------------------------

    /** Number of reference edges into each SSA id from elsewhere in [fn]. */
    private fun computeUseCounts(fn: DxirFunction): Map<Int, Int> {
        val counts = HashMap<Int, Int>()
        fun bump(id: Int) {
            counts[id] = (counts[id] ?: 0) + 1
        }
        fun walk(nodes: List<DxirNode>) {
            for (n in nodes) {
                when (n) {
                    is DxirOp -> {
                        for (o in n.operands) bump(o.id)
                        for (r in n.regions) for (b in r.blocks) {
                            walk(b.body)
                            for (term in b.terminator) bump(term.id)
                        }
                    }
                    is DxirCall -> for (o in n.args) bump(o.id)
                    else -> {}
                }
            }
        }
        walk(fn.body)
        for (r in fn.returns) bump(r.id)
        return counts
    }

    // ------------------------------------------------------------------------
    // Structural equality (fixpoint check)
    // ------------------------------------------------------------------------

    /**
     * Cheap structural equivalence check between two [DxirFunction]s — used by the
     * fixpoint loop to detect when no further rewrites fire. Two functions are
     * "equivalent" for fixpoint purposes if they have the same body op count + same
     * sequence of op kinds + same number of returns. Not a deep semantic check; the
     * goal is "no rewrite changed anything", which is a strict subset of "structurally
     * identical". A false-negative (claims equivalent when not) only matters if it
     * masks a real rewrite — guarded by the caller's `FIXPOINT_CAP` re-runs.
     */
    private fun structurallyEqual(a: DxirFunction, b: DxirFunction): Boolean {
        if (a.params.size != b.params.size) return false
        if (a.returns.size != b.returns.size) return false
        if (a.body.size != b.body.size) return false
        for (i in a.body.indices) {
            val na = a.body[i]
            val nb = b.body[i]
            if (na::class != nb::class) return false
            if (na is DxirOp && nb is DxirOp) {
                if (na.op != nb.op) return false
                if (na.operands.size != nb.operands.size) return false
                if (na.regions.size != nb.regions.size) return false
            }
        }
        return true
    }

    // ------------------------------------------------------------------------
    // §0.4.30 — Stage C.3a: coarsenLeaf — per-SOI-leaf coarsening
    // ------------------------------------------------------------------------

    /**
     * Coarsen a [RegionTreeNode] leaf by synthesising a mini-function whose params
     * are the leaf's free variables, whose body is the leaf's `directOps` (cloned
     * with operand references remapped to params or cloned locals), and whose
     * returns are the directOps consumed outside the leaf (via [DefUseChain]). The
     * mini-function is run through [apply] with [engine] and the size of the
     * resulting body becomes the authoritative post-coarsening op count.
     *
     * Stage C.3a scope:
     *  - Leaves only (no region-bearing ops in `directOps`). If a region-bearing op
     *    is encountered, returns [CoarsenResult.Failure] — caller falls back to raw
     *    op count.
     *  - Single-result ops only (no multi-result). Multi-result leaf ops are rare
     *    in practice; widen when a benchmark demands it.
     *  - `DxirOpResult` operand references (e.g., a WHILE's carried output consumed
     *    inside a leaf that sits after the WHILE) are tolerated — the clone routes
     *    through `.result(idx)` on the source op.
     *
     * Returns [CoarsenResult.Success] carrying the simplified function + its post-
     * coarsening body op count, or [CoarsenResult.Failure] with a reason. Never
     * throws — any internal error is wrapped into `Failure`.
     *
     * The simplified function is a **standalone** [DxirFunction] suitable for:
     *  - size measurement (this session's use case).
     *  - C.3b's gradient synthesis: run [DxirReverseTransform.apply] on it to get
     *    the pre-computed VJP body + splice both primal + gradient into an
     *    `OpKind.COARSENED` op in the parent function.
     */
    fun coarsenLeaf(
        leaf: RegionTreeNode,
        sourceFn: DxirFunction,
        engine: SymbolicEngine?,
    ): CoarsenResult {
        if (!leaf.isLeaf) return CoarsenResult.Failure("not a leaf")
        if (leaf.directOps.isEmpty()) return CoarsenResult.Failure("leaf is empty")
        for (n in leaf.directOps) {
            if (n is DxirOp && n.hasRegions) {
                return CoarsenResult.Failure("leaf contains region-bearing op ${n.op}")
            }
            if (n is DxirOp && n.isMultiResult) {
                return CoarsenResult.Failure("leaf contains multi-result op ${n.op}")
            }
        }

        val mini = try {
            buildLeafMiniFunction(leaf, sourceFn)
        } catch (t: Throwable) {
            return CoarsenResult.Failure(
                "mini-function construction failed: ${t::class.simpleName}: ${t.message}",
            )
        }
        val simplified = try {
            apply(mini, engine)
        } catch (t: Throwable) {
            return CoarsenResult.Failure(
                "PhiCalculus.apply failed on mini-function: ${t::class.simpleName}: ${t.message}",
            )
        }
        return CoarsenResult.Success(simplified = simplified, size = simplified.body.size)
    }

    /**
     * Synthesise a standalone [DxirFunction] representing a leaf's subtree:
     *  - params: one per free variable (operand ids referenced from inside the leaf
     *    but not declared by it). Order follows first-encounter-in-directOps.
     *  - body: clones of [RegionTreeNode.directOps] with operand references remapped
     *    to params (for free vars) or cloned predecessors (for same-scope refs).
     *  - returns: directOps consumed outside the leaf per [DefUseChain]. If none are,
     *    the last directOp is treated as the sole return (dead-leaf fallback).
     */
    private fun buildLeafMiniFunction(
        leaf: RegionTreeNode,
        sourceFn: DxirFunction,
    ): DxirFunction {
        val localIds = leaf.directOps.map { it.id }.toHashSet()
        val freeVarIds = LinkedHashSet<Int>()
        for (n in leaf.directOps) {
            if (n !is DxirOp) continue
            for (operand in n.operands) {
                if (operand.id !in localIds) freeVarIds += operand.id
            }
        }
        val sourceById = collectAllSourceNodes(sourceFn)
        val chain = DefUseChain.build(sourceFn)
        val returnIdsRaw = leaf.directOps.filter { n ->
            val users = chain.usersOf(n.id)
            // Consumed by -1 (function sink), by an id outside localIds (parent scope or
            // child region outer-ref), or by no one (dead node — don't surface as return).
            users.any { it == -1 || it !in localIds }
        }.map { it.id }
        val returnIds = if (returnIdsRaw.isEmpty()) listOf(leaf.directOps.last().id) else returnIdsRaw

        return DxirBuilder.function("soi_leaf_${leaf.regionOp?.id ?: "root"}") {
            val nodeMap = HashMap<Int, DxirNode>()
            for (fid in freeVarIds) {
                val src = sourceById[fid]
                    ?: error("buildLeafMiniFunction: free var id=$fid missing from source")
                val p = param("free_$fid", src.type)
                nodeMap[fid] = p
            }
            for (n in leaf.directOps) {
                val cloned: DxirNode = when (n) {
                    is DxirConst -> const(n.value, n.type, n.sharding)
                    is DxirOp -> {
                        val clonedOperands = n.operands.map {
                            nodeMap[it.id]
                                ?: error("buildLeafMiniFunction: operand id=${it.id} missing from nodeMap")
                        }
                        op(n.op, clonedOperands, n.type, n.attrs, n.sharding, emptyList())
                    }
                    else -> error("buildLeafMiniFunction: unsupported ${n::class.simpleName}")
                }
                nodeMap[n.id] = cloned
            }
            returnIds.map {
                nodeMap[it]
                    ?: error("buildLeafMiniFunction: return id=$it missing from nodeMap")
            }
        }
    }

    /**
     * Walk [fn] end-to-end and return `id → node` for every params + body + nested
     * region bodies + block args. Used by [buildLeafMiniFunction] to resolve free
     * variable types.
     */
    private fun collectAllSourceNodes(fn: DxirFunction): Map<Int, DxirNode> {
        val map = HashMap<Int, DxirNode>()
        for (p in fn.params) map[p.id] = p
        fun walk(node: DxirNode) {
            if (node is DxirOpResult) return
            map[node.id] = node
            if (node is DxirOp) for (region in node.regions) for (block in region.blocks) {
                for (arg in block.args) map[arg.id] = arg
                for (b in block.body) walk(b)
            }
        }
        for (n in fn.body) walk(n)
        return map
    }

    // ------------------------------------------------------------------------
    // §0.4.33 — Stage C.3b.3a: coarsenFunction — SOI splice pass
    // ------------------------------------------------------------------------

    /**
     * Run the Stage C SOI-identification + coarsening pipeline on [fn], producing a new
     * [DxirFunction] where each final SOI is replaced by a single [OpKind.COARSENED]
     * op carrying the pre-computed primal + gradient. Downstream consumers
     * ([DxirReverseTransform.apply]) use [DxirReverseTransform.handleCoarsenedAdjoint]
     * to splice the stored gradient when they encounter the COARSENED op.
     *
     * **C.3b.3a** (§0.4.33) — the root-is-leaf case: a region-free primal collapses to a
     * single COARSENED op at the top level.
     *
     * **C.3b.3b2** (§0.4.35) — the region-bearing case: when the primal contains IF
     * branches, each branch whose body is a viable SOI (non-empty, scalar-numeric
     * single-yield) gets its body replaced by a per-branch COARSENED op. The enclosing
     * IF structure is preserved; downstream `DxirReverseTransform.walkBranchReverse`
     * (extended in §0.4.34 to dispatch COARSENED) handles the gradient through each
     * branch. WHILE cond/body regions are deferred — cond yields Bool (non-differentiable
     * result type), and body yields multiple values (multi-result COARSENED not yet
     * supported).
     *
     * Single-return primals only. Multi-sink coarsening is conceptually
     * straightforward (one COARSENED op per sink) but defers until a benchmark demands
     * it.
     *
     * If the primal doesn't meet the scope (multi-return, no viable SOIs, coarsening
     * failure), returns [fn] unchanged — callers can check structurally whether any
     * replacement happened.
     *
     * @param sizeLimit The SOI size limit `L` (plan §8.2 default: 50). Drives which
     *   nodes in the region tree [SoiIdentification.identifyWithSizeLimit] marks as
     *   large; first-cut coarsenFunction does whole-leaf coarsening without further
     *   size discrimination at the leaf level.
     */
    fun coarsenFunction(
        fn: DxirFunction,
        engine: SymbolicEngine?,
        sizeLimit: Int = 50,
    ): DxirFunction {
        if (fn.returns.size != 1) return fn
        val tree = RegionTree.build(fn)

        // C.3b.3a: region-free primal. Whole body collapses to one COARSENED op.
        if (tree.root.isLeaf) return coarsenRootLeaf(fn, engine)

        // C.3b.3b2: region-bearing primal. Splice per-SOI COARSENEDs into IF branches.
        return coarsenMultiSoi(fn, tree, engine, sizeLimit)
    }

    /**
     * §0.4.33 — wrap the entire region-free function body in one COARSENED op. Returns
     * [fn] unchanged on any failure (PhiCalculus.apply threw, resulting body has
     * regions, gradient body generation failed, etc.).
     */
    private fun coarsenRootLeaf(fn: DxirFunction, engine: SymbolicEngine?): DxirFunction {
        if (fn.body.isEmpty()) return fn

        val primalBody = try {
            apply(fn, engine)
        } catch (_: Throwable) {
            return fn
        }
        if (primalBody.body.any { it is DxirOp && it.hasRegions }) return fn

        val gradientBody = try {
            DxirReverseTransform.apply(
                primal = primalBody,
                includeForward = false,
                seedAsParam = true,
            )
        } catch (_: Throwable) {
            return fn
        }
        if (gradientBody.body.any { it is DxirOp && it.hasRegions }) return fn

        val reads = computeGradientReads(gradientBody)

        return DxirBuilder.function(fn.name) {
            val clonedParams = fn.params.map { p -> param(p.name, p.type, p.sharding) }
            val c = coarsened(
                operands = clonedParams,
                primalBody = primalBody,
                gradientBody = gradientBody,
                readsPrimalIndices = reads,
            )
            listOf(c)
        }
    }

    /**
     * §0.4.35 — Stage C.3b.3b2 multi-SOI splice. For a region-bearing primal, identify
     * SOIs that are IF-branch leaves (single-yield, scalar numeric) and replace each
     * branch body with a single COARSENED op carrying the coarsened primal + gradient.
     * Non-SOI regions + non-IF region-bearing ops (WHILE in particular) survive unchanged.
     *
     * Replacement shape — for each SOI IF-branch leaf:
     *  1. Build a mini-function via [coarsenLeaf]: params = free vars referenced by the
     *     branch body, body = the branch body ops, returns = the yielded value.
     *  2. Generate the gradient body via [DxirReverseTransform.apply] with
     *     `seedAsParam = true` — signature `(upstream, *free_vars) → (*grads)`.
     *  3. Construct a COARSENED op whose operands are the free-var outer-scope refs
     *     (resolved through the new function's `nodeMap`) and whose attrs carry the
     *     primal + gradient bodies.
     *  4. Rewrite the branch's block body to `[COARSENED]` with the terminator yielding
     *     the COARSENED's result.
     *
     * Returns [fn] unchanged if no coarsenable SOIs exist or all coarsening attempts
     * fail.
     */
    private fun coarsenMultiSoi(
        fn: DxirFunction,
        @Suppress("UNUSED_PARAMETER") tree: RegionTree,
        engine: SymbolicEngine?,
        sizeLimit: Int,
    ): DxirFunction {
        val soiResult = SoiIdentification.identifyWithSizeLimit(fn, sizeLimit, engine)
        // Only ORIGINAL tree nodes are safe to splice — splitOnReuses produces fragment
        // nodes whose directOps is a partition of an original leaf, and those fragments
        // reference each other via intermediate values that would be branch-internal
        // (not resolvable from the outer nodeMap). For C.3b.3b2 first cut, skip
        // fragments and only coarsen whole-leaf SOIs. Use the SoiResult's tree (not
        // the parameter) so reference-identity lookup matches what identifyWithSizeLimit
        // actually produced — §0.4.35 exposed `tree` on SoiResult precisely for this.
        val originalNodes: Set<RegionTreeNode> = soiResult.tree.bottomUp().toSet()
        // Collect per-SOI replacements. Keyed by (regionOpId, regionIndex).
        val replacements = HashMap<Pair<Int, Int>, CoarsenedReplacement>()
        for (soi in soiResult.sois) {
            val leaf = soi.node
            if (leaf !in originalNodes) continue  // skip split fragments
            if (!leaf.isLeaf) continue
            val regionOp = leaf.regionOp ?: continue
            if (regionOp.op != OpKind.IF) continue
            val block = regionOp.regions[leaf.regionIndex].blocks[leaf.blockIndex]
            if (leaf.directOps.isEmpty()) continue
            if (block.terminator.size != 1) continue
            val yieldType = block.terminator.single().type
            if (!yieldType.isScalar) continue
            if (yieldType.dtype !is io.tlaloc.core.F32 && yieldType.dtype !is io.tlaloc.core.F64) continue

            val repl = buildReplacement(leaf, fn, engine) ?: continue
            replacements[regionOp.id to leaf.regionIndex] = repl
        }
        if (replacements.isEmpty()) return fn

        // Clone the function, substituting branch bodies where replacements apply.
        return DxirBuilder.function(fn.name) {
            val nodeMap = HashMap<Int, DxirNode>()
            for (p in fn.params) nodeMap[p.id] = param(p.name, p.type, p.sharding)
            for (n in fn.body) {
                val cloned = if (
                    n is DxirOp && n.hasRegions &&
                    n.op == OpKind.IF &&
                    n.regions.indices.any { replacements.containsKey(n.id to it) }
                ) {
                    cloneIfWithReplacements(n, nodeMap, this, replacements)
                } else {
                    cloneNode(n, nodeMap, this as DxirEmitter)
                }
                nodeMap[n.id] = cloned
            }
            fn.returns.map { nodeMap[it.id] ?: error("return id=${it.id} missing from nodeMap") }
        }
    }

    /**
     * Build the per-SOI replacement artifact for a viable IF-branch leaf. Returns null
     * if any step fails (leaf coarsening, gradient body generation, region-bearing
     * outputs, etc.) so the caller can skip this SOI without bringing down the whole pass.
     */
    private fun buildReplacement(
        leaf: RegionTreeNode,
        fn: DxirFunction,
        engine: SymbolicEngine?,
    ): CoarsenedReplacement? {
        val primalBody = when (val r = coarsenLeaf(leaf, fn, engine)) {
            is CoarsenResult.Success -> r.simplified
            is CoarsenResult.Failure -> return null
        }
        if (primalBody.body.any { it is DxirOp && it.hasRegions }) return null
        val gradientBody = try {
            DxirReverseTransform.apply(
                primal = primalBody,
                includeForward = false,
                seedAsParam = true,
            )
        } catch (_: Throwable) {
            return null
        }
        if (gradientBody.body.any { it is DxirOp && it.hasRegions }) return null
        val reads = computeGradientReads(gradientBody)

        // Free-var ids: operand ids in directOps NOT declared locally. First-encounter order.
        val localIds = leaf.directOps.map { it.id }.toHashSet()
        val freeVarIds = LinkedHashSet<Int>()
        for (n in leaf.directOps) {
            if (n !is DxirOp) continue
            for (operand in n.operands) {
                if (operand.id !in localIds) freeVarIds += operand.id
            }
        }
        return CoarsenedReplacement(
            primalBody = primalBody,
            gradientBody = gradientBody,
            readsPrimalIndices = reads,
            freeVarIds = freeVarIds.toList(),
        )
    }

    /**
     * Clone an IF op, substituting specific regions' block bodies with their
     * [CoarsenedReplacement]-driven COARSENED ops. Branches without a replacement get
     * verbatim clones via [cloneRegion]. Block args are copied; the new block body is
     * `[COARSENED]` + a single-value yield of the COARSENED result.
     */
    private fun cloneIfWithReplacements(
        oldIf: DxirOp,
        nodeMap: MutableMap<Int, DxirNode>,
        builder: DxirBuilder,
        replacements: Map<Pair<Int, Int>, CoarsenedReplacement>,
    ): DxirOp {
        require(oldIf.op == OpKind.IF) { "cloneIfWithReplacements: not an IF" }
        val predClone = nodeMap[oldIf.operands[0].id]
            ?: error("cloneIfWithReplacements: predicate id=${oldIf.operands[0].id} missing")
        val clonedRegions = oldIf.regions.mapIndexed { rIdx, region ->
            val repl = replacements[oldIf.id to rIdx]
            if (repl != null) {
                builder.region { emitCoarsenedBranchBody(repl, nodeMap, this) }
            } else {
                cloneRegion(region, nodeMap, builder)
            }
        }
        return builder.ifOp(
            cond = predClone,
            types = oldIf.types,
            thenRegion = clonedRegions[0],
            elseRegion = clonedRegions[1],
        )
    }

    /**
     * Inside a new IF-branch region builder, emit the replacement COARSENED op and yield
     * its result. Operands map to the outer [nodeMap] via the recorded free-var ids.
     * IF-branch blocks have no args (per §3.1.1's shape constraint), so there's no block-
     * arg mapping to thread — the operand lookup happens entirely through the outer nodeMap.
     */
    private fun emitCoarsenedBranchBody(
        repl: CoarsenedReplacement,
        nodeMap: Map<Int, DxirNode>,
        rb: DxirRegionBuilder,
    ) {
        val operands = repl.freeVarIds.map { fid ->
            nodeMap[fid] ?: error(
                "emitCoarsenedBranchBody: free var id=$fid missing from outer nodeMap",
            )
        }
        // Emit a single-result COARSENED op directly through the region builder's op()
        // method. Using opMulti-vs-op would need the multi-result shape, which C.3b.3b2
        // doesn't produce (single-yield branches).
        val types = repl.primalBody.returns.map { it.type }
        require(types.size == 1) {
            "emitCoarsenedBranchBody: multi-result COARSENED in branch not supported (C.3b.3b2 first cut)"
        }
        val attrs: Map<String, Any> = mapOf(
            "primal_body" to repl.primalBody,
            "gradient_body" to repl.gradientBody,
            "reads_primal_indices" to repl.readsPrimalIndices,
        )
        val c = rb.op(
            kind = OpKind.COARSENED,
            operands = operands,
            type = types[0],
            attrs = attrs,
            sharding = null,
            regions = emptyList(),
        )
        rb.yields(c)
    }

    /** §0.4.35 — per-SOI replacement artifact produced by [buildReplacement]. */
    private data class CoarsenedReplacement(
        val primalBody: DxirFunction,
        val gradientBody: DxirFunction,
        val readsPrimalIndices: Set<Int>,
        /** Free variable ids (in emission order) consumed by the replaced branch body. */
        val freeVarIds: List<Int>,
    )

    /**
     * Scan [gradientBody]'s body for references to its primal operand params (params at
     * indices ≥ 1, since params[0] is the upstream). Returns the set of primal-operand
     * indices that actually appear in emitted op operands — the `reads_primal_indices`
     * attr the [OpKind.COARSENED] op needs for [DxirReverseTransform.computeUsedByAdjoint]
     * to know which operand subgraphs must be cloned into the gradient-function scope.
     */
    private fun computeGradientReads(gradientBody: DxirFunction): Set<Int> {
        val paramIds = gradientBody.params.map { it.id }
        val reads = LinkedHashSet<Int>()
        fun scanOp(n: DxirOp) {
            for (operand in n.operands) {
                val paramIdx = paramIds.indexOf(operand.id)
                if (paramIdx >= 1) reads += paramIdx - 1
            }
            for (region in n.regions) for (block in region.blocks) {
                for (bodyNode in block.body) if (bodyNode is DxirOp) scanOp(bodyNode)
            }
        }
        for (n in gradientBody.body) if (n is DxirOp) scanOp(n)
        // Also scan returns — a return might directly reference a primal param.
        for (r in gradientBody.returns) {
            val paramIdx = paramIds.indexOf(r.id)
            if (paramIdx >= 1) reads += paramIdx - 1
        }
        return reads
    }
}
