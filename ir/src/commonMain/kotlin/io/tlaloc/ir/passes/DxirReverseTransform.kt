package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * Source-code-transformation reverse-mode AD on a [DxirFunction]. Given a primal function
 * `f: P₁ × … × Pₙ → R` (R a single scalar), produces a gradient function whose returns are
 * the per-parameter gradients of `f`'s output. When [apply] is called with
 * `includeForward = true`, the primal return value is prepended to the returns list so
 * callers can synthesise `valueAndGrad` / `valueAndGrad2` without re-running the forward.
 *
 * This is the SCT baseline (§11.8.1 Stage A) — the prerequisite for the coarsening pass
 * in Stage B. It is intentionally narrow:
 *
 * - **Single scalar return only.** Vector / tensor outputs would require an explicit
 *   upstream-cotangent parameter (rather than the implicit 1.0 seed embedded here).
 * - **Straight-line bodies only.** Ops carrying nested regions (e.g. `MANUAL_COMPUTATION`,
 *   future `If`/`While`) are rejected — handling them is the φ-calculus pass in Stage B.
 * - **Single-result body ops only.** Multi-result ops (e.g. `SPLIT`, `ARGMAX`) need
 *   per-result accumulator threading; deferred.
 * - **Only ops with a registered [VjpRule] in [VjpRegistry].** Unsupported ops fail loudly.
 *
 * Multi-parameter primals are accepted (N ≥ 0). Each primal parameter gets its own
 * accumulator slot, and the returned function's gradient-return list is indexed positionally
 * by primal param order; unused params get a typed zero constant.
 *
 * ### Algorithm
 *
 * 1. Validate hard gates.
 * 2. Compute [usedByAdjoint] — the set of primal node ids whose *cloned* form must exist
 *    in the gradient function's body. Seed with `op.operands[i]` for every primal op and
 *    every `i ∈ VjpRegistry[op.op].readsPrimalOperandIndices`, plus the primal return's id
 *    when `includeForward` is on; transitively close through `DxirOp.operands` so every
 *    operand chain reachable from a used node lands in the set too. Primal subgraphs
 *    whose values no adjoint rule dereferences (e.g. the outer `MUL` in
 *    `ADD(MUL(x, x), CONST)` — `ADD`'s rule reads no operands) drop out of this set and
 *    do not appear in the gradient body.
 * 3. Re-create the primal's params + body in a fresh [DxirBuilder], cloning only nodes
 *    in [usedByAdjoint]. Populate [nodeMap] for every primal id: cloned nodes for ids in
 *    the set, primal nodes verbatim for ids not in it (the framework never dereferences
 *    those — they're used only for `indexOf` reference-identity matching).
 * 4. Allocate a constant `1.0` seed matching the return's dtype; install it as the
 *    upstream gradient of the return node in `gradAccum: Map<primal-id, contribution>`.
 * 5. Walk the primal body in reverse. For each op with non-null upstream gradient, look
 *    up its [VjpRule], invoke it, and accumulate the returned contributions into
 *    `gradAccum` keyed by the primal operand's id. Repeated contributions to the same id
 *    emit a fresh `ADD`, mirroring the runtime tape's `Tape.pushback` accumulation. For
 *    primal ops not in [usedByAdjoint] (no clone in body), construct a detached phantom
 *    [DxirOp] with [nodeMap]-resolved operands to hand to the rule; the phantom is never
 *    added to the gradient body.
 * 6. Return a function whose returns are the accumulated gradients for each primal
 *    parameter (or a typed `0.0` const for unused params — empty-graph edge case). When
 *    [includeForward] is true, the cloned primal return is prepended.
 *
 * @throws IllegalArgumentException if the primal violates a hard gate.
 * @throws IllegalStateException if a body op references an [OpKind] without a [VjpRule].
 */
object DxirReverseTransform {

    /**
     * Reverse-mode transform.
     *
     * @param primal The forward function. Must have exactly 1 scalar return.
     * @param includeForward If true, prepend the cloned primal return to the output's
     *   returns list — `valueAndGrad` shape.
     * @param seedAsParam §0.4.33 — if true, the gradient function's first param is an
     *   explicit `upstream` (typed to match the primal's return type) that seeds
     *   `gradAccum[ret.id]`. The resulting signature is
     *   `(upstream, *primal_params) → (*grads)`, which is what
     *   [OpKind.COARSENED]'s `gradient_body` attribute expects. When false (default),
     *   the seed is `const(1.0)` — the existing `grad` / `valueAndGrad` behaviour.
     *   Mutually exclusive with `includeForward` (a COARSENED gradient_body never
     *   emits a forward value).
     */
    fun apply(
        primal: DxirFunction,
        includeForward: Boolean = false,
        seedAsParam: Boolean = false,
    ): DxirFunction {
        require(!(includeForward && seedAsParam)) {
            "DxirReverseTransform: includeForward + seedAsParam are incompatible — the " +
                "COARSENED gradient_body signature doesn't accommodate a forward return"
        }
        require(primal.returns.size == 1) {
            "DxirReverseTransform v1 requires exactly 1 return value (got ${primal.returns.size})"
        }
        val ret = primal.returns.single()
        require(ret.type.isScalar) {
            "DxirReverseTransform v1 requires a scalar return (got ${ret.type})"
        }
        for (n in primal.body) {
            if (n is DxirOp) {
                if (n.hasRegions) {
                    require(n.op == OpKind.IF) {
                        "DxirReverseTransform: op ${n.op} has regions but no rule supports " +
                            "regions for it (only IF is handled directly via the §0.4.23 " +
                            "branch reverse walk; WHILE and others must be coarsened by " +
                            "Stage B's PhiCalculus pass before SCT)"
                    }
                }
                require(!n.isMultiResult) {
                    "DxirReverseTransform v1 rejects multi-result ops (got ${n.op})"
                }
            }
        }

        val primalById: Map<Int, DxirNode> =
            (primal.params + primal.body).associateBy { it.id }
        val usedByAdjoint = computeUsedByAdjoint(primal, primalById, includeForward)

        return DxirBuilder.function(primal.name + "_grad") {
            // §0.4.33 — when seedAsParam is set, prepend `upstream` as the first param
            // so the resulting function has the COARSENED gradient_body signature:
            // `(upstream, *primal_params) → (*grads)`. The upstream's type matches the
            // primal's return type (the gradient type of the value flowing INTO this
            // gradient — same type as what a downstream op would hand us).
            val upstreamParam: DxirParam? = if (seedAsParam) {
                param("__upstream__", ret.type)
            } else {
                null
            }
            // --- 1. Clone primal params + selectively clone body into the new function's
            //        scope. Nodes outside `usedByAdjoint` stay as primal references in
            //        `nodeMap` — they're consumed only by phantom operand lookups below,
            //        never dereferenced by adjoint body-ops. ---
            val nodeMap = HashMap<Int, DxirNode>()
            for (p in primal.params) {
                nodeMap[p.id] = param(p.name, p.type)
            }
            for (n in primal.body) {
                if (n.id !in usedByAdjoint) {
                    // Not needed in gradient body. Record primal node for indexOf-style
                    // operand resolution; never dereferenced as a body operand.
                    nodeMap[n.id] = n
                    continue
                }
                val cloned: DxirNode = when (n) {
                    is DxirConst -> const(n.value, n.type)
                    is DxirOp -> {
                        if (n.op == OpKind.IF) {
                            // Don't clone the IF here — its regions reference outer-scope
                            // SSA values that must resolve through nodeMap, but the inner
                            // body is also re-cloned during [walkBranchReverse]. Cloning
                            // both would emit duplicate computation. Map to the primal
                            // node; the IF's id is referenced only by adjoint synthesis
                            // (which uses outerNodeMap[predicate] and primal regions).
                            n
                        } else {
                            op(
                                kind = n.op,
                                operands = n.operands.map {
                                    nodeMap[it.id]
                                        ?: error("primal body op ${n.id} references unknown id ${it.id}")
                                },
                                type = n.type,
                                attrs = n.attrs,
                            )
                        }
                    }
                    else -> error(
                        "DxirReverseTransform: unsupported primal body node ${n::class.simpleName}",
                    )
                }
                nodeMap[n.id] = cloned
            }

            // --- 2. Seed the return's adjoint. When `seedAsParam` is set, use the
            //        upstream param as the seed so the gradient function chains properly
            //        with a downstream consumer (COARSENED's handleCoarsenedAdjoint); else
            //        seed with const(1.0) for the standalone `grad` / `valueAndGrad` path. ---
            val seed: DxirNode = upstreamParam
                ?: const(seedValueFor(ret.type.dtype), ret.type)
            val gradAccum = HashMap<Int, DxirNode>()
            gradAccum[ret.id] = seed

            // --- 3. Reverse walk: emit per-op adjoint contributions. ---
            for (n in primal.body.asReversed()) {
                if (n !is DxirOp) continue
                val upstream = gradAccum[n.id] ?: continue

                // Special case: IF op (added §0.4.23). Per paper C2 — `d/dx(φ(a, b)) =
                // φ(da/dx, db/dx)` — the gradient of an IF distributes through its
                // branches. Recursively reverse-walk each branch's body with `upstream`
                // seeded at the branch's yield, then synthesize an IF that picks the
                // appropriate per-branch adjoint based on the runtime predicate.
                if (n.op == OpKind.IF) {
                    handleIfAdjoint(n, upstream, gradAccum, nodeMap, primalById, this)
                    continue
                }
                // §0.4.32 — COARSENED adjoint: splice the stored gradient_body into the
                // outer gradient function. Matches [handleIfAdjoint]'s structural role as
                // a "special case bypassing the VjpRule contract" (the COARSENED op's
                // gradient is precomputed + stored in attrs rather than derived from a
                // static rule). See the doc-comment on [handleCoarsenedAdjoint].
                if (n.op == OpKind.COARSENED) {
                    handleCoarsenedAdjoint(n, upstream, gradAccum, nodeMap, primalById, this)
                    continue
                }

                val rule = VjpRegistry[n.op] ?: error(
                    "DxirReverseTransform: no VJP rule registered for ${n.op}",
                )
                // Real in-body clone when `usedByAdjoint` contains the primal id, else
                // a detached phantom whose operands come from nodeMap (cloned nodes for
                // `usedByAdjoint` ids, primal verbatim otherwise). The rule will only
                // dereference operands at indices it declared in `readsPrimalOperandIndices`,
                // and those are in `usedByAdjoint` by construction — so phantom operand
                // fallbacks are never dereferenced by emitted body-ops.
                val clonedOp: DxirOp = if (n.id in usedByAdjoint) {
                    nodeMap[n.id] as DxirOp
                } else {
                    DxirOp(
                        id = allocateId(),
                        op = n.op,
                        operands = n.operands.map { nodeMap[it.id] ?: it },
                        attrs = n.attrs,
                        type = n.type,
                        sharding = n.sharding,
                    )
                }
                val contributions = rule.apply(clonedOp, upstream, this)
                for ((operandKey, contribution) in contributions) {
                    val operandIdx = clonedOp.operands.indexOf(operandKey)
                    require(operandIdx >= 0) {
                        "VjpRule for ${n.op} returned an operand key that is not in op.operands"
                    }
                    val primalOperand = n.operands[operandIdx]
                    // Constants have no gradient surface (they're literals).
                    if (primalById[primalOperand.id] is DxirConst) continue
                    val existing = gradAccum[primalOperand.id]
                    gradAccum[primalOperand.id] = when {
                        existing == null -> contribution
                        // §0.4.45 — SCATTER_ADD fusion: if the contribution is a
                        // SCATTER_ADD and we already have an accumulator, rewrite the
                        // new SCATTER_ADD's base operand to thread `existing` instead
                        // of the contribution's own (now-dead) zero-broadcast base.
                        // Collapses the outer-ADD + SCATTER_ADD chain into a single
                        // SCATTER_ADD chain — one op per gather instead of two.
                        contribution is DxirOp && contribution.op == OpKind.SCATTER_ADD -> {
                            op(
                                OpKind.SCATTER_ADD,
                                listOf(existing, contribution.operands[1], contribution.operands[2]),
                                contribution.type,
                            )
                        }
                        else -> op(OpKind.ADD, listOf(existing, contribution), contribution.type)
                    }
                }
            }

            // --- 4. Returns: gradient per primal param. Unused params get a typed zero.
            //        §0.4.54 — Integer-typed params (I32, I64, Bool) aren't differentiable.
            //        Emit a typed zero regardless of what the VJP chain accumulated, so
            //        the grad function's return types match `primal.params.types` for the
            //        synthesis's Pair/Triple boxing. Without this, PowRule-style chains
            //        that CAST Int exponents to Float for adjoint arithmetic produce a
            //        Float-typed dExp which doesn't type-check against an Int-typed param.
            //        When `includeForward`, prepend the cloned primal return so the caller
            //        can emit `valueAndGrad` / `valueAndGrad2` without re-running forward. ---
            val gradReturns = primal.params.map { p ->
                if (isIntegerDtype(p.type.dtype)) {
                    const(zeroValueFor(p.type.dtype), p.type)
                } else {
                    gradAccum[p.id] ?: const(zeroValueFor(p.type.dtype), p.type)
                }
            }
            if (includeForward) {
                val forwardReturn = nodeMap[ret.id]
                    ?: error("DxirReverseTransform: cloned return node missing for id ${ret.id}")
                listOf(forwardReturn) + gradReturns
            } else {
                gradReturns
            }
        }.let(::dropUnreachableBody)
            .let(::applyCSE)
            .let(::applyConstFold)
            .let(::dropUnreachableBody)
            .let(::tagSingleUseScatterAdds)
    }

    /**
     * §0.4.48 — common sub-expression elimination on the gradient body. Adjoint
     * rules often emit structurally-identical ops: [MulRule] on `x*x` produces
     * two identical `MUL(upstream, x)` contributions (one per operand slot),
     * [SqrtRule] emits a fresh `SQRT(op.operands[0])` whose input already has a
     * cloned primal `SQRT` in the body. CSE walks the body in declaration order,
     * hashes each scalar-result op by `(op.op, operand canonical ids, attrs)`,
     * and replaces duplicates with references to the canonical op. Duplicates
     * become unreachable and are dropped by the following [dropUnreachableBody]
     * step.
     *
     * Scope: single-result, region-free, scalar-or-rank-1 ops only. Multi-result
     * and region-bearing ops (IF / WHILE / COARSENED) are kept verbatim — their
     * structural equivalence is subtler and not load-bearing for any benchmark
     * today. [SCATTER_ADD]'s `in_place` attr is preserved (two SCATTER_ADDs with
     * the same operands but different in-place tags would be an SSA use-graph
     * inconsistency; we shouldn't see that in practice).
     */
    private fun applyCSE(fn: DxirFunction): DxirFunction {
        // Region-bearing ops (IF / COARSENED / WHILE — the latter shouldn't survive
        // SCT, but COARSENED and IF can) carry internal operand references inside
        // their regions. Rewriting those references safely needs a recursive walk
        // that preserves block-arg identity; deferred. For now, skip CSE entirely
        // when any body op carries a region — conservative but correct.
        if (fn.body.any { it is DxirOp && it.hasRegions }) return fn
        val byId = HashMap<Int, DxirNode>()
        for (p in fn.params) byId[p.id] = p
        // CSE consts too — adjoint emission produces many duplicate const literals
        // (e.g., const 2.0f in multiple SqrtRule adjoints). Signature = (value, type).
        val const2canon = HashMap<Pair<Any, DxirType>, DxirConst>()
        val sig2canon = HashMap<Triple<OpKind, List<Int>, Map<String, Any>>, DxirNode>()
        val newBody = mutableListOf<DxirNode>()
        var mutated = false
        for (n in fn.body) {
            when (n) {
                is DxirConst -> {
                    val key = n.value to n.type
                    val existing = const2canon[key]
                    if (existing != null) {
                        byId[n.id] = existing
                        mutated = true
                    } else {
                        const2canon[key] = n
                        byId[n.id] = n
                        newBody += n
                    }
                }
                is DxirOp -> {
                    if (n.isMultiResult || n.hasRegions) {
                        byId[n.id] = n
                        newBody += n
                        continue
                    }
                    val canonicalOperands = n.operands.map { byId[it.id] ?: it }
                    val opIds = canonicalOperands.map { it.id }
                    val sig = Triple(n.op, opIds, n.attrs)
                    val existing = sig2canon[sig]
                    if (existing != null) {
                        byId[n.id] = existing
                        mutated = true
                    } else {
                        // Always rebuild with canonical operand references — even if
                        // operand ids are unchanged, the REFERENCES may now point to
                        // rebuilt ops in newBody rather than the originals. Using the
                        // id-equality shortcut can leave stale references pointing to
                        // ops that are no longer in the body.
                        val rebuilt = DxirOp(
                            id = n.id,
                            op = n.op,
                            operands = canonicalOperands,
                            attrs = n.attrs,
                            types = n.types,
                            sharding = n.sharding,
                            regions = emptyList(),
                        )
                        sig2canon[sig] = rebuilt
                        byId[n.id] = rebuilt
                        newBody += rebuilt
                        if (rebuilt !== n) mutated = true
                    }
                }
                else -> {
                    byId[n.id] = n
                    newBody += n
                }
            }
        }
        if (!mutated) return fn
        val newReturns = fn.returns.map { byId[it.id] ?: it }
        return DxirFunction(fn.name, fn.params, newBody, newReturns, fn.meshes)
    }

    /**
     * §0.4.48 — peephole constant folding on scalar numeric ops. Handles:
     * - `MUL(const a, const b)` → `const(a*b)`
     * - `ADD(const a, const b)` → `const(a+b)`
     * - `SUB(const a, const b)` → `const(a-b)`
     * - `DIV(const a, const b)` → `const(a/b)` (when b ≠ 0)
     * - `NEG(const a)` → `const(-a)`
     * - `MUL(x, const 1)` / `MUL(const 1, x)` → `x`
     * - `MUL(x, const 0)` / `MUL(const 0, x)` → `const(0)`
     * - `ADD(x, const 0)` / `ADD(const 0, x)` → `x`
     * - `SUB(x, const 0)` → `x`
     *
     * Runs AFTER [applyCSE] so `ADD(%a, %a)` (the CSE-merged form of `ADD(%a,
     * %a_duplicate)`) is visible as a same-id operand pair, enabling the `x + x`
     * = `2 * x` folding as a side effect of recognising that the operand ids
     * match. (Not yet implemented — [MulRule]-heavy adjoint chains will keep
     * benefiting from CSE without this folding.)
     *
     * Scope: F32 / F64 scalar const ops only — bool / int arithmetic isn't
     * exercised by any adjoint rule today.
     */
    private fun applyConstFold(fn: DxirFunction): DxirFunction {
        // Same region-bearing guard as applyCSE — const-fold rewrites operand
        // references and can leave region-internal references dangling.
        if (fn.body.any { it is DxirOp && it.hasRegions }) return fn
        val byId = HashMap<Int, DxirNode>()
        for (p in fn.params) byId[p.id] = p
        val newBody = mutableListOf<DxirNode>()
        var mutated = false

        fun asFloatConst(n: DxirNode): Float? {
            val resolved = byId[n.id] ?: return null
            if (resolved !is DxirConst) return null
            return (resolved.value as? Number)?.toFloat()
        }

        fun emit(replacement: DxirNode, original: DxirNode) {
            byId[original.id] = replacement
            if (replacement !in newBody && (replacement is DxirOp || replacement is DxirConst)) {
                // Avoid adding the replacement twice — if it's a pre-existing node from `byId`,
                // it's already been emitted earlier. Only add when it's a fresh const.
                if (newBody.none { it.id == replacement.id }) newBody += replacement
            }
            mutated = true
        }

        for (n in fn.body) {
            when (n) {
                is DxirConst -> {
                    byId[n.id] = n
                    newBody += n
                }
                is DxirOp -> {
                    // Non-foldable shapes (multi-result, region-bearing, non-scalar,
                    // non-F32/F64) still need operand rebuilding — prior CSE / fold
                    // passes may have replaced operands, and keeping stale references
                    // would leave dangling id references in the function body.
                    val shouldOnlyRebuild = n.isMultiResult || n.hasRegions ||
                        !n.type.isScalar ||
                        (n.type.dtype != io.tlaloc.core.F32 && n.type.dtype != io.tlaloc.core.F64)
                    if (shouldOnlyRebuild) {
                        val canonicalOperands = n.operands.map { byId[it.id] ?: it }
                        val rebuilt = DxirOp(
                            id = n.id,
                            op = n.op,
                            operands = canonicalOperands,
                            attrs = n.attrs,
                            types = n.types,
                            sharding = n.sharding,
                            regions = n.regions,
                        )
                        if (rebuilt !== n) mutated = true
                        byId[n.id] = rebuilt
                        newBody += rebuilt
                        continue
                    }
                    // Try to fold.
                    val folded: DxirNode? = when (n.op) {
                        OpKind.MUL -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null -> DxirConst(n.id, a * b, n.type)
                                a == 1.0f -> byId[n.operands[1].id] ?: n.operands[1]
                                b == 1.0f -> byId[n.operands[0].id] ?: n.operands[0]
                                a == 0.0f || b == 0.0f -> DxirConst(n.id, 0.0f, n.type)
                                else -> null
                            }
                        }
                        OpKind.ADD -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null -> DxirConst(n.id, a + b, n.type)
                                a == 0.0f -> byId[n.operands[1].id] ?: n.operands[1]
                                b == 0.0f -> byId[n.operands[0].id] ?: n.operands[0]
                                else -> null
                            }
                        }
                        OpKind.SUB -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null -> DxirConst(n.id, a - b, n.type)
                                b == 0.0f -> byId[n.operands[0].id] ?: n.operands[0]
                                else -> null
                            }
                        }
                        OpKind.DIV -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null && b != 0.0f -> DxirConst(n.id, a / b, n.type)
                                b == 1.0f -> byId[n.operands[0].id] ?: n.operands[0]
                                else -> null
                            }
                        }
                        OpKind.NEG -> {
                            val a = asFloatConst(n.operands[0])
                            if (a != null) DxirConst(n.id, -a, n.type) else null
                        }
                        else -> null
                    }
                    if (folded != null) {
                        emit(folded, n)
                    } else {
                        // Always rebuild with canonical operand references (see applyCSE
                        // for the "stale reference" reasoning).
                        val canonicalOperands = n.operands.map { byId[it.id] ?: it }
                        val rebuilt = DxirOp(
                            id = n.id,
                            op = n.op,
                            operands = canonicalOperands,
                            attrs = n.attrs,
                            types = n.types,
                            sharding = n.sharding,
                            regions = emptyList(),
                        )
                        if (rebuilt !== n) mutated = true
                        byId[n.id] = rebuilt
                        newBody += rebuilt
                    }
                }
                else -> {
                    byId[n.id] = n
                    newBody += n
                }
            }
        }
        if (!mutated) return fn
        val newReturns = fn.returns.map { byId[it.id] ?: it }
        return DxirFunction(fn.name, fn.params, newBody, newReturns, fn.meshes)
    }

    /**
     * §0.4.45 — simple DCE over a function's body. Marks reachable nodes by walking
     * from [DxirFunction.returns] through operand + nested-region references,
     * transitively. Unreferenced top-level body ops are filtered out. Params are
     * always preserved (so the declared signature survives even if the gradient
     * doesn't dereference a param, which is common — e.g., the idx operand of a
     * GATHER whose adjoint is zero). Nested region bodies aren't pruned here —
     * region-internal DCE is deferred; the Stage B.3 DCE pass the PhiCalculus
     * comments reference was scoped to a future session.
     *
     * Primary motivation: [GatherRule] emits `SCATTER_ADD(zero_bcast, idx, upstream)`;
     * [applyC5Pass]-side gradAccum fusion rewrites subsequent SCATTER_ADDs to thread
     * the accumulator forward, orphaning the per-gather zero-broadcast. Without DCE
     * those orphans survive into synthesis and re-introduce the per-gather
     * allocation overhead the fusion was meant to eliminate.
     */
    private fun dropUnreachableBody(fn: DxirFunction): DxirFunction {
        val reachable = HashSet<Int>()
        val queue = ArrayDeque<Int>()
        fun enqueue(id: Int) { if (reachable.add(id)) queue += id }
        for (r in fn.returns) enqueue(r.id)
        val byId = fn.body.associateBy { it.id }
        while (queue.isNotEmpty()) {
            val id = queue.removeFirst()
            val node = byId[id] ?: continue // params have ids but aren't in body
            if (node is DxirOp) {
                for (o in node.operands) enqueue(o.id)
                for (region in node.regions) for (block in region.blocks) {
                    for (b in block.body) enqueue(b.id)
                    for (term in block.terminator) enqueue(term.id)
                }
            }
        }
        val prunedBody = fn.body.filter { it.id in reachable }
        if (prunedBody.size == fn.body.size) return fn
        return DxirFunction(fn.name, fn.params, prunedBody, fn.returns, fn.meshes)
    }

    /**
     * §0.4.46 — tag each `OpKind.SCATTER_ADD` whose `operand[0]` (the accumulator
     * base) has exactly one use in the function. The single-use invariant makes
     * destructive in-place mutation safe: the cloned buffer isn't aliased anywhere
     * else, so `base[idx] += value` can write through to `base`'s storage without
     * corrupting a shared view.
     *
     * GatherRule's post-§0.4.45 emission pattern is exactly this shape — a linear
     * chain where each SCATTER_ADD's base is the previous SCATTER_ADD's result
     * (or, for the first, a BROADCAST(0) used only as that SCATTER_ADD's base).
     * Every SCATTER_ADD in the accumulator chain qualifies; in-place lowering
     * eliminates the `FloatArray.copyOf` inside [io.tlaloc.core.ops.scatterAddInto]
     * across the whole chain, not just the terminal step.
     *
     * The tag lands as an attr on the dxir op (`"in_place" = true`); the attr is
     * consumed by `DxirToIrSynthesis.irScatterAdd` which picks
     * `scatterAddInPlace` vs `scatterAddInto` accordingly. Interpreter + stablehlo
     * lowering are NOT affected by the tag — they preserve non-destructive
     * semantics regardless, because the tag's correctness depends on the SSA
     * use-graph being exactly the grad body (interpreter callers could evaluate
     * the same function with different aliasing guarantees; stablehlo lowering
     * should always be value-semantic at the MLIR level).
     *
     * Use-graph: for every `DxirNode` id in the body, count the number of
     * positions where it appears as an operand — in top-level body ops, in
     * nested-region ops, and in `fn.returns`. An id with count = 1 is
     * single-use. A SCATTER_ADD whose base's id is single-use gets tagged.
     *
     * Narrow scope: only tags top-level SCATTER_ADDs whose base is another
     * top-level body op or a param (not nested-region ops). Nested region
     * SCATTER_ADDs are rare today; extending the tag to them needs richer
     * region-aware use counting.
     */
    private fun tagSingleUseScatterAdds(fn: DxirFunction): DxirFunction {
        // 1. Build the use-count table.
        val useCount = HashMap<Int, Int>()
        fun bump(id: Int) { useCount[id] = (useCount[id] ?: 0) + 1 }
        fun walkOperandsIn(nodes: List<DxirNode>) {
            for (n in nodes) {
                if (n !is DxirOp) continue
                for (o in n.operands) bump(o.id)
                for (region in n.regions) for (block in region.blocks) {
                    walkOperandsIn(block.body)
                    for (term in block.terminator) bump(term.id)
                }
            }
        }
        walkOperandsIn(fn.body)
        for (r in fn.returns) bump(r.id)

        // 2. Re-emit body with tagged SCATTER_ADD attrs where safe.
        var mutated = false
        val rewritten = fn.body.map { n ->
            if (n !is DxirOp) return@map n
            if (n.op != OpKind.SCATTER_ADD) return@map n
            if (n.operands.isEmpty()) return@map n
            val baseId = n.operands[0].id
            if ((useCount[baseId] ?: 0) != 1) return@map n
            // Already tagged? skip.
            if (n.attrs["in_place"] == true) return@map n
            mutated = true
            DxirOp(
                id = n.id,
                op = n.op,
                operands = n.operands,
                attrs = n.attrs + ("in_place" to true),
                types = n.types,
                sharding = n.sharding,
                regions = n.regions,
            )
        }
        if (!mutated) return fn
        return DxirFunction(fn.name, fn.params, rewritten, fn.returns, fn.meshes)
    }

    /**
     * Primal node ids whose cloned form must appear in the gradient body. Seeded by every
     * operand each [VjpRule] dereferences, plus the primal return when [includeForward]
     * is on, then transitively closed through [DxirOp.operands] so every operand-reference
     * chain reaching a used node is itself cloneable.
     */
    private fun computeUsedByAdjoint(
        primal: DxirFunction,
        primalById: Map<Int, DxirNode>,
        includeForward: Boolean,
    ): Set<Int> {
        val used = HashSet<Int>()
        val worklist = ArrayDeque<Int>()
        fun enqueue(id: Int) {
            if (used.add(id)) worklist.addLast(id)
        }
        for (n in primal.body) {
            if (n !is DxirOp) continue
            // §0.4.23 — IF ops have no entry in VjpRegistry but the synthesized adjoint
            // IF dereferences the predicate, so it must be in usedByAdjoint.
            // §0.4.24 — also enqueue every outer-scope id referenced INSIDE an IF's
            // regions. `walkBranchReverse` clones branch body ops into the grad builder
            // and resolves their operands through `outerNodeMap`; if an outer-scope
            // operand isn't cloned (i.e., absent from usedByAdjoint), outerNodeMap
            // points at the primal node, whose id collides with the grad builder's
            // fresh allocations → semantically wrong (but ref-integrity-valid)
            // gradients. PhiCalculus F2/C1 distribution exposes this path by pushing
            // previously-top-level ops into branch bodies; B.4a requires the fix.
            if (n.op == OpKind.IF) {
                enqueue(n.operands[0].id)
                enqueueRegionOuterRefs(n, ::enqueue)
                continue
            }
            // §0.4.32 — COARSENED op: the reads set lives in attrs (per-instance data,
            // not a static VjpRule val). [handleCoarsenedAdjoint] uses the cloned
            // primal operands at those indices when splicing the gradient_body, so
            // those operand subgraphs must be in the gradient body.
            if (n.op == OpKind.COARSENED) {
                val reads = n.attrs["reads_primal_indices"] as? Set<*>
                if (reads != null) {
                    for (v in reads) if (v is Int) enqueue(n.operands[v].id)
                }
                continue
            }
            val rule = VjpRegistry[n.op] ?: continue
            for (i in rule.readsPrimalOperandIndices) {
                enqueue(n.operands[i].id)
            }
        }
        if (includeForward) {
            enqueue(primal.returns.single().id)
        }
        while (worklist.isNotEmpty()) {
            val id = worklist.removeFirst()
            val node = primalById[id] ?: continue
            if (node is DxirOp) {
                for (operand in node.operands) enqueue(operand.id)
            }
        }
        return used
    }

    /**
     * §0.4.23 — handle the IF op as a special case in [DxirReverseTransform.apply]'s
     * reverse walk. Per paper C2 (`d/dx(φ(a, b)) = φ(da/dx, db/dx)`), the gradient of an
     * IF result distributes through its branches: for each outer-scope value `x`
     * referenced in a branch, the contribution to `x`'s gradient is an IF that picks
     * the per-branch gradient based on the runtime predicate.
     *
     * Algorithm:
     *  1. For each branch, [walkBranchReverse] performs a recursive reverse walk:
     *     clone the branch body ops into the outer gradient builder, seed `gradAccum`
     *     with `upstream` at the branch's yield id, walk in reverse applying VjpRules
     *     per the standard pattern. Returns a per-id gradient map for outer-scope
     *     values referenced in the branch.
     *  2. For each id with at least one per-branch gradient, synthesize an IF in the
     *     outer gradient body: `IF(pred, thenAdj, elseAdj)`. Missing-branch
     *     contributions become `const(0)`. Accumulate into the outer `gradAccum`.
     *
     * Limitations (first cut, §0.4.23):
     *  - Single-result IF only.
     *  - Branch bodies must NOT contain nested control flow (no nested IF/WHILE).
     *  - Branch bodies must not contain multi-result ops.
     *  - `usedByAdjoint` analysis isn't extended into branches — all branch body ops
     *    are unconditionally cloned into the gradient body.
     */
    private fun handleIfAdjoint(
        ifNode: DxirOp,
        upstream: DxirNode,
        outerGradAccum: MutableMap<Int, DxirNode>,
        outerNodeMap: Map<Int, DxirNode>,
        primalById: Map<Int, DxirNode>,
        builder: DxirBuilder,
    ) {
        require(ifNode.op == OpKind.IF) { "handleIfAdjoint: not an IF (got ${ifNode.op})" }
        require(ifNode.types.size == 1) {
            "handleIfAdjoint: only single-result IF supported (got ${ifNode.types.size} results)"
        }
        val predClone = outerNodeMap[ifNode.operands[0].id]
            ?: error("handleIfAdjoint: predicate id=${ifNode.operands[0].id} not in nodeMap")

        val thenBlock = ifNode.regions[0].blocks.single()
        val elseBlock = ifNode.regions[1].blocks.single()

        val thenAdjoints = walkBranchReverse(thenBlock, upstream, outerNodeMap, primalById, builder)
        val elseAdjoints = walkBranchReverse(elseBlock, upstream, outerNodeMap, primalById, builder)

        // Combine per-branch adjoints into a synthesized IF for each outer-scope id.
        // LinkedHashSet preserves insertion order so the IF emission order is
        // deterministic (helps test stability).
        val allIds = LinkedHashSet<Int>()
        allIds.addAll(thenAdjoints.keys)
        allIds.addAll(elseAdjoints.keys)

        for (id in allIds) {
            val primal = primalById[id] ?: continue
            // Skip constants — they have no gradient surface.
            if (primal is DxirConst) continue
            val primalType = primal.type
            val thenAdj = thenAdjoints[id]
                ?: builder.const(zeroValueFor(primalType.dtype), primalType)
            val elseAdj = elseAdjoints[id]
                ?: builder.const(zeroValueFor(primalType.dtype), primalType)

            val condIf = builder.ifOp(
                cond = predClone,
                types = listOf(primalType),
                thenRegion = builder.region { yields(thenAdj) },
                elseRegion = builder.region { yields(elseAdj) },
            )

            val existing = outerGradAccum[id]
            outerGradAccum[id] = if (existing == null) condIf
            else builder.op(OpKind.ADD, listOf(existing, condIf), primalType)
        }
    }

    /**
     * §0.4.32 — handle the [OpKind.COARSENED] op in the reverse walk. Mirrors
     * [handleIfAdjoint]'s role: the standard [VjpRule] contract doesn't fit COARSENED
     * (the reads set is per-instance data in attrs, not a static val), so
     * [DxirReverseTransform.apply]'s dispatch routes COARSENED to this helper directly.
     *
     * Algorithm:
     *  1. Read `attrs["gradient_body"]` — a [DxirFunction] with signature
     *     `(upstream, *primal_operands) → (d_operand_i, …)`.
     *  2. Build a `gradNodeMap: old-id → new-node` seeded with:
     *      - `gradient_body.params[0]` → `upstream` (the gradient flowing into the
     *        COARSENED op's primal result).
     *      - `gradient_body.params[i+1]` → `outerNodeMap[coarsened.operands[i].id]`
     *        (the cloned primal operand; in the gradient builder's scope).
     *  3. Clone each op in `gradient_body.body` into [builder] with operand references
     *     resolved through `gradNodeMap`.
     *  4. Each `gradient_body.returns[i]` is the gradient contribution for
     *     `coarsened.operands[i]`. Accumulate into `outerGradAccum` via the standard
     *     ADD-on-existing convention.
     *
     * Limitations (C.3b.2 first cut):
     *  - Single-result COARSENED only (the top-level guard still rejects multi-result).
     *  - `gradient_body.body` must be straight-line (no regions, no multi-result).
     *    Real VJPs produced by the coarsening pipeline (C.3b.3) satisfy this; hand-
     *    built gradient_bodies with regions would need recursive handling.
     */
    private fun handleCoarsenedAdjoint(
        coarsened: DxirOp,
        upstream: DxirNode,
        outerGradAccum: MutableMap<Int, DxirNode>,
        outerNodeMap: Map<Int, DxirNode>,
        primalById: Map<Int, DxirNode>,
        builder: DxirBuilder,
    ) {
        require(coarsened.op == OpKind.COARSENED) {
            "handleCoarsenedAdjoint: not a COARSENED op (got ${coarsened.op})"
        }
        require(coarsened.types.size == 1) {
            "handleCoarsenedAdjoint: only single-result COARSENED supported " +
                "(got ${coarsened.types.size} results)"
        }
        val gradBody = coarsened.attrs["gradient_body"] as? io.tlaloc.ir.DxirFunction
            ?: error("handleCoarsenedAdjoint: COARSENED op id=${coarsened.id} missing gradient_body attr")
        require(gradBody.params.size == 1 + coarsened.operands.size) {
            "handleCoarsenedAdjoint: gradient_body param count ${gradBody.params.size} ≠ " +
                "1 + ${coarsened.operands.size} (upstream + primal operands)"
        }
        require(gradBody.returns.size == coarsened.operands.size) {
            "handleCoarsenedAdjoint: gradient_body return count ${gradBody.returns.size} ≠ " +
                "operand count ${coarsened.operands.size}"
        }

        // Step 2: seed gradient-body param ids → outer nodes.
        val gradNodeMap = HashMap<Int, DxirNode>()
        gradNodeMap[gradBody.params[0].id] = upstream
        for (i in coarsened.operands.indices) {
            val operandClone = outerNodeMap[coarsened.operands[i].id]
                ?: error(
                    "handleCoarsenedAdjoint: primal operand id=${coarsened.operands[i].id} " +
                        "missing from outerNodeMap (expected cloned or primal verbatim)",
                )
            gradNodeMap[gradBody.params[i + 1].id] = operandClone
        }

        // Step 3: clone gradient_body.body into the outer gradient builder. Straight-
        // line only — we reject regions + multi-result to keep C.3b.2 first-cut simple.
        for (n in gradBody.body) {
            val cloned: DxirNode = when (n) {
                is DxirConst -> builder.const(n.value, n.type, n.sharding)
                is DxirOp -> {
                    require(!n.hasRegions) {
                        "handleCoarsenedAdjoint: gradient_body op ${n.op} has regions " +
                            "(nested control flow in gradient body not supported in C.3b.2)"
                    }
                    require(!n.isMultiResult) {
                        "handleCoarsenedAdjoint: gradient_body op ${n.op} is multi-result " +
                            "(not supported in C.3b.2)"
                    }
                    val clonedOperands = n.operands.map {
                        gradNodeMap[it.id]
                            ?: error(
                                "handleCoarsenedAdjoint: gradient_body op id=${n.id} references " +
                                    "unknown id=${it.id} (gradient_body has broken SSA?)",
                            )
                    }
                    builder.op(n.op, clonedOperands, n.type, n.attrs, n.sharding, emptyList())
                }
                else -> error(
                    "handleCoarsenedAdjoint: unsupported gradient_body node " +
                        "${n::class.simpleName} (id=${n.id})",
                )
            }
            gradNodeMap[n.id] = cloned
        }

        // Step 4: map each gradient_body return → primal operand contribution.
        for (i in gradBody.returns.indices) {
            val contribution = gradNodeMap[gradBody.returns[i].id]
                ?: error(
                    "handleCoarsenedAdjoint: gradient_body.returns[$i] id=" +
                        "${gradBody.returns[i].id} missing from gradNodeMap",
                )
            val primalOperand = coarsened.operands[i]
            // Constants have no gradient surface — skip.
            if (primalById[primalOperand.id] is DxirConst) continue
            val existing = outerGradAccum[primalOperand.id]
            outerGradAccum[primalOperand.id] = if (existing == null) contribution
            else builder.op(OpKind.ADD, listOf(existing, contribution), contribution.type)
        }
    }

    /**
     * Reverse-walk a single IF branch's body. Clones branch body ops into [builder]'s
     * scope so adjoints can reference them, then performs the standard reverse-walk
     * pattern (seed `gradAccum` at the yield's id with `upstream`, walk body in reverse
     * applying VjpRules, accumulate per-id contributions).
     *
     * Returns the per-id gradient map. Outer-scope ids may appear (when the branch
     * yields or references an outer SSA value); branch-internal ids appear too but are
     * filtered out by the caller (only outer-scope ids contribute back to the outer
     * `gradAccum`).
     *
     * The current implementation rejects nested IF/WHILE in the branch body — recursion
     * into nested IFs is a §0.4.23+ extension.
     */
    private fun walkBranchReverse(
        block: io.tlaloc.ir.DxirBlock,
        upstream: DxirNode,
        outerNodeMap: Map<Int, DxirNode>,
        primalById: Map<Int, DxirNode>,
        builder: DxirBuilder,
    ): MutableMap<Int, DxirNode> {
        // Step 1: clone branch body ops into the outer builder. branchNodeMap starts as
        // a copy of outerNodeMap (so outer-scope refs resolve) and accumulates clones
        // for each branch body op.
        val branchNodeMap = HashMap<Int, DxirNode>(outerNodeMap)
        for (n in block.body) {
            when (n) {
                is DxirConst -> {
                    branchNodeMap[n.id] = builder.const(n.value, n.type, n.sharding)
                }
                is DxirOp -> {
                    require(!n.hasRegions) {
                        "walkBranchReverse: nested control-flow op ${n.op} in IF branch " +
                            "not yet supported (§0.4.23 first cut)"
                    }
                    require(!n.isMultiResult) {
                        "walkBranchReverse: multi-result op ${n.op} in IF branch not yet supported"
                    }
                    // §0.4.36 — don't clone COARSENED into the grad body. The reverse-walk
                    // dispatch (below) splices the stored gradient_body via
                    // [handleCoarsenedAdjoint]; a cloned COARSENED would be orphaned in the
                    // grad body (no downstream consumer of its result) and [DxirToIrSynthesis]
                    // has no lowering arm for it, so it would cause synthesis to abort. Skip
                    // cloning + keep branchNodeMap pointing at the original COARSENED so
                    // handleCoarsenedAdjoint can read its attrs at reverse-walk time.
                    if (n.op == OpKind.COARSENED) {
                        branchNodeMap[n.id] = n
                    } else {
                        val clonedOperands = n.operands.map {
                            branchNodeMap[it.id]
                                ?: error("walkBranchReverse: operand id=${it.id} of branch op ${n.id} not in nodeMap")
                        }
                        branchNodeMap[n.id] = builder.op(n.op, clonedOperands, n.type, n.attrs)
                    }
                }
                else -> error(
                    "walkBranchReverse: unsupported branch body node ${n::class.simpleName}",
                )
            }
        }

        // Step 2: seed the per-branch gradAccum with `upstream` at the yield's id.
        val yieldNode = block.terminator.single()
        val gradAccum = HashMap<Int, DxirNode>()
        gradAccum[yieldNode.id] = upstream

        // Step 3: reverse walk through the branch body.
        for (n in block.body.asReversed()) {
            if (n !is DxirOp) continue
            val upstreamForN = gradAccum[n.id] ?: continue
            // §0.4.34 — COARSENED in a branch body: splice the gradient_body via the
            // same helper the outer reverse walk uses. Shared logic keeps the
            // gradient-through-coarsened semantics identical regardless of whether the
            // COARSENED sits at the function's top level or inside an IF branch. The
            // clone (`branchNodeMap[n.id]`) carries branch-scope operand references;
            // handleCoarsenedAdjoint reads gradient_body and emits contributions into
            // the branch's gradAccum (not the outer one).
            if (n.op == OpKind.COARSENED) {
                val clonedCoarsened = branchNodeMap[n.id] as DxirOp
                handleCoarsenedAdjoint(
                    coarsened = clonedCoarsened,
                    upstream = upstreamForN,
                    outerGradAccum = gradAccum,
                    outerNodeMap = branchNodeMap,
                    primalById = primalById,
                    builder = builder,
                )
                continue
            }
            val rule = VjpRegistry[n.op] ?: error(
                "walkBranchReverse: no VJP rule registered for ${n.op} in branch body",
            )
            val clonedOp = branchNodeMap[n.id] as DxirOp
            val contributions = rule.apply(clonedOp, upstreamForN, builder)
            for ((operandKey, contribution) in contributions) {
                val operandIdx = clonedOp.operands.indexOf(operandKey)
                require(operandIdx >= 0) {
                    "VjpRule for ${n.op} returned an operand key that is not in op.operands"
                }
                val primalOperand = n.operands[operandIdx]
                if (primalById[primalOperand.id] is DxirConst) continue
                val existing = gradAccum[primalOperand.id]
                gradAccum[primalOperand.id] = if (existing == null) contribution
                else builder.op(OpKind.ADD, listOf(existing, contribution), contribution.type)
            }
        }

        return gradAccum
    }

    /**
     * §0.4.24 — walk an IF (or any region-bearing op) and enqueue every operand id
     * referenced by body ops inside its regions. Recurses into nested regions. The
     * enqueue worklist filters out region-internal ids later (via `primalById[id] ?: continue`),
     * so pushing every operand id is safe — the transitive closure ignores ids that
     * don't correspond to top-level body nodes.
     */
    private fun enqueueRegionOuterRefs(node: DxirOp, enqueue: (Int) -> Unit) {
        for (region in node.regions) {
            for (block in region.blocks) {
                for (bodyOp in block.body) {
                    if (bodyOp !is DxirOp) continue
                    for (operand in bodyOp.operands) enqueue(operand.id)
                    if (bodyOp.hasRegions) enqueueRegionOuterRefs(bodyOp, enqueue)
                }
                for (term in block.terminator) enqueue(term.id)
            }
        }
    }

    private fun seedValueFor(dtype: io.tlaloc.core.DType): Any = when (dtype) {
        F32 -> 1.0f
        F64 -> 1.0
        I32 -> 1
        I64 -> 1L
        else -> error("DxirReverseTransform: cannot seed adjoint for dtype $dtype")
    }

    private fun zeroValueFor(dtype: io.tlaloc.core.DType): Any = when (dtype) {
        F32 -> 0.0f
        F64 -> 0.0
        I32 -> 0
        I64 -> 0L
        io.tlaloc.core.Bool -> 0.0f  // Bool dtype uses F32 0/1 encoding (see OpKind.STEP/NOT comments).
        else -> error("DxirReverseTransform: cannot zero-seed gradient for dtype $dtype")
    }

    /** §0.4.54 — integer dtypes whose gradients are structurally zero. */
    private fun isIntegerDtype(dtype: io.tlaloc.core.DType): Boolean =
        dtype == I32 || dtype == I64 || dtype == io.tlaloc.core.Bool
}
