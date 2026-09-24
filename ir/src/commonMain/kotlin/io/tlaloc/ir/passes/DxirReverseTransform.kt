package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBlock
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirRegion
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * Gradient-accumulator key. The contribution that feeds back into a
 * primal value is keyed by `(primal node id, result index)`. For single-result
 * primal nodes (params, consts, single-result `DxirOp`) the index is always 0.
 * For [DxirOpResult] (a multi-result op's result reference), the index is
 * `node.index` and the id is `node.source.id`. Centralising this mapping in one
 * extension keeps every gradAccum read / write at every call site (top-level,
 * branch, COARSENED) consistent, and lets contributions be seeded at distinct
 * result indices without changing the accumulator.
 */
private fun DxirNode.gradKey(): Pair<Int, Int> = when (this) {
    is DxirOpResult -> source.id to index
    else -> id to 0
}

/**
 * Source-code-transformation reverse-mode AD on a [DxirFunction]. Given a primal function
 * `f: P₁ × … × Pₙ → R` (R a single scalar), produces a gradient function whose returns are
 * the per-parameter gradients of `f`'s output. When [apply] is called with
 * `includeForward = true`, the primal return value is prepended to the returns list so
 * callers can synthesise `valueAndGrad` / `valueAndGrad2` without re-running the forward.
 *
 * The coarsening pass ([PhiCalculus]) builds on this transform. It is intentionally narrow:
 *
 * - **Single scalar return only** on the default (const-1.0-seed) path. Vector / tensor
 *   outputs require an explicit upstream-cotangent parameter — which is exactly what
 *   `seedAsParam = true` provides: with a caller-supplied seed the reverse walk
 *   is seed-agnostic, so the single return may be any type and the transform is a true
 *   pullback `(ȳ, x) → x̄`.
 * - **Straight-line bodies only.** Ops carrying nested regions (e.g. `MANUAL_COMPUTATION`,
 *   `WHILE`) are rejected, except for the `IF` and `COARSENED` arms; loops are removed
 *   beforehand by the φ-calculus coarsening pass.
 * - **Single-result body ops only.** Multi-result ops (other than the sanctioned
 *   `IF`/`COARSENED` arms) would need per-result accumulator threading and are rejected.
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
 *    emit a fresh `ADD`. For
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
     * @param seedAsParam If true, the gradient function's first param is an
     *   explicit `upstream` (typed to match the primal's return type) that seeds
     *   `gradAccum[ret.id]`. The resulting signature is
     *   `(upstream, *primal_params) → (*grads)`, which is what
     *   [OpKind.COARSENED]'s `gradient_body` attribute expects. When false (default),
     *   the seed is `const(1.0)` — the existing `grad` / `valueAndGrad` behaviour.
     *   May be combined with `includeForward`: the combined
     *   mode `(upstream, *primal_params) → (y, *grads)` is the `valueAndVjp` shape
     *   (a COARSENED gradient_body still never passes both, but the seeded-cotangent
     *   user surface needs the primal value alongside the pullback).
     * @param inputOnlyTrailingParams How many of the primal's LAST params
     *   are inputs only: no gradient is emitted for them, so `returns` carries
     *   `params.size - inputOnlyTrailingParams` gradients instead of one per param.
     *   This is what the K2 plugin's captured runtime values are — a `grad { x -> f(x)
     *   * scale }` lowers `scale` to a trailing param that the IR phase binds at the
     *   CALL SITE, and the user asked for the derivative with respect to the lambda's
     *   declared parameters only. The adjoint chain that fed a dropped gradient becomes
     *   unreachable and the following `dropUnreachableBody` removes it, so the returned
     *   function is the same one a primal without that param would have produced,
     *   plus the param. Default 0: every param gets a gradient.
     */
    fun apply(
        primal: DxirFunction,
        includeForward: Boolean = false,
        seedAsParam: Boolean = false,
        inputOnlyTrailingParams: Int = 0,
    ): DxirFunction {
        // §0.4.212 — Pre-pass `PhiCalculus.liftIfRegionBodies` to hoist safe arithmetic
        // ops out of IF region bodies. Without this, IFs with non-empty regions (e.g.,
        // SUB(state, maxAngle) inside a coarsened-WHILE-unroll's collision IF) survive
        // into the clone-and-rewrite step. The §0.4.173 / §0.4.174 / §0.4.175 arc
        // documented this as a "KNOWN LEAK" — the §0.4.175 deep-clone arm only fires
        // when `regionsAllEmpty` holds. Pre-§0.4.212, only callers that explicitly
        // ran `liftIfRegionBodies` first (the K2 plugin's `TlalocIrGenerationExtension`)
        // got correct behaviour; direct consumers via `:benchmarks` (e.g., the
        // QWOP `Qwop.hipUpdatePrimal` test surfaced in §0.4.211) hit the leak. Calling
        // the lift pass here is idempotent (returns the input function unchanged when
        // not safe to lift) and self-contained — making `DxirReverseTransform.apply`
        // produce well-formed gradient functions regardless of whether the caller ran
        // the lift step.
        @Suppress("NAME_SHADOWING")
        val primal = PhiCalculus.liftIfRegionBodies(primal)
        // §0.4.501 — a bad count would silently drop a REAL gradient (or index past
        // the param list), which is the one failure mode a captured-value feature must
        // not have. Refuse by name.
        require(inputOnlyTrailingParams in 0..primal.params.size) {
            "DxirReverseTransform: inputOnlyTrailingParams=$inputOnlyTrailingParams is outside " +
                "0..${primal.params.size} for '${primal.name}'"
        }
        require(primal.returns.size == 1) {
            "DxirReverseTransform v1 requires exactly 1 return value (got ${primal.returns.size})"
        }
        val ret = primal.returns.single()
        // §0.4.398 — the scalar gate applies only to the const-1.0-seed path: a unit
        // seed is meaningful only for a scalar objective. With `seedAsParam` the
        // upstream param takes the primal return's type VERBATIM (tensor allowed) and
        // the walk below is seed-agnostic — every VjpRule already handles tensor
        // upstreams (that is how interior ops' adjoints flow under `grad {}`); the
        // final op's upstream being a tensor rather than const(1.0) changes nothing.
        require(seedAsParam || ret.type.isScalar) {
            "DxirReverseTransform v1 requires a scalar return (got ${ret.type}) — " +
                "tensor-returning primals need seedAsParam (the vjp pullback form)"
        }
        // §0.4.139 — multi-result IFs are allowed at the top level. The seed flows
        // into `gradAccum[(if.id, k)]` for each result index `k` referenced
        // downstream (DxirOpResult.id == source.id). The per-branch reverse walk
        // is seeded at every live index's terminator slot. §0.4.155 — Phase 5b:
        // multi-live-index now supported via `Set<Int>`; the §0.4.154 substrate
        // makes the per-index keying uniform across the dispatch chain.
        val ifLiveIndices = HashMap<Int, Set<Int>>()
        for (n in primal.body) {
            if (n is DxirOp) {
                // §0.4.448 — audit finding C: the demoted kinds refuse BY NAME
                // with the sanctioned alternative in the message (see
                // [demotedKindRefusal]), ahead of the generic multi-result gate
                // below (which would otherwise catch a demoted kind with no
                // directions)
                // and ahead of the walk's index-0 upstream lookup (which would
                // skip an op with no accumulated upstream silently). Branch
                // bodies get the same guard inside [walkBranchReverse].
                demotedKindRefusal(n.op, "DxirReverseTransform")?.let { error(it) }
                // §0.4.465 — Phase H1a: the INFERENCE-ONLY kinds refuse here
                // too, by name, with the TRAINING spelling in the message.
                // Same gate, different reason: these kinds DO execute (they
                // have interpreter arms and emission), they simply have no
                // adjoint that means anything — see [INFERENCE_ONLY_OP_KINDS].
                inferenceOnlyKindRefusal(n.op, "DxirReverseTransform")?.let { error(it) }
                if (n.hasRegions) {
                    require(n.op == OpKind.IF) {
                        "DxirReverseTransform: op ${n.op} has regions but no rule supports " +
                            "regions for it (only IF is handled directly, by a reverse walk of its " +
                            "branches; WHILE and others must be coarsened by the " +
                            "PhiCalculus pass first)"
                    }
                }
                if (n.isMultiResult) {
                    // §0.4.179 — Phase 5c: multi-result COARSENED accepted alongside
                    // multi-result IF. handleCoarsenedAdjoint reads K upstreams from
                    // the per-index gradAccum (§0.4.154 substrate) and seeds dead
                    // indices with const(0). No live-index pre-check needed for
                    // COARSENED — the dispatch in step 3 collects whichever indices
                    // have accumulated contribution and skips the op entirely if none.
                    require(n.op == OpKind.IF || n.op == OpKind.COARSENED) {
                        "DxirReverseTransform v1 rejects multi-result ops (got ${n.op})"
                    }
                    if (n.op == OpKind.IF) {
                        val liveIndices = findIfLiveResultIndices(primal, n)
                        require(liveIndices.isNotEmpty()) {
                            "DxirReverseTransform: multi-result IF id=${n.id} has zero result " +
                                "indices referenced downstream — the IF op is dead and should " +
                                "be DCE'd before SCT"
                        }
                        ifLiveIndices[n.id] = liveIndices
                    }
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
                            // §0.4.175 — when the IF's regions are EMPTY (single-block
                            // each, no body ops, just terminator), deep-clone the IF
                            // into the grad body so downstream consumers that reference
                            // it as a forward operand resolve to a grad-scope op rather
                            // than leaking the primal id. This is safe + cheap because
                            // PhiCalculus.liftIfRegionBodies (§0.4.174) hoists region
                            // body ops to top level, leaving IFs with empty regions
                            // yielding outer-scope refs only — exactly the shape this
                            // arm handles.
                            //
                            // §0.4.173 — KNOWN LEAK: when the lift pass DOESN'T fire (the
                            // IF has body ops with unsafe op kinds like DIV, SQRT, LOG;
                            // or the function has a non-IF region-bearing op that bails
                            // out the whole function), regions stay non-empty and we
                            // fall back to the pre-§0.4.175 "don't clone" path. The leak
                            // persists for those shapes; the next firing can either widen
                            // [SAFE_LIFT_OPS] or extend [irIfOp] to lower IF-with-body-ops
                            // as `IrBlock` branches.
                            val regionsAllEmpty = n.regions.all { region ->
                                region.blocks.size == 1 && region.blocks.single().body.isEmpty()
                            }
                            if (regionsAllEmpty) {
                                val predClone = resolveCloneOperand(n.operands[0], nodeMap, n.id)
                                val clonedRegions = n.regions.map { region ->
                                    val block = region.blocks.single()
                                    region {
                                        val terms = block.terminator.map {
                                            resolveCloneOperand(it, nodeMap, n.id)
                                        }
                                        yields(*terms.toTypedArray())
                                    }
                                }
                                if (n.types.size == 1) {
                                    op(
                                        OpKind.IF, listOf(predClone), n.type, n.attrs,
                                        n.sharding, clonedRegions,
                                    )
                                } else {
                                    opMulti(
                                        OpKind.IF, listOf(predClone), n.types, n.attrs,
                                        n.sharding, clonedRegions,
                                    )
                                }
                            } else {
                                n
                            }
                        } else {
                            // §0.4.179 — Phase 5c: dispatch on multi-result vs single-result.
                            // Multi-result non-IF ops (today: only COARSENED) must be cloned
                            // via opMulti() to preserve all types[]; using op() with
                            // type=type[0] loses the higher-indexed result types and trips
                            // `mappedSource.result(idx)` lookups downstream.
                            val clonedOperands = n.operands.map { resolveCloneOperand(it, nodeMap, n.id) }
                            if (n.isMultiResult) {
                                opMulti(
                                    kind = n.op,
                                    operands = clonedOperands,
                                    types = n.types,
                                    attrs = n.attrs,
                                    sharding = n.sharding,
                                )
                            } else {
                                op(
                                    kind = n.op,
                                    operands = clonedOperands,
                                    type = n.type,
                                    attrs = n.attrs,
                                )
                            }
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
            val gradAccum = HashMap<Pair<Int, Int>, DxirNode>()
            gradAccum[ret.gradKey()] = seed

            // --- 3. Reverse walk: emit per-op adjoint contributions. ---
            for (n in primal.body.asReversed()) {
                if (n !is DxirOp) continue

                // §0.4.155 — IF dispatch: collect per-(live-index) upstreams. For a
                // single-result IF, the only live index is 0; for a multi-result IF,
                // every index in `ifLiveIndices[n.id]` may have its own contribution.
                // Indices with no accumulated contribution are skipped (they're dead
                // from the gradient's perspective).
                if (n.op == OpKind.IF) {
                    val liveIndices: Set<Int> = if (n.isMultiResult) {
                        ifLiveIndices[n.id] ?: emptySet()
                    } else {
                        setOf(0)
                    }
                    val upstreams: Map<Int, DxirNode> = liveIndices.mapNotNull { idx ->
                        gradAccum[n.id to idx]?.let { idx to it }
                    }.toMap()
                    if (upstreams.isEmpty()) continue
                    handleIfAdjoint(n, upstreams, gradAccum, nodeMap, primalById, this)
                    continue
                }

                // §0.4.32 — COARSENED adjoint: splice the stored gradient_body into the
                // outer gradient function. Matches [handleIfAdjoint]'s structural role as
                // a "special case bypassing the VjpRule contract" (the COARSENED op's
                // gradient is precomputed + stored in attrs rather than derived from a
                // static rule). See the doc-comment on [handleCoarsenedAdjoint].
                //
                // §0.4.179 — Phase 5c: collect per-result-index upstreams (mirror IF
                // dispatch). Single-result COARSENED routes through index 0 (preserves
                // pre-§0.4.179 behavior bit-exactly). Multi-result COARSENED collects
                // every index that has accumulated contribution; indices with no
                // downstream consumer are absent from `upstreams` and seeded with
                // typed-zero inside [handleCoarsenedAdjoint].
                if (n.op == OpKind.COARSENED) {
                    val upstreams: Map<Int, DxirNode> = n.types.indices.mapNotNull { idx ->
                        gradAccum[n.id to idx]?.let { idx to it }
                    }.toMap()
                    if (upstreams.isEmpty()) continue
                    handleCoarsenedAdjoint(n, upstreams, gradAccum, nodeMap, primalById, this)
                    continue
                }

                // §0.4.154 — non-IF body ops are single-result; their upstream lives
                // at index 0.
                val upstream = gradAccum[n.id to 0] ?: continue

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
                        // §0.4.155 — preserve DxirOpResult wrapping in phantom operands
                        // so two operands that reference distinct result indices of the
                        // same multi-result source remain distinct objects (avoiding the
                        // associate-map / indexOf collapse that drops contributions to
                        // result(k) for k>0). Mirrors `resolveCloneOperand`'s logic, but
                        // with an `?: it` fallback for ids not in nodeMap (the phantom
                        // contract — fallback operands are never dereferenced).
                        operands = n.operands.map { o ->
                            if (o is DxirOpResult) {
                                val mappedSource = nodeMap[o.source.id] ?: o.source
                                if (mappedSource is DxirOp) mappedSource.result(o.index) else o
                            } else {
                                nodeMap[o.id] ?: o
                            }
                        },
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
                    val accumKey = primalOperand.gradKey()
                    val existing = gradAccum[accumKey]
                    gradAccum[accumKey] = when {
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
            //        §0.4.419 — Phase E1c-pre: integer TENSOR params emit their
            //        structural zero as ZEROS_LIKE on the cloned param itself, so
            //        the zero NAMES its param (an anonymous const's sentinel-dimmed
            //        type cannot — the reason `grad {}` synthesis was restricted to
            //        one integer param per lambda from §0.4.400 until now). Scalar
            //        integer params keep the plain const: a scalar carries no
            //        sentinel extents, so there is nothing to address.
            val gradReturns = primal.params.dropLast(inputOnlyTrailingParams).map { p ->
                if (isIntegerDtype(p.type.dtype)) {
                    if (p.type.isScalar) {
                        const(zeroValueFor(p.type.dtype), p.type)
                    } else {
                        op(OpKind.ZEROS_LIKE, listOf(nodeMap.getValue(p.id)), p.type)
                    }
                } else {
                    gradAccum[p.gradKey()] ?: const(zeroValueFor(p.type.dtype), p.type)
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
     * Canonicalize a node REFERENCE through an id-keyed map while
     * preserving a [DxirOpResult]'s index: `byId[ref.id]` maps a `%op#k`
     * reference to the rebuilt SOURCE op, and returning that op directly
     * collapses the reference to result 0 (for example, `MUL(seed, %c#1)`'s
     * surviving operand would fold to `%c` and the gradient would read result
     * 0's value). Every id-keyed
     * canonicalization in the CSE / const-fold plumbing routes through here.
     */
    private fun canonicalRef(ref: DxirNode, byId: Map<Int, DxirNode>): DxirNode {
        val mapped = byId[ref.id] ?: return ref
        return if (ref is DxirOpResult && mapped is DxirOp && mapped.isMultiResult) {
            // Same source instance → keep the original reference (preserves the
            // mutated-flag / idempotency contract of the callers' `!==` checks).
            if (mapped === ref.source) ref else mapped.result(ref.index)
        } else {
            mapped
        }
    }

    /**
     * Common sub-expression elimination on the gradient body. Adjoint
     * rules often emit structurally-identical ops: [MulRule] on `x*x` produces
     * two identical `MUL(upstream, x)` contributions (one per operand slot),
     * [SqrtRule] emits a fresh `SQRT(op.operands[0])` whose input already has a
     * cloned primal `SQRT` in the body. CSE walks the body in declaration order,
     * hashes each scalar-result op by `(op.op, operand canonical ids, attrs)`,
     * and replaces duplicates with references to the canonical op. Duplicates
     * become unreachable and are dropped by the following [dropUnreachableBody]
     * step.
     *
     * Recurses into IF region bodies. Each branch's block
     * gets its own scope-local CSE pass that inherits the outer canonical maps
     * (so inner ops can dedup against outer-scope canonical entries) but doesn't
     * leak its own registrations back. Block args are added to the inner scope
     * only. See [cseRegionBearingOp] for how WHILE and COARSENED bodies are handled.
     *
     * Scope: single-result ops only. Multi-result ops are kept verbatim — their
     * structural equivalence is subtler and not load-bearing for any benchmark
     * today. [SCATTER_ADD]'s `in_place` attr is preserved (two SCATTER_ADDs with
     * the same operands but different in-place tags would be an SSA use-graph
     * inconsistency; we shouldn't see that in practice).
     */
    internal fun applyCSE(fn: DxirFunction): DxirFunction {
        val byId = HashMap<Int, DxirNode>()
        for (p in fn.params) byId[p.id] = p
        // CSE consts too — adjoint emission produces many duplicate const literals
        // (e.g., const 2.0f in multiple SqrtRule adjoints). Signature = (value, type).
        val const2canon = HashMap<Pair<Any, DxirType>, DxirConst>()
        val sig2canon = HashMap<CseSig, DxirNode>()
        val newBody = mutableListOf<DxirNode>()
        var mutated = false
        for (n in fn.body) {
            val (newNode, nodeMutated) = cseNode(n, byId, sig2canon, const2canon)
            if (newNode != null) newBody += newNode
            if (nodeMutated) mutated = true
        }
        if (!mutated) return fn
        val newReturns = fn.returns.map { canonicalRef(it, byId) }
        return DxirFunction(fn.name, fn.params, newBody, newReturns, fn.meshes)
    }

    /**
     * Process a single node under [byId] / [sig2canon] / [const2canon]. Returns the
     * node to add to the surrounding body (or null if the node was deduplicated and
     * should be dropped) and a `mutated` flag indicating whether the input was
     * changed in any way (deduplicated, rebuilt with canonical operand refs, or
     * had a region rewritten).
     *
     * The maps are mutated by this method: deduplicated entries are recorded in
     * [byId]; canonical entries are registered in [sig2canon] / [const2canon].
     * Callers that need to scope these maps to a sub-region should pass copies.
     */
    /**
     * CSE signature. `types` is part of the key: two ops with
     * identical (kind, operands, attrs) can still differ in RESULT TYPE —
     * BROADCAST is the archetype (the same scalar seed splat to two different
     * shapes carries `broadcast_dimensions=[]` both times). Without the types
     * in the key, MeanRule's runtime-N `ones` broadcast would deduplicate onto
     * the rank-1 upstream splat and produce a wrong-shape gradient.
     */
    private data class CseSig(
        val op: OpKind,
        // §0.4.430 — (id, resultIndex) pairs, not bare ids: `%c#0` and `%c#1`
        // share an id, and a bare-id key merged `MUL(seed, %c#0)` with
        // `MUL(seed, %c#1)` — two different values off the same multi-result
        // source (the same result-identity discipline as the §0.4.366 types
        // component).
        val operandIds: List<Pair<Int, Int>>,
        val attrs: Map<String, Any>,
        val types: List<DxirType>,
    )

    private fun operandKey(n: DxirNode): Pair<Int, Int> =
        n.id to ((n as? DxirOpResult)?.index ?: 0)

    private fun cseNode(
        n: DxirNode,
        byId: HashMap<Int, DxirNode>,
        sig2canon: HashMap<CseSig, DxirNode>,
        const2canon: HashMap<Pair<Any, DxirType>, DxirConst>,
    ): Pair<DxirNode?, Boolean> {
        return when (n) {
            is DxirConst -> {
                val key = n.value to n.type
                val existing = const2canon[key]
                if (existing != null) {
                    byId[n.id] = existing
                    null to true
                } else {
                    const2canon[key] = n
                    byId[n.id] = n
                    n to false
                }
            }
            is DxirOp -> {
                if (n.hasRegions || n.op == OpKind.COARSENED) {
                    // §0.4.118 — IF regions get internal CSE. §0.4.119 — COARSENED
                    // has empty `regions` but stores `primal_body` / `gradient_body`
                    // as DxirFunctions in attrs; recurse into those. §0.4.129 —
                    // WHILE regions also get internal CSE (multi-result region-bearing
                    // ops route here ahead of the multi-result short-circuit so their
                    // cond / body bodies still benefit from dedup).
                    cseRegionBearingOp(n, byId, sig2canon, const2canon)
                } else if (n.isMultiResult) {
                    byId[n.id] = n
                    n to false
                } else {
                    val canonicalOperands = n.operands.map { canonicalRef(it, byId) }
                    val opIds = canonicalOperands.map { operandKey(it) }
                    val sig = CseSig(n.op, opIds, n.attrs, n.types)
                    val existing = sig2canon[sig]
                    if (existing != null) {
                        byId[n.id] = existing
                        null to true
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
                        rebuilt to (rebuilt !== n)
                    }
                }
            }
            else -> {
                byId[n.id] = n
                n to false
            }
        }
    }

    /**
     * Region-bearing op CSE. For [OpKind.IF], recurse into each branch's
     * region body using a scoped copy of the canonical maps. The same
     * recursion fires for [OpKind.WHILE]: the LoopInvariant break-loop
     * rewrite produces nested WHILEs inside IF arms, so WHILEs can survive
     * coarsening.
     * [cseRegion] processes each region's block with its own scoped copy of the
     * canonical maps, so the cond / body regions of a WHILE never cross-pollute
     * (their block args have distinct ids, and inner registrations don't leak
     * back to outer scope). The op's IMMEDIATE operands are still canonicalized
     * in all cases.
     *
     * For [OpKind.COARSENED], the regions list is empty (the op stores
     * its `primal_body` and `gradient_body` in attrs as full [DxirFunction]s, not
     * as regions). Run [applyCSE] on each nested function so inner redundancy
     * still gets deduplicated. `applyCSE` is idempotent (returns the same
     * function reference when nothing changes), so re-running on already-CSE'd
     * `gradient_body` produced by [DxirReverseTransform.apply] is a no-op.
     */
    private fun cseRegionBearingOp(
        n: DxirOp,
        outerById: HashMap<Int, DxirNode>,
        outerSig2canon: HashMap<CseSig, DxirNode>,
        outerConst2canon: HashMap<Pair<Any, DxirType>, DxirConst>,
    ): Pair<DxirNode?, Boolean> {
        val canonicalOperands = n.operands.map { canonicalRef(it, outerById) }
        var mutated = canonicalOperands.zip(n.operands).any { (a, b) -> a !== b }

        val newRegions = if (n.op == OpKind.IF || n.op == OpKind.WHILE) {
            n.regions.map { region ->
                val (newRegion, regionMutated) = cseRegion(
                    region, outerById, outerSig2canon, outerConst2canon,
                )
                if (regionMutated) mutated = true
                newRegion
            }
        } else {
            n.regions
        }

        val newAttrs: Map<String, Any> = if (n.op == OpKind.COARSENED) {
            val rebuiltAttrs = HashMap(n.attrs)
            var attrsMutated = false
            (n.attrs["primal_body"] as? DxirFunction)?.let { primalBody ->
                val csePrimal = applyCSE(primalBody)
                if (csePrimal !== primalBody) {
                    rebuiltAttrs["primal_body"] = csePrimal
                    attrsMutated = true
                }
            }
            (n.attrs["gradient_body"] as? DxirFunction)?.let { gradientBody ->
                val cseGradient = applyCSE(gradientBody)
                if (cseGradient !== gradientBody) {
                    rebuiltAttrs["gradient_body"] = cseGradient
                    attrsMutated = true
                }
            }
            if (attrsMutated) {
                mutated = true
                rebuiltAttrs
            } else {
                n.attrs
            }
        } else {
            n.attrs
        }

        val rebuilt = if (mutated) {
            DxirOp(
                id = n.id,
                op = n.op,
                operands = canonicalOperands,
                attrs = newAttrs,
                types = n.types,
                sharding = n.sharding,
                regions = newRegions,
            )
        } else {
            n
        }
        outerById[n.id] = rebuilt
        return rebuilt to mutated
    }

    /**
     * Recursively CSE a region's blocks under a scope-local copy of the
     * canonical maps. Block args are added to the inner scope so block-arg-rooted
     * ops can be deduplicated within the block. Inner registrations don't leak
     * back to the outer scope (different control-flow scope = different operand
     * visibility).
     */
    private fun cseRegion(
        region: DxirRegion,
        outerById: Map<Int, DxirNode>,
        outerSig2canon: HashMap<CseSig, DxirNode>,
        outerConst2canon: HashMap<Pair<Any, DxirType>, DxirConst>,
    ): Pair<DxirRegion, Boolean> {
        var anyMutated = false
        val newBlocks = region.blocks.map { block ->
            val innerById = HashMap<Int, DxirNode>(outerById)
            for (a in block.args) innerById[a.id] = a
            val innerSig2canon = HashMap(outerSig2canon)
            val innerConst2canon = HashMap(outerConst2canon)
            val newBody = mutableListOf<DxirNode>()
            var blockMutated = false
            for (n in block.body) {
                val (newNode, nodeMutated) = cseNode(n, innerById, innerSig2canon, innerConst2canon)
                if (newNode != null) newBody += newNode
                if (nodeMutated) blockMutated = true
            }
            val newTerminator = block.terminator.map { term ->
                val mapped = innerById[term.id] ?: return@map term
                // §0.4.130 — DxirOpResult terminators must route through `.result(k)`
                // when the source maps to a multi-result DxirOp. The pre-§0.4.130
                // path returned the source op directly, collapsing all multi-result
                // refs to index 0; that broke type validation when the terminator
                // referenced a non-zero index (e.g., an MR WHILE inside an IF arm
                // whose terminator yielded `whileOp.result(1)`). Mirrors the same
                // fix in [PhiCalculus.cloneRegion] from §0.4.128.
                if (term !is DxirOpResult) return@map mapped
                when {
                    mapped is DxirOp -> mapped.result(term.index)
                    term.index == 0 -> mapped
                    else -> error(
                        "applyCSE: terminator id=${term.id} index=${term.index} on non-Op " +
                            "rebuild (mapped=${mapped::class.simpleName}); only index 0 is " +
                            "tolerated for non-Op rebuilds",
                    )
                }
            }
            if (newTerminator.zip(block.terminator).any { (a, b) -> a !== b }) blockMutated = true
            if (blockMutated) anyMutated = true
            if (blockMutated) {
                DxirBlock(args = block.args, body = newBody, terminator = newTerminator)
            } else {
                block
            }
        }
        return (if (anyMutated) DxirRegion(newBlocks) else region) to anyMutated
    }

    /**
     * Peephole constant folding on scalar numeric ops. Handles:
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

        // §0.4.500 — the f64 twin. Every fold below used to build its replacement
        // constant from the FLOAT projection, whatever the node's dtype was, so an
        // f64 scalar body produced a `DxirConst` whose type said F64 and whose value
        // was a `java.lang.Float`. `DxirToIrSynthesis`'s `v as Double` then threw a
        // raw ClassCastException out of the IR generation extension: the whole
        // compilation died, with no Tlaloc diagnostic, on a program as ordinary as
        // `grad { x: Double -> x * 1.5 }`. Two bugs in one, and the second is the
        // worse: it neither worked nor refused by name. See `foldedConst` below.
        fun asDoubleConst(n: DxirNode): Double? {
            val resolved = byId[n.id] ?: return null
            if (resolved !is DxirConst) return null
            return (resolved.value as? Number)?.toDouble()
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
                        val canonicalOperands = n.operands.map { canonicalRef(it, byId) }
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
                    // §0.4.500 — a folded constant carries a value of the NODE's
                    // dtype, and the arithmetic happens at that width. The f32 arm
                    // is the pre-§0.4.500 expression, unchanged, so nothing about
                    // single precision moves; the f64 arm exists at all because
                    // folding an f64 product through Float and handing the result
                    // back as "f64" would be a single-precision answer wearing a
                    // double-precision type.
                    val wide = n.type.dtype == io.tlaloc.core.F64
                    fun foldedConst(f32: () -> Float, f64: () -> Double): DxirConst =
                        DxirConst(n.id, if (wide) f64() else f32(), n.type)

                    // §0.4.508 — the IDENTITY predicates must ask at the node's own width.
                    // §0.4.500 left them on the `Float` projection and argued the projection
                    // was lossless "because they compare against exact small integers". The
                    // comparison target is exact; the PROJECTED OPERAND is not. `1.0 + 1e-10`
                    // is a double that `toFloat()` rounds to exactly `1.0f`, so `a == 1.0f`
                    // fired and MUL collapsed to its other operand — turning d/dx x*1.0000000001
                    // into 1.0. Likewise `1e-50` projects to `0.0f`, so MUL-by-it folded the
                    // whole product to zero. Both are silent single-precision answers inside an
                    // f64 program, which is the same class of defect the dtype fix above closed,
                    // one level up: there the VALUE was narrow, here the DECISION is.
                    fun isExactly(operand: DxirNode, target: Double): Boolean =
                        if (wide) {
                            asDoubleConst(operand) == target
                        } else {
                            asFloatConst(operand) == target.toFloat()
                        }

                    // Try to fold.
                    val folded: DxirNode? = when (n.op) {
                        OpKind.MUL -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null -> foldedConst(
                                    { a * b },
                                    { asDoubleConst(n.operands[0])!! * asDoubleConst(n.operands[1])!! },
                                )
                                isExactly(n.operands[0], 1.0) -> canonicalRef(n.operands[1], byId)
                                isExactly(n.operands[1], 1.0) -> canonicalRef(n.operands[0], byId)
                                isExactly(n.operands[0], 0.0) || isExactly(n.operands[1], 0.0) ->
                                    foldedConst({ 0.0f }, { 0.0 })
                                else -> null
                            }
                        }
                        OpKind.ADD -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null -> foldedConst(
                                    { a + b },
                                    { asDoubleConst(n.operands[0])!! + asDoubleConst(n.operands[1])!! },
                                )
                                isExactly(n.operands[0], 0.0) -> canonicalRef(n.operands[1], byId)
                                isExactly(n.operands[1], 0.0) -> canonicalRef(n.operands[0], byId)
                                else -> null
                            }
                        }
                        OpKind.SUB -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null -> foldedConst(
                                    { a - b },
                                    { asDoubleConst(n.operands[0])!! - asDoubleConst(n.operands[1])!! },
                                )
                                isExactly(n.operands[1], 0.0) -> canonicalRef(n.operands[0], byId)
                                else -> null
                            }
                        }
                        OpKind.DIV -> {
                            val a = asFloatConst(n.operands[0])
                            val b = asFloatConst(n.operands[1])
                            when {
                                a != null && b != null && !isExactly(n.operands[1], 0.0) -> foldedConst(
                                    { a / b },
                                    { asDoubleConst(n.operands[0])!! / asDoubleConst(n.operands[1])!! },
                                )
                                isExactly(n.operands[1], 1.0) -> canonicalRef(n.operands[0], byId)
                                else -> null
                            }
                        }
                        OpKind.NEG -> {
                            val a = asFloatConst(n.operands[0])
                            if (a != null) {
                                foldedConst({ -a }, { -asDoubleConst(n.operands[0])!! })
                            } else {
                                null
                            }
                        }
                        else -> null
                    }
                    if (folded != null) {
                        emit(folded, n)
                    } else {
                        // Always rebuild with canonical operand references (see applyCSE
                        // for the "stale reference" reasoning).
                        val canonicalOperands = n.operands.map { canonicalRef(it, byId) }
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
        val newReturns = fn.returns.map { canonicalRef(it, byId) }
        return DxirFunction(fn.name, fn.params, newBody, newReturns, fn.meshes)
    }

    /**
     * Simple DCE over a function's body. Marks reachable nodes by walking
     * from [DxirFunction.returns] through operand + nested-region references,
     * transitively. Unreferenced top-level body ops are filtered out. Params are
     * always preserved (so the declared signature survives even if the gradient
     * doesn't dereference a param, which is common — e.g., the idx operand of a
     * GATHER whose adjoint is zero). Nested region bodies aren't pruned.
     *
     * Primary motivation: [GatherRule] emits `SCATTER_ADD(zero_bcast, idx, upstream)`;
     * the gradAccum fusion in [PhiCalculus] rewrites subsequent SCATTER_ADDs to thread
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
     * Tag each `OpKind.SCATTER_ADD` whose `operand[0]` (the accumulator
     * base) has exactly one use in the function. The single-use invariant makes
     * destructive in-place mutation safe: the cloned buffer isn't aliased anywhere
     * else, so `base[idx] += value` can write through to `base`'s storage without
     * corrupting a shared view.
     *
     * GatherRule's emission pattern (after DCE) is exactly this shape — a linear
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
    /**
     * Find the unique result index of [ifOp] that's referenced downstream
     * in [fn]'s body or returns. Returns null when zero or multiple indices are
     * referenced (zero = dead op, left to DCE; multiple = would need per-index
     * gradAccum, which is not supported). Used by the multi-
     * result-IF AD entry-point validation in [apply].
     *
     * Mirrors the structural shape of [PhiCalculus.findReferencedCarried] but
     * scoped to a single op (rather than a WHILE's all-result indices) and
     * returning a unique-or-null result.
     */
    /**
     * Operand resolution for the body-cloning step that preserves
     * [DxirOpResult] wrapping. Without this, a body op whose operand is a
     * `DxirOpResult` (e.g., `ADD(ifop.result(0), ifop.result(1))`) would have
     * both operands collapse to the same `nodeMap[ifop.id]` reference under
     * naive id-only lookup, conflating distinct result indices and silently
     * dropping gradient contributions to the higher-indexed slot. This helper
     * mirrors the pattern established in `PhiCalculus.cloneNode`'s `DxirOpResult`
     * arm: resolve through `nodeMap`, and if the source mapped to a `DxirOp`
     * (the common case for non-cloned multi-result primal IFs which map to
     * themselves), re-wrap with `.result(it.index)`.
     */
    private fun resolveCloneOperand(
        operand: DxirNode,
        nodeMap: Map<Int, DxirNode>,
        owningOpId: Int,
    ): DxirNode {
        if (operand is DxirOpResult) {
            val mappedSource = nodeMap[operand.source.id]
                ?: error(
                    "primal body op $owningOpId references unknown DxirOpResult source " +
                        "id=${operand.source.id}",
                )
            return when (mappedSource) {
                is DxirOp -> mappedSource.result(operand.index)
                else -> if (operand.index == 0) mappedSource
                else error(
                    "primal body op $owningOpId: DxirOpResult index=${operand.index} on " +
                        "non-Op clone (mapped=${mappedSource::class.simpleName}); only " +
                        "index 0 is tolerated for non-Op clones",
                )
            }
        }
        return nodeMap[operand.id]
            ?: error("primal body op $owningOpId references unknown id ${operand.id}")
    }

    private fun findIfLiveResultIndices(fn: DxirFunction, ifOp: DxirOp): Set<Int> {
        val ifId = ifOp.id
        val referenced = HashSet<Int>()
        fun checkRef(node: DxirNode) {
            if (node is DxirOpResult && node.source.id == ifId) {
                referenced += node.index
            } else if (node is DxirOp && node.id == ifId) {
                // Direct ref to a multi-result op implicitly targets index 0.
                referenced += 0
            }
        }
        fun walk(nodes: List<DxirNode>) {
            for (n in nodes) {
                if (n !is DxirOp) continue
                if (n === ifOp) continue
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
     * Block-local mirror of [findIfLiveResultIndex]. Walks the given
     * [block]'s body + terminator looking for downstream references to [ifOp]
     * (via [DxirOpResult] or direct [DxirOp] ref). Returns the unique result
     * index if exactly one is referenced, else null.
     *
     * Used by [walkBranchReverse] when accepting a nested multi-result IF as
     * a branch-body op: the "downstream scope" relative to the inner IF is the
     * outer branch's block, NOT the whole function. Walking the function would
     * also pick up references inside the inner IF's own regions (which are
     * region-internal to the inner IF, not downstream uses). Walking just the
     * outer block is the correct scope and matches the top-level
     * [findIfLiveResultIndex] shape exactly.
     */
    private fun findIfLiveResultIndicesInBlock(
        block: io.tlaloc.ir.DxirBlock,
        ifOp: DxirOp,
    ): Set<Int> {
        val ifId = ifOp.id
        val referenced = HashSet<Int>()
        fun checkRef(node: DxirNode) {
            if (node is DxirOpResult && node.source.id == ifId) {
                referenced += node.index
            } else if (node is DxirOp && node.id == ifId) {
                referenced += 0
            }
        }
        fun walk(nodes: List<DxirNode>) {
            for (n in nodes) {
                if (n !is DxirOp) continue
                if (n === ifOp) continue
                for (o in n.operands) checkRef(o)
                for (r in n.regions) for (b in r.blocks) {
                    walk(b.body)
                    for (term in b.terminator) checkRef(term)
                }
            }
        }
        walk(block.body)
        for (term in block.terminator) checkRef(term)
        return referenced
    }

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
                // §0.4.415 — Phase B5 (customVjp): a USER gradient_body's returns
                // are wrapped in CHECK_SHAPE_LIKE(contribution, operandClone) by
                // [handleCoarsenedAdjoint] whenever shapes aren't statically
                // decidable, so EVERY operand may be dereferenced as a shape
                // template regardless of whether the user's vjpFn reads it —
                // enqueue them all, or the wrap would reference an un-cloned
                // primal node (a leaked id in the gradient body's SSA).
                if (n.attrs["user_gradient"] == true) {
                    for (operand in n.operands) enqueue(operand.id)
                }
                continue
            }
            val rule = VjpRegistry[n.op] ?: continue
            // Phase A5c-2 — per-node, not the static property: SumRule/MeanRule only
            // dereference operand 0 (as their scalar seed's shape template) when the
            // target shape carries a sentinel, so a concrete-dims SUM does not drag
            // its summed operand into the gradient body.
            for (i in rule.readsPrimalOperands(n)) {
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
     * Handle the IF op as a special case in [DxirReverseTransform.apply]'s
     * reverse walk. Per the φ-node rule (`d/dx(φ(a, b)) = φ(da/dx, db/dx)`), the gradient of an
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
     * Limitations:
     *  - Multi-result IF: single-live-index only (top-level and nested).
     *  - Nested control flow in branch bodies: single-result IF supported via
     *    recursive dispatch; nested multi-result IF supported via
     *    block-local live-index analysis; WHILE errors.
     *  - Branch bodies must not contain non-IF multi-result ops.
     *  - `usedByAdjoint` analysis isn't extended into branches — all branch body ops
     *    are unconditionally cloned into the gradient body.
     */
    private fun handleIfAdjoint(
        ifNode: DxirOp,
        upstreams: Map<Int, DxirNode>,
        outerGradAccum: MutableMap<Pair<Int, Int>, DxirNode>,
        outerNodeMap: Map<Int, DxirNode>,
        primalById: Map<Int, DxirNode>,
        builder: DxirBuilder,
    ) {
        require(ifNode.op == OpKind.IF) { "handleIfAdjoint: not an IF (got ${ifNode.op})" }
        require(upstreams.isNotEmpty()) {
            "handleIfAdjoint: empty upstream map for IF id=${ifNode.id}; the caller must " +
                "supply at least one (live-index → upstream) pair"
        }
        require(upstreams.keys.all { it in ifNode.types.indices }) {
            "handleIfAdjoint: upstream live index out of bounds for ${ifNode.types.size}-" +
                "result IF id=${ifNode.id}; got keys=${upstreams.keys}"
        }
        val predClone = outerNodeMap[ifNode.operands[0].id]
            ?: error("handleIfAdjoint: predicate id=${ifNode.operands[0].id} not in nodeMap")

        val thenBlock = ifNode.regions[0].blocks.single()
        val elseBlock = ifNode.regions[1].blocks.single()

        // §0.4.155 — Phase 5b: seed each branch's reverse walk at terminator[k] for
        // every k in the live-index set; per-index contributions ADD-merge if two
        // indices yield the same SSA value (terminator[k1].id == terminator[k2].id).
        val thenAdjoints = walkBranchReverse(thenBlock, upstreams, outerNodeMap, primalById, builder)
        val elseAdjoints = walkBranchReverse(elseBlock, upstreams, outerNodeMap, primalById, builder)

        // Combine per-branch adjoints into a synthesized IF for each outer-scope key.
        // LinkedHashSet preserves insertion order so the IF emission order is
        // deterministic (helps test stability). §0.4.154 — keys are now
        // `(id, resultIndex)` so contributions to distinct result indices of the
        // same multi-result outer-scope op flow back through separate IF wrappers.
        val allKeys = LinkedHashSet<Pair<Int, Int>>()
        allKeys.addAll(thenAdjoints.keys)
        allKeys.addAll(elseAdjoints.keys)

        for (key in allKeys) {
            val (id, idx) = key
            // §0.4.140 — for the recursive (nested-IF) call, [outerNodeMap] is the
            // CALLER's branchNodeMap, which contains region-internal ids of the
            // outer branch (e.g., a `MUL(x, x)` defined in the outer-then block and
            // referenced by the inner-else's terminator). Those ids aren't in
            // [primalById] (which is built from `primal.params + primal.body` at the
            // function's top level only), but their gradient contributions still
            // need to accumulate into the caller's gradAccum so the caller's reverse
            // walk can route them through the outer branch's VJP rules. Fall back to
            // [outerNodeMap] for the type lookup; ids that miss BOTH maps are
            // branch-internal to one of the inner branches and skipped (their
            // gradients are local to that branch).
            val primal = primalById[id] ?: outerNodeMap[id] ?: continue
            // Skip constants — they have no gradient surface.
            if (primal is DxirConst) continue
            // §0.4.154 — for multi-result primal ops the gradient type at index `idx`
            // is `types[idx]`, not `type` (which collapses to `types[0]`).
            val primalType = if (primal is DxirOp && primal.isMultiResult) {
                primal.types[idx]
            } else {
                primal.type
            }
            val thenAdj = thenAdjoints[key]
                ?: builder.const(zeroValueFor(primalType.dtype), primalType)
            val elseAdj = elseAdjoints[key]
                ?: builder.const(zeroValueFor(primalType.dtype), primalType)

            val condIf = builder.ifOp(
                cond = predClone,
                types = listOf(primalType),
                thenRegion = builder.region { yields(thenAdj) },
                elseRegion = builder.region { yields(elseAdj) },
            )

            val existing = outerGradAccum[key]
            outerGradAccum[key] = if (existing == null) condIf
            else builder.op(OpKind.ADD, listOf(existing, condIf), primalType)
        }
    }

    /**
     * Handle the [OpKind.COARSENED] op in the reverse walk. Mirrors
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
     * Limitations:
     *  - Single-result COARSENED only (the top-level guard rejects multi-result).
     *  - `gradient_body.body` must be straight-line (no regions, no multi-result).
     *    VJPs produced by the coarsening pipeline satisfy this; hand-
     *    built gradient_bodies with regions would need recursive handling.
     */
    private fun handleCoarsenedAdjoint(
        coarsened: DxirOp,
        upstreams: Map<Int, DxirNode>,
        outerGradAccum: MutableMap<Pair<Int, Int>, DxirNode>,
        outerNodeMap: Map<Int, DxirNode>,
        primalById: Map<Int, DxirNode>,
        builder: DxirBuilder,
    ) {
        require(coarsened.op == OpKind.COARSENED) {
            "handleCoarsenedAdjoint: not a COARSENED op (got ${coarsened.op})"
        }
        require(upstreams.isNotEmpty()) {
            "handleCoarsenedAdjoint: empty upstream map for COARSENED id=${coarsened.id}; " +
                "the caller must supply at least one (result-index → upstream) pair"
        }
        require(upstreams.keys.all { it in coarsened.types.indices }) {
            "handleCoarsenedAdjoint: upstream result-index out of bounds for ${coarsened.types.size}-" +
                "result COARSENED id=${coarsened.id}; got keys=${upstreams.keys}"
        }
        // §0.4.416 — Phase B5 (customJvp): a USER node carrying a tangent_body
        // but NO gradient_body is forward-only. REFUSE loudly by name — the
        // exact mirror of DxirForwardTransform's customVjp refusal: silently
        // auto-differentiating primal_body here would make grad {} disagree
        // with the user's deliberately-supplied forward tangent (the §0.4.392
        // no-silent-fork principle, both directions).
        if (coarsened.attrs["gradient_body"] == null && coarsened.attrs["user_gradient"] == true) {
            error(
                "handleCoarsenedAdjoint: COARSENED id=${coarsened.id} carries a USER-supplied " +
                    "tangent but no gradient (user_gradient attr — a customJvp call-form): " +
                    "reverse mode would auto-differentiate primal_body and silently disagree " +
                    "with the user's forward tangent. Supply a vjpFn (customVjpJvp) or use " +
                    "forward mode (jvp {})",
            )
        }
        val gradBody = coarsened.attrs["gradient_body"] as? io.tlaloc.ir.DxirFunction
            ?: error("handleCoarsenedAdjoint: COARSENED op id=${coarsened.id} missing gradient_body attr")
        // §0.4.179 — Phase 5c. Single-result preserved as the K=1 case (1 + N params).
        // Multi-result uses K + N params: K upstreams (positional, one per result type)
        // + N primal operands (positional, aligned with coarsened.operands).
        val k = coarsened.types.size
        require(gradBody.params.size == k + coarsened.operands.size) {
            "handleCoarsenedAdjoint: gradient_body param count ${gradBody.params.size} ≠ " +
                "K + N (= $k + ${coarsened.operands.size}) (upstreams + primal operands)"
        }
        require(gradBody.returns.size == coarsened.operands.size) {
            "handleCoarsenedAdjoint: gradient_body return count ${gradBody.returns.size} ≠ " +
                "operand count ${coarsened.operands.size}"
        }

        // Step 2: seed gradient-body param ids → outer nodes.
        // §0.4.179 — Phase 5c: per-result-index upstream seeding. Indices NOT in
        // [upstreams] (no consumer accumulated into that result) get a typed-zero
        // const so the gradient_body's reverse walk sees a well-formed input even
        // when only a subset of results contribute downstream gradient.
        val gradNodeMap = HashMap<Int, DxirNode>()
        for (i in 0 until k) {
            val u = upstreams[i] ?: builder.const(zeroValueFor(coarsened.types[i].dtype), coarsened.types[i])
            gradNodeMap[gradBody.params[i].id] = u
        }
        for (i in coarsened.operands.indices) {
            val operandClone = outerNodeMap[coarsened.operands[i].id]
                ?: error(
                    "handleCoarsenedAdjoint: primal operand id=${coarsened.operands[i].id} " +
                        "missing from outerNodeMap (expected cloned or primal verbatim)",
                )
            gradNodeMap[gradBody.params[k + i].id] = operandClone
        }

        // Step 3: clone gradient_body.body into the outer gradient builder. §0.4.120 —
        // recursive cloning supports IF inside the gradient body via [cloneGradNode];
        // multi-result and other region-bearing ops still error.
        for (n in gradBody.body) {
            val cloned = cloneGradNode(n, gradNodeMap, builder)
            gradNodeMap[n.id] = cloned
        }

        // Step 4: map each gradient_body return → primal operand contribution.
        // §0.4.415 — Phase B5 (customVjp): a USER-supplied gradient_body
        // (`user_gradient = true`) is the user's assertion, not a machine
        // derivation, so each of its returns must honour the VJP shape contract
        // — d_operand shaped like its operand — before accumulation (design doc
        // §4.1; the `conv2dDataAdjoint` template-assert precedent). Statically
        // concrete-and-equal shapes need nothing; concrete-and-UNEQUAL shapes
        // fail the transform right here (loud, and check-time visible through
        // the intrinsic checker's probe); anything sentinel-bearing — which is
        // every `grad {}` body — wraps in a CHECK_SHAPE_LIKE whose interpreter
        // arm / host twin (`checkShapeLike`) asserts the RUNTIME dims at
        // execution. Machine-built gradient bodies stay byte-identical.
        val userGradient = coarsened.attrs["user_gradient"] == true
        for (i in gradBody.returns.indices) {
            var contribution = gradNodeMap[gradBody.returns[i].id]
                ?: error(
                    "handleCoarsenedAdjoint: gradient_body.returns[$i] id=" +
                        "${gradBody.returns[i].id} missing from gradNodeMap",
                )
            val primalOperand = coarsened.operands[i]
            // Constants have no gradient surface — skip.
            if (primalById[primalOperand.id] is DxirConst) continue
            if (userGradient) {
                val cDims = contribution.type.dims
                val oDims = primalOperand.type.dims
                val bothConcrete = cDims.all { it > 0 } && oDims.all { it > 0 }
                if (bothConcrete && cDims != oDims) {
                    error(
                        "handleCoarsenedAdjoint: customVjp gradient_body return $i has shape " +
                            "$cDims but its operand has shape $oDims — the user vjpFn violates " +
                            "the VJP shape contract (each d_operand must match its operand's shape)",
                    )
                }
                if (!bothConcrete || contribution.type != primalOperand.type) {
                    val operandClone = gradNodeMap[gradBody.params[k + i].id]
                        ?: error(
                            "handleCoarsenedAdjoint: operand clone for CHECK_SHAPE_LIKE " +
                                "missing (index $i)",
                        )
                    contribution = builder.op(
                        OpKind.CHECK_SHAPE_LIKE,
                        listOf(contribution, operandClone),
                        primalOperand.type,
                    )
                }
            }
            val operandKey = primalOperand.gradKey()
            val existing = outerGradAccum[operandKey]
            outerGradAccum[operandKey] = if (existing == null) contribution
            else builder.op(OpKind.ADD, listOf(existing, contribution), contribution.type)
        }
    }

    /**
     * Clone a single gradient_body node into [builder]'s scope, resolving
     * operand refs through [gradNodeMap]. Handles:
     *
     *  - [DxirConst]: re-emit verbatim (id may differ; gradNodeMap is updated by caller).
     *  - [DxirOp] without regions, single-result: clone operands canonically; emit via
     *    `builder.op` with the same kind/attrs/types.
     *  - [DxirOp] of kind [OpKind.IF], single-result: clone each branch region recursively.
     *    Multi-result IF and other region-bearing kinds error — those cases would
     *    need scope tracking that this single-purpose helper doesn't carry.
     *  - [DxirOpResult]: shouldn't appear in a single-result gradient_body's straight-
     *    line body, but if it does, throw — the caller's contract excludes multi-result
     *    sources.
     */
    private fun cloneGradNode(
        n: DxirNode,
        gradNodeMap: HashMap<Int, DxirNode>,
        builder: DxirBuilder,
    ): DxirNode = when (n) {
        is DxirConst -> builder.const(n.value, n.type, n.sharding)
        is DxirOp -> {
            require(!n.isMultiResult) {
                "handleCoarsenedAdjoint: gradient_body op ${n.op} is multi-result " +
                    "(not supported)"
            }
            if (n.hasRegions) {
                require(n.op == OpKind.IF) {
                    "handleCoarsenedAdjoint: gradient_body op ${n.op} has regions but " +
                        "only IF is supported (no WHILE/COARSENED inside gradient_body)"
                }
                cloneGradIf(n, gradNodeMap, builder)
            } else {
                val clonedOperands = n.operands.map {
                    gradNodeMap[it.id]
                        ?: error(
                            "handleCoarsenedAdjoint: gradient_body op id=${n.id} references " +
                                "unknown id=${it.id} (gradient_body has broken SSA?)",
                        )
                }
                builder.op(n.op, clonedOperands, n.type, n.attrs, n.sharding, emptyList())
            }
        }
        else -> error(
            "handleCoarsenedAdjoint: unsupported gradient_body node " +
                "${n::class.simpleName} (id=${n.id})",
        )
    }

    /**
     * Clone an IF op from the gradient body into [builder]'s scope. Each
     * branch's region is cloned via [cloneGradRegion]. Block args are scope-local;
     * inner-region operand refs to outer-scope ids resolve through the outer
     * [gradNodeMap], while inner block-arg ids are added to a per-region copy.
     */
    private fun cloneGradIf(
        n: DxirOp,
        gradNodeMap: HashMap<Int, DxirNode>,
        builder: DxirBuilder,
    ): DxirNode {
        val predClone = gradNodeMap[n.operands[0].id]
            ?: error(
                "handleCoarsenedAdjoint: IF predicate id=${n.operands[0].id} missing from gradNodeMap",
            )
        require(n.regions.size == 2) {
            "handleCoarsenedAdjoint: IF must have exactly 2 regions (then, else); got ${n.regions.size}"
        }
        return builder.ifOp(
            cond = predClone,
            types = n.types,
            thenRegion = builder.region {
                cloneGradRegion(n.regions[0], gradNodeMap, this)
            },
            elseRegion = builder.region {
                cloneGradRegion(n.regions[1], gradNodeMap, this)
            },
        )
    }

    /**
     * Clone a region's single block into the active region builder. The
     * outer [outerGradNodeMap] is copied so block args added inside don't leak back.
     * Block args live only within the cloned region's scope; their inner-only ids
     * never appear in the outer map.
     */
    private fun cloneGradRegion(
        region: io.tlaloc.ir.DxirRegion,
        outerGradNodeMap: HashMap<Int, DxirNode>,
        regionBuilder: io.tlaloc.ir.DxirRegionBuilder,
    ) {
        require(region.blocks.size == 1) {
            "handleCoarsenedAdjoint: gradient_body region must have exactly 1 block; got ${region.blocks.size}"
        }
        val block = region.blocks.single()
        val innerMap = HashMap(outerGradNodeMap)
        for (a in block.args) {
            val newArg = regionBuilder.arg(a.type, a.sharding)
            innerMap[a.id] = newArg
        }
        for (n in block.body) {
            val cloned = cloneGradBlockNode(n, innerMap, regionBuilder)
            innerMap[n.id] = cloned
        }
        val termNodes = block.terminator.map {
            innerMap[it.id]
                ?: error(
                    "handleCoarsenedAdjoint: gradient_body region terminator references " +
                        "unknown id=${it.id} (block has broken SSA?)",
                )
        }
        regionBuilder.yields(*termNodes.toTypedArray())
    }

    /**
     * Variant of [cloneGradNode] for nodes inside a region. Emits via the
     * region builder rather than the function builder. Supports
     * nested IF inside an IF arm by recursing into [cloneGradIfInRegion]. WHILE and
     * COARSENED inside an arm error with documented messages — those would
     * require coordinating with the loop/closed-form contracts that this surface
     * doesn't carry.
     */
    private fun cloneGradBlockNode(
        n: DxirNode,
        gradNodeMap: HashMap<Int, DxirNode>,
        regionBuilder: io.tlaloc.ir.DxirRegionBuilder,
    ): DxirNode = when (n) {
        is DxirConst -> regionBuilder.const(n.value, n.type, n.sharding)
        is DxirOp -> {
            require(!n.isMultiResult) {
                "handleCoarsenedAdjoint: gradient_body block op ${n.op} is multi-result " +
                    "(not supported)"
            }
            if (n.hasRegions) {
                require(n.op == OpKind.IF) {
                    "handleCoarsenedAdjoint: gradient_body block op ${n.op} has regions but " +
                        "only IF is supported (no WHILE/COARSENED inside gradient_body IF arms)"
                }
                cloneGradIfInRegion(n, gradNodeMap, regionBuilder)
            } else {
                val clonedOperands = n.operands.map {
                    gradNodeMap[it.id]
                        ?: error(
                            "handleCoarsenedAdjoint: gradient_body block op id=${n.id} references " +
                                "unknown id=${it.id}",
                        )
                }
                regionBuilder.op(n.op, clonedOperands, n.type, n.attrs, n.sharding, emptyList())
            }
        }
        else -> error(
            "handleCoarsenedAdjoint: unsupported gradient_body block node " +
                "${n::class.simpleName} (id=${n.id})",
        )
    }

    /**
     * Clone an IF op nested inside another IF's arm. Mirror of [cloneGradIf]
     * but emits via [io.tlaloc.ir.DxirRegionBuilder] instead of [DxirBuilder]. Both
     * builders share `region { ... }` and `ifOp(...)` surfaces, so the structure is
     * identical; only the receiver differs.
     */
    private fun cloneGradIfInRegion(
        n: DxirOp,
        gradNodeMap: HashMap<Int, DxirNode>,
        regionBuilder: io.tlaloc.ir.DxirRegionBuilder,
    ): DxirNode {
        val predClone = gradNodeMap[n.operands[0].id]
            ?: error(
                "handleCoarsenedAdjoint: nested IF predicate id=${n.operands[0].id} missing from gradNodeMap",
            )
        require(n.regions.size == 2) {
            "handleCoarsenedAdjoint: nested IF must have exactly 2 regions; got ${n.regions.size}"
        }
        return regionBuilder.ifOp(
            cond = predClone,
            types = n.types,
            thenRegion = regionBuilder.region {
                cloneGradRegion(n.regions[0], gradNodeMap, this)
            },
            elseRegion = regionBuilder.region {
                cloneGradRegion(n.regions[1], gradNodeMap, this)
            },
        )
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
     * Nested single-result IF in the branch body is supported via a
     * recursive [handleIfAdjoint] dispatch in step 3. Nested
     * multi-result IF with a unique downstream-referenced result index (the
     * top-level rule applied to the inner case) is also supported. Nested
     * WHILE errors: it needs WHILE-aware AD. Multi-live-index multi-result IF
     * errors: it needs a per-index gradAccum, which the top-level path does not
     * have either.
     */
    private fun walkBranchReverse(
        block: io.tlaloc.ir.DxirBlock,
        upstreams: Map<Int, DxirNode>,
        outerNodeMap: Map<Int, DxirNode>,
        primalById: Map<Int, DxirNode>,
        builder: DxirBuilder,
    ): MutableMap<Pair<Int, Int>, DxirNode> {
        // Step 1: clone branch body ops into the outer builder. branchNodeMap starts as
        // a copy of outerNodeMap (so outer-scope refs resolve) and accumulates clones
        // for each branch body op.
        //
        // §0.4.144 — track per-inner-IF live indices for nested multi-result IFs. For
        // single-result inner IFs the entry is unset and step 3 defaults to liveIdx=0.
        // §0.4.155 — Phase 5b: nested-IF live indices are now `Set<Int>` to mirror the
        // top-level path; multi-live-index nested MR IFs flow through the same
        // upstream-map dispatch as the top-level case.
        val branchNodeMap = HashMap<Int, DxirNode>(outerNodeMap)
        val nestedIfLiveIndices = HashMap<Int, Set<Int>>()
        for (n in block.body) {
            when (n) {
                is DxirConst -> {
                    branchNodeMap[n.id] = builder.const(n.value, n.type, n.sharding)
                }
                is DxirOp -> {
                    if (n.hasRegions) {
                        // §0.4.140 — single-result nested IF is allowed; the reverse
                        // walk in step 3 dispatches to [handleIfAdjoint] for it.
                        // §0.4.144 — multi-result nested IF is allowed when exactly
                        // one of its result indices is referenced in the outer
                        // branch's downstream scope. §0.4.155 — Phase 5b: multi-
                        // live-index nested MR IF AD lands; the per-index gradAccum
                        // (§0.4.154) lets nested IFs route per-index upstreams the
                        // same way as top-level. Other region-bearing shapes still
                        // error: WHILE needs WHILE-aware AD (Stage B's PhiCalculus
                        // pass should have coarsened it before SCT).
                        require(n.op == OpKind.IF) {
                            "walkBranchReverse: nested control-flow op ${n.op} in IF branch " +
                                "not yet supported; got op=${n.op}"
                        }
                        if (n.isMultiResult) {
                            val nestedLiveIndices = findIfLiveResultIndicesInBlock(block, n)
                            require(nestedLiveIndices.isNotEmpty()) {
                                "walkBranchReverse: nested multi-result IF id=${n.id} has " +
                                    "zero result indices referenced in the enclosing block — " +
                                    "the IF op is dead and should be DCE'd"
                            }
                            nestedIfLiveIndices[n.id] = nestedLiveIndices
                        }
                        // Mirror the top-level apply's "primal IF kept, not cloned" rule —
                        // [handleIfAdjoint] re-clones each branch's body during the recursive
                        // walk, so cloning the IF here would emit duplicate computation.
                        branchNodeMap[n.id] = n
                    } else {
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
                            // §0.4.155 — preserve DxirOpResult wrapping (mirror of the
                            // top-level body-cloning fix). Without this, branch body ops
                            // that reference different result indices of a nested multi-
                            // result IF collapse to the same primal-IF reference, dropping
                            // contributions to result(k) for k>0 inside the branch walk.
                            val clonedOperands = n.operands.map { o ->
                                if (o is DxirOpResult) {
                                    val mappedSource = branchNodeMap[o.source.id]
                                        ?: error(
                                            "walkBranchReverse: operand id=${o.source.id} of branch op " +
                                                "${n.id} not in nodeMap (DxirOpResult source)",
                                        )
                                    if (mappedSource is DxirOp) mappedSource.result(o.index) else mappedSource
                                } else {
                                    branchNodeMap[o.id]
                                        ?: error("walkBranchReverse: operand id=${o.id} of branch op ${n.id} not in nodeMap")
                                }
                            }
                            branchNodeMap[n.id] = builder.op(n.op, clonedOperands, n.type, n.attrs)
                        }
                    }
                }
                else -> error(
                    "walkBranchReverse: unsupported branch body node ${n::class.simpleName}",
                )
            }
        }

        // Step 2: seed the per-branch gradAccum with each (live-index → upstream)
        // contribution. §0.4.155 — Phase 5b: multiple live indices can each carry
        // their own upstream; if two indices yield the SAME SSA id (e.g.,
        // `yields(x, x)`), the gradient contributions ADD-merge naturally via the
        // existing accumulator pattern.
        val gradAccum = HashMap<Pair<Int, Int>, DxirNode>()
        for ((idx, upstream) in upstreams) {
            require(idx in block.terminator.indices) {
                "walkBranchReverse: live index $idx out of bounds for branch terminator " +
                    "size=${block.terminator.size}"
            }
            val yieldNode = block.terminator[idx]
            val key = yieldNode.gradKey()
            val existing = gradAccum[key]
            gradAccum[key] = if (existing == null) upstream
            else builder.op(OpKind.ADD, listOf(existing, upstream), upstream.type)
        }

        // Step 3: reverse walk through the branch body.
        for (n in block.body.asReversed()) {
            if (n !is DxirOp) continue

            // §0.4.448 — audit finding C: same demoted-kind refusal as the
            // top-level reverse walk, for ops living inside an IF branch body.
            demotedKindRefusal(n.op, "walkBranchReverse")?.let { error(it) }

            // §0.4.155 — IF dispatch (mirrors top-level): collect per-(live-index)
            // upstreams from the per-(id, idx) gradAccum. Single-result IFs use {0}.
            // Multi-result IFs use the set computed at Step 1.
            if (n.op == OpKind.IF) {
                val liveIndices: Set<Int> = if (n.isMultiResult) {
                    nestedIfLiveIndices[n.id] ?: emptySet()
                } else {
                    setOf(0)
                }
                val upstreamsForN: Map<Int, DxirNode> = liveIndices.mapNotNull { idx ->
                    gradAccum[n.id to idx]?.let { idx to it }
                }.toMap()
                if (upstreamsForN.isEmpty()) continue
                handleIfAdjoint(
                    ifNode = n,
                    upstreams = upstreamsForN,
                    outerGradAccum = gradAccum,
                    outerNodeMap = branchNodeMap,
                    primalById = primalById,
                    builder = builder,
                )
                continue
            }

            // §0.4.154 — non-IF body ops are single-result; their upstream lives
            // at index 0.
            val upstreamForN = gradAccum[n.id to 0] ?: continue
            // §0.4.34 — COARSENED in a branch body: splice the gradient_body via the
            // same helper the outer reverse walk uses. Shared logic keeps the
            // gradient-through-coarsened semantics identical regardless of whether the
            // COARSENED sits at the function's top level or inside an IF branch. The
            // clone (`branchNodeMap[n.id]`) carries branch-scope operand references;
            // handleCoarsenedAdjoint reads gradient_body and emits contributions into
            // the branch's gradAccum (not the outer one).
            if (n.op == OpKind.COARSENED) {
                val clonedCoarsened = branchNodeMap[n.id] as DxirOp
                // §0.4.179 — Phase 5c. Branch-body COARSENED stays single-result for
                // C.3b.3b2 (per [coarsenMultiSoi]'s `multi-result COARSENED in branch
                // not supported` guard). Single-result → upstream lives at index 0;
                // pass as a one-entry map.
                handleCoarsenedAdjoint(
                    coarsened = clonedCoarsened,
                    upstreams = mapOf(0 to upstreamForN),
                    outerGradAccum = gradAccum,
                    outerNodeMap = branchNodeMap,
                    primalById = primalById,
                    builder = builder,
                )
                continue
            }
            // §0.4.155 — nested IFs (single-result and multi-result) are handled by
            // the per-(live-index) dispatch above this block; only non-IF, non-COARSENED
            // body ops fall through to the VjpRule path below.
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
                val accumKey = primalOperand.gradKey()
                val existing = gradAccum[accumKey]
                gradAccum[accumKey] = if (existing == null) contribution
                else builder.op(OpKind.ADD, listOf(existing, contribution), contribution.type)
            }
        }

        return gradAccum
    }

    /**
     * Walk an IF (or any region-bearing op) and enqueue every operand id
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

    /** Integer dtypes whose gradients are structurally zero. */
    private fun isIntegerDtype(dtype: io.tlaloc.core.DType): Boolean =
        dtype == I32 || dtype == I64 || dtype == io.tlaloc.core.Bool
}
