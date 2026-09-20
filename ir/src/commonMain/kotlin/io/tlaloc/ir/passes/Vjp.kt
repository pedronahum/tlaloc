package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * A vector-Jacobian product (VJP) rule for one [OpKind]. Given a primal op, the upstream
 * gradient flowing into its result, and a [DxirBuilder] to emit into, returns one
 * (operand, contribution) pair per input that the rule wishes to backpropagate through.
 *
 * ### Calling convention
 *
 * The rule is invoked with a version of the primal op whose operand *nodes* live in the
 * gradient function's scope — either a proper clone in the gradient body, or a detached
 * "phantom" op whose operands point at cloned operand nodes. The first element of each
 * returned pair MUST be one of the passed-in `op.operands` (by reference identity); the
 * framework matches that back to an operand index to identify which primal SSA value each
 * contribution is intended for.
 *
 * Returning a list (rather than a map) preserves multiplicity for the aliased-operand
 * case — e.g., `x * x` has the same node at `op.operands[0]` and `op.operands[1]`, and a
 * map would collapse the two contributions into one.
 *
 * Operands the rule does not contribute back to (e.g., a pure-shape attribute) should be
 * omitted from the returned list.
 *
 * ### [readsPrimalOperandIndices] — used-by-adjoint analysis
 *
 * Each rule declares which primal operand indices its adjoint body-ops *dereference*.
 * Only those operands need to live in the gradient function's body; others may receive
 * contributions but their node identity is consumed only by `indexOf`-based index
 * resolution. [DxirReverseTransform] uses this declaration to skip cloning primal
 * subgraphs whose values are provably dead on the gradient side.
 *
 * Examples: `ADD`/`SUB`/`NEG` read no operands (the adjoint is a function of the upstream
 * gradient only). `MUL` and `DIV` read both operands (the adjoint mixes upstream with
 * primal operand values).
 *
 * ### Source of truth
 *
 * This registry is the canonical home for the math of every supported reverse-mode rule.
 * The runtime tape in `:autograd` mirrors a subset of these rules in float-array form for
 * performance; per-op equivalence tests (in `:compiler-plugin`'s test harness) verify the
 * two paths produce the same numerical answer for every case in the registry. With
 * MATMUL registered as of §0.4.9, the registry now covers every op the runtime tape
 * currently traces (ADD/SUB/MUL/DIV/NEG/RELU/SUM/MEAN/MATMUL) — the "single math source
 * of truth" goal of §11.8.1 step 1 is honestly met.
 */
interface VjpRule {
    /**
     * Primal operand indices whose node value the rule dereferences when emitting its
     * adjoint body-ops (i.e., passes to [DxirBuilder.op] as an operand). Indices NOT
     * listed here may still participate in the returned (operand, contribution) pairs —
     * but only as indexOf-resolved keys, not as dereferenced nodes — and therefore don't
     * constrain which primal subgraphs must be cloned into the gradient body.
     */
    val readsPrimalOperandIndices: Set<Int>

    /**
     * Per-node refinement of [readsPrimalOperandIndices]. Whether a rule dereferences
     * an operand can depend on the node itself: Phase A5c-2 has [SumRule] and
     * [MeanRule] attach a shape-only template operand to their scalar seed ONLY when
     * the target shape carries a -1 sentinel — with concrete dims no template is
     * needed, because synthesis bakes every extent as a const — so the clone that
     * keeps the template alive is only required in the sentinel case.
     *
     * [DxirReverseTransform] calls this (not the property) when seeding
     * `usedByAdjoint`, so the seeding and the rule's own dereferences cannot
     * disagree; the property remains the static over-approximation for callers that
     * have no node in hand.
     */
    fun readsPrimalOperands(op: DxirOp): Set<Int> = readsPrimalOperandIndices

    fun apply(
        op: DxirOp,
        upstream: DxirNode,
        builder: DxirBuilder,
    ): List<Pair<DxirNode, DxirNode>>
}

/**
 * Registered [VjpRule]s keyed by [OpKind]. Lookup yields `null` for ops with no rule
 * (callers decide whether that's a hard error or a fallback path).
 */
object VjpRegistry {

    // --- Elementwise binary ---

    /**
     * Phase A5c — unbroadcast an adjoint contribution back to the shape of the
     * [operand] it belongs to.
     *
     * With implicit broadcasting the contribution is shaped like the RESULT (the
     * operands broadcast against each other), while the gradient that accumulates
     * onto an operand must be shaped like that operand: `d/da Σ(a ⊙ b)` with
     * `a:[N,1]`, `b:[N,C]` is `[Σ_C upstream·b]` of shape `[N,1]`, not `[N,C]`.
     * The un-broadcast is NumPy's reduce-over-replicated-axes, which is exactly
     * `SUM_TO` (§0.4.373) reading the target extents from the operand's ACTUAL
     * runtime shape — so it is correct under the -1 sentinel dims of `grad {}`,
     * where which axes were size-1 (or which axes the operand lacked) is
     * statically unknowable.
     *
     * Skipped when the shapes are PROVABLY identical: concrete-and-equal dims need
     * no reduce, and a splat const's type is the shape it was splatted to, so its
     * contribution is already right-sized. Both cases would be a runtime no-op
     * anyway (`sumToLike` reduces nothing) — skipping them keeps the concrete-dims
     * IR that the coarsener, the emitter tests and the pinned gradient tests walk
     * byte-identical to pre-A5c. Anything else (differing dims, or any sentinel)
     * takes the SUM_TO path, which is sound in both directions.
     */
    private fun unbroadcast(builder: DxirBuilder, contribution: DxirNode, operand: DxirNode): DxirNode {
        val target = operand.type
        if (contribution.type == target && (target.dims.all { it > 0 } || operand is DxirConst)) {
            return contribution
        }
        return builder.op(OpKind.SUM_TO, listOf(contribution, operand), target)
    }

    /**
     * Phase A5c-2 — whether a scalar-seed splat to [target] needs a runtime shape
     * template operand. Concrete dims need none: synthesis bakes every extent as a
     * const. A -1 sentinel does, because the extents exist only at execution and the
     * static-atom axis-matching that would otherwise supply them is a guess — wrong
     * as soon as two params share atoms but differ at runtime, or a wide target
     * axis-matches a narrow param. [SumRule] and [MeanRule] consult it both when
     * emitting and when declaring [VjpRule.readsPrimalOperands], so the template is
     * cloned into the gradient body exactly when it is referenced.
     */
    private fun needsShapeTemplate(target: DxirType): Boolean = target.dims.any { it <= 0 }

    /**
     * Phase A5c-3 — emit a `broadcast_dimensions = []` BROADCAST of [value] to
     * [targetType], carrying [template] as a shape-only second operand whenever the
     * target's extents are not statically known ([needsShapeTemplate]).
     *
     * Every adjoint that splats or un-reduces back to a primal shape goes through
     * here, so the shape source is a value whose runtime dims ARE the target rather
     * than synthesis's atom-matching guess. That guess is unsafe once operands
     * broadcast, and doubly so for the STRETCH form: an un-reduce target is usually
     * an intermediate's shape (`(v * m).max(1)`'s adjoint stretches back to the
     * broadcast product), which no param need have — axis-matching a rank-2 target
     * against a rank-1 param's axis produced `[3,3]` where `[2,3]` was meant.
     *
     * [template] must be a node the rule already dereferences (i.e. in its
     * `readsPrimalOperandIndices`), so it is cloned into the gradient body anyway and
     * the extra reference costs nothing. Rules whose operand is NOT otherwise read
     * refine [VjpRule.readsPrimalOperands] per node instead — see [SumRule].
     */
    private fun broadcastTo(
        builder: DxirBuilder,
        value: DxirNode,
        template: DxirNode,
        targetType: DxirType,
    ): DxirNode = builder.op(
        OpKind.BROADCAST,
        if (needsShapeTemplate(targetType)) listOf(value, template) else listOf(value),
        targetType,
        attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
    )

    /**
     * Phase A5c-3 — a literal splatted over [template]'s shape. Under SYMBOLIC dims
     * this is a scalar const carried by a templated [broadcastTo], so synthesis reads
     * the target extents off a real runtime value; a bare shaped const has no runtime
     * shape source and synthesis would have to guess its extents by axis-matching
     * static atoms against the params. Under CONCRETE dims — and for a scalar target,
     * which needs no splat at all — this is exactly the plain shaped const it always
     * was, so no pre-A5c IR changes shape.
     */
    private fun splatConst(builder: DxirBuilder, value: Any, template: DxirNode, targetType: DxirType): DxirNode =
        if (needsShapeTemplate(targetType) && !targetType.isScalar) {
            broadcastTo(
                builder,
                builder.const(value, DxirType(targetType.dtype, emptyList())),
                template,
                targetType,
            )
        } else {
            builder.const(value, targetType)
        }

    /**
     * d(a + b)/da = 1, d(a + b)/db = 1 — the upstream, un-broadcast to each
     * operand's own shape (Phase A5c: the operands may broadcast against each
     * other, so the upstream is result-shaped and each side needs its own reduce).
     */
    val AddRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder) = listOf(
            op.operands[0] to unbroadcast(builder, upstream, op.operands[0]),
            op.operands[1] to unbroadcast(builder, upstream, op.operands[1]),
        )
    }

    /** d(a - b)/da = 1, d(a - b)/db = -1. */
    val SubRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val negUp = builder.op(OpKind.NEG, listOf(upstream), upstream.type)
            return listOf(
                op.operands[0] to unbroadcast(builder, upstream, op.operands[0]),
                op.operands[1] to unbroadcast(builder, negUp, op.operands[1]),
            )
        }
    }

    /** d(a * b)/da = b, d(a * b)/db = a. */
    val MulRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val a0 = op.operands[0]
            val a1 = op.operands[1]
            // Each product is result-shaped (the operands broadcast against each
            // other), so each is un-broadcast to its own operand afterwards.
            val da = builder.op(OpKind.MUL, listOf(upstream, a1), upstream.type)
            val db = builder.op(OpKind.MUL, listOf(upstream, a0), upstream.type)
            return listOf(
                a0 to unbroadcast(builder, da, a0),
                a1 to unbroadcast(builder, db, a1),
            )
        }
    }

    /**
     * d(a / b)/da = 1/b, d(a / b)/db = -a/b².  Emits using only ops the IR-side
     * synthesis path supports (ADD/SUB/MUL/DIV/NEG).
     */
    val DivRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val a0 = op.operands[0]
            val a1 = op.operands[1]
            // da = upstream / b
            val da = builder.op(OpKind.DIV, listOf(upstream, a1), upstream.type)
            // db = -upstream * a / (b * b)
            val bb = builder.op(OpKind.MUL, listOf(a1, a1), a1.type)
            // Phase A5c — `a / (b·b)` broadcasts a against b, so its shape is the
            // RESULT's, not a's (identical when the operands already agree).
            val aOverBB = builder.op(OpKind.DIV, listOf(a0, bb), op.type)
            val mul = builder.op(OpKind.MUL, listOf(upstream, aOverBB), upstream.type)
            val db = builder.op(OpKind.NEG, listOf(mul), mul.type)
            return listOf(
                a0 to unbroadcast(builder, da, a0),
                a1 to unbroadcast(builder, db, a1),
            )
        }
    }

    // --- Elementwise unary ---

    /** d(-a)/da = -1. Contribution is -upstream. */
    val NegRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val da = builder.op(OpKind.NEG, listOf(upstream), upstream.type)
            return listOf(op.operands[0] to da)
        }
    }

    /**
     * d(relu(x))/dx = step(x) = 1 if x > 0 else 0. Contribution is `upstream * step(x)`.
     *
     * The rule dereferences the primal operand (to feed it into [OpKind.STEP]), so
     * `readsPrimalOperandIndices` includes index 0 — without it the `usedByAdjoint`
     * analysis would drop the primal operand's clone and [DxirFunction]'s init-time
     * ref-integrity check would fire at construction.
     *
     * Declared before [rules] because Kotlin initialises `object` properties in source
     * order; a forward reference from `rules` to a later val fails to compile.
     */
    val ReluRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val mask = builder.op(OpKind.STEP, listOf(x), x.type)
            val contribution = builder.op(OpKind.MUL, listOf(upstream, mask), upstream.type)
            return listOf(x to contribution)
        }
    }

    // --- Reductions ---

    /**
     * d(sum(x))/dx = 1 broadcast to x's shape. Contribution is `BROADCAST(upstream, x.dims)`
     * with `broadcast_dimensions = []` — the operand is scalar (sum's output) and the
     * output shape is the operand's primal shape.
     *
     * [readsPrimalOperandIndices] = `emptySet()`: the rule only inspects the primal
     * operand's *type* (to populate BROADCAST's target shape), never dereferences its
     * *value*. Shape metadata on the node is available regardless of whether the node
     * is cloned into the gradient body or only referenced as a phantom.
     */
    val SumRule: VjpRule = object : VjpRule {
        // Phase A5c-2 — the scalar-seed splat below may carry `x` as a shape-only
        // template operand, in which case `x` must survive into the gradient body.
        // The static property over-approximates; [readsPrimalOperands] refines it per
        // node so a concrete-dims SUM does not clone its summed operand (which would
        // both recompute it and, if it is a rank-changing op the gradient scope cannot
        // synthesise, break synthesis outright).
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun readsPrimalOperands(op: DxirOp): Set<Int> =
            if (op.operands.isNotEmpty() && needsShapeTemplate(op.operands[0].type)) setOf(0) else emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val targetType = DxirType(upstream.type.dtype, x.type.dims)
            // §0.4.366 — axis-aware arm (Phase A1): for `sum(dims)` the upstream
            // has the reduced shape; RESHAPE it to the keepdims spelling (a
            // no-op when the primal kept dims) so the interpreter's equal-rank
            // stretch BROADCAST can un-reduce it over the reduced axes. The
            // full-reduce path keeps the scalar-splat BROADCAST unchanged.
            val up = reshapeToKeepdims(op, x, upstream, builder)
            // Phase A5c-2/A5c-3 — the seed carries `x` as a shape-only template
            // whenever the target's extents are symbolic, in BOTH forms: the scalar
            // splat (full reduce) and the equal-rank stretch (axis reduce). Synthesis
            // resolves a splat/stretch target by axis-matching static IrType atoms
            // against the params, which is a guess that stops being safe once operands
            // broadcast: two params sharing atoms can differ at runtime (`[2,1]` and
            // `[2,3]` are both `Rank2<Sym, Lit<Int>>`), and a rank-2 target can match
            // a rank-1 param's axis outright — either way the seed came out with the
            // wrong shape and the un-broadcast SUM_TO then threw. `x` is the one value
            // whose runtime shape IS the target: the SUM_TO/PAD_TO convention.
            val contribution = broadcastTo(builder, up, x, targetType)
            return listOf(x to contribution)
        }
    }

    /**
     * d(mean(x))/dx = (1/N) broadcast to x's shape, where N is the element count of x.
     * Emitted as `BROADCAST(MUL(upstream, const(1/N)), x.dims)`.
     *
     * [readsPrimalOperandIndices] = `emptySet()`: like [SumRule], the rule only reads
     * operand shape metadata (to derive N and to populate BROADCAST's target), never
     * the operand's value.
     *
     * For an empty primal operand (N = 0), the emitted const would be `1/0` — but
     * BROADCAST to a zero-length shape produces a zero-length `FloatArray`, so the
     * Inf / NaN value is vacuously never materialised at any index. No callers exercise
     * this today; if one does, specialise to a zero const here.
     */
    val MeanRule: VjpRule = object : VjpRule {
        // Phase A5c-2 — the scalar-seed splats below may carry `x` as a shape-only
        // template operand; [readsPrimalOperands] refines the static property per node
        // so a concrete-dims MEAN does not clone its operand (see [SumRule]).
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun readsPrimalOperands(op: DxirOp): Set<Int> =
            if (op.operands.isNotEmpty() && needsShapeTemplate(op.operands[0].type)) setOf(0) else emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            // §0.4.366 — axis-aware N (Phase A1): the divisor is the count of
            // elements actually folded into each output cell — the product of
            // the REDUCED extents only, not x's full element count.
            val rd = reductionDimsOf(op)
            val reducedExtents = rd?.map { x.type.dims[it] } ?: x.type.dims
            val scaled = if (reducedExtents.all { it > 0 }) {
                // Concrete dims: bake 1/N as a const (the fast path — all
                // IR-level callers).
                val n = if (reducedExtents.isEmpty()) 1 else reducedExtents.fold(1) { a, d -> a * d }
                val invN: Any = when (upstream.type.dtype) {
                    F32 -> 1.0f / n
                    F64 -> 1.0 / n
                    else -> error("MeanRule: unsupported dtype ${upstream.type.dtype}")
                }
                val invNConst = builder.const(invN, upstream.type)
                builder.op(OpKind.MUL, listOf(upstream, invNConst), upstream.type)
            } else {
                // §0.4.366 — symbolic (sentinel) dims: the K2 plugin lowers
                // grad-lambda shapes with -1 placeholders, so N cannot be baked
                // at transform time (doing so produced 1/-1 = -1 — the g1
                // E2E bug). Materialise N at RUNTIME with existing ops:
                // ones(x) → the same reduction → N at upstream's shape → DIV.
                val one: Any = if (upstream.type.dtype == F64) 1.0 else 1.0f
                val seed = builder.const(one, DxirType(upstream.type.dtype, emptyList()))
                // Phase A5c-2 — x rides along as a shape-only template so the splat
                // reads its target extents at runtime (see [broadcastTo]).
                val ones = broadcastTo(builder, seed, x, x.type)
                val nT = builder.op(OpKind.SUM, listOf(ones), op.type, attrs = op.attrs)
                builder.op(OpKind.DIV, listOf(upstream, nT), upstream.type)
            }
            val up = reshapeToKeepdims(op, x, scaled, builder)
            val targetType = DxirType(upstream.type.dtype, x.type.dims)
            // Phase A5c-2/3 — x is the shape template in both the splat and the
            // stretch form (see [broadcastTo]).
            val contribution = broadcastTo(builder, up, x, targetType)
            return listOf(x to contribution)
        }
    }

    // --- Linear algebra ---

    /**
     * d(A @ B)/dA = upstream @ Bᵀ, d(A @ B)/dB = Aᵀ @ upstream.  Emits two TRANSPOSEs
     * and two MATMULs into the gradient body.
     *
     * Shape derivation: `op.operands[0].type.dims = [M, K]`, `op.operands[1].type.dims
     * = [K, N]`, `upstream.type.dims = [M, N]`. Aᵀ is typed `[K, M]`; Bᵀ is typed
     * `[N, K]`; dA is `[M, K]`; dB is `[K, N]`. The permutation attr is `[1, 0]`,
     * matching the `:stablehlo` emitter's `intListAttr(node, "permutation")` contract.
     *
     * [readsPrimalOperandIndices] = `setOf(0, 1)`: the rule dereferences BOTH primal
     * operands (A feeds the Aᵀ in dB, B feeds the Bᵀ in dA) as input nodes to emitted
     * body-ops. This matches the MulRule / DivRule pattern — contrast with SumRule /
     * MeanRule which only read shape metadata and declare `emptySet()`.
     *
     * §0.4.137 — extended from rank-2 to rank-2-or-3 batched. §0.4.138 — generalised
     * to any rank ≥ 2 with arbitrary batch axes. The shape contract is the same
     * canonical batched-matmul convention §0.4.135's substrate uses: `(B0..Bk, M, K)
     * × (B0..Bk, K, N) → (B0..Bk, M, N)`. The TRANSPOSE permutation becomes `[0..r-3,
     * r-1, r-2]` — preserve all batch axes, swap the last two. The MATMUL kind is
     * the same op for all ranks; the substrate dispatches by shape.
     *
     * Declared before [rules] because Kotlin initialises `object` properties in source
     * order; a forward reference from `rules` to a later val fails to compile
     * (§0.4.3 object-init trap — worth repeating in every new VjpRule).
     */
    /**
     * §0.4.353 — rank-1 dot product `s = Σ a_i·b_i` (OpKind.DOT, the
     * `rank1 contract rank1` lowering): d s/d a = upstream ⊙ b,
     * d s/d b = upstream ⊙ a (upstream is the scalar seed; MUL
     * broadcasts scalar × vector). Contributions are typed with the
     * *receiving* operand's DxirType so named axes accumulate onto the
     * right parameter. Gap found by the §0.4.353 check-time
     * differentiability probe — a grad over rank-1 contract previously
     * failed only at runtime.
     */
    val DotRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val a = op.operands[0]
            val b = op.operands[1]
            require(a.type.rank == 1 && b.type.rank == 1) {
                "DotRule: rank-1 operands required, got ${a.type.dims} x ${b.type.dims}"
            }
            // Scalar upstream broadcast to vector shape first (the SumRule
            // convention). Phase A5c-3 — each splat carries its own operand as the
            // shape template (see [broadcastTo]); both are already cloned by this
            // rule, so the references are free.
            val upA = broadcastTo(builder, upstream, a, DxirType(upstream.type.dtype, a.type.dims))
            val upB = broadcastTo(builder, upstream, b, DxirType(upstream.type.dtype, b.type.dims))
            val da = builder.op(OpKind.MUL, listOf(upA, b), a.type)
            val db = builder.op(OpKind.MUL, listOf(upB, a), b.type)
            return listOf(a to da, b to db)
        }
    }

    /**
     * §0.4.359 — `d/dx reshape(x)` = reshape the upstream back to x's
     * shape (element-count-preserving relayout has an identity Jacobian
     * under the row-major flat view). Gap flagged by the DiffKT
     * comparison: RESHAPE was lowerable but undifferentiable.
     */
    val ReshapeRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val dx = builder.op(OpKind.RESHAPE, listOf(upstream), x.type)
            return listOf(x to dx)
        }
    }

    /**
     * §0.4.366 — shared axis-reduction plumbing (Phase A1).
     *
     * [reductionDimsOf] reads the op's `reduction_dims` attr (absent/empty →
     * null = full reduce). [reshapeToKeepdims] adapts a reduced-shape value
     * (the upstream, or a recomputed reduction output) for the interpreter's
     * equal-rank stretch BROADCAST: when the primal squeezed the reduced axes
     * (`op.type.rank != x.rank`), RESHAPE to the keepdims spelling (size-1 at
     * each reduced axis); when the primal kept dims, or was a full reduce to
     * scalar (the scalar-splat BROADCAST arm), pass through unchanged.
     */
    private fun reductionDimsOf(op: DxirOp): List<Int>? =
        (op.attrs["reduction_dims"] as? List<*>)
            ?.map { (it as Number).toInt() }
            ?.takeIf { it.isNotEmpty() }

    private fun reshapeToKeepdims(
        op: DxirOp,
        x: DxirNode,
        value: DxirNode,
        builder: DxirBuilder,
    ): DxirNode {
        val rd = reductionDimsOf(op) ?: return value
        // Scalar values (an attr-bearing reduce that covered every axis) take
        // the scalar-splat BROADCAST arm directly — no reshape needed, and the
        // synthesis's unsqueeze helper is tensor-only.
        if (value.type.isScalar) return value
        if (value.type.rank == x.type.rank) return value
        val kd = DxirType(
            value.type.dtype,
            x.type.dims.mapIndexed { i, d -> if (i in rd) 1 else d },
        )
        return builder.op(OpKind.RESHAPE, listOf(value), kd)
    }

    private fun reduceExtremumRule(kind: OpKind): VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            // Recompute the reduction (TanhRule convention) and broadcast it
            // and the upstream back over the reduced axes. Full-reduce flows
            // through the interpreter's scalar-splat BROADCAST arm; axis
            // reductions RESHAPE to keepdims first (§0.4.366) so the
            // equal-rank stretch arm applies whether the primal kept or
            // squeezed the reduced axes.
            val yRe = builder.op(kind, listOf(x), op.type, attrs = op.attrs)
            // Phase A5c-3 — x is the shape template for both un-reduce stretches
            // (see [broadcastTo]): under broadcasting the extremum's operand is often
            // an intermediate whose shape no param has, so axis-matching the target
            // against the params can pick the wrong extents. x is already cloned
            // (`readsPrimalOperandIndices = setOf(0)`), so the reference is free.
            val yB = broadcastTo(builder, reshapeToKeepdims(op, x, yRe, builder), x, x.type)
            val upB = broadcastTo(builder, reshapeToKeepdims(op, x, upstream, builder), x, x.type)
            // Indicator of the extremum: for MAX, yB - x ≥ 0 with equality
            // exactly at maxima → 1 - sign(yB - x); MIN mirrors with x - yB.
            // Tie convention: FULL upstream to every tied element (the
            // JAX-select convention; PyTorch's amax splits evenly — both are
            // valid subgradients, ours matches select(x == extremum, g, 0)).
            val diff = if (kind == OpKind.MAX) {
                builder.op(OpKind.SUB, listOf(yB, x), x.type)
            } else {
                builder.op(OpKind.SUB, listOf(x, yB), x.type)
            }
            val sgn = builder.op(OpKind.SIGN, listOf(diff), x.type)
            // Phase A5c-3 — the 1.0 splat carries x as its shape template under
            // symbolic dims: a shaped CONST has no runtime shape source, so synthesis
            // materialises it by axis-matching static atoms against the params, and
            // with broadcasting that guess can land on the wrong param (`[3,3]` where
            // the mask belongs at `[2,3]`). Concrete dims keep the plain shaped const.
            val one = splatConst(builder, floatLiteralForDtype(1.0, x.type.dtype), x, x.type)
            val mask = builder.op(OpKind.SUB, listOf(one, sgn), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upB, mask), x.type)
            return listOf(x to dx)
        }
    }

    /** §0.4.359 — max-reduction subgradient (see [reduceExtremumRule]). */
    val MaxRule: VjpRule = reduceExtremumRule(OpKind.MAX)

    /** §0.4.359 — min-reduction subgradient (see [reduceExtremumRule]). */
    val MinRule: VjpRule = reduceExtremumRule(OpKind.MIN)

    /**
     * §0.4.359 — raw softmax adjoint: with `y = softmax(x, axis)`,
     * `dx = y ⊙ (upstream − Σ_axis(upstream ⊙ y))`. Recomputes y
     * (TanhRule convention); the inner sum keeps dims for the stretch
     * broadcast back over the axis. Until now softmax differentiated
     * only through coarsened pattern bodies — a bare softmax in a user
     * lambda failed (flagged by the DiffKT comparison).
     */
    val SoftmaxRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val rank = x.type.rank
            val axisRaw = (op.attrs["axis"] as? Number)?.toInt() ?: (rank - 1)
            val axis = if (axisRaw < 0) axisRaw + rank else axisRaw
            val y = builder.op(OpKind.SOFTMAX, listOf(x), op.type, attrs = op.attrs)
            val t = builder.op(OpKind.MUL, listOf(upstream, y), x.type)
            val keepdimsType = DxirType(
                x.type.dtype,
                x.type.dims.mapIndexed { i, d -> if (i == axis) 1 else d },
            )
            val srow = builder.op(
                OpKind.SUM, listOf(t), keepdimsType,
                attrs = mapOf("reduction_dims" to listOf(axis)),
            )
            // Phase A5c-3 — x is the shape template for the row-sum stretch
            // (see [broadcastTo]); x is already cloned by this rule.
            val sB = broadcastTo(builder, srow, x, x.type)
            val diff = builder.op(OpKind.SUB, listOf(upstream, sB), x.type)
            val dx = builder.op(OpKind.MUL, listOf(y, diff), x.type)
            return listOf(x to dx)
        }
    }

    /**
     * §0.4.360 — `d/dx_i concat(x_1..x_n, dim)` = the upstream sliced back
     * to each operand's window along `dimension`.
     */
    val ConcatRule: VjpRule = object : VjpRule {
        // Phase A2b — variadic, so the static property cannot express "all
        // operands"; [readsPrimalOperands] is authoritative (the reverse transform
        // calls it). The symbolic branch below dereferences EVERY operand as a
        // SLICE_LIKE shape template, so all of them must be cloned into the
        // gradient body; the concrete branch dereferences none and keeps the
        // pre-A2b behaviour of not cloning the concat's operands at all.
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun readsPrimalOperands(op: DxirOp): Set<Int> =
            if (concatShapeIsSymbolic(op)) op.operands.indices.toSet() else emptySet()

        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val dim = (op.attrs["dimension"] as? Number)?.toInt() ?: 0
            val rank = op.type.rank
            // Phase A2b — SYMBOLIC dims: operand i's window starts at the cumulative
            // sum of the PRIOR operands' runtime axis extents and runs for its own,
            // and none of those extents exist at transform time (-1 sentinels under
            // `grad {}`). Bake nothing: SLICE_LIKE reads both bounds off shape-only
            // template operands at execution, the SUM_TO/PAD_TO convention.
            if (concatShapeIsSymbolic(op)) {
                return op.operands.mapIndexed { i, x ->
                    x to builder.op(
                        OpKind.SLICE_LIKE,
                        listOf(upstream, x) + op.operands.take(i),
                        x.type,
                        attrs = mapOf("axis" to dim),
                    )
                }
            }
            var offset = 0
            return op.operands.map { x ->
                val len = x.type.dims[dim]
                val starts = (0 until rank).map { if (it == dim) offset else 0 }
                val limits = (0 until rank).map { if (it == dim) offset + len else op.type.dims[it] }
                offset += len
                val dx = builder.op(
                    OpKind.SLICE, listOf(upstream), x.type,
                    attrs = mapOf(
                        "start_indices" to starts,
                        "limit_indices" to limits,
                        "strides" to List(rank) { 1 },
                    ),
                )
                x to dx
            }
        }
    }

    /**
     * Phase A2b — whether a CONCAT's window offsets are statically knowable. Any
     * sentinel on the result or on an operand means no: the axis extents that
     * [ConcatRule] would otherwise accumulate into `start_indices` are -1, and a
     * baked offset of -1 is not a wrong answer so much as a meaningless one.
     */
    private fun concatShapeIsSymbolic(op: DxirOp): Boolean =
        needsShapeTemplate(op.type) || op.operands.any { needsShapeTemplate(it.type) }

    /**
     * §0.4.360 — `d/dx slice(x)` = the upstream zero-padded back into x's
     * shape (SLICE's adjoint IS a pad). v1 requires unit strides — the
     * strided adjoint needs interior padding, deferred until demanded.
     */
    val SliceRule: VjpRule = object : VjpRule {
        // §0.4.374 — the adjoint dereferences the primal input (operand 0) as
        // PAD_TO's `template` operand (a shape source: `high` is derived from its
        // runtime extent), so the input subgraph must be cloned into the gradient
        // body.
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            @Suppress("UNCHECKED_CAST")
            val starts = op.attrs["start_indices"] as List<Int>
            @Suppress("UNCHECKED_CAST")
            val strides = op.attrs["strides"] as List<Int>
            require(strides.all { it == 1 }) {
                "SliceRule: strided slices are not differentiable in v1 (needs interior padding)"
            }
            // §0.4.374 — zero-pad the upstream back into x's window. The trailing
            // pad per axis is `high[i] = x.dim[i] − start[i] − upstream.dim[i]`,
            // which reads x's extent — a -1 SENTINEL under `grad {}`. PAD_TO reads
            // that extent from x's ACTUAL runtime shape at execution instead of
            // baking it as an attr (`low` = the user's slice starts, all literals).
            // On the concrete-dims IR path this is numerically identical to the old
            // PAD(low = starts, high = x.dim − limit) adjoint.
            val dx = builder.op(
                OpKind.PAD_TO, listOf(upstream, x), x.type,
                attrs = mapOf("low" to starts),
            )
            return listOf(x to dx)
        }
    }

    /**
     * §0.4.360 — `where(pred, a, b)`: d/da = upstream ⊙ mask,
     * d/db = upstream ⊙ (1 − mask), mask = cast(pred). No contribution to
     * pred (boolean routing carries no gradient).
     */
    val WhereRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val pred = op.operands[0]
            val a = op.operands[1]
            val b = op.operands[2]
            val maskType = DxirType(upstream.type.dtype, pred.type.dims)
            val mask = builder.op(OpKind.CAST, listOf(pred), maskType)
            val da = builder.op(OpKind.MUL, listOf(upstream, mask), a.type)
            // Phase A5c-3 — the 1.0 splat carries `pred` as its shape template under
            // symbolic dims (see [broadcastTo]): a shaped const has no runtime shape
            // source of its own.
            val one = splatConst(builder, floatLiteralForDtype(1.0, upstream.type.dtype), pred, maskType)
            val inv = builder.op(OpKind.SUB, listOf(one, mask), maskType)
            val db = builder.op(OpKind.MUL, listOf(upstream, inv), b.type)
            return listOf(a to da, b to db)
        }
    }

    /** §0.4.360 — comparisons are piecewise-constant: zero gradient to both
     * operands (the SignRule convention). */
    val CompareRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> =
            op.operands.map { x ->
                x to builder.const(floatLiteralForDtype(0.0, x.type.dtype), x.type)
            }
    }

    /** §0.4.360 — `d/dx pad(x)` = the upstream sliced back to x's window
     * (PAD's adjoint IS a slice — the dual of [SliceRule]). */
    val PadRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            @Suppress("UNCHECKED_CAST")
            val low = op.attrs["low"] as List<Int>
            val dx = builder.op(
                OpKind.SLICE, listOf(upstream), x.type,
                attrs = mapOf(
                    "start_indices" to low,
                    "limit_indices" to x.type.dims.indices.map { low[it] + x.type.dims[it] },
                    "strides" to List(x.type.rank) { 1 },
                ),
            )
            return listOf(x to dx)
        }
    }

    val MatmulRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val a = op.operands[0]
            val b = op.operands[1]
            val rank = a.type.rank
            require(rank >= 2 && b.type.rank == rank) {
                "MatmulRule: rank ≥ 2 operands required (matching ranks), got " +
                    "${a.type.dims} x ${b.type.dims}"
            }
            val dtype = upstream.type.dtype
            val m = a.type.dims[rank - 2]
            val k = a.type.dims[rank - 1]
            val n = b.type.dims[rank - 1]
            val batchDims = if (rank == 2) emptyList() else a.type.dims.subList(0, rank - 2)
            // Permutation: preserve batch axes [0..r-3], swap last two ([r-1, r-2]).
            val perm = (0 until rank - 2).toList() + listOf(rank - 1, rank - 2)
            val transposeAttrs = mapOf("permutation" to perm)
            val aT = builder.op(
                OpKind.TRANSPOSE,
                listOf(a),
                DxirType(dtype, batchDims + listOf(k, m)),
                attrs = transposeAttrs,
            )
            val bT = builder.op(
                OpKind.TRANSPOSE,
                listOf(b),
                DxirType(dtype, batchDims + listOf(n, k)),
                attrs = transposeAttrs,
            )
            val dA = builder.op(OpKind.MATMUL, listOf(upstream, bT), DxirType(dtype, batchDims + listOf(m, k)))
            val dB = builder.op(OpKind.MATMUL, listOf(aT, upstream), DxirType(dtype, batchDims + listOf(k, n)))
            return listOf(a to dA, b to dB)
        }
    }

    /**
     * §0.4.362 — CONV2D adjoint (NCHW / OIHW, the [StablehloEmitter]
     * layouts). The two classical results, expressed with existing ops:
     *
     * - `dX`: a lhs-dilated, tap-reversed transposed conv of dY against W —
     *   W's OIHW dims read under the IOHW layout contract over `o` and emit
     *   `i` for free — padded so the output lands exactly back on X's shape.
     * - `dW`: the batch↔feature transposed trick — X as `[Ci, N, H, W]`
     *   against dY as `[Co, N, Ho, Wo]` (an OIHW kernel with `o = Co`,
     *   `i = N`), `window_strides = rhs_dilation` and `rhs_dilation = stride`
     *   swapped, padded so the result is exactly `[kh, kw]`, transposed back.
     *
     * §0.4.385 — both are emitted as FUSED, runtime-extent ops
     * ([OpKind.CONV2D_DATA_ADJOINT] / [OpKind.CONV2D_KERNEL_ADJOINT]) rather
     * than as the explicit CONV_TRANSPOSE2D / TRANSPOSE+CONV2D+TRANSPOSE
     * chains this rule used to build. The reason is sentinel-safety: both
     * paddings are SOLVED from the primal's extents, and under `grad {}` those
     * extents are -1 sentinels, so solving here baked arithmetic garbage
     * (`[[-3,1],[-3,1]]` where `[[1,1],[1,1]]` is correct for a stride-1
     * padding-1 conv) that the interpreter, the emitter and the host twins all
     * honoured faithfully — a silently wrong gradient, with no downstream gate
     * to catch it. The fused ops take the tensor whose extents are the target
     * as a shape-only template operand (the PAD_TO / SUM_TO / SLICE_LIKE
     * convention) and solve at EXECUTION time; this rule now passes down
     * nothing but the primal's own literal attrs, so it reads no extent at all.
     * Fusing the transposes into the kernel adjoint also leaves the gradient
     * body with no rank-4 TRANSPOSE nodes to type, and makes each adjoint's
     * result IrType simply its template's.
     *
     * v1 scope: `lhs_dilation = [1, 1]` and groups = 1 on the primal
     * (matching the interpreter); strides, padding, and rhs_dilation are
     * fully general. CONV_TRANSPOSE2D's own adjoint is deferred — it only
     * arises when a user differentiates *through* a transposed conv, and
     * the derivation mirrors this one.
     */
    val Conv2dRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val wgt = op.operands[1]

            fun intPair(key: String, def: List<Int>): List<Int> =
                (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
            val s = intPair("window_strides", listOf(1, 1))
            val d = intPair("rhs_dilation", listOf(1, 1))
            val lhsDil = intPair("lhs_dilation", listOf(1, 1))
            require(lhsDil == listOf(1, 1)) {
                "Conv2dRule: primal lhs_dilation must be [1, 1] in v1; got $lhsDil"
            }
            val p = (op.attrs["padding"] as? List<*>)
                ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
                ?: listOf(listOf(0, 0), listOf(0, 0))

            val primalAttrs = mapOf<String, Any>(
                "window_strides" to s,
                "padding" to p,
                "rhs_dilation" to d,
            )
            val dX = builder.op(
                OpKind.CONV2D_DATA_ADJOINT, listOf(upstream, wgt, x), x.type, attrs = primalAttrs,
            )
            val dW = builder.op(
                OpKind.CONV2D_KERNEL_ADJOINT, listOf(x, upstream, wgt), wgt.type, attrs = primalAttrs,
            )
            return listOf(x to dX, wgt to dW)
        }
    }

    /**
     * §0.4.391 — CONV_TRANSPOSE2D adjoint: differentiating THROUGH a transposed
     * conv (a deconvolution / fractionally-strided upsample), which until here was a
     * loud "no VJP rule registered" even though the primal's FIR arm and host twin
     * shipped in §0.4.384.
     *
     * Structurally [Conv2dRule]'s mirror — two fused ops, each carrying the tensor
     * whose shape it produces as its last operand — but simpler in one important way:
     * NO padding solve. A transposed conv's tap maps input↔output through
     * `yDil = yo·s + ky·d − p_low` with `yDil` a multiple of the lhs dilation, so the
     * adjoints invert that one equation per tap and keep it only when it divides
     * evenly and lands in range. Padding, both dilations, the strides and the kernel
     * reversal all fall out of that test, so everything this rule passes down is a
     * literal attr off the primal and it reads no extent at all — sentinel-safe by
     * construction, with no runtime solve to get wrong.
     *
     * Verified against central differences over six configurations (lhs_dilation 1
     * and 2, window_strides 1 and 2, rhs_dilation, reversal, asymmetric padding, and
     * all combined) before implementation.
     *
     * All three engines have an arm; the emitter goes through the
     * conv-of-the-dilated-input identity rather than the index inversion (§0.4.393)
     * and rejects `window_reversal`, which nothing user-reachable sets.
     */
    val ConvTranspose2dRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val wgt = op.operands[1]

            fun intPair(key: String, def: List<Int>): List<Int> =
                (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
            val primalAttrs = mapOf<String, Any>(
                "window_strides" to intPair("window_strides", listOf(1, 1)),
                "padding" to ((op.attrs["padding"] as? List<*>)
                    ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
                    ?: listOf(listOf(0, 0), listOf(0, 0))),
                "lhs_dilation" to intPair("lhs_dilation", listOf(1, 1)),
                "rhs_dilation" to intPair("rhs_dilation", listOf(1, 1)),
                "window_reversal" to ((op.attrs["window_reversal"] as? List<*>)
                    ?.map { it as Boolean } ?: listOf(false, false)),
            )
            val dX = builder.op(
                OpKind.CONV_TRANSPOSE2D_DATA_ADJOINT, listOf(upstream, wgt, x), x.type,
                attrs = primalAttrs,
            )
            val dW = builder.op(
                OpKind.CONV_TRANSPOSE2D_KERNEL_ADJOINT, listOf(x, upstream, wgt), wgt.type,
                attrs = primalAttrs,
            )
            return listOf(x to dX, wgt to dW)
        }
    }

    /**
     * §0.4.363 — AVGPOOL2D adjoint: each input element receives
     * `Σ dY/(kh·kw)` over every window containing it — exactly a
     * transposed convolution of dY with a uniform `1/(kh·kw)` kernel.
     * Fully general strides/padding (count_include_pad — the interpreter's
     * convention, which divides by the FULL window).
     *
     * §0.4.386 — emitted as the fused [OpKind.AVGPOOL2D_GRAD] rather than the
     * original `RESHAPE → CONV_TRANSPOSE2D → RESHAPE` channel-folding chain, for
     * the same sentinel reason [Conv2dRule] gives: that chain solved its padding
     * from the primal's extents AND baked `n * c` as a reshape target, both of
     * which are arithmetic on -1s under `grad {}`. The fused op inverts the window
     * at execution time and needs no channel fold at all — the adjoint is
     * per-channel, so the depthwise-via-batch-folding trick (which only existed to
     * dodge grouped-conv support) is unnecessary.
     */
    val AvgPool2dRule: VjpRule = object : VjpRule {
        // §0.4.386 — the adjoint carries `x` as a shape-only template operand, so
        // its producer must survive into the gradient body. (It was `emptySet()`
        // while the rule only inspected `x.type`; a template OPERAND is a value
        // reference as far as the clone walk is concerned, and omitting it here
        // would leave the emitted node pointing at a primal id that collides with
        // the grad builder's fresh ids — ref-integrity-valid, semantically wrong.)
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            fun intPair(key: String, def: List<Int>): List<Int> =
                (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
            val k = intPair("window", emptyList())
            require(k.size == 2) { "AvgPool2dRule: primal needs `window` [kh, kw]; got $k" }
            val s = intPair("window_strides", k)
            val p = (op.attrs["padding"] as? List<*>)
                ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
                ?: listOf(listOf(0, 0), listOf(0, 0))

            // §0.4.386 — one fused, runtime-extent op instead of the channel-folded
            // `RESHAPE → CONV_TRANSPOSE2D(splat, lhs_dilation = stride, solved
            // padding) → RESHAPE` chain. That chain read `x`'s and `y`'s EXTENTS at
            // transform time in two places (the solved padding, and the reshape's
            // `n * c` target — which under `grad {}`'s -1 sentinels is 1, so the
            // reshape silently claimed a shape the data did not have). The fused op
            // inverts the window at execution time instead, so all this rule passes
            // down is the primal's own literal attrs and it reads no extent at all.
            val dX = builder.op(
                OpKind.AVGPOOL2D_GRAD, listOf(upstream, x), x.type,
                attrs = mapOf("window" to k, "window_strides" to s, "padding" to p),
            )
            return listOf(x to dX)
        }
    }

    /**
     * §0.4.363 — MAXPOOL2D adjoint: recompute `y = maxpool(x)`, then
     * `dx = where(x == y_per_window, dY_per_window, 0)` — each input element
     * receives the upstream of every window it wins.
     *
     * §0.4.389 — emitted as the fused [OpKind.MAXPOOL2D_GRAD] instead of the
     * original nearest-upsample-and-mask chain (`reshape → identity-dims stretch
     * broadcast → reshape` on both `y` and `dY`, then COMPARE + WHERE). Those
     * rank-6 intermediates are why the plan kept maxpool LAST: their types bake
     * `n`/`c`/`Ho`/`Wo`, so under `grad {}`'s -1 sentinels the reshape targets are
     * meaningless, and the synthesis has no rank-6 shape witness to type them with.
     * Inverting the window per input element needs no upsample at all, so the body
     * stays rank-4 and the result type is just `x`'s.
     *
     * v1 scope: the classic non-overlapping pool — `strides == window`, zero
     * padding (PyTorch's `MaxPool2d(k)` default shape). Window-DIVISIBLE spatial
     * dims are no longer required here (that check read extents); a remainder is
     * handled correctly by the host and interpreter, and is rejected by the emitter
     * alone, whose upsample expansion genuinely needs exact tiling.
     *
     * Tie convention: full upstream to every within-window tie (the MaxRule/JAX
     * -select convention; XLA's `select_and_scatter` picks a single winner — which
     * is why the emitter expands to compare+select rather than to it).
     */
    val MaxPool2dRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            fun intPair(key: String, def: List<Int>): List<Int> =
                (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
            val k = intPair("window", emptyList())
            require(k.size == 2) { "MaxPool2dRule: primal needs `window` [kh, kw]; got $k" }
            val s = intPair("window_strides", k)
            val p = (op.attrs["padding"] as? List<*>)
                ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
                ?: listOf(listOf(0, 0), listOf(0, 0))
            // §0.4.389 — the v1 restriction that SURVIVES is literal-attr-only and
            // therefore sentinel-safe: non-overlapping windows and no padding, which
            // is what the emitter's upsample-and-mask expansion can express. The old
            // `h % k[0] == 0 && w % k[1] == 0` divisibility half is GONE from here —
            // it read extents, and under `grad {}` those are -1 (which is also what
            // made this rule reject every symbolic maxpool, since `-1 % 2 == -1`).
            // The host and interpreter handle a remainder correctly (inputs the
            // truncated last window never covered simply get no gradient), and the
            // emitter checks divisibility itself, where dims are concrete.
            require(s == k && p.all { it == listOf(0, 0) }) {
                "MaxPool2dRule v1: strides == window and zero padding required; " +
                    "got window=$k strides=$s padding=$p"
            }

            // The pooled value, recomputed in the gradient body (the mask needs it),
            // and passed to the fused op so no engine recomputes the window max —
            // and so the emitter gets a ready SSA value instead of having to emit a
            // second reduce_window.
            val y = builder.op(op.op, listOf(x), op.type, attrs = op.attrs)
            val dx = builder.op(
                OpKind.MAXPOOL2D_GRAD, listOf(upstream, x, y), x.type,
                attrs = mapOf("window" to k, "window_strides" to s, "padding" to p),
            )
            return listOf(x to dx)
        }
    }

    /**
     * d(base^exp)/d(base) = exp · base^(exp-1)
     * d(base^exp)/d(exp)  = base^exp · ln(base)
     *
     * Added in §0.4.21 (Stage B.3 follow-up) to unblock gradient-through-coarsened-loops:
     * C6/C7/C8/C9's closed forms for symbolic trip counts emit POW ops (e.g., `a^n · p`
     * for C6 with symbolic n), which Stage A SCT couldn't previously differentiate. With
     * PowRule, `grad { r -> coarsenedBgdOuterLoop(r, ...) }` pipelines work end-to-end.
     *
     * `readsPrimalOperandIndices = setOf(0, 1)`: both base and exp are dereferenced by
     * the emitted adjoint ops (base appears in base^(exp-1), in ln(base), and in base^exp;
     * exp appears in exp·X and in base^(exp-1)).
     *
     * The exp-gradient branch emits `LOG(base)` unconditionally; when the primal's exp
     * is a `DxirConst`, the framework's constant-filter in [DxirReverseTransform] skips
     * the contribution but the LOG(base) still lives as a dead op in the gradient body.
     * Stage B.3's DCE (deferred) will strip it. An alternative — gate the exp-gradient
     * emission on `op.operands[1] !is DxirConst` — is correct but couples the rule to
     * the framework's const-filter assumption; keeping the emission unconditional keeps
     * the contracts independent.
     *
     * **Scope (first cut)**: F32/F64 operands only. Integer-typed POW (e.g., `int^int`)
     * is not differentiable cleanly (ln of an integer isn't an integer); guard with a
     * runtime check on the operand dtype. Declared before [rules] (§0.4.3 object-init
     * trap).
     */
    /**
     * `d/dx(exp(x)) = exp(x)`. Stage A §0.4.22 — a unary elementwise rule that reuses
     * the forward op's structure by emitting a fresh `EXP(x)` in the gradient body
     * rather than trying to share the primal's result. Slight redundancy (one extra
     * EXP in the gradient body) — acceptable; the interpreter handles it and downstream
     * CSE would dedupe.
     *
     * `readsPrimalOperandIndices = setOf(0)`: the adjoint dereferences `x` (via EXP(x)).
     */
    val ExpRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val expX = builder.op(OpKind.EXP, listOf(x), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, expX), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(log(x)) = 1/x`. Simple: `upstream / x`. Undefined for `x ≤ 0` (produces NaN
     * or Inf) but that's a user-code correctness issue, not ours.
     */
    val LogRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val dx = builder.op(OpKind.DIV, listOf(upstream, x), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(|x|) = sign(x)` with `sign(0) = 0`. §0.4.167 — primitive for CartPole's
     * loss-clipping `(2.4 - |xt+1,0|)` and `(0.21 - |xt+1,2|)` per
     * docs/CARTPOLE_PORT_PLAN.md Phase 0a-2.
     *
     * Implementation: `STEP(x) - STEP(-x)` evaluates to +1 / -1 / 0 at x>0 / x<0 / x=0.
     * Both STEPs are typed in the operand's dtype (F32 for scalar Float), matching
     * the convention `ReluRule` uses for `STEP(x)`. Multiplied by `upstream` to give
     * the contribution. At x=0 the gradient is 0 — a discontinuous adjoint at the
     * non-differentiable point, following the standard AD convention (PyTorch / JAX
     * agree).
     */
    val AbsRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val stepPos = builder.op(OpKind.STEP, listOf(x), x.type)
            val negX = builder.op(OpKind.NEG, listOf(x), x.type)
            val stepNeg = builder.op(OpKind.STEP, listOf(negX), x.type)
            val sign = builder.op(OpKind.SUB, listOf(stepPos, stepNeg), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, sign), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * §0.4.204 — CartPole Phase 3 sixth slice. `d/dx sign(x) = 0` everywhere
     * except at the non-differentiable origin (where it's a Dirac delta).
     * Practical AD convention: gradient is identically zero. SignRule emits a
     * zero const at x's shape; [readsPrimalOperandIndices] = `emptySet()` since
     * the gradient body doesn't dereference x's value.
     *
     * For CartPole's `a = sign(tanh(...) - ε)` discretisation, this means the
     * loss gradient correctly stops at the action-discretisation boundary —
     * the policy / weight gradients flow through `tanh(...) - ε` only when
     * downstream consumers don't pass through `sign()`. Practical RL training
     * mechanisms (REINFORCE, etc.) live above this layer.
     */
    val SignRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val zero = builder.const(floatLiteralForDtype(0.0, x.type.dtype), x.type)
            return listOf(x to zero)
        }
    }

    /**
     * `d/dx(sin(x)) = cos(x)`. §0.4.166 — Trigonometric primitive for the CartPole
     * physics step. Mirrors ExpRule's "emit a fresh primal-shape op in the gradient
     * body" approach to avoid sharing the primal's result with the adjoint.
     */
    val SinRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val cosX = builder.op(OpKind.COS, listOf(x), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, cosX), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(cos(x)) = -sin(x)`. §0.4.166 — companion to SinRule. The negation is
     * folded into the multiplication via NEG(MUL(upstream, sin(x))) rather than
     * MUL(upstream, NEG(sin(x))) — both produce the same value; the former keeps
     * the SIN op's structure unchanged for downstream CSE if multiple cos calls
     * use the same x.
     */
    val CosRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val sinX = builder.op(OpKind.SIN, listOf(x), x.type)
            val product = builder.op(OpKind.MUL, listOf(upstream, sinX), upstream.type)
            val dx = builder.op(OpKind.NEG, listOf(product), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(tan(x)) = 1 + tan²(x)` (= sec²(x)). §0.4.395 — Phase C2 trig tail.
     * The tan-recompute form (rather than `1/cos²`) mirrors TanhRule: the fresh
     * TAN over the cloned primal operand CSEs with the primal's own TAN node,
     * reads no extents (sentinel-safe), and the `1` splat rides the §0.4.380
     * shape-template machinery via [splatConst].
     */
    val TanRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val tanX = builder.op(OpKind.TAN, listOf(x), x.type)
            val tanSq = builder.op(OpKind.MUL, listOf(tanX, tanX), x.type)
            val one = splatConst(builder, floatLiteralForDtype(1.0, x.type.dtype), x, x.type)
            val sec2 = builder.op(OpKind.ADD, listOf(one, tanSq), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, sec2), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(atan(x)) = 1 / (1 + x²)`. §0.4.395 — companion to TanRule. Bounded in
     * (0, 1], so numerically benign everywhere; the denominator reads only the
     * primal operand's values, never its extents.
     */
    val AtanRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val xSq = builder.op(OpKind.MUL, listOf(x, x), x.type)
            val one = splatConst(builder, floatLiteralForDtype(1.0, x.type.dtype), x, x.type)
            val denom = builder.op(OpKind.ADD, listOf(one, xSq), x.type)
            val dx = builder.op(OpKind.DIV, listOf(upstream, denom), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(lgamma(x)) = ψ(x)` (digamma). §0.4.402 — Phase C1 special functions.
     * The adjoint is a fresh DIGAMMA over the cloned primal operand — a
     * different special function, so unlike TanhRule there is nothing to
     * recompute from the primal's own value stream; it reads only the operand's
     * VALUES, never its extents, so the rule is sentinel-safe by construction.
     */
    val LgammaRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val psi = builder.op(OpKind.DIGAMMA, listOf(x), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, psi), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(digamma(x)) = ψ₁(x)` (trigamma). §0.4.402 — companion to
     * [LgammaRule]; TRIGAMMA is the internal op this rule exists to emit.
     * TRIGAMMA itself deliberately has NO VjpRule (its derivative is
     * polygamma(2), out of C1's scope) — second-order reverse through DIGAMMA
     * fails loudly with "no VJP rule registered for TRIGAMMA", pinned in
     * DxirLgammaDigammaGradTest.
     */
    val DigammaRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val psi1 = builder.op(OpKind.TRIGAMMA, listOf(x), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, psi1), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(sqrt(x)) = 1 / (2 · sqrt(x))`. For `x = 0` the adjoint is infinite (divide
     * by zero); follows IEEE semantics in the interpreter. Avoid this on primals where
     * `x` can reach zero at the differentiation point.
     */
    val SqrtRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val two = splatConst(builder, floatLiteralForDtype(2.0, x.type.dtype), x, x.type)
            val sqrtX = builder.op(OpKind.SQRT, listOf(x), x.type)
            val denom = builder.op(OpKind.MUL, listOf(two, sqrtX), x.type)
            val dx = builder.op(OpKind.DIV, listOf(upstream, denom), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(tanh(x)) = 1 - tanh(x)²`. The hyperbolic-tangent adjoint. Numerically
     * well-behaved since tanh(x) is bounded in [-1, 1], so the derivative is in [0, 1].
     */
    val TanhRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val tanhX = builder.op(OpKind.TANH, listOf(x), x.type)
            val tanhSq = builder.op(OpKind.MUL, listOf(tanhX, tanhX), x.type)
            val one = splatConst(builder, floatLiteralForDtype(1.0, x.type.dtype), x, x.type)
            val diff = builder.op(OpKind.SUB, listOf(one, tanhSq), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, diff), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * `d/dx(sigmoid(x)) = sigmoid(x) · (1 - sigmoid(x))`. The logistic-function adjoint,
     * ubiquitous in neural-network activations. Same numerical-stability caveats as
     * forward sigmoid for very large |x| (but kotlin.math.exp handles overflow).
     */
    val SigmoidRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val sigX = builder.op(OpKind.SIGMOID, listOf(x), x.type)
            val one = splatConst(builder, floatLiteralForDtype(1.0, x.type.dtype), x, x.type)
            val oneMinusSig = builder.op(OpKind.SUB, listOf(one, sigX), x.type)
            val product = builder.op(OpKind.MUL, listOf(sigX, oneMinusSig), x.type)
            val dx = builder.op(OpKind.MUL, listOf(upstream, product), upstream.type)
            return listOf(x to dx)
        }
    }

    /**
     * Convert a Double literal to the dtype-appropriate numeric value for [DxirConst].
     * Used by rules that need a dtype-polymorphic constant (e.g., the `1` in SigmoidRule's
     * `1 - sigmoid(x)`).
     */
    private fun floatLiteralForDtype(value: Double, dtype: io.tlaloc.core.DType): Any = when (dtype) {
        io.tlaloc.core.F32 -> value.toFloat()
        io.tlaloc.core.F64 -> value
        else -> error("floatLiteralForDtype: only F32/F64 supported (got $dtype)")
    }

    val PowRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val base = op.operands[0]
            val rawExp = op.operands[1]
            require(base.type.dtype == io.tlaloc.core.F32 || base.type.dtype == io.tlaloc.core.F64) {
                "PowRule: only F32/F64 base supported (got ${base.type.dtype})"
            }
            // §0.4.53 — integer exponents (I32/I64) flow through C6's closed-form
            // `a^n` when `n` is a symbolic Int trip count. Cast to the base's float
            // dtype for the adjoint arithmetic; the dExp gradient is unused by
            // callers that propagate only floating-point upstream adjoints, but we
            // still compute it for API uniformity (Int gradients round to 0).
            val exp = if (rawExp.type.dtype == io.tlaloc.core.F32 || rawExp.type.dtype == io.tlaloc.core.F64) {
                rawExp
            } else {
                require(rawExp.type.dtype == io.tlaloc.core.I32 || rawExp.type.dtype == io.tlaloc.core.I64) {
                    "PowRule: only F32/F64/I32/I64 exp supported (got ${rawExp.type.dtype})"
                }
                builder.op(OpKind.CAST, listOf(rawExp), op.type)
            }
            val oneValue: Any = when (exp.type.dtype) {
                io.tlaloc.core.F32 -> 1.0f
                io.tlaloc.core.F64 -> 1.0
                else -> error("PowRule: unsupported exp dtype ${exp.type.dtype}")
            }
            // d/dbase = upstream · exp · base^(exp-1)
            val one = splatConst(builder, oneValue, exp, exp.type)
            val expMinus1 = builder.op(OpKind.SUB, listOf(exp, one), exp.type)
            val basePowerExpMinus1 = builder.op(OpKind.POW, listOf(base, expMinus1), op.type)
            val dBaseFactor = builder.op(OpKind.MUL, listOf(exp, basePowerExpMinus1), op.type)
            val dBase = builder.op(OpKind.MUL, listOf(upstream, dBaseFactor), op.type)

            // d/dexp = upstream · base^exp · ln(base)
            val basePowExp = builder.op(OpKind.POW, listOf(base, exp), op.type)
            val logBase = builder.op(OpKind.LOG, listOf(base), base.type)
            val dExpFactor = builder.op(OpKind.MUL, listOf(basePowExp, logBase), op.type)
            val dExp = builder.op(OpKind.MUL, listOf(upstream, dExpFactor), op.type)

            return listOf(rawExp to dExp, base to dBase).let {
                // Preserve original operand-order pairing (base then exp). The dExp
                // slot must use rawExp (the original I32/I64 handle) so gradient
                // accumulation resolves to the original SSA id.
                listOf(base to dBase, rawExp to dExp)
            }
        }
    }

    /**
     * §0.4.40 — `CAST` is emitted by the FIR lowering for dtype conversions like
     * `i.toFloat()` (where `i` is the loop counter). The operand is typically an
     * Int counter or a concrete-constant post-C5-unroll; either way it's not a
     * differentiable surface. Contribution is empty — no adjoint flows back to the
     * cast's operand. If future work needs Float→Int or Float→Double conversions
     * WITH gradient flow (i.e., both sides are differentiable), this rule must be
     * generalised to emit an identity-or-reverse-cast contribution; today no
     * benchmark needs that.
     */
    val CastRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder) =
            emptyList<Pair<DxirNode, DxirNode>>()
    }

    /**
     * `y = transpose(x, perm)` where `y[I] = x[perm(I)]` (perm is the output→input
     * axis mapping the :stablehlo emitter and MatmulRule both use). The adjoint is
     * a transpose of upstream by the *inverse* permutation: `dx[J] = dy[invPerm(J)]`,
     * i.e. `dx = transpose(dy, invPerm)`. For rank-2 axis swaps perm = [1,0] is its
     * own inverse, but for general rank we compute invPerm explicitly.
     *
     * Until this rule landed, MatmulRule could *emit* TRANSPOSE in its VJP but no
     * primal TRANSPOSE op could be differentiated — so any model that pre-transposed
     * a tensor (e.g. K^T in attention) would fail at DxirReverseTransform with
     * "no VJP rule registered for TRANSPOSE".
     *
     * [readsPrimalOperandIndices] = `emptySet()`: the rule only reads the primal's
     * permutation attr (a structural concern), not the operand's value.
     */
    val TransposeRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            @Suppress("UNCHECKED_CAST")
            val perm = (op.attrs["permutation"] as? List<Int>)
                ?: error("TransposeRule: TRANSPOSE op is missing required 'permutation' attr")
            require(perm.size == x.type.rank) {
                "TransposeRule: permutation length ${perm.size} must equal operand rank ${x.type.rank}"
            }
            require(perm.toSortedSet() == (0 until perm.size).toSortedSet()) {
                "TransposeRule: permutation $perm is not a valid permutation of [0..${perm.size - 1}]"
            }
            val invPerm = IntArray(perm.size).also { inv -> perm.forEachIndexed { i, p -> inv[p] = i } }.toList()
            val dx = builder.op(
                OpKind.TRANSPOSE,
                listOf(upstream),
                x.type,
                attrs = mapOf("permutation" to invPerm),
            )
            return listOf(x to dx)
        }
    }

    /**
     * §0.4.396 — `y = flip(x, axes)` (Phase C3, DiffKT `flip`). REVERSE is a
     * permutation of the elements and an involution, so it is SELF-ADJOINT:
     * `dx = REVERSE(dy, same axes)` — flipping is linear, its permutation
     * matrix is symmetric, and applying the same flip to the upstream undoes
     * the coordinate change exactly. The `dimensions` attr is a compile-time
     * user literal (axis POSITIONS, never extents), so the rule reads nothing
     * from any shape and is sentinel-safe by construction — no runtime-extent
     * template needed, unlike SUM_TO/PAD_TO/SLICE_LIKE.
     *
     * [readsPrimalOperandIndices] = `emptySet()`: like [TransposeRule], only
     * the structural attr is read, never the operand's value.
     */
    val ReverseRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val axes = (op.attrs["dimensions"] as? List<*>)?.map { (it as Number).toInt() }
                ?: error("ReverseRule: REVERSE op is missing required 'dimensions' attr")
            require(axes.isNotEmpty() && axes.toSet().size == axes.size && axes.all { it in 0 until x.type.rank }) {
                "ReverseRule: axes $axes must be distinct and in range for rank ${x.type.rank}"
            }
            val dx = builder.op(
                OpKind.REVERSE,
                listOf(upstream),
                x.type,
                attrs = mapOf("dimensions" to axes),
            )
            return listOf(x to dx)
        }
    }

    /**
     * §0.4.41 — d(arr[idx])/d(arr) is a one-hot vector at slot [idx] with value 1;
     * scaled by [upstream], the adjoint is `SCATTER(zeros_like(arr), idx, upstream)`.
     * §0.4.111 — same shape generalises to rank-2 `arr`: d(arr[idx, :])/d(arr) is a
     * matrix of zeros with row [idx] equal to `upstream` (rank-1). The structural
     * BROADCAST → SCATTER_ADD chain works unchanged because every operand's type
     * is derived from `arr.type` or `upstream`'s type.
     *
     * The scalar `idx` operand is non-differentiable (Int), so no contribution
     * flows back to it.
     *
     * The adjoint is emitted as two ops: `BROADCAST(const(0), arr.type)` produces a
     * zero tensor matching the primal array's shape, then `SCATTER_ADD` places
     * `upstream` at slot `idx`. When the same `arr` is gathered at multiple (or
     * identical) indices across the primal, `gradAccum`'s outer ADD accumulation
     * correctly sums the one-hot contributions elementwise.
     *
     * [readsPrimalOperandIndices] = `setOf(1)` — the idx operand is NOT read for
     * chain-rule math, but its value is spliced into the emitted SCATTER as a
     * forwarded operand reference. The grad body must therefore contain cloned
     * versions of the primal idx and its dependency closure; without marking
     * index 1, `usedByAdjoint` leaves the primal CAST/const dangling and the
     * gradient function's ref-integrity check fails with "references unknown
     * node ids". `readsPrimalOperandIndices`'s semantic is "operand subgraphs that
     * must live in the gradient body", not strictly "values the rule dereferences"
     * — same as how [MulRule] marks both operands even though only the product's
     * derivative structure needs them.
     *
     * Phase A5c-3 — index 0 joins the set as well, but only per NODE: the zero base
     * is a scalar splat to `arr`'s shape, so under symbolic dims it carries `arr` as
     * a shape template (see [broadcastTo]). With concrete dims no template is
     * emitted and `arr` stays uncloned, which is why [readsPrimalOperands] refines
     * the static over-approximation.
     */
    val GatherRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun readsPrimalOperands(op: DxirOp): Set<Int> =
            if (op.operands.isNotEmpty() && needsShapeTemplate(op.operands[0].type)) setOf(0, 1) else setOf(1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            // §0.4.45 — emit SCATTER_ADD(zero_bcast, idx, upstream) instead of the
            // pre-§0.4.45 three-op chain BROADCAST + SCATTER + outer-ADD. When the
            // primal has multiple gathers on the same `arr`, DxirReverseTransform's
            // gradAccum fusion path (see `applyGradAccumFusion`) rewrites each
            // subsequent SCATTER_ADD's base operand to the previous accumulator,
            // producing a linear SCATTER_ADD chain instead of a BROADCAST/SCATTER/
            // ADD-per-gather fanout. Net: ~1 transient FloatArray alloc per gather
            // instead of ~3 (plus one initial zero-broadcast for the first gather).
            val arr = op.operands[0]
            val idx = op.operands[1]
            val scalarDType = DxirType(arr.type.dtype, emptyList())
            val zeroScalarValue: Any = when (arr.type.dtype) {
                F32 -> 0.0f
                F64 -> 0.0
                else -> error("GatherRule: unsupported arr dtype ${arr.type.dtype}")
            }
            val zeroScalar = builder.const(zeroScalarValue, scalarDType)
            // §0.4.111 — name reflects the generalised shape: a zero tensor with the
            // same rank/dims as `arr`, whether rank-1 or rank-2. Phase A5c-3 — `arr`
            // rides along as the shape template under symbolic dims.
            val zeroBase = broadcastTo(builder, zeroScalar, arr, arr.type)
            val scatterAdded = builder.op(
                OpKind.SCATTER_ADD,
                listOf(zeroBase, idx, upstream),
                arr.type,
            )
            return listOf(arr to scatterAdded)
        }
    }

    /**
     * §0.4.370 — reverse of EMBEDDING (DiffKT-parity `embedding` gradient).
     * `EMBEDDING(table, indices)` gathers `table[indices[p], :]` for each flat
     * index position `p`; its adjoint w.r.t. `table` scatter-ADDs each upstream
     * row back to the vocab slot its index selected:
     * `dTable[indices[p], :] += upstream[p, :]`, summing collisions when the same
     * vocab row is embedded at multiple positions. Expressed as the single fused
     * [OpKind.EMBEDDING_GRAD] op (indices, upstream, tableTemplate) → dTable,
     * mirroring how [GatherRule] fuses its scatter-add adjoint.
     *
     * §0.4.400 — the primal table rides along as a SHAPE-ONLY template operand
     * (the SUM_TO/PAD_TO convention): under `grad {}`'s -1 sentinel dims the
     * result type's vocab extent is unknowable at compile time, and the
     * template's runtime dims are the only sound source for the synthesis's
     * host twin `embeddingGrad(upstream, indices, tableTemplate)`. Its values
     * are never read by the interpreter or emitter.
     *
     * The `indices` operand is non-differentiable (integer), so no contribution
     * flows to it. [readsPrimalOperandIndices] = `setOf(0, 1)`: both the idx
     * subgraph (forwarded into the emitted EMBEDDING_GRAD, the same reason
     * GatherRule marks its idx operand) and the table subgraph (the shape
     * template) must be cloned into the gradient body.
     */
    val EmbeddingRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val table = op.operands[0]
            val indices = op.operands[1]
            val dTable = builder.op(
                OpKind.EMBEDDING_GRAD,
                listOf(indices, upstream, table),
                table.type,
            )
            return listOf(table to dTable)
        }
    }

    /**
     * §0.4.77 — reverse of BROADCAST. When a lower-rank input is broadcast to a
     * higher-rank output, the gradient flowing back must be SUM-reduced across
     * the inserted dims to return to the input's shape.
     *
     * §0.4.84 — generalised to axis-aware partial SUM. The `broadcast_dimensions`
     * attr on the primal BROADCAST names which output axes the input's axes map
     * to; the reverse sums over the *other* output axes (those inserted by
     * broadcasting). Falls back to the scalar-input special case (SUM over all
     * dims → scalar, no `reduction_dims` attr) when broadcast_dims is empty.
     *
     * [readsPrimalOperandIndices] = `emptySet()`: the rule only needs the
     * upstream's shape (to know what to sum) and the input's type (to shape
     * the sum). Neither requires reading the primal operand's cached value.
     */
    val BroadcastRule: VjpRule = object : VjpRule {
        // §0.4.373 — the in-place size-1 stretch adjoint dereferences the primal
        // input (operand 0) as SUM_TO's `template` operand (a shape source), so
        // the input subgraph must be cloned into the gradient body. The scalar
        // and rank-increasing SUM paths don't read it, but the declaration is
        // static per rule — conservatively including 0 keeps the runtime-extent
        // adjoint's operand live without affecting the other paths' correctness.
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val input = op.operands[0]
            @Suppress("UNCHECKED_CAST")
            val broadcastDims = (op.attrs["broadcast_dimensions"] as? List<Int>) ?: emptyList()
            if (broadcastDims.isEmpty()) {
                // Scalar input case: SUM over all dims → scalar. The interpreter's
                // bridge-SUM arm defaults to "all dims" when `reduction_dims` is
                // absent, so no attr needed here.
                require(input.type.isScalar) {
                    "BroadcastRule: empty broadcast_dimensions requires scalar input; got ${input.type}"
                }
                val contribution = builder.op(OpKind.SUM, listOf(upstream), input.type)
                return listOf(input to contribution)
            }
            // Non-scalar input: reduce over output dims NOT listed in broadcast_dims.
            val outputRank = op.type.rank
            val reduceDims = (0 until outputRank).filter { it !in broadcastDims }
            if (reduceDims.isEmpty()) {
                // Equal-rank broadcast_dimensions (identity axis map). Either a
                // TRUE identity (input shape == output shape) OR an in-place
                // size-1 stretch (`[1,C]→[N,C]`, `[N,1]→[N,C]`): the adjoint must
                // sum over exactly the axes that were size-1 in `input` and keep
                // them size-1. Which axes those are is unknowable under the -1
                // sentinel dims of `grad {}` (BroadcastRule can't tell a stretched
                // size-1 axis from a matched one), so we defer the extent to
                // runtime via SUM_TO(upstream, template=input): it reads `input`'s
                // ACTUAL runtime shape and numpy-unbroadcasts. True identity is the
                // no-op case (SUM_TO reduces nothing → passes upstream through).
                // §0.4.373.
                val contribution = builder.op(OpKind.SUM_TO, listOf(upstream, input), input.type)
                return listOf(input to contribution)
            }
            val contribution = builder.op(
                OpKind.SUM,
                listOf(upstream),
                input.type,
                attrs = mapOf("reduction_dims" to reduceDims),
            )
            return listOf(input to contribution)
        }
    }

    /**
     * §0.4.399 — the runtime-extent family closes under differentiation.
     *
     * SUM_TO's adjoint w.r.t. `value` broadcasts the upstream (shaped like the
     * TEMPLATE) back up to `value`'s shape — but that shape is a -1 sentinel
     * under `grad {}`, so the target extents cannot be baked. `BROADCAST_LIKE`
     * reads them off the primal `value` operand's ACTUAL runtime shape (value
     * becomes the shape-only template of its own adjoint — the same inversion
     * BroadcastRule performs with SUM_TO). The template operand (a pure shape
     * source, values never read) gets no contribution.
     *
     * `readsPrimalOperandIndices = setOf(0)`: the adjoint dereferences the
     * primal `value` operand as BROADCAST_LIKE's template, so its subgraph
     * must be cloned into the gradient body. Before this rule existed,
     * reverse-mode THROUGH a gradient body — reverse-over-reverse, the one
     * second-order composition forward-over-reverse (§0.4.394's hessian)
     * cannot substitute for — failed loudly with "no VJP rule registered for
     * SUM_TO".
     */
    val SumToRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val value = op.operands[0]
            val dValue = builder.op(OpKind.BROADCAST_LIKE, listOf(upstream, value), value.type)
            return listOf(value to dValue)
        }
    }

    /**
     * §0.4.399 — BROADCAST_LIKE's own VJP is the numpy unbroadcast back down to
     * `value`'s runtime shape: `SUM_TO(upstream, template=value)`. Together with
     * [SumToRule] the pair is closed — each op's adjoint is the other, so any
     * order of differentiation through them terminates.
     */
    val BroadcastLikeRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val value = op.operands[0]
            val dValue = builder.op(OpKind.SUM_TO, listOf(upstream, value), value.type)
            return listOf(value to dValue)
        }
    }

    /**
     * §0.4.399 — PAD_TO's adjoint w.r.t. `value` cuts the upstream (shaped like
     * the TEMPLATE) back down to `value`'s window: the extents are `value`'s
     * runtime shape (-1 sentinels under `grad {}`, so `SLICE_AT` reads them off
     * the primal `value` operand at execution) and the offset is the PAD_TO
     * node's own `low` attr — already a literal, carried verbatim. SLICE_LIKE
     * cannot express this (its offset is a SUM of prior templates' runtime
     * extents along one axis; this one is a multi-axis literal), which is why
     * SLICE_AT exists.
     */
    val PadToRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val value = op.operands[0]
            val dValue = builder.op(
                OpKind.SLICE_AT, listOf(upstream, value), value.type,
                attrs = mapOf("low" to op.attrs["low"]!!),
            )
            return listOf(value to dValue)
        }
    }

    /**
     * §0.4.399 — SLICE_AT's own VJP zero-pads the upstream back into `value`'s
     * window at the same literal `low`: `PAD_TO(upstream, template=value, low)`.
     * The PAD_TO ⇄ SLICE_AT pair is closed under differentiation, like
     * SUM_TO ⇄ BROADCAST_LIKE. (SLICE_LIKE stays without a rule: its window
     * offset is a runtime SUM of prior templates' extents, which no literal
     * `low` can carry — the documented remaining gap.)
     */
    val SliceAtRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val value = op.operands[0]
            val dValue = builder.op(
                OpKind.PAD_TO, listOf(upstream, value), value.type,
                attrs = mapOf("low" to op.attrs["low"]!!),
            )
            return listOf(value to dValue)
        }
    }

    private val rules: Map<OpKind, VjpRule> = mapOf(
        OpKind.ADD to AddRule,
        OpKind.SUB to SubRule,
        OpKind.MUL to MulRule,
        OpKind.DIV to DivRule,
        OpKind.NEG to NegRule,
        OpKind.ABS to AbsRule,
        OpKind.RELU to ReluRule,
        OpKind.SUM to SumRule,
        OpKind.MEAN to MeanRule,
        OpKind.MATMUL to MatmulRule,
        OpKind.CONV2D to Conv2dRule,
        OpKind.CONV_TRANSPOSE2D to ConvTranspose2dRule,
        OpKind.AVGPOOL2D to AvgPool2dRule,
        OpKind.MAXPOOL2D to MaxPool2dRule,
        OpKind.RESHAPE to ReshapeRule,
        OpKind.MAX to MaxRule,
        OpKind.MIN to MinRule,
        OpKind.SOFTMAX to SoftmaxRule,
        OpKind.CONCAT to ConcatRule,
        OpKind.SLICE to SliceRule,
        OpKind.WHERE to WhereRule,
        OpKind.COMPARE to CompareRule,
        OpKind.PAD to PadRule,
        OpKind.DOT to DotRule,
        OpKind.POW to PowRule,
        OpKind.EXP to ExpRule,
        OpKind.LOG to LogRule,
        OpKind.SIN to SinRule,
        OpKind.COS to CosRule,
        OpKind.TAN to TanRule,
        OpKind.ATAN to AtanRule,
        // §0.4.402 — Phase C1 special functions. TRIGAMMA is deliberately
        // ABSENT: no rule (d trigamma = polygamma(2), out of C1's scope).
        OpKind.LGAMMA to LgammaRule,
        OpKind.DIGAMMA to DigammaRule,
        OpKind.SIGN to SignRule,
        OpKind.SQRT to SqrtRule,
        OpKind.TANH to TanhRule,
        OpKind.SIGMOID to SigmoidRule,
        OpKind.CAST to CastRule,
        OpKind.TRANSPOSE to TransposeRule,
        OpKind.REVERSE to ReverseRule,
        OpKind.GATHER to GatherRule,
        OpKind.EMBEDDING to EmbeddingRule,
        OpKind.BROADCAST to BroadcastRule,
        // §0.4.399 — the runtime-extent family's own rules: each pair is the
        // other's adjoint, so reverse-mode composes to any order through them.
        OpKind.SUM_TO to SumToRule,
        OpKind.BROADCAST_LIKE to BroadcastLikeRule,
        OpKind.PAD_TO to PadToRule,
        OpKind.SLICE_AT to SliceAtRule,
    )

    operator fun get(kind: OpKind): VjpRule? = rules[kind]

    fun contains(kind: OpKind): Boolean = kind in rules

    val supportedKinds: Set<OpKind> get() = rules.keys
}
