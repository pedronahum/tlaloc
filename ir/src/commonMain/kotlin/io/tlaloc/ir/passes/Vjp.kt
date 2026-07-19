package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
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

    /** d(a + b)/da = 1, d(a + b)/db = 1. Contribution is upstream itself for both. */
    val AddRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder) =
            listOf(op.operands[0] to upstream, op.operands[1] to upstream)
    }

    /** d(a - b)/da = 1, d(a - b)/db = -1. */
    val SubRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val negUp = builder.op(OpKind.NEG, listOf(upstream), upstream.type)
            return listOf(op.operands[0] to upstream, op.operands[1] to negUp)
        }
    }

    /** d(a * b)/da = b, d(a * b)/db = a. */
    val MulRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0, 1)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val a0 = op.operands[0]
            val a1 = op.operands[1]
            val da = builder.op(OpKind.MUL, listOf(upstream, a1), upstream.type)
            val db = builder.op(OpKind.MUL, listOf(upstream, a0), upstream.type)
            return listOf(a0 to da, a1 to db)
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
            val aOverBB = builder.op(OpKind.DIV, listOf(a0, bb), a0.type)
            val mul = builder.op(OpKind.MUL, listOf(upstream, aOverBB), upstream.type)
            val db = builder.op(OpKind.NEG, listOf(mul), mul.type)
            return listOf(a0 to da, a1 to db)
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
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val targetType = DxirType(upstream.type.dtype, x.type.dims)
            // §0.4.366 — axis-aware arm (Phase A1): for `sum(dims)` the upstream
            // has the reduced shape; RESHAPE it to the keepdims spelling (a
            // no-op when the primal kept dims) so the interpreter's equal-rank
            // stretch BROADCAST can un-reduce it over the reduced axes. The
            // full-reduce path keeps the scalar-splat BROADCAST unchanged.
            val up = reshapeToKeepdims(op, x, upstream, builder)
            val contribution = builder.op(
                OpKind.BROADCAST,
                listOf(up),
                targetType,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
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
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
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
                val ones = builder.op(
                    OpKind.BROADCAST,
                    listOf(builder.const(one, DxirType(upstream.type.dtype, emptyList()))),
                    x.type,
                    attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
                )
                val nT = builder.op(OpKind.SUM, listOf(ones), op.type, attrs = op.attrs)
                builder.op(OpKind.DIV, listOf(upstream, nT), upstream.type)
            }
            val up = reshapeToKeepdims(op, x, scaled, builder)
            val targetType = DxirType(upstream.type.dtype, x.type.dims)
            val contribution = builder.op(
                OpKind.BROADCAST,
                listOf(up),
                targetType,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
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
            // convention — the interpreter's MUL is same-size only).
            val upA = builder.op(
                OpKind.BROADCAST, listOf(upstream), DxirType(upstream.type.dtype, a.type.dims),
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            val upB = builder.op(
                OpKind.BROADCAST, listOf(upstream), DxirType(upstream.type.dtype, b.type.dims),
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
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
            val bcast = mapOf("broadcast_dimensions" to emptyList<Int>())
            val yB = builder.op(
                OpKind.BROADCAST, listOf(reshapeToKeepdims(op, x, yRe, builder)), x.type, attrs = bcast,
            )
            val upB = builder.op(
                OpKind.BROADCAST, listOf(reshapeToKeepdims(op, x, upstream, builder)), x.type, attrs = bcast,
            )
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
            val one = builder.const(floatLiteralForDtype(1.0, x.type.dtype), x.type)
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
            val sB = builder.op(
                OpKind.BROADCAST, listOf(srow), x.type,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
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
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val dim = (op.attrs["dimension"] as? Number)?.toInt() ?: 0
            val rank = op.type.rank
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
     * §0.4.360 — `d/dx slice(x)` = the upstream zero-padded back into x's
     * shape (SLICE's adjoint IS a pad). v1 requires unit strides — the
     * strided adjoint needs interior padding, deferred until demanded.
     */
    val SliceRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            @Suppress("UNCHECKED_CAST")
            val starts = op.attrs["start_indices"] as List<Int>
            @Suppress("UNCHECKED_CAST")
            val limits = op.attrs["limit_indices"] as List<Int>
            @Suppress("UNCHECKED_CAST")
            val strides = op.attrs["strides"] as List<Int>
            require(strides.all { it == 1 }) {
                "SliceRule: strided slices are not differentiable in v1 (needs interior padding)"
            }
            val dx = builder.op(
                OpKind.PAD, listOf(upstream), x.type,
                attrs = mapOf(
                    "low" to starts,
                    "high" to x.type.dims.indices.map { x.type.dims[it] - limits[it] },
                ),
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
            val one = builder.const(floatLiteralForDtype(1.0, upstream.type.dtype), maskType)
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
     * - `dX = conv_transpose(dY, W)`: [OpKind.CONV_TRANSPOSE2D] reads W's
     *   OIHW dims under its IOHW layout — contracting over `o` and emitting
     *   `i` for free — with `window_reversal` flipping the taps spatially,
     *   `lhs_dilation = stride` undoing the stride, and padding solved
     *   numerically so the output lands exactly back on X's shape.
     * - `dW = conv(Xᵀ, dYᵀ)`: the batch↔feature transposed trick — X as
     *   `[Ci, N, H, W]` against dY as `[Co, N, Ho, Wo]` (an OIHW kernel with
     *   `o = Co`, `i = N`), `window_strides = rhs_dilation` and
     *   `rhs_dilation = stride` swapped, original padding kept; the `[Ci,
     *   Co, kh, kw]` result transposes back to OIHW.
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
            val dtype = upstream.type.dtype

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

            val (n, cIn, h, w) = x.type.dims
            val cOut = wgt.type.dims[0]
            val kh = wgt.type.dims[2]
            val kw = wgt.type.dims[3]
            val hOut = op.type.dims[2]
            val wOut = op.type.dims[3]
            val kEff = listOf((kh - 1) * d[0] + 1, (kw - 1) * d[1] + 1)

            // dX: pad low = kEff−1−p_low; pad high solved so the transposed
            // conv's output size is exactly the input's.
            val dxPad = listOf(0, 1).map { axis ->
                val inDim = x.type.dims[2 + axis]
                val dilSize = (op.type.dims[2 + axis] - 1) * s[axis] + 1
                val low = kEff[axis] - 1 - p[axis][0]
                listOf(low, inDim - 1 + kEff[axis] - dilSize - low)
            }
            val dX = builder.op(
                OpKind.CONV_TRANSPOSE2D, listOf(upstream, wgt), x.type,
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to dxPad,
                    "lhs_dilation" to s,
                    "rhs_dilation" to d,
                    "window_reversal" to listOf(true, true),
                ),
            )

            // dW: batch↔feature transposes, stride and rhs_dilation swap roles.
            val swap = mapOf("permutation" to listOf(1, 0, 2, 3))
            val xT = builder.op(
                OpKind.TRANSPOSE, listOf(x), DxirType(dtype, listOf(cIn, n, h, w)), attrs = swap,
            )
            val upT = builder.op(
                OpKind.TRANSPOSE, listOf(upstream),
                DxirType(dtype, listOf(cOut, n, hOut, wOut)), attrs = swap,
            )
            // Padding high solved so the dW conv's output is exactly [kh, kw]:
            // the primal's floor-division can leave unused input rows/cols
            // (e.g. stride 2 over height 5), which the adjoint must crop —
            // reusing the primal's padding high would overshoot.
            val dwPad = listOf(0, 1).map { axis ->
                val inDim = x.type.dims[2 + axis]
                val kSpatial = wgt.type.dims[2 + axis]
                val kEffDw = (op.type.dims[2 + axis] - 1) * s[axis] + 1
                val low = p[axis][0]
                listOf(low, (kSpatial - 1) * d[axis] + kEffDw - inDim - low)
            }
            val dWt = builder.op(
                OpKind.CONV2D, listOf(xT, upT), DxirType(dtype, listOf(cIn, cOut, kh, kw)),
                attrs = mapOf(
                    "window_strides" to d,
                    "padding" to dwPad,
                    "rhs_dilation" to s,
                ),
            )
            val dW = builder.op(OpKind.TRANSPOSE, listOf(dWt), wgt.type, attrs = swap)
            return listOf(x to dX, wgt to dW)
        }
    }

    /**
     * §0.4.363 — AVGPOOL2D adjoint: each input element receives
     * `Σ dY/(kh·kw)` over every window containing it — exactly a
     * transposed convolution of dY with a uniform `1/(kh·kw)` kernel.
     * Channels fold into the batch dim (reshape `[N,C,·,·] →
     * [N·C,1,·,·]`) so the single-channel splat kernel applies depthwise
     * without grouped-conv support; padding is solved numerically as in
     * [Conv2dRule]. Fully general strides/padding (count_include_pad —
     * the interpreter's convention).
     */
    val AvgPool2dRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val dtype = upstream.type.dtype
            fun intPair(key: String, def: List<Int>): List<Int> =
                (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
            val k = intPair("window", emptyList())
            require(k.size == 2) { "AvgPool2dRule: primal needs `window` [kh, kw]; got $k" }
            val s = intPair("window_strides", k)
            val p = (op.attrs["padding"] as? List<*>)
                ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
                ?: listOf(listOf(0, 0), listOf(0, 0))

            val (n, c, h, w) = x.type.dims
            val hOut = op.type.dims[2]
            val wOut = op.type.dims[3]

            val upR = builder.op(
                OpKind.RESHAPE, listOf(upstream), DxirType(dtype, listOf(n * c, 1, hOut, wOut)),
            )
            val kernel = builder.const(
                floatLiteralForDtype(1.0 / (k[0] * k[1]), dtype),
                DxirType(dtype, listOf(1, 1, k[0], k[1])),
            )
            val dxPad = listOf(0, 1).map { axis ->
                val inDim = x.type.dims[2 + axis]
                val dilSize = (op.type.dims[2 + axis] - 1) * s[axis] + 1
                val low = k[axis] - 1 - p[axis][0]
                listOf(low, inDim - 1 + k[axis] - dilSize - low)
            }
            val dxR = builder.op(
                OpKind.CONV_TRANSPOSE2D, listOf(upR, kernel),
                DxirType(dtype, listOf(n * c, 1, h, w)),
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "padding" to dxPad,
                    "lhs_dilation" to s,
                ),
            )
            val dX = builder.op(OpKind.RESHAPE, listOf(dxR), x.type)
            return listOf(x to dX)
        }
    }

    /**
     * §0.4.363 — MAXPOOL2D adjoint via the upsample-and-mask
     * formulation: recompute `y = maxpool(x)`, nearest-upsample y and dY
     * back to x's shape (reshape → identity-dims stretch broadcast →
     * reshape — emitter- and interpreter-legal), then
     * `dx = where(x == U(y), U(dY), 0)`.
     *
     * v1 scope: the classic non-overlapping pool — `strides == window`,
     * zero padding, spatial dims divisible by the window (PyTorch's
     * `MaxPool2d(k)` default shape). Tie convention: full upstream to
     * every within-window tie (the MaxRule/JAX-select convention; XLA's
     * select_and_scatter picks a single winner — divergence exists only
     * on exact float ties).
     */
    val MaxPool2dRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val dtype = upstream.type.dtype
            fun intPair(key: String, def: List<Int>): List<Int> =
                (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() } ?: def
            val k = intPair("window", emptyList())
            require(k.size == 2) { "MaxPool2dRule: primal needs `window` [kh, kw]; got $k" }
            val s = intPair("window_strides", k)
            val p = (op.attrs["padding"] as? List<*>)
                ?.map { row -> (row as List<*>).map { (it as Number).toInt() } }
                ?: listOf(listOf(0, 0), listOf(0, 0))
            val (n, c, h, w) = x.type.dims
            require(s == k && p.all { it == listOf(0, 0) } && h % k[0] == 0 && w % k[1] == 0) {
                "MaxPool2dRule v1: strides == window, zero padding, and window-divisible " +
                    "spatial dims required; got window=$k strides=$s padding=$p dims=${x.type.dims}"
            }
            val hOut = op.type.dims[2]
            val wOut = op.type.dims[3]

            // Nearest-upsample: [N,C,Ho,Wo] → [N,C,Ho,1,Wo,1] → stretch →
            // [N,C,Ho,kh,Wo,kw] → [N,C,H,W]. Row-major flattening makes
            // this exactly per-window replication.
            val stretch = mapOf("broadcast_dimensions" to (0 until 6).toList())
            val narrow6 = DxirType(dtype, listOf(n, c, hOut, 1, wOut, 1))
            val wide6 = DxirType(dtype, listOf(n, c, hOut, k[0], wOut, k[1]))
            fun upsample(src: DxirNode): DxirNode {
                val r6 = builder.op(OpKind.RESHAPE, listOf(src), narrow6)
                val b6 = builder.op(OpKind.BROADCAST, listOf(r6), wide6, attrs = stretch)
                return builder.op(OpKind.RESHAPE, listOf(b6), x.type)
            }

            val yRe = builder.op(op.op, listOf(x), op.type, attrs = op.attrs)
            val mask = builder.op(
                OpKind.COMPARE, listOf(x, upsample(yRe)),
                DxirType(io.tlaloc.core.Bool, x.type.dims),
                attrs = mapOf("direction" to "EQ"),
            )
            val zero = builder.const(floatLiteralForDtype(0.0, dtype), x.type)
            val dx = builder.op(OpKind.WHERE, listOf(mask, upsample(upstream), zero), x.type)
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
     * `d/dx(sqrt(x)) = 1 / (2 · sqrt(x))`. For `x = 0` the adjoint is infinite (divide
     * by zero); follows IEEE semantics in the interpreter. Avoid this on primals where
     * `x` can reach zero at the differentiation point.
     */
    val SqrtRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(0)
        override fun apply(op: DxirOp, upstream: DxirNode, builder: DxirBuilder): List<Pair<DxirNode, DxirNode>> {
            val x = op.operands[0]
            val two = builder.const(floatLiteralForDtype(2.0, x.type.dtype), x.type)
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
            val one = builder.const(floatLiteralForDtype(1.0, x.type.dtype), x.type)
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
            val one = builder.const(floatLiteralForDtype(1.0, x.type.dtype), x.type)
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
            val one = builder.const(oneValue, exp.type)
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
     */
    val GatherRule: VjpRule = object : VjpRule {
        override val readsPrimalOperandIndices: Set<Int> = setOf(1)
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
            // same rank/dims as `arr`, whether rank-1 or rank-2.
            val zeroBase = builder.op(
                OpKind.BROADCAST,
                listOf(zeroScalar),
                arr.type,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            val scatterAdded = builder.op(
                OpKind.SCATTER_ADD,
                listOf(zeroBase, idx, upstream),
                arr.type,
            )
            return listOf(arr to scatterAdded)
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
        override val readsPrimalOperandIndices: Set<Int> = emptySet()
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
                // Degenerate: input and output have the same shape (broadcast is an
                // identity). Upstream passes straight through.
                return listOf(input to upstream)
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
        OpKind.SIGN to SignRule,
        OpKind.SQRT to SqrtRule,
        OpKind.TANH to TanhRule,
        OpKind.SIGMOID to SigmoidRule,
        OpKind.CAST to CastRule,
        OpKind.TRANSPOSE to TransposeRule,
        OpKind.GATHER to GatherRule,
        OpKind.BROADCAST to BroadcastRule,
    )

    operator fun get(kind: OpKind): VjpRule? = rules[kind]

    fun contains(kind: OpKind): Boolean = kind in rules

    val supportedKinds: Set<OpKind> get() = rules.keys
}
