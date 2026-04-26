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
            val contribution = builder.op(
                OpKind.BROADCAST,
                listOf(upstream),
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
            val n = if (x.type.dims.isEmpty()) 1 else x.type.dims.fold(1) { acc, d -> acc * d }
            val invN: Any = when (upstream.type.dtype) {
                F32 -> 1.0f / n
                F64 -> 1.0 / n
                else -> error("MeanRule: unsupported dtype ${upstream.type.dtype}")
            }
            val invNConst = builder.const(invN, upstream.type)
            val scaled = builder.op(OpKind.MUL, listOf(upstream, invNConst), upstream.type)
            val targetType = DxirType(upstream.type.dtype, x.type.dims)
            val contribution = builder.op(
                OpKind.BROADCAST,
                listOf(scaled),
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
        OpKind.POW to PowRule,
        OpKind.EXP to ExpRule,
        OpKind.LOG to LogRule,
        OpKind.SIN to SinRule,
        OpKind.COS to CosRule,
        OpKind.SQRT to SqrtRule,
        OpKind.TANH to TanhRule,
        OpKind.SIGMOID to SigmoidRule,
        OpKind.CAST to CastRule,
        OpKind.GATHER to GatherRule,
        OpKind.BROADCAST to BroadcastRule,
    )

    operator fun get(kind: OpKind): VjpRule? = rules[kind]

    fun contains(kind: OpKind): Boolean = kind in rules

    val supportedKinds: Set<OpKind> get() = rules.keys
}
