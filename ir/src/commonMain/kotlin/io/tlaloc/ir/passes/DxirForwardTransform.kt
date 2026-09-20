package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * §0.4.361 — **forward-mode automatic differentiation** over DXIR: the
 * JVP (Jacobian-vector product) transform, the DiffKT-parity milestone
 * (its `forward/` package) and the missing half of the AD pair.
 *
 * [apply] rewrites a primal `f(x₁..xₙ) → (y₁..yₘ)` into
 * `jvp_f(x₁..xₙ, dx₁..dxₙ) → (y₁..yₘ, dy₁..dyₘ)`: the forward values
 * plus the directional derivatives along the tangent inputs, computed
 * in **one forward pass** (dual-number semantics, materialized as a
 * second value stream through the same graph).
 *
 * # Why both modes matter
 *
 * Reverse mode ([DxirReverseTransform]) is O(1)-pass for many-inputs →
 * scalar (training losses). Forward mode is O(1)-pass for few-inputs →
 * many-outputs (sensitivities, per-sample Jacobian columns) — and the
 * two **compose**: `forward(reverse(f))` is a Hessian-vector product,
 * pinned in the §0.4.361 tests. The cross-validation identity
 * `⟨∇f(x), v⟩ = jvp_f(x, v).tangent` makes each transform the other's
 * oracle.
 *
 * # Tangent rules
 *
 * Linear ops pass tangents through their own operation; bilinear ops
 * (MUL, MATMUL, DOT) apply the product rule; elementwise nonlinearities
 * scale by f′(x) (recomputed from the primal value stream, which is
 * available in-pass — forward mode never needs the tape/recompute
 * choice reverse mode faces). Piecewise-constant ops (SIGN, STEP,
 * COMPARE) carry zero tangents; MAX/MIN route the tangent through the
 * extremum mask (full weight to every tie — the same convention as
 * [VjpRegistry]'s MaxRule). Consts carry structural-zero tangents.
 *
 * # v1 scope
 *
 * Straight-line bodies over the differentiable op set (the same
 * surface [VjpRegistry] + [DxirInterpreter] cover). Region-bearing ops
 * (IF/WHILE/COARSENED) error loudly — the coarsening story for forward
 * mode mirrors reverse-v1 history and lands separately.
 */
object DxirForwardTransform {

    fun apply(primal: DxirFunction): DxirFunction {
        primal.body.filterIsInstance<DxirOp>().firstOrNull { it.regions.isNotEmpty() }?.let {
            error(
                "DxirForwardTransform: region-bearing op ${it.op} (id=${it.id}) is out of v1 " +
                    "scope (straight-line bodies only; forward-mode coarsening lands separately)",
            )
        }

        return DxirBuilder.function("${primal.name}_jvp") {
            val valueMap = HashMap<Int, DxirNode>()
            val tangentMap = HashMap<Int, DxirNode>()

            for (p in primal.params) {
                valueMap[p.id] = param(p.name, p.type, p.sharding)
            }
            for (p in primal.params) {
                tangentMap[p.id] = param("d_${p.name}", p.type, p.sharding)
            }

            fun value(n: DxirNode): DxirNode = valueMap[n.id]
                ?: error("DxirForwardTransform: no value for id=${n.id}")

            fun zeroLike(t: DxirType): DxirNode = const(
                when (t.dtype) {
                    F32 -> 0.0f
                    F64 -> 0.0
                    else -> 0.0f
                },
                t,
            )

            fun tangent(n: DxirNode): DxirNode = tangentMap[n.id] ?: zeroLike(n.type)

            for (node in primal.body) {
                when (node) {
                    is DxirParam -> Unit
                    is DxirConst -> {
                        valueMap[node.id] = const(node.value, node.type, node.sharding)
                        // Structural zero tangent — resolved lazily by tangent().
                    }
                    is DxirOp -> {
                        val vOperands = node.operands.map { value(it) }
                        val v = op(node.op, vOperands, node.type, node.attrs, node.sharding)
                        valueMap[node.id] = v
                        tangentMap[node.id] = tangentOf(node, v, vOperands, ::tangent, this)
                    }
                    else -> error("DxirForwardTransform: unsupported body node $node")
                }
            }

            primal.returns.map { value(it) } + primal.returns.map { tangent(it) }
        }
    }

    /** Emit the tangent of [node] given its primal-value clone [v] and
     * cloned operand values [vOps]; [t] resolves operand tangents. */
    private fun tangentOf(
        node: DxirOp,
        v: DxirNode,
        vOps: List<DxirNode>,
        t: (DxirNode) -> DxirNode,
        b: DxirBuilder,
    ): DxirNode {
        val ty = node.type
        fun one(x: DxirType): DxirNode = b.const(if (x.dtype == F64) 1.0 else 1.0f, x)
        fun zero(): DxirNode = b.const(if (ty.dtype == F64) 0.0 else 0.0f, ty)

        return when (node.op) {
            // Linear: the op is its own tangent rule.
            OpKind.ADD -> b.op(OpKind.ADD, listOf(t(node.operands[0]), t(node.operands[1])), ty)
            OpKind.SUB -> b.op(OpKind.SUB, listOf(t(node.operands[0]), t(node.operands[1])), ty)
            OpKind.NEG -> b.op(OpKind.NEG, listOf(t(node.operands[0])), ty)
            OpKind.SUM, OpKind.MEAN, OpKind.RESHAPE, OpKind.TRANSPOSE,
            OpKind.SLICE, OpKind.PAD, OpKind.CONCAT, OpKind.AVGPOOL2D ->
                b.op(node.op, node.operands.map { t(it) }, ty, node.attrs)

            // BROADCAST is linear in its value (operand[0]). Phase A5c-2 lets a
            // scalar-seed splat carry a second, SHAPE-ONLY template operand (the
            // SUM_TO / PAD_TO convention) so synthesis reads the target extents off
            // a real runtime value instead of guessing them from static atoms —
            // so pass the template's primal VALUE clone, never its tangent.
            OpKind.BROADCAST ->
                if (node.operands.size == 2) {
                    b.op(OpKind.BROADCAST, listOf(t(node.operands[0]), vOps[1]), ty, node.attrs)
                } else {
                    b.op(OpKind.BROADCAST, listOf(t(node.operands[0])), ty, node.attrs)
                }

            // §0.4.373 — SUM_TO is linear in `value` (operand[0]); the template
            // (operand[1]) contributes SHAPE ONLY, so its tangent is irrelevant —
            // pass its primal VALUE clone (vOps[1]), never its tangent.
            OpKind.SUM_TO -> b.op(OpKind.SUM_TO, listOf(t(node.operands[0]), vOps[1]), ty, node.attrs)

            // §0.4.374 — PAD_TO is linear in `value` (operand[0]); the template
            // (operand[1]) contributes SHAPE ONLY, so pass its primal VALUE clone
            // (vOps[1]), never its tangent — same shape-only treatment as SUM_TO.
            OpKind.PAD_TO -> b.op(OpKind.PAD_TO, listOf(t(node.operands[0]), vOps[1]), ty, node.attrs)

            // Phase A2b — SLICE_LIKE is linear in `value` (operand[0]); EVERY
            // template (operands[1..], variadic: `thisTemplate` then the priors)
            // contributes SHAPE ONLY, so each takes its primal VALUE clone, never
            // its tangent — the same shape-only treatment as SUM_TO/PAD_TO.
            OpKind.SLICE_LIKE -> b.op(
                OpKind.SLICE_LIKE,
                listOf(t(node.operands[0])) + vOps.drop(1),
                ty,
                node.attrs,
            )

            // §0.4.363 — maxpool tangent: route dx through the argmax mask,
            // then window-sum via avgpool × kh·kw. Same v1 scope and tie
            // convention as [VjpRegistry.MaxPool2dRule] (its KDoc has the
            // derivation); ties sum, keeping the forward/reverse identity
            // exact.
            OpKind.MAXPOOL2D -> {
                val x = node.operands[0]
                val k = (node.attrs["window"] as? List<*>)?.map { (it as Number).toInt() }
                    ?: error("MAXPOOL2D tangent: missing `window` attr")
                val (n, c, h, w) = x.type.dims
                val hOut = ty.dims[2]
                val wOut = ty.dims[3]
                val stretch = mapOf("broadcast_dimensions" to (0 until 6).toList())
                val r6 = b.op(
                    OpKind.RESHAPE, listOf(v), DxirType(ty.dtype, listOf(n, c, hOut, 1, wOut, 1)),
                )
                val b6 = b.op(
                    OpKind.BROADCAST, listOf(r6),
                    DxirType(ty.dtype, listOf(n, c, hOut, k[0], wOut, k[1])), attrs = stretch,
                )
                val yUp = b.op(OpKind.RESHAPE, listOf(b6), x.type)
                val mask = b.op(
                    OpKind.COMPARE, listOf(vOps[0], yUp),
                    DxirType(io.tlaloc.core.Bool, x.type.dims),
                    attrs = mapOf("direction" to "EQ"),
                )
                val zeroX = b.const(if (x.type.dtype == F64) 0.0 else 0.0f, x.type)
                val masked = b.op(OpKind.WHERE, listOf(mask, t(x), zeroX), x.type)
                val pooled = b.op(OpKind.AVGPOOL2D, listOf(masked), ty, node.attrs)
                val scale = b.const(
                    if (ty.dtype == F64) (k[0] * k[1]).toDouble() else (k[0] * k[1]).toFloat(), ty,
                )
                b.op(OpKind.MUL, listOf(pooled, scale), ty)
            }

            // Bilinear: product rule.
            OpKind.MUL -> {
                val (a, c) = node.operands
                b.op(
                    OpKind.ADD,
                    listOf(
                        b.op(OpKind.MUL, listOf(t(a), vOps[1]), ty),
                        b.op(OpKind.MUL, listOf(vOps[0], t(c)), ty),
                    ),
                    ty,
                )
            }
            OpKind.DIV -> {
                // d(a/b) = (da − y·db) / b
                val (a, c) = node.operands
                val num = b.op(
                    OpKind.SUB,
                    listOf(t(a), b.op(OpKind.MUL, listOf(v, t(c)), ty)),
                    ty,
                )
                b.op(OpKind.DIV, listOf(num, vOps[1]), ty)
            }
            OpKind.MATMUL, OpKind.DOT, OpKind.CONV2D, OpKind.CONV_TRANSPOSE2D -> {
                val (a, c) = node.operands
                b.op(
                    OpKind.ADD,
                    listOf(
                        b.op(node.op, listOf(t(a), vOps[1]), ty, node.attrs),
                        b.op(node.op, listOf(vOps[0], t(c)), ty, node.attrs),
                    ),
                    ty,
                )
            }
            OpKind.POW -> {
                // d(a^c) = c·a^(c−1)·da + y·ln(a)·db
                val (a, c) = node.operands
                val cMinus1 = b.op(OpKind.SUB, listOf(vOps[1], one(vOps[1].type)), vOps[1].type)
                val aPow = b.op(OpKind.POW, listOf(vOps[0], cMinus1), ty)
                val term1 = b.op(
                    OpKind.MUL,
                    listOf(b.op(OpKind.MUL, listOf(vOps[1], aPow), ty), t(a)),
                    ty,
                )
                val lnA = b.op(OpKind.LOG, listOf(vOps[0]), vOps[0].type)
                val term2 = b.op(
                    OpKind.MUL,
                    listOf(b.op(OpKind.MUL, listOf(v, lnA), ty), t(c)),
                    ty,
                )
                b.op(OpKind.ADD, listOf(term1, term2), ty)
            }

            // Elementwise nonlinearities: f′(x) ⊙ dx, f′ from the in-pass value stream.
            OpKind.EXP -> b.op(OpKind.MUL, listOf(v, t(node.operands[0])), ty)
            OpKind.LOG -> b.op(OpKind.DIV, listOf(t(node.operands[0]), vOps[0]), ty)
            OpKind.SQRT -> {
                val twoY = b.op(OpKind.ADD, listOf(v, v), ty)
                b.op(OpKind.DIV, listOf(t(node.operands[0]), twoY), ty)
            }
            OpKind.RSQRT -> {
                // y = x^-1/2; dy = −½·y³·dx
                val y2 = b.op(OpKind.MUL, listOf(v, v), ty)
                val y3 = b.op(OpKind.MUL, listOf(y2, v), ty)
                val halfNeg = b.const(if (ty.dtype == F64) -0.5 else -0.5f, ty)
                b.op(OpKind.MUL, listOf(b.op(OpKind.MUL, listOf(halfNeg, y3), ty), t(node.operands[0])), ty)
            }
            OpKind.TANH -> {
                val y2 = b.op(OpKind.MUL, listOf(v, v), ty)
                val d = b.op(OpKind.SUB, listOf(one(ty), y2), ty)
                b.op(OpKind.MUL, listOf(d, t(node.operands[0])), ty)
            }
            OpKind.SIGMOID -> {
                val d = b.op(OpKind.MUL, listOf(v, b.op(OpKind.SUB, listOf(one(ty), v), ty)), ty)
                b.op(OpKind.MUL, listOf(d, t(node.operands[0])), ty)
            }
            OpKind.RELU -> {
                val mask = b.op(OpKind.STEP, listOf(vOps[0]), ty)
                b.op(OpKind.MUL, listOf(mask, t(node.operands[0])), ty)
            }
            OpKind.SIN -> b.op(
                OpKind.MUL, listOf(b.op(OpKind.COS, listOf(vOps[0]), ty), t(node.operands[0])), ty,
            )
            OpKind.COS -> b.op(
                OpKind.NEG,
                listOf(b.op(OpKind.MUL, listOf(b.op(OpKind.SIN, listOf(vOps[0]), ty), t(node.operands[0])), ty)),
                ty,
            )
            // §0.4.395 — Phase C2 trig tails. TAN reads its own value stream
            // (y = tan(x), dy = (1 + y²)·dx — the TanhRule-style recompute-free
            // form); ATAN reads the operand (dy = dx / (1 + x²)).
            OpKind.TAN -> {
                val y2 = b.op(OpKind.MUL, listOf(v, v), ty)
                val sec2 = b.op(OpKind.ADD, listOf(one(ty), y2), ty)
                b.op(OpKind.MUL, listOf(sec2, t(node.operands[0])), ty)
            }
            OpKind.ATAN -> {
                val x2 = b.op(OpKind.MUL, listOf(vOps[0], vOps[0]), ty)
                val denom = b.op(OpKind.ADD, listOf(one(ty), x2), ty)
                b.op(OpKind.DIV, listOf(t(node.operands[0]), denom), ty)
            }
            OpKind.ABS -> b.op(
                OpKind.MUL, listOf(b.op(OpKind.SIGN, listOf(vOps[0]), ty), t(node.operands[0])), ty,
            )
            OpKind.SOFTMAX -> {
                // dy = y ⊙ (dx − Σ_axis(y ⊙ dx))  (the VJP formula's forward twin).
                val rank = ty.rank
                val axisRaw = (node.attrs["axis"] as? Number)?.toInt() ?: (rank - 1)
                val axis = if (axisRaw < 0) axisRaw + rank else axisRaw
                val dx = t(node.operands[0])
                val yd = b.op(OpKind.MUL, listOf(v, dx), ty)
                val kd = DxirType(ty.dtype, ty.dims.mapIndexed { i, d -> if (i == axis) 1 else d })
                val s = b.op(OpKind.SUM, listOf(yd), kd, attrs = mapOf("reduction_dims" to listOf(axis)))
                val sB = b.op(
                    OpKind.BROADCAST, listOf(s), ty,
                    attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
                )
                b.op(OpKind.MUL, listOf(v, b.op(OpKind.SUB, listOf(dx, sB), ty)), ty)
            }
            OpKind.MAX, OpKind.MIN -> {
                // Route the tangent through the extremum mask, then reduce.
                // Ties get full weight (the VJP MaxRule convention).
                // §0.4.366 — axis reductions that squeeze the reduced axes
                // RESHAPE the primal output to the keepdims spelling first, so
                // the interpreter's equal-rank stretch BROADCAST arm applies
                // (mirrors VjpRegistry.reshapeToKeepdims).
                val x = node.operands[0]
                val bcast = mapOf("broadcast_dimensions" to emptyList<Int>())
                val rd = (node.attrs["reduction_dims"] as? List<*>)
                    ?.map { (it as Number).toInt() }
                    ?.takeIf { it.isNotEmpty() }
                val vK = if (rd == null || v.type.isScalar || v.type.rank == x.type.rank) v else b.op(
                    OpKind.RESHAPE, listOf(v),
                    DxirType(ty.dtype, x.type.dims.mapIndexed { i, d -> if (i in rd) 1 else d }),
                )
                val yB = b.op(OpKind.BROADCAST, listOf(vK), x.type, attrs = bcast)
                val diff = if (node.op == OpKind.MAX) {
                    b.op(OpKind.SUB, listOf(yB, vOps[0]), x.type)
                } else {
                    b.op(OpKind.SUB, listOf(vOps[0], yB), x.type)
                }
                val sgn = b.op(OpKind.SIGN, listOf(diff), x.type)
                val mask = b.op(OpKind.SUB, listOf(one(x.type), sgn), x.type)
                val masked = b.op(OpKind.MUL, listOf(mask, t(x)), x.type)
                b.op(OpKind.SUM, listOf(masked), ty, attrs = node.attrs)
            }
            OpKind.WHERE -> b.op(
                OpKind.WHERE, listOf(vOps[0], t(node.operands[1]), t(node.operands[2])), ty,
            )
            OpKind.GATHER -> b.op(OpKind.GATHER, listOf(t(node.operands[0]), vOps[1]), ty, node.attrs)
            // §0.4.370 — EMBEDDING is linear in the table (a gather along the
            // vocab axis): dY = EMBEDDING(dTable, indices). Indices carry no
            // tangent (integer); the primal index clone vOps[1] rides through.
            OpKind.EMBEDDING -> b.op(OpKind.EMBEDDING, listOf(t(node.operands[0]), vOps[1]), ty, node.attrs)
            OpKind.CAST -> b.op(OpKind.CAST, listOf(t(node.operands[0])), ty)

            // Piecewise-constant / boolean: zero tangent.
            OpKind.SIGN, OpKind.STEP, OpKind.COMPARE, OpKind.NOT, OpKind.LAND -> zero()

            else -> error(
                "DxirForwardTransform: no tangent rule for ${node.op} " +
                    "(widen the transform, don't guess)",
            )
        }
    }
}
