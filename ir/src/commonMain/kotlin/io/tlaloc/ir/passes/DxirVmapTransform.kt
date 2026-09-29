package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind

/**
 * An op the vmap transform has no batching rule for. [kind] is the op kind, or null
 * when the refusal is about the function rather than one op.
 */
@ExperimentalTlalocApi
class VmapUnsupportedException(val kind: OpKind?, detail: String) : RuntimeException(
    if (kind != null) "vmap has no batching rule for $kind: $detail" else "vmap: $detail",
)

/**
 * **Batching** over DXIR: rewrites a function over one example into the same
 * function over a batch, the batch axis leading every batched value.
 *
 * [apply] takes a per-example `f(x₁..xₙ) → (y₁..yₘ)` and a flag per parameter. A
 * batched parameter's type gains a leading axis of [batchSize] (`-1` for
 * plugin-lowered functions, whose extents are known only at run time); an unbatched
 * one keeps its type and is shared by every example. Every return is batched.
 *
 * Each node is tracked as a value plus a batched flag. An op whose operands are all
 * unbatched is copied unchanged, so work that does not depend on the batch is done
 * once. An op with at least one batched operand is rewritten by its batching rule
 * (the `when` in [Walk.batchOp]): axis attributes shift by one, and an unbatched
 * operand that has to be batched is materialized (see [Walk.materialize]). An op
 * kind without a rule throws [VmapUnsupportedException]; there is no fallback that
 * loops over examples.
 *
 * The batch axis is unnamed in the DXIR types, and batched types carry no axis names:
 * the emitter infers a `MATMUL`'s contraction from shared axis names, and a batch
 * name on both operands would be taken for one. The name lives in the Kotlin type.
 *
 * Design: docs/design/vmap.md.
 */
@ExperimentalTlalocApi
object DxirVmapTransform {

    fun apply(fn: DxirFunction, batched: List<Boolean>, batchSize: Int): DxirFunction {
        if (batched.size != fn.params.size) {
            throw VmapUnsupportedException(
                null,
                "${batched.size} batching flags for ${fn.params.size} parameters of '${fn.name}'",
            )
        }
        if (batched.none { it }) {
            throw VmapUnsupportedException(null, "no parameter of '${fn.name}' is batched")
        }
        require(batchSize == -1 || batchSize >= 1) { "vmap: batch size must be -1 or at least 1, got $batchSize" }
        return DxirBuilder.function("${fn.name}_vmap") {
            for (m in fn.meshes) declareMesh(m)
            val w = Walk(this, batchSize)
            for ((i, p) in fn.params.withIndex()) {
                if (batched[i]) {
                    val np = param(p.name, w.bt(p.type))
                    w.bind(p, np, true)
                    if (w.firstBatchedParam == null) w.firstBatchedParam = np
                } else {
                    w.bind(p, param(p.name, p.type, p.sharding), false)
                }
            }
            for (node in fn.body) w.process(node)
            fn.returns.map { w.batchedValue(it) }
        }
    }

    private class Walk(val b: DxirBuilder, val batchSize: Int) {
        /** New value per source id (index 0 for single-result nodes). */
        private val values = HashMap<Int, List<DxirNode>>()
        private val batchedFlags = HashMap<Int, List<Boolean>>()
        var firstBatchedParam: DxirNode? = null
        private val zeroVectors = HashMap<DType, DxirNode>()

        fun bind(src: DxirNode, v: DxirNode, isBatched: Boolean) {
            values[src.id] = listOf(v)
            batchedFlags[src.id] = listOf(isBatched)
        }

        private fun index(n: DxirNode): Int = (n as? DxirOpResult)?.index ?: 0

        fun value(n: DxirNode): DxirNode =
            values[n.id]?.get(index(n)) ?: error("DxirVmapTransform: no value for id=${n.id}")

        fun isBatched(n: DxirNode): Boolean =
            batchedFlags[n.id]?.get(index(n)) ?: error("DxirVmapTransform: no batching flag for id=${n.id}")

        /** Per-example type [t] with the batch axis in front, names dropped. */
        fun bt(t: DxirType): DxirType = DxirType(t.dtype, listOf(batchSize) + t.dims)

        private val concrete: Boolean get() = batchSize >= 0

        private fun allConcrete(t: DxirType) = concrete && t.dims.all { it >= 0 }

        /** The value of [n] as a batched node, materializing it if it is not batched. */
        fun batchedValue(n: DxirNode): DxirNode = if (isBatched(n)) value(n) else materialize(value(n), n.type)

        /**
         * An unbatched value [u] of per-example type [t] repeated along a new leading
         * batch axis. With concrete extents this is a `BROADCAST`. With `-1` extents
         * synthesized code learns the batch size only from a batched runtime value, so
         * [u] is added to a batched zero vector reshaped to `[B, 1, ..., 1]`, which the
         * right-aligned broadcasting of `ADD` expands to `[B] + t` (a scalar, a primitive
         * in synthesized code, is splatted against the zero vector instead).
         */
        fun materialize(u: DxirNode, t: DxirType): DxirNode {
            if (allConcrete(t)) {
                return b.op(
                    OpKind.BROADCAST, listOf(u), bt(t),
                    attrs = mapOf("broadcast_dimensions" to (1..t.rank).toList()),
                )
            }
            if (t.dtype == Bool) {
                throw VmapUnsupportedException(
                    null, "a boolean value that does not depend on the batch cannot be repeated along it " +
                        "when the extents are known only at run time",
                )
            }
            val z = zeroVector(t.dtype)
            // A scalar is a primitive in synthesized code: splat it against the zero vector.
            if (t.rank == 0) {
                return b.op(OpKind.BROADCAST, listOf(u, z), bt(t), attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
            }
            val zr = b.op(OpKind.RESHAPE, listOf(z), DxirType(t.dtype, listOf(batchSize) + List(t.rank) { 1 }))
            return b.op(OpKind.ADD, listOf(zr, u), bt(t))
        }

        /** Zeros of shape `[B]` in [dtype], built from the first batched parameter. */
        private fun zeroVector(dtype: DType): DxirNode = zeroVectors.getOrPut(dtype) {
            val p = firstBatchedParam ?: error("DxirVmapTransform: no batched parameter")
            val zeros = b.op(
                OpKind.BROADCAST, listOf(zero(dtype), p), DxirType(dtype, p.type.dims),
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            if (p.type.rank == 1) zeros else b.op(
                OpKind.SUM, listOf(zeros), DxirType(dtype, listOf(batchSize)),
                attrs = mapOf("reduction_dims" to (1 until p.type.rank).toList()),
            )
        }

        private fun floatMask(op: DxirOp): Boolean =
            op.operands.any { isBatched(it) && value(it).type.dtype != Bool }

        /** NOT / LAND of a batched float mask (the predicate form under `-1` extents): 1 − m and m · m'. */
        private fun maskLogic(op: DxirOp): DxirNode {
            val masks = op.operands.map { batchedValue(it) }
            if (masks.any { it.type.dtype == Bool }) {
                throw VmapUnsupportedException(op.op, "a boolean operand combined with a per-example condition")
            }
            val mt = bt(DxirType(masks[0].type.dtype, op.type.dims))
            return if (op.op == OpKind.NOT) {
                b.op(OpKind.SUB, listOf(splat(zeroLikeOne(mt.dtype), mt, masks[0]), masks[0]), mt)
            } else {
                b.op(OpKind.MUL, masks, mt)
            }
        }

        private fun zeroLikeOne(dtype: DType): DxirNode = b.const(if (dtype == F64) 1.0 else 1.0f, DxirType(dtype, emptyList()))

        private fun zero(dtype: DType): DxirNode = b.const(
            when (dtype) {
                F64 -> 0.0
                I32 -> 0
                I64 -> 0L
                else -> 0.0f
            },
            DxirType(dtype, emptyList()),
        )

        /**
         * A constant of a tensor type with `-1` extents: a splat whose extents synthesized
         * code infers by matching its axes against the parameters, which batching makes
         * ambiguous. Such a constant is re-emitted as a scalar splatted against a batched
         * operand instead.
         */
        private fun isShapedSplat(n: DxirNode): Boolean =
            n is DxirConst && n.type.rank > 0 && n.type.dims.any { it < 0 } && n.value is Number

        private fun scalarOf(n: DxirNode): DxirNode =
            if (n is DxirConst && n.type.rank > 0) b.const(n.value, DxirType(n.type.dtype, emptyList())) else value(n)

        /** Scalar [u] repeated to batched type [ty], against [template] (batched, of type [ty]) when extents are `-1`. */
        private fun splat(u: DxirNode, ty: DxirType, template: DxirNode): DxirNode =
            if (ty.dims.all { it >= 0 }) {
                b.op(OpKind.BROADCAST, listOf(u), ty, attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
            } else {
                b.op(OpKind.BROADCAST, listOf(u, template), ty, attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
            }

        /** Batched [v] of per-example type [t] reshaped to per-example rank [rank] by unit axes after the batch axis. */
        private fun padRankBatched(v: DxirNode, t: DxirType, rank: Int): DxirNode =
            if (t.rank >= rank) v
            else b.op(OpKind.RESHAPE, listOf(v), bt(DxirType(t.dtype, List(rank - t.rank) { 1 } + t.dims)))

        fun process(node: DxirNode) {
            when (node) {
                is DxirParam -> Unit
                is DxirConst -> bind(node, b.const(node.value, node.type, node.sharding), false)
                is DxirOp -> processOp(node)
                else -> throw VmapUnsupportedException(
                    null, "node ${node::class.simpleName} (id=${node.id}) is not an op, constant or parameter",
                )
            }
        }

        private fun processOp(op: DxirOp) {
            if (op.op == OpKind.IF) return processIf(op)
            if (op.regions.isNotEmpty()) {
                throw VmapUnsupportedException(op.op, "region-bearing ops other than IF are not batched")
            }
            if (op.isMultiResult) {
                throw VmapUnsupportedException(op.op, "multi-result ops other than IF are not batched")
            }
            if (op.operands.none { isBatched(it) }) {
                val v = b.op(op.op, op.operands.map { value(it) }, op.type, op.attrs, op.sharding)
                bind(op, v, false)
                return
            }
            bind(op, batchOp(op), true)
        }

        private fun intList(op: DxirOp, key: String): List<Int>? =
            (op.attrs[key] as? List<*>)?.map { (it as Number).toInt() }

        private fun withAttrs(op: DxirOp, vararg changes: Pair<String, Any>): Map<String, Any> =
            op.attrs + changes.toMap()

        /** The rule table: [op] has at least one batched operand. */
        private fun batchOp(op: DxirOp): DxirNode {
            val ty = bt(op.type)
            return when (op.op) {
                // A batched predicate (a per-example condition) under `-1` extents keeps its
                // operand's float dtype: at run time a boolean tensor is a 0/1 float mask, and
                // synthesis types a mask tensor, not a Bool one.
                OpKind.STEP -> {
                    val x = value(op.operands[0])
                    val t = if (op.type.dtype == Bool && !allConcrete(op.type)) {
                        bt(DxirType(x.type.dtype, op.type.dims))
                    } else ty
                    b.op(OpKind.STEP, listOf(x), t, op.attrs)
                }

                // NOT / LAND of a batched float mask (the predicate form under `-1` extents above):
                // 1 − m and m · m', in the mask's dtype.
                OpKind.NOT if floatMask(op) -> maskLogic(op)
                OpKind.LAND if floatMask(op) -> maskLogic(op)

                OpKind.NEG, OpKind.ABS, OpKind.EXP, OpKind.LOG, OpKind.SQRT, OpKind.RSQRT, OpKind.TANH,
                OpKind.SIGMOID, OpKind.RELU, OpKind.GELU, OpKind.SILU, OpKind.SIN, OpKind.COS,
                OpKind.TAN, OpKind.ATAN, OpKind.LGAMMA, OpKind.DIGAMMA, OpKind.TRIGAMMA, OpKind.POLYGAMMA,
                OpKind.SIGN, OpKind.NOT, OpKind.CAST,
                -> b.op(op.op, listOf(value(op.operands[0])), ty, op.attrs)

                OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.LAND -> {
                    val r = op.type.rank
                    // A batched operand of the result's full per-example shape: the template an
                    // unbatched scalar is splatted against (a batched tensor meets a scalar only
                    // as a tensor in synthesized code).
                    val full = op.operands.firstOrNull { isBatched(it) && it.type.dims == op.type.dims }
                    val operands = op.operands.map { o ->
                        when {
                            isBatched(o) -> padRankBatched(value(o), o.type, r)
                            full != null && (o.type.rank == 0 || isShapedSplat(o)) ->
                                splat(scalarOf(o), ty, value(full))
                            else -> value(o)
                        }
                    }
                    b.op(op.op, operands, ty, op.attrs)
                }

                OpKind.COMPARE -> {
                    for (o in op.operands) {
                        if (o.type.dims != op.type.dims) {
                            throw VmapUnsupportedException(
                                op.op, "operands of different shapes (${op.operands.map { it.type }}) — " +
                                    "only same-shape operands are batched",
                            )
                        }
                    }
                    val operands = op.operands.map { batchedValue(it) }
                    val t = if (!allConcrete(op.type)) bt(DxirType(operands[0].type.dtype, op.type.dims)) else ty
                    b.op(OpKind.COMPARE, operands, t, op.attrs)
                }

                OpKind.POW, OpKind.WHERE -> {
                    for (o in op.operands) {
                        if (o.type.dims != op.type.dims) {
                            throw VmapUnsupportedException(
                                op.op, "operands of different shapes (${op.operands.map { it.type }}) — " +
                                    "only same-shape operands are batched",
                            )
                        }
                    }
                    val full = op.operands.firstOrNull { isBatched(it) && it.type.dims == op.type.dims }
                    b.op(
                        op.op,
                        op.operands.map {
                            if (full != null && isShapedSplat(it)) {
                                splat(scalarOf(it), bt(it.type), value(full))
                            } else {
                                batchedValue(it)
                            }
                        },
                        ty, op.attrs,
                    )
                }

                OpKind.BROADCAST -> batchBroadcast(op)

                OpKind.SUM, OpKind.MEAN, OpKind.MAX, OpKind.MIN -> {
                    val x = op.operands[0]
                    val r = x.type.rank
                    if (r == 0) return value(x)
                    val dims = intList(op, "reduction_dims").orEmpty()
                        .map { if (it < 0) it + r else it }
                        .ifEmpty { (0 until r).toList() }
                    b.op(op.op, listOf(value(x)), ty, withAttrs(op, "reduction_dims" to dims.map { it + 1 }))
                }

                OpKind.SOFTMAX, OpKind.LOGSUMEXP, OpKind.ARGMAX -> {
                    val r = op.operands[0].type.rank
                    if (r == 0) {
                        // Its axis would land on the batch axis and mix the examples.
                        throw VmapUnsupportedException(op.op, "a per-example scalar operand (it has no axis of its own)")
                    }
                    val raw = (op.attrs["axis"] as? Number)?.toInt() ?: (r - 1)
                    val axis = if (raw < 0) raw + r else raw
                    b.op(op.op, listOf(value(op.operands[0])), ty, withAttrs(op, "axis" to axis + 1))
                }

                OpKind.TRANSPOSE -> {
                    val r = op.operands[0].type.rank
                    val perm = intList(op, "permutation") ?: (0 until r).reversed().toList()
                    b.op(
                        OpKind.TRANSPOSE, listOf(value(op.operands[0])), ty,
                        withAttrs(op, "permutation" to listOf(0) + perm.map { it + 1 }),
                    )
                }

                // `leading_kept` counts the leading axes the reshape leaves alone: under `-1`
                // extents a batched flatten ([B, -1, -1] -> [B, -1]) cannot be told from
                // other reshapes of the same ranks without it. Nested vmap adds one each.
                // A reshape that merges axes (a shared-rhs MATMUL's adjoint) merges the same
                // axes, one further along.
                OpKind.RESHAPE -> {
                    val shift = mutableListOf<Pair<String, Any>>(
                        "leading_kept" to ((op.attrs["leading_kept"] as? Number)?.toInt() ?: 0) + 1,
                    )
                    if ("merge_leading" in op.attrs) {
                        shift += "merge_from" to ((op.attrs["merge_from"] as? Number)?.toInt() ?: 0) + 1
                    }
                    b.op(OpKind.RESHAPE, listOf(value(op.operands[0])), ty, withAttrs(op, *shift.toTypedArray()))
                }

                OpKind.REVERSE -> {
                    val dims = intList(op, "dimensions")
                        ?: throw VmapUnsupportedException(op.op, "missing `dimensions` attribute")
                    b.op(OpKind.REVERSE, listOf(value(op.operands[0])), ty, withAttrs(op, "dimensions" to dims.map { it + 1 }))
                }

                OpKind.SLICE -> {
                    val starts = intList(op, "start_indices")
                        ?: throw VmapUnsupportedException(op.op, "missing `start_indices` attribute")
                    val limits = intList(op, "limit_indices")
                        ?: throw VmapUnsupportedException(op.op, "missing `limit_indices` attribute")
                    val strides = intList(op, "strides") ?: List(starts.size) { 1 }
                    val changes = mutableListOf<Pair<String, Any>>(
                        "start_indices" to listOf(0) + starts,
                        "limit_indices" to listOf(batchSize) + limits,
                        "strides" to listOf(1) + strides,
                    )
                    (op.attrs["slice_axis"] as? Number)?.let { changes += "slice_axis" to it.toInt() + 1 }
                    b.op(OpKind.SLICE, listOf(value(op.operands[0])), ty, withAttrs(op, *changes.toTypedArray()))
                }

                OpKind.PAD -> {
                    val changes = listOf("low", "high", "interior").mapNotNull { k ->
                        intList(op, k)?.let { k to (listOf(0) + it) }
                    }
                    b.op(OpKind.PAD, listOf(value(op.operands[0])), ty, withAttrs(op, *changes.toTypedArray()))
                }

                OpKind.CONCAT -> {
                    val dim = (op.attrs["dimension"] as? Number)?.toInt() ?: 0
                    b.op(OpKind.CONCAT, op.operands.map { batchedValue(it) }, ty, withAttrs(op, "dimension" to dim + 1))
                }

                OpKind.MATMUL -> batchMatmul(op)

                OpKind.DOT -> {
                    val (x, y) = op.operands
                    val prodType = DxirType(op.type.dtype, listOf(batchSize) + x.type.dims)
                    val prod = b.op(OpKind.MUL, listOf(value(x), value(y)), prodType)
                    b.op(OpKind.SUM, listOf(prod), ty, mapOf("reduction_dims" to listOf(1)))
                }

                // Runtime-extent ops (they appear in gradients). Value and templates are batched
                // together; SUM_TO and BROADCAST_LIKE align right, so a lower-rank side gets unit
                // axes after the batch axis to keep the batch axes aligned.
                OpKind.SUM_TO -> {
                    val (v, t) = op.operands
                    val vB = batchedValue(v)
                    val tB = batchedValue(t)
                    if (t.type.rank == v.type.rank) {
                        b.op(OpKind.SUM_TO, listOf(vB, tB), ty, op.attrs)
                    } else {
                        val gap = v.type.rank - t.type.rank
                        val tPad = b.op(OpKind.RESHAPE, listOf(tB), bt(DxirType(t.type.dtype, List(gap) { 1 } + t.type.dims)))
                        val s = b.op(OpKind.SUM_TO, listOf(vB, tPad), tPad.type, op.attrs)
                        b.op(OpKind.RESHAPE, listOf(s), ty)
                    }
                }

                OpKind.BROADCAST_LIKE -> {
                    val (v, t) = op.operands
                    val tB = batchedValue(t)
                    val vIn = when {
                        !isBatched(v) -> value(v)
                        v.type.rank == t.type.rank -> value(v)
                        else -> b.op(
                            OpKind.RESHAPE, listOf(value(v)),
                            bt(DxirType(v.type.dtype, List(t.type.rank - v.type.rank) { 1 } + v.type.dims)),
                        )
                    }
                    b.op(OpKind.BROADCAST_LIKE, listOf(vIn, tB), ty, op.attrs)
                }

                OpKind.PAD_TO, OpKind.SLICE_AT -> {
                    val low = intList(op, "low") ?: throw VmapUnsupportedException(op.op, "missing `low` attribute")
                    b.op(op.op, op.operands.map { batchedValue(it) }, ty, withAttrs(op, "low" to listOf(0) + low))
                }

                OpKind.SLICE_LIKE, OpKind.PAD_LIKE -> {
                    val axis = (op.attrs["axis"] as? Number)?.toInt()
                        ?: throw VmapUnsupportedException(op.op, "missing `axis` attribute")
                    b.op(op.op, op.operands.map { batchedValue(it) }, ty, withAttrs(op, "axis" to axis + 1))
                }

                OpKind.CHECK_SHAPE_LIKE, OpKind.ZEROS_LIKE ->
                    b.op(op.op, op.operands.map { batchedValue(it) }, ty, op.attrs)

                OpKind.EMBEDDING -> {
                    val (table, idx) = op.operands
                    if (isBatched(table)) {
                        throw VmapUnsupportedException(op.op, "a batched embedding table (only the indices may be batched)")
                    }
                    b.op(OpKind.EMBEDDING, listOf(value(table), value(idx)), ty, op.attrs)
                }

                // `x[i]` with a constant i: the slice at position i of the example's first axis,
                // which is axis 1 of the batch, then that axis dropped.
                OpKind.GATHER -> {
                    val (arr, idx) = op.operands
                    val i = constIndex(idx)
                    if (!isBatched(arr) || isBatched(idx) || i == null) {
                        throw VmapUnsupportedException(
                            op.op, "only a batched array indexed at a compile-time constant position is batched",
                        )
                    }
                    val dims = arr.type.dims
                    checkIndex(op, i, dims)
                    val sliced = b.op(
                        OpKind.SLICE, listOf(value(arr)), bt(DxirType(arr.type.dtype, listOf(1) + dims.drop(1))),
                        mapOf(
                            "start_indices" to listOf(0, i) + List(dims.size - 1) { 0 },
                            "limit_indices" to listOf(batchSize, i + 1) + dims.drop(1),
                            "strides" to List(dims.size + 1) { 1 },
                            "slice_axis" to 1, "slice_start" to i, "slice_end" to i + 1,
                        ),
                    )
                    b.op(OpKind.RESHAPE, listOf(sliced), ty)
                }

                // `base[i] += value` with a constant i (GATHER's adjoint): the value placed at
                // position i of axis 1 by PAD_TO (zeros elsewhere) and added to the base.
                OpKind.SCATTER_ADD -> {
                    val (base, idx, v) = op.operands
                    val i = constIndex(idx)
                    if (isBatched(idx) || i == null) {
                        throw VmapUnsupportedException(op.op, "only a compile-time constant position is batched")
                    }
                    checkIndex(op, i, base.type.dims)
                    val baseB = batchedValue(base)
                    val vB = batchedValue(v)
                    val rest = v.type.dims
                    val v1 = b.op(OpKind.RESHAPE, listOf(vB), bt(DxirType(v.type.dtype, listOf(1) + rest)))
                    val placed = b.op(
                        OpKind.PAD_TO, listOf(v1, baseB), ty,
                        mapOf("low" to listOf(0, i) + List(rest.size) { 0 }),
                    )
                    b.op(OpKind.ADD, listOf(baseB, placed), ty)
                }

                // Linear algebra with native leading batch axes (stablehlo.cholesky and
                // stablehlo.triangular_solve take them): the same op one rank higher.
                OpKind.CHOLESKY, OpKind.TRIANGLE -> b.op(op.op, listOf(value(op.operands[0])), ty, op.attrs)

                OpKind.TRIANGULAR_SOLVE -> b.op(op.op, op.operands.map { batchedValue(it) }, ty, op.attrs)

                // LU-based (a stablehlo.while loop): the loop runs over all matrices at once.
                OpKind.SOLVE -> b.op(OpKind.SOLVE, op.operands.map { batchedValue(it) }, ty, op.attrs)
                OpKind.DET -> b.op(OpKind.DET, listOf(value(op.operands[0])), ty, op.attrs)
                // Householder and Jacobi loops (stablehlo.while), run over all matrices at once.
                OpKind.QR_Q, OpKind.QR_R, OpKind.EIGH_W, OpKind.EIGH_V ->
                    b.op(op.op, listOf(value(op.operands[0])), ty, op.attrs)

                else -> throw VmapUnsupportedException(op.op, "no rule is defined for this op kind")
            }
        }

        /**
         * `BROADCAST(value[, template]) {broadcast_dimensions}`. With concrete extents the
         * op keeps no template and the dimension map gains the batch axis. With `-1`
         * extents the batched form is a `BROADCAST_LIKE` against a batched template: the
         * value is first reshaped so its axes align right with the template's.
         */
        private fun batchBroadcast(op: DxirOp): DxirNode {
            val v = op.operands[0]
            val rv = v.type.rank
            val rOut = op.type.rank
            val bd = intList(op, "broadcast_dimensions").orEmpty()
            val eff: List<Int> = when {
                bd.isNotEmpty() -> bd
                rv == 0 -> emptyList()
                rv == rOut -> (0 until rv).toList()
                else -> throw VmapUnsupportedException(
                    op.op, "an empty broadcast_dimensions from rank $rv to rank $rOut",
                )
            }
            val ty = bt(op.type)
            if (allConcrete(op.type)) {
                val newBd = if (isBatched(v)) listOf(0) + eff.map { it + 1 } else eff.map { it + 1 }
                return b.op(
                    OpKind.BROADCAST, listOf(value(v)), ty,
                    attrs = op.attrs.filterKeys { it != "broadcast_dimensions" } + ("broadcast_dimensions" to newBd),
                )
            }
            if (op.operands.size != 2) {
                throw VmapUnsupportedException(
                    op.op, "a broadcast to extents known only at run time needs a shape template operand",
                )
            }
            if (!isBatched(v) && rv == 0) {
                // A scalar splat stays one, against the batched template.
                return b.op(
                    OpKind.BROADCAST, listOf(value(v), batchedValue(op.operands[1])), ty,
                    attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
                )
            }
            if (eff.zipWithNext().any { (a, c) -> a >= c }) {
                throw VmapUnsupportedException(op.op, "broadcast_dimensions $eff are not increasing")
            }
            // Aligned per-example shape: the value's axes at their output positions, 1 elsewhere.
            val aligned = (0 until rOut).map { k ->
                val j = eff.indexOf(k)
                if (j >= 0) v.type.dims[j] else 1
            }
            val alignedValue = if (isBatched(v)) {
                if (aligned == v.type.dims) value(v)
                else b.op(OpKind.RESHAPE, listOf(value(v)), bt(DxirType(v.type.dtype, aligned)))
            } else {
                // Unbatched: BROADCAST_LIKE right-aligns, so the unit axes in front of the
                // value's first axis need not be spelled out.
                val trimmed = aligned.subList(eff.firstOrNull() ?: rOut, rOut)
                if (trimmed == v.type.dims) value(v)
                else b.op(OpKind.RESHAPE, listOf(value(v)), DxirType(v.type.dtype, trimmed))
            }
            val template = batchedValue(op.operands[1])
            return b.op(OpKind.BROADCAST_LIKE, listOf(alignedValue, template), ty)
        }

        /**
         * Canonical `MATMUL` (no dimension attributes). A batched lhs against an unbatched
         * rank-2 rhs (`x · W`) keeps W shared; otherwise both operands are batched (an
         * unbatched one materialized) and the op is the same one rank higher — the canonical
         * batched matmul every engine takes.
         */
        private fun batchMatmul(op: DxirOp): DxirNode {
            val dimKeys = listOf("lhs_contracting_dims", "rhs_contracting_dims", "lhs_batching_dims", "rhs_batching_dims")
            val explicit = dimKeys.filter { it in op.attrs }
            if (explicit.isNotEmpty() && !canonicalDims(op)) {
                throw VmapUnsupportedException(
                    op.op, "a MATMUL with explicit dimension attributes ($explicit, the `contract` form) other than " +
                        "the canonical product (contract the lhs's last axis with the rhs's second-to-last, " +
                        "leading axes as batch) is not batched",
                )
            }
            // A named `contract` that is the canonical product batches as one; its dimension
            // attributes (which would shift) are dropped, the names kept.
            val attrs = op.attrs.filterKeys { it !in dimKeys }
            val (x, y) = op.operands
            // Already a shared-rhs MATMUL (an inner vmap's `x · W`): W stays shared when it is
            // still unbatched.
            if (y.type.rank == 2 && x.type.rank > 2) {
                if (isBatched(y)) {
                    throw VmapUnsupportedException(
                        op.op, "the shared rank-2 operand of an inner vmap's matmul is batched by the outer one",
                    )
                }
                return b.op(OpKind.MATMUL, listOf(value(x), value(y)), bt(op.type), attrs)
            }
            if (x.type.rank < 2 || x.type.rank != y.type.rank) {
                throw VmapUnsupportedException(op.op, "operands of ranks ${x.type.rank} and ${y.type.rank}")
            }
            // `x · W` with W not batched: W is shared by every example (a MATMUL of a batched
            // lhs and a rank-2 rhs), not copied per example.
            if (isBatched(x) && !isBatched(y) && y.type.rank == 2) {
                return b.op(OpKind.MATMUL, listOf(value(x), value(y)), bt(op.type), attrs)
            }
            return b.op(OpKind.MATMUL, listOf(batchedValue(x), batchedValue(y)), bt(op.type), attrs)
        }

        /**
         * The unbatched op checks its index at run time; the batched slice / pad cannot, so a
         * constant index is checked here, and an array of rank > 1 under `-1` extents (two
         * run-time extents in one reshape) is refused.
         */
        private fun checkIndex(op: DxirOp, i: Int, dims: List<Int>) {
            if (i < 0 || (dims[0] >= 0 && i >= dims[0])) {
                throw VmapUnsupportedException(op.op, "index $i outside the array's first axis (${dims[0]})")
            }
            if (dims.size > 1 && (batchSize < 0 || dims.any { it < 0 })) {
                throw VmapUnsupportedException(op.op, "an array of rank ${dims.size} with extents known only at run time")
            }
        }

        /** The value of a constant integer index (possibly behind the lowering's CAST), else null. */
        private fun constIndex(n: DxirNode): Int? = when {
            n is DxirConst -> (n.value as? Number)?.toInt()
            n is DxirOp && n.op == OpKind.CAST && n.operands.single() is DxirConst ->
                ((n.operands.single() as DxirConst).value as? Number)?.toInt()
            else -> null
        }

        /**
         * Whether [op]'s dimension attributes (absent, or those of a named `contract`) describe
         * the canonical product: equal ranks r, the lhs's axis r−1 contracted with the rhs's
         * axis r−2, and the leading r−2 axes batching axes in order.
         */
        private fun canonicalDims(op: DxirOp): Boolean {
            val (x, y) = op.operands
            val r = x.type.rank
            if (y.type.rank != r || r < 2) return false
            val lead = (0 until r - 2).toList()
            return intList(op, "lhs_contracting_dims") == listOf(r - 1) &&
                intList(op, "rhs_contracting_dims") == listOf(r - 2) &&
                (intList(op, "lhs_batching_dims") ?: emptyList()) == lead &&
                (intList(op, "rhs_batching_dims") ?: emptyList()) == lead
        }

        /**
         * `IF`: both branches' bodies are flattened into the outer stream (both evaluate),
         * as the forward transform does. An unbatched condition keeps an `IF` over the
         * branch yields; a batched one selects per example with `WHERE`.
         */
        private fun processIf(op: DxirOp) {
            val blocks = op.regions.map { r ->
                r.blocks.singleOrNull()
                    ?: throw VmapUnsupportedException(op.op, "an IF region with ${r.blocks.size} blocks")
            }
            if (blocks.size != 2 || blocks.any { it.args.isNotEmpty() }) {
                throw VmapUnsupportedException(op.op, "an IF that is not a two-branch, argument-free conditional")
            }
            for (blk in blocks) for (inner in blk.body) process(inner)
            val cond = op.operands[0]
            val results = op.types.indices.map { k ->
                val a = blocks[0].terminator[k]
                val c = blocks[1].terminator[k]
                Triple(a, c, isBatched(a) || isBatched(c))
            }
            if (!isBatched(cond)) {
                val yieldsThen = results.map { (a, _, bat) -> if (bat) batchedValue(a) else value(a) }
                val yieldsElse = results.map { (_, c, bat) -> if (bat) batchedValue(c) else value(c) }
                val types = results.mapIndexed { k, (_, _, bat) -> if (bat) bt(op.types[k]) else op.types[k] }
                val newIf = b.ifOp(
                    value(cond), types,
                    b.region { yields(*yieldsThen.toTypedArray()) },
                    b.region { yields(*yieldsElse.toTypedArray()) },
                )
                values[op.id] = types.indices.map { newIf.result(it) }
                batchedFlags[op.id] = results.map { it.third }
                return
            }
            val predB = value(cond)
            // Bool, or (under `-1` extents) the float mask a batched STEP / COMPARE produces.
            val predDtype = predB.type.dtype
            values[op.id] = op.types.mapIndexed { k, t ->
                val (a, c, _) = results[k]
                val aB = batchedValue(a)
                val cB = batchedValue(c)
                val pred = when {
                    t.rank == 0 -> predB
                    allConcrete(t) -> b.op(
                        OpKind.BROADCAST, listOf(predB), bt(DxirType(predDtype, t.dims)),
                        attrs = mapOf("broadcast_dimensions" to listOf(0)),
                    )
                    else -> b.op(
                        OpKind.BROADCAST_LIKE,
                        listOf(b.op(OpKind.RESHAPE, listOf(predB), bt(DxirType(predDtype, List(t.rank) { 1 }))), aB),
                        bt(DxirType(predDtype, t.dims)),
                    )
                }
                b.op(OpKind.WHERE, listOf(pred, aB, cB), bt(t))
            }
            batchedFlags[op.id] = op.types.map { true }
        }
    }
}
