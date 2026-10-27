package io.tlaloc.stablehlo.mosaic

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * A Mosaic TPU kernel written in Kotlin at the level of a Pallas kernel: a grid, blocks of the
 * inputs and outputs chosen by index maps over the grid indices, VMEM scratch, and a body of
 * whole-block vector operations. [mosaicKernel] returns the module Pallas lowers for the
 * equivalent `pl.pallas_call`, op for op, so `harness/python/check_kmosaic_forms.py` can compare
 * the two (equal after `cse`) and run the Pallas twin in TPU interpret mode for its values.
 *
 * ```kotlin
 * val text = mosaicKernel("matmul") {
 *     grid(m / tm, n / tn, k / tk, semantics = listOf(PARALLEL, PARALLEL, ARBITRARY))
 *     val x = input(listOf(tm, tk), TileType.BF16) { (i, _, kk) -> listOf(i, kk) }
 *     val w = input(listOf(tk, tn), TileType.BF16) { (_, j, kk) -> listOf(kk, j) }
 *     val o = output(listOf(tm, tn), TileType.BF16) { (i, j, _) -> listOf(i, j) }
 *     val acc = scratch(listOf(tm, tn), TileType.F32)
 *     body {
 *         whenEq(programId(2), 0) { acc.store(zeros(acc)) }
 *         acc.store(acc.load() + dot(x.load(), w.load()))
 *         whenEq(programId(2), k / tk - 1) { o.store(acc.load().cast(TileType.BF16)) }
 *     }
 * }
 * ```
 *
 * Each body call emits what Pallas emits for its `jnp` / `pl` counterpart, in call order:
 * `ref.load()` is `ref[...]`, `+` with a differently shaped operand broadcasts it first (as
 * `jnp` does), a [Float] operand is a constant broadcast to the tile, `sum(axis)` /
 * `max(axis)` keep the axis (`keepdims=True`), [TileBody.dot] is `jax.lax.dot_general` with an
 * f32 result. Write the body in the order of the Pallas twin.
 */
@ExperimentalTlalocApi
fun mosaicKernel(name: String, serialized: Boolean = false, build: MosaicKernelScope.() -> Unit): String =
    MosaicKernelScope(name, serialized).apply(build).emit()

/** Element types of tiles. */
@ExperimentalTlalocApi
enum class TileType(val mlir: String) { F32("f32"), BF16("bf16"), I32("i32"), I1("i1") }

/** A grid axis's dimension semantics. */
@ExperimentalTlalocApi
enum class GridSemantics(val mlir: String) { PARALLEL("parallel"), ARBITRARY("arbitrary") }

/** A grid index in an index map. */
@ExperimentalTlalocApi
class GridIndex internal constructor(val axis: Int)

/** A block of an input, output or scratch buffer, in VMEM, as the body sees it. */
@ExperimentalTlalocApi
class TileRef internal constructor(val shape: List<Int>, val type: TileType, internal val index: ((List<GridIndex>) -> List<Any>)?) {
    internal var value: MosaicWriter.V? = null
    internal val memref: String get() = "memref<${shape.joinToString("x")}x${type.mlir}, #tpu.memory_space<vmem>>"
}

/** A vector value in the body: a whole tile or a scalar (an empty shape). */
@ExperimentalTlalocApi
class Tile internal constructor(internal val v: MosaicWriter.V, val shape: List<Int>, val type: TileType)

@ExperimentalTlalocApi
class MosaicKernelScope internal constructor(private val name: String, private val serialized: Boolean) {
    private var grid: List<Int> = emptyList()
    private var semantics: List<GridSemantics> = emptyList()
    private val inputs = ArrayList<TileRef>()
    private val outputs = ArrayList<TileRef>()
    private val scratch = ArrayList<TileRef>()
    private var body: (TileBody.() -> Unit)? = null

    /** The grid: [bounds] iterations per axis, with [semantics] (all parallel by default). */
    fun grid(vararg bounds: Int, semantics: List<GridSemantics> = bounds.map { GridSemantics.PARALLEL }) {
        require(bounds.all { it > 0 } && semantics.size == bounds.size) { "mosaicKernel: grid $bounds with semantics $semantics" }
        grid = bounds.toList()
        this.semantics = semantics
    }

    /** An input block of [shape]; [index] maps the grid indices to block indices (a [GridIndex] or an Int each). */
    fun input(shape: List<Int>, type: TileType, index: (List<GridIndex>) -> List<Any> = { emptyList() }): TileRef =
        TileRef(shape, type, index).also { inputs += it }

    fun output(shape: List<Int>, type: TileType, index: (List<GridIndex>) -> List<Any> = { emptyList() }): TileRef =
        TileRef(shape, type, index).also { outputs += it }

    /** VMEM scratch of [shape], kept across grid iterations. */
    fun scratch(shape: List<Int>, type: TileType): TileRef = TileRef(shape, type, null).also { scratch += it }

    fun body(block: TileBody.() -> Unit) {
        body = block
    }

    internal fun emit(): String {
        val w = MosaicWriter(serialized)
        val refs = inputs + outputs + scratch
        val windowed = inputs + outputs
        val attrs = ArrayList<Pair<String, String>>()
        attrs += "dimension_semantics" to "[${semantics.joinToString(", ") { "#tpu.dimension_semantics<${it.mlir}>" }}]"
        if (grid.isNotEmpty()) attrs += "iteration_bounds" to "array<i64: ${grid.joinToString(", ")}>"
        attrs += "scalar_prefetch" to "0 : i64"
        attrs += "scratch_operands" to "${scratch.size} : i64"
        attrs += "tpu.core_type" to "#tpu.core_type<tc>"
        if (grid.isNotEmpty()) {
            val indices = grid.indices.map { GridIndex(it) }
            attrs += "window_params" to windowed.withIndex().joinToString(", ", "[", "]") { (i, r) ->
                // A block whose index map reads no grid index is the same block at every step:
                // Pallas fetches it once (no double buffer).
                val fixed = r.index!!(indices).none { it is GridIndex }
                val mode = if (fixed) "pipeline_mode = #tpu.pipeline_mode<synchronous>, " else ""
                "{${mode}transform_indices = @transform_$i, window_bounds = array<i64: ${r.shape.joinToString(", ")}>}"
            }
        }
        w.function(name, grid.map { "i32" } + refs.map { it.memref }, emptyList(), attrs) { args ->
            for ((i, r) in refs.withIndex()) r.value = args[grid.size + i]
            TileBody(this, args.take(grid.size)).apply(body ?: error("mosaicKernel '$name' has no body"))
            ret()
        }
        if (grid.isNotEmpty()) {
            val indices = grid.indices.map { GridIndex(it) }
            for ((i, r) in windowed.withIndex()) {
                val map = r.index!!(indices)
                require(map.size == r.shape.size) { "mosaicKernel '$name': operand $i's index map gives ${map.size} indices for a rank-${r.shape.size} block" }
                w.function("transform_$i", grid.map { "i32" }, r.shape.map { "i32" }, emptyList()) { a ->
                    // Pallas's index-map functions start with a zero constant, used or not.
                    i32Const(0)
                    ret(map.map { e -> if (e is GridIndex) a[e.axis] else i32Const(e as Int) })
                }
            }
        }
        return w.module(name)
    }
}

/** The kernel body: whole-tile vector operations, emitted in call order. */
@ExperimentalTlalocApi
class TileBody internal constructor(private val w: MosaicWriter, private val grid: List<MosaicWriter.V>) {

    private fun vec(shape: List<Int>, type: TileType) =
        if (shape.isEmpty()) type.mlir else "vector<${shape.joinToString("x")}x${type.mlir}>"

    private fun tile(v: MosaicWriter.V, shape: List<Int>, type: TileType) = Tile(v, shape, type)

    /** `pl.program_id(axis)`: an i32 scalar. */
    fun programId(axis: Int): Tile = tile(grid[axis], emptyList(), TileType.I32)

    /** `ref[...]`. */
    fun TileRef.load(): Tile {
        val idx = shape.map { w.indexConst(0) }
        return tile(w.load(value!!, idx, vec(shape, type)), shape, type)
    }

    /** `ref[...] = t`. */
    fun TileRef.store(t: Tile) {
        require(t.shape == shape && t.type == type) { "store: a ${t.type} ${t.shape} tile into a $type $shape block" }
        w.store(t.v, value!!, shape.map { w.indexConst(0) })
    }

    /** `jnp.zeros_like(ref)`: an f32 zero broadcast to the block. */
    fun zeros(like: TileRef): Tile = splat(0f, like.shape, like.type)

    /** A constant broadcast to [shape] (`jnp.full`). */
    fun splat(value: Float, shape: List<Int>, type: TileType = TileType.F32): Tile {
        val c = w.f32Const("${hex(value, type)} : ${type.mlir}", type.mlir)
        return tile(w.broadcast(c, vec(shape, type)), shape, type)
    }

    /** `jax.lax.broadcasted_iota(jnp.int32, shape, dimension)`. */
    fun iota(shape: List<Int>, dimension: Int): Tile =
        tile(w.op1("tpu.iota", emptyList(), vec(shape, TileType.I32), listOf("dimensions" to "array<i32: $dimension>")), shape, TileType.I32)

    private fun broadcastTo(t: Tile, shape: List<Int>): Tile = if (t.shape == shape) t else tile(w.broadcast(t.v, vec(shape, t.type)), shape, t.type)

    private fun broadcastShape(a: List<Int>, b: List<Int>): List<Int> {
        if (a.isEmpty()) return b
        if (b.isEmpty()) return a
        require(a.size == b.size) { "broadcast: ranks of $a and $b differ" }
        return a.indices.map { i ->
            require(a[i] == b[i] || a[i] == 1 || b[i] == 1) { "broadcast: $a and $b do not broadcast" }
            maxOf(a[i], b[i])
        }
    }

    private fun binary(op: String, a: Tile, b: Tile, resultType: TileType? = null): Tile {
        require(a.type == b.type) { "$op: ${a.type} and ${b.type} operands; cast one first" }
        val shape = broadcastShape(a.shape, b.shape)
        val x = broadcastTo(a, shape)
        val y = broadcastTo(b, shape)
        val float = a.type == TileType.F32 || a.type == TileType.BF16
        val name = if (float) "arith.${op}f" else "arith.${op}i"
        val v = if (float) w.float(name, x.v, y.v)
        else w.op1(name, listOf(x.v, y.v), vec(shape, a.type), listOf("overflowFlags" to "#arith.overflow<none>"))
        return tile(v, shape, resultType ?: a.type)
    }

    private fun scalarConst(value: Number, type: TileType): Tile =
        if (type == TileType.I32) tile(w.i32Const(value.toInt()), emptyList(), type)
        else tile(w.f32Const("${hex(value.toFloat(), type)} : ${type.mlir}", type.mlir), emptyList(), type)

    operator fun Tile.plus(o: Tile) = binary("add", this, o)
    operator fun Tile.minus(o: Tile) = binary("sub", this, o)
    operator fun Tile.times(o: Tile) = binary("mul", this, o)
    operator fun Tile.div(o: Tile) = binary("div", this, o)
    operator fun Tile.plus(s: Number) = binary("add", this, scalarConst(s, type))
    operator fun Tile.minus(s: Number) = binary("sub", this, scalarConst(s, type))
    operator fun Tile.times(s: Number) = binary("mul", this, scalarConst(s, type))
    operator fun Tile.div(s: Number) = binary("div", this, scalarConst(s, type))

    /** `jnp.maximum`. */
    fun maximum(a: Tile, b: Tile): Tile = binary("maximum", a, b)
    fun maximum(a: Tile, s: Number): Tile = binary("maximum", a, scalarConst(s, a.type))

    private fun unary(name: String, a: Tile): Tile = tile(w.float(name, a.v), a.shape, a.type)
    fun exp(a: Tile) = unary("math.exp", a)
    fun rsqrt(a: Tile) = unary("math.rsqrt", a)
    fun tanh(a: Tile) = unary("math.tanh", a)
    fun log(a: Tile) = unary("math.log", a)

    /** `x.astype(type)` between f32 and bf16. */
    fun Tile.cast(to: TileType): Tile {
        if (to == type) return this
        val name = when {
            type == TileType.BF16 && to == TileType.F32 -> "arith.extf"
            type == TileType.F32 && to == TileType.BF16 -> "arith.truncf"
            else -> error("cast: $type to $to is not supported")
        }
        return tile(w.cast(name, v, vec(shape, to)), shape, to)
    }

    private fun reduce(kind: String, init: String, a: Tile, axis: Int, keepDims: Boolean): Tile {
        require(a.type == TileType.F32 && axis in a.shape.indices) { "reduce: an f32 tile and one of its axes, got ${a.type} ${a.shape} axis $axis" }
        val reduced = a.shape.filterIndexed { i, _ -> i != axis }
        val acc = w.f32Const("dense<$init> : ${vec(reduced, a.type)}", vec(reduced, a.type))
        val r = w.op1(
            "vector.multi_reduction", listOf(a.v, acc), vec(reduced, a.type),
            listOf("kind" to "#vector.kind<$kind>", "reduction_dims" to "array<i64: $axis>"),
        )
        if (!keepDims) return tile(r, reduced, a.type)
        val kept = a.shape.mapIndexed { i, d -> if (i == axis) 1 else d }
        return tile(w.op1("vector.shape_cast", listOf(r), vec(kept, a.type)), kept, a.type)
    }

    /** `jnp.sum(x, axis, keepdims=True)` (or without the axis when [keepDims] is false). */
    fun Tile.sum(axis: Int, keepDims: Boolean = true) = reduce("add", hex(0f), this, axis, keepDims)

    /** `jnp.max(x, axis, keepdims=True)`. */
    fun Tile.max(axis: Int, keepDims: Boolean = true) = reduce("maximumf", hex(Float.NEGATIVE_INFINITY), this, axis, keepDims)

    /**
     * `jax.lax.dot_general` of two rank-2 tiles with an f32 result: `a·b`, `a·bᵀ`
     * ([transposeB]) or `aᵀ·b` ([transposeA]), spelled by dimension numbers (Mosaic rejects
     * `transpose_lhs`).
     */
    fun dot(a: Tile, b: Tile, transposeA: Boolean = false, transposeB: Boolean = false): Tile {
        require(a.shape.size == 2 && b.shape.size == 2 && a.type == b.type) { "dot: rank-2 tiles of one type, got ${a.type} ${a.shape} and ${b.type} ${b.shape}" }
        require(!(transposeA && transposeB)) { "dot: transpose one operand at most" }
        val (m, ka) = if (transposeA) a.shape.reversed() else a.shape
        val (kb, n) = if (transposeB) b.shape.reversed() else b.shape
        require(ka == kb) { "dot: contracting extents $ka and $kb differ" }
        val dims = when {
            transposeA -> "[0], [0], [1], [1], [0, 1, 1, 1], [], []"
            transposeB -> "[1], [1], [0], [0], [0, 0, 1, 0], [], []"
            else -> "[1], [0], [0], [1], [0, 0, 1, 1], [], []"
        }
        val out = vec(listOf(m, n), TileType.F32)
        val init = w.f32Const("dense<${hex(0f)}> : $out", out)
        return tile(w.matmul(a.v, b.v, init, "#tpu.dot_dimension_numbers<$dims>"), listOf(m, n), TileType.F32)
    }

    private fun compare(predicate: Int, a: Tile, b: Tile): Tile {
        require(a.type == TileType.I32 && b.type == TileType.I32) { "compare: i32 tiles" }
        val shape = broadcastShape(a.shape, b.shape)
        val x = broadcastTo(a, shape)
        val y = broadcastTo(b, shape)
        return tile(w.op1("arith.cmpi", listOf(x.v, y.v), vec(shape, TileType.I1), listOf("predicate" to "$predicate : i64")), shape, TileType.I1)
    }

    /** `a <= b` on i32 tiles (`arith.cmpi sle`). */
    fun lessOrEqual(a: Tile, b: Tile) = compare(3, a, b)

    /** `a < b` on i32 tiles (`arith.cmpi slt`). */
    fun less(a: Tile, b: Tile) = compare(2, a, b)

    /** `jnp.where(cond, a, b)`; a [Float] [b] is broadcast first. */
    fun where(cond: Tile, a: Tile, b: Float): Tile {
        val bb = splat(b, a.shape, a.type)
        return tile(w.op1("arith.select", listOf(cond.v, a.v, bb.v), vec(a.shape, a.type)), a.shape, a.type)
    }

    /** `@pl.when(s == value)`. */
    fun whenEq(s: Tile, value: Int, then: TileBody.() -> Unit) {
        require(s.shape.isEmpty() && s.type == TileType.I32) { "whenEq: an i32 scalar such as programId(axis)" }
        val c = w.i32Const(value)
        val eq = w.cmpi(MosaicWriter.EQ, s.v, c)
        val ext = w.extui(eq, "i32")
        val ne = w.cmpi(MosaicWriter.NE, ext, w.i32Const(0))
        w.ifThen(ne) { then() }
    }

    private companion object {
        /** [v] as MLIR's exact hex spelling for [type] (bf16: the upper 16 bits, rounded to nearest even). */
        fun hex(v: Float, type: TileType = TileType.F32): String {
            val bits = v.toRawBits()
            if (type != TileType.BF16) return "0x" + bits.toUInt().toString(16).uppercase().padStart(8, '0')
            val rounded = (bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16
            return "0x" + (rounded and 0xFFFF).toString(16).uppercase().padStart(4, '0')
        }
    }
}
