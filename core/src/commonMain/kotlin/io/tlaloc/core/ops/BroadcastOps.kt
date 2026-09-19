package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32

// Phase A5c-2 (DiffKT parity) — implicit broadcasting on the elementwise binaries.
//
// THIS IS A SEPARATE FILE ON PURPOSE. Kotlin generics erase, so a broadcasting
// `operator fun <S1, S2> DTensor<S1, F32>.plus(other: DTensor<S2, F32>)` and the
// shape-preserving `operator fun <S> DTensor<S, F32>.plus(other: DTensor<S, F32>)`
// in HostOps.kt have the IDENTICAL JVM signature
// `plus(Lio/tlaloc/core/DTensor;Lio/tlaloc/core/DTensor;)Lio/tlaloc/core/DTensor;`
// — a platform declaration clash if they share a file (top-level functions compile
// into one facade class named after the file, and `@JvmName` is not available in
// commonMain). A separate file gives them separate facade classes while keeping
// them in the SAME package, so `import io.tlaloc.core.ops.plus` still brings in
// both and Kotlin's most-specific-wins resolution picks the shape-preserving one
// whenever the operands already agree. Folding this file back into HostOps.kt
// breaks the build.

/**
 * The broadcasting elementwise walk behind `a + b` for differently-shaped
 * operands. NumPy semantics: the two shapes right-align, each aligned pair must be
 * equal or 1 on one side, the result takes the max of each pair, and a
 * rank-deficient operand gains replicated leading axes.
 *
 * EQUAL dims take the same flat zip [elementwise] has always used. That matters
 * because the K2 plugin routes every tensor binary in a gradient body through
 * here, and under `grad {}`'s -1 sentinel dims two operands with the same static
 * shape can still turn out differently shaped at runtime — `[N,1]` and `[N,C]` are
 * both `Rank2<Sym, Lit<Int>>`. Only genuinely mixed shapes pay for the stride walk.
 *
 * The result shape comes from the operands' RUNTIME dims, never from a static
 * witness; [R] is supplied by the caller (synthesis threads the derived result
 * IrType's shape arg, the operator overloads below pass `Shape`) exactly the way
 * `sumToLike` and `broadcastLike` already do.
 */
internal fun <R : Shape> elementwiseBroadcast(
    a: DTensor<*, F32>,
    b: DTensor<*, F32>,
    f: (Float, Float) -> Float,
): DTensor<R, F32> {
    val ad = a.dims
    val bd = b.dims
    if (ad.contentEquals(bd)) {
        val av = a.hostF32()
        val bv = b.hostF32()
        val same = FloatArray(av.size)
        for (i in av.indices) same[i] = f(av[i], bv[i])
        return DTensor(HostF32Storage(same), ad.copyOf(), F32)
    }
    val r = maxOf(ad.size, bd.size)
    val outDims = IntArray(r)
    for (k in 0 until r) {
        val x = dimOrOne(ad, k, r)
        val y = dimOrOne(bd, k, r)
        require(x == y || x == 1 || y == 1) {
            "elementwiseBroadcast: shapes ${ad.toList()} and ${bd.toList()} are not " +
                "broadcast-compatible (axis $k: $x vs $y)"
        }
        outDims[k] = maxOf(x, y)
    }
    val aStrides = broadcastStridesFor(ad, outDims, r)
    val bStrides = broadcastStridesFor(bd, outDims, r)
    val outStrides = IntArray(r)
    run {
        var s = 1
        for (k in r - 1 downTo 0) { outStrides[k] = s; s *= outDims[k] }
    }
    val av = a.hostF32()
    val bv = b.hostF32()
    val out = FloatArray(outDims.fold(1) { acc, d -> acc * d })
    for (flat in out.indices) {
        var rem = flat
        var ia = 0
        var ib = 0
        for (k in 0 until r) {
            val coord = rem / outStrides[k]
            rem -= coord * outStrides[k]
            ia += coord * aStrides[k]
            ib += coord * bStrides[k]
        }
        out[flat] = f(av[ia], bv[ib])
    }
    return DTensor(HostF32Storage(out), outDims, F32)
}

/** Right-aligned axis [k] of [dims] within a rank-[rank] result: 1 for axes the operand lacks. */
private fun dimOrOne(dims: IntArray, k: Int, rank: Int): Int {
    val i = k - (rank - dims.size)
    return if (i < 0) 1 else dims[i]
}

/**
 * Per-result-axis strides for one operand of a broadcasting binary: its own
 * row-major strides right-aligned into the result rank, with 0 on the axes it lacks
 * and on its size-1 axes — both replicate, so they never advance the flat index.
 * Mirrors the dxir interpreter's `broadcastStrides` and the emitter's
 * right-aligned `broadcast_in_dim` map, so all three agree on one rule.
 */
private fun broadcastStridesFor(dims: IntArray, outDims: IntArray, rank: Int): IntArray {
    val own = IntArray(dims.size)
    run {
        var s = 1
        for (k in dims.indices.reversed()) { own[k] = s; s *= dims[k] }
    }
    val strides = IntArray(rank)
    val offset = rank - dims.size
    for (k in 0 until rank) {
        val i = k - offset
        if (i >= 0 && dims[i] != 1) strides[k] = own[i]
    }
    return strides
}

/**
 * The broadcasting binaries the K2 plugin synthesises tensor ADD/SUB/MUL/DIV with.
 * Star-projected operands plus an explicit result-shape witness — the `sumToLike` /
 * `broadcastLike` calling convention. These are top-level (not extensions) so each
 * name resolves uniquely by `CallableId`.
 */
fun <R : Shape> plusBroadcast(a: DTensor<*, F32>, b: DTensor<*, F32>): DTensor<R, F32> =
    elementwiseBroadcast(a, b) { x, y -> x + y }

fun <R : Shape> minusBroadcast(a: DTensor<*, F32>, b: DTensor<*, F32>): DTensor<R, F32> =
    elementwiseBroadcast(a, b) { x, y -> x - y }

fun <R : Shape> timesBroadcast(a: DTensor<*, F32>, b: DTensor<*, F32>): DTensor<R, F32> =
    elementwiseBroadcast(a, b) { x, y -> x * y }

fun <R : Shape> divBroadcast(a: DTensor<*, F32>, b: DTensor<*, F32>): DTensor<R, F32> =
    elementwiseBroadcast(a, b) { x, y -> x / y }

/**
 * The DiffKT-parity user surface: `a + b` broadcasts when the shapes differ.
 * Deliberately LESS specific than HostOps.kt's same-`S` operators, so resolution
 * keeps preferring those (and their precise shape witness) whenever the operands
 * already agree, and falls through to broadcasting only when they don't. The result
 * is `DTensor<Shape, F32>` — the broadcast of two arbitrary shapes has no static
 * witness worth claiming, and every downstream op is generic in `S` anyway.
 */
operator fun <S1 : Shape, S2 : Shape> DTensor<S1, F32>.plus(other: DTensor<S2, F32>): DTensor<Shape, F32> =
    plusBroadcast(this, other)

operator fun <S1 : Shape, S2 : Shape> DTensor<S1, F32>.minus(other: DTensor<S2, F32>): DTensor<Shape, F32> =
    minusBroadcast(this, other)

operator fun <S1 : Shape, S2 : Shape> DTensor<S1, F32>.times(other: DTensor<S2, F32>): DTensor<Shape, F32> =
    timesBroadcast(this, other)

operator fun <S1 : Shape, S2 : Shape> DTensor<S1, F32>.div(other: DTensor<S2, F32>): DTensor<Shape, F32> =
    divBroadcast(this, other)
