package io.tlaloc.autograd

import io.tlaloc.core.I32
import io.tlaloc.core.SelectionKernels
import io.tlaloc.core.Shape
import io.tlaloc.ir.OpKind

// Index selection in a trace: the positions of the largest entries, and the gather of
// rows by those positions. The positions are I32 tracers and carry no gradient; the
// gathered values do (TAKE_ALONG's adjoint scatters back to the rows they came from).

/**
 * The positions of the [k] largest entries along the last axis, largest first (equal
 * values in increasing position), as an I32 tracer of this shape with the last extent
 * [k]. Records [OpKind.TOP_K].
 */
fun Tracer<*>.topK(k: Int): Tracer<Shape> {
    require(rank >= 1) { "topK: needs an axis (got a scalar)" }
    require(k in 1..dims[rank - 1]) { "topK: k = $k outside 1..${dims[rank - 1]}" }
    val idx = SelectionKernels.topK(DoubleArray(size) { entry.value[it].toDouble() }, dims, k)
    val outDims = dims.copyOf().also { it[rank - 1] = k }
    val e = tape.op(
        OpKind.TOP_K, intArrayOf(id), outDims, FloatArray(idx.size) { idx[it].toFloat() },
        attrs = mapOf("k" to k), dtype = I32,
    )
    return Tracer(tape, e)
}

/**
 * The first position of the largest entry along [axis], which is dropped, as an I32
 * tracer. Records [OpKind.ARGMAX].
 */
fun Tracer<*>.argmax(axis: Int = -1): Tracer<Shape> {
    val a = if (axis < 0) axis + rank else axis
    require(rank >= 1 && a in 0 until rank) { "argmax: axis $axis out of range for rank $rank" }
    val n = dims[a]
    val inner = dims.drop(a + 1).fold(1) { x, y -> x * y }
    val outer = dims.take(a).fold(1) { x, y -> x * y }
    val v = entry.value
    val out = FloatArray(outer * inner) { o ->
        val base = (o / inner) * n * inner + o % inner
        var best = 0
        for (i in 1 until n) if (v[base + i * inner] > v[base + best * inner]) best = i
        best.toFloat()
    }
    val outDims = dims.filterIndexed { i, _ -> i != a }.toIntArray()
    val e = tape.op(OpKind.ARGMAX, intArrayOf(id), outDims, out, attrs = mapOf("axis" to a), dtype = I32)
    return Tracer(tape, e)
}

/**
 * `out[g, j, …] = this[g, indices[g, j], …]` for this tracer `[G, N, R…]` and the I32
 * [indices] `[G, M]`: a gather along axis 1 with axis 0 as a batch, extents `[G, M, R…]`.
 * The receiver may itself be I32 (positions of positions). Records [OpKind.TAKE_ALONG].
 */
fun Tracer<*>.takeAlong(indices: Tracer<*>): Tracer<Shape> {
    require(rank >= 2) { "takeAlong: the tensor must be [G, N, …] (got ${dims.toList()})" }
    require(indices.dtype == I32) { "takeAlong: indices must be I32 (got ${indices.dtype.name})" }
    require(indices.rank == 2 && indices.dims[0] == dims[0]) {
        "takeAlong: indices must be [${dims[0]}, M] (got ${indices.dims.toList()})"
    }
    val tape = sameTape(this, indices)
    val m = indices.dims[1]
    val idx = IntArray(indices.size) { indices.entry.value[it].toInt() }
    val out = SelectionKernels.takeAlong(DoubleArray(size) { entry.value[it].toDouble() }, dims, idx, m)
    val outDims = dims.copyOf().also { it[1] = m }
    val e = tape.op(
        OpKind.TAKE_ALONG, intArrayOf(id, indices.id), outDims, FloatArray(out.size) { out[it].toFloat() },
        dtype = dtype,
    )
    return Tracer(tape, e)
}
