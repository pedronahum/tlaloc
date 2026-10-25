package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.SelectionKernels
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.hostI32
import kotlin.jvm.JvmName

// Index selection ([io.tlaloc.core.SelectionKernels]): the host twins of TOP_K,
// TAKE_ALONG and TAKE_ALONG_GRAD. The F64 twins are in HostOpsF64.kt.

/**
 * The positions of the [k] largest entries along the last axis, largest first (equal
 * values in increasing position): extents of this tensor with the last replaced by [k].
 */
fun <S : Shape> DTensor<S, F32>.topKIndices(k: Int): DTensor<Shape, I32> {
    val v = hostF32()
    val out = SelectionKernels.topK(DoubleArray(v.size) { v[it].toDouble() }, dims, k)
    return DTensor(HostI32Storage(out), dims.copyOf().also { it[it.size - 1] = k }, I32)
}

/**
 * `out[g, j, …] = this[g, indices[g, j], …]` for this tensor `[G, N, R…]` and [indices]
 * `[G, M]`: extents `[G, M, R…]`.
 */
fun DTensor<*, F32>.takeAlong(indices: DTensor<*, I32>): DTensor<Shape, F32> {
    val v = hostF32()
    val m = checkTakeAlong(dims, indices.dims)
    val out = SelectionKernels.takeAlong(DoubleArray(v.size) { v[it].toDouble() }, dims, indices.hostI32(), m)
    return DTensor(HostF32Storage(FloatArray(out.size) { out[it].toFloat() }), takeAlongDims(dims, m), F32)
}

/** [takeAlong] of an integer tensor. */
@JvmName("takeAlongI32")
fun DTensor<*, I32>.takeAlong(indices: DTensor<*, I32>): DTensor<Shape, I32> {
    val v = hostI32()
    val m = checkTakeAlong(dims, indices.dims)
    val out = SelectionKernels.takeAlong(DoubleArray(v.size) { v[it].toDouble() }, dims, indices.hostI32(), m)
    return DTensor(HostI32Storage(IntArray(out.size) { out[it].toInt() }), takeAlongDims(dims, m), I32)
}

/**
 * The adjoint of [takeAlong]: this upstream `[G, M, R…]` added into zeros shaped like
 * [template] `[G, N, R…]` at [indices] `[G, M]` (repeated indices accumulate).
 */
fun DTensor<*, F32>.takeAlongGrad(indices: DTensor<*, I32>, template: DTensor<*, F32>): DTensor<Shape, F32> {
    val v = hostF32()
    val m = checkTakeAlong(template.dims, indices.dims)
    val out = SelectionKernels.takeAlongGrad(DoubleArray(v.size) { v[it].toDouble() }, indices.hostI32(), m, template.dims)
    return DTensor(HostF32Storage(FloatArray(out.size) { out[it].toFloat() }), template.dims.copyOf(), F32)
}

internal fun checkTakeAlong(xDims: IntArray, idxDims: IntArray): Int {
    require(xDims.size >= 2) { "takeAlong: the tensor must be [G, N, …] (got ${xDims.toList()})" }
    require(idxDims.size == 2 && idxDims[0] == xDims[0]) {
        "takeAlong: indices must be [${xDims[0]}, M] (got ${idxDims.toList()})"
    }
    return idxDims[1]
}

internal fun takeAlongDims(xDims: IntArray, m: Int): IntArray = xDims.copyOf().also { it[1] = m }
