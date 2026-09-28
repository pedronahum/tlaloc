package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.broadcastToLike
import io.tlaloc.core.ops.max
import io.tlaloc.core.ops.softmax
import io.tlaloc.core.ops.transpose
import io.tlaloc.ir.OpKind

// The trace spellings a transformer layer needs beyond the elementwise and
// reduction set in TracedOps.kt: axis permutation, softmax, axis max, NumPy
// broadcasting and a scalar fill. Each records the op kind and attrs that
// DxirInterpreter, StablehloEmitter and the Vjp registry already read
// (TransposeRule, SoftmaxRule, MaxRule, BroadcastRule), and computes its
// forward value through the `:core` host twin.

/**
 * Permutes the axes: output axis `i` is input axis `perm[i]`
 * (`stablehlo.transpose`). Records [OpKind.TRANSPOSE] with `permutation`.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.transpose(vararg perm: Int): Tracer<S> {
    require(perm.size == rank) {
        "transpose: permutation ${perm.toList()} must have one entry per axis (rank $rank)"
    }
    require(perm.sorted() == (0 until rank).toList()) {
        "transpose: ${perm.toList()} is not a permutation of 0..${rank - 1}"
    }
    val out = toF32Tensor().transpose(*perm)
    val e = tape.op(
        OpKind.TRANSPOSE,
        intArrayOf(id),
        out.dims.copyOf(),
        out.hostF32(),
        attrs = mapOf("permutation" to perm.toList()),
        dtype = entry.dtype,
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * Softmax along [axis] (negative counts from the back). Records
 * [OpKind.SOFTMAX] with the normalized `axis`.
 */
fun <S : Shape> Tracer<S>.softmax(axis: Int = -1): Tracer<S> {
    val a = if (axis < 0) axis + rank else axis
    require(rank >= 1 && a in 0 until rank) { "softmax: axis $axis out of range for rank $rank" }
    val out = toDTensor().softmax(a)
    val e = tape.op(OpKind.SOFTMAX, intArrayOf(id), dims.copyOf(), out.hostF32(), attrs = mapOf("axis" to a))
    return Tracer(tape, e)
}

/**
 * Maximum over [axes], which are dropped from the result. Records
 * [OpKind.MAX] with `reduction_dims`. Its gradient goes to every element
 * equal to the maximum.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.max(axes: IntArray): Tracer<S> {
    val sorted = axes.map { if (it < 0) it + rank else it }.distinct().sorted()
    require(sorted.isNotEmpty() && sorted.all { it in 0 until rank }) {
        "max(axes): axes ${axes.toList()} out of range for rank $rank (dims ${dims.toList()})"
    }
    val out = toF32Tensor().max(*sorted.toIntArray())
    val e = tape.op(
        OpKind.MAX,
        intArrayOf(id),
        out.dims.copyOf(),
        out.hostF32(),
        attrs = mapOf("reduction_dims" to sorted),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * NumPy broadcasting to [targetDims]: the receiver's axes align with the
 * trailing axes of the target, and each must equal the target's extent or be 1.
 *
 * A receiver of lower rank is first reshaped to the target rank with leading
 * 1s, so the recorded [OpKind.BROADCAST] is always equal-rank with the
 * identity `broadcast_dimensions`. Its adjoint is then `SUM_TO`, which sums
 * exactly the stretched axes. A rank-increasing BROADCAST that also stretches
 * a size-1 axis would reach BroadcastRule's rank-increasing arm, which sums
 * only the inserted axes.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<*>.broadcastTo(targetDims: IntArray): Tracer<S> {
    require(targetDims.size >= rank) {
        "broadcastTo: target ${targetDims.toList()} has lower rank than ${dims.toList()}"
    }
    val offset = targetDims.size - rank
    for (i in 0 until rank) {
        require(dims[i] == targetDims[offset + i] || dims[i] == 1) {
            "broadcastTo: axis $i of ${dims.toList()} cannot broadcast to ${targetDims.toList()}"
        }
    }
    if (dims.contentEquals(targetDims)) return this as Tracer<S>
    val aligned: Tracer<Shape> =
        if (offset == 0) this as Tracer<Shape>
        else reshape(IntArray(targetDims.size) { if (it < offset) 1 else dims[it - offset] })
    val template = DTensor<Shape, F32>(HostF32Storage(FloatArray(sizeOfDims(targetDims))), targetDims.copyOf(), F32)
    val out = broadcastToLike(aligned.toDTensor(), template).hostF32()
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(aligned.id),
        targetDims.copyOf(),
        out,
        attrs = mapOf("broadcast_dimensions" to targetDims.indices.toList()),
    )
    return Tracer<Shape>(tape, e) as Tracer<S>
}

/**
 * A tensor of the receiver's shape and dtype with every element [value],
 * recorded as one scalar constant and a BROADCAST. [constantLike] records a
 * full-size constant instead, which the StableHLO emitter writes out element
 * by element; use this for scales and fills on large activations.
 */
fun <S : Shape> Tracer<S>.splat(value: Float): Tracer<S> {
    var scalar: Tracer<io.tlaloc.core.ScalarShape> = constant(value)
    if (dtype == io.tlaloc.core.BF16) scalar = scalar.cast(io.tlaloc.core.BF16)
    val e = tape.op(
        OpKind.BROADCAST,
        intArrayOf(scalar.id),
        dims.copyOf(),
        FloatArray(size) { scalar.entry.value[0] },
        attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
    )
    return Tracer(tape, e)
}

/**
 * A constant of the given [dims] in the receiver's dtype: an F32 constant
 * leaf, cast to BF16 when the receiver is BF16, so it can combine with
 * the receiver inside a mixed-precision trace.
 */
fun <S : Shape> Tracer<*>.constantMatching(values: FloatArray, dims: IntArray): Tracer<S> {
    val c: Tracer<S> = constant(values, dims)
    return if (dtype == io.tlaloc.core.BF16) c.cast(io.tlaloc.core.BF16) else c
}

/**
 * `log(softmax(x))` along [axis], in the stable form
 * `x − m − log Σ exp(x − m)` with `m` the axis max.
 */
@Suppress("UNCHECKED_CAST")
fun <S : Shape> Tracer<S>.logSoftmax(axis: Int = -1): Tracer<S> {
    val a = if (axis < 0) axis + rank else axis
    require(rank >= 1 && a in 0 until rank) { "logSoftmax: axis $axis out of range for rank $rank" }
    val keep = IntArray(rank) { if (it == a) 1 else dims[it] }
    val m: Tracer<Shape> = max<Shape>(intArrayOf(a)).reshape<Shape>(keep).broadcastTo(dims)
    val shifted = (this as Tracer<Shape>) - m
    val lse: Tracer<Shape> = shifted.exp().sum<Shape>(intArrayOf(a)).log().reshape<Shape>(keep).broadcastTo(dims)
    return (shifted - lse) as Tracer<S>
}

private fun Tracer<*>.toF32Tensor(): DTensor<Shape, F32> =
    DTensor(HostF32Storage(entry.value.copyOf()), dims.copyOf(), F32)

private fun sizeOfDims(dims: IntArray): Int = dims.fold(1) { a, d -> a * d }
