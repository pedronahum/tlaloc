package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.div
import io.tlaloc.autograd.logSoftmax
import io.tlaloc.autograd.neg
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times

/**
 * Cross-entropy of [logits] `[..., classes]` against target distributions
 * [targets] of the same shape (one-hot rows from [oneHot], or smoothed
 * labels): `−Σ targets · logSoftmax(logits) / Σ targets`.
 *
 * With one-hot rows the divisor is the number of rows that have a target,
 * so rows left all-zero by [oneHot]'s `ignoreIndex` add nothing and are not
 * counted. This is PyTorch's `cross_entropy(logits, ids, ignore_index=…)`
 * with `reduction="mean"`.
 */
fun crossEntropy(logits: Tracer<Shape>, targets: Tracer<Shape>): Tracer<ScalarShape> {
    require(logits.dims.contentEquals(targets.dims)) {
        "crossEntropy: logits ${logits.dims.toList()} and targets ${targets.dims.toList()} must have the same shape"
    }
    require(logits.rank >= 1) { "crossEntropy: logits must have a class axis" }
    val classes = logits.dims[logits.rank - 1]
    val rows = logits.size / classes
    val flatLogits = logits.reshape<Shape>(intArrayOf(rows, classes))
    val flatTargets = targets.reshape<Shape>(intArrayOf(rows, classes))
    val picked = (flatTargets * flatLogits.logSoftmax(-1)).sum()
    return picked.neg() / flatTargets.sum()
}

/**
 * One-hot target rows for [ids]: a tensor `[*dims, numClasses]` with a 1 at
 * each id. Positions whose id equals [ignoreIndex] get an all-zero row, which
 * [crossEntropy] leaves out. [dims] defaults to `[ids.size]`.
 */
fun oneHot(
    ids: IntArray,
    numClasses: Int,
    dims: IntArray = intArrayOf(ids.size),
    ignoreIndex: Int? = null,
): DTensor<*, F32> {
    require(dims.fold(1) { a, d -> a * d } == ids.size) {
        "oneHot: dims ${dims.toList()} do not hold ${ids.size} ids"
    }
    val out = FloatArray(ids.size * numClasses)
    for ((row, id) in ids.withIndex()) {
        if (id == ignoreIndex) continue
        require(id in 0 until numClasses) { "oneHot: id $id at position $row is outside [0, $numClasses)" }
        out[row * numClasses + id] = 1f
    }
    return DTensor<Shape, F32>(HostF32Storage(out), dims + numClasses, F32)
}
