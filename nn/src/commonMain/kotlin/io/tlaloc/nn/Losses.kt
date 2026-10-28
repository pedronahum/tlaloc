package io.tlaloc.nn

import io.tlaloc.core.BF16
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.after
import io.tlaloc.autograd.cast
import io.tlaloc.autograd.div
import io.tlaloc.autograd.logSoftmax
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.neg
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.recompute
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.softCrossEntropy
import io.tlaloc.autograd.splat
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.takeAlong
import io.tlaloc.autograd.times

/**
 * Cross-entropy of [logits] `[..., classes]` against [targets], either target
 * distributions of the same shape (one-hot rows from [oneHot], or smoothed
 * labels): `−Σ targets · logSoftmax(logits) / Σ targets`, or I32 class ids
 * `[...]`: the mean of `−logSoftmax(logits)[id]` over the positions whose id
 * is not negative.
 *
 * With one-hot rows the divisor is the number of rows that have a target,
 * so rows left all-zero by [oneHot]'s `ignoreIndex` add nothing and are not
 * counted. Both forms are PyTorch's `cross_entropy(logits, ids,
 * ignore_index=-100)` with `reduction="mean"`; ids are a `[...]` input where
 * one-hot rows are `[..., classes]`.
 */
fun crossEntropy(logits: Tracer<Shape>, targets: Tracer<Shape>): Tracer<ScalarShape> {
    if (targets.dtype == I32) {
        require(logits.rank >= 1 && targets.dims.contentEquals(logits.dims.copyOf(logits.rank - 1))) {
            "crossEntropy: ids ${targets.dims.toList()} must be the logits' ${logits.dims.toList()} without the class axis"
        }
        val classes = logits.dims[logits.rank - 1]
        val rows = logits.size / classes
        val (ids, valid) = idsAndMask(targets, rows)
        return pickedLogProbs(logits.reshape(intArrayOf(rows, classes)), ids, valid).neg() / valid.sum()
    }
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
 * The cross-entropy of an LM head's logits `hidden · weight` against the I32 [ids], computed
 * [chunk] rows at a time, so that a rematerialized step never holds more than one chunk's
 * `[chunk, vocab]` logits: [hidden] is `[..., d]`, [weight] `[d, vocab]`, [ids] `[...]` with
 * negative ids ignored. The value is [crossEntropy] of the full logits.
 *
 * Each chunk starts after the previous chunk's loss ([after]). Its logits and log-probabilities
 * are marked [recompute]: under
 * [CapturedStep.withRemat] the backward pass computes them again chunk by chunk instead of keeping
 * them all from the forward pass; [io.tlaloc.ir.passes.DxirRemat.SAVE_ALL] recomputes only them.
 * Without rematerialization the gradient keeps every chunk, as the unchunked loss does. Logits
 * of a bf16 matmul are cast to f32 first.
 */
fun lmHeadCrossEntropy(hidden: Tracer<Shape>, weight: Tracer<Shape>, ids: Tracer<Shape>, chunk: Int): Tracer<ScalarShape> {
    require(weight.rank == 2 && hidden.rank >= 1 && hidden.dims[hidden.rank - 1] == weight.dims[0]) {
        "lmHeadCrossEntropy: weight ${weight.dims.toList()} must be [d, vocab] for hidden ${hidden.dims.toList()}"
    }
    require(ids.dtype == I32 && ids.dims.contentEquals(hidden.dims.copyOf(hidden.rank - 1))) {
        "lmHeadCrossEntropy: ids must be I32 ${hidden.dims.copyOf(hidden.rank - 1).toList()} (got ${ids.dtype.name} ${ids.dims.toList()})"
    }
    require(chunk >= 1) { "lmHeadCrossEntropy: chunk must be positive; got $chunk" }
    val d = weight.dims[0]
    val rows = hidden.size / d
    val flat = hidden.reshape<Shape>(intArrayOf(rows, d))
    val (positions, valid) = idsAndMask(ids, rows)
    @Suppress("UNCHECKED_CAST")
    val w = weight as Tracer<Rank2<Sym, Sym>>
    var total: Tracer<ScalarShape>? = null
    for (start in 0 until rows step chunk) {
        val end = minOf(rows, start + chunk)
        @Suppress("UNCHECKED_CAST")
        // Each chunk starts after the previous one's loss: XLA would otherwise be free to run the
        // chunks together, or merge their products back into one.
        val h = flat.slice<Shape>(start, end, 0).let { if (total == null) it else it.after(total!!) } as Tracer<Rank2<Sym, Sym>>
        val product = (h matmul w) as Tracer<Shape>
        val logits = (if (product.dtype == BF16) product.cast(F32) else product).recompute()
        val part = pickedLogProbs(logits, positions.slice(start, end, 0), valid.slice(start, end, 0), recomputed = true)
        total = total?.let { it + part } ?: part
    }
    return total!!.neg() / valid.sum()
}

/** Ids `[rows, 1]` with the negative ones set to 0, and the mask `[rows, 1]` of the others. */
private fun idsAndMask(ids: Tracer<Shape>, rows: Int): Pair<Tracer<Shape>, Tracer<Shape>> {
    val f = ids.cast(F32).reshape<Shape>(intArrayOf(rows, 1))
    // For integers: relu(1 − relu(−id)) is 1 when id ≥ 0 and 0 when id ≤ −1.
    val valid = (f.splat(1f) - f.neg().relu()).relu()
    return f.relu().cast(I32) to valid
}

/** `Σ valid · logSoftmax(logits)[id]` over the rows of [logits] `[rows, classes]`. */
private fun pickedLogProbs(logits: Tracer<Shape>, ids: Tracer<Shape>, valid: Tracer<Shape>, recomputed: Boolean = false): Tracer<ScalarShape> {
    val lp = logits.logSoftmax(-1).let { if (recomputed) it.recompute() else it }
    return (lp.takeAlong(ids) * valid).sum()
}

/**
 * Distillation loss: the soft-target cross entropy of [student] logits `[..., classes]`
 * against [teacher] logits of the same shape at [temperature] `T`,
 * `T² · mean_rows(−Σ softmax(teacher / T) · logSoftmax(student / T))` (Hinton, Vinyals
 * and Dean's scaling, which keeps the gradient's size independent of `T`).
 *
 * The student's gradient `softmax(student / T) − softmax(teacher / T)` is computed from
 * the logit difference in log space, so it does not cancel to zero as the student
 * approaches the teacher (arXiv 2610.31642). Pass the teacher's logits, not its
 * probabilities. The teacher receives a gradient too; trace it as a constant to train
 * the student alone.
 */
fun distillationLoss(
    student: Tracer<Shape>,
    teacher: Tracer<Shape>,
    temperature: Float = 1f,
): Tracer<ScalarShape> {
    require(student.dims.contentEquals(teacher.dims)) {
        "distillationLoss: student ${student.dims.toList()} and teacher ${teacher.dims.toList()} must have the same shape"
    }
    require(student.rank >= 1) { "distillationLoss: logits must have a class axis" }
    require(temperature > 0f) { "distillationLoss: temperature must be positive; got $temperature" }
    val classes = student.dims[student.rank - 1]
    val rows = student.size / classes
    var s = student.reshape<Shape>(intArrayOf(rows, classes))
    var t = teacher.reshape<Shape>(intArrayOf(rows, classes))
    if (temperature != 1f) {
        s = s / s.splat(temperature)
        t = t / t.splat(temperature)
    }
    return s.softCrossEntropy(t, -1).sum() * (temperature * temperature / rows)
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
