package io.tlaloc.autograd

import io.tlaloc.core.Rank2
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.cholesky
import io.tlaloc.core.ops.scaleTriangles
import io.tlaloc.core.ops.triangularSolve
import io.tlaloc.ir.OpKind

// The linear-algebra ops of `:core/ops/Linalg.kt` for the capture API. The forward
// value comes from the same host twin the plugin's synthesis calls; the op is
// recorded with the attrs the reverse transform's rules read.

/** Capture-API [io.tlaloc.core.ops.tril]. */
fun <R : ShapeAtom, C : ShapeAtom> Tracer<Rank2<R, C>>.tril(): Tracer<Rank2<R, C>> = scaleTriangles(1f, 1f, 0f)

/** Capture-API [io.tlaloc.core.ops.triu]. */
fun <R : ShapeAtom, C : ShapeAtom> Tracer<Rank2<R, C>>.triu(): Tracer<Rank2<R, C>> = scaleTriangles(0f, 1f, 1f)

/** Capture-API [io.tlaloc.core.ops.scaleTriangles]. */
fun <R : ShapeAtom, C : ShapeAtom> Tracer<Rank2<R, C>>.scaleTriangles(
    lower: Float,
    diagonal: Float,
    upper: Float,
): Tracer<Rank2<R, C>> {
    val out = toDTensor().scaleTriangles(lower, diagonal, upper).hostF32()
    val e = tape.op(
        OpKind.TRIANGLE, intArrayOf(id), dims.copyOf(), out,
        attrs = mapOf("lower" to lower.toDouble(), "diagonal" to diagonal.toDouble(), "upper" to upper.toDouble()),
    )
    return Tracer(tape, e)
}

/** Capture-API [io.tlaloc.core.ops.cholesky]. */
fun <N : ShapeAtom> Tracer<Rank2<N, N>>.cholesky(): Tracer<Rank2<N, N>> {
    val out = toDTensor().cholesky().hostF32()
    val e = tape.op(OpKind.CHOLESKY, intArrayOf(id), dims.copyOf(), out)
    return Tracer(tape, e)
}

/** Capture-API [io.tlaloc.core.ops.triangularSolve]. */
fun <N : ShapeAtom, K : ShapeAtom> Tracer<Rank2<N, N>>.triangularSolve(
    b: Tracer<Rank2<N, K>>,
    lower: Boolean,
): Tracer<Rank2<N, K>> = triangularSolve(b, lower, false, false)

/** Capture-API [io.tlaloc.core.ops.triangularSolve]. */
fun <N : ShapeAtom, K : ShapeAtom> Tracer<Rank2<N, N>>.triangularSolve(
    b: Tracer<Rank2<N, K>>,
    lower: Boolean,
    transposeA: Boolean,
    unitDiagonal: Boolean,
): Tracer<Rank2<N, K>> {
    val tape = sameTape(this, b)
    val out = toDTensor().triangularSolve(b.toDTensor(), lower, transposeA, unitDiagonal)
    val e = tape.op(
        OpKind.TRIANGULAR_SOLVE, intArrayOf(id, b.id), out.dims.copyOf(), out.hostF32(),
        attrs = mapOf("lower" to lower, "transpose_a" to transposeA, "unit_diagonal" to unitDiagonal),
    )
    return Tracer(tape, e)
}

/** Capture-API [io.tlaloc.core.ops.solveSpd]: recorded as `cholesky` and two triangular solves. */
fun <N : ShapeAtom, K : ShapeAtom> Tracer<Rank2<N, N>>.solveSpd(b: Tracer<Rank2<N, K>>): Tracer<Rank2<N, K>> {
    val l = cholesky()
    return l.triangularSolve(l.triangularSolve(b, true, false, false), true, true, false)
}

/** Capture-API [io.tlaloc.core.ops.logDetSpd]: recorded as `cholesky`, the diagonal's row sum, `log` and `sum`. */
fun <N : ShapeAtom> Tracer<Rank2<N, N>>.logDetSpd(): Tracer<io.tlaloc.core.ScalarShape> {
    val half = cholesky().scaleTriangles(0f, 1f, 0f).sum<io.tlaloc.core.Rank1<N>>(intArrayOf(1)).log().sum()
    return half + half
}

/** Capture-API [io.tlaloc.core.ops.identityLike]: a constant leaf. */
fun <N : ShapeAtom> Tracer<Rank2<N, N>>.identityLike(): Tracer<Rank2<N, N>> {
    val n = dims[0]
    require(rank == 2 && dims[1] == n) { "identityLike requires a square rank-2 matrix; got dims ${dims.toList()}" }
    val entry = tape.leaf(dims = dims.copyOf(), value = FloatArray(n * n) { if (it / n == it % n) 1f else 0f }, isConstant = true)
    return Tracer(tape, entry)
}

/** Capture-API [io.tlaloc.core.ops.invSpd]. */
fun <N : ShapeAtom> Tracer<Rank2<N, N>>.invSpd(): Tracer<Rank2<N, N>> = solveSpd(identityLike())
