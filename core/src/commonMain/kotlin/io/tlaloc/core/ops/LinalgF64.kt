package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.LinalgKernels
import io.tlaloc.core.Rank2
import io.tlaloc.core.ShapeAtom

// F64 overloads of Linalg.kt, computed in Double throughout. They are in their own
// file because each erases to the same JVM signature as its F32 twin.
//
// These run on the host. `grad {}` differentiates F32 tensors only (for every op,
// not only these); the F64 derivative rules are exercised through the StableHLO
// path (`PjrtSession.runOnF64`).

private fun f64Data(t: DTensor<*, F64>): DoubleArray {
    val s = t.storage
    require(s is HostF64Storage) { "expected host F64 storage, got ${s::class.simpleName}" }
    return s.data
}

private fun squareDimF64(t: DTensor<*, *>, what: String): Int {
    require(t.rank == 2 && t.dims[0] == t.dims[1]) {
        "$what requires a square rank-2 matrix; got dims ${t.dims.toList()}"
    }
    return t.dims[0]
}

/** F64 [tril]. */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F64>.tril(): DTensor<Rank2<R, C>, F64> =
    scaleTriangles(1.0, 1.0, 0.0)

/** F64 [triu]. */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F64>.triu(): DTensor<Rank2<R, C>, F64> =
    scaleTriangles(0.0, 1.0, 1.0)

/** F64 [scaleTriangles]. */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F64>.scaleTriangles(
    lower: Double,
    diagonal: Double,
    upper: Double,
): DTensor<Rank2<R, C>, F64> {
    require(rank == 2) { "scaleTriangles requires a rank-2 matrix; got dims ${dims.toList()}" }
    val out = LinalgKernels.triangle(f64Data(this), dims[0], dims[1], lower, diagonal, upper)
    return DTensor(HostF64Storage(out), dims.copyOf(), F64)
}

/** F64 [cholesky]. */
fun <N : ShapeAtom> DTensor<Rank2<N, N>, F64>.cholesky(): DTensor<Rank2<N, N>, F64> {
    val n = squareDimF64(this, "cholesky")
    return DTensor(HostF64Storage(LinalgKernels.cholesky(f64Data(this), n)), intArrayOf(n, n), F64)
}

/** F64 [triangularSolve]. */
fun <N : ShapeAtom, K : ShapeAtom> DTensor<Rank2<N, N>, F64>.triangularSolve(
    b: DTensor<Rank2<N, K>, F64>,
    lower: Boolean,
): DTensor<Rank2<N, K>, F64> = triangularSolve(b, lower, false, false)

/** F64 [triangularSolve]. */
fun <N : ShapeAtom, K : ShapeAtom> DTensor<Rank2<N, N>, F64>.triangularSolve(
    b: DTensor<Rank2<N, K>, F64>,
    lower: Boolean,
    transposeA: Boolean,
    unitDiagonal: Boolean,
): DTensor<Rank2<N, K>, F64> {
    val n = squareDimF64(this, "triangularSolve")
    require(b.rank == 2 && b.dims[0] == n) {
        "triangularSolve: B must be $n×k for a $n×$n A; got dims ${b.dims.toList()}"
    }
    val m = b.dims[1]
    val x = LinalgKernels.triangularSolve(f64Data(this), f64Data(b), n, m, lower, transposeA, unitDiagonal)
    return DTensor(HostF64Storage(x), intArrayOf(n, m), F64)
}

/** F64 [solveSpd]. */
fun <N : ShapeAtom, K : ShapeAtom> DTensor<Rank2<N, N>, F64>.solveSpd(
    b: DTensor<Rank2<N, K>, F64>,
): DTensor<Rank2<N, K>, F64> {
    val l = cholesky()
    return l.triangularSolve(l.triangularSolve(b, true, false, false), true, true, false)
}

/** F64 [logDetSpd]. */
fun <N : ShapeAtom> DTensor<Rank2<N, N>, F64>.logDetSpd(): DTensor<io.tlaloc.core.ScalarShape, F64> {
    val n = squareDimF64(this, "logDetSpd")
    val l = LinalgKernels.cholesky(f64Data(this), n)
    var s = 0.0
    for (i in 0 until n) s += kotlin.math.ln(l[i * n + i])
    return DTensor(HostF64Storage(doubleArrayOf(2 * s)), intArrayOf(), F64)
}
