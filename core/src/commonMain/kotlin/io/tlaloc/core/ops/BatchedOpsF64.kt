package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.LinalgKernels
import io.tlaloc.core.F64
import io.tlaloc.core.hostF64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.Shape

/**
 * The F64 twin of [matmulBatched] over F32 tensors (see there). Canonical batched matrix product: `(B0..Bk, M, K) x (B0..Bk, K, N) -> (B0..Bk, M, N)`,
 * every leading axis a batch axis. The host twin of a DXIR `MATMUL` whose operands have
 * rank 3 or more (the DXIR interpreter's arm and `stablehlo.dot_general` with batching
 * dimensions); the K2 plugin synthesizes it for the batched programs `vmap` produces.
 * Each batch element runs the loop of the rank-2 [matmul], so a batch of one gives
 * the same bits.
 *
 * Star-projected operands plus a result-shape witness [R], the calling convention of
 * [plusBroadcast]: the batched ranks have no typed overload.
 */
@ExperimentalTlalocApi
fun <R : Shape> matmulBatched(a: DTensor<*, F64>, b: DTensor<*, F64>): DTensor<R, F64> {
    val r = a.dims.size
    require(r >= 2 && b.dims.size == r) {
        "matmulBatched: operands must have the same rank, at least 2; got ${a.dims.toList()} x ${b.dims.toList()}"
    }
    for (axis in 0 until r - 2) {
        require(a.dims[axis] == b.dims[axis]) {
            "matmulBatched: batch axis $axis differs: ${a.dims.toList()} x ${b.dims.toList()}"
        }
    }
    val m = a.dims[r - 2]
    val k = a.dims[r - 1]
    val n = b.dims[r - 1]
    require(b.dims[r - 2] == k) { "matmulBatched: inner dim mismatch: ${a.dims.toList()} x ${b.dims.toList()}" }
    var batch = 1
    for (axis in 0 until r - 2) batch *= a.dims[axis]
    val av = a.hostF64()
    val bv = b.hostF64()
    val out = DoubleArray(batch * m * n)
    for (s in 0 until batch) {
        val aBase = s * m * k
        val bBase = s * k * n
        val oBase = s * m * n
        for (i in 0 until m) {
            for (p in 0 until k) {
                val aip = av[aBase + i * k + p]
                if (aip == 0.0) continue
                val rowOff = oBase + i * n
                val bOff = bBase + p * n
                for (j in 0 until n) {
                    out[rowOff + j] += aip * bv[bOff + j]
                }
            }
        }
    }
    val outDims = a.dims.copyOf()
    outDims[r - 1] = n
    return DTensor(HostF64Storage(out), outDims, F64)
}

/**
 * The first [axis] axes of [x] kept, the rest flattened into one (row-major, so the data
 * is unchanged): `[B, m, n]` with `axis = 1` gives `[B, m * n]`. The host twin of the
 * `RESHAPE` a batched `flatten()` becomes under `vmap`.
 */
@ExperimentalTlalocApi
fun <R : Shape> flattenFrom(x: DTensor<*, F64>, axis: Int): DTensor<R, F64> {
    require(axis in 0 until x.dims.size) { "flattenFrom: axis $axis outside rank ${x.dims.size}" }
    var rest = 1
    for (k in axis until x.dims.size) rest *= x.dims[k]
    return DTensor(HostF64Storage(x.hostF64().copyOf()), x.dims.copyOfRange(0, axis) + rest, F64)
}

/**
 * [kernel] on each `rows x cols` matrix along the leading (batch) axes of [x], with the
 * matching matrices of [others] (each `n x k`, given as its per-matrix size), widened to
 * Double as the rank-2 linalg host twins do.
 */
private fun perMatrixF64(
    x: DTensor<*, F64>,
    rows: Int,
    cols: Int,
    others: List<Pair<DTensor<*, F64>, Int>>,
    outSize: Int,
    kernel: (DoubleArray, List<DoubleArray>) -> DoubleArray,
): DoubleArray {
    val xv = x.hostF64()
    val count = xv.size / (rows * cols)
    val ov = others.map { (t, n) -> t.hostF64() to n }
    val out = DoubleArray(count * outSize)
    for (i in 0 until count) {
        val v = xv.copyOfRange(i * rows * cols, (i + 1) * rows * cols)
        val parts = ov.map { (o, n) -> o.copyOfRange(i * n, (i + 1) * n).let { w -> DoubleArray(w.size) { j -> w[j].toDouble() } } }
        val d = kernel(DoubleArray(v.size) { j -> v[j].toDouble() }, parts)
        for (j in 0 until outSize) out[i * outSize + j] = d[j]
    }
    return out
}

private fun squareLastF64(x: DTensor<*, F64>, what: String): Int {
    val r = x.dims.size
    require(r >= 2 && x.dims[r - 1] == x.dims[r - 2]) {
        "$what: square matrices along leading batch axes required; got ${x.dims.toList()}"
    }
    return x.dims[r - 1]
}

/** [cholesky] of each matrix along the leading (batch) axes of [a]: the host twin of a batched `CHOLESKY`. */
@ExperimentalTlalocApi
fun <R : Shape> choleskyBatched(a: DTensor<*, F64>): DTensor<R, F64> {
    val n = squareLastF64(a, "choleskyBatched")
    val out = perMatrixF64(a, n, n, emptyList(), n * n) { m, _ -> LinalgKernels.cholesky(m, n) }
    return DTensor(HostF64Storage(out), a.dims.copyOf(), F64)
}

/** [triangularSolve] per matrix along the leading (batch) axes: the host twin of a batched `TRIANGULAR_SOLVE`. */
@ExperimentalTlalocApi
fun <R : Shape> triangularSolveBatched(
    a: DTensor<*, F64>,
    b: DTensor<*, F64>,
    lower: Boolean,
    transposeA: Boolean,
    unitDiagonal: Boolean,
): DTensor<R, F64> {
    val n = squareLastF64(a, "triangularSolveBatched")
    val r = a.dims.size
    require(b.dims.size == r && b.dims[r - 2] == n && b.dims.copyOfRange(0, r - 2).contentEquals(a.dims.copyOfRange(0, r - 2))) {
        "triangularSolveBatched: B must be [..., $n, k] with A's leading axes; got ${b.dims.toList()}"
    }
    val k = b.dims[r - 1]
    val out = perMatrixF64(a, n, n, listOf(b to n * k), n * k) { m, o ->
        LinalgKernels.triangularSolve(m, o[0], n, k, lower, transposeA, unitDiagonal)
    }
    return DTensor(HostF64Storage(out), b.dims.copyOf(), F64)
}

/** [scaleTriangles] per matrix along the leading (batch) axes: the host twin of a batched `TRIANGLE`. */
@ExperimentalTlalocApi
fun <R : Shape> scaleTrianglesBatched(x: DTensor<*, F64>, lower: Double, diagonal: Double, upper: Double): DTensor<R, F64> {
    val r = x.dims.size
    require(r >= 2) { "scaleTrianglesBatched: rank 2 or more required; got ${x.dims.toList()}" }
    val rows = x.dims[r - 2]
    val cols = x.dims[r - 1]
    val out = perMatrixF64(x, rows, cols, emptyList(), rows * cols) { m, _ ->
        LinalgKernels.triangle(m, rows, cols, lower.toDouble(), diagonal.toDouble(), upper.toDouble())
    }
    return DTensor(HostF64Storage(out), x.dims.copyOf(), F64)
}

/** [solve] per matrix along the leading (batch) axes: the host twin of a batched `SOLVE`. */
@ExperimentalTlalocApi
fun <R : Shape> solveBatched(a: DTensor<*, F64>, b: DTensor<*, F64>, transposeA: Boolean): DTensor<R, F64> {
    val n = squareLastF64(a, "solveBatched")
    val r = a.dims.size
    require(b.dims.size == r && b.dims[r - 2] == n && b.dims.copyOfRange(0, r - 2).contentEquals(a.dims.copyOfRange(0, r - 2))) {
        "solveBatched: B must be [..., $n, k] with A's leading axes; got ${b.dims.toList()}"
    }
    val k = b.dims[r - 1]
    val out = perMatrixF64(a, n, n, listOf(b to n * k), n * k) { m, o -> LinalgKernels.solve(m, o[0], n, k, transposeA) }
    return DTensor(HostF64Storage(out), b.dims.copyOf(), F64)
}

/** [det] per matrix along the leading (batch) axes: the host twin of a batched `DET`, one value per matrix. */
@ExperimentalTlalocApi
fun <R : Shape> detBatched(a: DTensor<*, F64>): DTensor<R, F64> {
    val n = squareLastF64(a, "detBatched")
    val out = perMatrixF64(a, n, n, emptyList(), 1) { m, _ -> doubleArrayOf(LinalgKernels.det(m, n)) }
    return DTensor(HostF64Storage(out), a.dims.copyOfRange(0, a.dims.size - 2), F64)
}
