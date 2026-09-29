package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.LinalgKernels
import io.tlaloc.core.F32
import io.tlaloc.core.hostF32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape

/**
 * Canonical batched matrix product: `(B0..Bk, M, K) x (B0..Bk, K, N) -> (B0..Bk, M, N)`,
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
fun <R : Shape> matmulBatched(a: DTensor<*, F32>, b: DTensor<*, F32>): DTensor<R, F32> {
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
    val av = a.hostF32()
    val bv = b.hostF32()
    val out = FloatArray(batch * m * n)
    for (s in 0 until batch) {
        val aBase = s * m * k
        val bBase = s * k * n
        val oBase = s * m * n
        for (i in 0 until m) {
            for (p in 0 until k) {
                val aip = av[aBase + i * k + p]
                if (aip == 0f) continue
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
    return DTensor(HostF32Storage(out), outDims, F32)
}

/**
 * The first [axis] axes of [x] kept, the rest flattened into one (row-major, so the data
 * is unchanged): `[B, m, n]` with `axis = 1` gives `[B, m * n]`. The host twin of the
 * `RESHAPE` a batched `flatten()` becomes under `vmap`.
 */
@ExperimentalTlalocApi
fun <R : Shape> flattenFrom(x: DTensor<*, F32>, axis: Int): DTensor<R, F32> {
    require(axis in 0 until x.dims.size) { "flattenFrom: axis $axis outside rank ${x.dims.size}" }
    var rest = 1
    for (k in axis until x.dims.size) rest *= x.dims[k]
    return DTensor(HostF32Storage(x.hostF32().copyOf()), x.dims.copyOfRange(0, axis) + rest, F32)
}

/**
 * [kernel] on each `rows x cols` matrix along the leading (batch) axes of [x], with the
 * matching matrices of [others] (each `n x k`, given as its per-matrix size), widened to
 * Double as the rank-2 linalg host twins do.
 */
private fun perMatrixF32(
    x: DTensor<*, F32>,
    rows: Int,
    cols: Int,
    others: List<Pair<DTensor<*, F32>, Int>>,
    outSize: Int,
    kernel: (DoubleArray, List<DoubleArray>) -> DoubleArray,
): FloatArray {
    val xv = x.hostF32()
    var count = 1
    for (axis in 0 until x.dims.size - 2) count *= x.dims[axis]
    val ov = others.map { (t, n) -> t.hostF32() to n }
    val out = FloatArray(count * outSize)
    for (i in 0 until count) {
        val v = xv.copyOfRange(i * rows * cols, (i + 1) * rows * cols)
        val parts = ov.map { (o, n) -> o.copyOfRange(i * n, (i + 1) * n).let { w -> DoubleArray(w.size) { j -> w[j].toDouble() } } }
        val d = kernel(DoubleArray(v.size) { j -> v[j].toDouble() }, parts)
        for (j in 0 until outSize) out[i * outSize + j] = d[j].toFloat()
    }
    return out
}

private fun squareLastF32(x: DTensor<*, F32>, what: String): Int {
    val r = x.dims.size
    require(r >= 2 && x.dims[r - 1] == x.dims[r - 2]) {
        "$what: square matrices along leading batch axes required; got ${x.dims.toList()}"
    }
    return x.dims[r - 1]
}

/** [cholesky] of each matrix along the leading (batch) axes of [a]: the host twin of a batched `CHOLESKY`. */
@ExperimentalTlalocApi
fun <R : Shape> choleskyBatched(a: DTensor<*, F32>): DTensor<R, F32> {
    val n = squareLastF32(a, "choleskyBatched")
    val out = perMatrixF32(a, n, n, emptyList(), n * n) { m, _ -> LinalgKernels.cholesky(m, n) }
    return DTensor(HostF32Storage(out), a.dims.copyOf(), F32)
}

/** [triangularSolve] per matrix along the leading (batch) axes: the host twin of a batched `TRIANGULAR_SOLVE`. */
@ExperimentalTlalocApi
fun <R : Shape> triangularSolveBatched(
    a: DTensor<*, F32>,
    b: DTensor<*, F32>,
    lower: Boolean,
    transposeA: Boolean,
    unitDiagonal: Boolean,
): DTensor<R, F32> {
    val n = squareLastF32(a, "triangularSolveBatched")
    val r = a.dims.size
    require(b.dims.size == r && b.dims[r - 2] == n && b.dims.copyOfRange(0, r - 2).contentEquals(a.dims.copyOfRange(0, r - 2))) {
        "triangularSolveBatched: B must be [..., $n, k] with A's leading axes; got ${b.dims.toList()}"
    }
    val k = b.dims[r - 1]
    val out = perMatrixF32(a, n, n, listOf(b to n * k), n * k) { m, o ->
        LinalgKernels.triangularSolve(m, o[0], n, k, lower, transposeA, unitDiagonal)
    }
    return DTensor(HostF32Storage(out), b.dims.copyOf(), F32)
}

/** [scaleTriangles] per matrix along the leading (batch) axes: the host twin of a batched `TRIANGLE`. */
@ExperimentalTlalocApi
fun <R : Shape> scaleTrianglesBatched(x: DTensor<*, F32>, lower: Float, diagonal: Float, upper: Float): DTensor<R, F32> {
    val r = x.dims.size
    require(r >= 2) { "scaleTrianglesBatched: rank 2 or more required; got ${x.dims.toList()}" }
    val rows = x.dims[r - 2]
    val cols = x.dims[r - 1]
    val out = perMatrixF32(x, rows, cols, emptyList(), rows * cols) { m, _ ->
        LinalgKernels.triangle(m, rows, cols, lower.toDouble(), diagonal.toDouble(), upper.toDouble())
    }
    return DTensor(HostF32Storage(out), x.dims.copyOf(), F32)
}

/** [solve] per matrix along the leading (batch) axes: the host twin of a batched `SOLVE`. */
@ExperimentalTlalocApi
fun <R : Shape> solveBatched(a: DTensor<*, F32>, b: DTensor<*, F32>, transposeA: Boolean): DTensor<R, F32> {
    val n = squareLastF32(a, "solveBatched")
    val r = a.dims.size
    require(b.dims.size == r && b.dims[r - 2] == n && b.dims.copyOfRange(0, r - 2).contentEquals(a.dims.copyOfRange(0, r - 2))) {
        "solveBatched: B must be [..., $n, k] with A's leading axes; got ${b.dims.toList()}"
    }
    val k = b.dims[r - 1]
    val out = perMatrixF32(a, n, n, listOf(b to n * k), n * k) { m, o -> LinalgKernels.solve(m, o[0], n, k, transposeA) }
    return DTensor(HostF32Storage(out), b.dims.copyOf(), F32)
}

/** [det] per matrix along the leading (batch) axes: the host twin of a batched `DET`, one value per matrix. */
@ExperimentalTlalocApi
fun <R : Shape> detBatched(a: DTensor<*, F32>): DTensor<R, F32> {
    val n = squareLastF32(a, "detBatched")
    val out = perMatrixF32(a, n, n, emptyList(), 1) { m, _ -> doubleArrayOf(LinalgKernels.det(m, n)) }
    return DTensor(HostF32Storage(out), a.dims.copyOfRange(0, a.dims.size - 2), F32)
}

private fun qrShapeF32(a: DTensor<*, F32>, what: String): Pair<Int, Int> {
    val r = a.dims.size
    require(r >= 2 && a.dims[r - 2] >= a.dims[r - 1]) {
        "$what: matrices along leading batch axes with rows ≥ columns required; got ${a.dims.toList()}"
    }
    return a.dims[r - 2] to a.dims[r - 1]
}

/** [qrQ] per matrix along the leading (batch) axes: the host twin of a batched `QR_Q`. */
@ExperimentalTlalocApi
fun <R : Shape> qrQBatched(a: DTensor<*, F32>): DTensor<R, F32> {
    val (m, n) = qrShapeF32(a, "qrQBatched")
    val out = perMatrixF32(a, m, n, emptyList(), m * n) { x, _ -> LinalgKernels.qr(x, m, n).first }
    return DTensor(HostF32Storage(out), a.dims.copyOf(), F32)
}

/** [qrR] per matrix along the leading (batch) axes: the host twin of a batched `QR_R`. */
@ExperimentalTlalocApi
fun <R : Shape> qrRBatched(a: DTensor<*, F32>): DTensor<R, F32> {
    val (m, n) = qrShapeF32(a, "qrRBatched")
    val out = perMatrixF32(a, m, n, emptyList(), n * n) { x, _ -> LinalgKernels.qr(x, m, n).second }
    return DTensor(HostF32Storage(out), a.dims.copyOfRange(0, a.dims.size - 2) + intArrayOf(n, n), F32)
}

/** [eighValues] per matrix along the leading (batch) axes: the host twin of a batched `EIGH_W`. */
@ExperimentalTlalocApi
fun <R : Shape> eighValuesBatched(a: DTensor<*, F32>): DTensor<R, F32> {
    val n = squareLastF32(a, "eighValuesBatched")
    val out = perMatrixF32(a, n, n, emptyList(), n) { x, _ -> LinalgKernels.eigh(x, n).first }
    return DTensor(HostF32Storage(out), a.dims.copyOfRange(0, a.dims.size - 1), F32)
}

/** [eighVectors] per matrix along the leading (batch) axes: the host twin of a batched `EIGH_V`. */
@ExperimentalTlalocApi
fun <R : Shape> eighVectorsBatched(a: DTensor<*, F32>): DTensor<R, F32> {
    val n = squareLastF32(a, "eighVectorsBatched")
    val out = perMatrixF32(a, n, n, emptyList(), n * n) { x, _ -> LinalgKernels.eigh(x, n).second }
    return DTensor(HostF32Storage(out), a.dims.copyOf(), F32)
}

/**
 * `[..., m, k] x [k, n] -> [..., m, n]`: a matrix product with the rank-2 [w] shared by every
 * leading index of [a] — the host twin of a `MATMUL` of a batched lhs and a rank-2 rhs, which
 * `vmap` emits for `x · W` with W not batched. Each leading index runs the loop of the rank-2
 * `matmul`, so it gives that function's bits.
 */
@ExperimentalTlalocApi
fun <R : Shape> matmulSharedRhs(a: DTensor<*, F32>, w: DTensor<*, F32>): DTensor<R, F32> {
    val r = a.dims.size
    require(r >= 2 && w.dims.size == 2 && a.dims[r - 1] == w.dims[0]) {
        "matmulSharedRhs: expected [..., m, k] x [k, n]; got ${a.dims.toList()} x ${w.dims.toList()}"
    }
    val m = a.dims[r - 2]
    val k = a.dims[r - 1]
    val n = w.dims[1]
    var batch = 1
    for (axis in 0 until r - 2) batch *= a.dims[axis]
    val av = a.hostF32()
    val wv = w.hostF32()
    val out = FloatArray(batch * m * n)
    for (s in 0 until batch) {
        val aBase = s * m * k
        val oBase = s * m * n
        for (i in 0 until m) {
            for (p in 0 until k) {
                val aip = av[aBase + i * k + p]
                if (aip == 0f) continue
                val rowOff = oBase + i * n
                val wOff = p * n
                for (j in 0 until n) {
                    out[rowOff + j] += aip * wv[wOff + j]
                }
            }
        }
    }
    val outDims = a.dims.copyOf()
    outDims[r - 1] = n
    return DTensor(HostF32Storage(out), outDims, F32)
}

/**
 * [count] consecutive axes of [x] from axis [from] merged into one (row-major, so the data is
 * unchanged): `[B, m, k]` with `from = 0, count = 2` gives `[B * m, k]`. The host twin of the
 * `RESHAPE` a shared-rhs matmul's gradient folds its leading axes with.
 */
@ExperimentalTlalocApi
fun <R : Shape> mergeAxes(x: DTensor<*, F32>, from: Int, count: Int): DTensor<R, F32> {
    require(from >= 0 && count >= 1 && from + count <= x.dims.size) {
        "mergeAxes: axes $from until ${from + count} outside rank ${x.dims.size}"
    }
    var merged = 1
    for (axis in from until from + count) merged *= x.dims[axis]
    val dims = x.dims.copyOfRange(0, from) + intArrayOf(merged) + x.dims.copyOfRange(from + count, x.dims.size)
    return DTensor(HostF32Storage(x.hostF32().copyOf()), dims, F32)
}
