package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.ExperimentalTlalocApi
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
