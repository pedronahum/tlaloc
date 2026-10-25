package io.tlaloc.core

/**
 * Index selection shared by the host ops in `:core/ops` and the `:ir` interpreters'
 * `TOP_K`, `TAKE_ALONG` and `TAKE_ALONG_GRAD` arms, in Double so the F32 and F64 engines
 * agree.
 *
 * `takeAlong(x, idx)` with `x` of extents `[G, N, R…]` and `idx` of extents `[G, M]` is
 * `out[g, j, …] = x[g, idx[g, j], …]`, extents `[G, M, R…]`: a gather along axis 1 with
 * axis 0 as a batch. `takeAlongGrad` is its adjoint, the scatter-add back to `[G, N, R…]`.
 */
object SelectionKernels {

    /**
     * The positions of the [k] largest entries of each row along the last axis of [x]
     * (extents [dims]), largest first; equal values in increasing position. Extents
     * `dims` with the last replaced by [k].
     */
    fun topK(x: DoubleArray, dims: IntArray, k: Int): IntArray {
        require(dims.isNotEmpty()) { "topK: needs an axis" }
        val n = dims.last()
        require(k in 1..n) { "topK: k = $k outside 1..$n" }
        val rows = x.size / n
        val out = IntArray(rows * k)
        val order = arrayOfNulls<Int>(n)
        for (r in 0 until rows) {
            for (j in 0 until n) order[j] = j
            val base = r * n
            val sorted = order.sortedWith { a, b ->
                val c = x[base + b!!].compareTo(x[base + a!!])
                if (c != 0) c else a.compareTo(b)
            }
            for (j in 0 until k) out[r * k + j] = sorted[j]!!
        }
        return out
    }

    /** `x[g, idx[g, j], …]`; [xDims] `[G, N, R…]`, [idx] `G × M` row-major. */
    fun takeAlong(x: DoubleArray, xDims: IntArray, idx: IntArray, m: Int): DoubleArray {
        require(xDims.size >= 2) { "takeAlong: x must be at least rank 2 (got ${xDims.toList()})" }
        val g = xDims[0]
        val n = xDims[1]
        val inner = xDims.drop(2).fold(1) { a, b -> a * b }
        require(idx.size == g * m) { "takeAlong: ${idx.size} indices for $g rows of $m" }
        val out = DoubleArray(g * m * inner)
        for (b in 0 until g) {
            for (j in 0 until m) {
                val i = idx[b * m + j]
                require(i in 0 until n) { "takeAlong: index $i outside 0..${n - 1}" }
                x.copyInto(out, (b * m + j) * inner, (b * n + i) * inner, (b * n + i + 1) * inner)
            }
        }
        return out
    }

    /** The adjoint of [takeAlong]: [upstream] `[G, M, R…]` added into zeros of [xDims]. */
    fun takeAlongGrad(upstream: DoubleArray, idx: IntArray, m: Int, xDims: IntArray): DoubleArray {
        val g = xDims[0]
        val n = xDims[1]
        val inner = xDims.drop(2).fold(1) { a, b -> a * b }
        require(upstream.size == g * m * inner) { "takeAlongGrad: upstream has ${upstream.size} elements" }
        val out = DoubleArray(g * n * inner)
        for (b in 0 until g) {
            for (j in 0 until m) {
                val i = idx[b * m + j]
                require(i in 0 until n) { "takeAlongGrad: index $i outside 0..${n - 1}" }
                val src = (b * m + j) * inner
                val dst = (b * n + i) * inner
                for (e in 0 until inner) out[dst + e] += upstream[src + e]
            }
        }
        return out
    }
}
