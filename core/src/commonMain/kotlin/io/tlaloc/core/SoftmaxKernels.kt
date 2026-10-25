package io.tlaloc.core

import kotlin.math.exp
import kotlin.math.ln1p

/**
 * The log-softmax kernel shared by the CPU engines: the F32 and F64 host twins in
 * `:core/ops` and the `:ir` interpreters' `LOG_SOFTMAX` arms. One implementation, in
 * Double, so the host and the interpreters agree bit for bit.
 *
 * Along an axis with maximum `M`,
 *
 * ```
 * logSoftmax(x)_k = (x_k − M) − log1p(S),   S = Σ_{j ≠ i} exp(x_j − M)
 * ```
 *
 * where `i` is one position of the maximum (Barron, Gnanapragasam and Tschannen, "A
 * Fix for Softmax and Cross-Entropy", arXiv 2610.31642). The usual `x − M − log Σ_j
 * exp(x_j − M)` adds the maximum's own `exp(0) = 1` to the sum before the logarithm,
 * which rounds away an `S` below the precision's epsilon: the maximum's log
 * probability `−log1p(S)` then comes out as exactly 0. With ties at the maximum `S`
 * counts each further tied position as `exp(0) = 1`.
 */
object SoftmaxKernels {

    /**
     * Log-softmax of the row-major tensor [x] of extents [dims] along [axis]
     * (non-negative). An axis whose maximum is not finite gives NaN.
     */
    fun logSoftmax(x: DoubleArray, dims: IntArray, axis: Int): DoubleArray {
        val rank = dims.size
        require(axis in 0 until rank) { "logSoftmax: axis $axis out of range for rank $rank" }
        val axisLen = dims[axis]
        var inner = 1
        for (k in axis + 1 until rank) inner *= dims[k]
        var outer = 1
        for (k in 0 until axis) outer *= dims[k]
        require(x.size == outer * axisLen * inner) {
            "logSoftmax: ${x.size} elements for extents ${dims.toList()}"
        }
        val out = DoubleArray(x.size)
        for (o in 0 until outer) {
            for (i in 0 until inner) {
                val base = o * axisLen * inner + i
                var mx = Double.NEGATIVE_INFINITY
                for (j in 0 until axisLen) mx = maxOf(mx, x[base + j * inner])
                var ties = 0
                var rest = 0.0
                for (j in 0 until axisLen) {
                    val v = x[base + j * inner]
                    if (v == mx) ties++ else rest += exp(v - mx)
                }
                val logNorm = if (mx.isFinite()) ln1p((ties - 1) + rest) else Double.NaN
                for (j in 0 until axisLen) out[base + j * inner] = (x[base + j * inner] - mx) - logNorm
            }
        }
        return out
    }
}
