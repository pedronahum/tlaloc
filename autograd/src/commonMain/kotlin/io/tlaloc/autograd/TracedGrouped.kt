package io.tlaloc.autograd

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.ir.OpKind

/**
 * Matrix products over consecutive groups of rows: this `[M, K]` times [rhs] `[G, K, N]`, where
 * the first `sizes[0]` rows are multiplied by `rhs[0]`, the next `sizes[1]` by `rhs[1]`, and so
 * on ([sizes] is I32 `[G]`); rows after the last group are zero. The grouped matmul of a mixture
 * of experts whose rows are sorted by expert. Records [OpKind.RAGGED_DOT], which XLA emits as
 * `chlo.ragged_dot` (a TPU's grouped matmul); differentiable in this and [rhs].
 */
fun Tracer<*>.raggedDot(rhs: Tracer<*>, sizes: Tracer<*>): Tracer<Shape> {
    require(rank == 2 && dtype == F32) { "raggedDot: lhs must be f32 [M, K] (got ${dtype.name} ${dims.toList()})" }
    require(sizes.dtype == I32 && sizes.rank == 1) { "raggedDot: sizes must be i32 [G] (got ${sizes.dtype.name} ${sizes.dims.toList()})" }
    val (m, k) = dims[0] to dims[1]
    val g = sizes.dims[0]
    require(rhs.dtype == F32 && rhs.rank == 3 && rhs.dims[0] == g && rhs.dims[1] == k) {
        "raggedDot: rhs must be f32 [$g, $k, N] (got ${rhs.dtype.name} ${rhs.dims.toList()})"
    }
    val n = rhs.dims[2]
    val tape = sameTape(this, rhs).also { sameTape(this, sizes) }
    val lhsV = entry.value
    val rhsV = rhs.entry.value
    val out = FloatArray(m * n)
    var row = 0
    for (group in 0 until g) {
        val count = sizes.entry.value[group].toInt()
        require(count >= 0) { "raggedDot: sizes must be non-negative (group $group has $count)" }
        repeat(count) {
            if (row < m) {
                for (j in 0 until n) {
                    var acc = 0.0
                    for (l in 0 until k) acc += lhsV[row * k + l].toDouble() * rhsV[(group * k + l) * n + j]
                    out[row * n + j] = acc.toFloat()
                }
            }
            row++
        }
    }
    val e = tape.op(OpKind.RAGGED_DOT, intArrayOf(id, rhs.id, sizes.id), intArrayOf(m, n), out)
    return Tracer(tape, e)
}
