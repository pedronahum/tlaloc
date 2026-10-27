package io.tlaloc.ir

import io.tlaloc.core.I32

/**
 * [OpKind.RAGGED_DOT]: matrix products over consecutive groups of rows, one parser for every
 * layer. The groups are given by `sizes [G]` (i32): group g is the `sizes[g]` rows after the
 * first `sizes[0] + … + sizes[g−1]`. Rows after the last group belong to none.
 *
 * Attribute `ragged`:
 * - `"rows"` (the default): `lhs [M, K]`, `rhs [G, K, N]` → `[M, N]`; each row of group g is
 *   multiplied by `rhs[g]`, rows in no group give zeros. The grouped matmul of a mixture of
 *   experts with its rows sorted by expert.
 * - `"contraction"`: `lhs [M, K]`, `rhs [M, N]` → `[G, K, N]`; `out[g] = lhs_gᵀ · rhs_g` over
 *   group g's rows. The gradient of a `"rows"` product for its right operand.
 *
 * Emitted as `chlo.ragged_dot` (XLA's ragged non-contracting and ragged contracting modes).
 * The gradients close over the two forms: for `"rows"`, `d lhs = rows(dy, rhsᵀ)` and
 * `d rhs = contraction(lhs, dy)`; for `"contraction"`, `d lhs = rows(rhs, dyᵀ)` and
 * `d rhs = rows(lhs, dy)`. The sizes get no gradient.
 */
object RaggedDotAttrs {

    const val ROWS: String = "rows"
    const val CONTRACTION: String = "contraction"

    data class Parsed(val contraction: Boolean, val m: Int, val k: Int, val n: Int, val groups: Int)

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.RAGGED_DOT) { "$layer: RaggedDotAttrs.parse called on ${op.op} (a compiler bug)" }
        require(op.attrs.keys.all { it == "ragged" }) { "$layer: RAGGED_DOT takes only the attribute ragged, got ${op.attrs.keys}" }
        val ragged = op.attrs["ragged"] ?: ROWS
        require(ragged == ROWS || ragged == CONTRACTION) { "$layer: RAGGED_DOT ragged must be \"$ROWS\" or \"$CONTRACTION\", got $ragged" }
        require(op.operands.size == 3 && op.types.size == 1) { "$layer: RAGGED_DOT takes lhs, rhs and sizes and returns one value" }
        val (l, r, s) = op.operands.map { it.type }
        require(s.dtype == I32 && s.rank == 1) { "$layer: RAGGED_DOT sizes must be i32 [G], got $s" }
        require(l.rank == 2 && l.dtype == r.dtype && op.type.dtype == l.dtype) {
            "$layer: RAGGED_DOT lhs must be [M, K] with rhs and the result in its dtype, got $l, $r -> ${op.type}"
        }
        val g = s.dims[0]
        val (m, k) = l.dims
        return if (ragged == ROWS) {
            require(r.rank == 3 && r.dims[0] == g && r.dims[1] == k) { "$layer: RAGGED_DOT rows: rhs must be [$g, $k, N], got $r" }
            val n = r.dims[2]
            require(op.type.dims == listOf(m, n)) { "$layer: RAGGED_DOT rows of $l and $r is [$m, $n], got ${op.type}" }
            Parsed(false, m, k, n, g)
        } else {
            require(r.rank == 2 && r.dims[0] == m) { "$layer: RAGGED_DOT contraction: rhs must be [$m, N], got $r" }
            val n = r.dims[1]
            require(op.type.dims == listOf(g, k, n)) { "$layer: RAGGED_DOT contraction of $l and $r over $g groups is [$g, $k, $n], got ${op.type}" }
            Parsed(true, m, k, n, g)
        }
    }

    /** The group of each of the [m] rows (−1 for none), from [sizes]. */
    fun groupOfRows(sizes: IntArray, m: Int, layer: String): IntArray {
        require(sizes.all { it >= 0 }) { "$layer: RAGGED_DOT sizes must be non-negative, got ${sizes.toList()}" }
        val out = IntArray(m) { -1 }
        var row = 0
        for ((g, size) in sizes.withIndex()) for (i in 0 until size) {
            if (row < m) out[row] = g
            row++
        }
        return out
    }

    /** The value of [op] on row-major [lhs], [rhs] and [sizes], accumulated in double. */
    fun evaluate(op: DxirOp, lhs: DoubleArray, rhs: DoubleArray, sizes: IntArray, layer: String): DoubleArray {
        val p = parse(op, layer)
        val group = groupOfRows(sizes, p.m, layer)
        val (m, k, n) = Triple(p.m, p.k, p.n)
        return if (!p.contraction) {
            val out = DoubleArray(m * n)
            for (i in 0 until m) {
                val g = group[i]
                if (g < 0) continue
                for (j in 0 until n) {
                    var acc = 0.0
                    for (l in 0 until k) acc += lhs[i * k + l] * rhs[(g * k + l) * n + j]
                    out[i * n + j] = acc
                }
            }
            out
        } else {
            val out = DoubleArray(p.groups * k * n)
            for (i in 0 until m) {
                val g = group[i]
                if (g < 0) continue
                for (l in 0 until k) for (j in 0 until n) out[(g * k + l) * n + j] += lhs[i * k + l] * rhs[i * n + j]
            }
            out
        }
    }
}
