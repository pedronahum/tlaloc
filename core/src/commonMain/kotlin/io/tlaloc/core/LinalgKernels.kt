package io.tlaloc.core

import kotlin.math.sqrt

/**
 * Double-precision kernels behind the linear-algebra ops, shared by every CPU engine:
 * the F32 and F64 host twins in `:core/ops` and the `:ir` interpreter arms
 * (`CHOLESKY`, `TRIANGULAR_SOLVE`, `TRIANGLE`). One implementation, so the host
 * and the interpreter agree bit for bit. Matrices are dense, row-major, rank 2.
 *
 * These are reference implementations for correctness: O(n³), unblocked, single
 * threaded. The GPU path lowers to StableHLO, where XLA supplies its own kernels.
 */
object LinalgKernels {

    /**
     * Lower Cholesky factor `L` of `sym(A) = (A + Aᵀ) / 2`, `L·Lᵀ = sym(A)`, with the
     * strictly-upper triangle zero. Reading the symmetrized input is JAX's
     * `symmetrize_input=True` convention: the result depends on both triangles
     * equally, so its derivative is defined for every entry.
     *
     * A matrix that is not positive definite (a pivot ≤ 0 or NaN) yields a matrix
     * of NaN, the result XLA gives on the CPU and the GPU.
     */
    fun cholesky(a: DoubleArray, n: Int): DoubleArray {
        require(a.size == n * n) { "cholesky: ${a.size} elements for a $n×$n matrix" }
        val l = DoubleArray(n * n)
        for (j in 0 until n) {
            var d = 0.5 * (a[j * n + j] + a[j * n + j])
            for (k in 0 until j) d -= l[j * n + k] * l[j * n + k]
            if (!(d > 0.0)) return DoubleArray(n * n) { Double.NaN }
            val ljj = sqrt(d)
            l[j * n + j] = ljj
            for (i in j + 1 until n) {
                var s = 0.5 * (a[i * n + j] + a[j * n + i])
                for (k in 0 until j) s -= l[i * n + k] * l[j * n + k]
                l[i * n + j] = s / ljj
            }
        }
        return l
    }

    /**
     * Solves `op(A)·X = B` for `X`, where `A` is `n×n` and triangular, `B` is `n×m`,
     * and `op(A)` is `A` or, with [transposeA], `Aᵀ`. Only the triangle named by
     * [lower] is read; with [unitDiagonal] the diagonal is taken as ones and not read.
     * StableHLO `triangular_solve` with `left_side = true`.
     *
     * A zero on a read diagonal divides by zero (IEEE: ±∞ or NaN), as LAPACK's
     * `trsm` does; no check is made.
     */
    fun triangularSolve(
        a: DoubleArray,
        b: DoubleArray,
        n: Int,
        m: Int,
        lower: Boolean,
        transposeA: Boolean,
        unitDiagonal: Boolean,
    ): DoubleArray {
        require(a.size == n * n) { "triangularSolve: A has ${a.size} elements for $n×$n" }
        require(b.size == n * m) { "triangularSolve: B has ${b.size} elements for $n×$m" }
        // op(A)[i][k]: A[i][k], or A[k][i] when transposed. op(A) is lower
        // triangular exactly when `lower != transposeA`.
        fun opA(i: Int, k: Int): Double = if (transposeA) a[k * n + i] else a[i * n + k]
        val effLower = lower != transposeA
        val x = b.copyOf()
        for (c in 0 until m) {
            if (effLower) {
                for (i in 0 until n) {
                    var s = x[i * m + c]
                    for (k in 0 until i) s -= opA(i, k) * x[k * m + c]
                    x[i * m + c] = if (unitDiagonal) s else s / opA(i, i)
                }
            } else {
                for (i in n - 1 downTo 0) {
                    var s = x[i * m + c]
                    for (k in i + 1 until n) s -= opA(i, k) * x[k * m + c]
                    x[i * m + c] = if (unitDiagonal) s else s / opA(i, i)
                }
            }
        }
        return x
    }

    /**
     * Scales the three parts of an `r×c` matrix: entries below the diagonal by
     * [lower], on it by [diagonal], above it by [upper]. `(1, 1, 0)` is `tril`,
     * `(0, 1, 1)` is `triu`, `(0, 1, 0)` keeps the diagonal. A zero scale writes an
     * exact zero (it does not multiply, so an infinite or NaN entry there is dropped
     * rather than turned into NaN).
     */
    fun triangle(a: DoubleArray, r: Int, c: Int, lower: Double, diagonal: Double, upper: Double): DoubleArray {
        require(a.size == r * c) { "triangle: ${a.size} elements for $r×$c" }
        val out = DoubleArray(r * c)
        for (i in 0 until r) {
            for (j in 0 until c) {
                val s = if (i > j) lower else if (i == j) diagonal else upper
                out[i * c + j] = if (s == 0.0) 0.0 else s * a[i * c + j]
            }
        }
        return out
    }

    /**
     * An LU factorization with partial pivoting: `P·A = L·U`, [packed] holding `U`
     * on and above the diagonal and `L`'s multipliers below it (`L` has a unit
     * diagonal), [perm] the row order (`(P·A)[i] = A[perm[i]]`), [swaps] the number
     * of row exchanges.
     */
    class Lu(val packed: DoubleArray, val perm: IntArray, val swaps: Int)

    /**
     * LU with partial pivoting, the pivot of column `k` being the first row `i ≥ k`
     * with the largest `|A[i][k]|`. A zero pivot (a singular matrix) leaves its
     * column's multipliers zero instead of dividing, as LAPACK's `getrf` does, so
     * `U` has a zero on the diagonal and [det] is exactly 0.
     */
    fun lu(a: DoubleArray, n: Int): Lu {
        require(a.size == n * n) { "lu: ${a.size} elements for a $n×$n matrix" }
        val m = a.copyOf()
        val perm = IntArray(n) { it }
        var swaps = 0
        for (k in 0 until n) {
            var p = k
            var best = kotlin.math.abs(m[k * n + k])
            for (i in k + 1 until n) {
                val v = kotlin.math.abs(m[i * n + k])
                if (v > best) {
                    best = v
                    p = i
                }
            }
            if (p != k) {
                for (j in 0 until n) {
                    val t = m[k * n + j]
                    m[k * n + j] = m[p * n + j]
                    m[p * n + j] = t
                }
                val t = perm[k]
                perm[k] = perm[p]
                perm[p] = t
                swaps++
            }
            val piv = m[k * n + k]
            for (i in k + 1 until n) {
                val l = if (piv == 0.0) 0.0 else m[i * n + k] / piv
                m[i * n + k] = l
                for (j in k + 1 until n) m[i * n + j] -= l * m[k * n + j]
            }
        }
        return Lu(m, perm, swaps)
    }

    /**
     * Solves `op(A)·X = B` through [lu], `op(A)` being `A` or, with [transposeA],
     * `Aᵀ`. `A` is `n×n`, `B` is `n×m`. A singular `A` divides by zero (IEEE ±∞ or
     * NaN), as LAPACK's `getrs` does.
     */
    fun solve(a: DoubleArray, b: DoubleArray, n: Int, m: Int, transposeA: Boolean): DoubleArray {
        require(b.size == n * m) { "solve: B has ${b.size} elements for $n×$m" }
        val f = lu(a, n)
        return if (!transposeA) {
            // A = Pᵀ·L·U: X = U⁻¹·L⁻¹·(P·B).
            val pb = DoubleArray(n * m) { b[f.perm[it / m] * m + it % m] }
            val y = triangularSolve(f.packed, pb, n, m, lower = true, transposeA = false, unitDiagonal = true)
            triangularSolve(f.packed, y, n, m, lower = false, transposeA = false, unitDiagonal = false)
        } else {
            // Aᵀ = Uᵀ·Lᵀ·P: X = Pᵀ·L⁻ᵀ·U⁻ᵀ·B.
            val y = triangularSolve(f.packed, b, n, m, lower = false, transposeA = true, unitDiagonal = false)
            val z = triangularSolve(f.packed, y, n, m, lower = true, transposeA = true, unitDiagonal = true)
            val x = DoubleArray(n * m)
            for (i in 0 until n) for (c in 0 until m) x[f.perm[i] * m + c] = z[i * m + c]
            x
        }
    }

    /** `det A` through [lu]: the product of `U`'s diagonal, negated for an odd number of swaps. */
    fun det(a: DoubleArray, n: Int): Double {
        val f = lu(a, n)
        var d = if (f.swaps % 2 == 0) 1.0 else -1.0
        for (i in 0 until n) d *= f.packed[i * n + i]
        return d
    }
}
