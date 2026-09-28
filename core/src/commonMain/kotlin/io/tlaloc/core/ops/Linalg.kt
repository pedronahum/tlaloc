package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.LinalgKernels
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.hostF32

// Differentiable linear algebra, F32. Each function here is also the host twin the
// K2 plugin calls from a synthesized gradient body, and each computes through
// `LinalgKernels` in Double before narrowing, exactly as the `:ir` interpreter does.
// The F64 overloads are in LinalgF64.kt: the two sets erase to the same JVM
// signatures and need separate file facades.
//
// Under `grad {}` the boolean arguments must be literals, and none has a default:
// the plugin reads arguments by position, and K2 does not reorder named arguments
// before it runs.

private fun squareDim(t: DTensor<*, *>, what: String): Int {
    require(t.rank == 2 && t.dims[0] == t.dims[1]) {
        "$what requires a square rank-2 matrix; got dims ${t.dims.toList()}"
    }
    return t.dims[0]
}

private fun f64Of(t: DTensor<*, F32>): DoubleArray {
    val v = t.hostF32()
    return DoubleArray(v.size) { v[it].toDouble() }
}

private fun <S : io.tlaloc.core.Shape> f32Tensor(v: DoubleArray, dims: IntArray): DTensor<S, F32> =
    DTensor(HostF32Storage(FloatArray(v.size) { v[it].toFloat() }), dims, F32)

/**
 * Lower triangle of the matrix, the diagonal included; the rest is zero.
 * Differentiable; its derivative keeps the same triangle of the upstream.
 */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F32>.tril(): DTensor<Rank2<R, C>, F32> =
    scaleTriangles(1f, 1f, 0f)

/**
 * Upper triangle of the matrix, the diagonal included; the rest is zero.
 * Differentiable; its derivative keeps the same triangle of the upstream.
 */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F32>.triu(): DTensor<Rank2<R, C>, F32> =
    scaleTriangles(0f, 1f, 1f)

/**
 * Multiplies the entries below the diagonal by [lower], on it by [diagonal] and
 * above it by [upper]. A zero scale writes an exact zero. The matrix need not be
 * square. `tril()` is `scaleTriangles(1f, 1f, 0f)`.
 *
 * Linear in the matrix, and its own adjoint and tangent. The linear-algebra
 * derivative rules use it for their triangle masks.
 */
fun <R : ShapeAtom, C : ShapeAtom> DTensor<Rank2<R, C>, F32>.scaleTriangles(
    lower: Float,
    diagonal: Float,
    upper: Float,
): DTensor<Rank2<R, C>, F32> {
    require(rank == 2) { "scaleTriangles requires a rank-2 matrix; got dims ${dims.toList()}" }
    val out = LinalgKernels.triangle(
        f64Of(this), dims[0], dims[1], lower.toDouble(), diagonal.toDouble(), upper.toDouble(),
    )
    return f32Tensor(out, dims.copyOf())
}

/**
 * Lower Cholesky factor `L` of a symmetric positive-definite matrix: `L·Lᵀ = A`,
 * zero above the diagonal.
 *
 * The factor is taken of `(A + Aᵀ) / 2`, so both triangles are read and a matrix
 * that is symmetric up to rounding is handled as symmetric. A matrix that is not
 * positive definite gives a matrix of NaN, as XLA does.
 *
 * Differentiable in reverse and forward mode. The derivative is Murray's
 * (*Differentiation of the Cholesky decomposition*, 2016): with
 * `Φ(X)` = the lower triangle of `X` with its diagonal halved,
 * `L̇ = L·Φ(L⁻¹·sym(Ȧ)·L⁻ᵀ)` and `Ā = sym(L⁻ᵀ·Φ(Lᵀ·L̄)·L⁻¹)`; each costs two
 * triangular solves and a matrix product. Lowered to `stablehlo.cholesky`.
 */
fun <N : ShapeAtom> DTensor<Rank2<N, N>, F32>.cholesky(): DTensor<Rank2<N, N>, F32> {
    val n = squareDim(this, "cholesky")
    return f32Tensor(LinalgKernels.cholesky(f64Of(this), n), intArrayOf(n, n))
}

/**
 * Solves `A·X = B` for `X`, where `A` (the receiver) is triangular: lower when
 * [lower], upper otherwise. Only that triangle of `A` is read.
 *
 * Differentiable in both arguments, reverse and forward mode:
 * `B̄ = A⁻ᵀ·X̄` (one more triangular solve) and `Ā = −B̄·Xᵀ` restricted to the
 * triangle that was read. Lowered to `stablehlo.triangular_solve`.
 */
fun <N : ShapeAtom, K : ShapeAtom> DTensor<Rank2<N, N>, F32>.triangularSolve(
    b: DTensor<Rank2<N, K>, F32>,
    lower: Boolean,
): DTensor<Rank2<N, K>, F32> = triangularSolve(b, lower, false, false)

/**
 * Solves `op(A)·X = B` for `X`, where `A` (the receiver) is triangular (lower when
 * [lower]) and `op(A)` is `Aᵀ` when [transposeA], `A` otherwise. With
 * [unitDiagonal] the diagonal of `A` is taken as ones and not read.
 *
 * Differentiable in both arguments; see the two-argument overload.
 */
fun <N : ShapeAtom, K : ShapeAtom> DTensor<Rank2<N, N>, F32>.triangularSolve(
    b: DTensor<Rank2<N, K>, F32>,
    lower: Boolean,
    transposeA: Boolean,
    unitDiagonal: Boolean,
): DTensor<Rank2<N, K>, F32> {
    val n = squareDim(this, "triangularSolve")
    require(b.rank == 2 && b.dims[0] == n) {
        "triangularSolve: B must be $n×k for a $n×$n A; got dims ${b.dims.toList()}"
    }
    val m = b.dims[1]
    val x = LinalgKernels.triangularSolve(f64Of(this), f64Of(b), n, m, lower, transposeA, unitDiagonal)
    return f32Tensor(x, intArrayOf(n, m))
}

/**
 * Solves `A·X = B` for a symmetric positive-definite `A` (the receiver) through
 * its Cholesky factor: `X = L⁻ᵀ·(L⁻¹·B)` with `L = cholesky(A)`. `A` is read as
 * `(A + Aᵀ)/2`; a matrix that is not positive definite gives NaN.
 *
 * Differentiable in both arguments, reverse and forward mode. Under `grad {}` it
 * is lowered to `cholesky` and two `triangularSolve`s and differentiated through
 * their rules; the result equals implicit differentiation of `A·X = B`,
 * `B̄ = A⁻¹·X̄` and `Ā = −sym(B̄·Xᵀ)`. Cost: one factorization (n³/3) and
 * 2·n²·k for the solves; the reverse pass factors again only if the factor is not
 * shared, and adds four triangular solves.
 */
fun <N : ShapeAtom, K : ShapeAtom> DTensor<Rank2<N, N>, F32>.solveSpd(
    b: DTensor<Rank2<N, K>, F32>,
): DTensor<Rank2<N, K>, F32> {
    val l = cholesky()
    return l.triangularSolve(l.triangularSolve(b, true, false, false), true, true, false)
}
