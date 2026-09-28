package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.LinalgKernels
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * CHOLESKY, TRIANGULAR_SOLVE and TRIANGLE at IR level: forward values in the
 * interpreter, and both transforms against central finite differences.
 *
 * The finite differences are taken in Double on [LinalgKernels], not on the F32
 * interpreter, so the reference carries no F32 noise: with step h = 1e-5 on O(1)
 * entries its truncation error is O(h²) ≈ 1e-10 and its rounding error
 * O(ε/h) ≈ 1e-11. The derivative under test runs in F32, where each op rounds to
 * about 6e-8 relative; the matrices here have condition numbers below 30 and the
 * derivative bodies chain at most six ops, so its error is below 1e-5 of the
 * largest gradient entry. The tolerance, 1e-4 of the largest entry, is ten times
 * that and far below the size of any real mistake (a wrong factor of ½ or a
 * missing transpose moves entries by O(1)).
 */
class DxirLinalgGradTest {

    private val scalar = DxirType(F32, emptyList())

    // A = M·Mᵀ + 4I with a small antisymmetric part: CHOLESKY factors sym(A), so
    // the finite differences in the upper and lower triangles differ and both are
    // checked.
    private val n = 4
    private val spd = doubleArrayOf(
        6.2, 1.1, -0.7, 0.4,
        1.3, 5.1, 0.9, -1.2,
        -0.5, 0.9, 4.8, 0.6,
        0.4, -1.0, 0.6, 5.5,
    )

    // Lower triangular with an unrelated upper triangle, which the solve must not read.
    private val tri = doubleArrayOf(
        2.0, 9.0, -7.0, 5.0,
        0.6, 1.7, 8.0, -3.0,
        -0.4, 0.3, 2.4, 4.0,
        0.9, -0.8, 0.5, 1.9,
    )
    private val rhs = doubleArrayOf(0.7, -1.2, 0.4, 2.1, -0.3, 0.8, 1.5, -0.6)
    private val w44 = doubleArrayOf(
        0.3, -1.1, 0.7, 0.2, 1.4, -0.5, 0.9, -0.8, 0.6, 1.2, -0.4, 0.5, -0.9, 0.1, 1.3, -0.2,
    )
    private val w42 = doubleArrayOf(1.1, -0.4, 0.6, 0.9, -1.3, 0.2, 0.5, -0.7)

    private fun f(a: DoubleArray) = FloatArray(a.size) { a[it].toFloat() }

    private fun maxAbs(a: FloatArray) = a.maxOf { abs(it) }

    private fun assertClose(want: DoubleArray, got: FloatArray, relTol: Double, what: String) {
        assertEquals(want.size, got.size, "$what size")
        val scale = max(1e-12, want.maxOf { abs(it) })
        for (i in want.indices) {
            assertTrue(
                abs(want[i] - got[i]) <= relTol * scale,
                "$what[$i] = ${got[i]}, want ${want[i]} (tolerance ${relTol * scale})",
            )
        }
    }

    /** Central differences of [loss] at [x], in Double. */
    private fun fdGrad(x: DoubleArray, loss: (DoubleArray) -> Double): DoubleArray {
        val h = 1e-5
        return DoubleArray(x.size) { i ->
            val p = x.copyOf().also { it[i] += h }
            val m = x.copyOf().also { it[i] -= h }
            (loss(p) - loss(m)) / (2 * h)
        }
    }

    private fun dot(a: DoubleArray, b: DoubleArray): Double = a.indices.sumOf { a[it] * b[it] }

    // --- forward values ---

    @Test
    fun choleskyOfAKnownMatrix() {
        // The textbook example: L·Lᵀ = A with integer L.
        val a = floatArrayOf(4f, 12f, -16f, 12f, 37f, -43f, -16f, -43f, 98f)
        val t = DxirType(F32, listOf(3, 3))
        val fn = DxirBuilder.function("chol") { listOf(op(OpKind.CHOLESKY, listOf(param("a", t)), t)) }
        val l = DxirInterpreter.evalFunction(fn, listOf(a)).single()
        assertEquals(listOf(2f, 0f, 0f, 6f, 1f, 0f, -8f, 5f, 3f), l.toList())
    }

    @Test
    fun choleskyOneByOneAndNotPositiveDefinite() {
        val t = DxirType(F32, listOf(1, 1))
        val fn = DxirBuilder.function("chol1") { listOf(op(OpKind.CHOLESKY, listOf(param("a", t)), t)) }
        assertEquals(3f, DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(9f))).single().single())
        assertTrue(DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(-1f))).single().single().isNaN())
        val t2 = DxirType(F32, listOf(2, 2))
        val fn2 = DxirBuilder.function("chol2") { listOf(op(OpKind.CHOLESKY, listOf(param("a", t2)), t2)) }
        // Symmetric, indefinite (eigenvalues 3 and −1): every entry is NaN.
        val l = DxirInterpreter.evalFunction(fn2, listOf(floatArrayOf(1f, 2f, 2f, 1f))).single()
        assertTrue(l.all { it.isNaN() }, "indefinite input gives NaN, got ${l.toList()}")
    }

    @Test
    fun choleskyOfAnIllConditionedMatrix() {
        // Hilbert(6), condition number 1.5e7: in Double the factor reproduces the
        // matrix to 1e-15 of its largest entry.
        val k = 6
        val h = DoubleArray(k * k) { 1.0 / ((it / k) + (it % k) + 1) }
        val l = LinalgKernels.cholesky(h, k)
        for (i in 0 until k) for (j in 0 until k) {
            var s = 0.0
            for (p in 0 until k) s += l[i * k + p] * l[j * k + p]
            assertTrue(abs(s - h[i * k + j]) < 1e-15, "(L·Lᵀ)[$i,$j] = $s, want ${h[i * k + j]}")
        }
        for (i in 0 until k) for (j in i + 1 until k) assertEquals(0.0, l[i * k + j])
    }

    @Test
    fun triangularSolveReadsOnlyItsTriangle() {
        // L = [[2,0,0],[6,1,0],[-8,5,3]] with junk above the diagonal; X integer.
        val a = floatArrayOf(2f, 99f, -99f, 6f, 1f, 42f, -8f, 5f, 3f)
        val x = doubleArrayOf(1.0, -2.0, 0.5, 3.0, -1.0, 4.0)
        val lx = DoubleArray(6)
        val lc = doubleArrayOf(2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0)
        for (i in 0 until 3) for (c in 0 until 2) {
            for (p in 0 until 3) lx[i * 2 + c] += lc[i * 3 + p] * x[p * 2 + c]
        }
        val at = DxirType(F32, listOf(3, 3))
        val bt = DxirType(F32, listOf(3, 2))
        val fn = DxirBuilder.function("trsv") {
            val pa = param("a", at)
            val pb = param("b", bt)
            listOf(
                op(
                    OpKind.TRIANGULAR_SOLVE, listOf(pa, pb), bt,
                    attrs = mapOf("lower" to true, "transpose_a" to false, "unit_diagonal" to false),
                ),
            )
        }
        val got = DxirInterpreter.evalFunction(fn, listOf(a, f(lx))).single()
        assertClose(x, got, 1e-6, "X")
    }

    @Test
    fun triangleScalesItsThreeParts() {
        val t = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("tri") {
            listOf(
                op(
                    OpKind.TRIANGLE, listOf(param("a", t)), t,
                    attrs = mapOf("lower" to 2.0, "diagonal" to 0.5, "upper" to 0.0),
                ),
            )
        }
        val got = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, Float.NaN, 4f, 6f, 8f))).single()
        // A zero scale writes an exact zero, even over NaN.
        assertEquals(listOf(0.5f, 0f, 0f, 8f, 3f, 0f), got.toList())
    }

    // --- reverse and forward rules against finite differences ---

    private fun choleskyLoss(): io.tlaloc.ir.DxirFunction {
        val t = DxirType(F32, listOf(n, n))
        return DxirBuilder.function("chol_loss") {
            val a = param("a", t)
            val w = param("w", t)
            val l = op(OpKind.CHOLESKY, listOf(a), t)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(l, w), t)), scalar))
        }
    }

    private fun choleskyLossD(a: DoubleArray): Double = dot(LinalgKernels.cholesky(a, n), w44)

    @Test
    fun choleskyGradientMatchesFiniteDifferences() {
        val grad = DxirReverseTransform.apply(choleskyLoss())
        val g = DxirInterpreter.evalFunction(grad, listOf(f(spd), f(w44)))
        assertClose(fdGrad(spd, ::choleskyLossD), g[0], 1e-4, "dA")
        // The gradient is symmetric: Ā = sym(…).
        for (i in 0 until n) for (j in 0 until n) {
            assertTrue(abs(g[0][i * n + j] - g[0][j * n + i]) <= 1e-6f * maxAbs(g[0]), "dA not symmetric at $i,$j")
        }
    }

    @Test
    fun choleskyTangentMatchesFiniteDifferences() {
        val v = DoubleArray(n * n) { ((it * 7) % 11 - 5) / 5.0 }
        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(choleskyLoss()),
            listOf(f(spd), f(w44), f(v), FloatArray(n * n)),
        )
        val h = 1e-5
        val want = (
            choleskyLossD(DoubleArray(n * n) { spd[it] + h * v[it] }) -
                choleskyLossD(DoubleArray(n * n) { spd[it] - h * v[it] })
            ) / (2 * h)
        assertClose(doubleArrayOf(want), floatArrayOf(jvp[1].single()), 1e-4, "tangent")
    }

    private fun solveLoss(lower: Boolean, transposeA: Boolean, unit: Boolean): io.tlaloc.ir.DxirFunction {
        val at = DxirType(F32, listOf(n, n))
        val bt = DxirType(F32, listOf(n, 2))
        return DxirBuilder.function("trsm_loss") {
            val a = param("a", at)
            val b = param("b", bt)
            val w = param("w", bt)
            val x = op(
                OpKind.TRIANGULAR_SOLVE, listOf(a, b), bt,
                attrs = mapOf("lower" to lower, "transpose_a" to transposeA, "unit_diagonal" to unit),
            )
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, w), bt)), scalar))
        }
    }

    // For the upper cases the same junk-laden matrix is transposed, so the
    // triangle that is read is still well conditioned.
    private fun triFor(lower: Boolean) =
        if (lower) tri else DoubleArray(n * n) { tri[(it % n) * n + it / n] }

    @Test
    fun triangularSolveGradientsMatchFiniteDifferencesForEveryFlag() {
        for (lower in listOf(true, false)) for (transposeA in listOf(false, true)) for (unit in listOf(false, true)) {
            val a = triFor(lower)
            val fn = solveLoss(lower, transposeA, unit)
            val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(a), f(rhs), f(w42)))
            fun lossD(aa: DoubleArray, bb: DoubleArray) =
                dot(LinalgKernels.triangularSolve(aa, bb, n, 2, lower, transposeA, unit), w42)
            val tag = "lower=$lower transpose=$transposeA unit=$unit"
            assertClose(fdGrad(a) { lossD(it, rhs) }, g[0], 1e-4, "dA $tag")
            assertClose(fdGrad(rhs) { lossD(a, it) }, g[1], 1e-4, "dB $tag")

            val va = DoubleArray(n * n) { ((it * 5) % 7 - 3) / 3.0 }
            val vb = DoubleArray(n * 2) { ((it * 3) % 5 - 2) / 2.0 }
            val jvp = DxirInterpreter.evalFunction(
                DxirForwardTransform.apply(fn),
                listOf(f(a), f(rhs), f(w42), f(va), f(vb), FloatArray(n * 2)),
            )
            val h = 1e-5
            val want = (
                lossD(DoubleArray(n * n) { a[it] + h * va[it] }, DoubleArray(n * 2) { rhs[it] + h * vb[it] }) -
                    lossD(DoubleArray(n * n) { a[it] - h * va[it] }, DoubleArray(n * 2) { rhs[it] - h * vb[it] })
                ) / (2 * h)
            assertClose(doubleArrayOf(want), floatArrayOf(jvp[1].single()), 1e-4, "tangent $tag")
        }
    }

    @Test
    fun triangleIsItsOwnAdjointAndTangent() {
        val t = DxirType(F32, listOf(n, n))
        val fn = DxirBuilder.function("tri_loss") {
            val a = param("a", t)
            val w = param("w", t)
            val m = op(OpKind.TRIANGLE, listOf(a), t, attrs = mapOf("lower" to 1.0, "diagonal" to 0.5, "upper" to 0.0))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(m, w), t)), scalar))
        }
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(spd), f(w44)))
        val want = LinalgKernels.triangle(w44, n, n, 1.0, 0.5, 0.0)
        assertClose(want, g[0], 1e-7, "dA")
        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(f(spd), f(w44), f(w44), FloatArray(n * n)),
        )
        assertClose(doubleArrayOf(dot(want, w44)), floatArrayOf(jvp[1].single()), 1e-6, "tangent")
    }

    @Test
    fun secondOrderThroughCholeskyMatchesFiniteDifferencesOfTheGradient() {
        // Forward over reverse: the tangent of the gradient body, which is made of
        // CHOLESKY, TRIANGULAR_SOLVE, TRIANGLE, MATMUL and TRANSPOSE, so every
        // forward rule above is exercised on a reverse body. The reference is the
        // central difference of the Double gradient (itself exact to ~1e-9 here).
        val grad = DxirReverseTransform.apply(choleskyLoss())
        val hvp = DxirForwardTransform.apply(grad)
        val v = DoubleArray(n * n) { ((it * 7) % 11 - 5) / 5.0 }
        val out = DxirInterpreter.evalFunction(hvp, listOf(f(spd), f(w44), f(v), FloatArray(n * n)))
        // Outputs: dA, dW, then their tangents.
        val got = out[2]
        val h = 1e-4
        fun gradD(a: DoubleArray) = fdGrad(a, ::choleskyLossD)
        val gp = gradD(DoubleArray(n * n) { spd[it] + h * v[it] })
        val gm = gradD(DoubleArray(n * n) { spd[it] - h * v[it] })
        // Nested differences: truncation O(h²) ≈ 1e-8 relative, rounding
        // O(ε/(h·1e-5)) ≈ 1e-7, both below the F32 error of the second-order body.
        assertClose(DoubleArray(n * n) { (gp[it] - gm[it]) / (2 * h) }, got, 1e-3, "d(dA)")
    }
}
