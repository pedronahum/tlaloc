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

    // --- SOLVE and DET (LU with partial pivoting) ---

    // Nonsymmetric, not diagonally dominant: the first column's largest entry is in
    // row 2, so the factorization pivots.
    private val gen = doubleArrayOf(
        0.5, 2.0, -1.0, 0.3,
        1.2, -0.4, 0.8, 2.2,
        -3.0, 0.7, 1.5, -0.2,
        0.9, 1.1, -0.6, 1.4,
    )

    @Test
    fun luPivotsAndGivesExactDeterminants() {
        // A permutation matrix needs a pivot at the first step: det = −1.
        assertEquals(-1.0, LinalgKernels.det(doubleArrayOf(0.0, 1.0, 1.0, 0.0), 2))
        // Without pivoting, 1e-20 as the first pivot would lose the answer entirely.
        val x = LinalgKernels.solve(doubleArrayOf(1e-20, 1.0, 1.0, 1.0), doubleArrayOf(1.0, 2.0), 2, 1, false)
        assertTrue(abs(x[0] - 1.0) < 1e-15 && abs(x[1] - 1.0) < 1e-15, "x = ${x.toList()}")
        // Singular: U gets an exact zero on its diagonal, so det is exactly 0.
        assertEquals(0.0, abs(LinalgKernels.det(doubleArrayOf(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0), 3)))
        assertEquals(5.0, LinalgKernels.det(doubleArrayOf(5.0), 1))
        // An upper-triangular matrix's determinant is its diagonal's product.
        assertEquals(24.0, LinalgKernels.det(doubleArrayOf(2.0, 7.0, -1.0, 0.0, 3.0, 5.0, 0.0, 0.0, 4.0), 3))
    }

    @Test
    fun solveOfAnIllConditionedMatrixHasSmallResidual() {
        // Hilbert(8) with a perturbed first column (so it is not symmetric),
        // condition number about 1e10; both transpose flags.
        val k = 8
        val h = DoubleArray(k * k) { 1.0 / ((it / k) + (it % k) + 1) + if (it % k == 0) 1e-3 * (it / k) else 0.0 }
        val b = DoubleArray(k) { (it % 3) - 1.0 }
        for (tr in listOf(false, true)) {
            val x = LinalgKernels.solve(h, b, k, 1, tr)
            val xNorm = x.maxOf { abs(it) }
            for (i in 0 until k) {
                val r = (0 until k).sumOf { (if (tr) h[it * k + i] else h[i * k + it]) * x[it] } - b[i]
                assertTrue(abs(r) < 1e-13 * xNorm, "transpose=$tr residual[$i] = $r (‖X‖∞ = $xNorm)")
            }
        }
    }

    private fun generalSolveLoss(transposeA: Boolean): io.tlaloc.ir.DxirFunction {
        val at = DxirType(F32, listOf(n, n))
        val bt = DxirType(F32, listOf(n, 2))
        return DxirBuilder.function("solve_loss") {
            val a = param("a", at)
            val b = param("b", bt)
            val w = param("w", bt)
            val x = op(OpKind.SOLVE, listOf(a, b), bt, attrs = mapOf("transpose_a" to transposeA))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, w), bt)), scalar))
        }
    }

    @Test
    fun solveGradientsAndTangentMatchFiniteDifferences() {
        for (tr in listOf(false, true)) {
            val fn = generalSolveLoss(tr)
            val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(gen), f(rhs), f(w42)))
            fun lossD(aa: DoubleArray, bb: DoubleArray) = dot(LinalgKernels.solve(aa, bb, n, 2, tr), w42)
            assertClose(fdGrad(gen) { lossD(it, rhs) }, g[0], 1e-4, "dA transpose=$tr")
            assertClose(fdGrad(rhs) { lossD(gen, it) }, g[1], 1e-4, "dB transpose=$tr")
            val va = DoubleArray(n * n) { ((it * 5) % 7 - 3) / 3.0 }
            val vb = DoubleArray(n * 2) { ((it * 3) % 5 - 2) / 2.0 }
            val jvp = DxirInterpreter.evalFunction(
                DxirForwardTransform.apply(fn),
                listOf(f(gen), f(rhs), f(w42), f(va), f(vb), FloatArray(n * 2)),
            )
            val h = 1e-5
            val want = (
                lossD(DoubleArray(n * n) { gen[it] + h * va[it] }, DoubleArray(n * 2) { rhs[it] + h * vb[it] }) -
                    lossD(DoubleArray(n * n) { gen[it] - h * va[it] }, DoubleArray(n * 2) { rhs[it] - h * vb[it] })
                ) / (2 * h)
            assertClose(doubleArrayOf(want), floatArrayOf(jvp[1].single()), 1e-4, "tangent transpose=$tr")
        }
    }

    private fun detFn(): io.tlaloc.ir.DxirFunction {
        val t = DxirType(F32, listOf(n, n))
        return DxirBuilder.function("det") { listOf(op(OpKind.DET, listOf(param("a", t)), scalar)) }
    }

    @Test
    fun detGradientTangentAndSecondOrderMatchFiniteDifferences() {
        val fn = detFn()
        val detD = { a: DoubleArray -> LinalgKernels.det(a, n) }
        assertClose(doubleArrayOf(detD(gen)), DxirInterpreter.evalFunction(fn, listOf(f(gen))).single(), 1e-6, "det")
        val grad = DxirReverseTransform.apply(fn)
        val g = DxirInterpreter.evalFunction(grad, listOf(f(gen))).single()
        val gWant = fdGrad(gen, detD)
        assertClose(gWant, g, 1e-4, "d det")
        val v = DoubleArray(n * n) { ((it * 7) % 11 - 5) / 5.0 }
        val jvp = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(f(gen), f(v)))
        assertClose(doubleArrayOf(dot(gWant, v)), floatArrayOf(jvp[1].single()), 1e-4, "tangent")
        // Forward over reverse: the tangent of the gradient body (SOLVE, a splat,
        // TRIANGLE, DET) against differences of the Double gradient.
        val hv = DxirInterpreter.evalFunction(DxirForwardTransform.apply(grad), listOf(f(gen), f(v)))[1]
        val h = 1e-4
        val gp = fdGrad(DoubleArray(n * n) { gen[it] + h * v[it] }, detD)
        val gm = fdGrad(DoubleArray(n * n) { gen[it] - h * v[it] }, detD)
        assertClose(DoubleArray(n * n) { (gp[it] - gm[it]) / (2 * h) }, hv, 1e-3, "d(d det)")
    }

    // --- QR ---

    private val tall = doubleArrayOf(
        1.2, -0.7, 0.3,
        0.4, 2.1, -1.1,
        -0.9, 0.5, 1.7,
        0.6, -1.3, 0.2,
        1.5, 0.8, -0.4,
    )

    @Test
    fun qrFactorsWithOrthonormalColumns() {
        for ((m, k, a) in listOf(
            Triple(5, 3, tall),
            Triple(4, 4, gen),
            Triple(1, 1, doubleArrayOf(-2.0)),
            // Hilbert(6)'s first four columns, condition number about 3e4.
            Triple(6, 4, DoubleArray(24) { 1.0 / ((it / 4) + (it % 4) + 1) }),
        )) {
            val (q, r) = LinalgKernels.qr(a, m, k)
            for (i in 0 until m) for (j in 0 until k) {
                val s = (0 until k).sumOf { q[i * k + it] * r[it * k + j] }
                assertTrue(abs(s - a[i * k + j]) < 1e-14, "$m×$k: (Q·R)[$i,$j] = $s, want ${a[i * k + j]}")
            }
            for (i in 0 until k) for (j in 0 until k) {
                val s = (0 until m).sumOf { q[it * k + i] * q[it * k + j] }
                assertTrue(abs(s - if (i == j) 1.0 else 0.0) < 1e-14, "$m×$k: (QᵀQ)[$i,$j] = $s")
                if (i > j) assertEquals(0.0, r[i * k + j])
            }
        }
        // 1×1: LAPACK leaves a lone entry alone (τ = 0), so Q = 1 and R = A.
        assertEquals(1.0, LinalgKernels.qr(doubleArrayOf(-2.0), 1, 1).first.single())
    }

    private fun qrLoss(m: Int, k: Int): io.tlaloc.ir.DxirFunction {
        val at = DxirType(F32, listOf(m, k))
        val rt = DxirType(F32, listOf(k, k))
        return DxirBuilder.function("qr_loss") {
            val a = param("a", at)
            val wq = param("wq", at)
            val wr = param("wr", rt)
            val q = op(OpKind.QR_Q, listOf(a), at)
            val r = op(OpKind.QR_R, listOf(a), rt)
            val lq = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(q, wq), at)), scalar)
            val lr = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(r, wr), rt)), scalar)
            listOf(op(OpKind.ADD, listOf(lq, lr), scalar))
        }
    }

    @Test
    fun qrGradientTangentAndSecondOrderMatchFiniteDifferences() {
        val m = 5
        val k = 3
        val wq = DoubleArray(m * k) { ((it * 7) % 11 - 5) / 5.0 }
        val wr = DoubleArray(k * k) { ((it * 5) % 7 - 3) / 3.0 }
        fun lossD(a: DoubleArray): Double {
            val (q, r) = LinalgKernels.qr(a, m, k)
            return dot(q, wq) + dot(r, wr)
        }
        val fn = qrLoss(m, k)
        val grad = DxirReverseTransform.apply(fn)
        val g = DxirInterpreter.evalFunction(grad, listOf(f(tall), f(wq), f(wr)))
        val gWant = fdGrad(tall, ::lossD)
        assertClose(gWant, g[0], 1e-4, "dA")
        val v = DoubleArray(m * k) { ((it * 3) % 7 - 3) / 3.0 }
        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn),
            listOf(f(tall), f(wq), f(wr), f(v), FloatArray(m * k), FloatArray(k * k)),
        )
        assertClose(doubleArrayOf(dot(gWant, v)), floatArrayOf(jvp[1].single()), 1e-4, "tangent")
        val hv = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(grad),
            listOf(f(tall), f(wq), f(wr), f(v), FloatArray(m * k), FloatArray(k * k)),
        )[3]
        val h = 1e-4
        val gp = fdGrad(DoubleArray(m * k) { tall[it] + h * v[it] }, ::lossD)
        val gm = fdGrad(DoubleArray(m * k) { tall[it] - h * v[it] }, ::lossD)
        assertClose(DoubleArray(m * k) { (gp[it] - gm[it]) / (2 * h) }, hv, 1e-3, "d(dA)")
    }
}
