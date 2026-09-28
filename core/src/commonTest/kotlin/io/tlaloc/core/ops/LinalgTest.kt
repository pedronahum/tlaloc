package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** The host (forward) linear-algebra functions, F32 and F64. */
class LinalgTest {

    private fun f64(rows: Int, cols: Int, vararg v: Double): DTensor<Rank2<Sym, Sym>, F64> =
        DTensor(HostF64Storage(v.copyOf()), intArrayOf(rows, cols), F64)

    private fun DTensor<*, F64>.data(): DoubleArray = (storage as HostF64Storage).data

    private fun hilbert(k: Int) = f64(k, k, *DoubleArray(k * k) { 1.0 / ((it / k) + (it % k) + 1) })

    @Test
    fun choleskyOfTheTextbookMatrix() {
        val a = Tensors.f32Matrix<Sym, Sym>(3, 3, floatArrayOf(4f, 12f, -16f, 12f, 37f, -43f, -16f, -43f, 98f))
        assertContentEquals(floatArrayOf(2f, 0f, 0f, 6f, 1f, 0f, -8f, 5f, 3f), a.cholesky().hostF32())
        val a64 = f64(3, 3, 4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0)
        assertContentEquals(doubleArrayOf(2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0), a64.cholesky().data())
    }

    @Test
    fun oneByOne() {
        assertEquals(1.5f, Tensors.f32Matrix<Sym, Sym>(1, 1, floatArrayOf(2.25f)).cholesky().hostF32().single())
        val x = f64(1, 1, 4.0).triangularSolve(f64(1, 3, 2.0, -8.0, 1.0), true)
        assertContentEquals(doubleArrayOf(0.5, -2.0, 0.25), x.data())
    }

    @Test
    fun f64CholeskyOfHilbertReproducesTheMatrix() {
        // Condition number of Hilbert(8) is 1.5e10; the factor still reproduces it
        // to 1e-15 in Double (Cholesky is backward stable).
        val k = 8
        val h = hilbert(k)
        val l = h.cholesky().data()
        val hd = h.data()
        for (i in 0 until k) for (j in 0 until k) {
            var s = 0.0
            for (p in 0 until k) s += l[i * k + p] * l[j * k + p]
            assertTrue(abs(s - hd[i * k + j]) < 1e-15, "(L·Lᵀ)[$i,$j] = $s, want ${hd[i * k + j]}")
        }
    }

    @Test
    fun triangularSolveInverts() {
        // For every triangle / transpose, op(A)·X reproduces B.
        val l = f64(3, 3, 2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0)
        val b = f64(3, 2, 1.0, -2.0, 0.5, 3.0, -1.0, 4.0)
        for (lower in listOf(true, false)) for (transposeA in listOf(false, true)) {
            val a = if (lower) l else l.transposeF64()
            val x = a.triangularSolve(b, lower, transposeA, false).data()
            val op = if (transposeA) a.transposeF64().data() else a.data()
            for (i in 0 until 3) for (c in 0 until 2) {
                var s = 0.0
                for (p in 0 until 3) s += op[i * 3 + p] * x[p * 2 + c]
                assertTrue(abs(s - b.data()[i * 2 + c]) < 1e-12, "lower=$lower transpose=$transposeA ($i,$c)")
            }
        }
    }

    @Test
    fun unitDiagonalIsNotRead() {
        val a = f64(2, 2, 7.0, 0.0, 3.0, 9.0)
        val x = a.triangularSolve(f64(2, 1, 1.0, 5.0), true, false, true).data()
        assertContentEquals(doubleArrayOf(1.0, 2.0), x)
    }

    @Test
    fun trilTriuAndScaleTriangles() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        assertContentEquals(floatArrayOf(1f, 0f, 0f, 4f, 5f, 0f), a.tril().hostF32())
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 0f, 5f, 6f), a.triu().hostF32())
        assertContentEquals(floatArrayOf(-1f, 4f, 6f, 8f, -5f, 12f), a.scaleTriangles(2f, -1f, 2f).hostF32())
        assertContentEquals(doubleArrayOf(0.5, 0.0, 3.0, 0.5), f64(2, 2, 1.0, 2.0, 3.0, 1.0).scaleTriangles(1.0, 0.5, 0.0).data())
    }

    @Test
    fun wrongShapesRefuseByName() {
        val rect = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))
        assertTrue(assertFailsWith<IllegalArgumentException> { rect.cholesky() }.message!!.contains("cholesky requires a square"))
        val sq = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 0f, 0f, 1f))
        val b = Tensors.f32Matrix<Sym, Sym>(3, 1, FloatArray(3))
        assertTrue(assertFailsWith<IllegalArgumentException> { sq.triangularSolve(b, true) }.message!!.contains("triangularSolve"))
    }

    private fun DTensor<Rank2<Sym, Sym>, F64>.transposeF64(): DTensor<Rank2<Sym, Sym>, F64> {
        val (r, c) = dims.toList()
        val d = data()
        return f64(c, r, *DoubleArray(r * c) { d[(it % r) * c + it / r] })
    }

    @Test
    fun solveSpdSolves() {
        // A·X = B for the textbook matrix, X integer.
        val a = f64(3, 3, 4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0)
        val x = doubleArrayOf(1.0, -2.0, 3.0, 0.5, -1.0, 4.0)
        val ad = a.data()
        val b = DoubleArray(6)
        for (i in 0 until 3) for (c in 0 until 2) for (p in 0 until 3) b[i * 2 + c] += ad[i * 3 + p] * x[p * 2 + c]
        val got = a.solveSpd(f64(3, 2, *b)).data()
        for (i in x.indices) assertTrue(abs(got[i] - x[i]) < 1e-12, "X[$i] = ${got[i]}, want ${x[i]}")
        val got32 = Tensors.f32Matrix<Sym, Sym>(3, 3, FloatArray(9) { ad[it].toFloat() })
            .solveSpd(Tensors.f32Matrix<Sym, Sym>(3, 2, FloatArray(6) { b[it].toFloat() })).hostF32()
        for (i in x.indices) assertTrue(abs(got32[i] - x[i]) < 1e-3, "F32 X[$i] = ${got32[i]}, want ${x[i]}")
        // 1×1.
        assertContentEquals(doubleArrayOf(2.0), f64(1, 1, 4.0).solveSpd(f64(1, 1, 8.0)).data())
    }

    @Test
    fun solveSpdOfHilbertHasSmallResidual() {
        // Hilbert(8), cond 1.5e10: the error in X can be large, but the residual
        // A·X − B of a backward-stable solve is at the level of ε·‖A‖·‖X‖.
        val k = 8
        val h = hilbert(k)
        val b = f64(k, 1, *DoubleArray(k) { 1.0 })
        val x = h.solveSpd(b).data()
        val hd = h.data()
        val xNorm = x.maxOf { abs(it) }
        for (i in 0 until k) {
            val r = (0 until k).sumOf { hd[i * k + it] * x[it] } - 1.0
            assertTrue(abs(r) < 1e-14 * k * xNorm, "residual[$i] = $r (‖X‖∞ = $xNorm)")
        }
    }

    @Test
    fun logDetSpdValues() {
        // Textbook matrix: det = (2·1·3)² = 36.
        val a = f64(3, 3, 4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0)
        assertTrue(abs(a.logDetSpd().data().single() - kotlin.math.ln(36.0)) < 1e-14)
        assertEquals(
            kotlin.math.ln(36.0).toFloat(),
            Tensors.f32Matrix<Sym, Sym>(3, 3, FloatArray(9) { a.data()[it].toFloat() }).logDetSpd().hostF32().single(),
        )
        assertEquals(kotlin.math.ln(4.0), f64(1, 1, 4.0).logDetSpd().data().single())
        // Hilbert(8): det = 1/365356847125734485878112256000000 (exact, rational
        // elimination), log det = −74.97842732916048. The determinant itself is
        // 2.7e-33; the log is computed without forming it.
        val got = hilbert(8).logDetSpd().data().single()
        assertTrue(abs(got - -74.97842732916048) < 1e-8, "log det H8 = $got")
    }

    @Test
    fun invSpdInverts() {
        val a = f64(3, 3, 4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0)
        val inv = a.invSpd().data()
        val ad = a.data()
        for (i in 0 until 3) for (j in 0 until 3) {
            val s = (0 until 3).sumOf { ad[i * 3 + it] * inv[it * 3 + j] }
            assertTrue(abs(s - if (i == j) 1.0 else 0.0) < 1e-12, "(A·A⁻¹)[$i,$j] = $s")
        }
        assertContentEquals(doubleArrayOf(0.25), f64(1, 1, 4.0).invSpd().data())
        assertContentEquals(floatArrayOf(1f, 0f, 0f, 1f), Tensors.f32Matrix<Sym, Sym>(2, 2, FloatArray(4)).identityLike().hostF32())
        val inv32 = Tensors.f32Matrix<Sym, Sym>(3, 3, FloatArray(9) { ad[it].toFloat() }).invSpd().hostF32()
        for (i in inv.indices) assertTrue(abs(inv32[i] - inv[i]) < 1e-3 * 49.4, "F32 inverse[$i]")
    }

    @Test
    fun generalSolveAndDet() {
        // A permutation-heavy system: A·X = B with X integer, both transpose flags.
        val a = f64(3, 3, 0.0, 2.0, 1.0, 1.0, 0.0, 3.0, 4.0, 1.0, 0.0)
        val x = doubleArrayOf(1.0, -2.0, 3.0, 0.5, -1.0, 4.0)
        for (tr in listOf(false, true)) {
            val ad = a.data()
            val b = DoubleArray(6)
            for (i in 0 until 3) for (c in 0 until 2) for (p in 0 until 3) {
                b[i * 2 + c] += (if (tr) ad[p * 3 + i] else ad[i * 3 + p]) * x[p * 2 + c]
            }
            val got = a.solve(f64(3, 2, *b), tr).data()
            for (i in x.indices) assertTrue(abs(got[i] - x[i]) < 1e-13, "transpose=$tr X[$i] = ${got[i]}")
        }
        // det = 0·(0·0 − 3·1) − 2·(1·0 − 3·4) + 1·(1·1 − 0·4) = 25.
        assertEquals(25.0, a.det().data().single(), 1e-13)
        assertEquals(25f, Tensors.f32Matrix<Sym, Sym>(3, 3, floatArrayOf(0f, 2f, 1f, 1f, 0f, 3f, 4f, 1f, 0f)).det().hostF32().single(), 1e-5f)
        assertEquals(-3.0, f64(1, 1, -3.0).det().data().single())
        assertContentEquals(doubleArrayOf(-2.0), f64(1, 1, -3.0).solve(f64(1, 1, 6.0)).data())
        // Hilbert(8): det = 1/365356847125734485878112256000000 = 2.737e-33 (exact,
        // rational elimination); LU in Double gets it to 1e-6 relative, about what
        // a condition number of 1.5e10 allows.
        val d8 = hilbert(8).det().data().single()
        assertTrue(abs(d8 * 365356847125734485878112256000000.0 - 1.0) < 1e-5, "det H8 = $d8")
    }
}
