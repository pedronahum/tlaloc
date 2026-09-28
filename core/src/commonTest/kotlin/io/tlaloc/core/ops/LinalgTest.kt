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
}
