package io.tlaloc.core.ops

import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class HostOpsTest {

    @Test
    fun elementwiseAdd() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val b = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
        val c = a + b
        assertContentEquals(floatArrayOf(11f, 22f, 33f, 44f), c.hostF32())
        assertContentEquals(intArrayOf(2, 2), c.dims)
    }

    @Test
    fun elementwiseMulDivSub() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(6f, 8f, 10f))
        val b = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(2f, 4f, 5f))
        assertContentEquals(floatArrayOf(12f, 32f, 50f), (a * b).hostF32())
        assertContentEquals(floatArrayOf(3f, 2f, 2f), (a / b).hostF32())
        assertContentEquals(floatArrayOf(4f, 4f, 5f), (a - b).hostF32())
    }

    @Test
    fun shapeMismatchFailsFast() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))
        val b = Tensors.f32Matrix<Sym, Sym>(3, 2, FloatArray(6))
        assertFailsWith<IllegalArgumentException> { a + b }
    }

    @Test
    fun reluZerosNegatives() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(-1f, 0f, 2f, -3.5f))
        assertContentEquals(floatArrayOf(0f, 0f, 2f, 0f), a.relu().hostF32())
    }

    @Test
    fun negFlipsSigns() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(1f, -2f, 3f))
        assertContentEquals(floatArrayOf(-1f, 2f, -3f), a.neg().hostF32())
    }

    @Test
    fun sumProducesScalar() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val s = a.sum()
        assertEquals(0, s.rank)
        assertContentEquals(floatArrayOf(10f), s.hostF32())
    }

    @Test
    fun meanAverages() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(2f, 4f, 6f, 8f))
        assertContentEquals(floatArrayOf(5f), a.mean().hostF32())
    }

    @Test
    fun matmul2x3Times3x2() {
        // [1 2 3]   [7  8]    [1*7+2*9+3*11,  1*8+2*10+3*12]   [58  64]
        // [4 5 6] x [9  10] = [4*7+5*9+6*11,  4*8+5*10+6*12] = [139 154]
        //           [11 12]
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = Tensors.f32Matrix<Sym, Sym>(3, 2, floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f))
        val c: DTensorAlias = a matmul b
        assertContentEquals(intArrayOf(2, 2), c.dims)
        assertContentEquals(floatArrayOf(58f, 64f, 139f, 154f), c.hostF32())
    }

    @Test
    fun matmulInnerDimMismatchFails() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))
        val b = Tensors.f32Matrix<Sym, Sym>(4, 2, FloatArray(8))
        assertFailsWith<IllegalArgumentException> { a matmul b }
    }

    @Test
    fun identityMatmul() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(5f, 7f, 11f, 13f))
        val id = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 0f, 0f, 1f))
        assertContentEquals(a.hostF32(), (a matmul id).hostF32())
    }
}

private typealias DTensorAlias = io.tlaloc.core.DTensor<Rank2<Sym, Sym>, io.tlaloc.core.F32>
