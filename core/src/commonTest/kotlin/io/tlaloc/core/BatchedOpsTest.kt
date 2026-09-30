@file:OptIn(ExperimentalTlalocApi::class)

package io.tlaloc.core

import io.tlaloc.core.ops.cholesky
import io.tlaloc.core.ops.choleskyBatched
import io.tlaloc.core.ops.detBatched
import io.tlaloc.core.ops.flattenFrom
import io.tlaloc.core.ops.matmul
import io.tlaloc.core.ops.matmulBatched
import io.tlaloc.core.ops.scaleTrianglesBatched
import io.tlaloc.core.ops.solveBatched
import io.tlaloc.core.ops.triangularSolveBatched
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The batched host twins `vmap`'s synthesized code calls: each equals the rank-2 twin
 * applied to each matrix along the leading axes, bit for bit, and empty batches and
 * empty matrices are handled.
 */
class BatchedOpsTest {

    private fun data(n: Int, seed: Int) = FloatArray(n) { kotlin.math.sin(0.7 * it + seed).toFloat() }

    private fun <S : Shape> t(data: FloatArray, vararg dims: Int) = DTensor<S, F32>(HostF32Storage(data), dims, F32)

    @Test
    fun `matmulBatched equals matmul per matrix`() {
        val a = data(3 * 2 * 4, 1)
        val b = data(3 * 4 * 5, 2)
        val got = matmulBatched<Shape>(t<Shape>(a, 3, 2, 4), t<Shape>(b, 3, 4, 5))
        assertContentEquals(intArrayOf(3, 2, 5), got.dims)
        val want = (0 until 3).flatMap { i ->
            (t<Rank2<Sym, Sym>>(a.copyOfRange(8 * i, 8 * i + 8), 2, 4) matmul t<Rank2<Sym, Sym>>(b.copyOfRange(20 * i, 20 * i + 20), 4, 5))
                .hostF32().asList()
        }
        assertEquals(want.map { it.toRawBits() }, got.hostF32().map { it.toRawBits() })
    }

    @Test
    fun `choleskyBatched equals cholesky per matrix`() {
        val spd = FloatArray(2 * 9)
        val raw = data(18, 3)
        for (e in 0 until 2) for (i in 0 until 3) for (j in 0 until 3) {
            var s = 0f
            for (q in 0 until 3) s += raw[e * 9 + i * 3 + q] * raw[e * 9 + j * 3 + q]
            spd[e * 9 + i * 3 + j] = s + if (i == j) 3f else 0f
        }
        val got = choleskyBatched<Shape>(t<Shape>(spd, 2, 3, 3)).hostF32()
        val want = (0 until 2).flatMap { e -> t<Rank2<Sym, Sym>>(spd.copyOfRange(9 * e, 9 * e + 9), 3, 3).cholesky().hostF32().asList() }
        assertEquals(want.map { it.toRawBits() }, got.map { it.toRawBits() })
    }

    @Test
    fun `flattenFrom keeps the leading axes`() {
        val x = t<Shape>(data(24, 4), 2, 3, 4)
        assertContentEquals(intArrayOf(2, 12), flattenFrom<Shape>(x, 1).dims)
        assertContentEquals(intArrayOf(2, 3, 4), flattenFrom<Shape>(x, 2).dims)
        assertContentEquals(intArrayOf(24), flattenFrom<Shape>(x, 0).dims)
        assertContentEquals(x.hostF32(), flattenFrom<Shape>(x, 1).hostF32())
    }

    @Test
    fun `empty batches and empty matrices`() {
        assertContentEquals(intArrayOf(0, 3, 3), choleskyBatched<Shape>(t<Shape>(FloatArray(0), 0, 3, 3)).dims)
        assertContentEquals(intArrayOf(2, 0, 0), choleskyBatched<Shape>(t<Shape>(FloatArray(0), 2, 0, 0)).dims)
        assertContentEquals(intArrayOf(2, 0, 4), scaleTrianglesBatched<Shape>(t<Shape>(FloatArray(0), 2, 0, 4), 1f, 1f, 0f).dims)
        assertContentEquals(intArrayOf(2), detBatched<Shape>(t<Shape>(FloatArray(0), 2, 0, 0)).dims)
        val a = t<Shape>(FloatArray(0), 2, 0, 0)
        val b = t<Shape>(FloatArray(0), 2, 0, 3)
        assertContentEquals(intArrayOf(2, 0, 3), solveBatched<Shape>(a, b, false).dims)
        assertContentEquals(intArrayOf(2, 0, 3), triangularSolveBatched<Shape>(a, b, true, false, false).dims)
    }
}
