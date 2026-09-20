package io.tlaloc.core.ops

import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * §0.4.428 — host-level certs for the DiffKT `view`/`withChange`/`meld`/`split`
 * sugar surfaces (the A2 indexing-sugar tail). All values are exact on the
 * quarter-integer grid; routing ops are exact by construction, so every
 * assertion is analytic equality, no tolerance.
 */
class ViewWithChangeMeldTest {

    @Test
    fun viewRangeTakesContiguousWindow() {
        val a = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(0.5f, 1f, 1.5f, 2f, 2.5f, 3f, 3.5f, 4f))
        val v = a.view(1..2, 0)
        assertContentEquals(intArrayOf(2, 2), v.dims)
        assertContentEquals(floatArrayOf(1.5f, 2f, 2.5f, 3f), v.hostF32())
        // Column window on axis 1 (negative-axis spelling), axis survives.
        val c = a.view(0..0, -1)
        assertContentEquals(intArrayOf(4, 1), c.dims)
        assertContentEquals(floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f), c.hostF32())
    }

    @Test
    fun viewIndexDropsTheAxis() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(0.25f, 0.5f, 0.75f, 1f, 1.25f, 1.5f))
        val row = a.view(1, 0)
        assertContentEquals(intArrayOf(3), row.dims)
        assertContentEquals(floatArrayOf(1f, 1.25f, 1.5f), row.hostF32())
        val col = a.view(2, 1)
        assertContentEquals(intArrayOf(2), col.dims)
        assertContentEquals(floatArrayOf(0.75f, 1.5f), col.hostF32())
        assertFailsWith<IllegalArgumentException> { a.view(3, 1) }
    }

    @Test
    fun withChangeRowIsFunctional() {
        val a = Tensors.f32Matrix<Sym, Sym>(3, 2, floatArrayOf(0.5f, 1f, 2f, 3f, 4f, 5f))
        val r = Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f))
        val y = a.withChange(1, 0, r)
        assertContentEquals(intArrayOf(3, 2), y.dims)
        assertContentEquals(floatArrayOf(0.5f, 1f, 10f, 20f, 4f, 5f), y.hostF32())
        // The receiver is untouched — functional update, not mutation.
        assertContentEquals(floatArrayOf(0.5f, 1f, 2f, 3f, 4f, 5f), a.hostF32())
    }

    @Test
    fun withChangeRangeMidAxisReplacesWindow() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 4, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f))
        val r = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
        val y = a.withChange(1..2, 1, r)
        assertContentEquals(floatArrayOf(1f, 10f, 20f, 4f, 5f, 30f, 40f, 8f), y.hostF32())
        // Shape discipline: replacement must be exactly the window's shape.
        assertFailsWith<IllegalArgumentException> { a.withChange(0..2, 1, r) }
        assertFailsWith<IllegalArgumentException> { a.withChange(1..2, 1, Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f))) }
    }

    @Test
    fun meldFlattensRowMajorAndConcats() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(0.5f, 1f, 1.5f, 2f))
        val b = Tensors.f32Vector<Sym>(floatArrayOf(2.5f, 3f, 3.5f))
        val m = meld(a, b)
        assertContentEquals(intArrayOf(7), m.dims)
        assertContentEquals(floatArrayOf(0.5f, 1f, 1.5f, 2f, 2.5f, 3f, 3.5f), m.hostF32())
    }

    @Test
    fun splitInvertsMeld() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(0.5f, 1f, 1.5f, 2f))
        val b = Tensors.f32Vector<Sym>(floatArrayOf(2.5f, 3f, 3.5f))
        val pieces = meld(a, b).split(listOf(intArrayOf(2, 2), intArrayOf(3)))
        assertEquals(2, pieces.size)
        assertContentEquals(intArrayOf(2, 2), pieces[0].dims)
        assertContentEquals(a.hostF32(), pieces[0].hostF32())
        assertContentEquals(intArrayOf(3), pieces[1].dims)
        assertContentEquals(b.hostF32(), pieces[1].hostF32())
        // Shapes must consume every element exactly.
        assertFailsWith<IllegalArgumentException> {
            meld(a, b).split(listOf(intArrayOf(2, 2), intArrayOf(2)))
        }
    }
}
