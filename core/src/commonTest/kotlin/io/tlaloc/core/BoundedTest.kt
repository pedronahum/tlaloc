package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

private object MaxSeqForTest : DimBound(4096)

private object MaxBatchForTest : DimBound(8)

private class NonPositiveBound(max: Int) : DimBound(max)

class BoundedTest {

    @Test
    fun boundCarriesItsMaxAndName() {
        assertEquals(4096, MaxSeqForTest.max)
        assertEquals("MaxSeqForTest", MaxSeqForTest.boundName)
        assertEquals("MaxSeqForTest(<= 4096)", MaxSeqForTest.toString())
    }

    @Test
    fun boundBelowOneIsRefused() {
        val e = assertFailsWith<IllegalArgumentException> { NonPositiveBound(0) }
        assertTrue("at least 1" in e.message!!, e.message)
        assertFailsWith<IllegalArgumentException> { NonPositiveBound(-3) }
        assertEquals(1, NonPositiveBound(1).max)
    }

    @Test
    fun boundedAtomComposesPositionallyAndInsideNamed() {
        // Compiles: the atom goes where Sym goes. The phantom type is erased at run time,
        // so the tensor behaves as any other.
        val t: DTensor<Rank2<Named<SeqLen, Bounded<MaxSeqForTest>>, Named<Hidden, Sym>>, F32> =
            Tensors.f32Matrix(3, 2, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        assertEquals(listOf(3, 2), t.dims.toList())
        val v: DTensor<Rank1<Bounded<MaxBatchForTest>>, F32> = Tensors.f32Vector(floatArrayOf(1f, 2f))
        assertEquals(listOf(2), v.dims.toList())
        val atom: ShapeAtom = Bounded<MaxSeqForTest>()
        assertTrue(atom is Bounded<*>)
    }
}
