package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals

class ShapeTest {
    @Test
    fun scalarShapeRankIsZero() {
        assertEquals(0, ScalarShape.rank)
    }

    @Test
    fun ranksMatchDeclaredRank() {
        assertEquals(1, Rank1<Sym>().rank)
        assertEquals(2, Rank2<Sym, Sym>().rank)
        assertEquals(3, Rank3<Sym, Sym, Sym>().rank)
        assertEquals(4, Rank4<Sym, Sym, Sym, Sym>().rank)
    }

    @Test
    fun dynShapeReportsRuntimeRank() {
        assertEquals(3, DynShape(intArrayOf(2, 3, 5)).rank)
        assertEquals(0, DynShape(intArrayOf()).rank)
    }

    @Test
    fun symCarriesName() {
        assertEquals("B", Sym("B").toString())
    }
}
