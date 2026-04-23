package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals

class DScalarTest {
    @Test
    fun floatScalarRoundTrips() {
        val s: DScalar = FloatScalar(1.5f)
        assertEquals(1.5f, s.toFloat())
        assertEquals(1.5, s.toDouble())
    }

    @Test
    fun doubleScalarRoundTrips() {
        val s: DScalar = DoubleScalar(2.25)
        assertEquals(2.25f, s.toFloat())
        assertEquals(2.25, s.toDouble())
    }

    @Test
    fun floatScalarArithmetic() {
        val a = FloatScalar(3f)
        val b = FloatScalar(4f)
        assertEquals(7f, (a + b).v)
        assertEquals(-1f, (a - b).v)
        assertEquals(12f, (a * b).v)
        assertEquals(0.75f, (a / b).v)
    }

    @Test
    fun doubleScalarArithmetic() {
        val a = DoubleScalar(10.0)
        val b = DoubleScalar(2.0)
        assertEquals(12.0, (a + b).v)
        assertEquals(8.0, (a - b).v)
        assertEquals(20.0, (a * b).v)
        assertEquals(5.0, (a / b).v)
    }
}
