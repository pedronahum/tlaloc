package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Fp8Test {

    @Test
    fun knownCodesDecode() {
        assertEquals(1f, f8e4m3fnToFloat(0x38))
        assertEquals(-2f, f8e4m3fnToFloat(0xC0.toByte()))
        assertEquals(448f, f8e4m3fnToFloat(0x7E))
        assertEquals(1f / 512f, f8e4m3fnToFloat(0x01))
        assertTrue(f8e4m3fnToFloat(0x7F).isNaN() && f8e4m3fnToFloat(0xFF.toByte()).isNaN())
        assertEquals(listOf(0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -6f), (listOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15)).map { f4e2m1ToFloat(it) })
    }

    @Test
    fun everyFiniteCodeRoundTrips() {
        for (c in 0 until 256) {
            if (c == 0x7F || c == 0xFF) continue
            val v = f8e4m3fnToFloat(c.toByte())
            val back = floatToF8e4m3fn(v).toInt() and 0xFF
            // -0 encodes as 0x80; every other code is its own.
            assertEquals(c, back, "code 0x${c.toString(16)} = $v encoded as 0x${back.toString(16)}")
        }
    }

    @Test
    fun valuesBetweenCodesRoundToTheNearestAndTiesToEven() {
        val codes = (0..0x7E).map { f8e4m3fnToFloat(it.toByte()) }
        for (c in 0 until 0x7E) {
            val lo = codes[c]
            val hi = codes[c + 1]
            val mid = (lo + hi) / 2f
            assertEquals(if (c % 2 == 0) c else c + 1, floatToF8e4m3fn(mid).toInt(), "midpoint of 0x${c.toString(16)} and the next")
            assertEquals(c, floatToF8e4m3fn(lo + (hi - lo) * 0.25f).toInt())
            assertEquals(c + 1, floatToF8e4m3fn(lo + (hi - lo) * 0.75f).toInt())
            assertEquals(0x80 or c, floatToF8e4m3fn(-(lo + (hi - lo) * 0.25f)).toInt() and 0xFF)
        }
    }

    @Test
    fun aValuePastTheLargestIsRefused() {
        assertFailsWith<IllegalArgumentException> { floatToF8e4m3fn(480f) }
        assertEquals(0x7E, floatToF8e4m3fn(460f).toInt())
    }
}
