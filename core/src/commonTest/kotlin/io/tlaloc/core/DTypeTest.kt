package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals

class DTypeTest {
    @Test
    fun byteSizes() {
        assertEquals(4, F32.sizeBytes)
        assertEquals(8, F64.sizeBytes)
        assertEquals(4, I32.sizeBytes)
        assertEquals(8, I64.sizeBytes)
        assertEquals(1, Bool.sizeBytes)
    }

    @Test
    fun names() {
        assertEquals("f32", F32.name)
        assertEquals("f64", F64.name)
    }
}
