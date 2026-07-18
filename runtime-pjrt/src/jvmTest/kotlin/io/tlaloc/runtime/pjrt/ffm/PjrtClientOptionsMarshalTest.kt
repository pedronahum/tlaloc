package io.tlaloc.runtime.pjrt.ffm

import java.lang.foreign.Arena
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * §0.4.333 — pins the `PJRT_NamedValue[2]` byte layout
 * [PjrtFfm.marshalCreateOptions] hands to `PJRT_Client_Create`. Runs
 * everywhere (no GPU, no plugin): this is the offset arithmetic that keeps
 * the CUDA plugin's BFCAllocator from preallocating 75% of the GB10's
 * unified memory (the 2026-07-18 reboot incident — see [PjrtClientOptions]).
 */
class PjrtClientOptionsMarshalTest {

    @Test
    fun namedValueLayoutMatchesHeader() {
        // Offsets hand-checked against pjrt_c_api.h:229 (LP64).
        assertEquals(0L, PjrtFfm.OFF_NamedValue_StructSize)
        assertEquals(16L, PjrtFfm.OFF_NamedValue_Name)
        assertEquals(24L, PjrtFfm.OFF_NamedValue_NameSize)
        assertEquals(32L, PjrtFfm.OFF_NamedValue_Type)
        assertEquals(40L, PjrtFfm.OFF_NamedValue_Value)
        assertEquals(48L, PjrtFfm.OFF_NamedValue_ValueSize)
        assertEquals(56L, PjrtFfm.SZ_NamedValue)
    }

    @Test
    fun marshalsMemoryFractionAndPreallocate() {
        Arena.ofConfined().use { arena ->
            val options = PjrtClientOptions(memoryFraction = 0.25f, preallocate = false)
            val array = PjrtFfm.marshalCreateOptions(arena, options)

            // Entry 0: memory_fraction, kFloat.
            assertEquals(PjrtFfm.SZ_NamedValue, array.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_StructSize))
            assertEquals("memory_fraction", readName(array, 0))
            assertEquals(PjrtFfm.PJRT_NAMED_VALUE_TYPE_FLOAT, array.get(JAVA_INT, PjrtFfm.OFF_NamedValue_Type))
            assertEquals(0.25f, array.get(JAVA_FLOAT, PjrtFfm.OFF_NamedValue_Value))
            assertEquals(1L, array.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_ValueSize))

            // Entry 1: preallocate, kBool, false.
            val base = PjrtFfm.SZ_NamedValue
            assertEquals(PjrtFfm.SZ_NamedValue, array.get(JAVA_LONG, base + PjrtFfm.OFF_NamedValue_StructSize))
            assertEquals("preallocate", readName(array, 1))
            assertEquals(PjrtFfm.PJRT_NAMED_VALUE_TYPE_BOOL, array.get(JAVA_INT, base + PjrtFfm.OFF_NamedValue_Type))
            assertEquals(0, array.get(JAVA_BYTE, base + PjrtFfm.OFF_NamedValue_Value))
            assertEquals(1L, array.get(JAVA_LONG, base + PjrtFfm.OFF_NamedValue_ValueSize))
        }
    }

    @Test
    fun marshalsPreallocateTrueAsOneByte() {
        Arena.ofConfined().use { arena ->
            val array = PjrtFfm.marshalCreateOptions(
                arena, PjrtClientOptions(memoryFraction = 1.0f, preallocate = true),
            )
            val base = PjrtFfm.SZ_NamedValue
            assertEquals(1, array.get(JAVA_BYTE, base + PjrtFfm.OFF_NamedValue_Value))
        }
    }

    @Test
    fun defaultsAreUnifiedMemorySafe() {
        // Env-free CI/dev boxes must land on no-preallocate + a ≤0.5 cap;
        // an env override in the test JVM would be a deliberate choice.
        if (System.getenv("TLALOC_PJRT_MEMORY_FRACTION") == null &&
            System.getenv("TLALOC_PJRT_PREALLOCATE") == null
        ) {
            val resolved = PjrtClientOptions.resolve()
            assertEquals(0.5f, resolved.memoryFraction)
            assertEquals(false, resolved.preallocate)
        }
    }

    @Test
    fun rejectsOutOfRangeFraction() {
        assertFailsWith<IllegalArgumentException> { PjrtClientOptions(0f, false) }
        assertFailsWith<IllegalArgumentException> { PjrtClientOptions(1.5f, false) }
    }

    private fun readName(array: java.lang.foreign.MemorySegment, index: Int): String {
        val base = index * PjrtFfm.SZ_NamedValue
        val ptr = array.get(ADDRESS, base + PjrtFfm.OFF_NamedValue_Name).reinterpret(Long.MAX_VALUE)
        val size = array.get(JAVA_LONG, base + PjrtFfm.OFF_NamedValue_NameSize)
        val bytes = ByteArray(size.toInt()) { ptr.get(JAVA_BYTE, it.toLong()) }
        return String(bytes)
    }
}
