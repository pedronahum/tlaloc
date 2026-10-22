package io.tlaloc.runtime.pjrt.ffm

import java.lang.foreign.Arena
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** The one libtpu create option, `premapped_buffer_size`, as a `PJRT_NamedValue` (no TPU needed). */
class TpuClientOptionsTest {

    @Test
    fun marshalsPremappedBufferSizeAsOneInt64() {
        Arena.ofConfined().use { arena ->
            val nv = PjrtFfm.marshalTpuCreateOptions(arena, TpuClientOptions(premappedBufferSize = 1L shl 33))
            assertEquals(PjrtFfm.SZ_NamedValue, nv.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_StructSize))
            val size = nv.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_NameSize)
            val name = nv.get(ADDRESS, PjrtFfm.OFF_NamedValue_Name).reinterpret(size)
            assertEquals("premapped_buffer_size", String(name.toArray(JAVA_BYTE), Charsets.UTF_8))
            assertEquals(PjrtFfm.PJRT_NAMED_VALUE_TYPE_INT64, nv.get(JAVA_INT, PjrtFfm.OFF_NamedValue_Type))
            assertEquals(1L shl 33, nv.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_Value))
            assertEquals(1L, nv.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_ValueSize))
        }
    }

    @Test
    fun resolvesZeroUnlessTheEnvironmentSaysOtherwise() {
        assertEquals(0L, TpuClientOptions.resolve { null }.premappedBufferSize)
        assertEquals(4096L, TpuClientOptions.resolve { if (it == "TPU_PREMAPPED_BUFFER_SIZE") "4096" else null }.premappedBufferSize)
        assertFailsWith<IllegalArgumentException> { TpuClientOptions.resolve { "four" } }
        assertFailsWith<IllegalArgumentException> { TpuClientOptions(-1) }
    }
}
