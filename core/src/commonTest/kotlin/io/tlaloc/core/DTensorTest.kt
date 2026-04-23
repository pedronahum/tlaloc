package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DTensorTest {
    @Test
    fun rankAndSizeDerivedFromDims() {
        val t = DTensor<Rank2<Sym, Sym>, F32>(
            storage = HostF32Storage(FloatArray(6)),
            dims = intArrayOf(2, 3),
            dtype = F32,
        )
        assertEquals(2, t.rank)
        assertEquals(6, t.size)
    }

    @Test
    fun scalarTensorHasSizeOne() {
        val t = DTensor<ScalarShape, F32>(
            storage = HostF32Storage(FloatArray(1)),
            dims = intArrayOf(),
            dtype = F32,
        )
        assertEquals(0, t.rank)
        assertEquals(1, t.size)
    }

    @Test
    fun negativeDimsRejected() {
        assertFailsWith<IllegalArgumentException> {
            DTensor<Rank1<Sym>, F32>(
                storage = HostF32Storage(FloatArray(0)),
                dims = intArrayOf(-1),
                dtype = F32,
            )
        }
    }

    @Test
    fun closeReleasesStorage() {
        var released = false
        val storage = object : TensorStorage {
            override val sizeBytes: Long = 0
            override fun release() { released = true }
        }
        DTensor<ScalarShape, F32>(storage, intArrayOf(), F32).close()
        assertTrue(released)
    }

    @Test
    fun storageSizeBytesMatchesDtype() {
        val s = HostF32Storage(FloatArray(10))
        assertEquals(40L, s.sizeBytes)
    }
}
