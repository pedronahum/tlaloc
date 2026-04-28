package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Layer 2 §0.4.243+ — refcount + lifecycle tests for [BufferHandle] and
 * [HandleRef]. v1 contract: sequential workflow execution, plain-var
 * refcount.
 */
class BufferHandleTest {

    @Test
    fun newHandleStartsLiveAtRefcountOne() {
        val ref = HandleRef(nativeId = 1L)
        assertEquals(1, ref.refCount)
        assertTrue(ref.refCount > 0)
    }

    @Test
    fun retainIncrementsRefcount() {
        val ref = HandleRef(nativeId = 2L)
        ref.retain()
        ref.retain()
        assertEquals(3, ref.refCount)
    }

    @Test
    fun releaseDecrementsRefcount() {
        val ref = HandleRef(nativeId = 3L)
        ref.retain() // count = 2
        ref.release() // count = 1
        assertEquals(1, ref.refCount)
    }

    @Test
    fun onZeroFiresWhenRefcountDropsToZero() {
        var freed = false
        val ref = HandleRef(nativeId = 4L, onZero = { freed = true })
        assertFalse(freed)
        ref.release()
        assertTrue(freed, "onZero callback must fire when refcount transitions to 0")
        assertEquals(0, ref.refCount)
    }

    @Test
    fun retainOnZeroRefcountFails() {
        val ref = HandleRef(nativeId = 5L)
        ref.release()
        assertFails { ref.retain() }
    }

    @Test
    fun doubleReleaseFails() {
        val ref = HandleRef(nativeId = 6L)
        ref.release()
        assertFails { ref.release() }
    }

    @Test
    fun bufferHandleIsAValueClassWrapper() {
        val ref = HandleRef(nativeId = 7L)
        // BufferHandle's type parameters are phantom — no runtime cost.
        // We can construct one over any DTensor / Mesh combination.
        val handle: BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh1<DataAxis>> =
            BufferHandle(ref)
        assertTrue(handle.isLive)
        handle.retain()
        assertEquals(2, handle.ref.refCount)
        handle.release()
        handle.release()
        assertFalse(handle.isLive)
    }

    @Test
    fun closeIsAliasForRelease() {
        var freed = false
        val ref = HandleRef(nativeId = 8L, onZero = { freed = true })
        val handle: BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0> = BufferHandle(ref)
        handle.close()
        assertTrue(freed, "BufferHandle.close() must call HandleRef.release()")
    }

    @Test
    fun useWithBlockReleasesAfterBody() {
        var freed = false
        val ref = HandleRef(nativeId = 9L, onZero = { freed = true })
        val handle: BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0> = BufferHandle(ref)
        handle.use {
            assertTrue(it.isLive)
        }
        assertTrue(freed, "use { } must release on exit")
    }
}
