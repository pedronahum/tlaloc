package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import io.tlaloc.runtime.pjrt.ffm.PjrtRuntimeException
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.nio.file.Files
import java.util.Collections
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * [PjrtSession] used from many threads at once, [PjrtSession.close] racing
 * in-flight calls, and the load-time and option-parsing refusals of the PJRT
 * binding. The GPU tests run wherever a PJRT CUDA plugin and an NVIDIA device
 * resolve and skip by name elsewhere; the rest run on every host.
 */
class PjrtSessionConcurrencyTest {

    private val n = 256
    private val vec = DxirType(F32, listOf(n))

    private fun triple() = DxirBuilder.function("triple_concurrency") {
        val v = param("v", vec)
        val two = op(OpKind.ADD, listOf(v, v), vec)
        listOf(op(OpKind.ADD, listOf(two, v), vec))
    }

    private fun assumeGpu() {
        assumeTrue(PjrtBinaries.available, "PJRT CUDA plugin not resolved — skipping.\n${PjrtBinaries.pluginSearchReport}")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device (nvidia-smi -L) — skipping.")
    }

    /** Distinct per call: thread t, iteration i. Small integers stay exact in f32 ×3. */
    private fun valueFor(t: Int, i: Int): Float = (t * 1000 + i).toFloat()

    @Test
    fun concurrentExecuteOnGivesEveryCallerItsOwnResult() {
        assumeGpu()
        val threads = 8
        val iterations = 200
        val fn = triple()
        PjrtSession().use { session ->
            session.prepare(fn)
            val barrier = CyclicBarrier(threads)
            val failures = Collections.synchronizedList(mutableListOf<String>())
            val pool = Executors.newFixedThreadPool(threads)
            try {
                val futures = (0 until threads).map { t ->
                    pool.submit {
                        barrier.await()
                        for (i in 0 until iterations) {
                            val x = valueFor(t, i)
                            session.bufferFromHostF32(FloatArray(n) { x }, listOf(n)).use { input ->
                                val outs = session.executeOn(fn, listOf(input))
                                try {
                                    val got = outs.single().toFloatArray(n)
                                    val bad = got.indexOfFirst { it != 3f * x }
                                    if (bad >= 0) failures += "thread $t iter $i: expected ${3f * x}, got ${got[bad]} at [$bad]"
                                } finally {
                                    outs.forEach { it.close() }
                                }
                            }
                        }
                    }
                }
                futures.forEach { it.get(120, TimeUnit.SECONDS) }
            } finally {
                pool.shutdownNow()
            }
            assertTrue(failures.isEmpty(), "${failures.size} calls read another caller's result, first: ${failures.take(3)}")
            assertEquals(1, session.cacheSize, "one program, one executable")
        }
    }

    @Test
    fun closeWaitsForInFlightCallsAndLaterCallsRefuseByName() {
        assumeGpu()
        val threads = 6
        val fn = triple()
        val session = PjrtSession()
        session.prepare(fn)
        val completed = AtomicInteger()
        val refused = AtomicInteger()
        val failures = Collections.synchronizedList(mutableListOf<String>())
        val start = CyclicBarrier(threads + 1)
        val pool = Executors.newFixedThreadPool(threads)
        try {
            val futures = (0 until threads).map { t ->
                pool.submit {
                    start.await()
                    var i = 0
                    while (true) {
                        val x = valueFor(t, i++ % 1000)
                        try {
                            // runOn stages, executes and reads back inside the session, so
                            // the whole call is covered by the lifetime lock. (Buffers a
                            // caller holds across close() are the caller's to close first.)
                            val out = session.runOn(fn, listOf(FloatArray(n) { x })).single()
                            if (out.any { it != 3f * x }) failures += "thread $t: wrong result for $x"
                            completed.incrementAndGet()
                        } catch (e: IllegalStateException) {
                            if (e.message == "PjrtSession is closed") {
                                refused.incrementAndGet()
                                break
                            }
                            failures += "thread $t: ${e::class.simpleName}: ${e.message}"
                            break
                        } catch (e: Throwable) {
                            failures += "thread $t: ${e::class.simpleName}: ${e.message}"
                            break
                        }
                    }
                }
            }
            start.await()
            Thread.sleep(300)
            session.close()
            futures.forEach { it.get(120, TimeUnit.SECONDS) }
        } finally {
            pool.shutdownNow()
            session.close()
        }
        assertTrue(failures.isEmpty(), "calls racing close() failed other than by name: ${failures.take(3)}")
        assertTrue(completed.get() > 0, "no call completed before close()")
        assertEquals(threads, refused.get(), "every thread's first call after close() is refused by name")
        assertFailsWith<IllegalStateException> { session.runOn(fn, listOf(FloatArray(n))) }
    }

    @Test
    fun closeBlocksUntilAnInFlightCallReturns() {
        assumeGpu()
        val session = PjrtSession()
        val entered = java.util.concurrent.CountDownLatch(1)
        val release = java.util.concurrent.CountDownLatch(1)
        val caller = Thread {
            session.asInFlightCall {
                entered.countDown()
                release.await(30, TimeUnit.SECONDS)
                // Still usable: close() has not freed anything under this call.
                session.platformName()
            }
        }
        caller.start()
        assertTrue(entered.await(30, TimeUnit.SECONDS))
        val closer = Thread { session.close() }
        closer.start()
        closer.join(500)
        assertTrue(closer.isAlive, "close() returned while a call was still in flight")
        release.countDown()
        closer.join(30_000)
        caller.join(30_000)
        assertFalse(closer.isAlive, "close() did not return after the in-flight call finished")
        val e = assertFailsWith<IllegalStateException> { session.platformName() }
        assertEquals("PjrtSession is closed", e.message)
    }

    @Test
    fun aBufferOutlivingItsSessionRefusesReadsAndClosesQuietly() {
        assumeGpu()
        val session = PjrtSession()
        val buf = session.bufferFromHostF32(FloatArray(n) { 1f }, listOf(n))
        session.close()
        val e = assertFailsWith<IllegalStateException> { buf.toFloatArray(n) }
        assertEquals("PjrtBuffer used after its PJRT client was closed", e.message)
        buf.close()
        buf.close()
    }

    @Test
    fun theLoadedPluginPassesTheApiVersionCheck() {
        assumeGpu()
        Arena.ofConfined().use { arena ->
            // PjrtFfm.load runs checkedApi; a plugin it refuses never gets this far.
            PjrtFfm.load(PjrtBinaries.pluginPath!!, arena)
        }
    }

    // ---------------------------------------------------------------- every host

    @Test
    fun aFailedOpenClosesTheArenaItOpened() {
        val notAPlugin = Files.createTempFile("tlaloc-not-a-plugin-", ".so")
        try {
            Files.writeString(notAPlugin, "not an ELF file")
            var arena: Arena? = null
            assertFailsWith<IllegalArgumentException> {
                PjrtSession.openHandles(notAPlugin, PjrtTarget.Cuda, PjrtClientOptions.resolve { null }) {
                    Arena.ofShared().also { arena = it }
                }
            }
            assertFalse(arena!!.scope().isAlive, "the arena opened for a failed session must be closed")
        } finally {
            Files.deleteIfExists(notAPlugin)
        }
    }

    @Test
    fun anApiWithAnotherMajorVersionIsRefusedByName() {
        val e = assertFailsWith<PjrtRuntimeException> {
            PjrtFfm.checkApiCompatible("/p/plugin.so", structSize = 4096, major = 1, minor = 0)
        }
        assertTrue("/p/plugin.so" in e.message!! && "PJRT C API 1.0" in e.message!! && "major version 0" in e.message!!, e.message)
    }

    @Test
    fun anApiTableTooShortForTheBindingsIsRefusedByName() {
        val need = PjrtFfm.PJRT_API_MIN_STRUCT_SIZE
        assertEquals(PjrtFfm.OFFSET_PJRT_Buffer_ToHostBuffer + 8, need)
        val e = assertFailsWith<PjrtRuntimeException> {
            PjrtFfm.checkApiCompatible("/p/plugin.so", structSize = need - 8, major = 0, minor = 30)
        }
        assertTrue("${need - 8} bytes" in e.message!! && "at least $need bytes" in e.message!!, e.message)
        PjrtFfm.checkApiCompatible("/p/plugin.so", structSize = need, major = 0, minor = 30)
    }

    @Test
    fun malformedOptionVariablesAreRefusedByName() {
        fun resolveWith(vararg pairs: Pair<String, String>) = PjrtClientOptions.resolve(mapOf(*pairs)::get)
        val cases = listOf(
            "TLALOC_PJRT_MEMORY_FRACTION" to "half",
            "TLALOC_PJRT_PREALLOCATE" to "yes",
            "TLALOC_PJRT_NODE_ID" to "first",
            "TLALOC_PJRT_NUM_NODES" to "2.5",
        )
        for ((name, value) in cases) {
            val e = assertFailsWith<IllegalArgumentException> { resolveWith(name to value) }
            assertTrue(name in e.message!! && "'$value'" in e.message!!, e.message)
        }
        val range = assertFailsWith<IllegalArgumentException> { resolveWith("TLALOC_PJRT_MEMORY_FRACTION" to "2") }
        assertTrue("memoryFraction must be in (0, 1]" in range.message!! && "TLALOC_PJRT_MEMORY_FRACTION" in range.message!!, range.message)
        assertEquals(PjrtClientOptions(0.25f, true), resolveWith("TLALOC_PJRT_MEMORY_FRACTION" to "0.25", "TLALOC_PJRT_PREALLOCATE" to "true"))
        assertEquals(PjrtClientOptions(0.5f, false), resolveWith())
    }

    @Test
    fun aMissingPluginOffLinuxSaysTheCudaPluginIsLinuxOnly() {
        val mac = PjrtBinaries.missingCudaPluginMessage("PjrtSession", "Mac OS X", "REPORT")
        assertTrue("published for Linux only" in mac && "Mac OS X" in mac && mac.endsWith("REPORT"), mac)
        val linux = PjrtBinaries.missingCudaPluginMessage("runOnPjrt", "Linux", "REPORT")
        assertEquals("runOnPjrt: no PJRT CUDA plugin found.\nREPORT", linux)
    }
}
