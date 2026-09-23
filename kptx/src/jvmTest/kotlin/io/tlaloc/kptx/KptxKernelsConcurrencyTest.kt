package io.tlaloc.kptx

import java.util.concurrent.Callable
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The kernel module caches in [KptxKernels] are process-global. Threads that
 * ask for the same key at the same moment get the one cached instance.
 */
class KptxKernelsConcurrencyTest {

    private fun <T : Any> allThreadsGetOneInstance(get: () -> T) {
        val threads = 16
        val barrier = CyclicBarrier(threads)
        val pool = Executors.newFixedThreadPool(threads)
        try {
            val results = (0 until threads)
                .map { pool.submit(Callable { barrier.await(); get() }) }
                .map { it.get(60, TimeUnit.SECONDS) }
            val distinct = results.map { System.identityHashCode(it) }.toSet().size
            assertEquals(1, distinct, "$threads concurrent callers built $distinct different instances")
        } finally {
            pool.shutdownNow()
        }
    }

    // Keys no other test uses, so each test starts from an empty cache slot.
    @Test
    fun attentionModule() = allThreadsGetOneInstance { KptxKernels.attentionModule(1024 - 32) }

    @Test
    fun crossEntropyModule() = allThreadsGetOneInstance { KptxKernels.crossEntropyModule(1024 - 32) }

    @Test
    fun pagedAttentionModule() = allThreadsGetOneInstance { KptxKernels.pagedAttentionModule(64, 0.3125f) }

    @Test
    fun templateSpecialization() = allThreadsGetOneInstance {
        KptxKernels.rope.specialize(arch = "sm_89", shapes = mapOf("concurrency_probe" to 7))
    }
}
