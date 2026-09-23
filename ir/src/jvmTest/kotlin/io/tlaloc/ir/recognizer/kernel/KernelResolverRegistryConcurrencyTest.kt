package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals

/** [KernelResolverRegistry] is process-global; registrations from many threads are all kept. */
@OptIn(ExperimentalTlalocApi::class)
class KernelResolverRegistryConcurrencyTest {

    private class Named(override val backendId: String) : KernelResolver {
        override fun supports(kernelName: String, target: KernelTarget) = true
        override fun resolve(kernelName: String, target: KernelTarget, attrs: Map<String, Any>) =
            KernelResolution(backendId, "impl")
    }

    @AfterTest
    fun reset() = KernelResolverRegistry.clear()

    @Test
    fun concurrentRegistrationsAreAllKept() {
        KernelResolverRegistry.clear()
        val threads = 16
        val perThread = 500
        val barrier = CyclicBarrier(threads)
        val pool = Executors.newFixedThreadPool(threads)
        try {
            val futures = (0 until threads).map { t ->
                pool.submit {
                    barrier.await()
                    repeat(perThread) { i ->
                        KernelResolverRegistry.register(Named("b-$t-$i"))
                        // Readers run alongside writers and must never see a torn list.
                        KernelResolverRegistry.resolveAll("k", KernelTarget.NVIDIA_H100)
                    }
                }
            }
            futures.forEach { it.get(60, TimeUnit.SECONDS) }
        } finally {
            pool.shutdownNow()
        }
        val ids = KernelResolverRegistry.registered()
        assertEquals(threads * perThread, ids.size, "registrations were lost")
        assertEquals(threads * perThread, ids.toSet().size)
    }
}
