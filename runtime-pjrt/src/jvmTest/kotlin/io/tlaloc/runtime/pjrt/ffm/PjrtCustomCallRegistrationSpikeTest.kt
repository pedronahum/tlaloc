package io.tlaloc.runtime.pjrt.ffm

import io.tlaloc.runtime.pjrt.PjrtBinaries
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_INT
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.326 spike, refactored over the §0.4.327 [PjrtFfiRegistry]
 * production API — can an out-of-tree process register a **typed-FFI
 * custom-call handler** with the JAX-shipped `xla_cuda_plugin.so`, and
 * have an XLA-compiled program dispatch into a **Kotlin upcall stub**
 * at execute time?
 *
 * This was the load-bearing uncertainty for the KPTX kernel tier (the
 * pyptx-style "hand-written PTX kernels inside PJRT executables" plan —
 * see docs/KPTX_PLAN.md). Original spike evidence (2026-07-18, GB10,
 * jaxlib 0.10.0): extension chain `[20, 19, 12, 7, 6, 5, 4, 0, 3, 2, 1]`
 * (0 = PJRT_Gpu_Custom_Call), handler invoked at EXECUTE stage (3) with
 * one metadata query, negative control rejected with "No FFI handler
 * registered for ... on a platform CUDA".
 *
 * ABI facts (struct offsets, header provenance) live in
 * [PjrtFfiRegistry]'s KDoc since the §0.4.327 refactor.
 */
class PjrtCustomCallRegistrationSpikeTest {

    companion object {
        /** Execution stages observed by the handler (3 = EXECUTE). Written
         * from XLA's dispatch thread; read from the test thread. */
        private val observedStages = ConcurrentLinkedQueue<Int>()
    }

    private fun customCallMlir(target: String): String = """
        func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
          %0 = stablehlo.custom_call @$target(%arg0) {api_version = 4 : i32} : (tensor<4xf32>) -> tensor<4xf32>
          return %0 : tensor<4xf32>
        }
    """.trimIndent()

    @Test
    fun registerTypedFfiHandlerAndDispatchThroughCompiledExecutable() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!

        // 1. The plugin must expose the GPU custom-call extension at all.
        println("[kptx-spike] PJRT extension types exposed by $pluginPath: ${PjrtFfiRegistry.extensionTypes(pluginPath)}")
        assumeTrue(
            PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath),
            "plugin exposes no PJRT_Gpu_Custom_Call extension — " +
                "KPTX registration would need a self-built plugin.",
        )

        // 2. Register a Kotlin handler; metadata queries are answered by the
        //    registry, so only genuine execution frames arrive here.
        PjrtFfiRegistry.registerExecuteHandler(pluginPath, "tlaloc_kptx_spike") { frame ->
            observedStages.add(
                frame.reinterpret(64).get(JAVA_INT, PjrtFfiRegistry.OFF_FRAME_STAGE),
            )
            MemorySegment.NULL
        }

        // Duplicate registration must fail fast Kotlin-side.
        assertFailsWith<IllegalStateException> {
            PjrtFfiRegistry.registerExecuteHandler(pluginPath, "tlaloc_kptx_spike") { MemorySegment.NULL }
        }

        // 3. Compile + execute a program whose only op is the custom call.
        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()

                // Negative control first: an unregistered symbol must be
                // rejected — proves the positive path below isn't vacuous.
                val rejected = assertFailsWith<PjrtRuntimeException> {
                    client.compile(customCallMlir("tlaloc_kptx_never_registered")).use { exec ->
                        client.bufferFromHostF32(device, floatArrayOf(1f, 2f, 3f, 4f), listOf(4)).use { input ->
                            exec.execute(listOf(input), device).forEach { it.close() }
                        }
                    }
                }
                println("[kptx-spike] negative control rejected as expected: ${rejected.message?.take(200)}")

                client.compile(customCallMlir("tlaloc_kptx_spike")).use { exec ->
                    client.bufferFromHostF32(device, floatArrayOf(1f, 2f, 3f, 4f), listOf(4)).use { input ->
                        exec.execute(listOf(input), device).forEach { it.close() }
                    }
                }
            }
        }

        println("[kptx-spike] handler observed stages=$observedStages (3 = EXECUTE)")
        assertTrue(
            observedStages.contains(PjrtFfiRegistry.XLA_FFI_STAGE_EXECUTE),
            "expected the registered handler to be invoked at EXECUTE stage (3); " +
                "observed stages=$observedStages",
        )
    }
}
