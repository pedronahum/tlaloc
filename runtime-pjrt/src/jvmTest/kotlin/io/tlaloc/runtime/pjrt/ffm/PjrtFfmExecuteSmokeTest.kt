package io.tlaloc.runtime.pjrt.ffm

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.304 — FFM compile + dispatch smoke. Builds a tiny StableHLO module
 * (`f(x) = x + 1`), compiles via PJRT_Client_Compile, uploads a host buffer
 * via PJRT_Client_BufferFromHostBuffer, runs PJRT_LoadedExecutable_Execute,
 * pulls the output back via PJRT_Buffer_ToHostBuffer.
 *
 * **No subprocess, no Python.** All FFM downcalls into the bundled
 * `xla_cuda_plugin.so` from the JDK.
 */
class PjrtFfmExecuteSmokeTest {

    private fun resolvePluginPath(): Path? {
        System.getenv("TLALOC_PJRT_PLUGIN_PATH")?.let { p ->
            val path = Path.of(p)
            if (Files.exists(path)) return path
        }
        val home = System.getProperty("user.home") ?: return null
        return Path.of(
            home, ".local", "venvs", "iree", "lib", "python3.12",
            "site-packages", "jax_plugins", "xla_cuda12", "xla_cuda_plugin.so",
        ).takeIf { Files.exists(it) }
    }

    @Test
    fun runsFxEqualsXPlusOneEndToEndViaFfm() {
        val plugin = resolvePluginPath()
        assumeTrue(
            plugin != null,
            "PJRT plugin not found — set TLALOC_PJRT_PLUGIN_PATH or pip-install jax[cuda12] into ~/.local/venvs/iree.",
        )

        val mlir = """
            module @add1 {
              func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
                %c = stablehlo.constant dense<1.0> : tensor<f32>
                %0 = stablehlo.add %arg0, %c : tensor<f32>
                return %0 : tensor<f32>
              }
            }
        """.trimIndent()

        Arena.ofShared().use { arena ->
            val api = PjrtFfm.load(plugin!!, arena)
            api.createClient().use { client ->
                val devices = client.addressableDevices()
                assertEquals(1, devices.size, "test host has exactly one CUDA device")
                val device = devices.single()

                client.compile(mlir).use { exec ->
                    assertEquals(1, exec.numOutputs, "f(x)=x+1 has 1 output")

                    client.bufferFromHostF32(device, floatArrayOf(2.0f), dims = emptyList()).use { input ->
                        val outputs = exec.execute(listOf(input), device)
                        assertEquals(1, outputs.size)
                        try {
                            val result = outputs[0].toFloatArray(1).single()
                            println("[pjrt-ffm-execute] f(2.0) = $result")
                            assertEquals(3.0f, result, "f(2.0) should be 3.0")
                        } finally {
                            outputs.forEach { it.close() }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun runsRank1VectorAddViaFfm() {
        // A slightly bigger smoke: 4-element vector add. Exercises non-scalar
        // dims handling in BufferFromHostBuffer + ToHostBuffer.
        val plugin = resolvePluginPath()
        assumeTrue(plugin != null, "PJRT plugin not found.")

        val mlir = """
            module @vadd {
              func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
                %0 = stablehlo.add %a, %b : tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        """.trimIndent()

        Arena.ofShared().use { arena ->
            val api = PjrtFfm.load(plugin!!, arena)
            api.createClient().use { client ->
                val device = client.addressableDevices().first()
                client.compile(mlir).use { exec ->
                    val a = client.bufferFromHostF32(device, floatArrayOf(1f, 2f, 3f, 4f), dims = listOf(4))
                    val b = client.bufferFromHostF32(device, floatArrayOf(10f, 20f, 30f, 40f), dims = listOf(4))
                    try {
                        val outputs = exec.execute(listOf(a, b), device)
                        try {
                            val result = outputs[0].toFloatArray(4).toList()
                            println("[pjrt-ffm-execute] vadd = $result")
                            assertEquals(listOf(11f, 22f, 33f, 44f), result)
                        } finally {
                            outputs.forEach { it.close() }
                        }
                    } finally {
                        a.close(); b.close()
                    }
                }
            }
        }
    }
}
