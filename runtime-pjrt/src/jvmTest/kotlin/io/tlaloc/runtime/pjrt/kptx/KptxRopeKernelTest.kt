package io.tlaloc.runtime.pjrt.kptx

import io.tlaloc.kptx.KptxKernels
import io.tlaloc.kptx.emitPtx
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.349 — **RoPE forward** ([KptxKernels.rope], DSL-authored) vs an
 * XLA decompose-path oracle on the same PJRT-CUDA client:
 * `out = x_real·cos(θ) − x_imag·sin(θ)` at the LlamaDecoder-medium
 * shape (256×512, θ ~ N(0, 0.05²) matching the benchmark synthesis).
 *
 * The kernel uses PTX's `cos.approx`/`sin.approx` (the only sin/cos
 * PTX offers) while XLA lowers `stablehlo.cosine/sine` to its precise
 * routines — the tolerance covers the approx error, which at RoPE's
 * small angle magnitudes sits around 1e-6 absolute (asserted at 1e-4
 * with the achieved value printed).
 */
class KptxRopeKernelTest {

    private val rows = 256
    private val cols = 512
    private val blockThreads = 256

    private val ropePtx = KptxKernels.rope.specialize().emitPtx()

    private val kptxMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${rows}x${cols}xf32>, %arg2: tensor<${rows}x${cols}xf32>) -> tensor<${rows}x${cols}xf32> {
          %0 = stablehlo.custom_call @kptx_rope(%arg0, %arg1, %arg2) {api_version = 4 : i32} : (tensor<${rows}x${cols}xf32>, tensor<${rows}x${cols}xf32>, tensor<${rows}x${cols}xf32>) -> tensor<${rows}x${cols}xf32>
          return %0 : tensor<${rows}x${cols}xf32>
        }
    """.trimIndent()

    private val oracleMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${rows}x${cols}xf32>, %arg2: tensor<${rows}x${cols}xf32>) -> tensor<${rows}x${cols}xf32> {
          %c = stablehlo.cosine %arg2 : tensor<${rows}x${cols}xf32>
          %s = stablehlo.sine %arg2 : tensor<${rows}x${cols}xf32>
          %real = stablehlo.multiply %arg0, %c : tensor<${rows}x${cols}xf32>
          %imag = stablehlo.multiply %arg1, %s : tensor<${rows}x${cols}xf32>
          %out = stablehlo.subtract %real, %imag : tensor<${rows}x${cols}xf32>
          return %out : tensor<${rows}x${cols}xf32>
        }
    """.trimIndent()

    @Test
    fun kptxRopeMatchesXlaDecomposeOracle() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_rope",
            KptxKernelRegistry.LaunchConfig(
                ptx = ropePtx,
                entryName = "kptx_rope",
                grid = { args ->
                    val n = args[0].dims.fold(1L) { a, d -> a * d }.toInt()
                    KptxKernelRegistry.Dim3((n + blockThreads - 1) / blockThreads)
                },
                block = KptxKernelRegistry.Dim3(blockThreads),
                trailingI32Params = { args ->
                    intArrayOf(args[0].dims.fold(1L) { a, d -> a * d }.toInt())
                },
            ),
        )

        val rng = java.util.Random(99)
        val n = rows * cols
        val xr = FloatArray(n) { rng.nextFloat() * 2f - 1f }
        val xi = FloatArray(n) { rng.nextFloat() * 2f - 1f }
        val th = FloatArray(n) { (rng.nextGaussian() * 0.05).toFloat() }

        val kptxOut: FloatArray
        val oracleOut: FloatArray
        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()

                fun run(mlir: String): FloatArray =
                    client.compile(mlir).use { exec ->
                        client.bufferFromHostF32(device, xr, listOf(rows, cols)).use { a ->
                            client.bufferFromHostF32(device, xi, listOf(rows, cols)).use { b ->
                                client.bufferFromHostF32(device, th, listOf(rows, cols)).use { c ->
                                    val outputs = exec.execute(listOf(a, b, c), device)
                                    try {
                                        outputs.single().toFloatArray(n)
                                    } finally {
                                        outputs.forEach { it.close() }
                                    }
                                }
                            }
                        }
                    }

                kptxOut = run(kptxMlir)
                oracleOut = run(oracleMlir)
            }
        }

        var maxAbs = 0f
        for (i in kptxOut.indices) maxAbs = max(maxAbs, abs(kptxOut[i] - oracleOut[i]))
        println("[kptx-rope] ${rows}x${cols} f32 vs XLA cosine/sine oracle: max|diff|=$maxAbs (approx-sin grade)")
        assertTrue(maxAbs <= 1e-4f, "max abs diff $maxAbs exceeds 1e-4 vs XLA oracle")
    }
}
