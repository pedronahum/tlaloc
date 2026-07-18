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
 * §0.4.358 — **attention forward via the three-stage launch chain**
 * ([KptxKernels.attentionModule]) vs an XLA decompose-path oracle
 * mirroring the emitter's exact lowering (score matmul →
 * max-subtracting softmax → output matmul), on the same PJRT-CUDA
 * client at the LlamaDecoder-medium attention shape (Q[256,512],
 * Kᵀ[512,256], V[256,512], no scale factor — the primal's exact form).
 *
 * ONE `custom_call @kptx_attention` with results
 * `(O[256,512], S[256,256])` — the score matrix is the inter-stage
 * scratch; the registry launches scores → row-softmax → output
 * back-to-back on XLA's stream.
 */
class KptxAttentionKernelTest {

    private val t = 256
    private val d = 512
    private val blockThreads = 256

    private val attnPtx = KptxKernels.attentionModule(block = blockThreads).emitPtx()

    private val kptxMlir = """
        func.func @main(%arg0: tensor<${t}x${d}xf32>, %arg1: tensor<${d}x${t}xf32>, %arg2: tensor<${t}x${d}xf32>) -> tensor<${t}x${d}xf32> {
          %0:2 = stablehlo.custom_call @kptx_attention(%arg0, %arg1, %arg2) {api_version = 4 : i32} : (tensor<${t}x${d}xf32>, tensor<${d}x${t}xf32>, tensor<${t}x${d}xf32>) -> (tensor<${t}x${d}xf32>, tensor<${t}x${t}xf32>)
          return %0#0 : tensor<${t}x${d}xf32>
        }
    """.trimIndent()

    private val oracleMlir = """
        func.func @main(%arg0: tensor<${t}x${d}xf32>, %arg1: tensor<${d}x${t}xf32>, %arg2: tensor<${t}x${d}xf32>) -> tensor<${t}x${d}xf32> {
          %s = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<${t}x${d}xf32>, tensor<${d}x${t}xf32>) -> tensor<${t}x${t}xf32>
          %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
          %max = stablehlo.reduce(%s init: %ninf) applies stablehlo.maximum across dimensions = [1] : (tensor<${t}x${t}xf32>, tensor<f32>) -> tensor<${t}xf32>
          %maxb = stablehlo.broadcast_in_dim %max, dims = [0] : (tensor<${t}xf32>) -> tensor<${t}x${t}xf32>
          %sh = stablehlo.subtract %s, %maxb : tensor<${t}x${t}xf32>
          %e = stablehlo.exponential %sh : tensor<${t}x${t}xf32>
          %zero = stablehlo.constant dense<0.0> : tensor<f32>
          %sum = stablehlo.reduce(%e init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<${t}x${t}xf32>, tensor<f32>) -> tensor<${t}xf32>
          %sumb = stablehlo.broadcast_in_dim %sum, dims = [0] : (tensor<${t}xf32>) -> tensor<${t}x${t}xf32>
          %p = stablehlo.divide %e, %sumb : tensor<${t}x${t}xf32>
          %o = stablehlo.dot_general %p, %arg2, contracting_dims = [1] x [0] : (tensor<${t}x${t}xf32>, tensor<${t}x${d}xf32>) -> tensor<${t}x${d}xf32>
          return %o : tensor<${t}x${d}xf32>
        }
    """.trimIndent()

    @Test
    fun kptxAttentionChainMatchesXlaDecomposeOracle() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxKernelRegistry.registerKernelChain(
            pluginPath,
            "kptx_attention",
            attnPtx,
            listOf(
                KptxKernelRegistry.Stage(
                    entryName = "kptx_attn_scores",
                    grid = { args, _ -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) },
                    block = KptxKernelRegistry.Dim3(blockThreads),
                    paramBuffers = { args, rets -> listOf(args[0], args[1], rets[1]) },
                    trailingI32Params = { args, _ ->
                        intArrayOf(args[0].dims[0].toInt(), args[0].dims[1].toInt())
                    },
                ),
                KptxKernelRegistry.Stage(
                    entryName = "kptx_attn_softmax",
                    grid = { args, _ -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) },
                    block = KptxKernelRegistry.Dim3(blockThreads),
                    paramBuffers = { _, rets -> listOf(rets[1]) },
                    trailingI32Params = { args, _ -> intArrayOf(args[0].dims[0].toInt()) },
                ),
                KptxKernelRegistry.Stage(
                    entryName = "kptx_attn_out",
                    grid = { args, _ -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) },
                    block = KptxKernelRegistry.Dim3(blockThreads),
                    paramBuffers = { args, rets -> listOf(rets[1], args[2], rets[0]) },
                    trailingI32Params = { args, _ ->
                        intArrayOf(args[0].dims[0].toInt(), args[0].dims[1].toInt())
                    },
                ),
            ),
        )

        val rng = java.util.Random(2718)
        val q = FloatArray(t * d) { (rng.nextGaussian() * 0.05).toFloat() }
        val kt = FloatArray(d * t) { (rng.nextGaussian() * 0.05).toFloat() }
        val v = FloatArray(t * d) { (rng.nextGaussian() * 0.05).toFloat() }

        val kptxOut: FloatArray
        val oracleOut: FloatArray
        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()

                fun run(mlir: String): FloatArray =
                    client.compile(mlir).use { exec ->
                        client.bufferFromHostF32(device, q, listOf(t, d)).use { a ->
                            client.bufferFromHostF32(device, kt, listOf(d, t)).use { b ->
                                client.bufferFromHostF32(device, v, listOf(t, d)).use { c ->
                                    val outputs = exec.execute(listOf(a, b, c), device)
                                    try {
                                        outputs.first().toFloatArray(t * d)
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
        println("[kptx-attention] ${t}x${d} chain vs XLA decompose oracle: max|diff|=$maxAbs")
        assertTrue(maxAbs <= 1e-4f, "max abs diff $maxAbs exceeds 1e-4 vs XLA oracle")
    }
}
