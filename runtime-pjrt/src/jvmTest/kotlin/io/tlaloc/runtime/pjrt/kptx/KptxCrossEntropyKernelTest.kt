package io.tlaloc.runtime.pjrt.kptx

import io.tlaloc.kptx.KptxKernels
import io.tlaloc.kptx.emitPtx
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.350 — **CrossEntropy forward via a two-stage launch chain**
 * ([KptxKernels.crossEntropyModule]) vs an XLA decompose-path oracle
 * mirroring the emitter's exact lowering (max-subtracting softmax →
 * log → labels-left mul → full-reduce), on the same PJRT-CUDA client
 * at the LlamaDecoder-medium LM-head shape (256 tokens × 2048 vocab,
 * one-hot labels, logits from small-Gaussian matmul scale).
 *
 * The chain mechanism under test: ONE `custom_call @kptx_cross_entropy`
 * with results `(loss: f32 scalar, row_loss: f32[256])` — the second
 * result is the inter-stage scratch, XLA-owned per the standing
 * memory decision. The registry launches `_rows` (CTA per row) then
 * `_sum` (one CTA) back-to-back on XLA's stream inside one handler
 * dispatch.
 *
 * Numerics: `ex2.approx`/`lg2.approx` are PTX's only exp/log; asserted
 * at rel ≤ 1e-4 on the scalar with the achieved diff printed.
 */
class KptxCrossEntropyKernelTest {

    private val rows = 256
    private val cols = 2048
    private val blockThreads = 256

    private val cePtx = KptxKernels.crossEntropyModule(block = blockThreads).emitPtx()

    private val kptxMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${rows}x${cols}xf32>) -> tensor<f32> {
          %0:2 = stablehlo.custom_call @kptx_cross_entropy(%arg0, %arg1) {api_version = 4 : i32} : (tensor<${rows}x${cols}xf32>, tensor<${rows}x${cols}xf32>) -> (tensor<f32>, tensor<${rows}xf32>)
          return %0#0 : tensor<f32>
        }
    """.trimIndent()

    /** The emitter's CE decomposition, verbatim shape: max-subtracting
     * softmax, log, labels-left multiply, full-sum to scalar. */
    private val oracleMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${rows}x${cols}xf32>) -> tensor<f32> {
          %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
          %max = stablehlo.reduce(%arg0 init: %ninf) applies stablehlo.maximum across dimensions = [1] : (tensor<${rows}x${cols}xf32>, tensor<f32>) -> tensor<${rows}xf32>
          %maxb = stablehlo.broadcast_in_dim %max, dims = [0] : (tensor<${rows}xf32>) -> tensor<${rows}x${cols}xf32>
          %sh = stablehlo.subtract %arg0, %maxb : tensor<${rows}x${cols}xf32>
          %e = stablehlo.exponential %sh : tensor<${rows}x${cols}xf32>
          %zero = stablehlo.constant dense<0.0> : tensor<f32>
          %s = stablehlo.reduce(%e init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<${rows}x${cols}xf32>, tensor<f32>) -> tensor<${rows}xf32>
          %sb = stablehlo.broadcast_in_dim %s, dims = [0] : (tensor<${rows}xf32>) -> tensor<${rows}x${cols}xf32>
          %p = stablehlo.divide %e, %sb : tensor<${rows}x${cols}xf32>
          %logp = stablehlo.log %p : tensor<${rows}x${cols}xf32>
          %pw = stablehlo.multiply %arg1, %logp : tensor<${rows}x${cols}xf32>
          %loss = stablehlo.reduce(%pw init: %zero) applies stablehlo.add across dimensions = [0, 1] : (tensor<${rows}x${cols}xf32>, tensor<f32>) -> tensor<f32>
          return %loss : tensor<f32>
        }
    """.trimIndent()

    @Test
    fun kptxCrossEntropyChainMatchesXlaDecomposeOracle() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxKernelRegistry.registerKernelChain(
            pluginPath,
            "kptx_cross_entropy",
            cePtx,
            listOf(
                KptxKernelRegistry.Stage(
                    entryName = "kptx_cross_entropy_rows",
                    grid = { args, _ -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) },
                    block = KptxKernelRegistry.Dim3(blockThreads),
                    paramBuffers = { args, rets -> listOf(args[0], args[1], rets[1]) },
                    trailingI32Params = { args, _ -> intArrayOf(args[0].dims[1].toInt()) },
                ),
                KptxKernelRegistry.Stage(
                    entryName = "kptx_cross_entropy_sum",
                    grid = { _, _ -> KptxKernelRegistry.Dim3(1) },
                    block = KptxKernelRegistry.Dim3(blockThreads),
                    paramBuffers = { _, rets -> listOf(rets[1], rets[0]) },
                    trailingI32Params = { args, _ -> intArrayOf(args[0].dims[0].toInt()) },
                ),
            ),
        )

        // One-hot labels + small-Gaussian logits, the benchmark regime.
        val rng = java.util.Random(4242)
        val logits = FloatArray(rows * cols) { (rng.nextGaussian() * 0.5).toFloat() }
        val labels = FloatArray(rows * cols)
        for (i in 0 until rows) labels[i * cols + rng.nextInt(cols)] = 1f

        val kptxLoss: Float
        val oracleLoss: Float
        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()

                fun run(mlir: String): Float =
                    client.compile(mlir).use { exec ->
                        client.bufferFromHostF32(device, logits, listOf(rows, cols)).use { a ->
                            client.bufferFromHostF32(device, labels, listOf(rows, cols)).use { b ->
                                val outputs = exec.execute(listOf(a, b), device)
                                try {
                                    outputs.first().toFloatArray(1).single()
                                } finally {
                                    outputs.forEach { it.close() }
                                }
                            }
                        }
                    }

                kptxLoss = run(kptxMlir)
                oracleLoss = run(oracleMlir)
            }
        }

        val rel = abs(kptxLoss - oracleLoss) / abs(oracleLoss)
        println("[kptx-cross-entropy] ${rows}x${cols} chain: loss kptx=$kptxLoss vs oracle=$oracleLoss, rel=$rel")
        assertTrue(rel <= 1e-4f, "loss rel diff $rel exceeds 1e-4 vs XLA oracle")
    }
}
