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
 * KPTX v1.7 (§0.4.335) — **RMS-norm backward**, hand-written PTX
 * (f32, LlamaDecoder-medium 256×512, eps = 1e-5), validated against a
 * decompose-path oracle carrying the analytical VJP — the same math the
 * RmsNorm coarsener's `gradient_body` emits (§0.4.264/§0.4.292) —
 * compiled by the same PJRT-CUDA client.
 *
 * With `r_i = 1/sqrt(mean_j(x_ij²) + eps)`, `D = cols`, and
 * `s_i = Σ_j dy_ij·w_j·x_ij`:
 *
 *     dx_ij = dy_ij·w_j·r_i − x_ij·r_i³·s_i/D
 *     dw_j  = Σ_i dy_ij·x_ij·r_i
 *
 * `d_eps` is **not** a kernel output — it stays on the analytic
 * decompose path when the VJP is partitioned (no const-zero shortcut;
 * the adjoint is simply not claimed by this kernel).
 *
 * # Two chained custom_calls, not one
 *
 * `dw` is a cross-row reduction and doesn't fit the one-CTA-per-row
 * shape, so backward is **two kernels chained through the emitted
 * program**:
 *
 *   1. `@kptx_rms_norm_bwd_dx(x, w, dy) -> (dx, inv_rms)` — one CTA per
 *      row, 256 threads; two strided accumulations (sum-of-squares and
 *      `s_i`) reduced by a dual shared-memory tree (2 KB static SMEM),
 *      `sqrt.rn`+`rcp.rn` normalizer; thread 0 also writes `r_i` to the
 *      `inv_rms` result vector.
 *   2. `@kptx_rms_norm_bwd_dw(x, dy, inv_rms) -> dw` — one thread per
 *      column (⌈cols/256⌉ CTAs), row-loop accumulating
 *      `dy·x·r_i` with the `inv_rms` produced by kernel 1.
 *
 * XLA sequences both custom_calls on the same stream via the
 * `inv_rms` data dependence — the framework owns ordering, per the
 * KPTX standing decisions. The registry (§0.4.330) needs no changes:
 * per-name registration composes.
 *
 * Tolerance: reduction-order differences vs XLA's codegen put f32
 * sums over 256–512 elements at ~1e-7 relative; asserted at
 * max |diff| ≤ 1e-4 per output with achieved values printed.
 */
class KptxRmsNormBackwardKernelTest {

    private val rows = 256
    private val cols = 512
    private val blockThreads = 256

    // §0.4.344 — kernel text now sourced from the :kptx production
    // library (KptxKernels.rmsNormBwdDx, block=256); this oracle test
    // re-certifies the DSL transcription numerically on every run.
    private val bwdDxPtx = KptxKernels.rmsNormBwdDx
        .specialize(shapes = mapOf("block" to 256))
        .emitPtx()

    private val bwdDwPtx = KptxKernels.rmsNormBwdDw
        .specialize()
        .emitPtx()

    private val kptxMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${cols}xf32>, %arg2: tensor<${rows}x${cols}xf32>) -> (tensor<${rows}x${cols}xf32>, tensor<${cols}xf32>) {
          %0:2 = stablehlo.custom_call @kptx_rms_norm_bwd_dx(%arg0, %arg1, %arg2) {api_version = 4 : i32} : (tensor<${rows}x${cols}xf32>, tensor<${cols}xf32>, tensor<${rows}x${cols}xf32>) -> (tensor<${rows}x${cols}xf32>, tensor<${rows}xf32>)
          %1 = stablehlo.custom_call @kptx_rms_norm_bwd_dw(%arg0, %arg2, %0#1) {api_version = 4 : i32} : (tensor<${rows}x${cols}xf32>, tensor<${rows}x${cols}xf32>, tensor<${rows}xf32>) -> tensor<${cols}xf32>
          return %0#0, %1 : tensor<${rows}x${cols}xf32>, tensor<${cols}xf32>
        }
    """.trimIndent()

    /** The analytical VJP as decomposed StableHLO — XLA's own codegen is
     * the oracle, same discipline as the §0.4.331 forward test. */
    private val oracleMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${cols}xf32>, %arg2: tensor<${rows}x${cols}xf32>) -> (tensor<${rows}x${cols}xf32>, tensor<${cols}xf32>) {
          %sq = stablehlo.multiply %arg0, %arg0 : tensor<${rows}x${cols}xf32>
          %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %sum = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<${rows}x${cols}xf32>, tensor<f32>) -> tensor<${rows}xf32>
          %n = stablehlo.constant dense<${cols}.0> : tensor<${rows}xf32>
          %mean = stablehlo.divide %sum, %n : tensor<${rows}xf32>
          %eps = stablehlo.constant dense<1.000000e-05> : tensor<${rows}xf32>
          %meps = stablehlo.add %mean, %eps : tensor<${rows}xf32>
          %r = stablehlo.rsqrt %meps : tensor<${rows}xf32>
          %rb = stablehlo.broadcast_in_dim %r, dims = [0] : (tensor<${rows}xf32>) -> tensor<${rows}x${cols}xf32>
          %wb = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<${cols}xf32>) -> tensor<${rows}x${cols}xf32>
          %dyw = stablehlo.multiply %arg2, %wb : tensor<${rows}x${cols}xf32>
          %dx1 = stablehlo.multiply %dyw, %rb : tensor<${rows}x${cols}xf32>
          %t = stablehlo.multiply %dyw, %arg0 : tensor<${rows}x${cols}xf32>
          %s = stablehlo.reduce(%t init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<${rows}x${cols}xf32>, tensor<f32>) -> tensor<${rows}xf32>
          %r2 = stablehlo.multiply %r, %r : tensor<${rows}xf32>
          %r3 = stablehlo.multiply %r2, %r : tensor<${rows}xf32>
          %sd = stablehlo.divide %s, %n : tensor<${rows}xf32>
          %c = stablehlo.multiply %r3, %sd : tensor<${rows}xf32>
          %cb = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<${rows}xf32>) -> tensor<${rows}x${cols}xf32>
          %xc = stablehlo.multiply %arg0, %cb : tensor<${rows}x${cols}xf32>
          %dx = stablehlo.subtract %dx1, %xc : tensor<${rows}x${cols}xf32>
          %dyx = stablehlo.multiply %arg2, %arg0 : tensor<${rows}x${cols}xf32>
          %dyxr = stablehlo.multiply %dyx, %rb : tensor<${rows}x${cols}xf32>
          %dw = stablehlo.reduce(%dyxr init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<${rows}x${cols}xf32>, tensor<f32>) -> tensor<${cols}xf32>
          return %dx, %dw : tensor<${rows}x${cols}xf32>, tensor<${cols}xf32>
        }
    """.trimIndent()

    @Test
    fun kptxRmsNormBackwardMatchesXlaDecomposeOracle() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_rms_norm_bwd_dx",
            KptxKernelRegistry.LaunchConfig(
                ptx = bwdDxPtx,
                entryName = "kptx_rms_norm_bwd_dx",
                grid = { args -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) }, // one CTA per row
                block = KptxKernelRegistry.Dim3(blockThreads),
                trailingI32Params = { args -> intArrayOf(args[0].dims[1].toInt()) },
            ),
        )
        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_rms_norm_bwd_dw",
            KptxKernelRegistry.LaunchConfig(
                ptx = bwdDwPtx,
                entryName = "kptx_rms_norm_bwd_dw",
                grid = { args ->
                    val nCols = args[0].dims[1].toInt()
                    KptxKernelRegistry.Dim3((nCols + blockThreads - 1) / blockThreads) // one thread per column
                },
                block = KptxKernelRegistry.Dim3(blockThreads),
                trailingI32Params = { args ->
                    intArrayOf(args[0].dims[0].toInt(), args[0].dims[1].toInt())
                },
            ),
        )

        // Deterministic pseudo-random inputs: x in [-1, 1), w in [0.5, 1.5),
        // dy in [-1, 1) — a generic cotangent, not all-ones, so every term
        // of the VJP is exercised.
        val rng = java.util.Random(1234)
        val x = FloatArray(rows * cols) { rng.nextFloat() * 2f - 1f }
        val w = FloatArray(cols) { rng.nextFloat() + 0.5f }
        val dy = FloatArray(rows * cols) { rng.nextFloat() * 2f - 1f }

        val kptxOut: List<FloatArray>
        val oracleOut: List<FloatArray>
        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()

                fun run(mlir: String): List<FloatArray> =
                    client.compile(mlir).use { exec ->
                        client.bufferFromHostF32(device, x, listOf(rows, cols)).use { xBuf ->
                            client.bufferFromHostF32(device, w, listOf(cols)).use { wBuf ->
                                client.bufferFromHostF32(device, dy, listOf(rows, cols)).use { dyBuf ->
                                    val outputs = exec.execute(listOf(xBuf, wBuf, dyBuf), device)
                                    try {
                                        listOf(
                                            outputs[0].toFloatArray(rows * cols),
                                            outputs[1].toFloatArray(cols),
                                        )
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

        fun compare(label: String, got: FloatArray, want: FloatArray) {
            var maxAbs = 0f
            var maxRel = 0f
            for (i in got.indices) {
                val d = abs(got[i] - want[i])
                maxAbs = max(maxAbs, d)
                val denom = max(abs(want[i]), 1e-6f)
                maxRel = max(maxRel, d / denom)
            }
            println("[kptx-rms-norm-bwd] $label vs XLA decompose oracle: max|diff|=$maxAbs, max rel=$maxRel")
            assertTrue(maxAbs <= 1e-4f, "$label: max abs diff $maxAbs exceeds 1e-4 vs XLA oracle")
        }
        compare("dx ${rows}x${cols}", kptxOut[0], oracleOut[0])
        compare("dw $cols", kptxOut[1], oracleOut[1])
    }
}
