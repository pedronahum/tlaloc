package io.tlaloc.runtime.pjrt.kptx

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
 * KPTX v1.5 (§0.4.331) — the first *real* KPTX kernel: **RMS-norm
 * forward** (f32, LlamaDecoder-medium shape 256 tokens × 512 dModel,
 * eps = 1e-5 — matching the Llama decoder's RmsNorm), hand-written PTX,
 * validated against a **decompose-path oracle**: the same math as plain
 * StableHLO (multiply / reduce / rsqrt / broadcast) compiled by the same
 * PJRT-CUDA client — XLA's own codegen as reference.
 *
 * Kernel shape: one CTA per token row, 256 threads; strided
 * sum-of-squares accumulation (fma), shared-memory tree reduction
 * (1 KB static SMEM), `sqrt.rn` + `rcp.rn` for the normalizer
 * (correctly-rounded ops keep the numerics within float-ulp of the
 * oracle), then a strided normalize-and-scale write loop.
 *
 * Tolerance: the kernel reduces sequentially-per-thread + tree; XLA
 * reduces in its own order — f32 sums over 512 elements differ at
 * ~1e-7 relative. Asserted at max |diff| ≤ 1e-4 with the achieved
 * value printed (observed well below; see test output).
 *
 * DXIR-emitter integration (recognizer → KernelDescriptor →
 * `custom_call @kptx_rms_norm` emitted by StablehloEmitter) is v1.6's
 * scope — this test hand-writes both MLIR modules to isolate kernel
 * correctness from emitter plumbing.
 */
class KptxRmsNormKernelTest {

    private val rows = 256
    private val cols = 512
    private val blockThreads = 256

    // eps 1e-5f == 0x3727C5AC
    private val rmsNormPtx = """
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry kptx_rms_norm(
            .param .u64 x_ptr,
            .param .u64 w_ptr,
            .param .u64 out_ptr,
            .param .u32 n_cols
        )
        {
            .reg .pred %p<4>;
            .reg .b32 %r<10>;
            .reg .f32 %f<16>;
            .reg .b64 %rd<21>;
            .shared .align 4 .b8 sdata[1024];

            ld.param.u64 %rd1, [x_ptr];
            ld.param.u64 %rd2, [w_ptr];
            ld.param.u64 %rd3, [out_ptr];
            ld.param.u32 %r1, [n_cols];
            cvta.to.global.u64 %rd4, %rd1;
            cvta.to.global.u64 %rd5, %rd2;
            cvta.to.global.u64 %rd6, %rd3;

            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %tid.x;
            mov.u32 %r4, %ntid.x;

            // row base byte offset = row * n_cols * 4
            mul.lo.u32 %r5, %r2, %r1;
            mul.wide.u32 %rd7, %r5, 4;
            add.s64 %rd8, %rd4, %rd7;
            add.s64 %rd9, %rd6, %rd7;

            // strided sum of squares
            mov.f32 %f1, 0f00000000;
            mov.u32 %r6, %r3;
        LOOP_SUM:
            setp.ge.u32 %p1, %r6, %r1;
            @%p1 bra SUM_DONE;
            mul.wide.u32 %rd10, %r6, 4;
            add.s64 %rd11, %rd8, %rd10;
            ld.global.f32 %f2, [%rd11];
            fma.rn.f32 %f1, %f2, %f2, %f1;
            add.u32 %r6, %r6, %r4;
            bra LOOP_SUM;
        SUM_DONE:

            // sdata[tid] = partial
            mul.wide.u32 %rd12, %r3, 4;
            mov.u64 %rd13, sdata;
            add.s64 %rd14, %rd13, %rd12;
            st.shared.f32 [%rd14], %f1;
            bar.sync 0;

            // shared-memory tree reduction: stride = ntid/2 .. 1
            shr.u32 %r7, %r4, 1;
        RED_LOOP:
            setp.eq.u32 %p2, %r7, 0;
            @%p2 bra RED_DONE;
            setp.ge.u32 %p3, %r3, %r7;
            @%p3 bra RED_SKIP;
            add.u32 %r8, %r3, %r7;
            mul.wide.u32 %rd15, %r8, 4;
            add.s64 %rd16, %rd13, %rd15;
            ld.shared.f32 %f3, [%rd16];
            ld.shared.f32 %f4, [%rd14];
            add.f32 %f5, %f3, %f4;
            st.shared.f32 [%rd14], %f5;
        RED_SKIP:
            bar.sync 0;
            shr.u32 %r7, %r7, 1;
            bra RED_LOOP;
        RED_DONE:

            // inv_rms = 1 / sqrt(sum/n + eps), computed by every thread
            ld.shared.f32 %f6, [%rd13];
            cvt.rn.f32.u32 %f7, %r1;
            div.rn.f32 %f8, %f6, %f7;
            add.f32 %f9, %f8, 0f3727C5AC;
            sqrt.rn.f32 %f10, %f9;
            rcp.rn.f32 %f11, %f10;

            // strided write: out = x * inv_rms * w
            mov.u32 %r9, %r3;
        LOOP_OUT:
            setp.ge.u32 %p1, %r9, %r1;
            @%p1 bra OUT_DONE;
            mul.wide.u32 %rd17, %r9, 4;
            add.s64 %rd18, %rd8, %rd17;
            ld.global.f32 %f12, [%rd18];
            add.s64 %rd19, %rd5, %rd17;
            ld.global.f32 %f13, [%rd19];
            mul.f32 %f14, %f12, %f11;
            mul.f32 %f15, %f14, %f13;
            add.s64 %rd20, %rd9, %rd17;
            st.global.f32 [%rd20], %f15;
            add.u32 %r9, %r9, %r4;
            bra LOOP_OUT;
        OUT_DONE:
            ret;
        }
    """.trimIndent()

    private val kptxMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${cols}xf32>) -> tensor<${rows}x${cols}xf32> {
          %0 = stablehlo.custom_call @kptx_rms_norm(%arg0, %arg1) {api_version = 4 : i32} : (tensor<${rows}x${cols}xf32>, tensor<${cols}xf32>) -> tensor<${rows}x${cols}xf32>
          return %0 : tensor<${rows}x${cols}xf32>
        }
    """.trimIndent()

    /** Same math as decomposed StableHLO — XLA's own codegen is the oracle. */
    private val oracleMlir = """
        func.func @main(%arg0: tensor<${rows}x${cols}xf32>, %arg1: tensor<${cols}xf32>) -> tensor<${rows}x${cols}xf32> {
          %sq = stablehlo.multiply %arg0, %arg0 : tensor<${rows}x${cols}xf32>
          %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %sum = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<${rows}x${cols}xf32>, tensor<f32>) -> tensor<${rows}xf32>
          %n = stablehlo.constant dense<${cols}.0> : tensor<${rows}xf32>
          %mean = stablehlo.divide %sum, %n : tensor<${rows}xf32>
          %eps = stablehlo.constant dense<1.000000e-05> : tensor<${rows}xf32>
          %meps = stablehlo.add %mean, %eps : tensor<${rows}xf32>
          %rms = stablehlo.rsqrt %meps : tensor<${rows}xf32>
          %rmsb = stablehlo.broadcast_in_dim %rms, dims = [0] : (tensor<${rows}xf32>) -> tensor<${rows}x${cols}xf32>
          %xn = stablehlo.multiply %arg0, %rmsb : tensor<${rows}x${cols}xf32>
          %wb = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<${cols}xf32>) -> tensor<${rows}x${cols}xf32>
          %out = stablehlo.multiply %xn, %wb : tensor<${rows}x${cols}xf32>
          return %out : tensor<${rows}x${cols}xf32>
        }
    """.trimIndent()

    @Test
    fun kptxRmsNormMatchesXlaDecomposeOracle() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_rms_norm",
            KptxKernelRegistry.LaunchConfig(
                ptx = rmsNormPtx,
                entryName = "kptx_rms_norm",
                grid = { args -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) }, // one CTA per row
                block = KptxKernelRegistry.Dim3(blockThreads),
                trailingI32Params = { args -> intArrayOf(args[0].dims[1].toInt()) },
            ),
        )

        // Deterministic pseudo-random inputs: x in [-1, 1), w in [0.5, 1.5).
        val rng = java.util.Random(42)
        val x = FloatArray(rows * cols) { rng.nextFloat() * 2f - 1f }
        val w = FloatArray(cols) { rng.nextFloat() + 0.5f }

        val kptxOut: FloatArray
        val oracleOut: FloatArray
        Arena.ofShared().use { arena ->
            PjrtFfm.load(pluginPath, arena).createClient().use { client ->
                val device = client.addressableDevices().first()

                fun run(mlir: String): FloatArray =
                    client.compile(mlir).use { exec ->
                        client.bufferFromHostF32(device, x, listOf(rows, cols)).use { xBuf ->
                            client.bufferFromHostF32(device, w, listOf(cols)).use { wBuf ->
                                val outputs = exec.execute(listOf(xBuf, wBuf), device)
                                try {
                                    outputs.single().toFloatArray(rows * cols)
                                } finally {
                                    outputs.forEach { it.close() }
                                }
                            }
                        }
                    }

                kptxOut = run(kptxMlir)
                oracleOut = run(oracleMlir)
            }
        }

        var maxAbs = 0f
        var maxRel = 0f
        for (i in kptxOut.indices) {
            val d = abs(kptxOut[i] - oracleOut[i])
            maxAbs = max(maxAbs, d)
            val denom = max(abs(oracleOut[i]), 1e-6f)
            maxRel = max(maxRel, d / denom)
        }
        println(
            "[kptx-rms-norm] ${rows}x${cols} f32 vs XLA decompose oracle: " +
                "max|diff|=$maxAbs, max rel=$maxRel",
        )
        assertTrue(maxAbs <= 1e-4f, "max abs diff $maxAbs exceeds 1e-4 vs XLA oracle")
    }
}
