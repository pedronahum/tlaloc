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

    // eps 1e-5f == 0x3727C5AC
    private val bwdDxPtx = """
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry kptx_rms_norm_bwd_dx(
            .param .u64 x_ptr,
            .param .u64 w_ptr,
            .param .u64 dy_ptr,
            .param .u64 dx_ptr,
            .param .u64 invr_ptr,
            .param .u32 n_cols
        )
        {
            .reg .pred %p<6>;
            .reg .b32 %r<12>;
            .reg .f32 %f<32>;
            .reg .b64 %rd<36>;
            .shared .align 4 .b8 sdata[1024];
            .shared .align 4 .b8 sdata2[1024];

            ld.param.u64 %rd1, [x_ptr];
            ld.param.u64 %rd2, [w_ptr];
            ld.param.u64 %rd3, [dy_ptr];
            ld.param.u64 %rd4, [dx_ptr];
            ld.param.u64 %rd5, [invr_ptr];
            ld.param.u32 %r1, [n_cols];
            cvta.to.global.u64 %rd6, %rd1;
            cvta.to.global.u64 %rd7, %rd2;
            cvta.to.global.u64 %rd8, %rd3;
            cvta.to.global.u64 %rd9, %rd4;
            cvta.to.global.u64 %rd10, %rd5;

            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %tid.x;
            mov.u32 %r4, %ntid.x;

            // row base byte offset = row * n_cols * 4
            mul.lo.u32 %r5, %r2, %r1;
            mul.wide.u32 %rd11, %r5, 4;
            add.s64 %rd12, %rd6, %rd11;    // x row
            add.s64 %rd13, %rd8, %rd11;    // dy row
            add.s64 %rd14, %rd9, %rd11;    // dx row

            // strided accumulate: sumsq += x*x ; s += (dy*w)*x
            mov.f32 %f1, 0f00000000;
            mov.f32 %f2, 0f00000000;
            mov.u32 %r6, %r3;
        LOOP_ACC:
            setp.ge.u32 %p1, %r6, %r1;
            @%p1 bra ACC_DONE;
            mul.wide.u32 %rd15, %r6, 4;
            add.s64 %rd16, %rd12, %rd15;
            ld.global.f32 %f3, [%rd16];
            add.s64 %rd17, %rd13, %rd15;
            ld.global.f32 %f4, [%rd17];
            add.s64 %rd18, %rd7, %rd15;
            ld.global.f32 %f5, [%rd18];
            fma.rn.f32 %f1, %f3, %f3, %f1;
            mul.f32 %f6, %f4, %f5;
            fma.rn.f32 %f2, %f6, %f3, %f2;
            add.u32 %r6, %r6, %r4;
            bra LOOP_ACC;
        ACC_DONE:

            // sdata[tid] = sumsq partial, sdata2[tid] = s partial
            mul.wide.u32 %rd19, %r3, 4;
            mov.u64 %rd20, sdata;
            add.s64 %rd21, %rd20, %rd19;
            st.shared.f32 [%rd21], %f1;
            mov.u64 %rd22, sdata2;
            add.s64 %rd23, %rd22, %rd19;
            st.shared.f32 [%rd23], %f2;
            bar.sync 0;

            // dual shared-memory tree reduction: stride = ntid/2 .. 1
            shr.u32 %r7, %r4, 1;
        RED_LOOP:
            setp.eq.u32 %p2, %r7, 0;
            @%p2 bra RED_DONE;
            setp.ge.u32 %p3, %r3, %r7;
            @%p3 bra RED_SKIP;
            add.u32 %r8, %r3, %r7;
            mul.wide.u32 %rd24, %r8, 4;
            add.s64 %rd25, %rd20, %rd24;
            ld.shared.f32 %f7, [%rd25];
            ld.shared.f32 %f8, [%rd21];
            add.f32 %f9, %f7, %f8;
            st.shared.f32 [%rd21], %f9;
            add.s64 %rd26, %rd22, %rd24;
            ld.shared.f32 %f10, [%rd26];
            ld.shared.f32 %f11, [%rd23];
            add.f32 %f12, %f10, %f11;
            st.shared.f32 [%rd23], %f12;
        RED_SKIP:
            bar.sync 0;
            shr.u32 %r7, %r7, 1;
            bra RED_LOOP;
        RED_DONE:

            // r = 1/sqrt(sumsq/n + eps); c = r^3 * (s/n)
            ld.shared.f32 %f13, [%rd20];
            ld.shared.f32 %f14, [%rd22];
            cvt.rn.f32.u32 %f15, %r1;
            div.rn.f32 %f16, %f13, %f15;
            add.f32 %f17, %f16, 0f3727C5AC;
            sqrt.rn.f32 %f18, %f17;
            rcp.rn.f32 %f19, %f18;
            mul.f32 %f20, %f19, %f19;
            mul.f32 %f21, %f20, %f19;
            div.rn.f32 %f22, %f14, %f15;
            mul.f32 %f23, %f21, %f22;

            // thread 0 writes inv_rms[row]
            setp.ne.u32 %p4, %r3, 0;
            @%p4 bra SKIP_INVR;
            mul.wide.u32 %rd27, %r2, 4;
            add.s64 %rd28, %rd10, %rd27;
            st.global.f32 [%rd28], %f19;
        SKIP_INVR:

            // strided write: dx = (dy*w)*r - x*c
            mov.u32 %r9, %r3;
        LOOP_OUT:
            setp.ge.u32 %p5, %r9, %r1;
            @%p5 bra OUT_DONE;
            mul.wide.u32 %rd29, %r9, 4;
            add.s64 %rd30, %rd12, %rd29;
            ld.global.f32 %f24, [%rd30];
            add.s64 %rd31, %rd13, %rd29;
            ld.global.f32 %f25, [%rd31];
            add.s64 %rd32, %rd7, %rd29;
            ld.global.f32 %f26, [%rd32];
            mul.f32 %f27, %f25, %f26;
            mul.f32 %f28, %f27, %f19;
            mul.f32 %f29, %f24, %f23;
            sub.f32 %f30, %f28, %f29;
            add.s64 %rd33, %rd14, %rd29;
            st.global.f32 [%rd33], %f30;
            add.u32 %r9, %r9, %r4;
            bra LOOP_OUT;
        OUT_DONE:
            ret;
        }
    """.trimIndent()

    private val bwdDwPtx = """
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry kptx_rms_norm_bwd_dw(
            .param .u64 x_ptr,
            .param .u64 dy_ptr,
            .param .u64 invr_ptr,
            .param .u64 dw_ptr,
            .param .u32 n_rows,
            .param .u32 n_cols
        )
        {
            .reg .pred %p<3>;
            .reg .b32 %r<10>;
            .reg .f32 %f<8>;
            .reg .b64 %rd<16>;

            ld.param.u64 %rd1, [x_ptr];
            ld.param.u64 %rd2, [dy_ptr];
            ld.param.u64 %rd3, [invr_ptr];
            ld.param.u64 %rd4, [dw_ptr];
            ld.param.u32 %r1, [n_rows];
            ld.param.u32 %r2, [n_cols];
            cvta.to.global.u64 %rd5, %rd1;
            cvta.to.global.u64 %rd6, %rd2;
            cvta.to.global.u64 %rd7, %rd3;
            cvta.to.global.u64 %rd8, %rd4;

            // column j = ctaid * ntid + tid; guard j < n_cols
            mov.u32 %r3, %ctaid.x;
            mov.u32 %r4, %ntid.x;
            mov.u32 %r5, %tid.x;
            mad.lo.u32 %r6, %r3, %r4, %r5;
            setp.ge.u32 %p1, %r6, %r2;
            @%p1 bra DONE;

            // dw_j = sum_i dy[i,j] * x[i,j] * inv_rms[i]
            mov.f32 %f1, 0f00000000;
            mov.u32 %r7, 0;
        LOOP_ROWS:
            setp.ge.u32 %p2, %r7, %r1;
            @%p2 bra ROWS_DONE;
            mad.lo.u32 %r8, %r7, %r2, %r6;
            mul.wide.u32 %rd9, %r8, 4;
            add.s64 %rd10, %rd5, %rd9;
            ld.global.f32 %f2, [%rd10];
            add.s64 %rd11, %rd6, %rd9;
            ld.global.f32 %f3, [%rd11];
            mul.wide.u32 %rd12, %r7, 4;
            add.s64 %rd13, %rd7, %rd12;
            ld.global.f32 %f4, [%rd13];
            mul.f32 %f5, %f3, %f2;
            fma.rn.f32 %f1, %f5, %f4, %f1;
            add.u32 %r7, %r7, 1;
            bra LOOP_ROWS;
        ROWS_DONE:
            mul.wide.u32 %rd14, %r6, 4;
            add.s64 %rd15, %rd8, %rd14;
            st.global.f32 [%rd15], %f1;
        DONE:
            ret;
        }
    """.trimIndent()

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
