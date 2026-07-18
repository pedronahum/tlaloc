package io.tlaloc.benchmarks

import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.RmsNormKernel
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.kptx.KptxKernelRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * KPTX v1.8 (§0.4.336) — **the recognizer-driven loop closes**: the
 * LlamaDecoder-medium forward pipeline
 * `recognize → coarsen → lowerKernelChoice(GB10, {RmsNorm→[RmsNormKernel]})
 * → decomposeCoarsened → toStablehlo` emits two typed-FFI
 * `stablehlo.custom_call @kptx_rms_norm` ops (pre-attn + pre-MLP norms),
 * and that emitter-produced program **executes on the GB10 with the
 * hand-written PTX rms_norm computing inside it**, agreeing with the
 * full-decompose program on the same PJRT-CUDA client.
 *
 * Unlike §0.4.331/§0.4.335 (hand-written MLIR isolating kernel
 * correctness) and §0.4.332 (probe handler isolating the attr pipeline),
 * nothing here is hand-written except the PTX itself: the MLIR comes
 * from the recognizer pipeline, the kernel launch from
 * [KptxKernelRegistry], the numbers from a real Llama-shaped forward.
 *
 * The kernel is the **eps-operand variant** of §0.4.331's rms_norm
 * (the coarsened RmsNorm is unweighted — Llama's weight-multiply lives
 * outside the recognized pattern — and eps arrives as the coarsener's
 * `[rows, 1]` operand, loaded per CTA): one CTA per token
 * row, 256 threads, strided fma sum-of-squares, shared-memory tree
 * reduction, `sqrt.rn`+`rcp.rn` normalizer, `out = x · r`.
 */
class LlamaDecoderKptxRmsNormE2ETest {

    private val kptxRegistry: Map<String, KernelTemplate> = mapOf("RmsNorm" to RmsNormKernel)

    private val rmsNormEpsPtx = """
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry kptx_rms_norm(
            .param .u64 x_ptr,
            .param .u64 eps_ptr,
            .param .u64 out_ptr,
            .param .u32 n_cols
        )
        {
            .reg .pred %p<4>;
            .reg .b32 %r<10>;
            .reg .f32 %f<16>;
            .reg .b64 %rd<22>;
            .shared .align 4 .b8 sdata[1024];

            ld.param.u64 %rd1, [x_ptr];
            ld.param.u64 %rd2, [eps_ptr];
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

            // inv_rms = 1 / sqrt(sum/n + eps[row]); eps is the coarsener's
            // [rows, 1] operand, one value per token row
            ld.shared.f32 %f6, [%rd13];
            cvt.rn.f32.u32 %f7, %r1;
            div.rn.f32 %f8, %f6, %f7;
            mul.wide.u32 %rd20, %r2, 4;
            add.s64 %rd21, %rd5, %rd20;
            ld.global.f32 %f9, [%rd21];
            add.f32 %f10, %f8, %f9;
            sqrt.rn.f32 %f11, %f10;
            rcp.rn.f32 %f12, %f11;

            // strided write: out = x * inv_rms
            mov.u32 %r9, %r3;
        LOOP_OUT:
            setp.ge.u32 %p1, %r9, %r1;
            @%p1 bra OUT_DONE;
            mul.wide.u32 %rd17, %r9, 4;
            add.s64 %rd18, %rd8, %rd17;
            ld.global.f32 %f13, [%rd18];
            mul.f32 %f14, %f13, %f12;
            add.s64 %rd19, %rd9, %rd17;
            st.global.f32 [%rd19], %f14;
            add.u32 %r9, %r9, %r4;
            bra LOOP_OUT;
        OUT_DONE:
            ret;
        }
    """.trimIndent()

    /** GPU-less pin: the recognizer pipeline emits exactly two typed-FFI
     * `@kptx_rms_norm` custom_calls on GB10 and zero on CPU_GENERIC. */
    @Test
    fun emitsTwoKptxRmsNormCustomCallsOnGb10() {
        val gb10 = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10, kptxRegistry,
        ).toStablehlo()
        assertEquals(
            2,
            Regex("stablehlo\\.custom_call @kptx_rms_norm\\(").findAll(gb10).count(),
            "expected the pre-attn and pre-MLP RmsNorms to lower to @kptx_rms_norm",
        )
        assertTrue(
            gb10.contains("api_version = 4 : i32"),
            "KPTX custom_calls must use the typed-FFI convention",
        )

        val cpu = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC, kptxRegistry,
        ).toStablehlo()
        assertEquals(0, Regex("custom_call").findAll(cpu).count(), "CPU_GENERIC must fully decompose")
    }

    /** GPU-gated: the emitter-produced GB10 program runs with the PTX
     * kernel inside and agrees with the full-decompose program. */
    @Test
    fun kptxLoweredForwardAgreesWithDecomposePath() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val pluginPath = PjrtBinaries.pluginPath!!
        assumeTrue(PjrtFfiRegistry.isGpuCustomCallSupported(pluginPath), "no GPU custom-call extension — skipping.")

        KptxKernelRegistry.registerKernel(
            pluginPath,
            "kptx_rms_norm",
            KptxKernelRegistry.LaunchConfig(
                ptx = rmsNormEpsPtx,
                entryName = "kptx_rms_norm",
                grid = { args -> KptxKernelRegistry.Dim3(args[0].dims[0].toInt()) }, // one CTA per row
                block = KptxKernelRegistry.Dim3(256),
                trailingI32Params = { args -> intArrayOf(args[0].dims[1].toInt()) },
            ),
        )

        val kptxFn = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10, kptxRegistry,
        )
        val decomposeFn = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC, kptxRegistry,
        )
        val inputs = llamaSynthesizeInputs(seed = 42L, fn = kptxFn)

        val kptxOut: List<FloatArray>
        val decomposeOut: List<FloatArray>
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            kptxOut = session.runOn(kptxFn, inputs)
            decomposeOut = session.runOn(decomposeFn, inputs)
        }

        assertEquals(decomposeOut.size, kptxOut.size)
        var maxRel = 0f
        for (o in kptxOut.indices) {
            for (i in kptxOut[o].indices) {
                val d = abs(kptxOut[o][i] - decomposeOut[o][i])
                maxRel = max(maxRel, d / max(abs(decomposeOut[o][i]), 1e-6f))
            }
        }
        println(
            "[kptx-llama-e2e] LlamaDecoder-medium forward, 2×@kptx_rms_norm inside: " +
                "loss kptx=${kptxOut[0][0]}, decompose=${decomposeOut[0][0]}, max rel=$maxRel",
        )
        assertTrue(maxRel <= 1e-4f, "kptx-lowered forward diverges from decompose path: max rel $maxRel")
    }
}
