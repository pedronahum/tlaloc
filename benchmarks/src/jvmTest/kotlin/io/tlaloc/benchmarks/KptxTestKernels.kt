package io.tlaloc.benchmarks

import io.tlaloc.runtime.pjrt.kptx.KptxKernelRegistry
import java.nio.file.Path

/**
 * KPTX v1.8/1.9 (§0.4.336/§0.4.337) — the hand-written PTX kernels the
 * benchmarks-module tests register, plus idempotent registration (the
 * registry is process-permanent and throws on duplicate (plugin, name);
 * multiple tests in the same Gradle worker JVM share one registration).
 *
 * v1 keeps kernel text in test sources; the v2 DSL arc moves kernel
 * authoring into main source sets (docs/KPTX_PLAN.md tasks 9–15).
 */
internal object KptxTestKernels {

    /**
     * RMS-norm forward, eps-operand variant matching the [io.tlaloc.ir.recognizer.kernel.RmsNormKernel]
     * template's claimed COARSENED signature `(x, eps[rows,1]) -> y` (the
     * coarsened RmsNorm is unweighted; eps loaded per CTA). One CTA per
     * token row, 256 threads, strided fma sum-of-squares, shared-memory
     * tree reduction (1 KB static SMEM), `sqrt.rn`+`rcp.rn` normalizer.
     */
    val rmsNormEpsPtx = """
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

    private val registeredPlugins = HashSet<Path>()

    /** Register `@kptx_rms_norm` on [pluginPath] exactly once per JVM. */
    @Synchronized
    fun ensureRmsNormRegistered(pluginPath: Path) {
        if (!registeredPlugins.add(pluginPath)) return
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
    }
}
