package io.tlaloc.kptx
/**
 * KPTX v2.2 (§0.4.339) — the byte-identical round-trip corpus: the five
 * real v1 kernels (§0.4.328–337), copied **verbatim** from their source
 * tests. `parsePtx(text).emitPtx() == text` must hold byte-for-byte on
 * every entry — this is the contract that makes the IR a lossless model
 * of the text and the emitter's format the single canonical style.
 *
 * Copies, not references: the originals live in other modules' test
 * sources (listed per entry). Task 15 inverts the dependency — the DSL
 * becomes the single source and the runtime tests consume emitted PTX.
 * Until then, drift between an original and its corpus copy is caught
 * by the corpus test failing to round-trip any *new* construct.
 */
internal object PtxRoundTripCorpus {
    /** §0.4.328 CudaDriverFfmSmokeTest.addOnePtx. */
    val addOne = """
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry add_one(
            .param .u64 in_ptr,
            .param .u64 out_ptr,
            .param .u32 n
        )
        {
            .reg .pred %p<2>;
            .reg .b32 %r<6>;
            .reg .f32 %f<3>;
            .reg .b64 %rd<8>;

            ld.param.u64 %rd1, [in_ptr];
            ld.param.u64 %rd2, [out_ptr];
            ld.param.u32 %r1, [n];
            cvta.to.global.u64 %rd3, %rd1;
            cvta.to.global.u64 %rd4, %rd2;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.s32 %r5, %r2, %r3, %r4;
            setp.ge.s32 %p1, %r5, %r1;
            @%p1 bra DONE;
            mul.wide.s32 %rd5, %r5, 4;
            add.s64 %rd6, %rd3, %rd5;
            ld.global.f32 %f1, [%rd6];
            add.f32 %f2, %f1, 0f3F800000;
            add.s64 %rd7, %rd4, %rd5;
            st.global.f32 [%rd7], %f2;
        DONE:
            ret;
        }
    """.trimIndent()

    /** §0.4.331 KptxRmsNormKernelTest.rmsNormPtx (w-variant, hard-coded eps). */
    val rmsNormForward = """
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

    /** §0.4.336 KptxTestKernels.rmsNormEpsPtx (eps-operand variant). */
    val rmsNormEps = """
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

    /** §0.4.335 KptxRmsNormBackwardKernelTest.bwdDxPtx (dual smem tree, trailing comments). */
    val rmsNormBwdDx = """
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

    /** §0.4.335 KptxRmsNormBackwardKernelTest.bwdDwPtx (per-column row loop). */
    val rmsNormBwdDw = """
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

    val all = listOf(addOne, rmsNormForward, rmsNormEps, rmsNormBwdDx, rmsNormBwdDw)
}
