package io.tlaloc.kptx

import kotlin.jvm.Synchronized

/**
 * The KPTX production kernel library, written as [PtxKernelTemplate]s
 * and multi-kernel [PtxModule]s. This file is the **single source** for
 * the KPTX kernels — the runtime/benchmark tests register
 * `specialize(...).emitPtx()` output, and the GPU oracle and end-to-end
 * tests check the emitted kernels numerically on every run.
 *
 * The symbolic `block` shape sizes the shared-memory reduction scratch
 * (`4·block` bytes per array), so specializing at any block size emits a
 * consistent kernel rather than overflowing a fixed-size smem array.
 *
 * Emitted text is canonical: byte-stable under parse/emit and ISA-clean
 * by construction (every `inst()` is validated at build time).
 */
object KptxKernels {

    /**
     * RMS-norm forward, eps-operand variant — the kernel the
     * [io.tlaloc.ir.recognizer.kernel] RmsNormKernel template claims as
     * `custom_call @kptx_rms_norm(x, eps[rows,1]) -> y`.
     * One CTA per token row, `block` threads: strided fma
     * sum-of-squares, shared-memory tree reduction, `sqrt.rn`+`rcp.rn`
     * normalizer, strided `out = x · r` writes. Launch signature:
     * `(x_ptr, eps_ptr, out_ptr, n_cols)` — n_cols stays a runtime
     * scalar (the launch registry's trailing-i32 marshalling).
     */
    val rmsNormEps: PtxKernelTemplate = PtxKernelTemplate("kptx_rms_norm") { env ->
        val block = env.shape("block")
        val xPtr = param(".u64", "x_ptr")
        val epsPtr = param(".u64", "eps_ptr")
        val outPtr = param(".u64", "out_ptr")
        val nColsP = param(".u32", "n_cols")

        val p1 = pred(); val p2 = pred(); val p3 = pred()
        val r = List(9) { r32() }                       // %r1..%r9
        val f = List(14) { f32() }                      // %f1..%f14
        val rd = List(21) { r64() }                     // %rd1..%rd21
        val sdata = shared("sdata", sizeBytes = 4 * block)

        inst("ld.param.u64", rd[0], mem(xPtr))
        inst("ld.param.u64", rd[1], mem(epsPtr))
        inst("ld.param.u64", rd[2], mem(outPtr))
        inst("ld.param.u32", r[0], mem(nColsP))
        inst("cvta.to.global.u64", rd[3], rd[0])
        inst("cvta.to.global.u64", rd[4], rd[1])
        inst("cvta.to.global.u64", rd[5], rd[2])
        blank()
        inst("mov.u32", r[1], ctaidX)
        inst("mov.u32", r[2], tidX)
        inst("mov.u32", r[3], ntidX)
        blank()
        comment("row base byte offset = row * n_cols * 4")
        inst("mul.lo.u32", r[4], r[1], r[0])
        inst("mul.wide.u32", rd[6], r[4], imm(4))
        inst("add.s64", rd[7], rd[3], rd[6])
        inst("add.s64", rd[8], rd[5], rd[6])
        blank()
        comment("strided sum of squares")
        inst("mov.f32", f[0], imm("0f00000000"))
        inst("mov.u32", r[5], r[2])
        val loopSum = label("LOOP_SUM")
        val sumDone = label("SUM_DONE")
        place(loopSum)
        inst("setp.ge.u32", p1, r[5], r[0])
        inst("bra", sumDone, guard = p1)
        inst("mul.wide.u32", rd[9], r[5], imm(4))
        inst("add.s64", rd[10], rd[7], rd[9])
        inst("ld.global.f32", f[1], mem(rd[10]))
        inst("fma.rn.f32", f[0], f[1], f[1], f[0])
        inst("add.u32", r[5], r[5], r[3])
        inst("bra", loopSum)
        place(sumDone)
        blank()
        inst("mul.wide.u32", rd[11], r[2], imm(4))
        inst("mov.u64", rd[12], sdata)
        inst("add.s64", rd[13], rd[12], rd[11])
        inst("st.shared.f32", mem(rd[13]), f[0])
        inst("bar.sync", imm(0))
        blank()
        comment("shared-memory tree reduction: stride = ntid/2 .. 1")
        inst("shr.u32", r[6], r[3], imm(1))
        val redLoop = label("RED_LOOP")
        val redSkip = label("RED_SKIP")
        val redDone = label("RED_DONE")
        place(redLoop)
        inst("setp.eq.u32", p2, r[6], imm(0))
        inst("bra", redDone, guard = p2)
        inst("setp.ge.u32", p3, r[2], r[6])
        inst("bra", redSkip, guard = p3)
        inst("add.u32", r[7], r[2], r[6])
        inst("mul.wide.u32", rd[14], r[7], imm(4))
        inst("add.s64", rd[15], rd[12], rd[14])
        inst("ld.shared.f32", f[2], mem(rd[15]))
        inst("ld.shared.f32", f[3], mem(rd[13]))
        inst("add.f32", f[4], f[2], f[3])
        inst("st.shared.f32", mem(rd[13]), f[4])
        place(redSkip)
        inst("bar.sync", imm(0))
        inst("shr.u32", r[6], r[6], imm(1))
        inst("bra", redLoop)
        place(redDone)
        blank()
        comment("inv_rms = 1 / sqrt(sum/n + eps[row]); eps is the coarsener's")
        comment("[rows, 1] operand, one value per token row")
        inst("ld.shared.f32", f[5], mem(rd[12]))
        inst("cvt.rn.f32.u32", f[6], r[0])
        inst("div.rn.f32", f[7], f[5], f[6])
        inst("mul.wide.u32", rd[16], r[1], imm(4))
        inst("add.s64", rd[17], rd[4], rd[16])
        inst("ld.global.f32", f[8], mem(rd[17]))
        inst("add.f32", f[9], f[7], f[8])
        inst("sqrt.rn.f32", f[10], f[9])
        inst("rcp.rn.f32", f[11], f[10])
        blank()
        comment("strided write: out = x * inv_rms")
        inst("mov.u32", r[8], r[2])
        val loopOut = label("LOOP_OUT")
        val outDone = label("OUT_DONE")
        place(loopOut)
        inst("setp.ge.u32", p1, r[8], r[0])
        inst("bra", outDone, guard = p1)
        inst("mul.wide.u32", rd[18], r[8], imm(4))
        inst("add.s64", rd[19], rd[7], rd[18])
        inst("ld.global.f32", f[12], mem(rd[19]))
        inst("mul.f32", f[13], f[12], f[11])
        inst("add.s64", rd[20], rd[8], rd[18])
        inst("st.global.f32", mem(rd[20]), f[13])
        inst("add.u32", r[8], r[8], r[3])
        inst("bra", loopOut)
        place(outDone)
        inst("ret")
    }

    /**
     * RMS-norm backward, dx half: with
     * `r = 1/sqrt(mean(x²)+eps)`, `s = Σ dy·w·x`, computes
     * `dx = (dy·w)·r − x·r³·s/D` and writes `inv_rms[row] = r` for the
     * dw kernel to consume. One CTA per row, dual shared-memory tree
     * (two `4·block`-byte arrays). Launch signature:
     * `(x_ptr, w_ptr, dy_ptr, dx_ptr, invr_ptr, n_cols)`.
     */
    val rmsNormBwdDx: PtxKernelTemplate = PtxKernelTemplate("kptx_rms_norm_bwd_dx") { env ->
        val block = env.shape("block")
        val xPtr = param(".u64", "x_ptr")
        val wPtr = param(".u64", "w_ptr")
        val dyPtr = param(".u64", "dy_ptr")
        val dxPtr = param(".u64", "dx_ptr")
        val invrPtr = param(".u64", "invr_ptr")
        val nColsP = param(".u32", "n_cols")

        val p1 = pred(); val p2 = pred(); val p3 = pred(); val p4 = pred(); val p5 = pred()
        val r = List(9) { r32() }
        val f = List(30) { f32() }
        val rd = List(33) { r64() }
        val sdata = shared("sdata", sizeBytes = 4 * block)
        val sdata2 = shared("sdata2", sizeBytes = 4 * block)

        inst("ld.param.u64", rd[0], mem(xPtr))
        inst("ld.param.u64", rd[1], mem(wPtr))
        inst("ld.param.u64", rd[2], mem(dyPtr))
        inst("ld.param.u64", rd[3], mem(dxPtr))
        inst("ld.param.u64", rd[4], mem(invrPtr))
        inst("ld.param.u32", r[0], mem(nColsP))
        inst("cvta.to.global.u64", rd[5], rd[0])
        inst("cvta.to.global.u64", rd[6], rd[1])
        inst("cvta.to.global.u64", rd[7], rd[2])
        inst("cvta.to.global.u64", rd[8], rd[3])
        inst("cvta.to.global.u64", rd[9], rd[4])
        blank()
        inst("mov.u32", r[1], ctaidX)
        inst("mov.u32", r[2], tidX)
        inst("mov.u32", r[3], ntidX)
        blank()
        comment("row base byte offset = row * n_cols * 4")
        inst("mul.lo.u32", r[4], r[1], r[0])
        inst("mul.wide.u32", rd[10], r[4], imm(4))
        inst("add.s64", rd[11], rd[5], rd[10], comment = "x row")
        inst("add.s64", rd[12], rd[7], rd[10], comment = "dy row")
        inst("add.s64", rd[13], rd[8], rd[10], comment = "dx row")
        blank()
        comment("strided accumulate: sumsq += x*x ; s += (dy*w)*x")
        inst("mov.f32", f[0], imm("0f00000000"))
        inst("mov.f32", f[1], imm("0f00000000"))
        inst("mov.u32", r[5], r[2])
        val loopAcc = label("LOOP_ACC")
        val accDone = label("ACC_DONE")
        place(loopAcc)
        inst("setp.ge.u32", p1, r[5], r[0])
        inst("bra", accDone, guard = p1)
        inst("mul.wide.u32", rd[14], r[5], imm(4))
        inst("add.s64", rd[15], rd[11], rd[14])
        inst("ld.global.f32", f[2], mem(rd[15]))
        inst("add.s64", rd[16], rd[12], rd[14])
        inst("ld.global.f32", f[3], mem(rd[16]))
        inst("add.s64", rd[17], rd[6], rd[14])
        inst("ld.global.f32", f[4], mem(rd[17]))
        inst("fma.rn.f32", f[0], f[2], f[2], f[0])
        inst("mul.f32", f[5], f[3], f[4])
        inst("fma.rn.f32", f[1], f[5], f[2], f[1])
        inst("add.u32", r[5], r[5], r[3])
        inst("bra", loopAcc)
        place(accDone)
        blank()
        comment("sdata[tid] = sumsq partial, sdata2[tid] = s partial")
        inst("mul.wide.u32", rd[18], r[2], imm(4))
        inst("mov.u64", rd[19], sdata)
        inst("add.s64", rd[20], rd[19], rd[18])
        inst("st.shared.f32", mem(rd[20]), f[0])
        inst("mov.u64", rd[21], sdata2)
        inst("add.s64", rd[22], rd[21], rd[18])
        inst("st.shared.f32", mem(rd[22]), f[1])
        inst("bar.sync", imm(0))
        blank()
        comment("dual shared-memory tree reduction: stride = ntid/2 .. 1")
        inst("shr.u32", r[6], r[3], imm(1))
        val redLoop = label("RED_LOOP")
        val redSkip = label("RED_SKIP")
        val redDone = label("RED_DONE")
        place(redLoop)
        inst("setp.eq.u32", p2, r[6], imm(0))
        inst("bra", redDone, guard = p2)
        inst("setp.ge.u32", p3, r[2], r[6])
        inst("bra", redSkip, guard = p3)
        inst("add.u32", r[7], r[2], r[6])
        inst("mul.wide.u32", rd[23], r[7], imm(4))
        inst("add.s64", rd[24], rd[19], rd[23])
        inst("ld.shared.f32", f[6], mem(rd[24]))
        inst("ld.shared.f32", f[7], mem(rd[20]))
        inst("add.f32", f[8], f[6], f[7])
        inst("st.shared.f32", mem(rd[20]), f[8])
        inst("add.s64", rd[25], rd[21], rd[23])
        inst("ld.shared.f32", f[9], mem(rd[25]))
        inst("ld.shared.f32", f[10], mem(rd[22]))
        inst("add.f32", f[11], f[9], f[10])
        inst("st.shared.f32", mem(rd[22]), f[11])
        place(redSkip)
        inst("bar.sync", imm(0))
        inst("shr.u32", r[6], r[6], imm(1))
        inst("bra", redLoop)
        place(redDone)
        blank()
        comment("r = 1/sqrt(sumsq/n + eps); c = r^3 * (s/n)")
        inst("ld.shared.f32", f[12], mem(rd[19]))
        inst("ld.shared.f32", f[13], mem(rd[21]))
        inst("cvt.rn.f32.u32", f[14], r[0])
        inst("div.rn.f32", f[15], f[12], f[14])
        inst("add.f32", f[16], f[15], imm("0f3727C5AC"))
        inst("sqrt.rn.f32", f[17], f[16])
        inst("rcp.rn.f32", f[18], f[17])
        inst("mul.f32", f[19], f[18], f[18])
        inst("mul.f32", f[20], f[19], f[18])
        inst("div.rn.f32", f[21], f[13], f[14])
        inst("mul.f32", f[22], f[20], f[21])
        blank()
        comment("thread 0 writes inv_rms[row]")
        val skipInvr = label("SKIP_INVR")
        inst("setp.ne.u32", p4, r[2], imm(0))
        inst("bra", skipInvr, guard = p4)
        inst("mul.wide.u32", rd[26], r[1], imm(4))
        inst("add.s64", rd[27], rd[9], rd[26])
        inst("st.global.f32", mem(rd[27]), f[18])
        place(skipInvr)
        blank()
        comment("strided write: dx = (dy*w)*r - x*c")
        inst("mov.u32", r[8], r[2])
        val loopOut = label("LOOP_OUT")
        val outDone = label("OUT_DONE")
        place(loopOut)
        inst("setp.ge.u32", p5, r[8], r[0])
        inst("bra", outDone, guard = p5)
        inst("mul.wide.u32", rd[28], r[8], imm(4))
        inst("add.s64", rd[29], rd[11], rd[28])
        inst("ld.global.f32", f[23], mem(rd[29]))
        inst("add.s64", rd[30], rd[12], rd[28])
        inst("ld.global.f32", f[24], mem(rd[30]))
        inst("add.s64", rd[31], rd[6], rd[28])
        inst("ld.global.f32", f[25], mem(rd[31]))
        inst("mul.f32", f[26], f[24], f[25])
        inst("mul.f32", f[27], f[26], f[18])
        inst("mul.f32", f[28], f[23], f[22])
        inst("sub.f32", f[29], f[27], f[28])
        inst("add.s64", rd[32], rd[13], rd[28])
        inst("st.global.f32", mem(rd[32]), f[29])
        inst("add.u32", r[8], r[8], r[3])
        inst("bra", loopOut)
        place(outDone)
        inst("ret")
    }

    /**
     * RoPE forward, the recognizer's SUB-form/cos-first
     * recombination (the LlamaDecoder shape):
     * `out = x_real·cos(θ) − x_imag·sin(θ)`, elementwise. Grid-stride
     * one-thread-per-element; `n` (total elements) stays a runtime
     * trailing-i32 scalar. Uses PTX's `cos.approx`/`sin.approx` — the
     * only sin/cos PTX offers; at RoPE's angle magnitudes the approx
     * error is ~1e-6 absolute, pinned against the XLA oracle by
     * KptxRopeKernelTest. Launch signature:
     * `(x_real_ptr, x_imag_ptr, theta_ptr, out_ptr, n)`.
     */
    val rope: PtxKernelTemplate = PtxKernelTemplate("kptx_rope") { _ ->
        val xrPtr = param(".u64", "x_real_ptr")
        val xiPtr = param(".u64", "x_imag_ptr")
        val thPtr = param(".u64", "theta_ptr")
        val outPtr = param(".u64", "out_ptr")
        val nP = param(".u32", "n")

        val p1 = pred()
        val r = List(6) { r32() }
        val f = List(8) { f32() }
        val rd = List(13) { r64() }

        inst("ld.param.u64", rd[0], mem(xrPtr))
        inst("ld.param.u64", rd[1], mem(xiPtr))
        inst("ld.param.u64", rd[2], mem(thPtr))
        inst("ld.param.u64", rd[3], mem(outPtr))
        inst("ld.param.u32", r[0], mem(nP))
        inst("cvta.to.global.u64", rd[4], rd[0])
        inst("cvta.to.global.u64", rd[5], rd[1])
        inst("cvta.to.global.u64", rd[6], rd[2])
        inst("cvta.to.global.u64", rd[7], rd[3])
        blank()
        comment("element index = ctaid * ntid + tid; guard idx < n")
        inst("mov.u32", r[1], ctaidX)
        inst("mov.u32", r[2], ntidX)
        inst("mov.u32", r[3], tidX)
        inst("mad.lo.u32", r[4], r[1], r[2], r[3])
        val done = label("DONE")
        inst("setp.ge.u32", p1, r[4], r[0])
        inst("bra", done, guard = p1)
        blank()
        comment("out = x_real*cos(theta) - x_imag*sin(theta)")
        inst("mul.wide.u32", rd[8], r[4], imm(4))
        inst("add.s64", rd[9], rd[6], rd[8])
        inst("ld.global.f32", f[0], mem(rd[9]), comment = "theta")
        inst("cos.approx.f32", f[1], f[0])
        inst("sin.approx.f32", f[2], f[0])
        inst("add.s64", rd[10], rd[4], rd[8])
        inst("ld.global.f32", f[3], mem(rd[10]), comment = "x_real")
        inst("add.s64", rd[11], rd[5], rd[8])
        inst("ld.global.f32", f[4], mem(rd[11]), comment = "x_imag")
        inst("mul.f32", f[5], f[3], f[1])
        inst("mul.f32", f[6], f[4], f[2])
        inst("sub.f32", f[7], f[5], f[6])
        inst("add.s64", rd[12], rd[7], rd[8])
        inst("st.global.f32", mem(rd[12]), f[7])
        place(done)
        inst("ret")
    }

    /**
     * CrossEntropy forward as a **two-stage launch chain**
     * behind one custom_call: the coarsened CE returns a scalar
     * (`loss = Σ_ij labels·log(softmax(logits))`, labels-left, full
     * reduction), which needs a cross-row stage; the intermediate
     * `row_loss[rows]` lives in an extra XLA-owned custom_call result
     * (never handler-allocated scratch), and the registry launches both
     * stages back-to-back on XLA's stream.
     *
     * Stage 1, `kptx_cross_entropy_rows(logits, labels, row_loss,
     * n_cols)` — one CTA per row: strided max reduction, then one
     * strided pass accumulating (Σ exp(l−max), Σ lab·(l−max), Σ lab)
     * into a triple shared-memory tree; thread 0 writes
     * `row_loss = Σ lab·(l−max) − (Σ lab)·ln(Σ exp(l−max))` — the exact
     * algebra of the emitter's max-subtracting softmax → log → mul →
     * sum decomposition. Natural exp/log via the approx-only PTX
     * transcendentals: `ex2(x·log2e)`, `lg2(x)·ln2` (hex-spelled
     * constants; grade pinned vs the XLA oracle).
     *
     * Stage 2, `kptx_cross_entropy_sum(row_loss, loss, n_rows)` — one
     * CTA, strided sum + tree, thread 0 writes the scalar.
     */
    /* Multi-kernel modules come from [ptxModule]; [PtxKernelTemplate]
     * is single-kernel, so the CE chain is cached here by block size. */
    // The module caches below are process-global and read from any thread;
    // each accessor is @Synchronized so a module is built once per key.
    private val ceCache = HashMap<Int, PtxModule>()

    @Synchronized
    fun crossEntropyModule(block: Int): PtxModule = ceCache.getOrPut(block) {
        // log2(e) = 0x3FB8AA3B, ln(2) = 0x3F317218 (f32, bit-exact).
        ptxModule {
            kernel("kptx_cross_entropy_rows") {
                val logitsPtr = param(".u64", "logits_ptr")
                val labelsPtr = param(".u64", "labels_ptr")
                val rowLossPtr = param(".u64", "row_loss_ptr")
                val nColsP = param(".u32", "n_cols")

                val p1 = pred(); val p2 = pred(); val p3 = pred(); val p4 = pred()
                val r = List(8) { r32() }
                val f = List(20) { f32() }
                val rd = List(20) { r64() }
                val smax = shared("smax", sizeBytes = 4 * block)
                val ssum = shared("ssum", sizeBytes = 4 * block)
                val sdot = shared("sdot", sizeBytes = 4 * block)
                val slab = shared("slab", sizeBytes = 4 * block)

                inst("ld.param.u64", rd[0], mem(logitsPtr))
                inst("ld.param.u64", rd[1], mem(labelsPtr))
                inst("ld.param.u64", rd[2], mem(rowLossPtr))
                inst("ld.param.u32", r[0], mem(nColsP))
                inst("cvta.to.global.u64", rd[3], rd[0])
                inst("cvta.to.global.u64", rd[4], rd[1])
                inst("cvta.to.global.u64", rd[5], rd[2])
                blank()
                inst("mov.u32", r[1], ctaidX)
                inst("mov.u32", r[2], tidX)
                inst("mov.u32", r[3], ntidX)
                comment("row bases")
                inst("mul.lo.u32", r[4], r[1], r[0])
                inst("mul.wide.u32", rd[6], r[4], imm(4))
                inst("add.s64", rd[7], rd[3], rd[6], comment = "logits row")
                inst("add.s64", rd[8], rd[4], rd[6], comment = "labels row")
                blank()
                comment("pass 1: strided row max")
                inst("mov.f32", f[0], imm("0fFF800000"), comment = "-inf")
                inst("mov.u32", r[5], r[2])
                val maxLoop = label("MAX_LOOP")
                val maxDone = label("MAX_DONE")
                place(maxLoop)
                inst("setp.ge.u32", p1, r[5], r[0])
                inst("bra", maxDone, guard = p1)
                inst("mul.wide.u32", rd[9], r[5], imm(4))
                inst("add.s64", rd[10], rd[7], rd[9])
                inst("ld.global.f32", f[1], mem(rd[10]))
                inst("max.f32", f[0], f[0], f[1])
                inst("add.u32", r[5], r[5], r[3])
                inst("bra", maxLoop)
                place(maxDone)
                inst("mul.wide.u32", rd[11], r[2], imm(4))
                inst("mov.u64", rd[12], smax)
                inst("add.s64", rd[13], rd[12], rd[11])
                inst("st.shared.f32", mem(rd[13]), f[0])
                inst("bar.sync", imm(0))
                comment("max tree")
                inst("shr.u32", r[6], r[3], imm(1))
                val mredLoop = label("MRED_LOOP")
                val mredSkip = label("MRED_SKIP")
                val mredDone = label("MRED_DONE")
                place(mredLoop)
                inst("setp.eq.u32", p2, r[6], imm(0))
                inst("bra", mredDone, guard = p2)
                inst("setp.ge.u32", p3, r[2], r[6])
                inst("bra", mredSkip, guard = p3)
                inst("add.u32", r[7], r[2], r[6])
                inst("mul.wide.u32", rd[14], r[7], imm(4))
                inst("add.s64", rd[15], rd[12], rd[14])
                inst("ld.shared.f32", f[2], mem(rd[15]))
                inst("ld.shared.f32", f[3], mem(rd[13]))
                inst("max.f32", f[4], f[2], f[3])
                inst("st.shared.f32", mem(rd[13]), f[4])
                place(mredSkip)
                inst("bar.sync", imm(0))
                inst("shr.u32", r[6], r[6], imm(1))
                inst("bra", mredLoop)
                place(mredDone)
                inst("ld.shared.f32", f[5], mem(rd[12]), comment = "row max")
                inst("bar.sync", imm(0))
                blank()
                comment("pass 2: strided (sum exp(l-max), sum lab*(l-max), sum lab)")
                inst("mov.f32", f[6], imm("0f00000000"))
                inst("mov.f32", f[7], imm("0f00000000"))
                inst("mov.f32", f[8], imm("0f00000000"))
                inst("mov.u32", r[5], r[2])
                val accLoop = label("ACC_LOOP")
                val accDone = label("ACC_DONE")
                place(accLoop)
                inst("setp.ge.u32", p1, r[5], r[0])
                inst("bra", accDone, guard = p1)
                inst("mul.wide.u32", rd[9], r[5], imm(4))
                inst("add.s64", rd[10], rd[7], rd[9])
                inst("ld.global.f32", f[9], mem(rd[10]), comment = "logit")
                inst("add.s64", rd[16], rd[8], rd[9])
                inst("ld.global.f32", f[10], mem(rd[16]), comment = "label")
                inst("sub.f32", f[11], f[9], f[5], comment = "l - max")
                inst("mul.f32", f[12], f[11], imm("0f3FB8AA3B"), comment = "* log2(e)")
                inst("ex2.approx.f32", f[13], f[12])
                inst("add.f32", f[6], f[6], f[13])
                inst("fma.rn.f32", f[7], f[10], f[11], f[7])
                inst("add.f32", f[8], f[8], f[10])
                inst("add.u32", r[5], r[5], r[3])
                inst("bra", accLoop)
                place(accDone)
                inst("mov.u64", rd[12], ssum)
                inst("add.s64", rd[13], rd[12], rd[11])
                inst("st.shared.f32", mem(rd[13]), f[6])
                inst("mov.u64", rd[17], sdot)
                inst("add.s64", rd[18], rd[17], rd[11])
                inst("st.shared.f32", mem(rd[18]), f[7])
                inst("mov.u64", rd[19], slab)
                inst("add.s64", rd[15], rd[19], rd[11])
                inst("st.shared.f32", mem(rd[15]), f[8])
                inst("bar.sync", imm(0))
                comment("triple tree")
                inst("shr.u32", r[6], r[3], imm(1))
                val tredLoop = label("TRED_LOOP")
                val tredSkip = label("TRED_SKIP")
                val tredDone = label("TRED_DONE")
                place(tredLoop)
                inst("setp.eq.u32", p2, r[6], imm(0))
                inst("bra", tredDone, guard = p2)
                inst("setp.ge.u32", p3, r[2], r[6])
                inst("bra", tredSkip, guard = p3)
                inst("add.u32", r[7], r[2], r[6])
                inst("mul.wide.u32", rd[14], r[7], imm(4))
                inst("add.s64", rd[16], rd[12], rd[14])
                inst("ld.shared.f32", f[9], mem(rd[16]))
                inst("ld.shared.f32", f[10], mem(rd[13]))
                inst("add.f32", f[11], f[9], f[10])
                inst("st.shared.f32", mem(rd[13]), f[11])
                inst("add.s64", rd[16], rd[17], rd[14])
                inst("ld.shared.f32", f[9], mem(rd[16]))
                inst("ld.shared.f32", f[10], mem(rd[18]))
                inst("add.f32", f[11], f[9], f[10])
                inst("st.shared.f32", mem(rd[18]), f[11])
                inst("add.s64", rd[16], rd[19], rd[14])
                inst("ld.shared.f32", f[9], mem(rd[16]))
                inst("ld.shared.f32", f[10], mem(rd[15]))
                inst("add.f32", f[11], f[9], f[10])
                inst("st.shared.f32", mem(rd[15]), f[11])
                place(tredSkip)
                inst("bar.sync", imm(0))
                inst("shr.u32", r[6], r[6], imm(1))
                inst("bra", tredLoop)
                place(tredDone)
                blank()
                comment("thread 0: row_loss = dot - labSum * ln(sumExp)")
                val done = label("DONE")
                inst("setp.ne.u32", p4, r[2], imm(0))
                inst("bra", done, guard = p4)
                inst("ld.shared.f32", f[12], mem(rd[12]), comment = "sum exp")
                inst("ld.shared.f32", f[13], mem(rd[17]), comment = "dot")
                inst("ld.shared.f32", f[14], mem(rd[19]), comment = "lab sum")
                inst("lg2.approx.f32", f[15], f[12])
                inst("mul.f32", f[16], f[15], imm("0f3F317218"), comment = "* ln(2)")
                inst("mul.f32", f[17], f[14], f[16])
                inst("sub.f32", f[18], f[13], f[17])
                inst("mul.wide.u32", rd[9], r[1], imm(4))
                inst("add.s64", rd[10], rd[5], rd[9])
                inst("st.global.f32", mem(rd[10]), f[18])
                place(done)
                inst("ret")
            }

            kernel("kptx_cross_entropy_sum") {
                val rowLossPtr = param(".u64", "row_loss_ptr")
                val lossPtr = param(".u64", "loss_ptr")
                val nRowsP = param(".u32", "n_rows")

                val p1 = pred(); val p2 = pred(); val p3 = pred(); val p4 = pred()
                val r = List(6) { r32() }
                val f = List(6) { f32() }
                val rd = List(10) { r64() }
                val sacc = shared("sacc", sizeBytes = 4 * block)

                inst("ld.param.u64", rd[0], mem(rowLossPtr))
                inst("ld.param.u64", rd[1], mem(lossPtr))
                inst("ld.param.u32", r[0], mem(nRowsP))
                inst("cvta.to.global.u64", rd[2], rd[0])
                inst("cvta.to.global.u64", rd[3], rd[1])
                inst("mov.u32", r[1], tidX)
                inst("mov.u32", r[2], ntidX)
                blank()
                inst("mov.f32", f[0], imm("0f00000000"))
                inst("mov.u32", r[3], r[1])
                val loop = label("LOOP")
                val loopDone = label("LOOP_DONE")
                place(loop)
                inst("setp.ge.u32", p1, r[3], r[0])
                inst("bra", loopDone, guard = p1)
                inst("mul.wide.u32", rd[4], r[3], imm(4))
                inst("add.s64", rd[5], rd[2], rd[4])
                inst("ld.global.f32", f[1], mem(rd[5]))
                inst("add.f32", f[0], f[0], f[1])
                inst("add.u32", r[3], r[3], r[2])
                inst("bra", loop)
                place(loopDone)
                inst("mul.wide.u32", rd[6], r[1], imm(4))
                inst("mov.u64", rd[7], sacc)
                inst("add.s64", rd[8], rd[7], rd[6])
                inst("st.shared.f32", mem(rd[8]), f[0])
                inst("bar.sync", imm(0))
                inst("shr.u32", r[4], r[2], imm(1))
                val red = label("RED_LOOP")
                val redSkip = label("RED_SKIP")
                val redDone = label("RED_DONE")
                place(red)
                inst("setp.eq.u32", p2, r[4], imm(0))
                inst("bra", redDone, guard = p2)
                inst("setp.ge.u32", p3, r[1], r[4])
                inst("bra", redSkip, guard = p3)
                inst("add.u32", r[5], r[1], r[4])
                inst("mul.wide.u32", rd[9], r[5], imm(4))
                inst("add.s64", rd[4], rd[7], rd[9])
                inst("ld.shared.f32", f[2], mem(rd[4]))
                inst("ld.shared.f32", f[3], mem(rd[8]))
                inst("add.f32", f[4], f[2], f[3])
                inst("st.shared.f32", mem(rd[8]), f[4])
                place(redSkip)
                inst("bar.sync", imm(0))
                inst("shr.u32", r[4], r[4], imm(1))
                inst("bra", red)
                place(redDone)
                val done = label("DONE")
                inst("setp.ne.u32", p4, r[1], imm(0))
                inst("bra", done, guard = p4)
                inst("ld.shared.f32", f[5], mem(rd[7]))
                inst("st.global.f32", mem(rd[3]), f[5])
                place(done)
                inst("ret")
            }
        }
    }

    /**
     * Attention forward (`O = softmax(Q·Kᵀ)·V`, the
     * math of the coarsened FlashAttention op: operands `(Q[T,D], Kᵀ[D,T],
     * V[T,D])`, no scale factor, max-subtracting row softmax — exactly
     * the emitter's decomposition) as a **three-stage launch chain**
     * behind one custom_call. The score matrix `S[T,T]` lives in an
     * XLA-owned scratch result (the same mechanism as [crossEntropyModule]):
     *
     *   1. `kptx_attn_scores(q, kt, S, n_t, n_d)` — CTA per score row,
     *      thread-strided columns, D-loop dot per element.
     *   2. `kptx_attn_softmax(S, n_t)` — CTA per row: strided max →
     *      tree, strided Σ exp(s−max) via `ex2(x·log2e)` → tree,
     *      strided normalize-in-place.
     *   3. `kptx_attn_out(S, v, o, n_t, n_d)` — CTA per output row,
     *      thread-strided dims, T-loop dot per element.
     *
     * Correctness-tier f32 loops by design; there is no warp-specialized
     * mma (tensor-core) path.
     */
    private val attnCache = HashMap<Int, PtxModule>()

    @Synchronized
    fun attentionModule(block: Int): PtxModule = attnCache.getOrPut(block) {
        // log2(e) = 0x3FB8AA3B (f32, bit-exact).
        ptxModule {
            kernel("kptx_attn_scores") {
                val qPtr = param(".u64", "q_ptr")
                val ktPtr = param(".u64", "kt_ptr")
                val sPtr = param(".u64", "s_ptr")
                val nTP = param(".u32", "n_t")
                val nDP = param(".u32", "n_d")

                val p1 = pred(); val p2 = pred()
                val r = List(7) { r32() }
                val f = List(3) { f32() }
                val rd = List(14) { r64() }

                inst("ld.param.u64", rd[0], mem(qPtr))
                inst("ld.param.u64", rd[1], mem(ktPtr))
                inst("ld.param.u64", rd[2], mem(sPtr))
                inst("ld.param.u32", r[0], mem(nTP))
                inst("ld.param.u32", r[1], mem(nDP))
                inst("cvta.to.global.u64", rd[3], rd[0])
                inst("cvta.to.global.u64", rd[4], rd[1])
                inst("cvta.to.global.u64", rd[5], rd[2])
                blank()
                inst("mov.u32", r[2], ctaidX)
                inst("mov.u32", r[3], tidX)
                inst("mov.u32", r[4], ntidX)
                comment("row bases: q row i*n_d, s row i*n_t; kt row stride in bytes")
                inst("mul.lo.u32", r[5], r[2], r[1])
                inst("mul.wide.u32", rd[6], r[5], imm(4))
                inst("add.s64", rd[7], rd[3], rd[6], comment = "q row")
                inst("mul.lo.u32", r[5], r[2], r[0])
                inst("mul.wide.u32", rd[6], r[5], imm(4))
                inst("add.s64", rd[8], rd[5], rd[6], comment = "s row")
                inst("mul.wide.u32", rd[9], r[0], imm(4), comment = "kt row stride bytes")
                blank()
                comment("for j strided: S[i,j] = sum_d Q[i,d] * Kt[d,j]")
                inst("mov.u32", r[5], r[3])
                val colLoop = label("COL_LOOP")
                val colDone = label("COL_DONE")
                place(colLoop)
                inst("setp.ge.u32", p1, r[5], r[0])
                inst("bra", colDone, guard = p1)
                inst("mul.wide.u32", rd[10], r[5], imm(4))
                inst("add.s64", rd[11], rd[4], rd[10], comment = "kt walking ptr, starts at [0,j]")
                inst("mov.u64", rd[12], rd[7], comment = "q walking ptr")
                inst("mov.f32", f[0], imm("0f00000000"))
                inst("mov.u32", r[6], imm(0))
                val dLoop = label("D_LOOP")
                val dDone = label("D_DONE")
                place(dLoop)
                inst("setp.ge.u32", p2, r[6], r[1])
                inst("bra", dDone, guard = p2)
                inst("ld.global.f32", f[1], mem(rd[12]))
                inst("ld.global.f32", f[2], mem(rd[11]))
                inst("fma.rn.f32", f[0], f[1], f[2], f[0])
                inst("add.s64", rd[12], rd[12], imm(4))
                inst("add.s64", rd[11], rd[11], rd[9])
                inst("add.u32", r[6], r[6], imm(1))
                inst("bra", dLoop)
                place(dDone)
                inst("add.s64", rd[13], rd[8], rd[10])
                inst("st.global.f32", mem(rd[13]), f[0])
                inst("add.u32", r[5], r[5], r[4])
                inst("bra", colLoop)
                place(colDone)
                inst("ret")
            }

            rowSoftmaxKernel("kptx_attn_softmax", "n_t", block)

            kernel("kptx_attn_out") {
                val sPtr = param(".u64", "s_ptr")
                val vPtr = param(".u64", "v_ptr")
                val oPtr = param(".u64", "o_ptr")
                val nTP = param(".u32", "n_t")
                val nDP = param(".u32", "n_d")

                val p1 = pred(); val p2 = pred()
                val r = List(7) { r32() }
                val f = List(3) { f32() }
                val rd = List(14) { r64() }

                inst("ld.param.u64", rd[0], mem(sPtr))
                inst("ld.param.u64", rd[1], mem(vPtr))
                inst("ld.param.u64", rd[2], mem(oPtr))
                inst("ld.param.u32", r[0], mem(nTP))
                inst("ld.param.u32", r[1], mem(nDP))
                inst("cvta.to.global.u64", rd[3], rd[0])
                inst("cvta.to.global.u64", rd[4], rd[1])
                inst("cvta.to.global.u64", rd[5], rd[2])
                blank()
                inst("mov.u32", r[2], ctaidX)
                inst("mov.u32", r[3], tidX)
                inst("mov.u32", r[4], ntidX)
                comment("row bases: s row i*n_t, o row i*n_d; v row stride bytes = n_d*4")
                inst("mul.lo.u32", r[5], r[2], r[0])
                inst("mul.wide.u32", rd[6], r[5], imm(4))
                inst("add.s64", rd[7], rd[3], rd[6], comment = "s row")
                inst("mul.lo.u32", r[5], r[2], r[1])
                inst("mul.wide.u32", rd[6], r[5], imm(4))
                inst("add.s64", rd[8], rd[5], rd[6], comment = "o row")
                inst("mul.wide.u32", rd[9], r[1], imm(4), comment = "v row stride bytes")
                blank()
                comment("for d strided: O[i,d] = sum_j P[i,j] * V[j,d]")
                inst("mov.u32", r[5], r[3])
                val dimLoop = label("DIM_LOOP")
                val dimDone = label("DIM_DONE")
                place(dimLoop)
                inst("setp.ge.u32", p1, r[5], r[1])
                inst("bra", dimDone, guard = p1)
                inst("mul.wide.u32", rd[10], r[5], imm(4))
                inst("add.s64", rd[11], rd[4], rd[10], comment = "v walking ptr, starts at [0,d]")
                inst("mov.u64", rd[12], rd[7], comment = "p walking ptr")
                inst("mov.f32", f[0], imm("0f00000000"))
                inst("mov.u32", r[6], imm(0))
                val jLoop = label("J_LOOP")
                val jDone = label("J_DONE")
                place(jLoop)
                inst("setp.ge.u32", p2, r[6], r[0])
                inst("bra", jDone, guard = p2)
                inst("ld.global.f32", f[1], mem(rd[12]))
                inst("ld.global.f32", f[2], mem(rd[11]))
                inst("fma.rn.f32", f[0], f[1], f[2], f[0])
                inst("add.s64", rd[12], rd[12], imm(4))
                inst("add.s64", rd[11], rd[11], rd[9])
                inst("add.u32", r[6], r[6], imm(1))
                inst("bra", jLoop)
                place(jDone)
                inst("add.s64", rd[13], rd[8], rd[10])
                inst("st.global.f32", mem(rd[13]), f[0])
                inst("add.u32", r[5], r[5], r[4])
                inst("bra", dimLoop)
                place(dimDone)
                inst("ret")
            }
        }
    }

    /**
     * **Paged attention forward** as a three-stage
     * launch chain behind one `custom_call`, the serving-path sibling of
     * [attentionModule]. Same skeleton — scores → row softmax → output —
     * but K and V are read *through the block table* instead of from a
     * contiguous window, and every row carries its own `seqLen` bound.
     *
     * Operand order follows `OpKind.PAGED_ATTENTION`'s
     * `(query, keyCache, valueCache, blockTables, seqLens) → out`, and the
     * `S[numSeqs·numHeads, numMaxBlocks·blockSize]` score matrix lives in an
     * XLA-owned scratch result (the same mechanism as [crossEntropyModule]):
     *
     *   1. `kptx_paged_scores(q, kcache, btab, slens, S, n_h, n_d, n_bs,
     *      n_kv, n_mb)` — one CTA per (sequence, query head), context
     *      lanes distributed over the CTA's warps. A live lane `j` resolves
     *      `block = blockTables[seq, j / blockSize]`,
     *      `off = j % blockSize`, dots `Q[seq,h,:]` against
     *      `K[block, off, kvHead, :]` and scales; a lane at or past
     *      `seqLen` is written `−inf`. See the warp-per-lane mapping below.
     *   2. `kptx_paged_softmax(S, n_ctx)` — [rowSoftmaxKernel] verbatim.
     *      The `−inf` dead lanes exponentiate to exactly `0`, so no
     *      masking arm is needed there.
     *   3. `kptx_paged_out(S, vcache, btab, slens, o, n_h, n_d, n_bs,
     *      n_kv, n_mb)` — one CTA per output row, accumulating only over
     *      the live lanes, with the context split across the block. See
     *      the context split below.
     *
     * # The warp-per-lane mapping in stage 1
     *
     * Giving each *thread* a context lane `j` and walking `headDim`
     * serially inside it would have the 32 threads of a warp, at a fixed
     * `d`, read 32 **different pages**, which at Llama-3-8B shapes are
     * `numKvHeads · headDim · 4` = 4096 B apart: 32 separate 32-byte
     * sectors fetched for 128 bytes of useful data, an **8× read
     * amplification** (analysis in `docs/KPTX_PAGED_PERF.md`). Stage 3's
     * V walk does not have this problem; it is coalesced by construction.
     *
     * The stage therefore maps a **warp** to a context lane:
     *
     * ```
     *   nWarps = ntid / 32                    // CTA-wide, branch-uniform
     *   w = tid / 32,  lane = tid % 32
     *   warp w owns j = w, w+nWarps, w+2*nWarps, ... < ctx
     *   its 32 lanes split d = lane, lane+32, ... < n_d
     *   warpReduceSumF32(acc); lane 0 scales and stores S[row, j]
     * ```
     *
     * **No barrier and no shared memory.** A warp is already synchronous
     * and `shfl.sync` is what reconverges it; the whole warp shares `j`,
     * so the live/dead test, the page resolution and the block-table load
     * are all warp-uniform (that load is a broadcast, one transaction).
     * Each `K[block, off, kvh, lane…]` request is 32 consecutive f32
     * — one 128 B transaction — and so is the `Q` request. The pointer
     * stride is the literal `128`, which is `32 lanes × 4 B`: a **warp**
     * constant, not a dim-derived one, so no sentinel-dim rule is baked.
     *
     * **Three CTA-uniform reasons to decline the mapping**, each falling
     * back to the thread-per-lane program under `SCALAR_J`, the same
     * pattern as the `nsplit < 2` arm of stage 3:
     *
     * - **`ntid` is not a multiple of 32.** The reduction's member mask
     *   is `0xffffffff`; on a partial warp that is a lie and `shfl.sync`
     *   would wait on lanes that do not exist.
     * - **`ntid < 32`** (`nWarps == 0`), which would leave the `j` loop
     *   with no owner at all.
     * - **`headDim < 32`.** The mapping is still *correct* there — lanes
     *   past `n_d` contribute a zero to the reduction — but it cannot
     *   fill a 128 B transaction, which is the entire point, and it pays
     *   ten reduction ops for fewer than one FMA per lane.
     *
     * The floating-point sum is **reassociated** by this mapping: the dot
     * is a 32-way tree over `shfl` partials rather than a sequential
     * `fma.rn.f32` chain. Against the interpreter's Double paged walk at
     * the test fixture (permuted block table, ragged `seqLens`, GQA group
     * 4) the worst |delta| is **8.940697e-8**, versus 1.1920929e-7 for the
     * sequential chain — a tree sum of 64 terms rounds better than a chain
     * of 64. That is a measurement at one shape, not a guarantee.
     *
     * Measured on a GB10 at the two Llama-3-8B-shaped benchmark points
     * (`docs/KPTX_PAGED_PERF.md`), the warp mapping lowered the claimed
     * lane's device floor from 932.7/1097.5 µs to 596.8/838.3 µs and from
     * 1615.4/1578.1 µs to 1102.4/1211.8 µs (two sessions each,
     * non-overlapping ranges), with the unclaimed control lane unmoved; the
     * TinyLlama-shaped points stayed inside their own spread. Declared
     * registers rise 65 → 73 with **no occupancy change** (3 blocks/SM,
     * 50%, register-limited).
     *
     * # The context split in stage 3
     *
     * Striding threads over `headDim` alone would, at `headDim = 64` with
     * a 256-thread block, leave **192 of 256 threads idle** and have the
     * surviving 64 each walk the whole context serially. At the
     * latency-critical batch-1 shape that dominates: 32 CTAs x 64 live
     * threads is 2048 threads on a 48-SM device.
     *
     * The stage therefore decomposes its block as `(part, d)`:
     *
     * ```
     *   nsplit = ntid / n_d          // CTA-wide, so branch-uniform
     *   part   = tid / n_d,  d = tid % n_d
     *   part p accumulates j = p, p+nsplit, p+2*nsplit, ... < seqLen
     *   smem[tid] = partial; bar.sync
     *   part 0 sums smem[p*n_d + d] for p in 1..nsplit-1 and stores O[row,d]
     * ```
     *
     * Three properties this shape was chosen for, each of which is a way
     * it could have been got wrong:
     *
     * - **The barrier is branch-uniform.** `nsplit` is derived only from
     *   `ntid` and `n_d`, both CTA-wide, so either every thread takes
     *   the split path or none does. A thread whose `part >= nsplit`
     *   (`ntid` not a multiple of `n_d`) skips only the *accumulation*
     *   and still stores its zero and still reaches `bar.sync`.
     * - **`nsplit < 2` keeps the d-strided program**, which is what makes
     *   `headDim >= ntid` safe: there the d-strided loop is both
     *   correct and already fully parallel, and no shared memory or
     *   barrier is touched at all.
     * - **Coalescing is preserved, not sacrificed.** Consecutive `tid`
     *   within a part have consecutive `d`, so a warp's
     *   `V[block, off, kvh, d]` request is still one contiguous run; the
     *   partition index moved to the *slow* axis (`j`), not the fast
     *   one. Threads sharing a `part` also share a `j`, hence a page
     *   resolution and a `P[row, j]` broadcast.
     *
     * The floating-point sum is **reassociated** by this: partial sums
     * are per-partition and added in partition order rather than in
     * `j` order. That is a different rounding of the same mathematics;
     * against the interpreter's Double paged walk the worst |delta| at
     * the test shape (`headDim 64 / ctx 128`) is **1.1920929e-7** either
     * way, because each partition's chain is short enough that neither
     * ordering loses a bit the other keeps. That is a *result*, not a
     * guarantee: a longer context would round differently, and the
     * oracle test is what would say so.
     *
     * **The `nsplit < 2` arm** is not exercised by any shape in the suite
     * (every fixture has `headDim <= 128` against a 256-thread block). It
     * is reached only at `headDim >= 128` with a block of 256 — i.e.
     * `headDim 256`, or a 128-thread launch — and
     * `thePagedOutStageCarriesBothDecompositions` in
     * `PagedAttentionModuleTest` pins that both arms are in the emitted PTX.
     *
     * **GQA is indexing, not new math**: `kvHead = h / (numHeads /
     * numKvHeads)`, computed per CTA from the trailing shape params. The
     * `numHeads == numKvHeads` case falls out with `group == 1`.
     *
     * **The `scale` attribute is baked into the PTX** as an f32 immediate
     * and the module is cached per `(block, scale)`. This is the
     * specialization-cache pattern and is *not* a sentinel-dims violation:
     * `scale` is a compile-time literal on the op, not a value derived from
     * a tensor dimension. Every dim-derived quantity —
     * `numHeads`/`headDim`/`blockSize`/`numKvHeads`/`maxBlocksPerSeq` —
     * arrives as a trailing i32 read from the call frame's buffer shapes at
     * dispatch, and `seqLens` is read from device memory inside the kernel.
     *
     * `seqLens[seq]` is **clamped** to the padded context width before use:
     * an over-long sequence is a scheduler bug, and clamping keeps the
     * kernel inside its buffers while the decode bucket policy refuses the
     * over-cap request by name at the layer that can actually split it.
     *
     * f32 throughout, like the dense [attentionModule] chain; there is no
     * tensor-core path and no bf16 KV-cache pool support here.
     */
    private val pagedAttnCache = HashMap<Pair<Int, Int>, PtxModule>()

    @Synchronized
    fun pagedAttentionModule(block: Int, scale: Float): PtxModule =
        pagedAttnCache.getOrPut(block to scale.toRawBits()) {
            val scaleImm = "0f" + scale.toRawBits().toUInt().toString(16).uppercase().padStart(8, '0')
            ptxModule {
                kernel("kptx_paged_scores") {
                    val qPtr = param(".u64", "q_ptr")
                    val kPtr = param(".u64", "k_ptr")
                    val tabPtr = param(".u64", "tab_ptr")
                    val lenPtr = param(".u64", "len_ptr")
                    val sPtr = param(".u64", "s_ptr")
                    val nHP = param(".u32", "n_h")
                    val nDP = param(".u32", "n_d")
                    val nBsP = param(".u32", "n_bs")
                    val nKvP = param(".u32", "n_kv")
                    val nMbP = param(".u32", "n_mb")

                    val p1 = pred(); val p2 = pred(); val p3 = pred()
                    val r = List(25) { r32() }
                    val f = List(3) { f32() }
                    val rd = List(20) { r64() }

                    inst("ld.param.u64", rd[0], mem(qPtr))
                    inst("ld.param.u64", rd[1], mem(kPtr))
                    inst("ld.param.u64", rd[2], mem(tabPtr))
                    inst("ld.param.u64", rd[3], mem(lenPtr))
                    inst("ld.param.u64", rd[4], mem(sPtr))
                    inst("ld.param.u32", r[0], mem(nHP))
                    inst("ld.param.u32", r[1], mem(nDP))
                    inst("ld.param.u32", r[2], mem(nBsP))
                    inst("ld.param.u32", r[3], mem(nKvP))
                    inst("ld.param.u32", r[4], mem(nMbP))
                    inst("cvta.to.global.u64", rd[5], rd[0])
                    inst("cvta.to.global.u64", rd[6], rd[1])
                    inst("cvta.to.global.u64", rd[7], rd[2])
                    inst("cvta.to.global.u64", rd[8], rd[3])
                    inst("cvta.to.global.u64", rd[9], rd[4])
                    blank()
                    comment("row = ctaid = seq*n_h + h; kv head = h / (n_h / n_kv)")
                    inst("mov.u32", r[5], ctaidX)
                    inst("mov.u32", r[6], tidX)
                    inst("mov.u32", r[7], ntidX)
                    inst("div.u32", r[8], r[5], r[0], comment = "seq")
                    inst("mul.lo.u32", r[9], r[8], r[0])
                    inst("sub.u32", r[10], r[5], r[9], comment = "h")
                    inst("div.u32", r[11], r[0], r[3], comment = "GQA group")
                    inst("div.u32", r[11], r[10], r[11], comment = "kv head")
                    inst("mul.lo.u32", r[12], r[4], r[2], comment = "padded context width")
                    blank()
                    comment("live lanes = min(seqLens[seq], ctx); an over-long seqLen is clamped, never read")
                    inst("mul.wide.u32", rd[10], r[8], imm(4))
                    inst("add.s64", rd[10], rd[8], rd[10])
                    inst("ld.global.u32", r[13], mem(rd[10]))
                    inst("setp.gt.u32", p1, r[13], r[12])
                    inst("mov.u32", r[13], r[12], guard = p1)
                    blank()
                    inst("mul.lo.u32", r[14], r[5], r[1])
                    inst("mul.wide.u32", rd[11], r[14], imm(4))
                    inst("add.s64", rd[11], rd[5], rd[11], comment = "q row")
                    inst("mul.lo.u32", r[14], r[5], r[12])
                    inst("mul.wide.u32", rd[12], r[14], imm(4))
                    inst("add.s64", rd[12], rd[9], rd[12], comment = "s row")
                    inst("mul.lo.u32", r[14], r[8], r[4])
                    inst("mul.wide.u32", rd[13], r[14], imm(4))
                    inst("add.s64", rd[13], rd[7], rd[13], comment = "block-table row")
                    blank()
                    comment("0.4.494: warp-per-lane mapping, taken when a warp load is a full 128 B run")
                    comment("guard is CTA-uniform: ntid a multiple of 32, at least one warp, n_d >= 32")
                    val scalarPath = label("SCALAR_J")
                    inst("and.b32", r[20], r[7], imm(31))
                    inst("setp.ne.u32", p1, r[20], imm(0))
                    inst("bra", scalarPath, guard = p1, comment = "partial warp: shfl's full mask would be a lie")
                    inst("shr.u32", r[20], r[7], imm(5), comment = "nWarps")
                    inst("setp.eq.u32", p1, r[20], imm(0))
                    inst("bra", scalarPath, guard = p1)
                    inst("setp.lt.u32", p1, r[1], imm(32))
                    inst("bra", scalarPath, guard = p1, comment = "headDim < 32 cannot fill a warp's transaction")
                    blank()
                    comment("WARP: warp w owns j = w, w+nWarps, ...; its 32 lanes split d = lane, lane+32, ...")
                    inst("shr.u32", r[21], r[6], imm(5), comment = "w")
                    inst("and.b32", r[22], r[6], imm(31), comment = "lane")
                    inst("mul.wide.u32", rd[19], r[22], imm(4), comment = "lane's byte offset into a row")
                    val wLoop = label("W_J_LOOP")
                    val wDone = label("W_J_DONE")
                    val wLive = label("W_LIVE")
                    val wNext = label("W_NEXT")
                    inst("mov.u32", r[23], r[21])
                    place(wLoop)
                    inst("setp.ge.u32", p1, r[23], r[12])
                    inst("bra", wDone, guard = p1)
                    inst("mul.wide.u32", rd[14], r[23], imm(4))
                    inst("add.s64", rd[15], rd[12], rd[14], comment = "&S[row, j]")
                    inst("setp.lt.u32", p2, r[23], r[13])
                    inst("bra", wLive, guard = p2, comment = "UNIFORM: every lane of the warp shares j")
                    inst("setp.ne.u32", p3, r[22], imm(0))
                    inst("bra", wNext, guard = p3)
                    inst("mov.f32", f[0], imm("0fFF800000"), comment = "-inf")
                    inst("st.global.f32", mem(rd[15]), f[0])
                    inst("bra", wNext)
                    place(wLive)
                    comment("page resolution is warp-uniform; the block-table load broadcasts")
                    inst("div.u32", r[16], r[23], r[2], comment = "page index within the sequence")
                    inst("mul.lo.u32", r[17], r[16], r[2])
                    inst("sub.u32", r[17], r[23], r[17], comment = "offset within the page")
                    inst("mul.wide.u32", rd[16], r[16], imm(4))
                    inst("add.s64", rd[16], rd[13], rd[16])
                    inst("ld.global.u32", r[18], mem(rd[16]), comment = "physical block")
                    inst("mul.lo.u32", r[19], r[18], r[2])
                    inst("add.u32", r[19], r[19], r[17])
                    inst("mul.lo.u32", r[19], r[19], r[3])
                    inst("add.u32", r[19], r[19], r[11])
                    inst("mul.lo.u32", r[19], r[19], r[1])
                    inst("mul.wide.u32", rd[17], r[19], imm(4))
                    inst("add.s64", rd[17], rd[6], rd[17], comment = "&K[block, off, kvh, 0]")
                    inst("add.s64", rd[17], rd[17], rd[19], comment = "+ lane: 32 lanes = one 128 B transaction")
                    inst("add.s64", rd[18], rd[11], rd[19], comment = "&Q[seq, h, lane]")
                    inst("mov.f32", f[0], imm("0f00000000"))
                    inst("mov.u32", r[24], r[22])
                    val wdLoop = label("W_D_LOOP")
                    val wdDone = label("W_D_DONE")
                    place(wdLoop)
                    inst("setp.ge.u32", p3, r[24], r[1])
                    inst("bra", wdDone, guard = p3)
                    inst("ld.global.f32", f[1], mem(rd[18]))
                    inst("ld.global.f32", f[2], mem(rd[17]))
                    inst("fma.rn.f32", f[0], f[1], f[2], f[0])
                    inst("add.s64", rd[18], rd[18], imm(128), comment = "32 lanes x 4 B, a warp constant")
                    inst("add.s64", rd[17], rd[17], imm(128))
                    inst("add.u32", r[24], r[24], imm(32))
                    inst("bra", wdLoop)
                    place(wdDone)
                    comment("every lane reaches this: the d loop's trip count differs, the reduction does not")
                    warpReduceSumF32(f[0])
                    inst("setp.ne.u32", p3, r[22], imm(0))
                    inst("bra", wNext, guard = p3, comment = "lane 0 holds the dot and owns S[row, j]")
                    inst("mul.f32", f[0], f[0], imm(scaleImm), comment = "the op's scale attr, baked")
                    inst("st.global.f32", mem(rd[15]), f[0])
                    place(wNext)
                    inst("add.u32", r[23], r[23], r[20])
                    inst("bra", wLoop)
                    place(wDone)
                    inst("ret")
                    blank()
                    comment("for j strided: live -> scale * dot(Q[seq,h,:], K[page(j),kvh,:]), dead -> -inf")
                    place(scalarPath)
                    inst("mov.u32", r[15], r[6])
                    val jLoop = label("J_LOOP")
                    val jDone = label("J_DONE")
                    val live = label("LIVE")
                    val next = label("NEXT")
                    place(jLoop)
                    inst("setp.ge.u32", p1, r[15], r[12])
                    inst("bra", jDone, guard = p1)
                    inst("mul.wide.u32", rd[14], r[15], imm(4))
                    inst("add.s64", rd[15], rd[12], rd[14], comment = "&S[row, j]")
                    inst("setp.lt.u32", p2, r[15], r[13])
                    inst("bra", live, guard = p2)
                    inst("mov.f32", f[0], imm("0fFF800000"), comment = "-inf")
                    inst("st.global.f32", mem(rd[15]), f[0])
                    inst("bra", next)
                    place(live)
                    inst("div.u32", r[16], r[15], r[2], comment = "page index within the sequence")
                    inst("mul.lo.u32", r[17], r[16], r[2])
                    inst("sub.u32", r[17], r[15], r[17], comment = "offset within the page")
                    inst("mul.wide.u32", rd[16], r[16], imm(4))
                    inst("add.s64", rd[16], rd[13], rd[16])
                    inst("ld.global.u32", r[18], mem(rd[16]), comment = "physical block")
                    inst("mul.lo.u32", r[19], r[18], r[2])
                    inst("add.u32", r[19], r[19], r[17])
                    inst("mul.lo.u32", r[19], r[19], r[3])
                    inst("add.u32", r[19], r[19], r[11])
                    inst("mul.lo.u32", r[19], r[19], r[1])
                    inst("mul.wide.u32", rd[17], r[19], imm(4))
                    inst("add.s64", rd[17], rd[6], rd[17], comment = "&K[block, off, kvh, 0]")
                    inst("mov.u64", rd[18], rd[11], comment = "q walking ptr")
                    inst("mov.f32", f[0], imm("0f00000000"))
                    inst("mov.u32", r[16], imm(0))
                    val dLoop = label("D_LOOP")
                    val dDone = label("D_DONE")
                    place(dLoop)
                    inst("setp.ge.u32", p3, r[16], r[1])
                    inst("bra", dDone, guard = p3)
                    inst("ld.global.f32", f[1], mem(rd[18]))
                    inst("ld.global.f32", f[2], mem(rd[17]))
                    inst("fma.rn.f32", f[0], f[1], f[2], f[0])
                    inst("add.s64", rd[18], rd[18], imm(4))
                    inst("add.s64", rd[17], rd[17], imm(4))
                    inst("add.u32", r[16], r[16], imm(1))
                    inst("bra", dLoop)
                    place(dDone)
                    inst("mul.f32", f[0], f[0], imm(scaleImm), comment = "the op's scale attr, baked")
                    inst("st.global.f32", mem(rd[15]), f[0])
                    place(next)
                    inst("add.u32", r[15], r[15], r[7])
                    inst("bra", jLoop)
                    place(jDone)
                    inst("ret")
                }

                rowSoftmaxKernel("kptx_paged_softmax", "n_ctx", block)

                kernel("kptx_paged_out") {
                    val sPtr = param(".u64", "s_ptr")
                    val vPtr = param(".u64", "v_ptr")
                    val tabPtr = param(".u64", "tab_ptr")
                    val lenPtr = param(".u64", "len_ptr")
                    val oPtr = param(".u64", "o_ptr")
                    val nHP = param(".u32", "n_h")
                    val nDP = param(".u32", "n_d")
                    val nBsP = param(".u32", "n_bs")
                    val nKvP = param(".u32", "n_kv")
                    val nMbP = param(".u32", "n_mb")

                    val p1 = pred(); val p2 = pred()
                    val r = List(24) { r32() }
                    val f = List(3) { f32() }
                    val rd = List(21) { r64() }
                    // §0.4.482 — the cross-partition reduction buffer. One
                    // f32 per thread; the split path writes `ntid` of them
                    // and reads `nsplit * n_d <= ntid`, so `block` threads
                    // is the bound, exactly as it is for the row softmax.
                    val sacc = shared("spacc", sizeBytes = 4 * block)

                    inst("ld.param.u64", rd[0], mem(sPtr))
                    inst("ld.param.u64", rd[1], mem(vPtr))
                    inst("ld.param.u64", rd[2], mem(tabPtr))
                    inst("ld.param.u64", rd[3], mem(lenPtr))
                    inst("ld.param.u64", rd[4], mem(oPtr))
                    inst("ld.param.u32", r[0], mem(nHP))
                    inst("ld.param.u32", r[1], mem(nDP))
                    inst("ld.param.u32", r[2], mem(nBsP))
                    inst("ld.param.u32", r[3], mem(nKvP))
                    inst("ld.param.u32", r[4], mem(nMbP))
                    inst("cvta.to.global.u64", rd[5], rd[0])
                    inst("cvta.to.global.u64", rd[6], rd[1])
                    inst("cvta.to.global.u64", rd[7], rd[2])
                    inst("cvta.to.global.u64", rd[8], rd[3])
                    inst("cvta.to.global.u64", rd[9], rd[4])
                    blank()
                    inst("mov.u32", r[5], ctaidX)
                    inst("mov.u32", r[6], tidX)
                    inst("mov.u32", r[7], ntidX)
                    inst("div.u32", r[8], r[5], r[0], comment = "seq")
                    inst("mul.lo.u32", r[9], r[8], r[0])
                    inst("sub.u32", r[10], r[5], r[9], comment = "h")
                    inst("div.u32", r[11], r[0], r[3], comment = "GQA group")
                    inst("div.u32", r[11], r[10], r[11], comment = "kv head")
                    inst("mul.lo.u32", r[12], r[4], r[2], comment = "padded context width")
                    inst("mul.wide.u32", rd[10], r[8], imm(4))
                    inst("add.s64", rd[10], rd[8], rd[10])
                    inst("ld.global.u32", r[13], mem(rd[10]))
                    inst("setp.gt.u32", p1, r[13], r[12])
                    inst("mov.u32", r[13], r[12], guard = p1)
                    blank()
                    inst("mul.lo.u32", r[14], r[5], r[12])
                    inst("mul.wide.u32", rd[11], r[14], imm(4))
                    inst("add.s64", rd[11], rd[5], rd[11], comment = "s row")
                    inst("mul.lo.u32", r[14], r[5], r[1])
                    inst("mul.wide.u32", rd[12], r[14], imm(4))
                    inst("add.s64", rd[12], rd[9], rd[12], comment = "o row")
                    inst("mul.lo.u32", r[14], r[8], r[4])
                    inst("mul.wide.u32", rd[13], r[14], imm(4))
                    inst("add.s64", rd[13], rd[7], rd[13], comment = "block-table row")
                    blank()
                    comment("0.4.482: nsplit = ntid / n_d context partitions; < 2 takes the d-strided scalar path")
                    val scalarPath = label("SCALAR_D")
                    inst("div.u32", r[20], r[7], r[1])
                    inst("setp.lt.u32", p1, r[20], imm(2))
                    inst("bra", scalarPath, guard = p1)
                    blank()
                    comment("SPLIT: thread = (part, d). Part `p` walks j = p, p+nsplit, ...; then reduce over parts.")
                    inst("div.u32", r[21], r[6], r[1], comment = "part")
                    inst("mul.lo.u32", r[22], r[21], r[1])
                    inst("sub.u32", r[22], r[6], r[22], comment = "d")
                    inst("mov.f32", f[0], imm("0f00000000"))
                    val spStore = label("SP_STORE")
                    val spLoop = label("SP_J_LOOP")
                    val spLoopDone = label("SP_J_DONE")
                    val spRed = label("SP_RED")
                    val spRedDone = label("SP_RED_DONE")
                    val spDone = label("SP_DONE")
                    inst("setp.ge.u32", p1, r[21], r[20])
                    inst(
                        "bra", spStore, guard = p1,
                        comment = "ntid not a multiple of n_d: the tail threads contribute a zero",
                    )
                    inst("mov.u32", r[16], r[21])
                    place(spLoop)
                    inst("setp.ge.u32", p2, r[16], r[13])
                    inst("bra", spLoopDone, guard = p2)
                    inst("div.u32", r[17], r[16], r[2], comment = "page index within the sequence")
                    inst("mul.lo.u32", r[18], r[17], r[2])
                    inst("sub.u32", r[18], r[16], r[18], comment = "offset within the page")
                    inst("mul.wide.u32", rd[14], r[17], imm(4))
                    inst("add.s64", rd[14], rd[13], rd[14])
                    inst("ld.global.u32", r[19], mem(rd[14]), comment = "physical block")
                    inst("mul.lo.u32", r[14], r[19], r[2])
                    inst("add.u32", r[14], r[14], r[18])
                    inst("mul.lo.u32", r[14], r[14], r[3])
                    inst("add.u32", r[14], r[14], r[11])
                    inst("mul.lo.u32", r[14], r[14], r[1])
                    inst("add.u32", r[14], r[14], r[22])
                    inst("mul.wide.u32", rd[15], r[14], imm(4))
                    inst("add.s64", rd[15], rd[6], rd[15], comment = "&V[block, off, kvh, d]")
                    inst("mul.wide.u32", rd[16], r[16], imm(4))
                    inst("add.s64", rd[16], rd[11], rd[16], comment = "&P[row, j]")
                    inst("ld.global.f32", f[1], mem(rd[16]))
                    inst("ld.global.f32", f[2], mem(rd[15]))
                    inst("fma.rn.f32", f[0], f[1], f[2], f[0])
                    inst("add.u32", r[16], r[16], r[20])
                    inst("bra", spLoop)
                    place(spLoopDone)
                    place(spStore)
                    inst("mov.u64", rd[20], sacc)
                    inst("mul.wide.u32", rd[19], r[6], imm(4))
                    inst("add.s64", rd[19], rd[20], rd[19])
                    inst("st.shared.f32", mem(rd[19]), f[0])
                    inst(
                        "bar.sync", imm(0),
                        comment = "UNIFORM: nsplit is a CTA-wide quantity, so every thread reaches this",
                    )
                    inst("setp.ne.u32", p1, r[21], imm(0))
                    inst("bra", spDone, guard = p1, comment = "part 0 owns the output row (part 0 <=> tid < n_d)")
                    inst("mov.u32", r[23], imm(1))
                    place(spRed)
                    inst("setp.ge.u32", p2, r[23], r[20])
                    inst("bra", spRedDone, guard = p2)
                    inst("mad.lo.u32", r[14], r[23], r[1], r[22])
                    inst("mul.wide.u32", rd[14], r[14], imm(4))
                    inst("add.s64", rd[14], rd[20], rd[14])
                    inst("ld.shared.f32", f[1], mem(rd[14]))
                    inst("add.f32", f[0], f[0], f[1])
                    inst("add.u32", r[23], r[23], imm(1))
                    inst("bra", spRed)
                    place(spRedDone)
                    inst("mul.wide.u32", rd[15], r[22], imm(4))
                    inst("add.s64", rd[15], rd[12], rd[15])
                    inst("st.global.f32", mem(rd[15]), f[0])
                    place(spDone)
                    inst("ret")
                    blank()
                    comment("for d strided: O[row,d] = sum over LIVE j of P[row,j] * V[page(j),kvh,d]")
                    place(scalarPath)
                    inst("mov.u32", r[15], r[6])
                    val dimLoop = label("DIM_LOOP")
                    val dimDone = label("DIM_DONE")
                    place(dimLoop)
                    inst("setp.ge.u32", p1, r[15], r[1])
                    inst("bra", dimDone, guard = p1)
                    inst("mov.f32", f[0], imm("0f00000000"))
                    inst("mov.u32", r[16], imm(0))
                    val jLoop = label("J_LOOP")
                    val jDone = label("J_DONE")
                    place(jLoop)
                    inst("setp.ge.u32", p2, r[16], r[13])
                    inst("bra", jDone, guard = p2)
                    inst("div.u32", r[17], r[16], r[2], comment = "page index within the sequence")
                    inst("mul.lo.u32", r[18], r[17], r[2])
                    inst("sub.u32", r[18], r[16], r[18], comment = "offset within the page")
                    inst("mul.wide.u32", rd[14], r[17], imm(4))
                    inst("add.s64", rd[14], rd[13], rd[14])
                    inst("ld.global.u32", r[19], mem(rd[14]), comment = "physical block")
                    inst("mul.lo.u32", r[20], r[19], r[2])
                    inst("add.u32", r[20], r[20], r[18])
                    inst("mul.lo.u32", r[20], r[20], r[3])
                    inst("add.u32", r[20], r[20], r[11])
                    inst("mul.lo.u32", r[20], r[20], r[1])
                    inst("add.u32", r[20], r[20], r[15])
                    inst("mul.wide.u32", rd[15], r[20], imm(4))
                    inst("add.s64", rd[15], rd[6], rd[15], comment = "&V[block, off, kvh, d]")
                    inst("mul.wide.u32", rd[16], r[16], imm(4))
                    inst("add.s64", rd[16], rd[11], rd[16], comment = "&P[row, j]")
                    inst("ld.global.f32", f[1], mem(rd[16]))
                    inst("ld.global.f32", f[2], mem(rd[15]))
                    inst("fma.rn.f32", f[0], f[1], f[2], f[0])
                    inst("add.u32", r[16], r[16], imm(1))
                    inst("bra", jLoop)
                    place(jDone)
                    inst("mul.wide.u32", rd[17], r[15], imm(4))
                    inst("add.s64", rd[18], rd[12], rd[17])
                    inst("st.global.f32", mem(rd[18]), f[0])
                    inst("add.u32", r[15], r[15], r[7])
                    inst("bra", dimLoop)
                    place(dimDone)
                    inst("ret")
                }
            }
        }
    /**
     * RMS-norm backward, dw half: `dw_j = Σ_i dy_ij·x_ij·inv_rms_i`,
     * consuming the `inv_rms` vector [rmsNormBwdDx] produced (XLA
     * sequences the two custom_calls via the data dependence). One
     * thread per column. Launch signature:
     * `(x_ptr, dy_ptr, invr_ptr, dw_ptr, n_rows, n_cols)`.
     */
    val rmsNormBwdDw: PtxKernelTemplate = PtxKernelTemplate("kptx_rms_norm_bwd_dw") { _ ->
        val xPtr = param(".u64", "x_ptr")
        val dyPtr = param(".u64", "dy_ptr")
        val invrPtr = param(".u64", "invr_ptr")
        val dwPtr = param(".u64", "dw_ptr")
        val nRowsP = param(".u32", "n_rows")
        val nColsP = param(".u32", "n_cols")

        val p1 = pred(); val p2 = pred()
        val r = List(8) { r32() }
        val f = List(5) { f32() }
        val rd = List(15) { r64() }

        inst("ld.param.u64", rd[0], mem(xPtr))
        inst("ld.param.u64", rd[1], mem(dyPtr))
        inst("ld.param.u64", rd[2], mem(invrPtr))
        inst("ld.param.u64", rd[3], mem(dwPtr))
        inst("ld.param.u32", r[0], mem(nRowsP))
        inst("ld.param.u32", r[1], mem(nColsP))
        inst("cvta.to.global.u64", rd[4], rd[0])
        inst("cvta.to.global.u64", rd[5], rd[1])
        inst("cvta.to.global.u64", rd[6], rd[2])
        inst("cvta.to.global.u64", rd[7], rd[3])
        blank()
        comment("column j = ctaid * ntid + tid; guard j < n_cols")
        inst("mov.u32", r[2], ctaidX)
        inst("mov.u32", r[3], ntidX)
        inst("mov.u32", r[4], tidX)
        inst("mad.lo.u32", r[5], r[2], r[3], r[4])
        val done = label("DONE")
        inst("setp.ge.u32", p1, r[5], r[1])
        inst("bra", done, guard = p1)
        blank()
        comment("dw_j = sum_i dy[i,j] * x[i,j] * inv_rms[i]")
        inst("mov.f32", f[0], imm("0f00000000"))
        inst("mov.u32", r[6], imm(0))
        val loopRows = label("LOOP_ROWS")
        val rowsDone = label("ROWS_DONE")
        place(loopRows)
        inst("setp.ge.u32", p2, r[6], r[0])
        inst("bra", rowsDone, guard = p2)
        inst("mad.lo.u32", r[7], r[6], r[1], r[5])
        inst("mul.wide.u32", rd[8], r[7], imm(4))
        inst("add.s64", rd[9], rd[4], rd[8])
        inst("ld.global.f32", f[1], mem(rd[9]))
        inst("add.s64", rd[10], rd[5], rd[8])
        inst("ld.global.f32", f[2], mem(rd[10]))
        inst("mul.wide.u32", rd[11], r[6], imm(4))
        inst("add.s64", rd[12], rd[6], rd[11])
        inst("ld.global.f32", f[3], mem(rd[12]))
        inst("mul.f32", f[4], f[2], f[1])
        inst("fma.rn.f32", f[0], f[4], f[3], f[0])
        inst("add.u32", r[6], r[6], imm(1))
        inst("bra", loopRows)
        place(rowsDone)
        inst("mul.wide.u32", rd[13], r[5], imm(4))
        inst("add.s64", rd[14], rd[7], rd[13])
        inst("st.global.f32", mem(rd[14]), f[0])
        place(done)
        inst("ret")
    }
}

/**
 * The **row-softmax stage**, shared verbatim by the
 * dense attention chain ([KptxKernels.attentionModule], where the row width
 * is `n_t`) and the paged one ([KptxKernels.pagedAttentionModule], where it
 * is the padded context width `n_ctx`). One CTA per row: strided max → tree
 * reduce, strided Σ `ex2((s−max)·log2e)` → tree reduce, strided
 * normalize-in-place.
 *
 * Extracted rather than duplicated because the paged form needs *exactly*
 * this program: the paged score stage writes `−inf` into every lane at or past
 * `seqLen`, and `exp(−inf − max)` is `0` for any finite max — which it is,
 * since a sequence has at least one live lane. The dead lanes therefore
 * fall out of the softmax on their own and no masking arm is needed here.
 * The `name`/`paramName` parameters let each chain keep its own kernel and
 * parameter names.
 */
private fun ModuleScope.rowSoftmaxKernel(name: String, paramName: String, block: Int) {
    kernel(name) {
        val sPtr = param(".u64", "s_ptr")
        val nTP = param(".u32", paramName)

        val p1 = pred(); val p2 = pred(); val p3 = pred()
        val r = List(7) { r32() }
        val f = List(12) { f32() }
        val rd = List(11) { r64() }
        val smax = shared("smax", sizeBytes = 4 * block)

        inst("ld.param.u64", rd[0], mem(sPtr))
        inst("ld.param.u32", r[0], mem(nTP))
        inst("cvta.to.global.u64", rd[1], rd[0])
        blank()
        inst("mov.u32", r[1], ctaidX)
        inst("mov.u32", r[2], tidX)
        inst("mov.u32", r[3], ntidX)
        inst("mul.lo.u32", r[4], r[1], r[0])
        inst("mul.wide.u32", rd[2], r[4], imm(4))
        inst("add.s64", rd[3], rd[1], rd[2], comment = "s row")
        blank()
        comment("pass 1: strided row max + tree")
        inst("mov.f32", f[0], imm("0fFF800000"), comment = "-inf")
        inst("mov.u32", r[4], r[2])
        val maxLoop = label("MAX_LOOP")
        val maxDone = label("MAX_DONE")
        place(maxLoop)
        inst("setp.ge.u32", p1, r[4], r[0])
        inst("bra", maxDone, guard = p1)
        inst("mul.wide.u32", rd[4], r[4], imm(4))
        inst("add.s64", rd[5], rd[3], rd[4])
        inst("ld.global.f32", f[1], mem(rd[5]))
        inst("max.f32", f[0], f[0], f[1])
        inst("add.u32", r[4], r[4], r[3])
        inst("bra", maxLoop)
        place(maxDone)
        inst("mul.wide.u32", rd[6], r[2], imm(4))
        inst("mov.u64", rd[7], smax)
        inst("add.s64", rd[8], rd[7], rd[6])
        inst("st.shared.f32", mem(rd[8]), f[0])
        inst("bar.sync", imm(0))
        inst("shr.u32", r[5], r[3], imm(1))
        val mred = label("MRED_LOOP")
        val mredSkip = label("MRED_SKIP")
        val mredDone = label("MRED_DONE")
        place(mred)
        inst("setp.eq.u32", p2, r[5], imm(0))
        inst("bra", mredDone, guard = p2)
        inst("setp.ge.u32", p3, r[2], r[5])
        inst("bra", mredSkip, guard = p3)
        inst("add.u32", r[6], r[2], r[5])
        inst("mul.wide.u32", rd[9], r[6], imm(4))
        inst("add.s64", rd[10], rd[7], rd[9])
        inst("ld.shared.f32", f[2], mem(rd[10]))
        inst("ld.shared.f32", f[3], mem(rd[8]))
        inst("max.f32", f[4], f[2], f[3])
        inst("st.shared.f32", mem(rd[8]), f[4])
        place(mredSkip)
        inst("bar.sync", imm(0))
        inst("shr.u32", r[5], r[5], imm(1))
        inst("bra", mred)
        place(mredDone)
        inst("ld.shared.f32", f[5], mem(rd[7]), comment = "row max")
        inst("bar.sync", imm(0))
        blank()
        comment("pass 2: strided sum exp(s-max) + tree")
        inst("mov.f32", f[6], imm("0f00000000"))
        inst("mov.u32", r[4], r[2])
        val sumLoop = label("SUM_LOOP")
        val sumDone = label("SUM_DONE")
        place(sumLoop)
        inst("setp.ge.u32", p1, r[4], r[0])
        inst("bra", sumDone, guard = p1)
        inst("mul.wide.u32", rd[4], r[4], imm(4))
        inst("add.s64", rd[5], rd[3], rd[4])
        inst("ld.global.f32", f[1], mem(rd[5]))
        inst("sub.f32", f[7], f[1], f[5])
        inst("mul.f32", f[8], f[7], imm("0f3FB8AA3B"))
        inst("ex2.approx.f32", f[9], f[8])
        inst("add.f32", f[6], f[6], f[9])
        inst("add.u32", r[4], r[4], r[3])
        inst("bra", sumLoop)
        place(sumDone)
        inst("st.shared.f32", mem(rd[8]), f[6])
        inst("bar.sync", imm(0))
        inst("shr.u32", r[5], r[3], imm(1))
        val sred = label("SRED_LOOP")
        val sredSkip = label("SRED_SKIP")
        val sredDone = label("SRED_DONE")
        place(sred)
        inst("setp.eq.u32", p2, r[5], imm(0))
        inst("bra", sredDone, guard = p2)
        inst("setp.ge.u32", p3, r[2], r[5])
        inst("bra", sredSkip, guard = p3)
        inst("add.u32", r[6], r[2], r[5])
        inst("mul.wide.u32", rd[9], r[6], imm(4))
        inst("add.s64", rd[10], rd[7], rd[9])
        inst("ld.shared.f32", f[2], mem(rd[10]))
        inst("ld.shared.f32", f[3], mem(rd[8]))
        inst("add.f32", f[4], f[2], f[3])
        inst("st.shared.f32", mem(rd[8]), f[4])
        place(sredSkip)
        inst("bar.sync", imm(0))
        inst("shr.u32", r[5], r[5], imm(1))
        inst("bra", sred)
        place(sredDone)
        inst("ld.shared.f32", f[10], mem(rd[7]), comment = "row sum")
        blank()
        comment("pass 3: strided normalize in place")
        inst("mov.u32", r[4], r[2])
        val nrmLoop = label("NRM_LOOP")
        val nrmDone = label("NRM_DONE")
        place(nrmLoop)
        inst("setp.ge.u32", p1, r[4], r[0])
        inst("bra", nrmDone, guard = p1)
        inst("mul.wide.u32", rd[4], r[4], imm(4))
        inst("add.s64", rd[5], rd[3], rd[4])
        inst("ld.global.f32", f[1], mem(rd[5]))
        inst("sub.f32", f[7], f[1], f[5])
        inst("mul.f32", f[8], f[7], imm("0f3FB8AA3B"))
        inst("ex2.approx.f32", f[9], f[8])
        inst("div.rn.f32", f[11], f[9], f[10])
        inst("st.global.f32", mem(rd[5]), f[11])
        inst("add.u32", r[4], r[4], r[3])
        inst("bra", nrmLoop)
        place(nrmDone)
        inst("ret")
    }
}
