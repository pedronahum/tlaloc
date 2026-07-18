package io.tlaloc.runtime.cuda

import io.tlaloc.kptx.IsaRegClass
import io.tlaloc.kptx.KptxTranspiler
import io.tlaloc.kptx.PtxKernelTemplate
import io.tlaloc.kptx.emitPtx
import io.tlaloc.kptx.imm
import io.tlaloc.kptx.parsePtx
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * KPTX v3.2 (§0.4.346) — **the bootstrap workflow, end-to-end** (plan
 * task 17, the adoption artifact): *transpile an expert kernel, edit
 * it, benchmark it.*
 *
 * The premise KPTX sells: you don't start from a blank DSL page — you
 * start from expert-written PTX (a vendor sample, a Godbolt paste, a
 * teammate's hand-tuned kernel), mechanically lift it into Kotlin, and
 * then make the kinds of edits that are miserable in flat text but
 * trivial in a builder: hoisting a baked constant into a parameter,
 * making a hard-coded launch shape symbolic, splicing instrumentation.
 *
 * The three steps, all executed live by this test:
 *
 * 1. **Transpile** — [expertPtx] (the §0.4.331 hand-written rms_norm,
 *    playing the "expert kernel from outside") through
 *    [KptxTranspiler.transpile]; self-verified byte-identical by
 *    construction.
 * 2. **Edit** — [editedTemplate] is the transpiler's output after the
 *    two edits the DSL makes cheap: the hard-coded `eps = 0f3727C5AC`
 *    hex constant becomes a **template argument** (any eps, still
 *    bit-exact — the float is re-encoded to its hex spelling at
 *    specialization time), and the 1024-byte smem scratch becomes
 *    `4·block` via a symbolic shape. Ten changed lines.
 * 3. **Benchmark & verify** — both kernels driver-JIT'd and launched
 *    on the GB10: the eps=1e-5 specialization must agree with the
 *    original **bit-exactly** (identical instruction stream, only
 *    provenance differs); the eps=1e-1 specialization must diverge
 *    (the edit is live, not decorative); and a timing loop prints
 *    medians for both (expect parity — same SASS).
 */
class KptxBootstrapWorkflowTest {

    private val rows = 256
    private val cols = 512

    /** The "expert kernel": §0.4.331's hand-written rms_norm forward
     * (w-variant, eps hard-coded as `0f3727C5AC` = 1e-5f). */
    private val expertPtx = """
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry rms_norm_w(
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

            mul.lo.u32 %r5, %r2, %r1;
            mul.wide.u32 %rd7, %r5, 4;
            add.s64 %rd8, %rd4, %rd7;
            add.s64 %rd9, %rd6, %rd7;

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

            ld.shared.f32 %f6, [%rd13];
            cvt.rn.f32.u32 %f7, %r1;
            div.rn.f32 %f8, %f6, %f7;
            add.f32 %f9, %f8, 0f3727C5AC;
            sqrt.rn.f32 %f10, %f9;
            rcp.rn.f32 %f11, %f10;

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

    /** Re-encode a float to its bit-exact PTX hex spelling. */
    private fun hexF32(v: Float): String =
        "0f" + Integer.toHexString(java.lang.Float.floatToRawIntBits(v)).uppercase().padStart(8, '0')

    /**
     * Step 2 — the transpiler output after the edit: eps is a template
     * arg (re-encoded bit-exactly), the smem scratch is `4·block`.
     * Everything else is the transpiled expert kernel verbatim.
     */
    private val editedTemplate = PtxKernelTemplate("rms_norm_w") { env ->
        val epsHex = hexF32(env.arg("eps").toFloat())          // ← edit 1
        val block = env.shape("block")                          // ← edit 2
        val x_ptr = param(".u64", "x_ptr")
        val w_ptr = param(".u64", "w_ptr")
        val out_ptr = param(".u64", "out_ptr")
        val n_cols = param(".u32", "n_cols")
        bank(IsaRegClass.PRED, 4)
        bank(IsaRegClass.R32, 10)
        bank(IsaRegClass.F32, 16)
        bank(IsaRegClass.R64, 21)
        val p1 = reg(IsaRegClass.PRED, 1); val p2 = reg(IsaRegClass.PRED, 2); val p3 = reg(IsaRegClass.PRED, 3)
        val r1 = reg(IsaRegClass.R32, 1); val r2 = reg(IsaRegClass.R32, 2); val r3 = reg(IsaRegClass.R32, 3)
        val r4 = reg(IsaRegClass.R32, 4); val r5 = reg(IsaRegClass.R32, 5); val r6 = reg(IsaRegClass.R32, 6)
        val r7 = reg(IsaRegClass.R32, 7); val r8 = reg(IsaRegClass.R32, 8); val r9 = reg(IsaRegClass.R32, 9)
        val f = (1..15).map { reg(IsaRegClass.F32, it) }
        val rd = (1..20).map { reg(IsaRegClass.R64, it) }
        fun fr(i: Int) = f[i - 1]
        fun rdr(i: Int) = rd[i - 1]
        val sdata = shared("sdata", sizeBytes = 4 * block)      // ← edit 2
        val LOOP_SUM = label("LOOP_SUM"); val SUM_DONE = label("SUM_DONE")
        val RED_LOOP = label("RED_LOOP"); val RED_SKIP = label("RED_SKIP"); val RED_DONE = label("RED_DONE")
        val LOOP_OUT = label("LOOP_OUT"); val OUT_DONE = label("OUT_DONE")

        inst("ld.param.u64", rdr(1), mem(x_ptr))
        inst("ld.param.u64", rdr(2), mem(w_ptr))
        inst("ld.param.u64", rdr(3), mem(out_ptr))
        inst("ld.param.u32", r1, mem(n_cols))
        inst("cvta.to.global.u64", rdr(4), rdr(1))
        inst("cvta.to.global.u64", rdr(5), rdr(2))
        inst("cvta.to.global.u64", rdr(6), rdr(3))
        blank()
        inst("mov.u32", r2, ctaidX)
        inst("mov.u32", r3, tidX)
        inst("mov.u32", r4, ntidX)
        blank()
        inst("mul.lo.u32", r5, r2, r1)
        inst("mul.wide.u32", rdr(7), r5, imm(4))
        inst("add.s64", rdr(8), rdr(4), rdr(7))
        inst("add.s64", rdr(9), rdr(6), rdr(7))
        blank()
        inst("mov.f32", fr(1), imm("0f00000000"))
        inst("mov.u32", r6, r3)
        place(LOOP_SUM)
        inst("setp.ge.u32", p1, r6, r1)
        inst("bra", SUM_DONE, guard = p1)
        inst("mul.wide.u32", rdr(10), r6, imm(4))
        inst("add.s64", rdr(11), rdr(8), rdr(10))
        inst("ld.global.f32", fr(2), mem(rdr(11)))
        inst("fma.rn.f32", fr(1), fr(2), fr(2), fr(1))
        inst("add.u32", r6, r6, r4)
        inst("bra", LOOP_SUM)
        place(SUM_DONE)
        blank()
        inst("mul.wide.u32", rdr(12), r3, imm(4))
        inst("mov.u64", rdr(13), sdata)
        inst("add.s64", rdr(14), rdr(13), rdr(12))
        inst("st.shared.f32", mem(rdr(14)), fr(1))
        inst("bar.sync", imm(0))
        blank()
        inst("shr.u32", r7, r4, imm(1))
        place(RED_LOOP)
        inst("setp.eq.u32", p2, r7, imm(0))
        inst("bra", RED_DONE, guard = p2)
        inst("setp.ge.u32", p3, r3, r7)
        inst("bra", RED_SKIP, guard = p3)
        inst("add.u32", r8, r3, r7)
        inst("mul.wide.u32", rdr(15), r8, imm(4))
        inst("add.s64", rdr(16), rdr(13), rdr(15))
        inst("ld.shared.f32", fr(3), mem(rdr(16)))
        inst("ld.shared.f32", fr(4), mem(rdr(14)))
        inst("add.f32", fr(5), fr(3), fr(4))
        inst("st.shared.f32", mem(rdr(14)), fr(5))
        place(RED_SKIP)
        inst("bar.sync", imm(0))
        inst("shr.u32", r7, r7, imm(1))
        inst("bra", RED_LOOP)
        place(RED_DONE)
        blank()
        inst("ld.shared.f32", fr(6), mem(rdr(13)))
        inst("cvt.rn.f32.u32", fr(7), r1)
        inst("div.rn.f32", fr(8), fr(6), fr(7))
        inst("add.f32", fr(9), fr(8), imm(epsHex))              // ← edit 1
        inst("sqrt.rn.f32", fr(10), fr(9))
        inst("rcp.rn.f32", fr(11), fr(10))
        blank()
        inst("mov.u32", r9, r3)
        place(LOOP_OUT)
        inst("setp.ge.u32", p1, r9, r1)
        inst("bra", OUT_DONE, guard = p1)
        inst("mul.wide.u32", rdr(17), r9, imm(4))
        inst("add.s64", rdr(18), rdr(8), rdr(17))
        inst("ld.global.f32", fr(12), mem(rdr(18)))
        inst("add.s64", rdr(19), rdr(5), rdr(17))
        inst("ld.global.f32", fr(13), mem(rdr(19)))
        inst("mul.f32", fr(14), fr(12), fr(11))
        inst("mul.f32", fr(15), fr(14), fr(13))
        inst("add.s64", rdr(20), rdr(9), rdr(17))
        inst("st.global.f32", mem(rdr(20)), fr(15))
        inst("add.u32", r9, r9, r4)
        inst("bra", LOOP_OUT)
        place(OUT_DONE)
        inst("ret")
    }

    private fun cudaAvailable(): Boolean = runCatching {
        val p = ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start()
        if (!p.waitFor(5, TimeUnit.SECONDS)) {
            p.destroyForcibly(); false
        } else {
            p.exitValue() == 0 && p.inputStream.bufferedReader().readText().isNotBlank()
        }
    }.getOrElse { false }

    @Test
    fun bootstrapWorkflowTranspileEditBenchmark() {
        // Step 1 — transpile: mechanical, self-verified byte-identical.
        val transpiled = KptxTranspiler.transpile(parsePtx(expertPtx))
        assertTrue(transpiled.contains("imm(\"0f3727C5AC\")"), "the constant the edit will hoist")
        assertTrue(transpiled.contains("val sdata = shared(\"sdata\", sizeBytes = 1024)"))

        assumeTrue(cudaAvailable(), "no NVIDIA GPU/driver — skipping GPU half.")

        val eps5 = editedTemplate.specialize(shapes = mapOf("block" to 256), args = mapOf("eps" to "1e-5"))
        val eps1 = editedTemplate.specialize(shapes = mapOf("block" to 256), args = mapOf("eps" to "0.1"))

        Arena.ofShared().use { arena ->
            val cuda = try {
                CudaDriverFfm.load(arena)
            } catch (e: CudaDriverException) {
                assumeTrue(false, "CUDA driver unusable (${e.message}) — skipping.")
                return
            }
            val device = cuda.deviceGet(0)
            cuda.primaryCtxRetainAndSetCurrent(device)
            try {
                val rng = java.util.Random(7)
                val x = FloatArray(rows * cols) { rng.nextFloat() * 2f - 1f }
                val w = FloatArray(cols) { rng.nextFloat() + 0.5f }

                fun run(ptx: String, entry: String): Pair<FloatArray, Long> {
                    val module = cuda.moduleLoadPtx(ptx)
                    try {
                        val function = cuda.moduleGetFunction(module, entry)
                        val xB = rows * cols * 4L
                        val wB = cols * 4L
                        val host = arena.allocate(xB)
                        for (i in x.indices) host.set(JAVA_FLOAT, i * 4L, x[i])
                        val hostW = arena.allocate(wB)
                        for (i in w.indices) hostW.set(JAVA_FLOAT, i * 4L, w[i])
                        val dX = cuda.memAlloc(xB); val dW = cuda.memAlloc(wB); val dOut = cuda.memAlloc(xB)
                        try {
                            cuda.memcpyHtoD(dX, host, xB)
                            cuda.memcpyHtoD(dW, hostW, wB)
                            val pX = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dX) }
                            val pW = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dW) }
                            val pO = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dOut) }
                            val pN = arena.allocate(JAVA_INT).also { it.set(JAVA_INT, 0L, cols) }
                            val params = arena.allocate(4 * 8L)
                            params.set(ADDRESS, 0L, pX); params.set(ADDRESS, 8L, pW)
                            params.set(ADDRESS, 16L, pO); params.set(ADDRESS, 24L, pN)

                            fun launchOnce() {
                                cuda.launchKernel(
                                    function, rows, 1, 1, 256, 1, 1,
                                    sharedMemBytes = 0, stream = MemorySegment.NULL, kernelParams = params,
                                )
                            }
                            launchOnce(); cuda.ctxSynchronize() // warm (JIT + caches)
                            val t0 = System.nanoTime()
                            repeat(100) { launchOnce() }
                            cuda.ctxSynchronize()
                            val perLaunchNs = (System.nanoTime() - t0) / 100

                            val out = arena.allocate(xB)
                            cuda.memcpyDtoH(out, dOut, xB)
                            return FloatArray(rows * cols) { out.get(JAVA_FLOAT, it * 4L) } to perLaunchNs
                        } finally {
                            cuda.memFree(dX); cuda.memFree(dW); cuda.memFree(dOut)
                        }
                    } finally {
                        cuda.moduleUnload(module)
                    }
                }

                // Step 3 — verify + benchmark.
                val (expertOut, expertNs) = run(expertPtx, "rms_norm_w")
                val (editedOut, editedNs) = run(eps5.emitPtx(), "rms_norm_w")
                val (eps1Out, _) = run(eps1.emitPtx(), "rms_norm_w")

                // eps=1e-5 specialization: identical instruction stream → bit-exact.
                for (i in expertOut.indices) {
                    assertEquals(expertOut[i], editedOut[i], "element $i must be bit-exact")
                }
                // eps=0.1 specialization: the edit is live.
                val maxDelta = expertOut.indices.maxOf { abs(expertOut[it] - eps1Out[it]) }
                assertTrue(maxDelta > 1e-3f, "eps edit had no numerical effect (max delta $maxDelta)")

                println(
                    "[kptx-bootstrap] transpile→edit→benchmark: expert=${expertNs / 1_000} us/launch, " +
                        "edited(eps=1e-5)=${editedNs / 1_000} us/launch (bit-exact vs expert), " +
                        "edited(eps=0.1) diverges as intended (max delta $maxDelta)",
                )
            } finally {
                cuda.primaryCtxRelease(device)
            }
        }
    }
}
