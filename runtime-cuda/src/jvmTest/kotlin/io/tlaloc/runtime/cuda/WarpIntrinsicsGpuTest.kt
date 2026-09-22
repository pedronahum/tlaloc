package io.tlaloc.runtime.cuda

import io.tlaloc.kptx.KShflMode
import io.tlaloc.kptx.PtxModule
import io.tlaloc.kptx.emitPtx
import io.tlaloc.kptx.imm
import io.tlaloc.kptx.mmaSyncM16N8K16F32F16
import io.tlaloc.kptx.ptxKernel
import io.tlaloc.kptx.shflSync
import io.tlaloc.kptx.warpReduceSumF32
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * KPTX v2.6 (§0.4.343) — GPU-side proof for the warp-intrinsic
 * surface, DSL-authored end-to-end:
 *
 * 1. [shflWarpSumComputesCorrectly] — a `shfl.sync.down` butterfly
 *    warp reduction written with the typed wrapper, driver-JIT
 *    compiled and **numerically verified**: 32 lanes summing
 *    `in[i] = i` land 496 in `out[0]`.
 * 2. [shflF32WarpSumComputesCorrectly] — §0.4.493: the same reduction
 *    over **f32**, written as one [warpReduceSumF32] call. 32 lanes
 *    holding 0f..31f land exactly 496.0f, and the emitted PTX is
 *    asserted to contain no `mov.b32` — the "shfl needs %r, the
 *    accumulator is %f" blocker named in docs/KPTX_PAGED_PERF.md §7.4
 *    was a wrapper `require`, not the ISA.
 * 3. [mmaKernelDriverJitLoads] — an `mma.sync.aligned.m16n8k16` tile
 *    written with the typed wrapper passes `cuModuleLoadData` +
 *    `cuModuleGetFunction` (the driver validates the full fragment
 *    encoding at JIT; `.target sm_80` — the m16n8k16 floor — JITs to
 *    GB10 SASS). Numerical mma validation needs the per-lane fragment
 *    layout harness and lands with the first real tensor-core kernel.
 */
class WarpIntrinsicsGpuTest {

    private fun warpSumModule(): PtxModule = ptxKernel("warp_sum") {
        val inPtr = param(".u64", "in_ptr")
        val outPtr = param(".u64", "out_ptr")
        val p1 = pred()
        val rTid = r32(); val rVal = r32(); val rTmp = r32()
        val rdIn = r64(); val rdOut = r64(); val rdInG = r64(); val rdOutG = r64(); val rdAddr = r64()

        inst("ld.param.u64", rdIn, mem(inPtr))
        inst("ld.param.u64", rdOut, mem(outPtr))
        inst("cvta.to.global.u64", rdInG, rdIn)
        inst("cvta.to.global.u64", rdOutG, rdOut)
        inst("mov.u32", rTid, tidX)
        inst("mul.wide.u32", rdAddr, rTid, imm(4))
        inst("add.s64", rdAddr, rdInG, rdAddr)
        inst("ld.global.u32", rVal, mem(rdAddr))
        for (off in listOf(16, 8, 4, 2, 1)) {
            shflSync(KShflMode.DOWN, d = rTmp, a = rVal, b = imm(off), c = imm("0x1f"))
            inst("add.u32", rVal, rVal, rTmp)
        }
        val done = label("DONE")
        inst("setp.ne.u32", p1, rTid, imm(0))
        inst("bra", done, guard = p1)
        inst("st.global.u32", mem(rdOutG), rVal)
        place(done)
        inst("ret")
    }

    /**
     * §0.4.493 — the same reduction over **floats**, written with
     * [warpReduceSumF32]. The whole reduction is one DSL call and the
     * emitted PTX contains no `mov.b32` round trip: `shfl.sync.down.b32`
     * carries `%f` registers directly.
     */
    private fun warpSumF32Module(): PtxModule = ptxKernel("warp_sum_f32") {
        val inPtr = param(".u64", "in_ptr")
        val outPtr = param(".u64", "out_ptr")
        val p1 = pred()
        val rTid = r32()
        val fVal = f32()
        val rdIn = r64(); val rdOut = r64(); val rdInG = r64(); val rdOutG = r64(); val rdAddr = r64()

        inst("ld.param.u64", rdIn, mem(inPtr))
        inst("ld.param.u64", rdOut, mem(outPtr))
        inst("cvta.to.global.u64", rdInG, rdIn)
        inst("cvta.to.global.u64", rdOutG, rdOut)
        inst("mov.u32", rTid, tidX)
        inst("mul.wide.u32", rdAddr, rTid, imm(4))
        inst("add.s64", rdAddr, rdInG, rdAddr)
        inst("ld.global.f32", fVal, mem(rdAddr))
        warpReduceSumF32(fVal)
        val done = label("DONE")
        inst("setp.ne.u32", p1, rTid, imm(0))
        inst("bra", done, guard = p1)
        inst("st.global.f32", mem(rdOutG), fVal)
        place(done)
        inst("ret")
    }

    private fun mmaProbeModule(): PtxModule = ptxKernel("mma_probe", target = "sm_80") {
        val outPtr = param(".u64", "out_ptr")
        val f = List(4) { f32() }
        val r = List(6) { r32() }
        val rdOut = r64(); val rdOutG = r64()

        inst("ld.param.u64", rdOut, mem(outPtr))
        inst("cvta.to.global.u64", rdOutG, rdOut)
        for (reg in f) inst("mov.f32", reg, imm("0f00000000"))
        for (reg in r) inst("mov.u32", reg, imm(0))
        mmaSyncM16N8K16F32F16(d = f, a = r.take(4), b = r.drop(4), c = f)
        inst("st.global.f32", mem(rdOutG), f[0])
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

    private fun withCuda(block: (CudaDriverFfm, Arena) -> Unit) {
        assumeTrue(cudaAvailable(), "no NVIDIA GPU/driver — skipping.")
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
                block(cuda, arena)
            } finally {
                cuda.primaryCtxRelease(device)
            }
        }
    }

    @Test
    fun shflWarpSumComputesCorrectly() = withCuda { cuda, arena ->
        val module = cuda.moduleLoadPtx(warpSumModule().emitPtx())
        try {
            val function = cuda.moduleGetFunction(module, "warp_sum")

            val n = 32
            val bytes = n * 4L
            val host = arena.allocate(bytes)
            for (i in 0 until n) host.set(JAVA_INT, i * 4L, i)

            val dIn = cuda.memAlloc(bytes)
            val dOut = cuda.memAlloc(4L)
            try {
                cuda.memcpyHtoD(dIn, host, bytes)

                val pIn = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dIn) }
                val pOut = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dOut) }
                val params = arena.allocate(2 * 8L)
                params.set(ADDRESS, 0L, pIn)
                params.set(ADDRESS, 8L, pOut)

                cuda.launchKernel(
                    function, 1, 1, 1, 32, 1, 1,
                    sharedMemBytes = 0, stream = MemorySegment.NULL, kernelParams = params,
                )
                cuda.ctxSynchronize()

                val out = arena.allocate(4L)
                cuda.memcpyDtoH(out, dOut, 4L)
                assertEquals((0 until 32).sum(), out.get(JAVA_INT, 0L), "warp sum of 0..31")
                println("[kptx-warp] shfl.sync.down warp reduction: 32 lanes → ${out.get(JAVA_INT, 0L)} (exact)")
            } finally {
                cuda.memFree(dIn)
                cuda.memFree(dOut)
            }
        } finally {
            cuda.moduleUnload(module)
        }
    }

    @Test
    fun shflF32WarpSumComputesCorrectly() = withCuda { cuda, arena ->
        val ptx = warpSumF32Module().emitPtx()
        // The claim under test, asserted on the text before the driver
        // sees it: no bit-reinterpretation round trip.
        assertEquals(false, "mov.b32" in ptx, "float warp reduction needs no mov.b32:\n$ptx")
        val module = cuda.moduleLoadPtx(ptx)
        try {
            val function = cuda.moduleGetFunction(module, "warp_sum_f32")

            val n = 32
            val bytes = n * 4L
            val host = arena.allocate(bytes)
            // 0f..31f — integral, so the f32 sum (496) is EXACT and the
            // assertion can be an equality, not a tolerance.
            for (i in 0 until n) host.set(JAVA_FLOAT, i * 4L, i.toFloat())

            val dIn = cuda.memAlloc(bytes)
            val dOut = cuda.memAlloc(4L)
            try {
                cuda.memcpyHtoD(dIn, host, bytes)

                val pIn = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dIn) }
                val pOut = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dOut) }
                val params = arena.allocate(2 * 8L)
                params.set(ADDRESS, 0L, pIn)
                params.set(ADDRESS, 8L, pOut)

                cuda.launchKernel(
                    function, 1, 1, 1, 32, 1, 1,
                    sharedMemBytes = 0, stream = MemorySegment.NULL, kernelParams = params,
                )
                cuda.ctxSynchronize()

                val out = arena.allocate(4L)
                cuda.memcpyDtoH(out, dOut, 4L)
                val got = out.get(JAVA_FLOAT, 0L)
                assertEquals(496.0f, got, "f32 warp sum of 0..31")
                println("[kptx-warp] shfl.sync.down.b32 over %f registers: 32 lanes → $got (exact, no mov.b32)")
            } finally {
                cuda.memFree(dIn)
                cuda.memFree(dOut)
            }
        } finally {
            cuda.moduleUnload(module)
        }
    }

    @Test
    fun mmaKernelDriverJitLoads() = withCuda { cuda, _ ->
        val module = cuda.moduleLoadPtx(mmaProbeModule().emitPtx())
        try {
            cuda.moduleGetFunction(module, "mma_probe")
            println("[kptx-warp] mma.sync.aligned.m16n8k16 kernel: driver-JIT accepts the DSL-emitted fragment encoding")
        } finally {
            cuda.moduleUnload(module)
        }
    }
}
