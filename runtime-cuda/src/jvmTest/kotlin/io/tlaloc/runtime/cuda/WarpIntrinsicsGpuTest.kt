package io.tlaloc.runtime.cuda

import io.tlaloc.kptx.KShflMode
import io.tlaloc.kptx.PtxModule
import io.tlaloc.kptx.emitPtx
import io.tlaloc.kptx.imm
import io.tlaloc.kptx.mmaSyncM16N8K16F32F16
import io.tlaloc.kptx.ptxKernel
import io.tlaloc.kptx.shflSync
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
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
 * 2. [mmaKernelDriverJitLoads] — an `mma.sync.aligned.m16n8k16` tile
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
