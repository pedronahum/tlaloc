package io.tlaloc.runtime.cuda

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
 * KPTX v1.2 (§0.4.328) — end-to-end smoke of [CudaDriverFfm]: a
 * hand-written PTX kernel (`add_one`, f32 elementwise) driver-JIT
 * compiled via `cuModuleLoadData`, launched with `cuLaunchKernel` on a
 * self-allocated buffer, result copied back and asserted exactly
 * (x + 1.0f is representable — bit-exact expectation is safe).
 *
 * The PTX targets `.target sm_75` — the driver JITs anything ≤ the
 * device's compute capability up to actual SASS (GB10 = sm_121), which
 * is exactly the deployment story: PTX text in, no CUDA toolkit needed.
 *
 * Self-skips when no NVIDIA driver / GPU is present.
 */
class CudaDriverFfmSmokeTest {

    private val addOnePtx = """
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

    /** Mirrors PjrtBinaries.cudaAvailable / IreeBinaries.cudaAvailable —
     * `nvidia-smi -L` succeeding with output is enough evidence to attempt
     * the driver; a broken driver then fails loudly in CudaDriverFfm.load. */
    private fun cudaAvailable(): Boolean = runCatching {
        val p = ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start()
        if (!p.waitFor(5, TimeUnit.SECONDS)) {
            p.destroyForcibly(); false
        } else {
            p.exitValue() == 0 && p.inputStream.bufferedReader().readText().isNotBlank()
        }
    }.getOrElse { false }

    @Test
    fun driverJitCompilesAndLaunchesAddOneKernel() {
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
                val module = cuda.moduleLoadPtx(addOnePtx)
                try {
                    val function = cuda.moduleGetFunction(module, "add_one")

                    val n = 1024
                    val bytes = n * 4L
                    val host = arena.allocate(bytes)
                    for (i in 0 until n) host.set(JAVA_FLOAT, i * 4L, i.toFloat())

                    val dIn = cuda.memAlloc(bytes)
                    val dOut = cuda.memAlloc(bytes)
                    try {
                        cuda.memcpyHtoD(dIn, host, bytes)

                        // void** kernelParams: 3 slots, each pointing at the
                        // param's value (dptr, dptr, int).
                        val pIn = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dIn) }
                        val pOut = arena.allocate(JAVA_LONG).also { it.set(JAVA_LONG, 0L, dOut) }
                        val pN = arena.allocate(JAVA_INT).also { it.set(JAVA_INT, 0L, n) }
                        val params = arena.allocate(3 * 8L)
                        params.set(ADDRESS, 0L, pIn)
                        params.set(ADDRESS, 8L, pOut)
                        params.set(ADDRESS, 16L, pN)

                        val block = 256
                        val grid = (n + block - 1) / block
                        cuda.launchKernel(
                            function, grid, 1, 1, block, 1, 1,
                            sharedMemBytes = 0, stream = MemorySegment.NULL, kernelParams = params,
                        )
                        cuda.ctxSynchronize()

                        val out = arena.allocate(bytes)
                        cuda.memcpyDtoH(out, dOut, bytes)
                        for (i in 0 until n) {
                            assertEquals(
                                i.toFloat() + 1f, out.get(JAVA_FLOAT, i * 4L),
                                "element $i",
                            )
                        }
                        println("[kptx-cuda-smoke] add_one on ${n} f32 elements: driver-JIT + cuLaunchKernel OK")
                    } finally {
                        cuda.memFree(dIn)
                        cuda.memFree(dOut)
                    }
                } finally {
                    cuda.moduleUnload(module)
                }
            } finally {
                cuda.primaryCtxRelease(device)
            }
        }
    }
}
