package io.tlaloc.runtime.cuda

import io.tlaloc.kptx.PtxBlank
import io.tlaloc.kptx.PtxGuard
import io.tlaloc.kptx.PtxImm
import io.tlaloc.kptx.PtxInst
import io.tlaloc.kptx.PtxKernel
import io.tlaloc.kptx.PtxLabel
import io.tlaloc.kptx.PtxMem
import io.tlaloc.kptx.PtxModule
import io.tlaloc.kptx.PtxOperand
import io.tlaloc.kptx.PtxParam
import io.tlaloc.kptx.PtxReg
import io.tlaloc.kptx.PtxRegDecl
import io.tlaloc.kptx.PtxSym
import io.tlaloc.kptx.emitPtx
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
 * KPTX v2.1 (§0.4.338) — the driver is the referee: `add_one` built as
 * value-type [PtxModule] IR, emitted by [emitPtx], driver-JIT compiled
 * via `cuModuleLoadData`, launched, and asserted bit-exact on 1024 f32
 * elements. The [io.tlaloc.kptx.PtxEmitterTest] pins the *formatting*;
 * this pins that the emitted text is *valid PTX to the real driver* —
 * the two halves of task 9's DoD seed (a Kotlin-constructed kernel
 * executing on the GPU; the DSL surface over this IR is task 12).
 *
 * Mirrors [CudaDriverFfmSmokeTest]'s harness (same kernel semantics,
 * same launch shape) — by design: if both pass, IR-emitted and
 * hand-written PTX are interchangeable at the driver boundary.
 */
class PtxIrDriverJitTest {

    private fun i(op: String, vararg ops: PtxOperand) = PtxInst(op, ops.toList())
    private fun r(name: String) = PtxReg(name)
    private fun m(base: String) = PtxMem(base)

    private fun addOneIr(): PtxModule = PtxModule(
        kernels = listOf(
            PtxKernel(
                name = "add_one",
                params = listOf(
                    PtxParam(".u64", "in_ptr"),
                    PtxParam(".u64", "out_ptr"),
                    PtxParam(".u32", "n"),
                ),
                body = listOf(
                    PtxRegDecl(".pred", "%p", 2),
                    PtxRegDecl(".b32", "%r", 6),
                    PtxRegDecl(".f32", "%f", 3),
                    PtxRegDecl(".b64", "%rd", 8),
                    PtxBlank,
                    i("ld.param.u64", r("%rd1"), m("in_ptr")),
                    i("ld.param.u64", r("%rd2"), m("out_ptr")),
                    i("ld.param.u32", r("%r1"), m("n")),
                    i("cvta.to.global.u64", r("%rd3"), r("%rd1")),
                    i("cvta.to.global.u64", r("%rd4"), r("%rd2")),
                    i("mov.u32", r("%r2"), r("%ctaid.x")),
                    i("mov.u32", r("%r3"), r("%ntid.x")),
                    i("mov.u32", r("%r4"), r("%tid.x")),
                    i("mad.lo.s32", r("%r5"), r("%r2"), r("%r3"), r("%r4")),
                    i("setp.ge.s32", r("%p1"), r("%r5"), r("%r1")),
                    PtxInst("bra", listOf(PtxSym("DONE")), guard = PtxGuard("%p1")),
                    i("mul.wide.s32", r("%rd5"), r("%r5"), PtxImm("4")),
                    i("add.s64", r("%rd6"), r("%rd3"), r("%rd5")),
                    i("ld.global.f32", r("%f1"), m("%rd6")),
                    i("add.f32", r("%f2"), r("%f1"), PtxImm("0f3F800000")),
                    i("add.s64", r("%rd7"), r("%rd4"), r("%rd5")),
                    i("st.global.f32", m("%rd7"), r("%f2")),
                    PtxLabel("DONE"),
                    i("ret"),
                ),
            ),
        ),
    )

    private fun cudaAvailable(): Boolean = runCatching {
        val p = ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start()
        if (!p.waitFor(5, TimeUnit.SECONDS)) {
            p.destroyForcibly(); false
        } else {
            p.exitValue() == 0 && p.inputStream.bufferedReader().readText().isNotBlank()
        }
    }.getOrElse { false }

    @Test
    fun irEmittedPtxDriverJitsAndComputes() {
        assumeTrue(cudaAvailable(), "no NVIDIA GPU/driver — skipping.")

        val ptxText = addOneIr().emitPtx()

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
                val module = cuda.moduleLoadPtx(ptxText)
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
                        println("[kptx-ir-jit] IR-emitted add_one: driver-JIT + cuLaunchKernel OK, 1024 f32 bit-exact")
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
