package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.U8
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.Nvfp4MatmulAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * NVFP4_MATMUL as the CUDA kernel `tlaloc_fp4_gemm` (libtlaloc_kernels.so,
 * triton/build_backend.sh) and as XLA's unpack-and-dot form, against the
 * interpreter: partial tiles, several K splits, 1 to 16 rows. TLALOC_FP4_BENCH=1
 * also times both forms and an FP8 projection at Qwen3.8-27B's MLP shapes.
 */
class PjrtNvfp4MatmulTest {

    private val library = Path.of("..", "triton", "backends", "tlaloc", "libtlaloc_kernels.so").toAbsolutePath().normalize()

    private fun registered(): Boolean {
        if (!TestBackend.deviceAvailable || TestBackend.isTpu || !Files.isRegularFile(library)) return false
        PjrtFfiRegistry.registerNativeLibrary(PjrtBinaries.requireCudaPlugin("PjrtNvfp4MatmulTest"), library)
        return true
    }

    private fun fn(m: Int, n: Int, k: Int, fused: Boolean): DxirFunction = DxirBuilder.function("fp4") {
        val t = (n + 15) / 16
        val x = param("x", DxirType(F32, listOf(m, k)))
        val c = param("codes", DxirType(U8, listOf(t, k / 64, 512)))
        val s = param("scales", DxirType(U8, listOf(t, k / 64, 64)))
        val s2 = param("scale2", DxirType(F32, listOf(n)))
        val attrs: Map<String, Any> = if (fused) mapOf(Nvfp4MatmulAttrs.FUSED_KERNEL to true) else emptyMap()
        listOf(op(OpKind.NVFP4_MATMUL, listOf(x, c, s, s2), DxirType(F32, listOf(m, n)), attrs))
    }

    private class Weight(val codes: ByteArray, val scales: ByteArray, val scale2: FloatArray)

    /** A random NVFP4 weight [n, k] in the checkpoint's layout, packed. */
    private fun weight(n: Int, k: Int, rnd: Random): Weight {
        val codes = ByteArray(n * k / 2) { rnd.nextInt(256).toByte() }
        val scales = ByteArray(n * k / 16) { (0x28 + rnd.nextInt(0x20)).toByte() }
        val (pc, ps) = Nvfp4MatmulAttrs.pack(codes, scales, n, k)
        return Weight(pc, ps, FloatArray(n) { if (it < n / 2) 0.0123f else 0.0456f })
    }

    @Test
    fun theKernelAndXlasFormMultiplyAsTheInterpreter() {
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so (triton/build_backend.sh)")
        val rnd = Random(5)
        for ((m, n, k) in listOf(Triple(1, 40, 128), Triple(4, 72, 4160), Triple(16, 33, 2112))) {
            val w = weight(n, k, rnd)
            val x = FloatArray(m * k) { rnd.nextFloat() * 2 - 1 }
            val want = DxirInterpreter.evalFunction(
                fn(m, n, k, false),
                listOf(x, FloatArray(w.codes.size) { w.codes[it].toInt().and(0xFF).toFloat() },
                    FloatArray(w.scales.size) { w.scales[it].toInt().and(0xFF).toFloat() }, w.scale2),
            )[0]
            assertTrue("tlaloc_fp4_gemm" in fn(m, n, k, true).toStablehlo(), "the fused form is emitted")
            TestBackend.session().use { s ->
                val ins = listOf<Any>(x, w.codes, w.scales, w.scale2)
                val fused = s.runOnHost(fn(m, n, k, true), ins).single() as FloatArray
                val xla = s.runOnHost(fn(m, n, k, false), ins).single() as FloatArray
                var mag = 0f
                for (v in want) mag = maxOf(mag, abs(v))
                var dk = 0f
                var dx = 0f
                for (i in want.indices) {
                    dk = maxOf(dk, abs(fused[i] - want[i]))
                    dx = maxOf(dx, abs(xla[i] - want[i]))
                }
                println("[nvfp4] M=$m N=$n K=$k: worst |d| against the interpreter: kernel $dk, XLA's form $dx (largest |y| $mag)")
                assertTrue(dk <= 1e-5f * mag && dx <= 1e-5f * mag, "NVFP4_MATMUL differs: $dk / $dx of $mag")
            }
        }
    }

    @Test
    fun timings() {
        assumeTrue(System.getenv("TLALOC_FP4_BENCH") == "1", "set TLALOC_FP4_BENCH=1")
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so")
        val rnd = Random(9)
        TestBackend.session().use { s ->
            for ((label, n, k) in listOf(Triple("gate+up", 34816, 5120), Triple("down", 5120, 17408))) {
                val w = weight(n, k, rnd)
                val t = (n + 15) / 16
                val codes = s.bufferFromHostU8(w.codes, listOf(t, k / 64, 512))
                val scales = s.bufferFromHostU8(w.scales, listOf(t, k / 64, 64))
                val s2 = s.bufferFromHostF32(w.scale2, listOf(n))
                val fp8 = s.bufferFromHostBytes(ByteArray(k * n) { rnd.nextInt(0x70).toByte() }, listOf(k, n), F8E4M3FN)
                val fp8Scale = s.bufferFromHostF32(FloatArray(n) { 0.01f }, listOf(n))
                for (m in listOf(4, 16)) {
                    val x = s.bufferFromHostF32(FloatArray(m * k) { rnd.nextFloat() - 0.5f }, listOf(m, k))
                    fun time(f: DxirFunction, ins: List<PjrtBuffer>): Double {
                        repeat(3) { s.executeOn(f, ins).forEach { it.close() } }
                        val t0 = System.nanoTime()
                        repeat(20) { s.executeOn(f, ins).forEach { it.close() } }
                        return (System.nanoTime() - t0) / 1e6 / 20
                    }
                    val kernel = time(fn(m, n, k, true), listOf(x, codes, scales, s2))
                    val xla = time(fn(m, n, k, false), listOf(x, codes, scales, s2))
                    // The FP8 projection the graphs emit today: bf16(x) @ bf16(codes), times the column scales.
                    val f8 = DxirBuilder.function("fp8") {
                        val xp = param("x", DxirType(F32, listOf(m, k)))
                        val wp = param("w", DxirType(F8E4M3FN, listOf(k, n)))
                        val sp = param("s", DxirType(F32, listOf(n)))
                        val y = op(
                            OpKind.MATMUL,
                            listOf(op(OpKind.CAST, listOf(xp), DxirType(BF16, listOf(m, k))), op(OpKind.CAST, listOf(wp), DxirType(BF16, listOf(k, n)))),
                            DxirType(F32, listOf(m, n)),
                        )
                        val sb = op(OpKind.BROADCAST, listOf(op(OpKind.RESHAPE, listOf(sp), DxirType(F32, listOf(1, n)))), y.type, mapOf("broadcast_dimensions" to listOf(0, 1)))
                        listOf(op(OpKind.MUL, listOf(y, sb), y.type))
                    }
                    val fp8Ms = time(f8, listOf(x, fp8, fp8Scale))
                    println("[nvfp4-bench] 27B $label, $m rows: kernel %.3f ms, XLA's NVFP4 form %.3f ms, FP8 %.3f ms".format(kernel, xla, fp8Ms))
                    x.close()
                }
                listOf(codes, scales, s2, fp8, fp8Scale).forEach { it.close() }
            }
        }
    }
}
