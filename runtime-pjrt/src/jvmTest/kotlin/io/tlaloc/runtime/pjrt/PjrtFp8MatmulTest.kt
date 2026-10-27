package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.testing.Tolerances
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test

/**
 * FP8_MATMUL through XLA:GPU against the interpreter: f8 codes in, f32 and bf16 out. A CUDA
 * session compiles a program with an FP8 dot with cuBLAS GEMMs
 * ([io.tlaloc.runtime.pjrt.ffm.CompileOptions.withXlaGpuFixes]): the Triton GEMM XLA picks for
 * this product returns twice it on the GB10.
 */
class PjrtFp8MatmulTest {

    @Test
    fun theFp8ProductMatchesTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val m = 128
        val k = 256
        val n = 192
        val r = Random(6)
        // Finite e4m3fn codes (0x7F and 0xFF are NaN), small enough that sums stay well in range.
        fun codes(size: Int) = ByteArray(size) { (r.nextInt(0, 0x60) or if (r.nextBoolean()) 0x80 else 0).toByte() }
        val a = codes(m * k)
        val b = codes(k * n)
        val sa = FloatArray(m) { 0.01f + r.nextFloat() }
        val sb = FloatArray(n) { 0.01f + r.nextFloat() }
        TestBackend.session().use { s ->
            for ((out, rowwise) in listOf(F32 to true, BF16 to true, F32 to false)) {
                val fn = DxirBuilder.function("fp8_mm") {
                    val pa = param("a", DxirType(F8E4M3FN, listOf(m, k)))
                    val pb = param("b", DxirType(F8E4M3FN, listOf(k, n)))
                    val psa = param("sa", DxirType(F32, if (rowwise) listOf(m) else emptyList()))
                    val psb = param("sb", DxirType(F32, if (rowwise) listOf(n) else emptyList()))
                    val y = op(OpKind.FP8_MATMUL, listOf(pa, pb, psa, psb), DxirType(out, listOf(m, n)))
                    listOf(if (out == F32) y else op(OpKind.CAST, listOf(y), DxirType(F32, listOf(m, n))))
                }
                val scaleA = if (rowwise) sa else floatArrayOf(sa[0])
                val scaleB = if (rowwise) sb else floatArrayOf(sb[0])
                val want = DxirInterpreter.evalFunction(
                    fn, listOf(FloatArray(a.size) { f8e4m3fnToFloat(a[it]) }, FloatArray(b.size) { f8e4m3fnToFloat(b[it]) }, scaleA, scaleB),
                )[0]
                val got = s.runOnHost(fn, listOf(a, b, scaleA, scaleB))[0] as FloatArray
                Tolerances.assertClose(DoubleArray(got.size) { got[it].toDouble() }, DoubleArray(want.size) { want[it].toDouble() }, if (out == F32) 1e-5 else 1e-2, "fp8 -> $out, rowwise $rowwise")
            }
        }
    }
}
