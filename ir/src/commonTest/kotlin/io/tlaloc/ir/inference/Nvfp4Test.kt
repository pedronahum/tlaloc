package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.U8
import io.tlaloc.core.f4e2m1ToFloat
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.Nvfp4MatmulAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Nvfp4Test {

    private fun dequant(q: Nvfp4Quantizer.Quantized, rows: Int, cols: Int) = FloatArray(rows * cols) { i ->
        val b = q.codes[i / 2].toInt()
        val code = if (i % 2 == 0) b and 0xF else (b ushr 4) and 0xF
        f4e2m1ToFloat(code) * f8e4m3fnToFloat(q.scales[(i / cols) * (cols / 16) + (i % cols) / 16]) * q.scale2
    }

    @Test
    fun roundingKeepsEachValueWithinHalfAStepOfItsGroup() {
        val rnd = Random(3)
        val rows = 5
        val cols = 64
        val w = FloatArray(rows * cols) { (rnd.nextFloat() - 0.5f) * if (it % 16 == 0) 4f else 1f }
        val q = Nvfp4Quantizer.quantize(w, rows, cols)
        val d = dequant(q, rows, cols)
        for (g in 0 until rows * cols / 16) {
            var gmax = 0f
            for (i in 0 until 16) gmax = maxOf(gmax, abs(w[16 * g + i]))
            // The largest gap between e2m1 values is 2 (4 to 6), a third of the group's max.
            for (i in 0 until 16) assertTrue(abs(d[16 * g + i] - w[16 * g + i]) <= gmax / 6f * 1.07f + 1e-6f, "value ${16 * g + i}")
        }
        assertEquals(0, Nvfp4Quantizer.nearest(0.2f))
        assertEquals(1, Nvfp4Quantizer.nearest(0.25f + 1e-3f))
        assertEquals(2, Nvfp4Quantizer.nearest(0.75f), "a tie goes to the even mantissa")
        assertEquals(15, Nvfp4Quantizer.nearest(-9f))
    }

    @Test
    fun thePackedLayoutReadsBackTheCheckpointsWeight() {
        val rnd = Random(4)
        val rows = 40
        val cols = 192
        val q = Nvfp4Quantizer.quantize(FloatArray(rows * cols) { rnd.nextFloat() - 0.5f }, rows, cols)
        val (codes, scales) = Nvfp4MatmulAttrs.pack(q.codes, q.scales, rows, cols)
        assertEquals(3 * (cols / 64) * 512, codes.size)
        val cf = FloatArray(codes.size) { (codes[it].toInt() and 0xFF).toFloat() }
        val sf = FloatArray(scales.size) { (scales[it].toInt() and 0xFF).toFloat() }
        val d = dequant(q, rows, cols)
        for (n in 0 until rows) for (k in 0 until cols) {
            assertEquals(d[n * cols + k].toDouble(), Nvfp4MatmulAttrs.weight(cf, sf, n, k, cols) * q.scale2, 1e-12, "[$n, $k]")
        }
        // NVFP4_MATMUL in the interpreter: bf16(x) times that weight.
        val m = 3
        val x = FloatArray(m * cols) { rnd.nextFloat() - 0.5f }
        val fn = DxirBuilder.function("f") {
            val t = (rows + 15) / 16
            val xp = param("x", DxirType(F32, listOf(m, cols)))
            val c = param("c", DxirType(U8, listOf(t, cols / 64, 512)))
            val s = param("s", DxirType(U8, listOf(t, cols / 64, 64)))
            val s2 = param("s2", DxirType(F32, listOf(rows)))
            listOf(op(OpKind.NVFP4_MATMUL, listOf(xp, c, s, s2), DxirType(F32, listOf(m, rows))))
        }
        val y = DxirInterpreter.evalFunction(fn, listOf(x, cf, sf, FloatArray(rows) { q.scale2 }))[0]
        for (i in 0 until m) for (n in 0 until rows) {
            var want = 0.0
            for (k in 0 until cols) {
                val xb = io.tlaloc.core.bf16BitsToFloat(io.tlaloc.core.floatToBf16Bits(x[i * cols + k]))
                want += xb.toDouble() * d[n * cols + k]
            }
            assertEquals(want, y[i * rows + n].toDouble(), 1e-5 * (abs(want) + 1e-3))
        }
    }
}
