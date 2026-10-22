package io.tlaloc.ir

import io.tlaloc.core.f4e2m1ToFloat
import io.tlaloc.core.floatToF4e2m1
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The NVFP4 activation quantization of `act_quant = "nvfp4"` (moe_fp4.cu's QuantizeRowsKernel). */
class Nvfp4ActivationsTest {

    @Test
    fun e2m1RoundsToNearestWithTiesToTheEvenCode() {
        val cases = listOf(
            0f to 0, 0.25f to 0, 0.26f to 1, 0.5f to 1, 0.75f to 2, 1f to 2, 1.25f to 2, 1.3f to 3,
            1.75f to 4, 2.5f to 4, 2.6f to 5, 3.5f to 6, 5f to 6, 5.1f to 7, 6f to 7, 100f to 7,
            -0.75f to 10, -6f to 15, -0.1f to 0,
        )
        for ((q, code) in cases) assertEquals(code, floatToF4e2m1(q), "e2m1 of $q")
        for (c in 0 until 16) {
            val v = f4e2m1ToFloat(c)
            if (v != 0f) assertEquals(c, floatToF4e2m1(v), "the code of $v is $c")
        }
    }

    @Test
    fun aRowDequantizesWithinOneStepOfItsGroup() {
        val rnd = Random(3)
        val k = 256
        val row = FloatArray(k) { (rnd.nextFloat() * 2 - 1) * (1 + it / 32) }
        val out = DoubleArray(k)
        val g = Nvfp4MatmulAttrs.quantizeActivations(row, 0, k, out, 0)
        var amax = 0f
        for (v in row) amax = maxOf(amax, abs(v))
        assertEquals(amax / 2688f, g)
        for (q in 0 until k / 16) {
            var a = 0f
            for (i in 0 until 16) a = maxOf(a, abs(row[16 * q + i]))
            for (i in 0 until 16) {
                val d = abs(out[16 * q + i] * g - row[16 * q + i])
                // e2m1's widest gap is 2 (4 to 6) of 6: a value is within a third of its group's amax.
                assertTrue(d <= a / 3 + 1e-6, "value ${row[16 * q + i]} became ${out[16 * q + i] * g} (group amax $a)")
            }
        }
        val zeros = DoubleArray(16) { 1.0 }
        assertEquals(0f, Nvfp4MatmulAttrs.quantizeActivations(FloatArray(16), 0, 16, zeros, 0))
        assertTrue(zeros.all { it == 0.0 }, "a row of zeros quantizes to zeros")
    }
}
