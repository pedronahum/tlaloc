package io.tlaloc.ir.inference

import io.tlaloc.core.f4e2m1ToFloat
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.core.floatToF8e4m3fn

/**
 * Rounds a weight `[rows, cols]` to NVFP4 as NVIDIA's ModelOpt stores a
 * weight it quantizes without calibration data:
 * ```
 *   scale2   = max |w| / (6 * 448)                        (1 when w is all zero)
 *   scale[g] = e4m3fn(max |w in group g| / 6 / scale2)    groups of 16 along a row
 *   code     = the nearest e2m1 to w / (e4m3(scale[g]) * scale2)   (0 where the scale is 0)
 * ```
 * Codes are two per byte, the even column in the low nibble; ties go to the
 * code with an even mantissa.
 */
object Nvfp4Quantizer {

    class Quantized(val codes: ByteArray, val scales: ByteArray, val scale2: Float)

    fun quantize(w: FloatArray, rows: Int, cols: Int): Quantized = quantize(w, rows, cols, 16, null)

    /**
     * As [quantize], with groups of [group] values along a row and, when [scale2] is given,
     * that tensor scale in place of `max |w| / (6 * 448)` (a bound known without a pass over
     * the tensor: an FP8 tensor of scale s has `max |w| <= 448 s`, so `s / 6`).
     */
    fun quantize(w: FloatArray, rows: Int, cols: Int, group: Int, scale2: Float?): Quantized {
        require(group >= 2 && group % 2 == 0 && cols % group == 0) { "Nvfp4Quantizer: $cols columns are not groups of $group" }
        require(w.size.toLong() == rows.toLong() * cols) { "Nvfp4Quantizer: ${w.size} values for [$rows, $cols]" }
        val s2 = scale2 ?: run {
            var amax = 0f
            for (v in w) amax = maxOf(amax, kotlin.math.abs(v))
            if (amax == 0f) 1f else amax / (6f * io.tlaloc.core.F8E4M3FN_MAX)
        }
        require(s2 > 0f && s2.isFinite()) { "Nvfp4Quantizer: scale2 must be positive, got $s2" }
        val groups = cols / group
        val codes = ByteArray(rows * cols / 2)
        val scales = ByteArray(rows * groups)
        for (r in 0 until rows) for (g in 0 until groups) {
            val base = r * cols + g * group
            var gmax = 0f
            for (i in 0 until group) gmax = maxOf(gmax, kotlin.math.abs(w[base + i]))
            val sb = floatToF8e4m3fn((gmax / 6f / s2).coerceAtMost(io.tlaloc.core.F8E4M3FN_MAX))
            scales[r * groups + g] = sb
            val step = f8e4m3fnToFloat(sb) * s2
            for (i in 0 until group) {
                val code = if (step == 0f) 0 else nearest(w[base + i] / step)
                val at = (base + i) / 2
                codes[at] = if (i % 2 == 0) {
                    (codes[at].toInt() and 0xF0 or code).toByte()
                } else {
                    (codes[at].toInt() and 0x0F or (code shl 4)).toByte()
                }
            }
        }
        return Quantized(codes, scales, s2)
    }

    /** The e2m1 code nearest [v] (clamped to ±6). */
    fun nearest(v: Float): Int {
        val a = kotlin.math.abs(v)
        var best = 0
        for (m in 1 until 8) {
            val d = kotlin.math.abs(f4e2m1ToFloat(m) - a)
            val db = kotlin.math.abs(f4e2m1ToFloat(best) - a)
            if (d < db || (d == db && m % 2 == 0)) best = m
        }
        return if (v < 0f && best != 0) best or 8 else best
    }
}
