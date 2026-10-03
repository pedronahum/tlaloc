package io.tlaloc.core

// float8 e4m3fn and float4 e2m1 on the host: the decodings (exact, every
// value is an f32) and the e4m3fn encoding (round to nearest even, as XLA's
// convert). Compute stays in f32; these formats are storage.

/** The f32 value of an e4m3fn byte: `(-1)^s 2^(e-7) (1 + m/8)`, subnormal at e = 0, NaN at 0x7F/0xFF. */
fun f8e4m3fnToFloat(bits: Byte): Float {
    val b = bits.toInt() and 0xFF
    val sign = if (b and 0x80 != 0) -1f else 1f
    val e = (b ushr 3) and 0xF
    val m = b and 0x7
    if (e == 0xF && m == 0x7) return Float.NaN
    val v = if (e == 0) m / 8f * F8_MIN_NORMAL else (1f + m / 8f) * pow2(e - 7)
    return sign * v
}

/** The largest finite e4m3fn value. */
const val F8E4M3FN_MAX: Float = 448f

private const val F8_MIN_NORMAL: Float = 0.015625f // 2^-6

private fun pow2(e: Int): Float = Float.fromBits((e + 127) shl 23)

/**
 * The e4m3fn byte nearest [v] (ties to even), as XLA's f32 -> f8e4m3fn
 * convert gives it. A finite [v] that rounds past ±448 is refused: e4m3fn
 * has no infinity, so an encoder scales or clamps first. NaN encodes as 0x7F.
 */
fun floatToF8e4m3fn(v: Float): Byte {
    if (v.isNaN()) return 0x7F
    val bits = v.toRawBits()
    val sign = (bits ushr 24) and 0x80
    val a = Float.fromBits(bits and 0x7FFFFFFF)
    val code: Int = if (a < F8_MIN_NORMAL) {
        // Subnormal: multiples of 2^-9, ties to even (8 is the smallest normal).
        val lower = kotlin.math.floor(a * 512.0).toInt()
        val frac = a * 512.0 - lower
        when {
            frac > 0.5 -> lower + 1
            frac < 0.5 -> lower
            else -> if (lower % 2 == 0) lower else lower + 1
        }
    } else {
        val b = a.toRawBits()
        var e = (b ushr 23) - 127
        var m = b and 0x7FFFFF
        // Round the 23-bit mantissa to 3 bits, ties to even.
        val lsb = (m ushr 20) and 1
        m += 0x7FFFF + lsb
        if (m >= 0x800000) {
            m -= 0x800000
            e += 1
        }
        ((e + 7) shl 3) or (m ushr 20)
    }
    require(code <= 0x7E) { "floatToF8e4m3fn: $v rounds past e4m3fn's largest value 448; scale or clamp before encoding" }
    return (sign or code).toByte()
}

/** The f32 value of an e2m1 code (the low 4 bits of [nibble]): 0, 0.5, 1, 1.5, 2, 3, 4, 6, signed by bit 3. */
fun f4e2m1ToFloat(nibble: Int): Float {
    val n = nibble and 0xF
    val mag = E2M1[n and 0x7]
    return if (n and 0x8 != 0) -mag else mag
}

private val E2M1 = floatArrayOf(0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f)

/** [v] clamped to ±448 and rounded to the nearest e4m3fn value, as an f8 KV cache stores it. */
fun saturateToF8e4m3fn(v: Float): Float = f8e4m3fnToFloat(floatToF8e4m3fn(v.coerceIn(-F8E4M3FN_MAX, F8E4M3FN_MAX)))
