package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.455 (Phase G1a) — the bf16 foundation pins. Every conversion assert
 * is BIT-LEVEL (raw patterns via [Float.toRawBits] / hand-derived Shorts),
 * per the house bit-pin rule: rounding conventions are contracts, and a
 * contract you pin with a tolerance is not pinned. The RNE cases were
 * derived by hand from the u32 bias-and-carry arithmetic (see Bf16.kt) —
 * they are the oracle, not a re-implementation.
 */
class Bf16Test {

    private fun bits(f: Float): Short = floatToBf16Bits(f)

    @Test
    fun dtypeSurface() {
        assertEquals(2, BF16.sizeBytes)
        assertEquals("bf16", BF16.name)
    }

    /** Values whose f32 low 16 bits are zero narrow losslessly and round-trip. */
    @Test
    fun exactValuesRoundTrip() {
        val exact = floatArrayOf(0f, 1f, -1f, 0.5f, -2.5f, 3.5f, 256f, -1024f, 1.984375f)
        for (v in exact) {
            assertEquals(0, v.toRawBits() and 0xFFFF, "test vector $v is not bf16-exact")
            val b = bits(v)
            assertEquals((v.toRawBits() ushr 16).toShort(), b, "narrowing $v")
            assertEquals(v.toRawBits(), bf16BitsToFloat(b).toRawBits(), "round-trip $v")
        }
    }

    /**
     * The hand-derived RNE cases. 0x3F80_8000 is exactly halfway between
     * bf16 0x3F80 (1.0) and 0x3F81; the kept lsb is even, so the tie goes
     * DOWN. 0x3F81_8000 is halfway with an odd kept lsb, so the tie goes UP
     * to the even 0x3F82. One ulp either side of a tie breaks toward the
     * nearer value.
     */
    @Test
    fun roundToNearestEvenTies() {
        assertEquals(0x3F80.toShort(), bits(Float.fromBits(0x3F808000)), "tie, even keeps: down")
        assertEquals(0x3F82.toShort(), bits(Float.fromBits(0x3F818000)), "tie, odd keeps: up to even")
        assertEquals(0x3F81.toShort(), bits(Float.fromBits(0x3F808001)), "just above tie: up")
        assertEquals(0x3F81.toShort(), bits(Float.fromBits(0x3F817FFF)), "just below tie: down")
        // Sign is carried through, not special-cased: the mirrored tie.
        assertEquals(0xBF80.toShort(), bits(Float.fromBits(0xBF808000.toInt())), "negative tie, down")
        assertEquals(0xBF82.toShort(), bits(Float.fromBits(0xBF818000.toInt())), "negative tie, up to even")
    }

    /** Signed zeros keep their sign bit — the standing signed-zero rule. */
    @Test
    fun signedZeros() {
        assertEquals(0x0000.toShort(), bits(0f))
        assertEquals(0x8000.toShort(), bits(-0f))
        assertEquals((-0f).toRawBits(), bf16BitsToFloat(0x8000.toShort()).toRawBits())
    }

    @Test
    fun infinitiesPassThrough() {
        assertEquals(0x7F80.toShort(), bits(Float.POSITIVE_INFINITY))
        assertEquals(0xFF80.toShort(), bits(Float.NEGATIVE_INFINITY))
        assertEquals(Float.POSITIVE_INFINITY, bf16BitsToFloat(0x7F80.toShort()))
        assertEquals(Float.NEGATIVE_INFINITY, bf16BitsToFloat(0xFF80.toShort()))
    }

    /**
     * RNE overflow: values above the largest finite bf16 (0x7F7F) whose
     * rounding carries into the exponent land on infinity — including
     * Float.MAX_VALUE. 0x7F7F_7FFF is just below its tie and stays finite;
     * 0x7F7F_8000 ties with an odd kept lsb and rounds up into inf.
     */
    @Test
    fun overflowRoundsToInfinity() {
        assertEquals(0x7F80.toShort(), bits(Float.MAX_VALUE))
        assertEquals(0xFF80.toShort(), bits(-Float.MAX_VALUE))
        assertEquals(0x7F7F.toShort(), bits(Float.fromBits(0x7F7F7FFF)), "below tie: stays max finite")
        assertEquals(0x7F80.toShort(), bits(Float.fromBits(0x7F7F8000)), "tie, odd keeps: up into inf")
    }

    /**
     * NaN narrows to NaN, always: the payload's surviving top bits are kept
     * and the quiet bit 0x0040 is forced, so a signaling payload whose top
     * bits are all zero cannot decay to infinity.
     */
    @Test
    fun nanIsQuietedNeverLost() {
        assertEquals(0x7FC0.toShort(), bits(Float.NaN))
        assertEquals(0x7FC0.toShort(), bits(Float.fromBits(0x7F800001)), "signaling, tiny payload")
        assertEquals(0xFFC0.toShort(), bits(Float.fromBits(0xFF800001.toInt())), "negative signaling")
        assertEquals(0x7FEA.toShort(), bits(Float.fromBits(0x7FAA0000)), "payload top bits kept, quieted")
        assertTrue(bf16BitsToFloat(bits(Float.NaN)).isNaN())
    }

    /**
     * Subnormals need no special branch: f32 subnormals below half a bf16
     * subnormal ulp flush to SIGNED zero by rounding, and bf16-representable
     * subnormal patterns round-trip exactly.
     */
    @Test
    fun subnormals() {
        assertEquals(0x0000.toShort(), bits(Float.fromBits(0x00000001)), "min f32 subnormal: rounds to +0")
        assertEquals(0x8000.toShort(), bits(Float.fromBits(0x80000001.toInt())), "sign survives the flush")
        assertEquals(0x0001.toShort(), bits(Float.fromBits(0x00010000)), "min bf16 subnormal is exact")
        assertEquals(0x0001.toShort(), bits(bf16BitsToFloat(0x0001.toShort())), "and round-trips")
        assertEquals(0x0000.toShort(), bits(Float.fromBits(0x00008000)), "half-ulp tie: even, down to 0")
        assertEquals(0x0002.toShort(), bits(Float.fromBits(0x00018000)), "tie above odd subnormal: up")
    }

    /**
     * The 256-value exponent sweep, both signs, mantissa 0 and a mantissa
     * with the quiet bit set (0x55): every bf16 pattern widens exactly and
     * narrows back to itself. (Signaling-NaN patterns — exp 255, mantissa
     * without bit 6 — are deliberately excluded: quieting maps them to a
     * DIFFERENT NaN, which nanIsQuietedNeverLost pins.)
     */
    @Test
    fun exponentSweepRoundTripsBf16Identity() {
        for (sign in intArrayOf(0x0000, 0x8000)) {
            for (e in 0..255) {
                for (m in intArrayOf(0x00, 0x55)) {
                    if (e == 255 && m != 0 && (m and 0x40) == 0) continue
                    val p = (sign or (e shl 7) or m).toShort()
                    assertEquals(p, bits(bf16BitsToFloat(p)), "sweep pattern ${p.toInt() and 0xFFFF}")
                }
            }
        }
    }

    @Test
    fun storageAndConstructors() {
        val v = Tensors.bf16Vector<Sym>(floatArrayOf(1f, Float.fromBits(0x3F808000), -0f))
        assertEquals(BF16, v.dtype)
        assertEquals(1, v.rank)
        assertEquals(3L * 2, v.storage.sizeBytes)
        // The stored patterns are the RNE-narrowed ones, not truncations.
        assertEquals(listOf(0x3F80.toShort(), 0x3F80.toShort(), 0x8000.toShort()), v.hostBf16().toList())

        val m = Tensors.bf16Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        assertEquals(listOf(2, 2), m.dims.toList())
        assertEquals(4L * 2, m.storage.sizeBytes)
        assertFailsWith<IllegalArgumentException> { Tensors.bf16Matrix<Sym, Sym>(2, 2, floatArrayOf(1f)) }

        val s = Tensors.bf16Scalar(1.0f)
        assertEquals(0, s.rank)
        assertEquals(0x3F80.toShort(), s.hostBf16()[0])
    }

    /** The compute-in-f32-store-bf16 cast surfaces: widen exactly, narrow RNE. */
    @Test
    fun castSurfacesRoundTrip() {
        val f = Tensors.f32Vector<Sym>(floatArrayOf(1f, -2.5f, Float.fromBits(0x3F818000)))
        val b = f.toBf16()
        assertEquals(BF16, b.dtype)
        assertEquals(listOf(3), b.dims.toList())
        assertEquals(listOf(0x3F80.toShort(), 0xC020.toShort(), 0x3F82.toShort()), b.hostBf16().toList())
        val back = b.toF32()
        assertEquals(F32, back.dtype)
        // Exactly-representable elements survive the round trip bit-for-bit;
        // the tie case comes back as its rounded value, not the original.
        assertEquals(1f.toRawBits(), back.hostF32()[0].toRawBits())
        assertEquals((-2.5f).toRawBits(), back.hostF32()[1].toRawBits())
        assertEquals(Float.fromBits(0x3F820000).toRawBits(), back.hostF32()[2].toRawBits())
        // A second narrowing is a no-op: RNE is idempotent on bf16 values.
        assertEquals(b.hostBf16().toList(), back.toBf16().hostBf16().toList())
    }

    @Test
    fun hostBf16RefusesForeignStorage() {
        val f = Tensors.f32Scalar(1f)
        @Suppress("UNCHECKED_CAST")
        val lied = f as DTensor<ScalarShape, BF16>
        assertFailsWith<IllegalArgumentException> { lied.hostBf16() }
    }
}
