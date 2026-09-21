package io.tlaloc.ir.inference

import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.ir.recognizer.quant.KvQuantDtype
import io.tlaloc.ir.recognizer.quant.KvScaleStrategy
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.472 — Phase H5: the KV-pool quantization contract.
 *
 * The oracle story, strongest first:
 *
 * 1. **A HAND-DERIVED case with exact arithmetic.** The scales are chosen so
 *    that `absmax / qmax` is a power of two and every step is exact in binary
 *    floating point — so the expected codes, the expected dequantized values
 *    AND the expected errors are all written out as literals a reader can
 *    check on paper, with no tolerance anywhere.
 * 2. **The derived error bound, recomputed rather than tuned.** Over a large
 *    pseudo-random pool, every element's round-trip error is asserted against
 *    `scale/2` — the bound derived in [KvQuantPool]'s file comment — and the
 *    test additionally pins that the bound is TIGHT (some element gets close
 *    to it), so a codec that quietly rounded to zero could not pass by being
 *    conservative.
 * 3. **The named refusals**: fp8 is refused BY NAME (it is not an
 *    integer-coded format), and so are mismatched scale vectors and
 *    out-of-range codes.
 */
class KvQuantPoolTest {

    private val int8PerHead = KvQuantConfig(KvQuantDtype.INT8, KvScaleStrategy.PER_HEAD)
    private val int8PerTensor = KvQuantConfig.INT8_PER_TENSOR
    private val int4PerHead = KvQuantConfig(KvQuantDtype.INT4, KvScaleStrategy.PER_HEAD)

    // --- Oracle 1: the hand-derived case. ------------------------------
    //
    // Pool [numBlocks=1, blockSize=2, numKvHeads=2, headDim=2], laid out
    // row-major, so the flat order is
    //   slot0: head0 {a, b}, head1 {c, d}
    //   slot1: head0 {e, f}, head1 {g, h}
    //
    // head 0 values: 127, -64, 0.4, -0.6   -> absmax 127 -> scale = 127/127 = 1
    // head 1 values: 254,  -3,   1,   -1   -> absmax 254 -> scale = 254/127 = 2
    //
    // Both scales are exact in binary. The codes follow by round-half-AWAY:
    //   head0: 127, -64, round(0.4)=0, round(-0.6)=-1
    //   head1: round(127)=127, round(-1.5)=-2, round(0.5)=1, round(-0.5)=-1
    // and the dequantized values are code*scale:
    //   head0: 127, -64, 0, -1     head1: 254, -4, 2, -2
    // with per-element errors
    //   head0: 0, 0, 0.4, 0.4      head1: 0, 1, 1, 1
    // every one of which is <= scale/2 (0.5 and 1.0 respectively) — and head 1
    // sits exactly ON its bound three times, which is the tie case.

    private val handPool = floatArrayOf(
        127f, -64f, /* head0 slot0 */ 254f, -3f, /* head1 slot0 */
        0.4f, -0.6f, /* head0 slot1 */ 1f, -1f, /* head1 slot1 */
    )

    @Test
    fun handDerivedPerHeadScalesAreExact() {
        val scales = KvQuantPool.scalesOf(handPool, numKvHeads = 2, headDim = 2, config = int8PerHead)
        assertContentEquals(floatArrayOf(1f, 2f), scales, "absmax/127 per head, both exact powers of two")
    }

    @Test
    fun handDerivedCodesAndDequantAreExact() {
        val q = KvQuantPool.quantize(handPool, numKvHeads = 2, headDim = 2, config = int8PerHead)
        assertContentEquals(
            intArrayOf(127, -64, 127, -2, 0, -1, 1, -1),
            q.codes,
            "hand-derived codes, ties away from zero (-1.5 -> -2, 0.5 -> 1, -0.5 -> -1)",
        )
        assertContentEquals(
            floatArrayOf(127f, -64f, 254f, -4f, 0f, -1f, 2f, -2f),
            q.dequantize(numKvHeads = 2, headDim = 2),
            "dequantized values are code * scale, exactly",
        )
    }

    @Test
    fun handDerivedErrorsSitAtOrUnderTheDerivedBound() {
        val q = KvQuantPool.quantize(handPool, numKvHeads = 2, headDim = 2, config = int8PerHead)
        val back = q.dequantize(numKvHeads = 2, headDim = 2)
        val expected = floatArrayOf(0f, 0f, 0f, 1f, 0.4f, 0.4f, 1f, 1f)
        for (i in handPool.indices) {
            assertEquals(
                expected[i], abs(handPool[i] - back[i]), 1e-6f,
                "element $i's round-trip error is the hand-derived one",
            )
        }
        // Per-head bound: 0.5 for head 0 (scale 1), 1.0 for head 1 (scale 2).
        assertEquals(1f, KvQuantPool.maxRoundTripError(q.scales), 0f, "the widest scale is 2, so the bound is 1")
    }

    @Test
    fun roundHalfAwayFromZeroIsTheTieConvention() {
        assertEquals(1, KvQuantPool.roundHalfAway(0.5f))
        assertEquals(-1, KvQuantPool.roundHalfAway(-0.5f))
        assertEquals(-2, KvQuantPool.roundHalfAway(-1.5f))
        assertEquals(2, KvQuantPool.roundHalfAway(1.5f))
        assertEquals(0, KvQuantPool.roundHalfAway(-0.49f))
    }

    // --- Oracle 2: the bound over a big pool, and its tightness. --------

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 8f
        }
    }

    @Test
    fun everyElementIsWithinTheDerivedBoundAndTheBoundIsTight() {
        val numKvHeads = 3
        val headDim = 5
        val pool = pseudo(4 * 2 * numKvHeads * headDim, 20260921)
        val q = KvQuantPool.quantize(pool, numKvHeads, headDim, int8PerHead)
        val back = q.dequantize(numKvHeads, headDim)

        var worstRatio = 0f
        for (i in pool.indices) {
            val head = (i / headDim) % numKvHeads
            val bound = q.scales[head] / 2f
            val err = abs(pool[i] - back[i])
            assertTrue(
                err <= bound + 1e-6f,
                "element $i: error $err exceeds the derived bound $bound (scale ${q.scales[head]})",
            )
            val ratio = err / bound
            if (ratio > worstRatio) worstRatio = ratio
        }
        assertTrue(
            worstRatio > 0.9f,
            "the scale/2 bound must be TIGHT — some element should land near it; worst ratio $worstRatio",
        )
    }

    @Test
    fun int4IsTheSameContractWithACoarserCodeRange() {
        val pool = pseudo(2 * 2 * 2 * 4, 99)
        val q4 = KvQuantPool.quantize(pool, numKvHeads = 2, headDim = 4, config = int4PerHead)
        assertTrue(q4.codes.all { it in -7..7 }, "int4 codes live in [-7, 7]")
        val q8 = KvQuantPool.quantize(pool, numKvHeads = 2, headDim = 4, config = int8PerHead)
        val e4 = pool.indices.maxOf { abs(pool[it] - q4.dequantize(2, 4)[it]) }
        val e8 = pool.indices.maxOf { abs(pool[it] - q8.dequantize(2, 4)[it]) }
        assertTrue(e4 > e8, "a 4-bit code set must be strictly worse than an 8-bit one here; got $e4 vs $e8")
    }

    @Test
    fun perTensorUsesOneScaleAndIsNoBetterThanPerHead() {
        val numKvHeads = 2
        val headDim = 4
        // Head 1's values are 100x head 0's, which is precisely the case
        // per-head scaling exists for: one shared scale spends the whole code
        // range on the loud head.
        val pool = FloatArray(2 * 2 * numKvHeads * headDim) { i ->
            val head = (i / headDim) % numKvHeads
            val base = ((i % headDim) + 1).toFloat()
            if (head == 0) base else base * 100f
        }
        val perTensor = KvQuantPool.quantize(pool, numKvHeads, headDim, int8PerTensor)
        assertEquals(1, perTensor.scales.size, "per-tensor scaling keeps ONE scale")
        val perHead = KvQuantPool.quantize(pool, numKvHeads, headDim, int8PerHead)
        assertEquals(2, perHead.scales.size)

        val quietErrTensor = (0 until headDim).maxOf { abs(pool[it] - perTensor.dequantize(numKvHeads, headDim)[it]) }
        val quietErrHead = (0 until headDim).maxOf { abs(pool[it] - perHead.dequantize(numKvHeads, headDim)[it]) }
        assertTrue(
            quietErrHead < quietErrTensor,
            "per-head must beat per-tensor on the quiet head; got $quietErrHead vs $quietErrTensor",
        )
    }

    @Test
    fun anAllZeroGroupTakesScaleOneAndRoundTripsExactly() {
        val pool = FloatArray(8)
        val q = KvQuantPool.quantize(pool, numKvHeads = 2, headDim = 2, config = int8PerHead)
        assertContentEquals(floatArrayOf(1f, 1f), q.scales, "an all-zero group takes scale 1, never 0")
        assertTrue(q.dequantize(2, 2).all { it == 0f })
    }

    // --- Oracle 3: the named refusals. ---------------------------------

    @Test
    fun fp8IsRefusedByNameAndNotApproximated() {
        val ex = assertFailsWith<IllegalArgumentException> {
            KvQuantPool.quantize(handPool, 2, 2, KvQuantConfig.FP8_PER_HEAD)
        }
        val msg = ex.message ?: ""
        assertTrue("fp8_e4m3" in msg, "must name the refused dtype; got: $msg")
        assertTrue("REFUSED BY NAME" in msg, "must be a refusal, not a gap; got: $msg")
        assertTrue("int8" in msg, "must name what IS supported; got: $msg")
    }

    @Test
    fun codeMaxRefusesTheFloatFormatsByName() {
        val ex = assertFailsWith<IllegalStateException> { KvQuantDtype.FP8_E5M2.codeMax }
        assertTrue("isIntegerCoded" in (ex.message ?: ""), "must point at the predicate; got: ${ex.message}")
    }

    @Test
    fun aWrongLengthScaleVectorIsRefused() {
        val ex = assertFailsWith<IllegalArgumentException> {
            KvQuantPool.dequantize(intArrayOf(1, 2, 3, 4), floatArrayOf(1f, 1f, 1f), 2, 2, int8PerHead)
        }
        assertTrue("needs 2 scale(s)" in (ex.message ?: ""), "must state the expected count; got: ${ex.message}")
    }

    @Test
    fun anOutOfRangeCodeIsRefusedByTheCodeRange() {
        val ex = assertFailsWith<IllegalArgumentException> {
            KvQuantPool.dequantize(intArrayOf(0, 0, 128, 0), floatArrayOf(1f, 1f), 2, 2, int8PerHead)
        }
        val msg = ex.message ?: ""
        assertTrue("128" in msg && "int8" in msg, "must name the offending code and the format; got: $msg")
    }

    @Test
    fun aNonFinitePoolIsRefusedRatherThanQuantized() {
        val bad = floatArrayOf(1f, Float.NaN, 1f, 1f)
        val ex = assertFailsWith<IllegalArgumentException> { KvQuantPool.quantize(bad, 2, 1, int8PerHead) }
        assertTrue("not finite" in (ex.message ?: ""), "must say what is wrong; got: ${ex.message}")
    }

    @Test
    fun aScaleVectorSuppliedFromElsewhereClampsRatherThanOverflows() {
        // The clamp's whole purpose: scales that did NOT come from this pool.
        val codes = KvQuantPool.quantizeWithScales(
            floatArrayOf(1000f, -1000f, 1f, -1f), floatArrayOf(1f, 1f), numKvHeads = 2, headDim = 1,
            config = int8PerHead,
        )
        assertContentEquals(intArrayOf(127, -127, 1, -1), codes, "out-of-range values clamp to ±codeMax")
    }
}
