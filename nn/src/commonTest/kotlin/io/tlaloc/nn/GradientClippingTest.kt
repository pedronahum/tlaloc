package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * §0.4.502 (Tier 2 item 7) — gradient clipping, against HAND-COMPUTED
 * expectations. Every fixture below is chosen so the norm is an exact small
 * integer or a familiar radical, so the expected value is arithmetic a reader
 * can check rather than a number read off a run.
 *
 * The PyTorch cross-check (`clip_grad_norm_`, `clip_grad_value_`) lives in
 * `benchmarks/.../NnSchedulesAndClippingVsPytorchTest` and self-skips without
 * the oracle venv. It exists because of one deliberate divergence, recorded in
 * [GradientClipping.byGlobalNorm]: PyTorch scales by `maxNorm / (norm + 1e-6)`
 * and this scales by `maxNorm / norm`.
 */
class GradientClippingTest {

    private fun v(vararg x: Float): DTensor<*, F32> = Tensors.f32Vector<Sym>(x)

    private fun close(expected: Float, actual: Float, tag: String, tol: Float = 1e-6f) {
        assertTrue(
            abs(expected - actual) <= tol * maxOf(1f, abs(expected)),
            "$tag: expected $expected, got $actual",
        )
    }

    // ---- the norm ---------------------------------------------------------

    @Test
    fun theGlobalNormIsTheNormOfEveryEntryAcrossEveryTensor() {
        // 3-4-5: 3² + 4² = 25, so the norm is exactly 5, split across two keys
        // to prove the reduction really is global and not per tensor.
        assertEquals(5f, GradientClipping.globalNorm(mapOf("a" to v(3f), "b" to v(4f))))
        assertEquals(5f, GradientClipping.globalNorm(mapOf("a" to v(3f, 4f))))
        assertEquals(5f, GradientClipping.globalNorm(mapOf("a" to v(-3f), "b" to v(0f, 4f))))
        // 1+1+1+1 = 4, norm 2.
        assertEquals(2f, GradientClipping.globalNorm(mapOf("a" to v(1f, -1f), "b" to v(1f, 1f))))
        assertEquals(0f, GradientClipping.globalNorm(mapOf("a" to v(0f, 0f))))
        assertEquals(0f, GradientClipping.globalNorm(emptyMap()))
    }

    @Test
    fun theNormOfTenThousandSmallEntriesLandsOnTheExactValue() {
        // 10_000 entries of 1e-3: the norm is sqrt(10000 * 1e-6) = 0.1, and the
        // Double accumulator (see the KDoc) reaches it to 3e-8 relative.
        val g = (0 until 100).associate { k -> "t$k" to v(*FloatArray(100) { 1e-3f }) }
        close(0.1f, GradientClipping.globalNorm(g), "10,000 small entries", 1e-6f)
    }

    // ---- clip by global norm ----------------------------------------------

    @Test
    fun aGradientUnderTheThresholdIsReturnedUntouched() {
        val g = mapOf("a" to v(3f), "b" to v(4f))
        // Not merely equal — the SAME map, because copying would allocate a
        // full parameter set per step for nothing.
        assertSame(g, GradientClipping.byGlobalNorm(g, maxNorm = 5f))
        assertSame(g, GradientClipping.byGlobalNorm(g, maxNorm = 10f))
    }

    @Test
    fun aGradientOverTheThresholdIsScaledToExactlyTheThreshold() {
        // norm 5, maxNorm 1 -> scale 0.2
        val g = mapOf("a" to v(3f, 0f), "b" to v(0f, 4f))
        val c = GradientClipping.byGlobalNorm(g, maxNorm = 1f)
        close(0.6f, c.getValue("a").hostF32()[0], "a[0] = 3 * 1/5")
        close(0f, c.getValue("a").hostF32()[1], "a[1]")
        close(0.8f, c.getValue("b").hostF32()[1], "b[1] = 4 * 1/5")
        close(1f, GradientClipping.globalNorm(c), "the clipped norm IS the threshold")
    }

    @Test
    fun clippingByGlobalNormPreservesTheDirectionExactly() {
        val g = mapOf("a" to v(1f, 2f), "b" to v(-3f, 4f))
        val c = GradientClipping.byGlobalNorm(g, maxNorm = 0.5f)
        val norm = sqrt(1f + 4f + 9f + 16f) // sqrt(30)
        val scale = 0.5f / norm
        for (key in listOf("a", "b")) {
            val before = g.getValue(key).hostF32()
            val after = c.getValue(key).hostF32()
            for (i in before.indices) close(before[i] * scale, after[i], "$key[$i]", 2e-6f)
        }
        // Direction preserved means every ratio after/before is the same
        // number; `byValue` below is exactly the operation that breaks this.
        val ratios = listOf("a", "b").flatMap { key ->
            val before = g.getValue(key).hostF32()
            val after = c.getValue(key).hostF32()
            before.indices.map { after[it] / before[it] }
        }
        for (r in ratios) close(ratios[0], r, "ratio")
    }

    @Test
    fun theShapesAndKeysAreCarriedThrough() {
        val g = mapOf(
            "w" to Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 0f, 0f, 0f)),
            "b" to v(0f, 0f, 0f),
        )
        val c = GradientClipping.byGlobalNorm(g, maxNorm = 1f)
        assertEquals(g.keys, c.keys)
        assertEquals(listOf(2, 2), c.getValue("w").dims.toList())
        assertEquals(listOf(3), c.getValue("b").dims.toList())
        close(1f, c.getValue("w").hostF32()[0], "the single nonzero entry becomes the whole norm")
    }

    @Test
    fun aNonFiniteNormIsRefusedByNameInsteadOfScalingEverythingToNothing() {
        val nan = assertFailsWith<IllegalArgumentException> {
            GradientClipping.byGlobalNorm(mapOf("a" to v(1f, Float.NaN)), maxNorm = 1f)
        }
        assertTrue("NaN" in nan.message!!, nan.message!!)
        assertTrue("diverged" in nan.message!!, nan.message!!)
        val inf = assertFailsWith<IllegalArgumentException> {
            GradientClipping.byGlobalNorm(mapOf("a" to v(Float.POSITIVE_INFINITY)), maxNorm = 1f)
        }
        assertTrue("zero" in inf.message!!, inf.message!!)
        // An overflow that is finite per element but not in the sum of squares
        // is the subtle case: 1e30² overflows f32, and Double accumulation
        // catches it as a finite norm instead. This is therefore NOT refused,
        // and the recorded reason is that the norm is real: 1e30·sqrt(2).
        val big = GradientClipping.byGlobalNorm(mapOf("a" to v(1e30f, 1e30f)), maxNorm = 1f)
        close(1f, GradientClipping.globalNorm(big), "1e30 pair clips to the threshold", 1e-5f)
    }

    @Test
    fun aNonPositiveThresholdIsRefusedByName() {
        val e = assertFailsWith<IllegalArgumentException> {
            GradientClipping.byGlobalNorm(mapOf("a" to v(1f)), maxNorm = 0f)
        }
        assertTrue("maxNorm" in e.message!!, e.message!!)
    }

    // ---- clip by value ----------------------------------------------------

    @Test
    fun clippingByValueClampsEachElementIndependently() {
        val g = mapOf("a" to v(-5f, -0.5f, 0f, 0.5f, 5f))
        val c = GradientClipping.byValue(g, limit = 1f)
        assertEquals(
            listOf(-1f, -0.5f, 0f, 0.5f, 1f),
            c.getValue("a").hostF32().toList(),
        )
    }

    @Test
    fun clippingByValueTakesAnAsymmetricRange() {
        val g = mapOf("a" to v(-5f, 0f, 5f))
        val c = GradientClipping.byValue(g, minValue = -0.25f, maxValue = 2f)
        assertEquals(listOf(-0.25f, 0f, 2f), c.getValue("a").hostF32().toList())
    }

    @Test
    fun clippingByValueChangesTheDirectionAndThatIsWhyBothExist() {
        // The distinction between the two functions, made observable: by-value
        // clipping of (10, 1) flattens the ratio from 10:1 to 1:1.
        val g = mapOf("a" to v(10f, 1f))
        val byValue = GradientClipping.byValue(g, limit = 1f).getValue("a").hostF32()
        assertEquals(listOf(1f, 1f), byValue.toList())
        val byNorm = GradientClipping.byGlobalNorm(g, maxNorm = 1f).getValue("a").hostF32()
        close(10f, byNorm[0] / byNorm[1], "by-norm keeps the 10:1 ratio", 1e-5f)
    }

    @Test
    fun clippingByValueLeavesANaNAloneRatherThanInventingABound() {
        // `coerceIn` would return the bound for a NaN. Turning a NaN into
        // `maxValue` is inventing a gradient, so the comparison is written out.
        val c = GradientClipping.byValue(mapOf("a" to v(Float.NaN, 9f)), limit = 1f)
        assertTrue(c.getValue("a").hostF32()[0].isNaN(), "the NaN must survive as a NaN")
        assertEquals(1f, c.getValue("a").hostF32()[1])
        // An infinity, by contrast, clamps: it is a value above the bound.
        val d = GradientClipping.byValue(mapOf("a" to v(Float.POSITIVE_INFINITY)), limit = 2f)
        assertEquals(2f, d.getValue("a").hostF32()[0])
    }

    @Test
    fun anInvalidRangeIsRefusedByName() {
        assertTrue(
            "minValue" in assertFailsWith<IllegalArgumentException> {
                GradientClipping.byValue(mapOf("a" to v(1f)), minValue = 1f, maxValue = -1f)
            }.message!!,
        )
        // An infinite bound, not a NaN one: `0f <= NaN` is already false, so a
        // NaN bound is caught one require earlier by the ordering check.
        assertTrue(
            "finite" in assertFailsWith<IllegalArgumentException> {
                GradientClipping.byValue(
                    mapOf("a" to v(1f)),
                    minValue = 0f,
                    maxValue = Float.POSITIVE_INFINITY,
                )
            }.message!!,
        )
        assertTrue(
            "minValue" in assertFailsWith<IllegalArgumentException> {
                GradientClipping.byValue(mapOf("a" to v(1f)), minValue = 0f, maxValue = Float.NaN)
            }.message!!,
        )
        assertTrue(
            "limit" in assertFailsWith<IllegalArgumentException> {
                GradientClipping.byValue(mapOf("a" to v(1f)), limit = -1f)
            }.message!!,
        )
    }

    // ---- in the training loop --------------------------------------------

    @Test
    fun clippingFitsBetweenTheCapturedStepAndTheOptimizer() {
        // The documented recipe, run: clip a real gradient and confirm the
        // optimizer takes a visibly smaller step. `FixedLearningRate` is used
        // because its update IS `-alpha*g`, so the effect is exactly the
        // clip's scale and nothing else.
        val params = listOf(NamedParameter("w", v(0f, 0f)))
        val grads = mapOf<String, DTensor<*, F32>>("w" to v(30f, 40f)) // norm 50
        val opt = FixedLearningRate(0.1f)
        val unclipped = opt.step(params, grads, Unit).params.getValue("w").hostF32()
        assertEquals(listOf(-3f, -4f), unclipped.toList())
        val clipped = opt.step(params, GradientClipping.byGlobalNorm(grads, 5f), Unit)
            .params.getValue("w").hostF32()
        // scale = 5/50 = 0.1
        close(-0.3f, clipped[0], "clipped w[0]")
        close(-0.4f, clipped[1], "clipped w[1]")
    }
}
