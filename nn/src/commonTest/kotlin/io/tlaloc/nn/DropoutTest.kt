package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.uniformFloats
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import io.tlaloc.autograd.sum

/**
 * §0.4.441 — F5: Dropout certification. The mask is pinned BIT-EXACT against
 * the raw [uniformFloats] stream under DiffKT's own `u > p` comparison; the
 * train-mode gradient of a linear loss IS the mask (exact — `upstream · mask`
 * with upstream ≡ 1, even at the non-dyadic scale 4/3); eval is the identity.
 */
class DropoutTest {

    private val xData = floatArrayOf(0.25f, -0.5f, 1.0f, 0.75f, -1.25f, 1.5f)

    private fun input() = Tensors.f32Matrix<Sym, Sym>(2, 3, xData)

    private fun expectedMask(key: RandomKey, p: Float, n: Int): FloatArray {
        val u = uniformFloats(key, n)
        val scale = 1f / (1f - p)
        return FloatArray(n) { if (u[it] > p) scale else 0f }
    }

    /**
     * Train mode, `L = Σ dropout(x)`: the input gradient equals the mask
     * EXACTLY (assertContentEquals — the MUL adjoint against a unit upstream
     * reproduces the constant leaf bit-for-bit), the mask itself is the
     * thresholded [uniformFloats] draw, the loss is the flat-order f32
     * accumulation of `x ⊙ mask`, and dropout contributes no trainables.
     */
    @Test
    fun trainModeGradientIsTheMaskBitExact() {
        val key = RandomKey.fromSeed(42L)
        val p = 0.25f
        val mask = expectedMask(key, p, 6)
        assertTrue(
            mask.any { it == 0f } && mask.any { it != 0f },
            "degenerate draw for the pinned key — pick another seed",
        )

        val result = valueAndGradients(Sequential(Dropout(p, key)), listOf(input())) { y -> y.sum() }

        assertTrue(result.gradients.isEmpty())
        assertContentEquals(mask, result.inputGradients[0].hostF32())
        var lossRef = 0f
        for (i in xData.indices) lossRef += xData[i] * mask[i]
        assertEquals(lossRef, result.loss)
    }

    /** One captured step would reuse one mask for every run. */
    @Test
    fun oneCaptureDrawsEachStepsMaskFromTheModelsKey() {
        val k1 = RandomKey.fromSeed(7L)
        val k2 = RandomKey.fromSeed(8L)
        fun model(key: RandomKey) = Sequential(Sequential(Dropout(0.5f, key)))
        val step = capture(model(k1), listOf(input())) { y -> y.sum() }
        assertEquals(1, step.dropoutMaskCount)
        // L = Σ x·mask, so the input gradient is the mask the step ran with.
        assertContentEquals(expectedMask(k1, 0.5f, 6), step.run(model(k1), listOf(input())).inputGradients[0].hostF32())
        assertContentEquals(expectedMask(k2, 0.5f, 6), step.run(model(k2), listOf(input())).inputGradients[0].hostF32())
        assertContentEquals(expectedMask(k2, 0.5f, 6), step.dropoutMasks(model(k2)).single().hostF32())

        // A fixed mask is the capture's, whatever the key at run time; p = 0 takes no mask.
        val fixed = capture(model(k1), listOf(input()), fixedDropoutMask = true) { y -> y.sum() }
        assertEquals(0, fixed.dropoutMaskCount)
        assertContentEquals(expectedMask(k1, 0.5f, 6), fixed.run(model(k2), listOf(input())).inputGradients[0].hostF32())
        assertEquals(0, capture(Sequential(Dropout(0f, k1)), listOf(input())) { y -> y.sum() }.dropoutMaskCount)
    }

    /** Bit-determinism per key; [Dropout.withKey] swaps the stream, not the rate. */
    @Test
    fun maskIsAPureFunctionOfTheKey() {
        val key = RandomKey.fromSeed(42L)
        val layer = Dropout(0.25f, key)
        val a = valueAndGradients(Sequential(layer), listOf(input())) { y -> y.sum() }
        val b = valueAndGradients(Sequential(layer), listOf(input())) { y -> y.sum() }
        assertContentEquals(a.inputGradients[0].hostF32(), b.inputGradients[0].hostF32())

        val rekeyed = layer.withKey(RandomKey.fromSeed(43L))
        assertEquals(0.25f, rekeyed.p)
        val c = valueAndGradients(Sequential(rekeyed), listOf(input())) { y -> y.sum() }
        assertContentEquals(
            expectedMask(RandomKey.fromSeed(43L), 0.25f, 6),
            c.inputGradients[0].hostF32(),
        )
    }

    /** `inferenceMode` = identity: `L = Σx`, gradient all ones, exact. */
    @Test
    fun evalModeIsTheIdentity() {
        val frozen = Dropout(0.5f, RandomKey.fromSeed(7L)).inferenceMode()
        val result = valueAndGradients(Sequential(frozen), listOf(input())) { y -> y.sum() }
        assertEquals(1.75f, result.loss) // Σx on the quarter grid, exact
        assertContentEquals(FloatArray(6) { 1f }, result.inputGradients[0].hostF32())
    }

    /** `p = 0` keeps everything at scale 1 — the mask is all ones, exactly. */
    @Test
    fun pZeroKeepsEverythingAtScaleOne() {
        val result = valueAndGradients(
            Sequential(Dropout(0f, RandomKey.fromSeed(7L))),
            listOf(input()),
        ) { y -> y.sum() }
        assertEquals(1.75f, result.loss)
        assertContentEquals(FloatArray(6) { 1f }, result.inputGradients[0].hostF32())
    }

    @Test
    fun refusesRatesOutsideTheHalfOpenUnitInterval() {
        assertFailsWith<IllegalArgumentException> { Dropout(1f, RandomKey.fromSeed(1L)) }
        assertFailsWith<IllegalArgumentException> { Dropout(-0.1f, RandomKey.fromSeed(1L)) }
    }
}
