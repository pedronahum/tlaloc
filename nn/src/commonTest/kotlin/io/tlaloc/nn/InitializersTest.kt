package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.hostF32
import io.tlaloc.core.normalFloats
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.438 — F2: the initializer family, BIT-DETERMINISTIC against the D1
 * threefry streams. The layered oracle: [uniformInit]/[gaussianInit] pin
 * bit-exact against the raw `uniformFloats`/`normalFloats` draws under the
 * identical affine rescale; the fan/gain SCALE FACTORS pin analytically; the
 * Kaiming/Xavier variants pin against the base initializers with those
 * replicated bounds; and the Dense factory pins the F0 §4.0.7 key-split
 * discipline (one child per parameter tensor, declaration order).
 */
class InitializersTest {

    private val key = RandomKey.fromSeed(42L)

    @Test
    fun uniformInitIsTheBitExactAffineRescaleOfUniformFloats() {
        val t = uniformInit(key, intArrayOf(2, 3), min = -0.5f, max = 0.5f)
        assertContentEquals(intArrayOf(2, 3), t.dims)

        val u = uniformFloats(key, 6)
        val expected = FloatArray(6) { u[it] * (0.5f - (-0.5f)) + (-0.5f) }
        assertContentEquals(expected, t.hostF32())

        // Deterministic: the same key redraws the same bits.
        assertContentEquals(t.hostF32(), uniformInit(key, intArrayOf(2, 3), -0.5f, 0.5f).hostF32())
        // Range honoured: [min, max).
        for (v in t.hostF32()) assertTrue(v >= -0.5f && v < 0.5f)
    }

    @Test
    fun gaussianInitIsTheBitExactAffineRescaleOfNormalFloats() {
        val t = gaussianInit(key, intArrayOf(5), mean = 0.25f, variance = 4f)
        val z = normalFloats(key, 5)
        // sqrt(4f) == 2f exactly — the scale is analytic here, the draws bit-pinned.
        assertContentEquals(FloatArray(5) { z[it] * 2f + 0.25f }, t.hostF32())
        assertContentEquals(t.hostF32(), gaussianInit(key, intArrayOf(5), 0.25f, 4f).hostF32())
    }

    @Test
    fun fanFactorsAreDiffktsExactly() {
        // FanIn.fanSize = dims[1], FanOut.fanSize = dims[0], × Π dims[2:].
        assertEquals(2, fanOf(intArrayOf(8, 2), FanMode.FanIn))
        assertEquals(8, fanOf(intArrayOf(8, 2), FanMode.FanOut))
        assertEquals(2 * 9, fanOf(intArrayOf(4, 2, 3, 3), FanMode.FanIn))
        assertEquals(4 * 9, fanOf(intArrayOf(4, 2, 3, 3), FanMode.FanOut))
        assertFailsWith<IllegalArgumentException> { fanOf(intArrayOf(7), FanMode.FanIn) }
    }

    @Test
    fun gainConstantsAreDiffktsExactly() {
        assertEquals(1f, ActivationGain.Linear.gain)
        assertEquals(1f, ActivationGain.Conv.gain)
        assertEquals(1f, ActivationGain.Sigmoid.gain)
        assertEquals(5f / 3f, ActivationGain.Tanh.gain)
        assertEquals(sqrt(2f), ActivationGain.Relu.gain)
        // LeakyRelu(0) degenerates to Relu's gain; LeakyRelu(1) to √(2/2) = 1 — both exact.
        assertEquals(ActivationGain.Relu.gain, ActivationGain.LeakyRelu(0f).gain)
        assertEquals(1f, ActivationGain.LeakyRelu(1f).gain)
    }

    @Test
    fun kaimingUniformIsUniformInitAtTheAnalyticBound() {
        // fan = dims[1] = 2 under FanIn; bound = √(3/2)·√2 (DiffKT's formula verbatim).
        val bound = sqrt(3f / 2) * ActivationGain.Relu.gain
        val t = kaimingUniformInit(key, intArrayOf(8, 2), FanMode.FanIn, ActivationGain.Relu)
        assertContentEquals(
            uniformInit(key, intArrayOf(8, 2), -bound, bound).hostF32(),
            t.hostF32(),
        )
        for (v in t.hostF32()) assertTrue(v >= -bound && v < bound)

        // The conv-default spelling (FanIn, LeakyRelu(√5)) on a rank-4 kernel:
        // fan = 2·3·3 = 18, bound = √(3/18)·√(2/6).
        val convBound = sqrt(3f / 18) * ActivationGain.LeakyRelu(sqrt(5f)).gain
        assertContentEquals(
            uniformInit(key, intArrayOf(4, 2, 3, 3), -convBound, convBound).hostF32(),
            kaimingUniformInit(key, intArrayOf(4, 2, 3, 3), FanMode.FanIn, ActivationGain.LeakyRelu(sqrt(5f))).hostF32(),
        )
    }

    @Test
    fun kaimingNormalIsNormalFloatsAtTheAnalyticStd() {
        val std = ActivationGain.Relu.gain / sqrt(2f) // fan = 2 → √2/√2; kept symbolic, not simplified
        val z = normalFloats(key, 16)
        assertContentEquals(
            FloatArray(16) { z[it] * std },
            kaimingNormalInit(key, intArrayOf(8, 2), FanMode.FanIn, ActivationGain.Relu).hostF32(),
        )
    }

    @Test
    fun xavierPairIsTheGlorotFormulaOverBothFans() {
        // dims [4, 6]: fanIn = 6, fanOut = 4, sum = 10.
        val bound = sqrt(6f / 10)
        assertContentEquals(
            uniformInit(key, intArrayOf(4, 6), -bound, bound).hostF32(),
            xavierUniformInit(key, intArrayOf(4, 6)).hostF32(),
        )
        val std = sqrt(2f / 10)
        val z = normalFloats(key, 24)
        assertContentEquals(
            FloatArray(24) { z[it] * std },
            xavierNormalInit(key, intArrayOf(4, 6)).hostF32(),
        )
    }

    /**
     * The Phase-F key-split discipline, pinned: a Dense drawn from [key]
     * consumes `key.split(2)` with child 0 → w and child 1 → b (declaration
     * order), both `uniform(±√(1/numInputs))` — DiffKT's default for BOTH
     * tensors — and `bias = false` still draws w off child 0, so the weights
     * are identical with and without a bias.
     */
    @Test
    fun denseFactoryHonoursTheKeySplitDiscipline() {
        val bound = sqrt(1f / 2)
        val keys = key.split(2)
        val dense = Dense(2, 3, key)

        assertEquals(listOf("w", "b"), dense.parameters.map { it.key })
        assertContentEquals(intArrayOf(2, 3), dense.w.dims)
        assertContentEquals(
            uniformInit(keys[0], intArrayOf(2, 3), -bound, bound).hostF32(),
            dense.w.hostF32(),
        )
        assertContentEquals(
            uniformInit(keys[1], intArrayOf(3), -bound, bound).hostF32(),
            dense.b!!.hostF32(),
        )

        val noBias = Dense(2, 3, key, bias = false)
        assertEquals(listOf("w"), noBias.parameters.map { it.key })
        assertContentEquals(dense.w.hostF32(), noBias.w.hostF32())
    }
}
