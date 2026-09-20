package io.tlaloc.core

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * §0.4.408 — Phase D1: the stateless PRNG foundation.
 *
 * Oracles, in order of authority:
 *  1. The Random123 known-answer vectors for the threefry-2x32 block function
 *     (Salmon et al. SC'11; reproduced verbatim in JAX's own test suite).
 *  2. Reference values generated on this machine from JAX 0.10
 *     (`jax_threefry_partitionable=False`, the classic layout) for the
 *     counter layout of bits / split / fold_in / uniform — including the
 *     odd-length end-pad case, which is where a wrong layout shows first.
 *  3. Distribution moments at a FIXED key (deterministic — these are pins of
 *     a specific stream, not statistical tests that could flake).
 */
class RandomTest {

    private val key = RandomKey(7, 42)

    // -- 1. Block-function known-answer vectors (Random123 / JAX) ------------

    @Test
    fun threefryKnownAnswerVectors() {
        assertEquals(
            (0x6b200159L shl 32) or 0x99ba4efeL,
            threefry2x32(0, 0, 0, 0),
            "zero key / zero counter",
        )
        assertEquals(
            (0x1cb996fcL shl 32) or 0xbb002be7L,
            threefry2x32(-1, -1, -1, -1),
            "all-ones key / counter",
        )
        assertEquals(
            (0xc4923a9cL shl 32) or 0x483df7a0L,
            threefry2x32(0x13198a2e, 0x03707344, 0x243f6a88.toInt(), 0x85a308d3.toInt()),
            "pi-digits key / counter",
        )
    }

    // -- 2. Counter layout vs JAX (classic threefry2x32) ---------------------

    @Test
    fun bitsMatchJaxLayoutEven() {
        // jax._src.prng.threefry_2x32(key=(7,42), iota(6))
        assertContentEquals(
            intArrayOf(
                0x79dd3b0a, 0x60799add, 0x52d2cd96,
                0x91537eb1.toInt(), 0x5343c4a9, 0x24290b34,
            ),
            threefryBits(key, 6),
        )
    }

    @Test
    fun bitsMatchJaxLayoutOdd() {
        // The odd-n end-pad: counters [0..4] + one zero pad, drop the last
        // output word. Index 2 is the lane the pad touches — the pin that
        // distinguishes end-pad from front-pad.
        assertContentEquals(
            intArrayOf(
                0x8e1adabf.toInt(),
            ),
            threefryBits(key, 1),
        )
        assertContentEquals(
            intArrayOf(
                0x79dd3b0a, 0x60799add, 0x345de1ac,
                0x91537eb1.toInt(), 0x5343c4a9,
            ),
            threefryBits(key, 5),
        )
        assertContentEquals(
            intArrayOf(
                0x9c91dc40.toInt(), 0x8efafd6d.toInt(), 0x8489e9ec.toInt(),
                0x2d1f0610, 0x2352957c, 0x076e59ef, 0xe37294e0.toInt(),
            ),
            threefryBits(key, 7),
        )
    }

    @Test
    fun uniformMatchesJaxExactly() {
        // jax.random.uniform(key=(7,42), (6,)) and (5,) — the decimal strings
        // are exact f32 values, so equality is bit-exact.
        assertContentEquals(
            floatArrayOf(
                0.4760318994522095f, 0.3768554925918579f, 0.3235290050506592f,
                0.5676802396774292f, 0.3252527713775635f, 0.14125120639801025f,
            ),
            uniformFloats(key, 6),
        )
        assertContentEquals(
            floatArrayOf(
                0.4760318994522095f, 0.3768554925918579f, 0.2045574188232422f,
                0.5676802396774292f, 0.3252527713775635f,
            ),
            uniformFloats(key, 5),
        )
    }

    @Test
    fun splitMatchesJaxAndFoldInPins() {
        // jax.random.split(key=(7,42), 3)
        val children = key.split(3)
        assertEquals(RandomKey(0x79dd3b0a, 0x60799add), children[0])
        assertEquals(RandomKey(0x52d2cd96, 0x91537eb1.toInt()), children[1])
        assertEquals(RandomKey(0x5343c4a9, 0x24290b34), children[2])
        // jax.random.fold_in(key=(7,42), 13)
        assertEquals(RandomKey(0x327d35f1, 0xec8db68b.toInt()), key.foldIn(13))
    }

    @Test
    fun seedConstruction() {
        assertEquals(RandomKey(0, 42), RandomKey.fromSeed(42L))
        assertEquals(RandomKey(1, -1), RandomKey.fromSeed(0x1_FFFF_FFFFL))
    }

    // -- 3. Split independence + determinism ---------------------------------

    @Test
    fun splitChildrenAreDistinctAndDeterministic() {
        val a = key.split(4)
        val b = key.split(4)
        assertEquals(a, b, "split must be deterministic across calls")
        val all = a + key
        for (i in all.indices) for (j in i + 1 until all.size) {
            assertNotEquals(all[i], all[j], "children (and parent) must be pairwise distinct")
        }
        // Streams of sibling keys differ from each other and the parent's.
        val u0 = uniformFloats(a[0], 8)
        val u1 = uniformFloats(a[1], 8)
        val up = uniformFloats(key, 8)
        assertTrue(!u0.contentEquals(u1) && !u0.contentEquals(up), "sibling streams must differ")
        // fold_in with different data gives different keys, deterministically.
        assertEquals(key.foldIn(3), key.foldIn(3))
        assertNotEquals(key.foldIn(3), key.foldIn(4))
        assertNotEquals(key.foldIn(3), key)
    }

    @Test
    fun sameKeyAndShapeReproduceIdenticalBits() {
        assertContentEquals(uniformFloats(key, 100), uniformFloats(key, 100))
        assertContentEquals(normalFloats(key, 100), normalFloats(key, 100))
        assertContentEquals(threefryBits(key, 101), threefryBits(key, 101))
    }

    // -- 4. Distribution sanity at the fixed key (deterministic pins) --------

    @Test
    fun uniformRangeAndMoments() {
        val n = 100_000
        val u = uniformFloats(key, n)
        var sum = 0.0
        var sumSq = 0.0
        for (v in u) {
            assertTrue(v >= 0f && v < 1f, "uniform draw $v out of [0, 1)")
            sum += v
            sumSq += v.toDouble() * v
        }
        val mean = sum / n
        val variance = sumSq / n - mean * mean
        // Actual values for key (7,42): mean 0.501502, var 0.083100.
        assertTrue(abs(mean - 0.5) < 5e-3, "uniform mean $mean too far from 1/2")
        assertTrue(abs(variance - 1.0 / 12.0) < 3e-3, "uniform var $variance too far from 1/12")
    }

    @Test
    fun normalMoments() {
        val n = 100_000
        val z = normalFloats(key, n)
        var sum = 0.0
        for (v in z) sum += v
        val mean = sum / n
        var m2 = 0.0
        var m3 = 0.0
        for (v in z) {
            val d = v - mean
            m2 += d * d
            m3 += d * d * d
        }
        m2 /= n
        m3 /= n
        val skew = m3 / (m2 * kotlin.math.sqrt(m2))
        // Actual values for key (7,42): mean -0.00235, var 0.99333, skew 0.00549.
        assertTrue(abs(mean) < 1e-2, "normal mean $mean too far from 0")
        assertTrue(abs(m2 - 1.0) < 2e-2, "normal var $m2 too far from 1")
        assertTrue(abs(skew) < 3e-2, "normal skew $skew too large")
    }

    @Test
    fun normalBoxMullerReferencePins() {
        // The Box-Muller recipe over the pinned uniform stream, computed
        // independently in numpy float64 (transcendental libm differences are
        // ~1 ulp, hence the 1e-5 tolerance rather than bit-exactness).
        val want = floatArrayOf(
            0.2569464445114136f, 0.7317786812782288f, -0.5827164053916931f,
            1.8203144073486328f, -1.0202655792236328f, -2.2052712440490723f,
        )
        val got = normalFloats(key, 6)
        for (i in want.indices) {
            assertTrue(abs(got[i] - want[i]) < 1e-5f, "normal[$i] = ${got[i]}, want ${want[i]}")
        }
    }

    // -- 5. Host tensor surface ----------------------------------------------

    @Test
    fun hostTensorWrappers() {
        val v = key.uniformVector<Sym>(5)
        assertContentEquals(intArrayOf(5), v.dims)
        assertContentEquals(uniformFloats(key, 5), v.hostF32())
        val m = key.uniformMatrix<Sym, Sym>(2, 3)
        assertContentEquals(intArrayOf(2, 3), m.dims)
        assertContentEquals(uniformFloats(key, 6), m.hostF32(), "matrix draws over the flat index space")
        val z = key.normalMatrix<Sym, Sym>(3, 2)
        assertContentEquals(intArrayOf(3, 2), z.dims)
        assertContentEquals(normalFloats(key, 6), z.hostF32())
    }
}
