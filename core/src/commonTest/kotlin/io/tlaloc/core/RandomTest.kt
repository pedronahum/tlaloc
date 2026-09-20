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

    // -- 4b. §0.4.431 distributions (cauchy / exponential / chiSquare) -------
    //
    // Two oracles per distribution: (a) reference pins computed independently
    // in numpy float64/float32 over the SAME threefry stream (transcendental
    // libm differences are ~1 ulp, hence 1e-5-relative rather than bit-exact
    // — the normalBoxMullerReferencePins convention); (b) the transform
    // applied in THIS test to the pinned base stream, bit-exact, so the draw
    // provably IS the documented composition of the D1 kernels.

    private fun assertClose(want: FloatArray, got: FloatArray, tolScale: Float, label: String) {
        assertEquals(want.size, got.size, "$label size")
        for (i in want.indices) {
            val tol = tolScale * maxOf(1f, abs(want[i]))
            assertTrue(abs(got[i] - want[i]) < tol, "$label[$i] = ${got[i]}, want ${want[i]}")
        }
    }

    @Test
    fun cauchyQuantileTransformPins() {
        // numpy reference: tan(pi * (u - 1/2)) with f32 centring/scaling over
        // the pinned uniform stream for key (7,42) — even and odd n (the odd
        // lane rides the end-pad counter path through the base stream).
        assertClose(
            floatArrayOf(
                -0.07544064521789551f, -0.40740060806274414f, -0.6191755533218384f,
                0.2158869206905365f, -0.6117090582847595f, -2.1036055088043213f,
            ),
            cauchyFloats(key, 6),
            1e-5f,
            "cauchy6",
        )
        assertClose(
            floatArrayOf(
                -0.07544064521789551f, -0.40740060806274414f, -1.335739016532898f,
                0.2158869206905365f, -0.6117090582847595f,
            ),
            cauchyFloats(key, 5),
            1e-5f,
            "cauchy5",
        )
        // Bit-exact composition contract: the draw IS tan(π(u − ½)) over the
        // same uniform stream, arm for arm (f32 centre/scale, Double tan).
        val u = uniformFloats(key, 6)
        val byHand = FloatArray(6) {
            kotlin.math.tan(((u[it] - 0.5f) * kotlin.math.PI.toFloat()).toDouble()).toFloat()
        }
        assertContentEquals(byHand, cauchyFloats(key, 6), "cauchy must be the exact composition")
    }

    @Test
    fun exponentialQuantileTransformPins() {
        assertClose(
            floatArrayOf(
                0.6463244557380676f, 0.47297683358192444f, 0.22885660827159882f,
                0.8385897874832153f, 0.3934171199798584f,
            ),
            exponentialFloats(key, 5),
            1e-5f,
            "exp5",
        )
        val u = uniformFloats(key, 5)
        val byHand = FloatArray(5) { (-kotlin.math.ln((1.0f - u[it]).toDouble())).toFloat() }
        assertContentEquals(byHand, exponentialFloats(key, 5), "exponential must be the exact composition")
        for (v in exponentialFloats(key, 1000)) {
            assertTrue(v >= 0f, "exponential draw $v out of [0, ∞)")
        }
    }

    @Test
    fun chiSquareIsTheSumOfSquaredNormals() {
        // numpy reference pins, n=3 dof=2 and n=2 dof=3 (same 6-normal base
        // stream, regrouped — pins distinguish the block layout).
        assertClose(
            floatArrayOf(0.6015214920043945f, 3.6531028747558594f, 5.904162883758545f),
            chiSquareFloats(key, 3, 2),
            1e-4f,
            "chisq_3x2",
        )
        assertClose(
            floatArrayOf(0.9410799145698547f, 9.217707633972168f),
            chiSquareFloats(key, 2, 3),
            1e-4f,
            "chisq_2x3",
        )
        // Bit-exact definition: contiguous dof-blocks of the SAME normal
        // stream, squared, f32 left-to-right accumulation.
        val z = normalFloats(key, 6)
        val byHand = FloatArray(3) { i ->
            var acc = 0f
            for (j in 0 until 2) acc += z[i * 2 + j] * z[i * 2 + j]
            acc
        }
        assertContentEquals(byHand, chiSquareFloats(key, 3, 2), "chiSquare must be the exact definition")
        for (v in chiSquareFloats(key, 500, 4)) {
            assertTrue(v >= 0f, "chi-square draw $v out of [0, ∞)")
        }
    }

    @Test
    fun distributionMomentsAtFixedKey() {
        val momentKey = RandomKey(2026, 920)
        // Cauchy has NO mean — the sanity statistic is the sample median
        // (odd n → the exact middle order statistic). Actual: 0.0028017.
        val c = cauchyFloats(momentKey, 20_001).sortedArray()
        val median = c[c.size / 2]
        assertTrue(abs(median) < 0.02f, "cauchy median $median too far from 0")
        // Exponential(1): mean 1, var 1. Actual: 1.000532 / 0.989457.
        val e = exponentialFloats(momentKey, 20_000)
        var eSum = 0.0
        for (v in e) eSum += v
        val eMean = eSum / e.size
        var eM2 = 0.0
        for (v in e) {
            val d = v - eMean
            eM2 += d * d
        }
        eM2 /= e.size
        assertTrue(abs(eMean - 1.0) < 2e-2, "exponential mean $eMean too far from 1")
        assertTrue(abs(eM2 - 1.0) < 5e-2, "exponential var $eM2 too far from 1")
        // χ²(4): mean 4, var 8. Actual: 3.995723 / 7.712023.
        val x = chiSquareFloats(momentKey, 5_000, 4)
        var xSum = 0.0
        for (v in x) xSum += v
        val xMean = xSum / x.size
        var xM2 = 0.0
        for (v in x) {
            val d = v - xMean
            xM2 += d * d
        }
        xM2 /= x.size
        assertTrue(abs(xMean - 4.0) < 5e-2, "chi-square(4) mean $xMean too far from 4")
        assertTrue(abs(xM2 - 8.0) < 0.5, "chi-square(4) var $xM2 too far from 8")
    }

    @Test
    fun distributionTensorWrappers() {
        val cv = key.cauchyVector<Sym>(4)
        assertContentEquals(intArrayOf(4), cv.dims)
        assertContentEquals(cauchyFloats(key, 4), cv.hostF32())
        val cm = key.cauchyMatrix<Sym, Sym>(2, 3)
        assertContentEquals(intArrayOf(2, 3), cm.dims)
        assertContentEquals(cauchyFloats(key, 6), cm.hostF32())
        val ev = key.exponentialVector<Sym>(4)
        assertContentEquals(exponentialFloats(key, 4), ev.hostF32())
        val em = key.exponentialMatrix<Sym, Sym>(2, 2)
        assertContentEquals(exponentialFloats(key, 4), em.hostF32())
        val xv = key.chiSquareVector<Sym>(3, 2)
        assertContentEquals(chiSquareFloats(key, 3, 2), xv.hostF32())
        val xm = key.chiSquareMatrix<Sym, Sym>(2, 2, 3)
        assertContentEquals(intArrayOf(2, 2), xm.dims)
        assertContentEquals(chiSquareFloats(key, 4, 3), xm.hostF32())
        // Determinism rides the pure-function design — same key, same draw.
        assertContentEquals(cv.hostF32(), key.cauchyVector<Sym>(4).hostF32())
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
