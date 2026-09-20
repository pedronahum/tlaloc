package io.tlaloc.core.ops

import io.tlaloc.core.DScalar
import io.tlaloc.core.DoubleScalar
import io.tlaloc.core.FloatScalar
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class HostOpsTest {

    @Test
    fun elementwiseAdd() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val b = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
        val c = a + b
        assertContentEquals(floatArrayOf(11f, 22f, 33f, 44f), c.hostF32())
        assertContentEquals(intArrayOf(2, 2), c.dims)
    }

    @Test
    fun elementwiseMulDivSub() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(6f, 8f, 10f))
        val b = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(2f, 4f, 5f))
        assertContentEquals(floatArrayOf(12f, 32f, 50f), (a * b).hostF32())
        assertContentEquals(floatArrayOf(3f, 2f, 2f), (a / b).hostF32())
        assertContentEquals(floatArrayOf(4f, 4f, 5f), (a - b).hostF32())
    }

    @Test
    fun scalarMixingTensorOnLeft() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(6f, 8f, 10f, 12f))
        assertContentEquals(floatArrayOf(8f, 10f, 12f, 14f), (a + 2f).hostF32())
        assertContentEquals(floatArrayOf(4f, 6f, 8f, 10f), (a - 2f).hostF32())
        assertContentEquals(floatArrayOf(12f, 16f, 20f, 24f), (a * 2f).hostF32())
        assertContentEquals(floatArrayOf(3f, 4f, 5f, 6f), (a / 2f).hostF32())
        assertContentEquals(intArrayOf(1, 4), (a + 2f).dims)
    }

    @Test
    fun scalarMixingScalarOnLeft() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(1f, 2f, 4f, 8f))
        assertContentEquals(floatArrayOf(11f, 12f, 14f, 18f), (10f + a).hostF32())
        // Non-commutative: the scalar-on-left spelling is not `a - 10f`.
        assertContentEquals(floatArrayOf(9f, 8f, 6f, 2f), (10f - a).hostF32())
        assertContentEquals(floatArrayOf(10f, 20f, 40f, 80f), (10f * a).hostF32())
        assertContentEquals(floatArrayOf(10f, 5f, 2.5f, 1.25f), (10f / a).hostF32())
        assertContentEquals(intArrayOf(1, 4), (10f - a).dims)
    }

    @Test
    fun powElementwise() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(1f, 2f, 3f, 4f))
        val exp = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(2f, 2f, 3f, 0.5f))
        assertContentEquals(floatArrayOf(1f, 4f, 27f, 2f), a.pow(exp).hostF32())
        assertContentEquals(floatArrayOf(1f, 4f, 9f, 16f), a.pow(2.0f).hostF32())
        assertContentEquals(floatArrayOf(1f, 8f, 27f, 64f), a.pow(3).hostF32())
        assertContentEquals(intArrayOf(1, 4), a.pow(2.0f).dims)
    }

    @Test
    fun tanAtanElementwise() {
        // §0.4.395 — Phase C2 trig tails: elementwise tan/atan, and the
        // round-trip atan(tan(x)) = x on (−π/2, π/2).
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(0f, 0.5f, -1.0f, 1.2f))
        val t = a.tan().hostF32()
        val at = a.atan().hostF32()
        for (i in 0 until 4) {
            val x = a.hostF32()[i].toDouble()
            assertEquals(kotlin.math.tan(x).toFloat(), t[i], 1e-6f, "tan slot $i")
            assertEquals(kotlin.math.atan(x).toFloat(), at[i], 1e-6f, "atan slot $i")
        }
        val roundTrip = a.tan().atan().hostF32()
        for (i in 0 until 4) {
            assertEquals(a.hostF32()[i], roundTrip[i], 1e-5f, "atan(tan(x)) slot $i")
        }
        assertContentEquals(intArrayOf(1, 4), a.tan().dims)
    }

    @Test
    fun flipReversesAlongListedAxes() {
        // §0.4.396 — Phase C3: trailing axis, LEADING axis (rows swap — not a
        // contiguous flat reversal), both axes, negative-axis spelling, the
        // involution, and both refusals.
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        assertContentEquals(floatArrayOf(3f, 2f, 1f, 6f, 5f, 4f), a.flip(1).hostF32())
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 1f, 2f, 3f), a.flip(0).hostF32())
        assertContentEquals(floatArrayOf(6f, 5f, 4f, 3f, 2f, 1f), a.flip(0, 1).hostF32())
        assertContentEquals(a.flip(1).hostF32(), a.flip(-1).hostF32())
        assertContentEquals(a.hostF32(), a.flip(0, 1).flip(0, 1).hostF32())
        assertContentEquals(intArrayOf(2, 3), a.flip(0).dims)
        // Fixed-arity synthesis delegates route through the same walk.
        assertContentEquals(a.flip(0).hostF32(), flipAxes1<Rank2<Sym, Sym>>(a, 0).hostF32())
        assertContentEquals(a.flip(0, 1).hostF32(), flipAxes2<Rank2<Sym, Sym>>(a, 0, 1).hostF32())
        assertFailsWith<IllegalArgumentException> { a.flip() }
        assertFailsWith<IllegalArgumentException> { a.flip(2) }
        assertFailsWith<IllegalArgumentException> { a.flip(0, 0) }
    }

    @Test
    fun concatAndStackWindows() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val b = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f))
        // Trailing axis: each of `a`'s rows is followed by `b`'s.
        val c = concat(1, a, b)
        assertContentEquals(intArrayOf(2, 5), c.dims)
        assertContentEquals(
            floatArrayOf(1f, 2f, 10f, 20f, 30f, 3f, 4f, 40f, 50f, 60f),
            c.hostF32(),
        )
        // Leading axis: the copy is NOT one contiguous run.
        val d = concat(0, a, Tensors.f32Matrix<Sym, Sym>(1, 2, floatArrayOf(7f, 8f)))
        assertContentEquals(intArrayOf(3, 2), d.dims)
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f, 7f, 8f), d.hostF32())
        // Three operands fold through the same pairwise path.
        assertContentEquals(intArrayOf(2, 7), concat(1, a, b, a).dims)
        // stack = unsqueeze each, then concat along the new axis.
        val s = stack(0, a, a)
        assertContentEquals(intArrayOf(2, 2, 2), s.dims)
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f), s.hostF32())
        assertFailsWith<IllegalArgumentException> { concat(1, a) }
        assertFailsWith<IllegalArgumentException> { concat(0, a, b) }
    }

    @Test
    fun shapeMismatchFailsFast() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))
        val b = Tensors.f32Matrix<Sym, Sym>(3, 2, FloatArray(6))
        assertFailsWith<IllegalArgumentException> { a + b }
    }

    @Test
    fun reluZerosNegatives() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(-1f, 0f, 2f, -3.5f))
        assertContentEquals(floatArrayOf(0f, 0f, 2f, 0f), a.relu().hostF32())
    }

    @Test
    fun negFlipsSigns() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(1f, -2f, 3f))
        assertContentEquals(floatArrayOf(-1f, 2f, -3f), a.neg().hostF32())
    }

    @Test
    fun sumProducesScalar() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
        val s = a.sum()
        assertEquals(0, s.rank)
        assertContentEquals(floatArrayOf(10f), s.hostF32())
    }

    @Test
    fun meanAverages() {
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(2f, 4f, 6f, 8f))
        assertContentEquals(floatArrayOf(5f), a.mean().hostF32())
    }

    @Test
    fun compareAgainstFloatScalar() {
        // §0.4.397 — Phase A5c-3(iv): `a gt 1.0f` and friends, the Float-side
        // comparison overloads producing the same 0/1 F32 masks as the
        // tensor⊙tensor forms.
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(0.5f, 1.0f, 2.0f, -3.0f))
        assertContentEquals(floatArrayOf(0f, 0f, 1f, 0f), (a gt 1.0f).hostF32())
        assertContentEquals(floatArrayOf(0f, 1f, 1f, 0f), (a ge 1.0f).hostF32())
        assertContentEquals(floatArrayOf(1f, 0f, 0f, 1f), (a lt 1.0f).hostF32())
        assertContentEquals(floatArrayOf(1f, 1f, 0f, 1f), (a le 1.0f).hostF32())
        assertContentEquals(floatArrayOf(0f, 1f, 0f, 0f), (a eq 1.0f).hostF32())
        assertContentEquals(floatArrayOf(1f, 0f, 1f, 1f), (a ne 1.0f).hostF32())
        assertContentEquals(intArrayOf(2, 2), (a gt 1.0f).dims)
    }

    @Test
    fun statsReturnsMeanAndBiasedVariance() {
        // §0.4.397 — DiffKT's stats(): (mean, variance), variance BIASED
        // (divide by N — the §0.4.390 batchNorm convention).
        // v = [2, 4, 6, 8]: mean = 5, var = (9 + 1 + 1 + 9)/4 = 5.
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(2f, 4f, 6f, 8f))
        val (mu, variance) = a.stats()
        assertContentEquals(floatArrayOf(5f), mu.hostF32())
        assertContentEquals(floatArrayOf(5f), variance.hostF32())
        assertContentEquals(intArrayOf(), mu.dims)
        assertContentEquals(intArrayOf(), variance.dims)
    }

    @Test
    fun dscalarTimesTensorBothOrders() {
        // §0.4.397 — DiffKT `timesScalar` parity: DScalar × DTensor in both
        // operand orders, DoubleScalar narrowing through toFloat().
        val a = Tensors.f32Matrix<Sym, Sym>(1, 4, floatArrayOf(1f, 2f, 3f, 4f))
        val s: DScalar = FloatScalar(2.5f)
        assertContentEquals(floatArrayOf(2.5f, 5f, 7.5f, 10f), (a * s).hostF32())
        assertContentEquals(floatArrayOf(2.5f, 5f, 7.5f, 10f), (s * a).hostF32())
        val d: DScalar = DoubleScalar(0.5)
        assertContentEquals(floatArrayOf(0.5f, 1f, 1.5f, 2f), (a * d).hostF32())
        assertContentEquals(intArrayOf(1, 4), (s * a).dims)
    }

    @Test
    fun matmul2x3Times3x2() {
        // [1 2 3]   [7  8]    [1*7+2*9+3*11,  1*8+2*10+3*12]   [58  64]
        // [4 5 6] x [9  10] = [4*7+5*9+6*11,  4*8+5*10+6*12] = [139 154]
        //           [11 12]
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = Tensors.f32Matrix<Sym, Sym>(3, 2, floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f))
        val c: DTensorAlias = a matmul b
        assertContentEquals(intArrayOf(2, 2), c.dims)
        assertContentEquals(floatArrayOf(58f, 64f, 139f, 154f), c.hostF32())
    }

    @Test
    fun matmulInnerDimMismatchFails() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))
        val b = Tensors.f32Matrix<Sym, Sym>(4, 2, FloatArray(8))
        assertFailsWith<IllegalArgumentException> { a matmul b }
    }

    @Test
    fun identityMatmul() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(5f, 7f, 11f, 13f))
        val id = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 0f, 0f, 1f))
        assertContentEquals(a.hostF32(), (a matmul id).hostF32())
    }

    @Test
    fun broadcastDimsRectangularRank2() {
        val out = broadcastDims<Rank2<Sym, Sym>>(2.5f, intArrayOf(2, 3))
        assertContentEquals(intArrayOf(2, 3), out.dims)
        assertContentEquals(FloatArray(6) { 2.5f }, out.hostF32())
    }

    @Test
    fun broadcastDimsCopiesDimsArray() {
        val dims = intArrayOf(3, 4)
        val out = broadcastDims<Rank2<Sym, Sym>>(0f, dims)
        dims[0] = 99
        // Output's dims must remain (3, 4) — caller mutating their array doesn't poison the tensor.
        assertContentEquals(intArrayOf(3, 4), out.dims)
    }

    @Test
    fun broadcastDimsEmptyShapeYieldsScalarSized() {
        val out = broadcastDims<io.tlaloc.core.ScalarShape>(7f, intArrayOf())
        assertEquals(1, out.size)
        assertContentEquals(floatArrayOf(7f), out.hostF32())
    }

    @Test
    fun stepIsOneOnPositiveZeroOnNonPositive() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 0f, -1f, 0.5f, -0.0f, 100f))
        val s = a.step()
        assertContentEquals(floatArrayOf(1f, 0f, 0f, 1f, 0f, 1f), s.hostF32())
        assertContentEquals(intArrayOf(2, 3), s.dims)
    }

    @Test
    fun stepPreservesShape() {
        val a = Tensors.f32Matrix<Sym, Sym>(3, 2, FloatArray(6) { 1f })
        val s = a.step()
        assertContentEquals(intArrayOf(3, 2), s.dims)
    }

    @Test
    fun signMapsThreeWaysAtZeroPositiveAndNegative() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 0f, -1f, 0.5f, -0.0f, -100f))
        val s = a.sign()
        // 0f and -0f both map to 0; positives → 1, negatives → -1.
        assertContentEquals(floatArrayOf(1f, 0f, -1f, 1f, 0f, -1f), s.hostF32())
        assertContentEquals(intArrayOf(2, 3), s.dims)
    }

    @Test
    fun signPreservesShape() {
        val a = Tensors.f32Matrix<Sym, Sym>(4, 1, floatArrayOf(0.1f, -0.1f, 0.0f, 5.0f))
        val s = a.sign()
        assertContentEquals(intArrayOf(4, 1), s.dims)
    }

    /**
     * §0.4.399 — `broadcastToLike`: the forward twin (and VJP) of `sumToLike`.
     * Right-aligned NumPy broadcast up to the template's RUNTIME dims; the
     * template contributes shape only. Equal-rank stretch, rank extension,
     * identity, and the incompatible-axis refusal.
     */
    @Test
    fun broadcastToLikeStretchesToTheTemplatesRuntimeShape() {
        val t = Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6))
        // Equal-rank stretch [1,3] → [2,3].
        val row = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(1f, 2f, 3f))
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 1f, 2f, 3f), broadcastToLike(row, t).hostF32())
        // Rank extension [3] → [2,3] (missing leading axis replicated).
        val vec = Tensors.f32Vector<Sym>(floatArrayOf(4f, 5f, 6f))
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 4f, 5f, 6f), broadcastToLike(vec, t).hostF32())
        // Identity fast path: same dims → copy.
        val full = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        assertContentEquals(full.hostF32(), broadcastToLike(full, t).hostF32())
        // An aligned axis that is neither equal nor 1 refuses loudly.
        val bad = Tensors.f32Matrix<Sym, Sym>(2, 2, FloatArray(4))
        assertFailsWith<IllegalArgumentException> { broadcastToLike(bad, t) }
    }

    /**
     * §0.4.399 — `sliceAtLike`: the reverse mirror (and VJP) of `padToLike`.
     * Window of the template's RUNTIME dims at literal offset `low`; the
     * template contributes shape only. Pins the padToLike ⇄ sliceAtLike
     * round trip: slicing back out what was padded in recovers the value.
     */
    @Test
    fun sliceAtLikeCutsTheWindowBackOut() {
        val v = Tensors.f32Matrix<Sym, Sym>(3, 4, FloatArray(12) { it.toFloat() })
        val t = Tensors.f32Matrix<Sym, Sym>(2, 2, FloatArray(4))
        assertContentEquals(floatArrayOf(5f, 6f, 9f, 10f), sliceAtLike(v, t, intArrayOf(1, 1)).hostF32())
        // Round trip with padToLike: pad a [2] into a [5] at low=1, slice it back.
        val u = Tensors.f32Vector<Sym>(floatArrayOf(7f, -3f))
        val big = Tensors.f32Vector<Sym>(FloatArray(5))
        val padded = padToLike(u, big, intArrayOf(1))
        assertContentEquals(floatArrayOf(0f, 7f, -3f, 0f, 0f), padded.hostF32())
        assertContentEquals(u.hostF32(), sliceAtLike(padded, u, intArrayOf(1)).hostF32())
        // A window that overruns the value refuses loudly.
        assertFailsWith<IllegalArgumentException> { sliceAtLike(u, big, intArrayOf(0)) }
    }

    @Test
    fun timesScalarMultipliesEachElement() {
        val a = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = a * 0.5f
        assertContentEquals(floatArrayOf(0.5f, 1f, 1.5f, 2f, 2.5f, 3f), b.hostF32())
        assertContentEquals(intArrayOf(2, 3), b.dims)
    }

    /**
     * §0.4.390 — training-mode batchNorm: per-channel statistics over the batch AND
     * spatial extents, BIASED variance, then the per-channel affine. Pinned here
     * because the plugin's `grad {}` path DESUGARS the call into primitives and so
     * never executes this function — it is the plain-runtime twin, and the two have
     * to agree by construction.
     *
     * Hand-computed, and each choice is load-bearing:
     * - channel 0 = [1,2,3,4]: μ = 2.5, biased ν = (1.5²+0.5²+0.5²+1.5²)/4 = 1.25.
     *   An UNBIASED variance (÷3) would give 1.6667 and a different 1/√(ν+eps), so
     *   this pins the convention rather than just the shape of the formula.
     * - channel 1 = [10,10,10,10]: ν = 0, so `eps` is the only thing keeping
     *   1/√(ν+eps) finite — every element lands on β.
     * - γ = [2,1] and β = [0,−1] are DISTINCT per channel, so a swapped
     *   scale/offset changes the answer.
     */
    @Test
    fun batchNormTrainingModeUsesBiasedVarianceAndPerChannelAffine() {
        val x = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(
            1, 2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 10f, 10f, 10f, 10f),
        )
        val gamma = Tensors.f32Vector<Sym>(floatArrayOf(2f, 1f))
        val beta = Tensors.f32Vector<Sym>(floatArrayOf(0f, -1f))
        val eps = 1e-2f
        val y = x.batchNorm(gamma, beta, eps)

        assertContentEquals(intArrayOf(1, 2, 2, 2), y.dims)
        val got = y.hostF32()
        val invStd0 = 1.0 / kotlin.math.sqrt(1.25 + eps.toDouble())
        floatArrayOf(1f, 2f, 3f, 4f).forEachIndexed { i, v ->
            assertEquals(
                ((v - 2.5f) * invStd0 * 2.0 + 0.0).toFloat(), got[i], 1e-5f, "channel 0 [$i]",
            )
        }
        // channel 1: (10−10)/√(0+eps) · 1 + (−1) = −1 everywhere.
        for (i in 0 until 4) assertEquals(-1f, got[4 + i], 1e-5f, "channel 1 [$i]")
    }

    @Test
    fun timesScalarZeroProducesZeroTensor() {
        // Use all-positive inputs to avoid `(-x) * 0 = -0` which assertContentEquals
        // treats as distinct from +0.
        val a = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(7f, 3f, 11f, 9f))
        val b = a * 0f
        assertContentEquals(FloatArray(4) { 0f }, b.hostF32())
    }
}

private typealias DTensorAlias = io.tlaloc.core.DTensor<Rank2<Sym, Sym>, io.tlaloc.core.F32>
