package io.tlaloc.nn

import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.440 — F4: the conv-stack certification. Quarter-integer grids
 * throughout, tie-free maxpool windows (F0 landmine 6), every hand assertion
 * `==` — the arithmetic is dyadic end to end. The trace-vs-hand-built-DXIR
 * oracle is the strongest pin: the same conv net built through the `:nn`
 * capture and directly through [DxirBuilder], both through the SAME reverse
 * transform, elementwise equal.
 */
class ConvStackGradientTest {

    private val w22 = floatArrayOf(0.5f, -1.0f, 0.25f, 1.5f) // [1, 1, 2, 2] OIHW

    private fun filter22() = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 2, 2, w22)

    // ---------------------------------------------------------------- 1×1×3×3

    private val x33 = floatArrayOf(
        0.25f, 0.5f, -0.75f,
        1.0f, -1.25f, 1.5f,
        2.0f, -0.25f, 0.75f,
    )

    private fun input33() = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 3, 3, x33)

    /**
     * Hand-exact 1×1×3×3 valid conv through `valueAndGradients`, L = Σy:
     *   y[i,j] = Σ_{p,q} X[i+p, j+q]·W[p,q]  (stride 1, no padding, y is 2×2)
     *   L = 1.75;  ∂L/∂W[p,q] = Σ_{i,j} X[i+p, j+q]  (the four 2×2 block sums)
     *   ∂L/∂X[i,j] = Σ_{windows covering (i,j)} W[p,q]  (corner→1 tap,
     *   edge→2 taps, center→all four).
     */
    @Test
    fun conv1x1x3x3GradientHandExactThroughValueAndGradients() {
        val model = Conv2d(filter22())
        val result = valueAndGradients(model, listOf(input33())) { y -> y.sum() }

        assertEquals(1.75f, result.loss)
        assertEquals(listOf("filter"), result.gradients.keys.toList())
        assertContentEquals(
            floatArrayOf(0.5f, 0.0f, 1.5f, 0.75f),
            result.gradients["filter"]!!.hostF32(),
        )
        val dx = result.inputGradients[0]
        assertContentEquals(intArrayOf(1, 1, 3, 3), dx.dims)
        assertContentEquals(
            floatArrayOf(
                0.5f, -0.5f, -1.0f,
                0.75f, 1.25f, 0.5f,
                0.25f, 1.75f, 1.5f,
            ),
            dx.hostF32(),
        )
    }

    // ------------------------------------------------- the trace-vs-DXIR pin

    private val x44 = floatArrayOf(
        0.25f, -0.5f, 1.0f, 0.75f,
        1.5f, 2.0f, -1.25f, 0.5f,
        -0.75f, 1.25f, 0.25f, -0.25f,
        0.5f, -1.0f, 1.75f, 2.25f,
    )

    private fun input44() = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 1, 4, 4, x44)

    private fun convNet() = Sequential(
        Conv2d(filter22(), hStride = 2, vStride = 2, activation = Activation.Relu),
        MaxPool2d(2, 2),
    )

    /**
     * The trace-vs-hand-built-DXIR oracle on the conv net CONV2D(stride 2) →
     * RELU → MAXPOOL2D → SUM: built through the `:nn` capture AND directly
     * through [DxirBuilder] with the interpreter's exact attr spellings, both
     * through the SAME reverse transform, elementwise equal — the strongest
     * available pin that the F4 TRACE spellings reproduce the canonical graph.
     *
     * Hand forward: conv windows give z = [4.0, 0.1875, −3.0, 4.1875] (every
     * pre-relu value nonzero — unambiguous mask), relu zeroes the −3.0, the
     * pool's max 4.1875 is tie-free, L = 4.1875.
     */
    @Test
    fun tracedConvNetMatchesHandBuiltDxirThroughTheSameTransform() {
        val captured = capture(convNet(), listOf(input44())) { y -> y.sum() }
        assertEquals(2, captured.primal.params.size) // x, then "0.filter"

        val tx = DxirType(F32, listOf(1, 1, 4, 4))
        val tf = DxirType(F32, listOf(1, 1, 2, 2))
        val tz = DxirType(F32, listOf(1, 1, 2, 2))
        val tp = DxirType(F32, listOf(1, 1, 1, 1))
        val zeroPad = listOf(listOf(0, 0), listOf(0, 0))
        val hand = DxirBuilder.function("convNetHand") {
            val px = param("x", tx)
            val pf = param("f", tf)
            val conv = op(
                OpKind.CONV2D, listOf(px, pf), tz,
                attrs = mapOf("window_strides" to listOf(2, 2), "padding" to zeroPad),
            )
            val r = op(OpKind.RELU, listOf(conv), tz)
            val pool = op(
                OpKind.MAXPOOL2D, listOf(r), tp,
                attrs = mapOf(
                    "window" to listOf(2, 2),
                    "window_strides" to listOf(2, 2),
                    "padding" to zeroPad,
                ),
            )
            listOf(op(OpKind.SUM, listOf(pool), DxirType(F32, emptyList())))
        }

        val values = listOf(x44, w22)
        val fromTrace = DxirInterpreter.evalFunction(DxirReverseTransform.apply(captured.primal), values)
        val fromHand = DxirInterpreter.evalFunction(DxirReverseTransform.apply(hand), values)

        assertEquals(fromHand.size, fromTrace.size)
        for (i in fromHand.indices) assertContentEquals(fromHand[i], fromTrace[i])
    }

    /**
     * The same net's gradients hand-exact through `valueAndGradients`: only
     * the pool-winning conv window (the bottom-right stride-2 window, z =
     * 4.1875) carries gradient, so ∂L/∂W = that window's x taps and ∂L/∂x = W
     * scattered onto those four positions, zeros elsewhere.
     */
    @Test
    fun convNetGradientsHandExact() {
        val result = valueAndGradients(convNet(), listOf(input44())) { y -> y.sum() }

        assertEquals(4.1875f, result.loss)
        assertContentEquals(
            floatArrayOf(0.25f, -0.25f, 1.75f, 2.25f),
            result.gradients["0.filter"]!!.hostF32(),
        )
        assertContentEquals(
            floatArrayOf(
                0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f,
                0f, 0f, 0.5f, -1.0f,
                0f, 0f, 0.25f, 1.5f,
            ),
            result.inputGradients[0].hostF32(),
        )
    }

    // ------------------------------------------------------------- pooling

    /**
     * MaxPool2d on the shuffled well-separated grid (all 16 values distinct —
     * tie-free per F0 landmine 6), L = Σy²: each window's argmax receives
     * 2·max, everything else exactly zero.
     */
    @Test
    fun maxPoolOnTheShuffledWellSeparatedGrid() {
        val x = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(
            1, 1, 4, 4,
            floatArrayOf(
                0.25f, 3.5f, -1.75f, 2.0f,
                1.5f, -0.5f, 0.75f, -3.25f,
                4.25f, -2.5f, 1.25f, 5.5f,
                -0.25f, 2.75f, -4.5f, 3.25f,
            ),
        )
        val result = valueAndGradients(Sequential(MaxPool2d(2, 2)), listOf(x)) { y -> (y * y).sum() }

        // y = [3.5, 2.0, 4.25, 5.5]; L = 12.25 + 4 + 18.0625 + 30.25
        assertEquals(64.5625f, result.loss)
        assertTrue(result.gradients.isEmpty()) // pooling is not trainable
        assertContentEquals(
            floatArrayOf(
                0f, 7f, 0f, 4f,
                0f, 0f, 0f, 0f,
                8.5f, 0f, 0f, 11f,
                0f, 0f, 0f, 0f,
            ),
            result.inputGradients[0].hostF32(),
        )
    }

    /**
     * AvgPool2d hand-exact over two channels, L = Σy²: y is the per-channel
     * window mean, so ∂L/∂x = 2y/4 splat uniformly over each channel.
     */
    @Test
    fun avgPoolGradientHandExactOverTwoChannels() {
        val x = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(
            1, 2, 2, 2,
            floatArrayOf(
                1.0f, 0.5f, -0.25f, 0.75f, // ch0 → mean 0.5
                2.0f, -1.0f, 0.5f, 1.5f, // ch1 → mean 0.75
            ),
        )
        val result = valueAndGradients(Sequential(AvgPool2d(2, 2)), listOf(x)) { y -> (y * y).sum() }

        assertEquals(0.8125f, result.loss) // 0.25 + 0.5625
        assertContentEquals(
            floatArrayOf(
                0.25f, 0.25f, 0.25f, 0.25f,
                0.375f, 0.375f, 0.375f, 0.375f,
            ),
            result.inputGradients[0].hostF32(),
        )
    }

    /** DiffKT's own divisibility require survives at the trace spelling. */
    @Test
    fun poolRefusesIndivisibleSpatialDims() {
        assertFailsWith<IllegalArgumentException> {
            valueAndGradients(Sequential(MaxPool2d(2, 2)), listOf(input33())) { y -> y.sum() }
        }
    }

    // -------------------------------------------------------- Same padding

    /**
     * DiffKT's TF-SAME formula, pinned per axis independent of any conv:
     * `total = in % stride == 0 ? max(k − stride, 0) : max(k − in % stride, 0)`,
     * split before = total/2, after = total − before (the odd unit lands AFTER).
     */
    @Test
    fun samePaddingFormulaPinned() {
        assertEquals(0 to 1, samePadding(3, 2, 1))
        assertEquals(1 to 1, samePadding(4, 3, 1))
        assertEquals(0 to 0, samePadding(4, 2, 2))
        assertEquals(0 to 1, samePadding(5, 2, 2))
        assertEquals(1 to 1, samePadding(5, 3, 2))
        assertEquals(0 to 0, samePadding(7, 2, 4)) // k < in % stride → no pad
        assertEquals(1 to 2, samePadding(6, 5, 2)) // odd total, after-heavy
    }

    /**
     * Conv2d with `PaddingStyle.Same` at stride 1 (pad bottom/right by 1 for a
     * 2×2 kernel — output extent = input extent), L = Σy, all hand-exact:
     * ∂L/∂W[p,q] = Σ of x over the region shifted by (p,q); ∂L/∂x[i,j] =
     * Σ_{p≤min(i,1), q≤min(j,1)} W[p,q].
     */
    @Test
    fun samePaddingConvGradientHandExact() {
        val model = Conv2dWithSamePadding(filter22())
        val result = valueAndGradients(model, listOf(input33())) { y -> y.sum() }

        assertEquals(3.4375f, result.loss)
        assertContentEquals(
            floatArrayOf(3.75f, 0.5f, 3.75f, 0.75f),
            result.gradients["filter"]!!.hostF32(),
        )
        val dx = result.inputGradients[0]
        assertContentEquals(intArrayOf(1, 1, 3, 3), dx.dims) // Same: dX in the input's own extent
        assertContentEquals(
            floatArrayOf(
                0.5f, -0.5f, -0.5f,
                0.75f, 1.25f, 1.25f,
                0.75f, 1.25f, 1.25f,
            ),
            dx.hostF32(),
        )
    }

    // ------------------------------------------------------- layer plumbing

    /**
     * The companion draws the filter `kaimingUniform(FanIn, LeakyRelu(√5))` —
     * DiffKT's conv default — under the F2 key discipline (`split(1)[0]`),
     * pinned bit-exact against the initializer called directly.
     */
    @Test
    fun convCompanionDrawsTheDiffKtDefaultUnderTheKeyDiscipline(): Unit {
        val key = RandomKey.fromSeed(7L)
        val layer = Conv2d(intArrayOf(2, 3, 2, 2), key, hStride = 2, vStride = 1)
        val expected = kaimingUniformInit(
            key.split(1)[0],
            intArrayOf(2, 3, 2, 2),
            FanMode.FanIn,
            ActivationGain.LeakyRelu(sqrt(5f)),
        )
        assertContentEquals(intArrayOf(2, 3, 2, 2), layer.filter.dims)
        assertContentEquals(expected.hostF32(), layer.filter.hostF32())
        assertEquals(2, layer.hStride)
        assertEquals(1, layer.vStride)
    }

    @Test
    fun convWithParametersRebuildsFunctionally() {
        val layer = Conv2d(filter22(), hStride = 2, vStride = 2, paddingStyle = PaddingStyle.Same)
        val replacement = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(
            1, 1, 2, 2, floatArrayOf(1f, 2f, 3f, 4f),
        )
        val rebuilt = layer.withParameters(mapOf("filter" to replacement))
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), rebuilt.filter.hostF32())
        assertContentEquals(w22, layer.filter.hostF32()) // the original is untouched
        assertEquals(2, rebuilt.hStride)
        assertTrue(rebuilt.paddingStyle is PaddingStyle.Same)
        assertFailsWith<IllegalArgumentException> {
            layer.withParameters(mapOf("weight" to replacement))
        }
    }
}
