package io.tlaloc.nn

import io.tlaloc.core.F32
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * §0.4.438 — F2: the Dense/ReluLayer/Flatten gradient certification through
 * the F1 `valueAndGradients` route. Every value on the quarter-integer grid,
 * every relu mask unambiguous (no pre-relu zeros), every assertion `==`.
 *
 * Hand derivation for the Dense+Relu+Dense stack (L = Σ y², x [2,2]):
 *   z₁ = x·W₁ + b₁ = [[0.25, −1.75], [1.5, −2.375]]
 *   h  = relu(z₁)  = [[0.25, 0], [1.5, 0]]        (mask [[1,0],[1,0]])
 *   y  = h·W₂ + b₂ = [[0.25, 0.625], [2.75, 1.25]],  L = 9.578125
 *   ∂L/∂y = 2y = [[0.5, 1.25], [5.5, 2.5]]
 *   ∂L/∂b₂ = colsum(2y) = [6, 3.75];  ∂L/∂W₂ = hᵀ·2y
 *   ∂L/∂h = 2y·W₂ᵀ = [[1.625, 0.75], [12.25, −3]];  ∂L/∂z₁ = mask ⊙ ∂L/∂h
 *   ∂L/∂b₁ = colsum;  ∂L/∂W₁ = xᵀ·∂L/∂z₁;  ∂L/∂x[i,k] = Σⱼ ∂L/∂z₁[i,j]·W₁[k,j]
 */
class DenseStackGradientTest {

    private val x  = floatArrayOf(1f, -0.5f, 2f, 0.25f)          // [2, 2]
    private val w1 = floatArrayOf(0.5f, -1f, 1f, 0.5f)           // [2, 2]
    private val b1 = floatArrayOf(0.25f, -0.5f)                  // [2]
    private val w2 = floatArrayOf(2f, 0.5f, -1f, 1f)             // [2, 2]
    private val b2 = floatArrayOf(-0.25f, 0.5f)                  // [2]

    private fun dense1(activation: Activation = Activation.Identity) = Dense(
        Tensors.f32Matrix<Sym, Sym>(2, 2, w1),
        Tensors.f32Vector<Sym>(b1),
        activation,
    )

    private fun dense2() = Dense(
        Tensors.f32Matrix<Sym, Sym>(2, 2, w2),
        Tensors.f32Vector<Sym>(b2),
    )

    private fun input() = Tensors.f32Matrix<Sym, Sym>(2, 2, x)

    @Test
    fun denseReluDenseGradientsHandExactOnTheQuarterGrid() {
        val model = Sequential(dense1(), ReluLayer, dense2())
        val result = valueAndGradients(model, listOf(input())) { y -> (y * y).sum() }

        assertEquals(9.578125f, result.loss)
        assertEquals(listOf("0.w", "0.b", "2.w", "2.b"), result.gradients.keys.toList())

        assertContentEquals(floatArrayOf(26.125f, 0.0f, 2.25f, 0.0f), result.gradients["0.w"]!!.hostF32())
        assertContentEquals(floatArrayOf(13.875f, 0.0f), result.gradients["0.b"]!!.hostF32())
        assertContentEquals(floatArrayOf(8.375f, 4.0625f, 0.0f, 0.0f), result.gradients["2.w"]!!.hostF32())
        assertContentEquals(floatArrayOf(6.0f, 3.75f), result.gradients["2.b"]!!.hostF32())

        assertEquals(1, result.inputGradients.size)
        assertContentEquals(
            floatArrayOf(0.8125f, 1.625f, 6.125f, 12.25f),
            result.inputGradients[0].hostF32(),
        )
    }

    /**
     * The trace-vs-hand-built-DXIR oracle on ONE Dense with a composed Relu
     * activation: the same MATMUL → BROADCAST(b, dims=[1]) → ADD → RELU →
     * MUL → SUM graph built through the `:nn` capture and directly through
     * [DxirBuilder], both put through the SAME reverse transform, must agree
     * elementwise exactly — and both against the hand derivation. This is the
     * strongest available pin that Dense's TRACE spelling (matmul + row
     * broadcast + relu) reproduces the canonical graph.
     */
    @Test
    fun tracedDenseMatchesHandBuiltDxirThroughTheSameTransform() {
        val captured = capture(dense1(Activation.Relu), listOf(input())) { y -> (y * y).sum() }
        assertEquals(3, captured.primal.params.size) // x, w, b — inputs first, then key order

        val m22 = DxirType(F32, listOf(2, 2))
        val v2 = DxirType(F32, listOf(2))
        val hand = DxirBuilder.function("denseHand") {
            val px = param("x", m22)
            val pw = param("w", m22)
            val pb = param("b", v2)
            val mm = op(OpKind.MATMUL, listOf(px, pw), m22)
            val bb = op(
                OpKind.BROADCAST, listOf(pb), m22,
                attrs = mapOf("broadcast_dimensions" to listOf(1)),
            )
            val z = op(OpKind.ADD, listOf(mm, bb), m22)
            val y = op(OpKind.RELU, listOf(z), m22)
            val sq = op(OpKind.MUL, listOf(y, y), m22)
            val loss = op(OpKind.SUM, listOf(sq), DxirType(F32, emptyList()))
            listOf(loss)
        }

        val values = listOf(x, w1, b1)
        val fromTrace = DxirInterpreter.evalFunction(DxirReverseTransform.apply(captured.primal), values)
        val fromHand = DxirInterpreter.evalFunction(DxirReverseTransform.apply(hand), values)

        assertEquals(fromHand.size, fromTrace.size)
        for (i in fromHand.indices) assertContentEquals(fromHand[i], fromTrace[i])

        // Hand math: z = [[0.25, −1.75], [1.5, −2.375]], y = relu(z), L = 2.3125,
        // ∂L/∂z = 2y ⊙ mask = [[0.5, 0], [3, 0]].
        assertContentEquals(floatArrayOf(0.25f, 0.5f, 1.5f, 3.0f), fromTrace[0]) // ∂L/∂x
        assertContentEquals(floatArrayOf(6.5f, 0.0f, 0.5f, 0.0f), fromTrace[1])  // ∂L/∂w = xᵀ·∂L/∂z
        assertContentEquals(floatArrayOf(3.5f, 0.0f), fromTrace[2])              // ∂L/∂b = colsum
    }

    /**
     * The F2 RESHAPE spelling through the transform: Flatten alone on a
     * rank-3 input. ∂(Σy²)/∂x = 2x, handed back RESHAPED to the input's own
     * rank-3 dims by ReshapeRule — the gradient's SHAPE is the assertion that
     * the reverse reshape happened.
     */
    @Test
    fun flattenGradientComesBackInTheInputShape() {
        val x3 = Tensors.f32Tensor3<Sym, Sym, Sym>(2, 1, 2, floatArrayOf(1f, -0.5f, 2f, 0.25f))
        val result = valueAndGradients(Sequential(Flatten), listOf(x3)) { y -> (y * y).sum() }

        assertEquals(5.3125f, result.loss)
        val dx = result.inputGradients[0]
        assertContentEquals(intArrayOf(2, 1, 2), dx.dims)
        assertContentEquals(floatArrayOf(2f, -1f, 4f, 0.5f), dx.hostF32())
    }

    /**
     * Flatten feeding Dense — the reshape chained into the matmul, gradients
     * hand-exact end to end. x [2,1,2] flattens to the same [2,2] the stack
     * test uses; L = Σ(x·W₁ + b₁) gives ∂L/∂z = 1s, so ∂L/∂b = [2,2],
     * ∂L/∂W = xᵀ·1s (= column sums of x per row), and ∂L/∂x row k of W summed.
     */
    @Test
    fun flattenIntoDenseGradientsHandExact() {
        val x3 = Tensors.f32Tensor3<Sym, Sym, Sym>(2, 1, 2, x)
        val model = Sequential(Flatten, dense1())
        val result = valueAndGradients(model, listOf(x3)) { y -> y.sum() }

        // z₁ = [[0.25, −1.75], [1.5, −2.375]] → L = −2.375
        assertEquals(-2.375f, result.loss)
        assertEquals(listOf("1.w", "1.b"), result.gradients.keys.toList())
        assertContentEquals(floatArrayOf(3f, 3f, -0.25f, -0.25f), result.gradients["1.w"]!!.hostF32())
        assertContentEquals(floatArrayOf(2f, 2f), result.gradients["1.b"]!!.hostF32())

        val dx = result.inputGradients[0]
        assertContentEquals(intArrayOf(2, 1, 2), dx.dims)
        assertContentEquals(floatArrayOf(-0.5f, 1.5f, -0.5f, 1.5f), dx.hostF32())
    }
}
