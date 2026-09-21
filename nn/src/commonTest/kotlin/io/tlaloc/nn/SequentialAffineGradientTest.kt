package io.tlaloc.nn

import io.tlaloc.core.F32
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import kotlin.test.assertTrue

/**
 * §0.4.437 — F1: the compiler-route loop proven on a 2-layer Sequential of
 * AffineTransforms. Every value is on the quarter-integer grid, so every
 * hand-derived gradient is exact in f32 and asserted with `==`, not tolerance.
 *
 * Hand derivation (elementwise throughout; L = Σᵢ yᵢ²):
 *   h = m₁·x + b₁, y = m₂·h + b₂
 *   ∂L/∂y = 2y;  ∂L/∂b₂ = 2y;  ∂L/∂m₂ = 2y·h
 *   ∂L/∂h = 2y·m₂;  ∂L/∂b₁ = ∂L/∂h;  ∂L/∂m₁ = ∂L/∂h·x;  ∂L/∂x = ∂L/∂h·m₁
 */
class SequentialAffineGradientTest {

    private val x  = floatArrayOf(1.0f, -0.5f, 2.0f, 0.25f)
    private val m1 = floatArrayOf(0.5f, 1.0f, -0.25f, 2.0f)
    private val b1 = floatArrayOf(0.25f, -1.0f, 0.5f, 0.0f)
    private val m2 = floatArrayOf(2.0f, 0.5f, 1.0f, -0.5f)
    private val b2 = floatArrayOf(-0.25f, 0.5f, 0.0f, 1.0f)

    private fun model() = Sequential(
        AffineTransform(Tensors.f32Vector<Sym>(m1), Tensors.f32Vector<Sym>(b1)),
        AffineTransform(Tensors.f32Vector<Sym>(m2), Tensors.f32Vector<Sym>(b2)),
    )

    private fun input() = Tensors.f32Vector<Sym>(x)

    @Test
    fun twoLayerAffineGradientsHandExactOnTheQuarterGrid() {
        val result = valueAndGradients(model(), listOf(input())) { y -> (y * y).sum() }

        // h = [0.75, -1.5, 0.0, 0.5]; y = [1.25, -0.25, 0.0, 0.75]
        assertEquals(2.1875f, result.loss)

        assertEquals(listOf("0.m", "0.b", "1.m", "1.b"), result.gradients.keys.toList())
        assertContentEquals(floatArrayOf(5.0f, 0.125f, 0.0f, -0.1875f), result.gradients["0.m"]!!.hostF32())
        assertContentEquals(floatArrayOf(5.0f, -0.25f, 0.0f, -0.75f), result.gradients["0.b"]!!.hostF32())
        assertContentEquals(floatArrayOf(1.875f, 0.75f, 0.0f, 0.75f), result.gradients["1.m"]!!.hostF32())
        assertContentEquals(floatArrayOf(2.5f, -0.5f, 0.0f, 1.5f), result.gradients["1.b"]!!.hostF32())

        assertEquals(1, result.inputGradients.size)
        // Index 2 is −0.0 bit-exactly: ∂L/∂x₂ = (2y₂)·m₁₂ = 0.0 · (−0.25).
        assertContentEquals(floatArrayOf(2.5f, -0.25f, -0.0f, -1.5f), result.inputGradients[0].hostF32())
    }

    /**
     * The ROUTE PIN — the arbitrary-arity claim made concrete. The captured
     * primal is a real [DxirFunction] with one param per (input + parameter):
     * 5 here, one past the `grad {}` intrinsic surface's 4-arity ceiling,
     * which the IR never had.
     */
    @Test
    fun capturedPrimalIsARealDxirFunctionWithOneParamPerInputAndParameter() {
        val captured = capture(model(), listOf(input())) { y -> (y * y).sum() }

        val primal: DxirFunction = captured.primal
        assertEquals(1 + 4, primal.params.size)
        for (p in primal.params) assertEquals(listOf(4), p.type.dims)
        assertEquals(1, primal.returns.size)

        // includeForward = true: (loss, dInput, dParams...) — 1 + 5 returns.
        assertEquals(6, captured.gradient.returns.size)
        assertEquals(listOf("0.m", "0.b", "1.m", "1.b"), captured.parameterKeys)
        assertEquals(1, captured.inputCount)
    }

    /**
     * The trace-vs-hand-built-DXIR oracle — the strongest one available: the
     * same single-affine graph built through the `:nn` capture and directly
     * through [DxirBuilder], both put through the SAME reverse transform, must
     * agree ELEMENTWISE EXACTLY — and both against the hand derivation.
     */
    @Test
    fun tracedGraphMatchesHandBuiltDxirThroughTheSameTransform() {
        val affine = AffineTransform(Tensors.f32Vector<Sym>(m1), Tensors.f32Vector<Sym>(b1))
        val captured = capture(affine, listOf(input())) { y -> (y * y).sum() }

        val v4 = DxirType(F32, listOf(4))
        val hand = DxirBuilder.function("affineHand") {
            val px = param("x", v4)
            val pm = param("m", v4)
            val pb = param("b", v4)
            val mx = op(OpKind.MUL, listOf(pm, px), v4)
            val y = op(OpKind.ADD, listOf(mx, pb), v4)
            val sq = op(OpKind.MUL, listOf(y, y), v4)
            val loss = op(OpKind.SUM, listOf(sq), DxirType(F32, emptyList()))
            listOf(loss)
        }

        assertEquals(hand.params.size, captured.primal.params.size)

        val values = listOf(x, m1, b1)
        val fromTrace = DxirInterpreter.evalFunction(DxirReverseTransform.apply(captured.primal), values)
        val fromHand = DxirInterpreter.evalFunction(DxirReverseTransform.apply(hand), values)

        assertEquals(fromHand.size, fromTrace.size)
        for (i in fromHand.indices) assertContentEquals(fromHand[i], fromTrace[i])

        // Both routes against the hand math: y = [0.75, -1.5, 0.0, 0.5], 2y = [1.5, -3, 0, 1].
        // Index 2 is −0.0 bit-exactly: (2y₂)·m₂ = 0.0 · (−0.25).
        assertContentEquals(floatArrayOf(0.75f, -3.0f, -0.0f, 2.0f), fromTrace[0]) // ∂L/∂x = 2y·m
        assertContentEquals(floatArrayOf(1.5f, 1.5f, 0.0f, 0.25f), fromTrace[1])  // ∂L/∂m = 2y·x
        assertContentEquals(floatArrayOf(1.5f, -3.0f, 0.0f, 1.0f), fromTrace[2])  // ∂L/∂b = 2y
    }

    /**
     * The caching contract's F1 half: a [CapturedStep] re-binds VALUES per
     * call — updated parameter tensors flow through the SAME captured pair
     * without retracing, as long as the structure is unchanged.
     */
    @Test
    fun capturedStepReBindsUpdatedParametersWithoutRetracing() {
        val captured = capture(model(), listOf(input())) { y -> (y * y).sum() }

        val first = captured.run(model(), listOf(input()))
        assertEquals(2.1875f, first.loss)

        // b₂ → 0: y = m₂·h = [1.5, -0.75, 0.0, -0.25], L = 2.875, ∂L/∂b₂ = 2y.
        val updated = model().withParameters(
            mapOf("1.b" to Tensors.f32Vector<Sym>(floatArrayOf(0f, 0f, 0f, 0f))),
        )
        val second = captured.run(updated, listOf(input()))
        assertEquals(2.875f, second.loss)
        assertContentEquals(floatArrayOf(3.0f, -1.5f, 0.0f, -0.5f), second.gradients["1.b"]!!.hostF32())

        // Structure change is refused, not silently mis-bound.
        val smaller = Sequential(
            AffineTransform(Tensors.f32Vector<Sym>(m1), Tensors.f32Vector<Sym>(b1)),
        )
        assertFailsWith<IllegalArgumentException> { captured.run(smaller, listOf(input())) }
    }

    @Test
    fun componentsAreImmutableValuesAndRebuildFunctionally() {
        val original = model()
        val rebuilt = original.withParameters(
            mapOf("0.m" to Tensors.f32Vector<Sym>(floatArrayOf(1f, 1f, 1f, 1f))),
        )
        assertNotSame(original, rebuilt)
        // The original still holds its own tensor…
        assertContentEquals(m1, original.parameters[0].tensor.hostF32())
        // …the rebuilt one the update, with every other parameter carried over.
        assertContentEquals(floatArrayOf(1f, 1f, 1f, 1f), rebuilt.parameters[0].tensor.hostF32())
        assertContentEquals(b1, rebuilt.parameters[1].tensor.hostF32())
        assertEquals(
            original.parameters.map { it.key },
            rebuilt.parameters.map { it.key },
        )

        assertFailsWith<IllegalArgumentException> {
            original.withParameters(mapOf("2.m" to Tensors.f32Vector<Sym>(floatArrayOf(1f))))
        }
        assertFailsWith<IllegalArgumentException> {
            AffineTransform(Tensors.f32Vector<Sym>(m1), Tensors.f32Vector<Sym>(b1))
                .withParameters(mapOf("nope" to Tensors.f32Vector<Sym>(floatArrayOf(1f))))
        }
        assertTrue(original.parameters.map { it.key } == listOf("0.m", "0.b", "1.m", "1.b"))
    }
}
