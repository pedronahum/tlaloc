package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Frozen parameters: [capture] with `frozen` returns gradients for the
 * trained parameters only, equal to the ones a full capture computes; the
 * gradient function does less work; and [Optimizer.step] with `frozen`
 * leaves the frozen tensors untouched and keeps no state for them.
 */
class FrozenParametersTest {

    private val n = 8
    private val d = 4

    private val xv = uniformFloats(RandomKey.fromSeed(31), n * d).map { 2f * it - 1f }.toFloatArray()
    private val yv = FloatArray(n) { i -> xv[i * d] * xv[i * d + 1] - 0.5f * xv[i * d + 2] }
    private val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)

    private fun model(): Sequential {
        val keys = RandomKey.fromSeed(5).split(3)
        return Sequential(Dense(d, 6, keys[0]), ReluLayer, Dense(6, 6, keys[1]), ReluLayer, Dense(6, 1, keys[2]))
    }

    private fun mse(y: io.tlaloc.autograd.Tracer<io.tlaloc.core.Shape>): io.tlaloc.autograd.Tracer<*> {
        val diff = y - y.constant<io.tlaloc.core.Shape>(yv, intArrayOf(n, 1))
        return (diff * diff).mean()
    }

    private fun dotCount(step: CapturedStep): Int =
        step.gradient.body.count { it is DxirOp && it.op == OpKind.MATMUL }

    @Test
    fun frozenCaptureGivesTheFullCapturesGradientsForTheTrainedParameters() {
        val m = model()
        val full = capture(m, listOf(x)) { mse(it) }.run(m, listOf(x))
        val frozen = Frozen.prefixes("0", "2.w")
        val step = capture(m, listOf(x), frozen) { mse(it) }
        val part = step.run(m, listOf(x))

        assertEquals(listOf("2.b", "4.w", "4.b"), step.parameterKeys)
        assertEquals(listOf("0.w", "0.b", "2.w"), step.frozenKeys)
        assertEquals(part.gradients.keys, setOf("2.b", "4.w", "4.b"))
        assertEquals(full.loss, part.loss)
        for ((key, g) in part.gradients) {
            assertContentEquals(full.gradients.getValue(key).hostF32(), g.hostF32(), key)
        }
        // The gradient function returns (loss, dx, one per trained parameter).
        assertEquals(1 + 1 + 3, step.gradient.returns.size)
    }

    @Test
    fun freezingRemovesTheAdjointWorkOfTheFrozenWeights() {
        val m = model()
        val full = capture(m, listOf(x)) { mse(it) }
        // Freezing the first layer drops its weight gradient xᵀ·dy, and the
        // input gradient (still returned) is the only dot left for it.
        val first = capture(m, listOf(x), Frozen.prefixes("0")) { mse(it) }
        // Freezing every weight leaves the bias gradients, which need the
        // backward dots through layers 2 and 4 but no xᵀ·dy product.
        val weights = capture(m, listOf(x), Frozen.matching { it.endsWith(".w") }) { mse(it) }
        println("[frozen] MATMUL in the gradient function: full ${dotCount(full)}, " +
            "layer 0 frozen ${dotCount(first)}, all weights frozen ${dotCount(weights)}")
        assertTrue(dotCount(first) < dotCount(full), "freezing layer 0 must remove its weight-gradient dot")
        assertTrue(dotCount(weights) < dotCount(first), "freezing every weight must remove every weight-gradient dot")
        assertTrue(first.gradient.body.size < full.gradient.body.size)
    }

    @Test
    fun noneIsTheOrdinaryCapture() {
        val m = model()
        val plain = capture(m, listOf(x)) { mse(it) }
        val none = capture(m, listOf(x), Frozen.NONE) { mse(it) }
        assertEquals(plain.parameterKeys, none.parameterKeys)
        assertEquals(emptyList(), none.frozenKeys)
        assertEquals(plain.gradient.body.size, none.gradient.body.size)
        val a = plain.run(m, listOf(x))
        val b = none.run(m, listOf(x))
        for ((key, g) in a.gradients) assertContentEquals(g.hostF32(), b.gradients.getValue(key).hostF32(), key)
    }

    @Test
    fun trainingLeavesFrozenTensorsBitIdenticalAndKeepsNoStateForThem() {
        var m = model()
        val before = m.parameters.associate { it.key to it.tensor.hostF32().copyOf() }
        val frozen = Frozen.prefixes("0", "2")
        val step = capture(m, listOf(x), frozen) { mse(it) }
        val opt = AdamW(learningRate = 0.05f, weightDecay = 0.1f)
        var state = opt.initialState()
        val losses = ArrayList<Float>()
        repeat(30) {
            val r = step.run(m, listOf(x))
            losses += r.loss
            val (next, s) = opt.step(m, r.gradients, state, frozen)
            m = next
            state = s
        }
        assertTrue(losses.last() < losses.first(), "the trained layer must reduce the loss: $losses")
        for (p in m.parameters) {
            if (frozen.isFrozen(p.key)) {
                assertContentEquals(before.getValue(p.key), p.tensor.hostF32(), "frozen ${p.key} changed")
            } else {
                assertFalse(before.getValue(p.key).contentEquals(p.tensor.hostF32()), "trained ${p.key} did not change")
            }
        }
        assertEquals(setOf("4.w", "4.b"), state.m.keys)
        assertEquals(setOf("4.w", "4.b"), state.v.keys)
    }

    @Test
    fun frozenTensorsAreTheSameObjectsAfterAStep() {
        val m = model()
        val frozen = Frozen.prefixes("0")
        val r = capture(m, listOf(x), frozen) { mse(it) }.run(m, listOf(x))
        val (next, _) = SGD(0.1f).step(m, r.gradients, SGD(0.1f).initialState(), frozen)
        assertSame((m.layers[0] as Dense).w, (next.layers[0] as Dense).w)
        assertSame((m.layers[0] as Dense).b, (next.layers[0] as Dense).b)
    }

    @Test
    fun aGradientForAFrozenParameterIsRefused() {
        val m = model()
        val r = capture(m, listOf(x)) { mse(it) }.run(m, listOf(x))
        val e = assertFailsWith<IllegalArgumentException> {
            SGD(0.1f).step(m, r.gradients, SGD(0.1f).initialState(), Frozen.prefixes("0"))
        }
        assertTrue("frozen" in e.message!!, e.message)
    }

    @Test
    fun aModelWhoseStructureChangedIsRefused() {
        val m = model()
        val step = capture(m, listOf(x), Frozen.prefixes("0")) { mse(it) }
        val other = Sequential(m.layers.take(3))
        val e = assertFailsWith<IllegalArgumentException> { step.run(other, listOf(x)) }
        assertTrue("re-capture" in e.message!!, e.message)
    }

    @Test
    fun prefixesMatchWholePathSegments() {
        val f = Frozen.prefixes("blocks.1", "embed.")
        assertTrue(f.isFrozen("blocks.1"))
        assertTrue(f.isFrozen("blocks.1.attn.q.w"))
        assertFalse(f.isFrozen("blocks.10.attn.q.w"))
        assertTrue(f.isFrozen("embed.table"))
        assertFalse(f.isFrozen("embedding.table"))
        assertTrue((Frozen.keys("a") + Frozen.keys("b")).isFrozen("b"))
        assertFalse(Frozen.allExcept { it.startsWith("lora") }.isFrozen("lora_A"))
        assertFailsWith<IllegalArgumentException> { Frozen.prefixes("") }
    }

    @Test
    fun aKeyOrPrefixThatSelectsNothingIsRefused() {
        val m = model()
        val e = assertFailsWith<IllegalArgumentException> { capture(m, listOf(x), Frozen.prefixes("0", "9")) { mse(it) } }
        assertTrue("prefix '9'" in e.message!! && "select no parameter" in e.message!!, e.message)
        assertFailsWith<IllegalArgumentException> { capture(m, listOf(x), Frozen.keys("0.w", "0.weight")) { mse(it) } }
        val r = capture(m, listOf(x), Frozen.prefixes("0")) { mse(it) }.run(m, listOf(x))
        assertFailsWith<IllegalArgumentException> {
            SGD(0.1f).step(m, r.gradients, SGD(0.1f).initialState(), Frozen.prefixes("0") + Frozen.keys("4.bias"))
        }
        assertEquals(listOf("prefix '9'"), Frozen.prefixes("0", "9").unmatched(m.parameters.map { it.key }))
        // A predicate selects what it selects; an empty selection is allowed there.
        capture(m, listOf(x), Frozen.matching { it.startsWith("none") }) { mse(it) }
    }
}
