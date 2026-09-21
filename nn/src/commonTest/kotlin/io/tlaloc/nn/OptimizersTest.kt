package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.439 — F3: the optimizer certification. Hand-stepped references on the
 * 2-param toy (w [2] = [1, −2], b [1] = [0.5]) over THREE steps with distinct
 * quarter-grid gradients per step, so every state slot (velocity, mean-square,
 * m/v) is exercised past its first-visit seeding. Exact `==` where the
 * arithmetic is exact (FixedLearningRate throughout, SGD momentum throughout,
 * every first-visit RMSprop step — √g² = |g| is exact on the grid); 1e-6
 * against inline DOUBLE-precision hand math where sqrt/div enter (the double
 * spelling is an independent reference — it shares no code with the Float
 * implementation). The integration cert closes the loop: F1's
 * `valueAndGradients` + SGD drives a fixed-key Dense model's loss strictly
 * down over 10 deterministic steps.
 */
class OptimizersTest {

    private fun toyParams(w: FloatArray, b: FloatArray): List<NamedParameter> = listOf(
        NamedParameter("w", Tensors.f32Vector<Sym>(w)),
        NamedParameter("b", Tensors.f32Vector<Sym>(b)),
    )

    private fun grads(gw: FloatArray, gb: FloatArray): Map<String, DTensor<*, F32>> = mapOf(
        "w" to Tensors.f32Vector<Sym>(gw),
        "b" to Tensors.f32Vector<Sym>(gb),
    )

    // The three per-step gradient pairs, quarter grid throughout.
    private val gw1 = floatArrayOf(0.5f, -1.0f);  private val gb1 = floatArrayOf(1.0f)
    private val gw2 = floatArrayOf(0.25f, 0.5f);  private val gb2 = floatArrayOf(-0.5f)
    private val gw3 = floatArrayOf(-0.5f, 0.25f); private val gb3 = floatArrayOf(0.25f)

    private val w0 = floatArrayOf(1.0f, -2.0f)
    private val b0 = floatArrayOf(0.5f)

    private fun assertClose(expected: Double, actual: Float, tag: String) {
        assertTrue(abs(expected - actual) < 1e-6, "$tag: expected ~$expected, got $actual")
    }

    // -----------------------------------------------------------------------
    // FixedLearningRate: t − α·g, stateless, exact on the quarter grid.
    // -----------------------------------------------------------------------

    @Test
    fun fixedLearningRateThreeStepsHandExact() {
        val opt = FixedLearningRate(0.5f)
        var params = toyParams(w0, b0)

        val s1 = opt.step(params, grads(gw1, gb1), Unit)
        assertContentEquals(floatArrayOf(0.75f, -1.5f), s1.params["w"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.0f), s1.params["b"]!!.hostF32())

        params = listOf(NamedParameter("w", s1.params["w"]!!), NamedParameter("b", s1.params["b"]!!))
        val s2 = opt.step(params, grads(gw2, gb2), Unit)
        assertContentEquals(floatArrayOf(0.625f, -1.75f), s2.params["w"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.25f), s2.params["b"]!!.hostF32())

        params = listOf(NamedParameter("w", s2.params["w"]!!), NamedParameter("b", s2.params["b"]!!))
        val s3 = opt.step(params, grads(gw3, gb3), Unit)
        assertContentEquals(floatArrayOf(0.875f, -1.875f), s3.params["w"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.125f), s3.params["b"]!!.hostF32())
    }

    // -----------------------------------------------------------------------
    // SGD: the DiffKT EMA momentum (v' = μ·v + (1−μ)·g, first visit v = g)
    // and the lrDecay schedule (DiffKT's "weightDecay"). Exact where dyadic.
    // -----------------------------------------------------------------------

    @Test
    fun sgdWithEmaMomentumThreeStepsHandExact() {
        // μ = 0.5, lr = 0.5. Hand: v seeds to g on the first visit, then
        // v' = 0.5·v + 0.5·g — the EMA form, NOT PyTorch's μ·v + g.
        val opt = SGD(initialLearningRate = 0.5f, momentum = 0.5f)
        var state = opt.initialState()
        assertEquals(0, state.stepCount)
        assertTrue(state.velocity.isEmpty(), "state slots must be created lazily")

        // step 1: v_w = g = [0.5, −1]; w = [0.75, −1.5]. v_b = [1]; b = 0.
        var out = opt.step(toyParams(w0, b0), grads(gw1, gb1), state)
        assertContentEquals(floatArrayOf(0.75f, -1.5f), out.params["w"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.0f), out.params["b"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.5f, -1.0f), out.state.velocity["w"]!!.hostF32())
        state = out.state
        assertEquals(1, state.stepCount)

        // step 2: v_w = 0.5·[0.5,−1] + 0.5·[0.25,0.5] = [0.375, −0.25]
        //         w = [0.75−0.1875, −1.5+0.125] = [0.5625, −1.375]
        //         v_b = 0.5·1 + 0.5·(−0.5) = 0.25; b = −0.125
        out = opt.step(
            listOf(NamedParameter("w", out.params["w"]!!), NamedParameter("b", out.params["b"]!!)),
            grads(gw2, gb2), state,
        )
        assertContentEquals(floatArrayOf(0.5625f, -1.375f), out.params["w"]!!.hostF32())
        assertContentEquals(floatArrayOf(-0.125f), out.params["b"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.375f, -0.25f), out.state.velocity["w"]!!.hostF32())
        state = out.state

        // step 3: v_w = 0.5·[0.375,−0.25] + 0.5·[−0.5,0.25] = [−0.0625, 0]
        //         w = [0.5625+0.03125, −1.375−0] = [0.59375, −1.375]
        //         v_b = 0.5·0.25 + 0.5·0.25 = 0.25; b = −0.25
        out = opt.step(
            listOf(NamedParameter("w", out.params["w"]!!), NamedParameter("b", out.params["b"]!!)),
            grads(gw3, gb3), state,
        )
        assertContentEquals(floatArrayOf(0.59375f, -1.375f), out.params["w"]!!.hostF32())
        assertContentEquals(floatArrayOf(-0.25f), out.params["b"]!!.hostF32())
        assertEquals(3, out.state.stepCount)
    }

    @Test
    fun sgdLrDecayFollowsTheDiffktAfterFitSchedule() {
        // lr_k = lr₀ / (1 + decay·k), k = COMPLETED steps: 0.5, 0.25, 0.5/3.
        val opt = SGD(initialLearningRate = 0.5f, lrDecay = 1f)
        var state = opt.initialState()

        var out = opt.step(toyParams(w0, b0), grads(gw1, gb1), state)
        assertContentEquals(floatArrayOf(0.75f, -1.5f), out.params["w"]!!.hostF32())
        state = out.state

        out = opt.step(
            listOf(NamedParameter("w", out.params["w"]!!), NamedParameter("b", out.params["b"]!!)),
            grads(gw2, gb2), state,
        )
        // lr = 0.25: w = [0.75 − 0.0625, −1.5 − 0.125] = [0.6875, −1.625]
        assertContentEquals(floatArrayOf(0.6875f, -1.625f), out.params["w"]!!.hostF32())
        state = out.state

        out = opt.step(
            listOf(NamedParameter("w", out.params["w"]!!), NamedParameter("b", out.params["b"]!!)),
            grads(gw3, gb3), state,
        )
        // lr = 0.5/3: w = [0.6875 + 0.25/3, −1.625 − 0.125/3]
        val w = out.params["w"]!!.hostF32()
        assertClose(0.6875 + 0.25 / 3.0, w[0], "lrDecay step 3 w[0]")
        assertClose(-1.625 - 0.125 / 3.0, w[1], "lrDecay step 3 w[1]")
    }

    @Test
    fun sgdWithoutMomentumEqualsFixedLearningRateBitExact() {
        val sgd = SGD(initialLearningRate = 0.5f)
        val flr = FixedLearningRate(0.5f)
        var sgdState = sgd.initialState()
        var sgdParams = toyParams(w0, b0)
        var flrParams = toyParams(w0, b0)
        val stepGrads = listOf(grads(gw1, gb1), grads(gw2, gb2), grads(gw3, gb3))
        for (g in stepGrads) {
            val a = sgd.step(sgdParams, g, sgdState)
            val b = flr.step(flrParams, g, Unit)
            assertContentEquals(b.params["w"]!!.hostF32(), a.params["w"]!!.hostF32())
            assertContentEquals(b.params["b"]!!.hostF32(), a.params["b"]!!.hostF32())
            assertTrue(a.state.velocity.isEmpty(), "momentum = 0 stores no velocity")
            sgdState = a.state
            sgdParams = listOf(NamedParameter("w", a.params["w"]!!), NamedParameter("b", a.params["b"]!!))
            flrParams = listOf(NamedParameter("w", b.params["w"]!!), NamedParameter("b", b.params["b"]!!))
        }
    }

    // -----------------------------------------------------------------------
    // RMSprop: DiffKT-literal (no epsilon): first visit ms = g², else
    // ms' = β·ms + (1−β)·g², update t − α·g/√ms'. First step is EXACT
    // (√g² = |g| on the grid); later steps against inline double hand math.
    // -----------------------------------------------------------------------

    @Test
    fun rmspropThreeStepsAgainstHandMath() {
        val opt = RMSprop(alpha = 0.5f, beta = 0.5f)
        var state = opt.initialState()
        assertTrue(state.meanSquare.isEmpty(), "state slots must be created lazily")

        // step 1 (exact): ms = g², update = α·g/|g| = α·sign(g).
        // w = [1−0.5, −2+0.5] = [0.5, −1.5]; b = 0.5−0.5 = 0.
        var out = opt.step(toyParams(w0, b0), grads(gw1, gb1), state)
        assertContentEquals(floatArrayOf(0.5f, -1.5f), out.params["w"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.0f), out.params["b"]!!.hostF32())
        assertContentEquals(floatArrayOf(0.25f, 1.0f), out.state.meanSquare["w"]!!.hostF32())
        state = out.state

        // step 2: ms_w = 0.5·[0.25,1] + 0.5·[0.0625,0.25] = [0.15625, 0.625]
        //         ms_b = 0.5·1 + 0.5·0.25 = 0.625
        out = opt.step(
            listOf(NamedParameter("w", out.params["w"]!!), NamedParameter("b", out.params["b"]!!)),
            grads(gw2, gb2), state,
        )
        assertContentEquals(floatArrayOf(0.15625f, 0.625f), out.state.meanSquare["w"]!!.hostF32())
        var w = out.params["w"]!!.hostF32()
        val w20 = 0.5 - 0.5 * 0.25 / sqrt(0.15625)
        val w21 = -1.5 - 0.5 * 0.5 / sqrt(0.625)
        val b2 = 0.0 + 0.5 * 0.5 / sqrt(0.625)
        assertClose(w20, w[0], "rmsprop step 2 w[0]")
        assertClose(w21, w[1], "rmsprop step 2 w[1]")
        assertClose(b2, out.params["b"]!!.hostF32()[0], "rmsprop step 2 b")
        state = out.state

        // step 3: ms_w = 0.5·[0.15625,0.625] + 0.5·[0.25,0.0625] = [0.203125, 0.34375]
        //         ms_b = 0.5·0.625 + 0.5·0.0625 = 0.34375
        out = opt.step(
            listOf(NamedParameter("w", out.params["w"]!!), NamedParameter("b", out.params["b"]!!)),
            grads(gw3, gb3), state,
        )
        assertContentEquals(floatArrayOf(0.203125f, 0.34375f), out.state.meanSquare["w"]!!.hostF32())
        w = out.params["w"]!!.hostF32()
        assertClose(w20 + 0.5 * 0.5 / sqrt(0.203125), w[0], "rmsprop step 3 w[0]")
        assertClose(w21 - 0.5 * 0.25 / sqrt(0.34375), w[1], "rmsprop step 3 w[1]")
        assertClose(b2 - 0.5 * 0.25 / sqrt(0.34375), out.params["b"]!!.hostF32()[0], "rmsprop step 3 b")
    }

    @Test
    fun rmspropEpsilonPlacementAndTheDiffktLiteralHazard() {
        // A zero gradient on the first visit: ms = 0.
        // eps > 0 (the Tlaloc extension, PyTorch placement √ms + ε): the
        // update is α·0/ε = 0 — the parameter is untouched and finite.
        val zeroG = grads(floatArrayOf(0f, 0f), floatArrayOf(0f))
        val withEps = RMSprop(alpha = 0.5f, beta = 0.5f, eps = 1e-8f)
        val outEps = withEps.step(toyParams(w0, b0), zeroG, withEps.initialState())
        assertContentEquals(w0, outEps.params["w"]!!.hostF32())
        assertContentEquals(b0, outEps.params["b"]!!.hostF32())

        // eps = 0 (DiffKT literal parity): 0/√0 = 0/0 = NaN. Pinned as the
        // recorded hazard of the source formula, not papered over.
        val literal = RMSprop(alpha = 0.5f, beta = 0.5f)
        val outNaN = literal.step(toyParams(w0, b0), zeroG, literal.initialState())
        assertTrue(outNaN.params["w"]!!.hostF32()[0].isNaN(), "DiffKT-literal RMSprop NaNs on a zero grad")
    }

    // -----------------------------------------------------------------------
    // Adam: standard Kingma–Ba, bias-corrected (DiffKT's Adam is a TODO
    // placeholder — no parity target exists; see MODEL_LAYER_PLAN.md §4.0.2).
    // Step 1 collapses by hand: m̂ = g, v̂ = g², update = lr·g/(|g|+ε).
    // Steps 2–3 against an inline double-precision spelling of the paper.
    // -----------------------------------------------------------------------

    @Test
    fun adamThreeStepsBiasCorrectedAgainstHandMath() {
        val lr = 0.1; val beta1 = 0.9; val beta2 = 0.999; val eps = 1e-8
        val opt = Adam(learningRate = 0.1f)
        var state = opt.initialState()
        assertTrue(state.m.isEmpty() && state.v.isEmpty(), "state slots must be created lazily")

        // Double-precision hand reference, one element at a time.
        var m = DoubleArray(3); var v = DoubleArray(3) // [w0, w1, b]
        var p = doubleArrayOf(1.0, -2.0, 0.5)
        val stepGradsD = listOf(
            doubleArrayOf(0.5, -1.0, 1.0),
            doubleArrayOf(0.25, 0.5, -0.5),
            doubleArrayOf(-0.5, 0.25, 0.25),
        )
        val stepGradsF = listOf(grads(gw1, gb1), grads(gw2, gb2), grads(gw3, gb3))

        var params = toyParams(w0, b0)
        for (t in 1..3) {
            val g = stepGradsD[t - 1]
            for (i in 0..2) {
                m[i] = beta1 * m[i] + (1 - beta1) * g[i]
                v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i]
                val mHat = m[i] / (1 - beta1.pow(t))
                val vHat = v[i] / (1 - beta2.pow(t))
                p[i] = p[i] - lr * mHat / (sqrt(vHat) + eps)
            }
            val out = opt.step(params, stepGradsF[t - 1], state)
            val w = out.params["w"]!!.hostF32()
            assertClose(p[0], w[0], "adam step $t w[0]")
            assertClose(p[1], w[1], "adam step $t w[1]")
            assertClose(p[2], out.params["b"]!!.hostF32()[0], "adam step $t b")
            assertEquals(t, out.state.stepCount)
            if (t == 1) {
                // The step-1 hand collapse, pinned as literals: m̂ = g and
                // v̂ = g² exactly at t = 1, so the update is lr·g/(|g|+ε)
                // ≈ lr·sign(g): w₁ ≈ [0.9, −1.9], b₁ ≈ 0.4.
                assertClose(0.9, w[0], "adam step-1 collapse w[0]")
                assertClose(-1.9, w[1], "adam step-1 collapse w[1]")
                assertClose(0.4, out.params["b"]!!.hostF32()[0], "adam step-1 collapse b")
            }
            state = out.state
            params = listOf(NamedParameter("w", out.params["w"]!!), NamedParameter("b", out.params["b"]!!))
        }
    }

    // -----------------------------------------------------------------------
    // Purity, laziness, guards, and the Momentum finding.
    // -----------------------------------------------------------------------

    @Test
    fun stepMutatesNothingItWasGiven() {
        val params = toyParams(w0, b0)
        val g = grads(gw1, gb1)
        val opt = SGD(initialLearningRate = 0.5f, momentum = 0.5f)
        val state0 = opt.initialState()
        val out = opt.step(params, g, state0)

        // Inputs untouched: original tensors, gradient tensors, and the old
        // state object all still hold their pre-step values.
        assertContentEquals(floatArrayOf(1.0f, -2.0f), params[0].tensor.hostF32())
        assertContentEquals(floatArrayOf(0.5f, -1.0f), g["w"]!!.hostF32())
        assertEquals(0, state0.stepCount)
        assertTrue(state0.velocity.isEmpty())
        // And the new state is a distinct value.
        assertEquals(1, out.state.stepCount)
        assertEquals(setOf("w", "b"), out.state.velocity.keys)
    }

    @Test
    fun gradientKeyAndDimsGuards() {
        val opt = FixedLearningRate(0.5f)
        val params = toyParams(w0, b0)
        // Missing gradient.
        assertFailsWith<IllegalArgumentException> {
            opt.step(params, mapOf("w" to Tensors.f32Vector<Sym>(gw1)), Unit)
        }
        // Unknown key.
        assertFailsWith<IllegalArgumentException> {
            opt.step(params, grads(gw1, gb1) + mapOf("ghost" to Tensors.f32Vector<Sym>(gb1)), Unit)
        }
        // Dims mismatch.
        assertFailsWith<IllegalArgumentException> {
            opt.step(params, mapOf(
                "w" to Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)),
                "b" to Tensors.f32Vector<Sym>(gb1),
            ), Unit)
        }
    }

    @Test
    fun momentumUpdatedIsTheEmaHelperNotAnOptimizer() {
        // DiffKT's Momentum.kt finding (F0 §4.0.2): momentum weights the NEW
        // statistic. (1−0.25)·current + 0.25·new, exact on the quarter grid.
        assertEquals(1.75f, 2.0f.momentumUpdated(1.0f, 0.25f))
        val cur = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, -2.0f))
        val new = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, 0.5f))
        assertContentEquals(
            floatArrayOf(0.875f, -1.375f),
            cur.momentumUpdated(new, 0.25f).hostF32(),
        )
    }

    // -----------------------------------------------------------------------
    // The integration cert: F1's valueAndGradients + SGD drive a fixed-key
    // Dense model's loss strictly down over 10 deterministic steps.
    // -----------------------------------------------------------------------

    @Test
    fun sgdDecreasesDenseModelLossOverTenDeterministicSteps() {
        val x = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(
            1f, -0.5f,
            2f, 0.25f,
            -1f, 0.5f,
            0.5f, 1f,
        ))
        var model = Dense(2, 1, RandomKey.fromSeed(7))
        val opt = SGD(initialLearningRate = 0.1f)
        var state = opt.initialState()

        val losses = ArrayList<Float>(11)
        repeat(10) {
            val res = valueAndGradients(model, listOf(x)) { y ->
                val d = y - 1.5f
                (d * d).mean()
            }
            losses += res.loss
            val (next, nextState) = opt.step(model, res.gradients, state)
            model = next
            state = nextState
        }
        losses += valueAndGradients(model, listOf(x)) { y ->
            val d = y - 1.5f
            (d * d).mean()
        }.loss

        assertEquals(11, losses.size)
        for (i in 1 until losses.size) {
            assertTrue(losses[i] < losses[i - 1], "loss must strictly decrease every step: $losses")
        }
        assertEquals(10, state.stepCount)
    }
}
