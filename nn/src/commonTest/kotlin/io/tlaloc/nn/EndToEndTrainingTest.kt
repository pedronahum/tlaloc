package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.split
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.444 — Phase F8 (1) + (4): the end-to-end training certification on the
 * host lane. The whole Phase F stack in one loop: threefry-keyed data and
 * initializers, `capture` ONCE per model structure (the F1 caching contract —
 * the trained model re-binds through the SAME `CapturedStep` every step),
 * gradients from `DxirReverseTransform` through the captured graph, and the
 * pure F3 optimizers folding new models out of old.
 *
 * Oracle story, per the slice:
 *  - the MLP's step-0 gradients get an INDEPENDENT check: central finite
 *    differences of the captured loss itself (the same `CapturedStep.run`,
 *    perturbed parameters) on 3 parameter elements spanning all three
 *    trainable tensors' roles (first-layer weight, first-layer bias,
 *    output-layer weight). FD truncation is O(h²) = O(1e-4) at h = 1e-2 and
 *    the f32 loss eval adds ~1e-6 rel noise scaled by 1/h, so the 2e-3+2% pin
 *    is honest slack, not tuned slack.
 *  - convergence is pinned TWICE: a relative collapse (final ≤ initial/20 for
 *    the MLP, ≤ initial/10 for the conv net) and an absolute bound observed
 *    on the fixed threefry task with margin (the task is bit-deterministic —
 *    same key, same draws, same losses, forever).
 *  - "monotone-ish" is made precise: non-overlapping window MEANS (25-step
 *    for the MLP's 200-step run, 10-step for the conv net's 40) strictly
 *    decrease — Adam's per-step wiggle is real; windowed means are the
 *    honest shape of "the loss goes down".
 */
class EndToEndTrainingTest {

    // ------------------------------------------------------------------
    // The fixed threefry-keyed synthetic task (bit-deterministic forever).
    // ------------------------------------------------------------------

    private val n = 16
    private val d = 4

    /** x ∈ [−1, 1)^{16×4} off the data key's first child stream. */
    private fun mlpInputs(): FloatArray {
        val u = uniformFloats(RandomKey.fromSeed(1234).split(2)[0], n * d)
        return FloatArray(u.size) { i -> 2f * u[i] - 1f }
    }

    /** The target the MLP must learn: y = x₀·x₁ + 0.5·x₂ − 0.25·x₃ (the
     * product term keeps the hidden layer honest — no linear model fits it). */
    private fun mlpTargets(x: FloatArray): FloatArray =
        FloatArray(n) { i ->
            val r = i * d
            x[r] * x[r + 1] + 0.5f * x[r + 2] - 0.25f * x[r + 3]
        }

    private fun mlpModel(): Sequential = Sequential(
        Dense(d, 8, RandomKey.fromSeed(7).split(2)[0]),
        ReluLayer,
        Dense(8, 1, RandomKey.fromSeed(7).split(2)[1]),
    )

    private fun windowMeansStrictlyDecrease(losses: List<Float>, window: Int, tag: String) {
        val means = losses.chunked(window).filter { it.size == window }.map { w ->
            w.fold(0.0) { a, v -> a + v } / w.size
        }
        for (i in 1 until means.size) {
            assertTrue(
                means[i] < means[i - 1],
                "$tag: $window-step window means must strictly decrease, got $means (losses $losses)",
            )
        }
    }

    // ------------------------------------------------------------------
    // (1) The MLP: Dense→Relu→Dense + Adam, one capture, 200 steps.
    // ------------------------------------------------------------------

    @Test
    fun mlpAdamTrainingConvergesOnFixedThreefryTask() {
        val xv = mlpInputs()
        val yv = mlpTargets(xv)
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)

        var model = mlpModel()
        // ONE capture for the whole run — the F1 caching contract exercised:
        // structure never changes, so every step re-binds through this pair.
        val step = capture(model, listOf(x)) { y ->
            val t = y.constant<io.tlaloc.core.Shape>(yv, intArrayOf(n, 1))
            val diff = y - t
            (diff * diff).mean()
        }

        val opt = Adam(learningRate = 0.05f)
        var state = opt.initialState()
        val losses = ArrayList<Float>(201)
        repeat(200) {
            val res = step.run(model, listOf(x))
            losses += res.loss
            val (next, nextState) = opt.step(model, res.gradients, state)
            model = next
            state = nextState
        }
        losses += step.run(model, listOf(x)).loss

        println("[nn-e2e] MLP+Adam: loss ${losses[0]} -> ${losses.last()} over 200 steps")
        assertEquals(201, losses.size)
        // The pinned bounds (observed 0.30533 → 0.0012239 on this fixed task;
        // margin, not slack-to-hide-drift):
        assertTrue(losses[0] > 0.1f, "initial loss should start un-trained (got ${losses[0]})")
        assertTrue(
            losses.last() < 2e-3f,
            "final loss must land under the pinned bound 2e-3 (got ${losses.last()}; curve $losses)",
        )
        assertTrue(
            losses.last() < losses[0] / 20f,
            "final loss must collapse at least 20×: ${losses[0]} → ${losses.last()}",
        )
        windowMeansStrictlyDecrease(losses.dropLast(1), 25, "MLP+Adam")
    }

    // ------------------------------------------------------------------
    // (1b) The FD spot check at step 0: 3 parameter elements, central
    // differences THROUGH the captured loss itself.
    // ------------------------------------------------------------------

    @Test
    fun mlpStepZeroGradientsMatchFiniteDifferencesOnThreeParameters() {
        val xv = mlpInputs()
        val yv = mlpTargets(xv)
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        val model = mlpModel()

        val step = capture(model, listOf(x)) { y ->
            val t = y.constant<io.tlaloc.core.Shape>(yv, intArrayOf(n, 1))
            val diff = y - t
            (diff * diff).mean()
        }
        val grads = step.run(model, listOf(x)).gradients

        // One element per trainable tensor role: first-layer weight, first-
        // layer bias, output-layer weight.
        val picks = listOf("0.w" to 3, "0.b" to 2, "2.w" to 5)
        val h = 1e-2f
        for ((key, idx) in picks) {
            val analytic = grads[key]!!.hostF32()[idx]

            fun lossWith(delta: Float): Float {
                val base = model.parameters.first { it.key == key }.tensor
                val bumped = base.hostF32().copyOf().also { it[idx] += delta }
                val tensor = io.tlaloc.core.DTensor<io.tlaloc.core.Shape, io.tlaloc.core.F32>(
                    io.tlaloc.core.HostF32Storage(bumped), base.dims.copyOf(), io.tlaloc.core.F32,
                )
                return step.run(model.withParameters(mapOf(key to tensor)), listOf(x)).loss
            }

            val fd = (lossWith(h) - lossWith(-h)) / (2f * h)
            assertTrue(
                abs(fd - analytic) <= 2e-3f + 0.02f * abs(analytic),
                "FD check on '$key'[$idx]: analytic $analytic vs central-difference $fd " +
                    "(|diff| ${abs(fd - analytic)}) exceeds 2e-3 + 2%",
            )
        }
    }

    // ------------------------------------------------------------------
    // (4) The small conv net: Conv2d→Relu→Flatten→Dense trains and the
    // loss decreases.
    // ------------------------------------------------------------------

    @Test
    fun convNetTrainsAndLossDecreases() {
        val b = 4
        val u = uniformFloats(RandomKey.fromSeed(4321).split(2)[0], b * 1 * 4 * 4)
        val xv = FloatArray(u.size) { i -> 2f * u[i] - 1f }
        // Target: per-image mean of the 16 pixels — learnable through the
        // conv/dense stack, computed host-side (fixed with the key).
        val yv = FloatArray(b) { i ->
            var s = 0.0
            for (j in 0 until 16) s += xv[i * 16 + j]
            (s / 16.0).toFloat()
        }
        val x = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(b, 1, 4, 4, xv)

        val keys = RandomKey.fromSeed(99).split(2)
        var model = Sequential(
            Conv2d(intArrayOf(2, 1, 2, 2), keys[0]),  // [4,1,4,4] → [4,2,3,3]
            ReluLayer,
            Flatten,                                   // → [4,18]
            Dense(18, 1, keys[1]),
        )
        val step = capture(model, listOf(x)) { y ->
            val t = y.constant<io.tlaloc.core.Shape>(yv, intArrayOf(b, 1))
            val diff = y - t
            (diff * diff).mean()
        }

        val opt = Adam(learningRate = 0.02f)
        var state = opt.initialState()
        val losses = ArrayList<Float>(41)
        repeat(40) {
            val res = step.run(model, listOf(x))
            losses += res.loss
            val (next, nextState) = opt.step(model, res.gradients, state)
            model = next
            state = nextState
        }
        losses += step.run(model, listOf(x)).loss

        println("[nn-e2e] ConvNet+Adam: loss ${losses[0]} -> ${losses.last()} over 40 steps")
        assertTrue(
            losses.last() < losses[0] / 10f,
            "conv-net loss must collapse at least 10×: ${losses[0]} → ${losses.last()} ($losses)",
        )
        windowMeansStrictlyDecrease(losses.dropLast(1), 10, "ConvNet+Adam")
    }
}
