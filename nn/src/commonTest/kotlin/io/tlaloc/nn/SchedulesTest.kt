package io.tlaloc.nn

import io.tlaloc.core.RandomKey
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.502 (Tier 2 item 7) — the learning-rate schedules, against
 * HAND-COMPUTED ANALYTIC expectations.
 *
 * Every number below is written out as arithmetic a reader can check by eye,
 * not as a golden value read off a run. The PyTorch cross-check for the three
 * schedules PyTorch also ships (`StepLR`, `ExponentialLR`,
 * `CosineAnnealingLR`) lives in
 * `benchmarks/.../NnSchedulesAndClippingVsPytorchTest`, and self-skips when the
 * oracle venv is absent; this file is the certification that does not depend on
 * a venv existing.
 */
class SchedulesTest {

    private fun close(expected: Float, actual: Float, tag: String, tol: Float = 1e-6f) {
        assertTrue(
            abs(expected - actual) <= tol * maxOf(1f, abs(expected)),
            "$tag: expected $expected, got $actual (tol $tol relative)",
        )
    }

    @Test
    fun constantIsConstantAndStepZeroIsTheFirstUpdatesRate() {
        val s = ConstantLR(0.01f)
        for (k in listOf(0, 1, 7, 10_000)) assertEquals(0.01f, s.at(k))
    }

    @Test
    fun stepDecayIsAStaircaseWithTheFloorOnTheBoundary() {
        // lr(k) = 0.1 * 0.5^floor(k/3)
        val s = StepDecay(initialLearningRate = 0.1f, factor = 0.5f, everySteps = 3)
        close(0.1f, s.at(0), "k=0")
        close(0.1f, s.at(1), "k=1")
        close(0.1f, s.at(2), "k=2")
        close(0.05f, s.at(3), "k=3 — the first boundary")
        close(0.05f, s.at(5), "k=5")
        close(0.025f, s.at(6), "k=6")
        close(0.1f * 0.5f * 0.5f * 0.5f, s.at(9), "k=9")
    }

    @Test
    fun exponentialDecayWithDecayStepsOneIsTheClassicGammaToTheStep() {
        // lr(k) = 0.2 * 0.9^k
        val s = ExponentialDecay(initialLearningRate = 0.2f, decayRate = 0.9f)
        close(0.2f, s.at(0), "k=0")
        close(0.18f, s.at(1), "k=1")
        close(0.2f * 0.9f * 0.9f, s.at(2), "k=2")
        close(0.2f * 0.9f * 0.9f * 0.9f * 0.9f * 0.9f, s.at(5), "k=5")
    }

    @Test
    fun exponentialDecayInterpolatesBetweenBoundariesUnlessItIsAStaircase() {
        // decaySteps = 4: the exponent is k/4, continuous.
        val smooth = ExponentialDecay(1f, 0.5f, decaySteps = 4)
        close(1f, smooth.at(0), "smooth k=0")
        // 0.5^(2/4) = 0.5^0.5 = 1/sqrt(2)
        close(0.70710678f, smooth.at(2), "smooth k=2 — the half-way point is sqrt(0.5)")
        close(0.5f, smooth.at(4), "smooth k=4")
        val stair = ExponentialDecay(1f, 0.5f, decaySteps = 4, staircase = true)
        close(1f, stair.at(2), "staircase k=2 holds")
        close(0.5f, stair.at(4), "staircase k=4 drops")
    }

    @Test
    fun aStaircaseExponentialAndAStepDecayComputeTheSameFunction() {
        // Recorded in ExponentialDecay's KDoc; pinned here rather than trusted.
        val a = StepDecay(0.3f, 0.7f, 5)
        val b = ExponentialDecay(0.3f, 0.7f, decaySteps = 5, staircase = true)
        for (k in 0..40) close(a.at(k), b.at(k), "k=$k")
    }

    @Test
    fun cosineDecayRunsFromTheInitialRateToTheFinalOneAndThenHolds() {
        // lr(k) = final + (lr0-final) * 0.5*(1 + cos(pi*k/T)), T = 10
        val s = CosineDecay(initialLearningRate = 0.1f, decaySteps = 10)
        close(0.1f, s.at(0), "k=0 is the peak (cos 0 = 1)")
        close(0.05f, s.at(5), "k=T/2 is exactly half (cos pi/2 = 0)")
        close(0f, s.at(10), "k=T is the floor (cos pi = -1)")
        // The clamp: PyTorch's CosineAnnealingLR would climb back toward 0.1.
        close(0f, s.at(11), "past T it HOLDS, it does not restart")
        close(0f, s.at(1000), "still held")
        for (k in 0..10) {
            val want = 0.05 * (1.0 + cos(PI * k / 10.0))
            close(want.toFloat(), s.at(k), "closed form at k=$k", 2e-6f)
        }
    }

    @Test
    fun cosineDecayHonoursANonZeroFloor() {
        val s = CosineDecay(initialLearningRate = 1f, decaySteps = 4, finalLearningRate = 0.25f)
        close(1f, s.at(0), "k=0")
        close(0.625f, s.at(2), "k=2 = 0.25 + 0.75*0.5")
        close(0.25f, s.at(4), "k=4")
        close(0.25f, s.at(9), "clamped")
    }

    @Test
    fun linearWarmupStartsAtZeroAndHandsOverAtThePeak() {
        // HuggingFace's convention, stated in LinearWarmup's KDoc: at(0) = 0,
        // so the first update of a warmed-up run does nothing.
        val s = LinearWarmup(4, 0.2f, ConstantLR(0.2f))
        close(0f, s.at(0), "k=0 is zero, by the documented convention")
        close(0.05f, s.at(1), "k=1 = peak/4")
        close(0.1f, s.at(2), "k=2")
        close(0.15f, s.at(3), "k=3")
        close(0.2f, s.at(4), "k=4 hands over at the peak")
        close(0.2f, s.at(50), "and stays on the inner schedule")
    }

    @Test
    fun linearWarmupRebasesTheInnerSchedulesStepCount() {
        // The inner schedule's own step 0 is the first POST-warmup step, so a
        // warmup followed by a cosine decays over `decaySteps` after warmup,
        // not `decaySteps - warmupSteps`.
        val s = LinearWarmup(3, 1f, CosineDecay(1f, 10))
        close(1f, s.at(3), "handover is the cosine's own k=0")
        close(0.5f, s.at(8), "cosine k=5")
        close(0f, s.at(13), "cosine k=10")
    }

    @Test
    fun aNegativeStepIsRefusedByNameByEverySchedule() {
        val schedules: List<Pair<String, LearningRateSchedule>> = listOf(
            "ConstantLR" to ConstantLR(1f),
            "StepDecay" to StepDecay(1f, 0.5f, 2),
            "ExponentialDecay" to ExponentialDecay(1f, 0.5f),
            "CosineDecay" to CosineDecay(1f, 4),
            "LinearWarmup" to LinearWarmup(2, 1f, ConstantLR(1f)),
        )
        for ((name, s) in schedules) {
            val e = assertFailsWith<IllegalArgumentException>(name) { s.at(-1) }
            assertTrue(name in e.message!!, "$name: ${e.message}")
        }
    }

    @Test
    fun degenerateScheduleParametersAreRefusedByName() {
        assertTrue("everySteps" in assertFailsWith<IllegalArgumentException> {
            StepDecay(1f, 0.5f, 0)
        }.message!!)
        assertTrue("factor" in assertFailsWith<IllegalArgumentException> {
            StepDecay(1f, 0f, 2)
        }.message!!)
        assertTrue("decaySteps" in assertFailsWith<IllegalArgumentException> {
            ExponentialDecay(1f, 0.5f, 0)
        }.message!!)
        assertTrue("decayRate" in assertFailsWith<IllegalArgumentException> {
            ExponentialDecay(1f, -0.5f)
        }.message!!)
        assertTrue("decaySteps" in assertFailsWith<IllegalArgumentException> {
            CosineDecay(1f, 0)
        }.message!!)
        assertTrue("warmupSteps" in assertFailsWith<IllegalArgumentException> {
            LinearWarmup(0, 1f, ConstantLR(1f))
        }.message!!)
    }

    // ---- the combinator ---------------------------------------------------

    @Test
    fun aScheduledOptimizerTakesEachStepAtTheScheduledRate() {
        // The oracle is the SAME optimizer driven by hand at the rates the
        // schedule names: `Scheduled` must be exactly that, with the step
        // count bookkept for the caller.
        val p = listOf(NamedParameter("w", Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f))))
        val g = mapOf<String, io.tlaloc.core.DTensor<*, io.tlaloc.core.F32>>(
            "w" to Tensors.f32Vector<Sym>(floatArrayOf(0.5f, 0.25f)),
        )
        val schedule = StepDecay(0.1f, 0.5f, 2)
        val scheduled = Scheduled(schedule) { lr -> FixedLearningRate(lr) }

        var params = p
        var state = scheduled.initialState()
        val byScheduled = ArrayList<Float>()
        repeat(6) {
            val out = scheduled.step(params, g, state)
            params = params.map { NamedParameter(it.key, out.params.getValue(it.key)) }
            state = out.state
            byScheduled += params[0].tensor.hostF32()[0]
        }

        var hand = p
        val byHand = ArrayList<Float>()
        for (k in 0 until 6) {
            val out = FixedLearningRate(schedule.at(k)).step(hand, g, Unit)
            hand = hand.map { NamedParameter(it.key, out.params.getValue(it.key)) }
            byHand += hand[0].tensor.hostF32()[0]
        }
        for (k in byHand.indices) {
            assertEquals(byHand[k].toRawBits(), byScheduled[k].toRawBits(), "step $k")
        }
        assertEquals(6, state.stepCount)
        // And the rate really did change: 1 - (0.1+0.1+0.05+0.05+0.025+0.025)*0.5
        close(1f - 0.5f * 0.35f, byScheduled.last(), "final weight")
    }

    @Test
    fun aScheduledAdamActuallyAnnealsATrainingRun() {
        // The end-to-end shape: the same model and data, once at a constant
        // rate and once cosine-annealed to zero. What is pinned is not "the
        // schedule is better" — it is not, necessarily — but that the annealed
        // run's LAST update is tiny and the constant run's is not. That is the
        // observable difference a schedule is for.
        val n = 16
        val d = 4
        val u = io.tlaloc.core.uniformFloats(RandomKey.fromSeed(1234).split(2)[0], n * d)
        val xv = FloatArray(u.size) { i -> 2f * u[i] - 1f }
        val yv = FloatArray(n) { i ->
            val r = i * d
            xv[r] * xv[r + 1] + 0.5f * xv[r + 2] - 0.25f * xv[r + 3]
        }
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        fun build() = Sequential(
            Dense(d, 8, RandomKey.fromSeed(7).split(2)[0]),
            ReluLayer,
            Dense(8, 1, RandomKey.fromSeed(7).split(2)[1]),
        )
        val step = capture(build(), listOf(x)) { y ->
            val t = y.constant<io.tlaloc.core.Shape>(yv, intArrayOf(n, 1))
            val diff = y - t
            (diff * diff).mean()
        }

        fun lastUpdateMagnitude(optimizer: Optimizer<*>): Float {
            @Suppress("UNCHECKED_CAST")
            val o = optimizer as Optimizer<Any?>
            var m = build()
            var s: Any? = o.initialState()
            var before = m
            repeat(40) {
                before = m
                val r = step.run(m, listOf(x))
                val out = o.step(m.parameters, r.gradients, s)
                m = m.withParameters(out.params)
                s = out.state
            }
            var acc = 0f
            for (i in m.parameters.indices) {
                val a = m.parameters[i].tensor.hostF32()
                val b = before.parameters[i].tensor.hostF32()
                for (j in a.indices) acc = maxOf(acc, abs(a[j] - b[j]))
            }
            return acc
        }

        val constant = lastUpdateMagnitude(Adam(learningRate = 0.05f))
        val annealed = lastUpdateMagnitude(
            Scheduled(CosineDecay(0.05f, 40)) { lr -> Adam(learningRate = lr) },
        )
        println("[nn-sched] last-step max|Δw|: constant $constant, cosine-annealed $annealed")
        assertTrue(
            annealed < constant / 100f,
            "a cosine schedule to zero must make the final update vanish: constant $constant, " +
                "annealed $annealed",
        )
    }
}
