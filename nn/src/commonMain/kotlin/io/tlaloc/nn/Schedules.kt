/**
 * **Learning-rate schedules**, and the optimizer combinator that applies one.
 *
 * [SGD.lrDecay] is DiffKT's `1/(1 + d·k)` annealing, baked into one optimizer;
 * the schedules here apply to all four optimizers and cover step, exponential
 * and cosine decay.
 *
 * ## The contract: a schedule is a FUNCTION OF THE STEP COUNT
 *
 * `at(step): Float`, pure, with no state of its own. That is not a style
 * preference, it is forced by the module's own design: an optimizer here is an immutable VALUE and `step` is
 * `(params, grads, state) -> (params', state')`. A schedule holding a mutable
 * cursor — PyTorch's `LRScheduler.step()` shape — would be the only mutable
 * thing in the module, would produce a different trajectory depending on how
 * many times it had been called, and could not be checkpointed by value.
 *
 * Because the schedule is a function, the step COUNT is state, and it lives
 * where all the other optimizer state lives: in [ScheduledState], written to
 * and read back from a checkpoint like Adam's moments. A resumed run continues
 * the schedule instead of restarting it — see the [Scheduled] KDoc for why that
 * is the whole point.
 *
 * `step` is 0-BASED and counts COMPLETED steps: `at(0)` is the rate the first
 * update uses. This matches [SGD]'s existing `stepCount` (`lr₀ / (1 + d·k)`
 * with k = completed steps, so the first step runs at `lr₀`) and it matches
 * PyTorch, whose `last_epoch` starts at 0 for the first `get_lr()`.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.pow

/** The learning rate at a given number of COMPLETED steps. Pure; no state. */
fun interface LearningRateSchedule {
    fun at(step: Int): Float
}

private fun requireStep(what: String, step: Int) {
    require(step >= 0) { "$what: step must be >= 0 (got $step)" }
}

/** A flat rate. Exists so a [Scheduled] optimizer can be written uniformly. */
class ConstantLR(val learningRate: Float) : LearningRateSchedule {
    override fun at(step: Int): Float {
        requireStep("ConstantLR", step)
        return learningRate
    }
}

/**
 * Staircase decay: `lr₀ · factor^⌊step / everySteps⌋`.
 *
 * PyTorch's `StepLR(step_size, gamma)`, exactly — [factor] is `gamma` and
 * [everySteps] is `step_size`. The floor is what makes it a staircase: the
 * rate is constant inside a block of [everySteps] and drops on the boundary.
 */
class StepDecay(
    val initialLearningRate: Float,
    val factor: Float,
    val everySteps: Int,
) : LearningRateSchedule {

    init {
        require(everySteps > 0) { "StepDecay: everySteps must be > 0 (got $everySteps)" }
        require(factor > 0f) {
            "StepDecay: factor must be > 0 (got $factor) — a factor of 0 freezes the model " +
                "forever at the first boundary and a negative factor reverses every update"
        }
    }

    override fun at(step: Int): Float {
        requireStep("StepDecay", step)
        return initialLearningRate * factor.toDouble().pow(step / everySteps).toFloat()
    }
}

/**
 * Exponential decay: `lr₀ · decayRate^(step / decaySteps)`.
 *
 * With the default `decaySteps = 1` this is PyTorch's `ExponentialLR(gamma)`:
 * `lr₀ · gamma^step`. [decaySteps] generalises it to "decay by [decayRate]
 * every [decaySteps] steps" with a CONTINUOUS exponent, which is TensorFlow's
 * / optax's `exponential_decay`; [staircase] floors the exponent and recovers
 * [StepDecay]'s shape.
 *
 * Recorded so nobody has to rediscover it: `StepDecay(lr, f, n)` and
 * `ExponentialDecay(lr, f, n, staircase = true)` compute the same function.
 * Both exist because both names are what users come looking for, and the
 * equality is pinned by a test rather than asserted here.
 */
class ExponentialDecay(
    val initialLearningRate: Float,
    val decayRate: Float,
    val decaySteps: Int = 1,
    val staircase: Boolean = false,
) : LearningRateSchedule {

    init {
        require(decaySteps > 0) { "ExponentialDecay: decaySteps must be > 0 (got $decaySteps)" }
        require(decayRate > 0f) { "ExponentialDecay: decayRate must be > 0 (got $decayRate)" }
    }

    override fun at(step: Int): Float {
        requireStep("ExponentialDecay", step)
        val e = step.toDouble() / decaySteps
        val exponent = if (staircase) floor(e) else e
        return initialLearningRate * decayRate.toDouble().pow(exponent).toFloat()
    }
}

/**
 * Cosine decay from `lr₀` to [finalLearningRate] over [decaySteps] steps, then
 * flat:
 *
 *   `lr(k) = final + (lr₀ − final) · ½(1 + cos(π · min(k, decaySteps) / decaySteps))`
 *
 * PyTorch's `CosineAnnealingLR(T_max, eta_min)` closed form ([decaySteps] is
 * `T_max`, [finalLearningRate] is `eta_min`). PyTorch implements it
 * RECURSIVELY and its recursion reproduces this closed form as long as nothing
 * else writes the learning rate — which, here, nothing can, because the
 * schedule is a function. The CLAMP past [decaySteps] is the deliberate
 * difference: PyTorch's `CosineAnnealingLR` keeps going and rises back toward
 * `lr₀` (it is an *annealing* schedule with a period), and a run that trained
 * past `T_max` by accident would silently get its learning rate back. Clamping
 * refuses to do that; a warm-restart schedule would be its own class (not
 * implemented).
 */
class CosineDecay(
    val initialLearningRate: Float,
    val decaySteps: Int,
    val finalLearningRate: Float = 0f,
) : LearningRateSchedule {

    init {
        require(decaySteps > 0) { "CosineDecay: decaySteps must be > 0 (got $decaySteps)" }
    }

    override fun at(step: Int): Float {
        requireStep("CosineDecay", step)
        val k = if (step > decaySteps) decaySteps else step
        val c = 0.5 * (1.0 + cos(PI * k.toDouble() / decaySteps))
        return (finalLearningRate + (initialLearningRate - finalLearningRate) * c).toFloat()
    }
}

/**
 * Linear warmup from 0 to [peakLearningRate] over [warmupSteps] steps, then
 * [then] re-based so its own step 0 is the first post-warmup step.
 *
 * **`at(0)` is 0**, and that is HuggingFace's
 * `get_linear_schedule_with_warmup` convention (`step / max(1, warmup)`),
 * stated here rather than left to be discovered: the very first update of a
 * warmed-up run does nothing. The alternative, `(step + 1) / warmup`, reaches
 * the peak one step early and is what optax's `linear_schedule` does with
 * `init_value` set nonzero. Neither is wrong; this one is the one written down.
 */
class LinearWarmup(
    val warmupSteps: Int,
    val peakLearningRate: Float,
    val then: LearningRateSchedule,
) : LearningRateSchedule {

    init {
        require(warmupSteps > 0) { "LinearWarmup: warmupSteps must be > 0 (got $warmupSteps)" }
    }

    override fun at(step: Int): Float {
        requireStep("LinearWarmup", step)
        if (step >= warmupSteps) return then.at(step - warmupSteps)
        return peakLearningRate * step.toFloat() / warmupSteps.toFloat()
    }
}

/** [Scheduled]'s state: the completed-step count plus the inner optimizer's own state. */
class ScheduledState<S> internal constructor(
    val stepCount: Int,
    val inner: S,
)

/**
 * The combinator that applies a [LearningRateSchedule] to any
 * optimizer: `Scheduled(CosineDecay(0.01f, 600)) { lr -> Adam(lr) }`.
 *
 * ## Why a rebuild per step, and not a mutable learning rate
 *
 * Every optimizer in this module holds its learning rate as a `val` on an
 * immutable value, so "changing the learning rate" means constructing the
 * optimizer again — which costs nothing (an `Adam` is four floats) and keeps
 * the pure `(params, grads, state) -> (params', state')` contract intact. The
 * [build] lambda is therefore the whole mechanism: it is called with the
 * scheduled rate for the CURRENT step count and the resulting optimizer takes
 * that one step.
 *
 * REJECTED: a `withLearningRate(lr)` member on [Optimizer] — it would be a
 * member every implementer owes, including the two ([FixedLearningRate] whose
 * rate is `alpha`, [SGD] whose rate is already scheduled) whose rate is not
 * called `learningRate`. REJECTED: a mutable `var learningRate` — see the file
 * KDoc; the trajectory would depend on call history.
 *
 * ## Checkpointing
 *
 * [Scheduled] is resumable exactly when its inner optimizer is, and the inner
 * checkpoint is NESTED under `inner.` rather than merged, so an `Adam` slot
 * name cannot collide with the wrapper's own. The `kind` tag is
 * `Scheduled(<inner kind>)`, which means a checkpoint written by
 * `Scheduled(...) { Adam(it) }` is refused by name if it is loaded into a bare
 * `Adam` — correctly: the bare `Adam` has nowhere to put the step count that
 * says where on the schedule the run had got to, and would resume at the
 * schedule's *start*.
 */
class Scheduled<S>(
    val schedule: LearningRateSchedule,
    val build: (Float) -> Optimizer<S>,
) : CheckpointableOptimizer<ScheduledState<S>> {

    /** The inner optimizer as it is configured for a given completed-step count. */
    fun optimizerAt(step: Int): Optimizer<S> = build(schedule.at(step))

    /** The rate the next step will use. Useful in a log line; nothing depends on it. */
    fun learningRateAt(step: Int): Float = schedule.at(step)

    override fun initialState(): ScheduledState<S> =
        ScheduledState(0, optimizerAt(0).initialState())

    override fun step(
        params: List<NamedParameter>,
        grads: Map<String, DTensor<*, F32>>,
        state: ScheduledState<S>,
    ): OptimizerStep<ScheduledState<S>> {
        val out = optimizerAt(state.stepCount).step(params, grads, state.inner)
        return OptimizerStep(out.params, ScheduledState(state.stepCount + 1, out.state))
    }

    @Suppress("UNCHECKED_CAST")
    private fun checkpointableInner(): CheckpointableOptimizer<S> {
        val inner = optimizerAt(0)
        return (inner as? CheckpointableOptimizer<S>)
            ?: OptimizerCheckpoint.refuse(inner, "state (wrapped in Scheduled)")
    }

    override val checkpointKind: String
        get() = "$KIND(${checkpointableInner().checkpointKind})"

    override fun saveState(state: ScheduledState<S>): OptimizerCheckpoint {
        val ic = checkpointableInner().saveState(state.inner)
        return OptimizerCheckpoint(
            kind = "$KIND(${ic.kind})",
            scalars = mapOf("stepCount" to state.stepCount.toString()) +
                ic.scalars.mapKeys { "$INNER.${it.key}" },
            tensors = OptimizerCheckpoint.grouped(INNER, ic.tensors),
        )
    }

    override fun loadState(checkpoint: OptimizerCheckpoint): ScheduledState<S> {
        val inner = checkpointableInner()
        val innerKind = checkpoint.kind.removePrefix("$KIND(").removeSuffix(")")
        val ic = OptimizerCheckpoint(
            kind = innerKind,
            scalars = checkpoint.scalars
                .filterKeys { it.startsWith("$INNER.") }
                .mapKeys { it.key.removePrefix("$INNER.") },
            tensors = checkpoint.group(INNER),
        )
        return ScheduledState(checkpoint.int("stepCount"), inner.loadState(ic))
    }

    private companion object {
        const val KIND = "Scheduled"
        const val INNER = "inner"
    }
}
