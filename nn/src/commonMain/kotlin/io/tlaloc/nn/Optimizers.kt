/**
 * §0.4.439 — Phase F3: the optimizers, pure and functional, per the ratified
 * decision 4 of MODEL_LAYER_PLAN.md and the F0 §4.0.2 audit of DiffKT's exact
 * update formulas. An optimizer is an immutable VALUE; `step` is a pure
 * function `(params, grads, state) → (params', state')` with state keyed like
 * the parameters and created LAZILY (a key absent from the state map IS the
 * "first visit of a parameter" DiffKT detects with its mutable per-visit
 * cursor — same semantics, no cursor). REJECTED: DiffKT's mutable
 * `nextParameter` cursor + `afterFit()` reset (visit-order-indexed
 * `mutableListOf` state fights decision 2's immutability and breaks the moment
 * a parameter set is reordered); a `var`-bearing optimizer with an internal
 * step counter (same objection — the step count is state, so it lives in the
 * state value).
 *
 * Zero tape/trace involvement (the gap table's F3 row: "pure host math on
 * DTensor/FloatArray; no tape involvement at all") and, as everywhere in
 * `:nn`, zero gradient math — gradients arrive from `DxirReverseTransform`
 * via [valueAndGradients]/[CapturedStep.run], keyed like the parameters.
 *
 * DiffKT-parity findings recorded by the F0 audit and honoured here:
 * - `SGD` momentum is the EMA form `v' = μ·v + (1−μ)·g`, NOT PyTorch's
 *   `μ·v + g`; the first visit seeds `v = g`.
 * - DiffKT's `SGDOptimizer.weightDecay` is LEARNING-RATE decay
 *   (`lr = lr₀ / (1 + decay·steps)`), not L2 regularization — spelled
 *   [SGD.lrDecay] here so nobody reads weight decay into it.
 * - `RMSprop` has NO epsilon in DiffKT (`t − α·g/√ms`); [RMSprop.eps]
 *   defaults to 0f for literal parity, PyTorch placement (`√ms + eps`) when
 *   nonzero.
 * - **DiffKT's `AdamOptimizer` is a TODO placeholder** (`TODO("Not yet
 *   implemented")` in both methods) — there is no DiffKT Adam to be parity
 *   with, and no "does DiffKT bias-correct" question to answer: [Adam] here
 *   is standard Kingma–Ba, BIAS-CORRECTED (m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ),
 *   update `θ − lr·m̂/(√v̂ + ε)`, defaults lr=1e-3, β₁=.9, β₂=.999, ε=1e-8),
 *   oracled against hand-stepped values and (in F8) the PyTorch harness.
 * - **DiffKT's `Momentum.kt` is NOT an optimizer**: it is the EMA helper
 *   `momentumUpdated(new, momentum) = (1−momentum)·this + momentum·new`
 *   shared with BatchNorm's running stats (momentum weights the NEW stat —
 *   the PyTorch `running_stats` convention). The "Momentum optimizer" of the
 *   slice list IS `SGD(momentum = μ)`; the helper ships here as
 *   [momentumUpdated] for F5's BatchNorm.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import kotlin.math.pow
import kotlin.math.sqrt

/** One optimizer step's outcome: updated parameter tensors (keyed like the input) + new state. */
class OptimizerStep<S>(
    val params: Map<String, DTensor<*, F32>>,
    val state: S,
)

/**
 * The pure optimizer contract (decision 4): `step(params, grads, state)`
 * returns NEW parameter tensors and NEW state; nothing is mutated anywhere.
 * [initialState] is the empty state — per-key slots appear lazily on each
 * key's first visit, so a freshly-constructed optimizer needs no knowledge of
 * the parameter tree it will step.
 */
interface Optimizer<S> {
    fun initialState(): S

    fun step(
        params: List<NamedParameter>,
        grads: Map<String, DTensor<*, F32>>,
        state: S,
    ): OptimizerStep<S>
}

/**
 * The model-level convenience: step every parameter of [model] and rebuild it
 * via [Trainable.withParameters]. Returns the NEW model and the NEW state —
 * the training loop is a fold, exactly as decision 2 promised.
 */
fun <M, S> Optimizer<S>.step(
    model: M,
    grads: Map<String, DTensor<*, F32>>,
    state: S,
): Pair<M, S> where M : Trainable<M> {
    val out = step(model.parameters, grads, state)
    return model.withParameters(out.params) to out.state
}

// ---------------------------------------------------------------------------
// Shared host-math plumbing. `hostF32()` returns the BACKING array — inputs
// are read-only here; every output is a freshly allocated FloatArray.
// ---------------------------------------------------------------------------

private fun tensorOf(dims: IntArray, data: FloatArray): DTensor<*, F32> =
    DTensor<Shape, F32>(HostF32Storage(data), dims.copyOf(), F32)

private fun gradFor(p: NamedParameter, grads: Map<String, DTensor<*, F32>>): FloatArray {
    val g = requireNotNull(grads[p.key]) {
        "optimizer step: no gradient for parameter '${p.key}' (have: ${grads.keys})"
    }
    require(g.dims.contentEquals(p.tensor.dims)) {
        "optimizer step: gradient dims ${g.dims.toList()} != parameter '${p.key}' dims ${p.tensor.dims.toList()}"
    }
    return g.hostF32()
}

private fun requireKnownKeys(params: List<NamedParameter>, grads: Map<String, DTensor<*, F32>>) {
    val known = params.mapTo(HashSet()) { it.key }
    val unknown = grads.keys - known
    require(unknown.isEmpty()) {
        "optimizer step: gradients carry unknown keys $unknown (parameters: $known)"
    }
}

/**
 * DiffKT's `FixedLearningRateOptimizer(alpha)`: `t − α·g`, no state at all.
 * The F0 audit records no default for `alpha`, so none is offered here.
 */
class FixedLearningRate(val alpha: Float) : CheckpointableOptimizer<Unit> {

    override fun initialState() = Unit

    // §0.4.502 — checkpointing. This optimizer has NO state, so its
    // checkpoint is the kind tag alone. It is still checkpointable rather
    // than exempt: a training loop written against `Optimizer<S>` must be
    // able to save whatever optimizer it was handed, and a kind tag that
    // round-trips is what stops an `Adam` state from being loaded here.
    override val checkpointKind: String get() = "FixedLearningRate"

    override fun saveState(state: Unit): OptimizerCheckpoint = OptimizerCheckpoint(checkpointKind)

    override fun loadState(checkpoint: OptimizerCheckpoint) = Unit

    override fun step(
        params: List<NamedParameter>,
        grads: Map<String, DTensor<*, F32>>,
        state: Unit,
    ): OptimizerStep<Unit> {
        requireKnownKeys(params, grads)
        val updated = LinkedHashMap<String, DTensor<*, F32>>(params.size)
        for (p in params) {
            val t = p.tensor.hostF32()
            val g = gradFor(p, grads)
            updated[p.key] = tensorOf(p.tensor.dims, FloatArray(t.size) { i -> t[i] - alpha * g[i] })
        }
        return OptimizerStep(updated, Unit)
    }
}

/**
 * [SGD]'s state: completed-step count (drives the [SGD.lrDecay] schedule) +
 * the per-key velocity slots (present only when momentum ≠ 0, seeded on each
 * key's first visit).
 */
class SGDState internal constructor(
    val stepCount: Int,
    val velocity: Map<String, DTensor<*, F32>>,
)

/**
 * DiffKT's `SGDOptimizer(initialLearningRate = .001f, weightDecay = 0f,
 * momentum = 0f)`, formulas exact per F0 §4.0.2:
 * - momentum = 0: `v = g`; first visit of a key: `v = g` (seeded, stored);
 *   else `v' = μ·v + (1−μ)·g` — the EMA form, NOT PyTorch's `μ·v + g`.
 * - update: `t − lr·v'` with `lr = lr₀ / (1 + lrDecay·stepCount)` — DiffKT's
 *   `afterFit()` schedule, where `stepCount` counts COMPLETED steps (step 0
 *   runs at `lr₀`).
 * - [lrDecay] is DiffKT's `weightDecay`, renamed because it is LEARNING-RATE
 *   decay (`1/(1+d·k)` annealing), not L2 weight decay.
 */
class SGD(
    val initialLearningRate: Float = 0.001f,
    val lrDecay: Float = 0f,
    val momentum: Float = 0f,
) : CheckpointableOptimizer<SGDState> {

    init {
        require(momentum in 0f..1f) { "SGD: momentum must be in [0, 1] (got $momentum)" }
    }

    override fun initialState() = SGDState(0, emptyMap())

    override fun step(
        params: List<NamedParameter>,
        grads: Map<String, DTensor<*, F32>>,
        state: SGDState,
    ): OptimizerStep<SGDState> {
        requireKnownKeys(params, grads)
        val lr = initialLearningRate / (1f + lrDecay * state.stepCount)
        val updated = LinkedHashMap<String, DTensor<*, F32>>(params.size)
        val velocity: MutableMap<String, DTensor<*, F32>>? =
            if (momentum == 0f) null else HashMap(state.velocity)
        for (p in params) {
            val t = p.tensor.hostF32()
            val g = gradFor(p, grads)
            val v: FloatArray = if (velocity == null) {
                g // momentum = 0: velocity IS the gradient, nothing stored
            } else {
                val prev = state.velocity[p.key]
                val fresh =
                    if (prev == null) g.copyOf() // first visit: v = g, seeded and stored
                    else {
                        val pv = prev.hostF32()
                        FloatArray(g.size) { i -> momentum * pv[i] + (1f - momentum) * g[i] }
                    }
                velocity[p.key] = tensorOf(p.tensor.dims, fresh)
                fresh
            }
            updated[p.key] = tensorOf(p.tensor.dims, FloatArray(t.size) { i -> t[i] - lr * v[i] })
        }
        return OptimizerStep(updated, SGDState(state.stepCount + 1, velocity ?: emptyMap()))
    }

    // §0.4.502 — checkpointing. `stepCount` is load-bearing here and not
    // merely informational: it drives the [lrDecay] schedule, so a resume
    // that restarted it at 0 would silently raise the learning rate back to
    // `initialLearningRate` at the exact moment the run had annealed it.
    override val checkpointKind: String get() = "SGD"

    override fun saveState(state: SGDState): OptimizerCheckpoint = OptimizerCheckpoint(
        kind = checkpointKind,
        scalars = mapOf("stepCount" to state.stepCount.toString()),
        tensors = OptimizerCheckpoint.grouped("velocity", state.velocity),
    )

    override fun loadState(checkpoint: OptimizerCheckpoint): SGDState =
        SGDState(checkpoint.int("stepCount"), checkpoint.group("velocity"))
}

/** [RMSprop]'s state: the per-key mean-square slots, seeded `ms = g²` on first visit. */
class RMSpropState internal constructor(
    val meanSquare: Map<String, DTensor<*, F32>>,
)

/**
 * DiffKT's `RMSpropOptimizer(alpha = 0.005f, beta = 0.9f)`, exact per F0
 * §4.0.2: first visit `ms = g²`, else `ms' = β·ms + (1−β)·g²`; update
 * `t − α·g/√ms'`. DiffKT has NO epsilon — [eps] defaults to 0f (literal
 * parity) and, when nonzero, takes PyTorch's placement `√ms' + ε` (recorded:
 * a Tlaloc extension, never oracled against DiffKT).
 */
class RMSprop(
    val alpha: Float = 0.005f,
    val beta: Float = 0.9f,
    val eps: Float = 0f,
) : CheckpointableOptimizer<RMSpropState> {

    override fun initialState() = RMSpropState(emptyMap())

    override fun step(
        params: List<NamedParameter>,
        grads: Map<String, DTensor<*, F32>>,
        state: RMSpropState,
    ): OptimizerStep<RMSpropState> {
        requireKnownKeys(params, grads)
        val updated = LinkedHashMap<String, DTensor<*, F32>>(params.size)
        val meanSquare = HashMap(state.meanSquare)
        for (p in params) {
            val t = p.tensor.hostF32()
            val g = gradFor(p, grads)
            val prev = state.meanSquare[p.key]
            val ms: FloatArray = if (prev == null) {
                FloatArray(g.size) { i -> g[i] * g[i] }
            } else {
                val pm = prev.hostF32()
                FloatArray(g.size) { i -> beta * pm[i] + (1f - beta) * g[i] * g[i] }
            }
            meanSquare[p.key] = tensorOf(p.tensor.dims, ms)
            updated[p.key] = tensorOf(
                p.tensor.dims,
                FloatArray(t.size) { i -> t[i] - alpha * g[i] / (sqrt(ms[i]) + eps) },
            )
        }
        return OptimizerStep(updated, RMSpropState(meanSquare))
    }

    // §0.4.502 — checkpointing. No step count: RMSprop's update has none (it
    // is DiffKT's `t − α·g/√ms`, with no bias correction and no schedule), so
    // writing one would be inventing state the optimizer does not have.
    override val checkpointKind: String get() = "RMSprop"

    override fun saveState(state: RMSpropState): OptimizerCheckpoint = OptimizerCheckpoint(
        kind = checkpointKind,
        tensors = OptimizerCheckpoint.grouped("meanSquare", state.meanSquare),
    )

    override fun loadState(checkpoint: OptimizerCheckpoint): RMSpropState =
        RMSpropState(checkpoint.group("meanSquare"))
}

/**
 * [Adam]'s state: the shared completed-step count `t` (bias correction is
 * 1-based: the first step runs at t = 1) + per-key first/second-moment slots,
 * zero-seeded lazily. The step count is GLOBAL to the state, not per key —
 * the [Trainable] contract steps every parameter together, so per-key
 * counters could never disagree; recorded so nobody "fixes" it into disagreeing.
 */
class AdamState internal constructor(
    val stepCount: Int,
    val m: Map<String, DTensor<*, F32>>,
    val v: Map<String, DTensor<*, F32>>,
)

/**
 * Standard Kingma–Ba Adam, BIAS-CORRECTED — see the file KDoc for the
 * recorded finding: DiffKT's `AdamOptimizer` is a `TODO(...)` placeholder,
 * so this is a Tlaloc implementation of the paper, not a DiffKT parity item.
 *
 *   m' = β₁·m + (1−β₁)·g;  v' = β₂·v + (1−β₂)·g²;  t' = t + 1
 *   m̂ = m'/(1−β₁^t');      v̂ = v'/(1−β₂^t')
 *   θ' = θ − lr·m̂/(√v̂ + ε)
 */
class Adam(
    val learningRate: Float = 0.001f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val eps: Float = 1e-8f,
) : CheckpointableOptimizer<AdamState> {

    override fun initialState() = AdamState(0, emptyMap(), emptyMap())

    override fun step(
        params: List<NamedParameter>,
        grads: Map<String, DTensor<*, F32>>,
        state: AdamState,
    ): OptimizerStep<AdamState> {
        requireKnownKeys(params, grads)
        val t = state.stepCount + 1
        val c1 = 1f - beta1.pow(t)
        val c2 = 1f - beta2.pow(t)
        val updated = LinkedHashMap<String, DTensor<*, F32>>(params.size)
        val mSlots = HashMap(state.m)
        val vSlots = HashMap(state.v)
        for (p in params) {
            val theta = p.tensor.hostF32()
            val g = gradFor(p, grads)
            val pm = state.m[p.key]?.hostF32()
            val pv = state.v[p.key]?.hostF32()
            val m = FloatArray(g.size) { i -> beta1 * (pm?.get(i) ?: 0f) + (1f - beta1) * g[i] }
            val v = FloatArray(g.size) { i -> beta2 * (pv?.get(i) ?: 0f) + (1f - beta2) * g[i] * g[i] }
            mSlots[p.key] = tensorOf(p.tensor.dims, m)
            vSlots[p.key] = tensorOf(p.tensor.dims, v)
            updated[p.key] = tensorOf(
                p.tensor.dims,
                FloatArray(theta.size) { i -> theta[i] - learningRate * (m[i] / c1) / (sqrt(v[i] / c2) + eps) },
            )
        }
        return OptimizerStep(updated, AdamState(t, mSlots, vSlots))
    }

    // §0.4.502 — checkpointing. All three parts of Adam's state matter to a
    // resume, and the step count most of all: the bias corrections
    // `1−β₁ᵗ`/`1−β₂ᵗ` are ~0.1/0.001 at t=1 and ~1 by t=1000, so a resume
    // that restarted `t` would divide a fully-warmed first moment by 0.1 and
    // take a step ten times too large on the first batch after loading.
    override val checkpointKind: String get() = "Adam"

    override fun saveState(state: AdamState): OptimizerCheckpoint = OptimizerCheckpoint(
        kind = checkpointKind,
        scalars = mapOf("stepCount" to state.stepCount.toString()),
        tensors = OptimizerCheckpoint.grouped("m", state.m) + OptimizerCheckpoint.grouped("v", state.v),
    )

    override fun loadState(checkpoint: OptimizerCheckpoint): AdamState = AdamState(
        checkpoint.int("stepCount"),
        checkpoint.group("m"),
        checkpoint.group("v"),
    )
}

/**
 * DiffKT's `Momentum.kt` EMA helper (NOT an optimizer there — the F0 §4.0.2
 * finding): `this.momentumUpdated(new, momentum) = (1−momentum)·this +
 * momentum·new`. Momentum weights the NEW statistic — PyTorch's
 * `running_stats` convention (default 0.1 at the BatchNorm call site, F5's
 * consumer), NOT Flax's `1−momentum`. Float and tensor overloads, exactly
 * like the source.
 */
fun Float.momentumUpdated(new: Float, momentum: Float): Float =
    (1f - momentum) * this + momentum * new

fun DTensor<*, F32>.momentumUpdated(new: DTensor<*, F32>, momentum: Float): DTensor<*, F32> {
    require(dims.contentEquals(new.dims)) {
        "momentumUpdated: dims ${dims.toList()} != new dims ${new.dims.toList()}"
    }
    val a = hostF32()
    val b = new.hostF32()
    return tensorOf(dims, FloatArray(a.size) { i -> (1f - momentum) * a[i] + momentum * b[i] })
}
