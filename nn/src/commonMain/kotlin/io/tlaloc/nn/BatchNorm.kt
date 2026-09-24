/**
 * BatchNorm in DiffKT's exact training semantics (`BatchNormTraining` V2 + the `batchNorm` op),
 * on the compiler route: the batch statistics are DESUGARED into traced primitives (SUM with
 * `reduction_dims`, elementwise SUB/DIV/SQRT/MUL/ADD, the axis BROADCAST) — there is NO fused
 * BATCHNORM kind on the tape — so `DxirReverseTransform` differentiates the whole composition
 * through the existing registry rules. Zero gradient math in this file.
 *
 * Layout: the channel axis is AXIS 1 — Tlaloc-native NCHW, rank ≥ 2 with rank-2 `[N, C]` as the
 * Dense-stack form. DiffKT is C-last (NHWC); same maths, transposed layout, and the PyTorch
 * oracle is NCHW-native anyway.
 *
 * The running statistics are FUNCTIONAL state: DiffKT
 * wrote the pure `(output, newStats)` function and then wrapped it mutably
 * (their own TODO #172); we keep the pure function. V2's state triple is
 * (runningN, runningSum, runningSumOfSquares), each EMA'd through
 * [momentumUpdated] — momentum weights the NEW batch statistic (PyTorch's
 * `running_stats` convention), default 0.1f.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.broadcastAlong
import io.tlaloc.autograd.div
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.sqrt
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times

private fun channelTensor(data: FloatArray): DTensor<*, F32> =
    DTensor<Shape, F32>(HostF32Storage(data), intArrayOf(data.size), F32)

/** A rank-0 F32 tensor, the checkpoint form of a scalar buffer. */
private fun scalarTensor(v: Float): DTensor<*, F32> =
    DTensor<Shape, F32>(HostF32Storage(floatArrayOf(v)), IntArray(0), F32)

/**
 * DiffKT `BatchNormTraining` V2's running state, functionally held:
 * (runningN, runningSum `[C]`, runningSumOfSquares `[C]`), from which
 * inference statistics derive as `mean = runningSum / runningN`,
 * `var = runningSS / runningN − mean²`. Fresh state is all
 * zeros — statistics are undefined (0/0) until the first training update,
 * and [BatchNorm.inferenceMode] refuses loudly rather than emitting NaNs.
 */
class BatchNormStats(
    val runningN: Float,
    val runningSum: DTensor<*, F32>,
    val runningSumOfSquares: DTensor<*, F32>,
) {
    init {
        require(runningSum.dims.size == 1 && runningSumOfSquares.dims.contentEquals(runningSum.dims)) {
            "BatchNormStats: runningSum ${runningSum.dims.toList()} and runningSumOfSquares " +
                "${runningSumOfSquares.dims.toList()} must be rank-1 [C] of the same size"
        }
    }

    val numFeatures: Int get() = runningSum.dims[0]

    /** `runningSum / runningN` per channel. Requires at least one recorded batch. */
    fun mean(): FloatArray {
        requireTrained()
        val s = runningSum.hostF32()
        return FloatArray(s.size) { s[it] / runningN }
    }

    /** `runningSS / runningN − mean²` per channel (biased, like the batch statistic). */
    fun variance(): FloatArray {
        requireTrained()
        val m = mean()
        val q = runningSumOfSquares.hostF32()
        return FloatArray(q.size) { q[it] / runningN - m[it] * m[it] }
    }

    private fun requireTrained() {
        require(runningN > 0f) {
            "BatchNormStats: no batch statistics recorded yet (runningN = $runningN) — " +
                "run at least one training update before reading mean/variance"
        }
    }

    companion object {
        fun initial(numFeatures: Int): BatchNormStats {
            require(numFeatures > 0) { "BatchNormStats: numFeatures=$numFeatures must be positive" }
            return BatchNormStats(0f, channelTensor(FloatArray(numFeatures)), channelTensor(FloatArray(numFeatures)))
        }
    }
}

/** The functional pair DiffKT's `batchNormTrainV2` actually returns: (output, updated stats). */
class BatchNormTrainResult(
    val output: Tracer<Shape>,
    val stats: BatchNormStats,
)

/**
 * The per-channel affine layer `y = m·x + b` with `m, b` rank-1 `[C]`
 * broadcast along the channel axis (axis 1) — BatchNorm's frozen inference
 * form on the NCHW layout. DiffKT freezes to `AffineTransform(m, b)` and
 * leans on its implicit C-last broadcasting; our `AffineTransform` is
 * same-shape elementwise by design, so the channel-broadcast form is its own
 * (trainable, like DiffKT's frozen `TrainableTensor`s) layer. REJECTED:
 * widening `AffineTransform` with implicit broadcasting — the trace spelling
 * would silently depend on the input's rank, and the explicit
 * `broadcast_dimensions = [1]` op states the broadcast in the captured graph.
 */
class ChannelAffine(
    val m: DTensor<*, F32>,
    val b: DTensor<*, F32>,
) : TrainableLayer<ChannelAffine> {

    init {
        require(m.dims.size == 1 && b.dims.contentEquals(m.dims)) {
            "ChannelAffine: m ${m.dims.toList()} and b ${b.dims.toList()} must be rank-1 [C] of the same size"
        }
    }

    override val parameters: List<NamedParameter> =
        listOf(NamedParameter("m", m), NamedParameter("b", b))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): ChannelAffine {
        val unknown = updated.keys - setOf("m", "b")
        require(unknown.isEmpty()) { "ChannelAffine.withParameters: unknown keys $unknown" }
        return ChannelAffine(updated["m"] ?: m, updated["b"] ?: b)
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank >= 2 && x.dims[1] == m.dims[0]) {
            "ChannelAffine: input must be rank >= 2 with dim 1 = ${m.dims[0]} channels " +
                "(got dims ${x.dims.toList()})"
        }
        val mB: Tracer<Shape> = x.broadcastAlong(params["m"], 1)
        val bB: Tracer<Shape> = x.broadcastAlong(params["b"], 1)
        return mB * x + bB
    }
}

/**
 * DiffKT's BatchNorm (`BatchNormTraining` V2, the default variant), functional:
 *
 * - Trainable `gamma` (scale, init ones) and `beta` (shift, init zeros), both
 *   rank-1 `[C]` — DiffKT packs them as `scaleShift [2, C]`; two keyed tensors
 *   are the same parameters under our keyed-gradient contract.
 * - [forward] is the TRAINING forward (what DiffKT's `invoke` computes):
 *   per-channel batch statistics over all axes but the channel axis,
 *   `out = γ·(x − μ)/√(σ² + 1e-5) + β` with `μ = Σx/n`,
 *   `σ² = Σx²/n − μ²` (biased) — desugared into traced primitives, so the
 *   reverse transform differentiates through the statistics (the full
 *   batch-norm gradient, mean/var terms included).
 * - [trainForward] returns the functional `(output, updatedStats)` pair;
 *   [updatedStats] is the host-side stats step for a held [CapturedStep]
 *   (the stats are not differentiable state — they never enter the trace).
 * - [inferenceMode] freezes to [ChannelAffine] exactly like DiffKT's
 *   `freezeBatchNorm`: `m = γ/√(σ² + 1e-5)`, `b = β − m·μ` from the RUNNING
 *   statistics.
 *
 * Note on [forward] inside a container fold: the traced output is identical to
 * `trainForward(...).output`, but the stats update is discarded — a Sequential
 * cannot return per-layer state through the fold. Threading stats through
 * containers is not supported;
 * standalone use calls [updatedStats] per training batch.
 */
class BatchNorm(
    val gamma: DTensor<*, F32>,
    val beta: DTensor<*, F32>,
    val stats: BatchNormStats,
    val momentum: Float = 0.1f,
) : TrainableLayer<BatchNorm>, Stateful<BatchNorm> {

    init {
        require(gamma.dims.size == 1 && beta.dims.contentEquals(gamma.dims)) {
            "BatchNorm: gamma ${gamma.dims.toList()} and beta ${beta.dims.toList()} must be rank-1 [C] of the same size"
        }
        require(stats.numFeatures == gamma.dims[0]) {
            "BatchNorm: stats carry ${stats.numFeatures} channels but gamma has ${gamma.dims[0]}"
        }
    }

    val numFeatures: Int get() = gamma.dims[0]

    override val parameters: List<NamedParameter> =
        listOf(NamedParameter("gamma", gamma), NamedParameter("beta", beta))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): BatchNorm {
        val unknown = updated.keys - setOf("gamma", "beta")
        require(unknown.isEmpty()) { "BatchNorm.withParameters: unknown keys $unknown" }
        return BatchNorm(updated["gamma"] ?: gamma, updated["beta"] ?: beta, stats, momentum)
    }

    /**
     * The [Stateful] half: the running statistics as checkpoint
     * buffers. `runningN` is a RANK-0 tensor rather than a scalar channel of
     * its own, because safetensors has a rank-0 shape and a second scalar
     * channel would have been a second thing to keep in sync (see [Stateful]).
     *
     * Order is declaration order and the keys are the field names, so a
     * checkpoint's `buffer.runningSum` reads as the thing it is. [momentum] is
     * NOT a buffer: it is a hyperparameter of the layer, part of the model's
     * STRUCTURE, and a checkpoint that silently overwrote it would change what
     * the reloaded model does with its next batch.
     */
    override val buffers: List<NamedParameter> = listOf(
        NamedParameter("runningN", scalarTensor(stats.runningN)),
        NamedParameter("runningSum", stats.runningSum),
        NamedParameter("runningSumOfSquares", stats.runningSumOfSquares),
    )

    override fun withBuffers(updated: Map<String, DTensor<*, F32>>): BatchNorm {
        val known = setOf("runningN", "runningSum", "runningSumOfSquares")
        val unknown = updated.keys - known
        require(unknown.isEmpty()) { "BatchNorm.withBuffers: unknown keys $unknown (known: $known)" }
        val n = updated["runningN"]?.let {
            require(it.size == 1) {
                "BatchNorm.withBuffers: runningN must hold one element (got dims ${it.dims.toList()})"
            }
            it.hostF32()[0]
        } ?: stats.runningN
        return withStats(
            BatchNormStats(
                n,
                updated["runningSum"] ?: stats.runningSum,
                updated["runningSumOfSquares"] ?: stats.runningSumOfSquares,
            ),
        )
    }

    /** The functional stats rebuild — the state half of a training step. */
    fun withStats(newStats: BatchNormStats): BatchNorm {
        require(newStats.numFeatures == numFeatures) {
            "BatchNorm.withStats: ${newStats.numFeatures} channels != $numFeatures"
        }
        return BatchNorm(gamma, beta, newStats, momentum)
    }

    private fun checkInput(dims: IntArray) {
        require(dims.size >= 2 && dims[1] == numFeatures) {
            "BatchNorm: input must be rank >= 2 with channel axis 1 = $numFeatures " +
                "(NCHW; got dims ${dims.toList()})"
        }
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        checkInput(x.dims)
        val axes = IntArray(x.rank - 1) { if (it == 0) 0 else it + 1 } // all but axis 1
        val n = (x.size / numFeatures).toFloat()
        val sumX: Tracer<Shape> = x.sum(axes)
        val sumSq: Tracer<Shape> = (x * x).sum(axes)
        val mu = sumX / n
        val v = sumSq / n - mu * mu
        val denom = (v + EPS).sqrt()
        val xhat = (x - x.broadcastAlong<Shape>(mu, 1)) / x.broadcastAlong(denom, 1)
        return x.broadcastAlong<Shape>(params["gamma"], 1) * xhat + x.broadcastAlong(params["beta"], 1)
    }

    /** DiffKT `batchNormTrainV2`'s actual signature: the traced output AND the EMA'd stats. */
    fun trainForward(x: Tracer<Shape>, params: Params): BatchNormTrainResult =
        BatchNormTrainResult(forward(x, params), updatedStats(x.toDTensor()))

    /**
     * The host-side functional stats step: batch (n, Σx, Σx²) per channel from
     * [batch], each EMA'd into the running triple via [momentumUpdated]
     * (`running' = (1−momentum)·running + momentum·batchStat`). Accumulation
     * runs in flat row-major order — the same sequence the traced [sum]
     * spelling folds in, so the two never drift by a bit.
     */
    fun updatedStats(batch: DTensor<*, F32>): BatchNormStats {
        checkInput(batch.dims)
        val dims = batch.dims
        val values = batch.hostF32()
        var inner = 1
        for (k in 2 until dims.size) inner *= dims[k]
        val c = numFeatures
        val sum = FloatArray(c)
        val sumSq = FloatArray(c)
        for (flat in values.indices) {
            val ch = (flat / inner) % c
            val x = values[flat]
            sum[ch] += x
            sumSq[ch] += x * x
        }
        val batchN = (values.size / c).toFloat()
        return BatchNormStats(
            stats.runningN.momentumUpdated(batchN, momentum),
            stats.runningSum.momentumUpdated(channelTensor(sum), momentum),
            stats.runningSumOfSquares.momentumUpdated(channelTensor(sumSq), momentum),
        )
    }

    /**
     * DiffKT's `freezeBatchNorm`: `m = γ/√(σ² + 1e-5)`, `b = β − m·μ` from the
     * running statistics → the frozen [ChannelAffine]. Refuses on untrained
     * stats (0/0) rather than freezing NaNs.
     */
    fun inferenceMode(): ChannelAffine {
        val mean = stats.mean()
        val variance = stats.variance()
        val g = gamma.hostF32()
        val bt = beta.hostF32()
        val m = FloatArray(numFeatures) { g[it] / kotlin.math.sqrt(variance[it] + EPS) }
        val b = FloatArray(numFeatures) { bt[it] - m[it] * mean[it] }
        return ChannelAffine(channelTensor(m), channelTensor(b))
    }

    companion object {
        /** DiffKT's epsilon — a literal `1e-5f` inside the `batchNorm` op, not a knob. */
        const val EPS: Float = 1e-5f

        /** The DiffKT constructor surface: γ ones, β zeros (`scaleShift` init), fresh zero stats. */
        operator fun invoke(numFeatures: Int, momentum: Float = 0.1f): BatchNorm {
            require(numFeatures > 0) { "BatchNorm: numFeatures=$numFeatures must be positive" }
            return BatchNorm(
                channelTensor(FloatArray(numFeatures) { 1f }),
                channelTensor(FloatArray(numFeatures)),
                BatchNormStats.initial(numFeatures),
                momentum,
            )
        }
    }
}
