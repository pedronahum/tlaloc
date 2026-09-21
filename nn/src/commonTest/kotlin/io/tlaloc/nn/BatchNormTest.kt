package io.tlaloc.nn

import io.tlaloc.core.F32
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.broadcastAlong
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.plus
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
 * §0.4.441 — F5: BatchNorm certification. The trace-vs-hand-built-DXIR oracle
 * pins the three NEW spellings (`sum(axes)`, `mean(axes)`, `broadcastAlong`)
 * against the interpreter's canonical attr spellings through the SAME reverse
 * transform (`==`); the batch-norm gradients pin against an independent
 * DOUBLE-precision spelling of the analytic formulas (the F3 precedent where
 * sqrt/div enter — `EPS = 1e-5f` keeps the arithmetic off the dyadic grid);
 * the running-stats EMA runs at momentum 0.25 (dyadic) and pins `==`.
 */
class BatchNormTest {

    // x [N=1, C=2, H=2, W=2] — quarter grid, n = 4 elements per channel.
    private val xData = floatArrayOf(
        1.0f, 0.5f, -0.25f, 0.75f, // ch0: sum 2.0, sumSq 1.875
        2.0f, -1.0f, 0.5f, 1.5f, // ch1: sum 3.0, sumSq 7.5
    )

    private fun input() = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(1, 2, 2, 2, xData)

    private val gammaData = floatArrayOf(0.5f, 2.0f)
    private val betaData = floatArrayOf(0.25f, -1.0f)

    private fun channel(vararg v: Float) = Tensors.f32Vector<Sym>(v)

    // ------------------------------------------------- the trace-vs-DXIR pin

    /**
     * The strongest oracle for the F5 spellings: `L = Σ(broadcastAlong(x.sum([0,2,3])
     * · w + x.mean([0,2,3]), axis 1) ⊙ x)` built through the Tracer AND directly
     * through [DxirBuilder] with the interpreter's exact attr spellings
     * (`reduction_dims`, `broadcast_dimensions = [1]`), BOTH through the SAME
     * reverse transform, every output elementwise `==`.
     */
    @Test
    fun tracedAxisOpsMatchHandBuiltDxirThroughTheSameTransform() {
        val wData = floatArrayOf(0.5f, -0.25f)
        val traced = captureN(listOf(input(), channel(*wData)), "axisOps") { leaves ->
            val x = leaves[0]
            val w = leaves[1]
            val s: Tracer<Shape> = x.sum(intArrayOf(0, 2, 3))
            val m: Tracer<Shape> = x.mean(intArrayOf(0, 2, 3))
            val t = s * w + m
            val u: Tracer<Shape> = x.broadcastAlong(t, 1)
            (u * x).sum()
        }

        val tx = DxirType(F32, listOf(1, 2, 2, 2))
        val tc = DxirType(F32, listOf(2))
        val reduce = mapOf("reduction_dims" to listOf(0, 2, 3))
        val hand = DxirBuilder.function("axisOpsHand") {
            val px = param("x", tx)
            val pw = param("w", tc)
            val s = op(OpKind.SUM, listOf(px), tc, attrs = reduce)
            val m = op(OpKind.MEAN, listOf(px), tc, attrs = reduce)
            val sw = op(OpKind.MUL, listOf(s, pw), tc)
            val t = op(OpKind.ADD, listOf(sw, m), tc)
            val u = op(
                OpKind.BROADCAST, listOf(t), tx,
                attrs = mapOf("broadcast_dimensions" to listOf(1)),
            )
            val ux = op(OpKind.MUL, listOf(u, px), tx)
            listOf(op(OpKind.SUM, listOf(ux), DxirType(F32, emptyList())))
        }

        val values = listOf(xData, wData)
        val fromTrace = DxirInterpreter.evalFunction(DxirReverseTransform.apply(traced), values)
        val fromHand = DxirInterpreter.evalFunction(DxirReverseTransform.apply(hand), values)

        assertEquals(fromHand.size, fromTrace.size)
        for (i in fromHand.indices) assertContentEquals(fromHand[i], fromTrace[i])
    }

    @Test
    fun axisReductionRefusesOutOfRangeAndEmptyAxes() {
        captureN(listOf(input()), "refusals") { leaves ->
            val x = leaves[0]
            assertFailsWith<IllegalArgumentException> { x.sum<Shape>(intArrayOf(4)) }
            assertFailsWith<IllegalArgumentException> { x.mean<Shape>(intArrayOf()) }
            assertFailsWith<IllegalArgumentException> {
                x.broadcastAlong<Shape>(channelTracer(x, floatArrayOf(1f, 2f, 3f)), 1)
            }
            leaves[0].sum()
        }
    }

    /**
     * A rank-1 leaf on the same tape, via the constant surface (the shape
     * checks are what's under test — constness is irrelevant here).
     */
    private fun channelTracer(on: Tracer<Shape>, v: FloatArray): Tracer<Shape> =
        on.constant(v, intArrayOf(v.size))

    // ----------------------------------------- the analytic gradient oracle

    /**
     * The full batch-norm gradient — statistics included — against an
     * independent double-precision spelling of the analytic formulas: with
     * `L = Σy²`, `g = 2y`, `dβ_c = Σg`, `dγ_c = Σg·x̂`, and
     * `dx = (γ/σ)·(g − mean(g) − x̂·mean(g·x̂))` (means over the n = N·H·W
     * elements of each channel — the μ/σ² dependence on x is exactly what
     * makes the last two terms exist).
     */
    @Test
    fun batchNormGradientsMatchTheAnalyticDoubleReference() {
        val bn = BatchNorm(channel(*gammaData), channel(*betaData), BatchNormStats.initial(2))
        val result = valueAndGradients(bn, listOf(input())) { y -> (y * y).sum() }

        val eps = BatchNorm.EPS.toDouble()
        val n = 4
        var lossRef = 0.0
        val dGamma = DoubleArray(2)
        val dBeta = DoubleArray(2)
        val dx = DoubleArray(8)
        for (c in 0 until 2) {
            val xs = DoubleArray(n) { xData[c * n + it].toDouble() }
            val mu = xs.sum() / n
            val variance = xs.sumOf { it * it } / n - mu * mu
            val sigma = kotlin.math.sqrt(variance + eps)
            val xhat = DoubleArray(n) { (xs[it] - mu) / sigma }
            val gamma = gammaData[c].toDouble()
            val beta = betaData[c].toDouble()
            val y = DoubleArray(n) { gamma * xhat[it] + beta }
            lossRef += y.sumOf { it * it }
            val g = DoubleArray(n) { 2.0 * y[it] }
            dBeta[c] = g.sum()
            dGamma[c] = (0 until n).sumOf { g[it] * xhat[it] }
            val gMean = g.sum() / n
            val gxMean = (0 until n).sumOf { g[it] * xhat[it] } / n
            for (i in 0 until n) {
                dx[c * n + i] = (gamma / sigma) * (g[i] - gMean - xhat[i] * gxMean)
            }
        }

        assertEquals(lossRef.toFloat(), result.loss, 1e-4f)
        val gGamma = result.gradients["gamma"]!!.hostF32()
        val gBeta = result.gradients["beta"]!!.hostF32()
        for (c in 0 until 2) {
            assertEquals(dGamma[c].toFloat(), gGamma[c], 1e-4f)
            assertEquals(dBeta[c].toFloat(), gBeta[c], 1e-4f)
        }
        val gx = result.inputGradients[0].hostF32()
        assertContentEquals(intArrayOf(1, 2, 2, 2), result.inputGradients[0].dims)
        for (i in 0 until 8) assertEquals(dx[i].toFloat(), gx[i], 1e-4f)
    }

    /**
     * The rank-2 `[N, C]` Dense-stack form, default γ=1/β=0, `L = Σy`: x̂ has
     * zero mean per channel, so `dβ = n` EXACTLY (the broadcast reverse is a
     * plain count) while dγ = Σx̂ and dx only vanish analytically — pinned at
     * tolerance through the f32 cancellation.
     */
    @Test
    fun rank2SumLossGradientsPinTheZeroMeanCancellation() {
        val x = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1f, 2f, 3f, 6f))
        val result = valueAndGradients(BatchNorm(2), listOf(x)) { y -> y.sum() }

        assertContentEquals(floatArrayOf(2f, 2f), result.gradients["beta"]!!.hostF32())
        val gGamma = result.gradients["gamma"]!!.hostF32()
        val gx = result.inputGradients[0].hostF32()
        for (c in 0 until 2) assertEquals(0f, gGamma[c], 1e-4f)
        for (i in 0 until 4) assertEquals(0f, gx[i], 1e-4f)
    }

    // --------------------------------------------------- the functional stats

    /**
     * Two training calls at momentum 0.25 (dyadic — every EMA step is exact),
     * from the fresh zero state: `running' = 0.75·running + 0.25·batch`,
     * pinned `==` at both steps, through BOTH stats surfaces (the traced
     * [BatchNorm.trainForward] pair and the host [BatchNorm.updatedStats]).
     */
    @Test
    fun runningStatsUpdatePinnedAcrossTwoTrainingCalls() {
        val bn = BatchNorm(2, momentum = 0.25f)

        // Call 1 through trainForward (the functional (y, stats) pair).
        var stats1: BatchNormStats? = null
        captureN(listOf(input(), bn.gamma, bn.beta), "bnTrain") { leaves ->
            val params = Params { key -> if (key == "gamma") leaves[1] else leaves[2] }
            val r = bn.trainForward(leaves[0], params)
            stats1 = r.stats
            // The pair's output IS the training forward (same trace).
            assertContentEquals(bn.forward(leaves[0], params).toDTensor().hostF32(), r.output.toDTensor().hostF32())
            r.output.sum()
        }
        val s1 = stats1!!
        assertEquals(1.0f, s1.runningN) // 0.75·0 + 0.25·4
        assertContentEquals(floatArrayOf(0.5f, 0.75f), s1.runningSum.hostF32()) // 0.25·[2, 3]
        assertContentEquals(floatArrayOf(0.46875f, 1.875f), s1.runningSumOfSquares.hostF32()) // 0.25·[1.875, 7.5]

        // Call 2 through the host stats step on a constant batch.
        val batch2 = Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(
            1, 2, 2, 2,
            floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f, 1f, 1f, 1f, 1f),
        )
        val s2 = bn.withStats(s1).updatedStats(batch2)
        assertEquals(1.75f, s2.runningN) // 0.75·1 + 0.25·4
        assertContentEquals(floatArrayOf(0.875f, 1.5625f), s2.runningSum.hostF32())
        assertContentEquals(floatArrayOf(0.6015625f, 2.40625f), s2.runningSumOfSquares.hostF32())

        // Purity: the original layer still holds the fresh zero state.
        assertEquals(0f, bn.stats.runningN)
    }

    @Test
    fun defaultMomentumIsPyTorchsPointOne() {
        assertEquals(0.1f, BatchNorm(2).momentum)
    }

    // ------------------------------------------------------- inference mode

    /**
     * `inferenceMode` = DiffKT's `freezeBatchNorm`: `m = γ/√(σ²+1e-5)`,
     * `b = β − m·μ` off the RUNNING stats (μ = [0.5, 1], σ² = [0.25, 1] by
     * construction), against the double reference; the frozen forward then
     * reproduces `m·x + b` on a batch.
     */
    @Test
    fun inferenceModeFreezesToTheRunningStatsChannelAffine() {
        val stats = BatchNormStats(4f, channel(2f, 4f), channel(2f, 8f))
        assertContentEquals(floatArrayOf(0.5f, 1f), stats.mean())
        assertContentEquals(floatArrayOf(0.25f, 1f), stats.variance())

        val bn = BatchNorm(channel(*gammaData), channel(*betaData), stats)
        val frozen = bn.inferenceMode()
        val eps = BatchNorm.EPS.toDouble()
        val mRef = DoubleArray(2) { gammaData[it] / sqrt(doubleArrayOf(0.25, 1.0)[it] + eps) }
        val bRef = DoubleArray(2) { betaData[it] - mRef[it] * doubleArrayOf(0.5, 1.0)[it] }
        for (c in 0 until 2) {
            assertEquals(mRef[c].toFloat(), frozen.m.hostF32()[c], 1e-6f)
            assertEquals(bRef[c].toFloat(), frozen.b.hostF32()[c], 1e-6f)
        }

        val result = valueAndGradients(frozen, listOf(input())) { y -> y.sum() }
        var lossRef = 0.0
        for (c in 0 until 2) for (i in 0 until 4) lossRef += mRef[c] * xData[c * 4 + i] + bRef[c]
        assertEquals(lossRef.toFloat(), result.loss, 1e-4f)
    }

    /**
     * [ChannelAffine] gradients hand-exact (`==`, dyadic grid), `L = Σy`:
     * `dm_c = Σ_channel x` (the `broadcast_dimensions = [1]` reverse — SUM over
     * [0, 2, 3]), `db_c = n`, `dx = m` splat along the channel axis.
     */
    @Test
    fun channelAffineGradientsHandExact() {
        val layer = ChannelAffine(channel(0.5f, 2f), channel(0.25f, -1f))
        val result = valueAndGradients(layer, listOf(input())) { y -> y.sum() }

        assertEquals(4.0f, result.loss) // ch0: 0.5·2+4·0.25 = 2; ch1: 2·3−4 = 2
        assertContentEquals(floatArrayOf(2f, 3f), result.gradients["m"]!!.hostF32())
        assertContentEquals(floatArrayOf(4f, 4f), result.gradients["b"]!!.hostF32())
        assertContentEquals(
            floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f, 2f, 2f, 2f, 2f),
            result.inputGradients[0].hostF32(),
        )
    }

    // ------------------------------------------------------------- refusals

    @Test
    fun refusalsAreLoud() {
        val bn = BatchNorm(3)
        // Channel mismatch (input carries 2 channels, layer expects 3).
        assertFailsWith<IllegalArgumentException> {
            valueAndGradients(bn, listOf(input())) { y -> y.sum() }
        }
        // Fresh stats: no mean/variance, no freeze.
        assertFailsWith<IllegalArgumentException> { bn.stats.mean() }
        assertFailsWith<IllegalArgumentException> { bn.inferenceMode() }
        // Structural checks.
        assertFailsWith<IllegalArgumentException> { BatchNorm(0) }
        assertFailsWith<IllegalArgumentException> {
            BatchNorm(channel(1f, 1f), channel(0f), BatchNormStats.initial(2))
        }
        assertFailsWith<IllegalArgumentException> { bn.withParameters(mapOf("scale" to channel(1f, 1f, 1f))) }
        val rebuilt = bn.withParameters(mapOf("gamma" to channel(2f, 2f, 2f)))
        assertContentEquals(floatArrayOf(2f, 2f, 2f), rebuilt.gamma.hostF32())
        assertContentEquals(floatArrayOf(1f, 1f, 1f), bn.gamma.hostF32()) // original untouched
        assertTrue(rebuilt.stats.runningN == 0f)
    }
}
