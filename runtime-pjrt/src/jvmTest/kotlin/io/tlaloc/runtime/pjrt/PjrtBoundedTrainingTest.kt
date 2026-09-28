package io.tlaloc.runtime.pjrt

import io.tlaloc.autograd.BoundedExecutor
import io.tlaloc.autograd.BucketLadders
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.boundedProgram
import io.tlaloc.autograd.broadcastAlong
import io.tlaloc.autograd.div
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.specOf
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Bounded
import io.tlaloc.core.DTensor
import io.tlaloc.core.DimBound
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object MaxSeqTrain : DimBound(64)

/**
 * A training loop over batches of varying length (docs/design/bounded-dims.md): a bounded
 * program's training step ([io.tlaloc.autograd.BoundedProgram.valueAndGrad]) run through a
 * [PjrtSession], padded to power-of-two buckets, compiles once per bucket, where running each
 * batch at its own length compiles once per distinct length.
 */
class PjrtBoundedTrainingTest {

    private val d = 8

    /** `sum over real rows of (x·W - t)^2 / len`. */
    private val loss = boundedProgram(
        "masked_mse",
        listOf(
            specOf<Rank2<Bounded<MaxSeqTrain>, Sym>>(F32, d),
            specOf<Rank2<Sym, Sym>>(F32, d, d),
            specOf<Rank2<Bounded<MaxSeqTrain>, Sym>>(F32, d),
        ),
        specOf<ScalarShape>(F32),
    ) { xs, ctx ->
        @Suppress("UNCHECKED_CAST")
        val r = ((xs[0] as Tracer<Rank2<Sym, Sym>>) matmul (xs[1] as Tracer<Rank2<Sym, Sym>>)) as Tracer<Shape> - xs[2]
        (r * r * r.broadcastAlong<Shape>(ctx.validMask(MaxSeqTrain), 0)).sum() / ctx.validLength(MaxSeqTrain)
    }

    private fun tensor(dims: IntArray, values: FloatArray) = DTensor<Shape, F32>(HostF32Storage(values), dims, F32)

    @Test
    fun `SGD over mixed lengths on the GPU compiles once per bucket and fits the weights`() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)

        val step = loss.valueAndGrad(listOf(1))
        val ladders = BucketLadders.powersOfTwo(step.bounds, minBucket = 8)
        var state = 11L
        fun rnd(): Float {
            state = state * 6364136223846793005L + 1442695040888963407L
            return ((state ushr 40).toInt() / (1 shl 24).toFloat()) * 2f - 1f
        }
        val wTrue = FloatArray(d * d) { rnd() }
        val lengths = List(40) { ((rnd() + 1f) * 32f).toInt().coerceIn(1, MaxSeqTrain.max) }
        val batches = lengths.map { n ->
            val x = FloatArray(n * d) { rnd() }
            val t = FloatArray(n * d) { i ->
                val row = i / d; val col = i % d
                var acc = 0f
                for (k in 0 until d) acc += x[row * d + k] * wTrue[k * d + col]
                acc
            }
            tensor(intArrayOf(n, d), x) to tensor(intArrayOf(n, d), t)
        }

        TestBackend.session().use { session ->
            val gpu = BoundedExecutor { t, args -> session.runOn(t.function, args, t.cacheKey) }
            var w = FloatArray(d * d)
            val losses = ArrayList<Float>()
            for ((k, batch) in batches.withIndex()) {
                val (x, t) = batch
                val inputs = listOf(x, tensor(intArrayOf(d, d), w), t)
                val (value, grad) = step.runBucketedAll(inputs, ladders, gpu)
                if (k == 0) {
                    // The first step against the interpreter: the dot runs in TF32 on the GPU.
                    val (v0, g0) = step.runAll(inputs)
                    val scale = maxOf(1f, g0.hostF32().maxOf { abs(it) })
                    for (i in g0.hostF32().indices) {
                        assertTrue(abs(g0.hostF32()[i] - grad.hostF32()[i]) <= 1e-2f * scale, "dW[$i]")
                    }
                    assertTrue(abs(v0.hostF32()[0] - value.hostF32()[0]) <= 1e-2f * maxOf(1f, v0.hostF32()[0]))
                }
                losses += value.hostF32()[0]
                val g = grad.hostF32()
                w = FloatArray(d * d) { w[it] - 0.1f * g[it] }
            }
            val buckets = lengths.map { ladders.bucketFor(MaxSeqTrain, it) }.distinct().size
            assertEquals(buckets, session.cacheSize, "one executable per bucket the batches touched")
            val early = losses.take(5).average()
            val late = losses.takeLast(5).average()
            assertTrue(late < early / 10, "loss $early -> $late")

            // The same batches at their own lengths: one executable per distinct length.
            val before = session.cacheSize
            for ((x, t) in batches) step.runAll(listOf(x, tensor(intArrayOf(d, d), w), t), gpu)
            val distinct = lengths.distinct().size
            assertEquals(before + distinct, session.cacheSize)
            println(
                "[bounded-train] ${lengths.size} SGD steps, lengths 1..${MaxSeqTrain.max} ($distinct distinct): " +
                    "$buckets executables for the buckets ${ladders.ladder(MaxSeqTrain)}, $distinct at exact lengths; " +
                    "loss %.4f -> %.6f".format(early, late),
            )
        }
    }
}
