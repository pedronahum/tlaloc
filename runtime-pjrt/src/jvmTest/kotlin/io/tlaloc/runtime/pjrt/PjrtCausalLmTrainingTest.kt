package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CapturedStep
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.Precision
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.step
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A Llama-architecture [CausalLM]'s captured training step (I32 token ids in,
 * rebindable one-hot targets, grouped-query attention, RoPE, SwiGLU) compiled
 * by XLA and run on the GPU through [PjrtSession.runOn], against the
 * interpreter, and then trained with AdamW on the GPU.
 */
class PjrtCausalLmTrainingTest {

    private val config = CausalLmConfig(
        vocabSize = 32, dModel = 32, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 64, initStd = 0.1f,
    )
    private val batch = 4
    private val seq = 8

    private fun batchAt(k: Int): Pair<DTensor<*, I32>, DTensor<*, F32>> {
        // A repeating sequence per row: token t is followed by (3t + 1 + row) mod V.
        val ids = IntArray(batch * seq)
        val next = IntArray(batch * seq)
        for (r in 0 until batch) {
            var t = (r * 5 + k) % config.vocabSize
            for (s in 0 until seq) {
                ids[r * seq + s] = t
                t = (3 * t + 1 + r) % config.vocabSize
                next[r * seq + s] = t
            }
        }
        return DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(batch, seq), I32) to
            oneHot(next, config.vocabSize, intArrayOf(batch, seq))
    }

    private fun bind(step: CapturedStep, ids: DTensor<*, I32>, targets: DTensor<*, F32>, model: CausalLM): List<FloatArray> {
        val idv = (ids.storage as HostI32Storage).data
        return listOf(FloatArray(idv.size) { idv[it].toFloat() }, targets.hostF32()) +
            model.parameters.map { it.tensor.hostF32() }
    }

    @Test
    fun gradientsOnTheGpuMatchTheInterpreterAndAdamWTrainsThere() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)

        var model = CausalLM.llama(config, RandomKey.fromSeed(3))
        val (ids0, targets0) = batchAt(0)
        val step = capture(model, listOf(ids0), targets = listOf(targets0)) { logits, t -> crossEntropy(logits, t[0]) }

        TestBackend.session().use { session ->
            val values = bind(step, ids0, targets0, model)
            val want = DxirInterpreter.evalFunction(step.gradient, values)
            val got = session.runOn(step.gradient, values)
            assertEquals(want.size, got.size)
            // XLA runs f32 dots at TF32 by default, so compare against each
            // output's scale rather than element by element.
            var worst = 0f
            for (k in want.indices) {
                val scale = want[k].maxOfOrNull { abs(it) }?.coerceAtLeast(1e-6f) ?: continue
                for (i in want[k].indices) worst = maxOf(worst, abs(got[k][i] - want[k][i]) / scale)
            }
            println("[pjrt-causal-lm] loss GPU=${got[0][0]} host=${want[0][0]}; worst |diff|/max over ${want.size} outputs = $worst")
            assertTrue(worst < TestBackend.defaultDotRelTolerance, "${TestBackend.target} gradients differ from the interpreter by $worst of scale")
            assertTrue(abs(got[0][0] - want[0][0]) < 1e-3f * want[0][0])

            val optimizer = AdamW(learningRate = 3e-3f, weightDecay = 0.01f)
            var state = optimizer.initialState()
            val losses = ArrayList<Float>()
            repeat(60) { s ->
                val (ids, targets) = batchAt(s % 4)
                val outs = session.runOn(step.gradient, bind(step, ids, targets, model))
                losses += outs[0][0]
                val grads = step.parameterKeys.withIndex().associate { (j, key) ->
                    val dims = step.primal.params[step.inputCount + j].type.dims.toIntArray()
                    key to DTensor<Shape, F32>(HostF32Storage(outs[1 + step.inputCount + j]), dims, F32)
                }
                val (m, st) = optimizer.step(model, grads, state)
                model = m
                state = st
            }
            println("[pjrt-causal-lm] 60 AdamW steps on the GPU: loss ${losses.first()} -> ${losses.last()}, executables ${session.cacheSize}")
            assertEquals(1, session.cacheSize, "four target batches must share one compiled executable")
            assertTrue(losses.last() < 0.5f * losses.first(), "training did not reduce the loss: $losses")
        }
    }

    @Test
    fun theMixedPrecisionStepRunsOnTheGpu() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val model = CausalLM.llama(config, RandomKey.fromSeed(4))
        val (ids, targets) = batchAt(0)
        val step = capture(model, listOf(ids), targets = listOf(targets), precision = Precision.MIXED_BF16) { logits, t ->
            crossEntropy(logits, t[0])
        }
        val values = bind(step, ids, targets, model)
        val want = DxirInterpreter.evalFunction(step.gradient, values)
        TestBackend.session().use { session ->
            val got = session.runOn(step.gradient, values)
            println("[pjrt-causal-lm] bf16 step: loss GPU=${got[0][0]} host=${want[0][0]}")
            assertTrue(abs(got[0][0] - want[0][0]) < 2e-2f * want[0][0], "bf16 loss GPU ${got[0][0]} vs host ${want[0][0]}")
            assertTrue(got.all { o -> o.all { it.isFinite() } }, "non-finite bf16 gradients")
        }
    }
}
