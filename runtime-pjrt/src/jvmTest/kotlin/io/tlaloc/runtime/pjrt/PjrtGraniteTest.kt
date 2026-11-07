package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.nn.HfCausalLm
import io.tlaloc.nn.captureCausalLm
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The tiny Granite checkpoint (`tiny_granite`, its four multipliers away from 1) on the device
 * with XLA's own attention and matmuls: a prefill and greedy decode steps, and a training step
 * of the [io.tlaloc.nn.CausalLM] read from the checkpoint, each against the interpreter. The ids
 * must be the interpreter's; logits, pools and gradients agree within [TOLERANCE] of each
 * output's scale (the GB10 runs f32 dots at TF32).
 */
class PjrtGraniteTest {
    private val dir = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_granite")

    private fun worstRelative(want: List<FloatArray>, got: List<FloatArray>): Float {
        assertEquals(want.size, got.size)
        var worst = 0f
        for (k in want.indices) {
            val scale = want[k].maxOfOrNull { abs(it) }?.coerceAtLeast(1e-6f) ?: continue
            for (i in want[k].indices) worst = maxOf(worst, abs(got[k][i] - want[k][i]) / scale)
        }
        return worst
    }

    private fun argmax(v: FloatArray) = v.indices.maxByOrNull { v[it] }!!

    @Test
    fun prefillAndDecodeRunAsTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val (config, staged) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 32
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val bucket = DecodeBucket(1, context)
        val chunk = 8
        val prefill = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk), config)
        val decodeSpec = HfDecoderGraph.spec(config, model, bucket)
        val decode = HfDecoderGraph.build(decodeSpec, config)
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val prompt = listOf(3, 4, 50, 11, 12, 13)
        var worst = 0f
        TestBackend.session().use { s ->
            var pools: List<FloatArray> = decodeSpec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
            val pad = chunk - prompt.size
            val inputs = buildList {
                add(FloatArray(chunk) { if (it < pad) 0f else prompt[it - pad].toFloat() })
                add(FloatArray(chunk) { if (it < pad) 0f else (it - pad).toFloat() })
                add(tables)
                add(floatArrayOf(prompt.size.toFloat()))
                add(FloatArray(chunk) { if (it < pad) -1f else (bs + it - pad).toFloat() })
                addAll(pools)
                addAll(staged)
            }
            var want = DxirInterpreter.evalFunction(prefill, inputs)
            var got = s.runOn(prefill, inputs)
            worst = maxOf(worst, worstRelative(want, got))
            var n = prompt.size
            repeat(6) { step ->
                val next = argmax(want[0])
                assertEquals(next, argmax(got[0]), "step $step")
                pools = want.drop(1)
                val stepInputs = buildList {
                    add(floatArrayOf(next.toFloat()))
                    add(floatArrayOf(n.toFloat()))
                    add(tables)
                    add(floatArrayOf((n + 1).toFloat()))
                    add(floatArrayOf((bs + n).toFloat()))
                    addAll(pools)
                    addAll(staged)
                }
                want = DxirInterpreter.evalFunction(decode, stepInputs)
                got = s.runOn(decode, stepInputs)
                worst = maxOf(worst, worstRelative(want, got))
                n++
            }
        }
        assertTrue(worst < TOLERANCE, "outputs differ from the interpreter by $worst of scale")
        println("[pjrt-granite] prefill and six decode steps: ids == the interpreter, worst difference $worst of scale")
    }

    @Test
    fun aTrainingStepRunsAsTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val lm = HfCausalLm.load(dir)
        val ids = intArrayOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41)
        val next = IntArray(ids.size) { if (it + 1 < ids.size) ids[it + 1] else -1 }
        fun tensor(a: IntArray) = DTensor<Shape, I32>(HostI32Storage(a), intArrayOf(1, a.size), I32)
        val step = captureCausalLm(lm.model, tensor(ids), tensor(next), chunk = 4)
        val values = listOf(FloatArray(ids.size) { ids[it].toFloat() }, FloatArray(next.size) { next[it].toFloat() }) +
            step.parameterTensors(lm.model).map { (it.storage as io.tlaloc.core.HostF32Storage).data }
        TestBackend.session().use { s ->
            val want = DxirInterpreter.evalFunction(step.gradient, values)
            val got = s.runOn(step.gradient, values)
            assertTrue(abs(got[0][0] - want[0][0]) <= 1e-3f * abs(want[0][0]), "the loss: ${got[0][0]} against ${want[0][0]}")
            val worst = worstRelative(want, got)
            assertTrue(worst < TOLERANCE, "gradients differ from the interpreter by $worst of scale")
            println("[pjrt-granite] training step: loss ${got[0][0]} (interpreter ${want[0][0]}), worst gradient difference $worst of scale")
        }
    }

    private companion object {
        const val TOLERANCE = 0.025f
    }
}
