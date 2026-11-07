package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The committed `tiny_granite` checkpoint (embeddings times 12, attention scale 1/16, residual
 * adds times 0.22, logits divided by 8) read into a [CausalLM]: its logits are the serving
 * graph's, its gradients are finite differences of its loss, and it saves, reloads and takes
 * LoRA adapters with its multipliers kept.
 */
class HfGraniteCausalLmTest {
    private val dir: Path = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_granite")

    private fun idsTensor(ids: IntArray) = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(1, ids.size), I32)

    /** `[n, vocab]` logits of [ids] from the traced forward. */
    private fun logits(model: CausalLM, ids: IntArray): FloatArray {
        val params = model.parameters
        var out: Tracer<Shape>? = null
        captureN(listOf(idsTensor(ids)) + params.map { it.tensor }) { leaves ->
            val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
            model.forward(leaves[0], Params { byKey.getValue(it) }).also { out = it }
        }
        return FloatArray(out!!.size) { out!!.peek(it) }
    }

    /** The serving prefill graph's logits for the last token of [ids], the whole prompt in one call. */
    private fun servedLastLogits(ids: IntArray): FloatArray {
        val (config, staged) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 32
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context), DecodeGraphKind.PREFILL, prefillChunk = ids.size)
        val prefill = HfDecoderGraph.build(spec, config)
        val pools = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val out = DxirInterpreter.evalFunction(prefill, buildList {
            add(FloatArray(ids.size) { ids[it].toFloat() })
            add(FloatArray(ids.size) { it.toFloat() })
            add(FloatArray(context / bs) { (1 + it).toFloat() })
            add(floatArrayOf(ids.size.toFloat()))
            add(FloatArray(ids.size) { (bs + it).toFloat() })
            addAll(pools)
            addAll(staged)
        })
        return out[0]
    }

    @Test fun theCheckpointReadsWithItsMultipliers() {
        val m = HfCausalLm.load(dir).model
        assertEquals(12f, m.embedScale)
        assertEquals(1f / 8, m.logitScale)
        for (b in m.blocks) {
            assertEquals(0.0625f, b.attn.scale)
            assertEquals(0.22f, b.residualMultiplier)
            assertTrue(b.attn.rope != null && b.attn.qNorm == null && b.attn.gate == null)
        }
    }

    @Test fun logitsAreTheServingGraphs() {
        val lm = HfCausalLm.load(dir)
        val prompt = intArrayOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41, 8, 9)
        val all = logits(lm.model, prompt)
        var worst = 0f
        for (n in listOf(3, 7, prompt.size)) {
            val want = servedLastLogits(prompt.copyOf(n))
            for (v in 0 until VOCAB) {
                val got = all[(n - 1) * VOCAB + v]
                worst = maxOf(worst, abs(got - want[v]))
                assertTrue(abs(got - want[v]) <= 2e-4f, "prefix $n logit $v: $got vs the serving graph's ${want[v]}")
            }
        }
        println("tiny Granite CausalLM against the serving graph: worst |logit difference| $worst")
    }

    @Test fun gradientsAreFiniteDifferencesOfTheLoss() {
        val lm = HfCausalLm.load(dir)
        val ids = intArrayOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22)
        val next = IntArray(ids.size) { if (it + 1 < ids.size) ids[it + 1] else -1 }
        val step = captureCausalLm(lm.model, idsTensor(ids), idsTensor(next), chunk = 4)
        val got = step.run(lm.model, listOf(idsTensor(ids), idsTensor(next)))
        fun loss(model: CausalLM) = step.run(model, listOf(idsTensor(ids), idsTensor(next))).loss.toDouble()
        val byKey = lm.model.parameters.associate { it.key to it.tensor }
        val eps = 1e-3f
        for ((key, index) in listOf(
            "blocks.0.attn.q.w" to 5,
            "blocks.1.attn.k.w" to 17,
            "blocks.2.attn.o.w" to 3,
            "blocks.1.mlp.up.w" to 40,
            "blocks.0.mlpNorm.weight" to 7,
            "norm.weight" to 2,
            "head.w" to 9,
            "embed.table" to 4 * HIDDEN + 2,
        )) {
            val t = byKey.getValue(key)
            fun shifted(d: Float): CausalLM {
                val v = t.hostF32().copyOf().also { it[index] += d }
                return lm.model.withParameters(mapOf(key to DTensor<Shape, F32>(HostF32Storage(v), t.dims.copyOf(), F32)))
            }
            val fd = (loss(shifted(eps)) - loss(shifted(-eps))) / (2 * eps)
            val g = got.gradients.getValue(key).hostF32()[index].toDouble()
            assertTrue(abs(fd - g) <= 2e-3 + 3e-2 * abs(fd), "d loss / d $key[$index]: reverse mode $g, finite difference $fd")
        }
    }

    @Test fun itSavesAndReloadsWithTheSameLogits() {
        val lm = HfCausalLm.load(dir)
        val out = Files.createTempDirectory("tiny-granite-saved")
        try {
            lm.save(dir, out, dtype = F32)
            val prompt = intArrayOf(3, 5, 9, 12, 33, 7)
            val a = logits(lm.model, prompt)
            val b = logits(HfCausalLm.load(out).model, prompt)
            for (i in a.indices) assertEquals(a[i], b[i], "logit $i")
        } finally {
            out.toFile().deleteRecursively()
        }
    }

    @Test fun loraKeepsTheMultipliers() {
        val lm = HfCausalLm.load(dir)
        val adapted = Lora.apply(lm.model, LoraConfig(rank = 2, alpha = 4f, targetModules = listOf("q_proj", "v_proj", "down_proj")), RandomKey.fromSeed(3))
        val prompt = intArrayOf(3, 5, 9, 12, 33, 7)
        val a = logits(lm.model, prompt)
        val b = logits(adapted, prompt)
        for (i in a.indices) assertEquals(a[i], b[i], 1e-6f, "logit $i")
        assertEquals(lm.model.logitScale, adapted.logitScale)
        assertEquals(0.0625f, adapted.blocks[0].attn.scale)
        assertEquals(0.22f, adapted.blocks[0].residualMultiplier)
    }

    private companion object {
        const val VOCAB = 64
        const val HIDDEN = 32
    }
}
