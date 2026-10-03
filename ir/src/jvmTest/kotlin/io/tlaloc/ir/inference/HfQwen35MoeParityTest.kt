package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The qwen3_5_moe family against transformers on a tiny random checkpoint
 * committed with its oracle (`tiny_qwen3_5_moe/`, written by
 * `harness/python/tiny_qwen35_moe_fixture.py`): three Gated DeltaNet layers and
 * one gated attention layer, every MLP a mixture of 8 experts (top 2) with a
 * shared expert. Two prompts, prefilled in one call and decoded token by token
 * in the interpreter: the ids are transformers', and every logit of every step
 * is within [TOL] of its.
 */
class HfQwen35MoeParityTest {

    private val dir: Path = Path.of(javaClass.getResource("tiny_qwen3_5_moe/config.json")!!.toURI()).parent
    private val fixture = parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject

    @Test
    fun theTinyCheckpointReadsAsAMixtureOfExperts() {
        HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config
            assertEquals(HfModelFamily.Qwen3_5Moe, c.family)
            assertEquals(MoeConfig(numExperts = 8, topK = 2, expertIntermediate = 8, sharedIntermediate = 8), c.moe)
            assertEquals(MlpKind.MOE, c.layer(0).mlp)
            assertEquals(MlpKind.MOE, c.layer(3).mlp)
            assertTrue(ckpt.verifyInventory().all { it.startsWith("model.visual.") || it.startsWith("mtp.") })
        }
    }

    @Test
    fun greedyDecodingMatchesTransformersLogitForLogit() {
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 32
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs, stateSlots = 2)
        fun build(kind: DecodeGraphKind): DxirFunction =
            HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
        val prefill = build(DecodeGraphKind.PREFILL)
        val decode = build(DecodeGraphKind.DECODE)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context))
        var worst = 0.0
        for (p in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
            val prompt = ints("promptTokens")
            val want = ints("generatedTokens")
            val wantLogits = (p["logits"] as JsonArray).elements.map { row ->
                (row as JsonArray).elements.map { (it as JsonNumber).value }
            }
            var pools: List<FloatArray> = (0 until config.numLayers).flatMap { l ->
                spec.poolTypesOf(l).toList().map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
            }
            val pad = context - prompt.size
            val tables = FloatArray(context / bs) { (1 + it).toFloat() }
            val out = DxirInterpreter.evalFunction(prefill, buildList {
                add(FloatArray(context) { if (it < pad) 0f else prompt[it - pad].toFloat() })
                add(FloatArray(context) { if (it < pad) 0f else (it - pad).toFloat() })
                add(tables)
                add(floatArrayOf(prompt.size.toFloat()))
                add(FloatArray(context) { if (it < pad) -1f else (bs + it - pad).toFloat() })
                add(floatArrayOf(1f))
                addAll(pools)
                addAll(weights)
            })
            pools = out.drop(1)
            val logits = arrayListOf(out[0])
            fun argmax(v: FloatArray) = v.indices.maxByOrNull { v[it] }!!
            val got = arrayListOf(argmax(out[0]))
            while (got.size < want.size) {
                val pos = prompt.size + got.size - 1
                val step = DxirInterpreter.evalFunction(decode, buildList {
                    add(floatArrayOf(got.last().toFloat()))
                    add(floatArrayOf(pos.toFloat()))
                    add(tables)
                    add(floatArrayOf((pos + 1).toFloat()))
                    add(floatArrayOf((bs + pos).toFloat()))
                    add(floatArrayOf(1f))
                    addAll(pools)
                    addAll(weights)
                })
                pools = step.drop(1)
                logits += step[0]
                got += argmax(step[0])
            }
            assertEquals(want, got, "greedy ids for prompt $prompt")
            for ((s, row) in logits.withIndex()) {
                for (v in row.indices) {
                    val e = abs(row[v] - wantLogits[s][v])
                    worst = maxOf(worst, e)
                    assertTrue(e <= TOL, "step $s logit $v: ${row[v]} vs ${wantLogits[s][v]}")
                }
            }
        }
        println("tiny Qwen3.5-MoE parity: ids exact, worst |logit difference| $worst")
    }

    private companion object {
        const val TOL = 1e-4
    }
}
