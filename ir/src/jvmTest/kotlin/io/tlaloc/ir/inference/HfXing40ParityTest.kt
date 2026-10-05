package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Xing 4.0 truncated to its first 4 layers (2 dense, 2 mixtures of experts)
 * against the checkpoint's own modeling code in transformers
 * (`xing4_0_4layer_greedy.json`, `harness/python/hf_greedy_fixture.py
 * --trust-remote-code --num-layers 4`): each prompt prefilled in one call and
 * decoded token by token in the interpreter. The ids are transformers', the
 * first step's top logits and every chosen logit within [TOL].
 *
 * Stages about 10 GB of f32 weights: runs with TLALOC_XING_PARITY=1 and
 * `-PirTestHeap=24g`.
 */
class HfXing40ParityTest {

    private fun dir(rev: String): Path? {
        val d = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--XingChen-AGI--Xing4.0-29B-A4B/snapshots", rev)
        return d.takeIf { Files.isRegularFile(it.resolve("config.json")) }
    }

    @Test
    fun greedyDecodingOfFourLayersMatchesTransformers() {
        assumeTrue(System.getenv("TLALOC_XING_PARITY") == "1", "set TLALOC_XING_PARITY=1 (and -PirTestHeap=24g)")
        val fixture = parseJson(javaClass.getResourceAsStream("xing4_0_4layer_greedy.json")!!.readBytes().toString(Charsets.UTF_8)) as JsonObject
        val d = dir((fixture["revision"] as JsonString).value)
        assumeTrue(d != null, "no XingChen-AGI/Xing4.0-29B-A4B checkpoint at the fixture's revision")
        val layers = (fixture["numHiddenLayers"] as JsonNumber).value.toInt()
        val (config, weights) = HfCheckpoint.open(d!!).use { ckpt ->
            val c = ckpt.config.copy(numLayers = layers, weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 32
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        fun build(kind: DecodeGraphKind): DxirFunction =
            HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
        val prefill = build(DecodeGraphKind.PREFILL)
        val decode = build(DecodeGraphKind.DECODE)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context))
        var worst = 0.0
        for (p in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
            fun doubles(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value }
            val prompt = ints("promptTokens")
            val want = ints("generatedTokens")
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
                    addAll(pools)
                    addAll(weights)
                })
                pools = step.drop(1)
                logits += step[0]
                got += argmax(step[0])
            }
            // The fixture's logit of its own choice at each step, against ours at the same token.
            val chosenErr = doubles("chosenLogits").withIndex().map { (s, c) -> if (s < logits.size) abs(logits[s][want[s]] - c) else Double.NaN }
            println("[xing4-parity] prompt $prompt: |chosen logit difference| per step $chosenErr")
            assertEquals(want, got, "greedy ids for prompt $prompt")
            val top = ints("step1TopKIndices")
            val topV = doubles("step1TopKValues")
            for ((i, v) in top.withIndex()) {
                val e = abs(logits[0][v] - topV[i])
                worst = maxOf(worst, e)
                assertTrue(e <= TOL, "step 0 logit $v: ${logits[0][v]} vs ${topV[i]}")
            }
            for ((s, chosen) in doubles("chosenLogits").withIndex()) {
                val e = abs(logits[s][want[s]] - chosen)
                worst = maxOf(worst, e)
                assertTrue(e <= TOL, "step $s chosen logit: ${logits[s][want[s]]} vs $chosen")
            }
            println("[xing4-parity] prompt $prompt: ids $got equal transformers'")
        }
        println("[xing4-parity] worst |logit difference| $worst")
    }

    private companion object {
        const val TOL = 2e-3
    }
}
