package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The tiny random Qwen3.5-MoE checkpoint of HfQwen35MoeParityTest on the
 * device: transformers' greedy ids for both prompts, every logit within 1e-3.
 */
class PjrtQwen35MoeParityTest {

    private val dir = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_qwen3_5_moe")

    @Test
    fun greedyDecodingMatchesTransformersOnTheDevice() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val fixture = parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 32
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs, stateSlots = 2)
        fun build(kind: DecodeGraphKind) =
            HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
        val prefill = build(DecodeGraphKind.PREFILL)
        val decode = build(DecodeGraphKind.DECODE)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context))
        var worst = 0.0
        TestBackend.session().use { s ->
            for (p in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
                fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
                val prompt = ints("promptTokens")
                val want = ints("generatedTokens")
                val wantLogits = (p["logits"] as JsonArray).elements.map { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
                var pools: List<FloatArray> = (0 until config.numLayers).flatMap { l ->
                    spec.poolTypesOf(l).toList().map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
                }
                val pad = context - prompt.size
                val tables = FloatArray(context / bs) { (1 + it).toFloat() }
                val out = s.runOn(prefill, buildList {
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
                    val step = s.runOn(decode, buildList {
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
                assertEquals(want, got, "greedy ids on ${TestBackend.target}")
                for ((st, row) in logits.withIndex()) {
                    for (v in row.indices) worst = maxOf(worst, abs(row[v] - wantLogits[st][v]))
                }
            }
        }
        println("[pjrt-qwen35moe] ids == transformers, worst |logit difference| $worst")
        assertTrue(worst <= 1e-3, "worst logit difference $worst")
    }
}
