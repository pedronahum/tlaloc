package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Qwen3.5-0.8B (Gated DeltaNet and gated attention layers) greedy-decodes
 * transformers' 16 ids for both prompts of `qwen3_5_0_8b_greedy.json` on the
 * device: one prefill call, then one decode call per token, the linear
 * layers' state carried in their pools between calls. Weights staged f32.
 */
class PjrtQwen35GreedyParityTest {

    private val fixturePath = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "qwen3_5_0_8b_greedy.json")
    private val blockSize = 8
    private val numBlocks = 6

    @Test
    fun qwen35GreedyDecodesTransformersIdsOnTheBackend() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeTrue(Files.isRegularFile(fixturePath), "fixture not found at $fixturePath")
        val fixture = parseJson(Files.readString(fixturePath)) as JsonObject
        val dir = Path.of(
            System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots",
            (fixture["revision"] as JsonString).value,
        )
        assumeTrue(Files.isRegularFile(dir.resolve("config.json")), "no Qwen/Qwen3.5-0.8B checkpoint at the fixture's revision")
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize, stateSlots = 2)
        val decodeContext = numBlocks * blockSize
        fun build(kind: DecodeGraphKind, context: Int): DxirFunction =
            HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, decodeContext))
        val decode = build(DecodeGraphKind.DECODE, decodeContext)
        fun argmax(v: FloatArray): Int { var b = 0; for (i in v.indices) if (v[i] > v[b]) b = i; return b }

        TestBackend.session().use { session ->
            for (p in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
                fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
                val prompt = ints("promptTokens")
                val want = ints("generatedTokens")
                val slot = 1f
                var pools: List<FloatArray> = (0 until config.numLayers).flatMap { l ->
                    spec.poolTypesOf(l).toList().map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
                }
                val context = ((prompt.size + blockSize - 1) / blockSize) * blockSize
                val pad = context - prompt.size
                val t0 = System.nanoTime()
                val out = session.runOn(build(DecodeGraphKind.PREFILL, context), buildList {
                    add(FloatArray(context) { if (it < pad) 0f else prompt[it - pad].toFloat() })
                    add(FloatArray(context) { if (it < pad) 0f else (it - pad).toFloat() })
                    add(FloatArray(context / blockSize) { it.toFloat() })
                    add(floatArrayOf(prompt.size.toFloat()))
                    add(FloatArray(context) { if (it < pad) -1f else (it - pad).toFloat() })
                    add(floatArrayOf(slot))
                    addAll(pools)
                    addAll(weights)
                })
                pools = out.drop(1)
                val got = arrayListOf(argmax(out[0]))
                while (got.size < want.size) {
                    val pos = prompt.size + got.size - 1
                    val step = session.runOn(decode, buildList {
                        add(floatArrayOf(got.last().toFloat()))
                        add(floatArrayOf(pos.toFloat()))
                        add(FloatArray(decodeContext / blockSize) { it.toFloat() })
                        add(floatArrayOf((pos + 1).toFloat()))
                        add(floatArrayOf(pos.toFloat()))
                        add(floatArrayOf(slot))
                        addAll(pools)
                        addAll(weights)
                    })
                    pools = step.drop(1)
                    got += argmax(step[0])
                }
                val kind = (p["kind"] as JsonString).value
                println(
                    "[pjrt-qwen35:${TestBackend.target}] $kind: ${got.size} tokens in %.0f ms (compile included), ids %s transformers"
                        .format((System.nanoTime() - t0) / 1e6, if (got == want) "==" else "!="),
                )
                assertEquals(want, got, "${TestBackend.target} greedy ids for the $kind prompt")
            }
        }
    }
}
