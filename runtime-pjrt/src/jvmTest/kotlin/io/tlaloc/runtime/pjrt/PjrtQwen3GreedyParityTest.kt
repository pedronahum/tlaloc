package io.tlaloc.runtime.pjrt

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
import kotlin.test.assertTrue
import io.tlaloc.stablehlo.toStablehlo

/**
 * Qwen3-0.6B's prefill and decode graphs, compiled by XLA and run on the
 * [TestBackend] (the GPU, or a TPU with `TLALOC_TEST_PJRT_TARGET=tpu`),
 * greedy-decode the ids transformers generated (`qwen3_0_6b_greedy.json`) for
 * a plain and a chat prompt.
 *
 * Self-skips without the backend or without the checkpoint (the Hugging Face
 * cache, or TLALOC_QWEN3_CHECKPOINT).
 */
class PjrtQwen3GreedyParityTest {

    private val fixturePath = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "qwen3_0_6b_greedy.json")
    private val blockSize = 8
    private val numBlocks = 6

    private fun checkpointDir(fixture: JsonObject): Path? {
        System.getenv("TLALOC_QWEN3_CHECKPOINT")?.let { return Path.of(it) }
        val rev = (fixture["revision"] as JsonString).value
        return Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots", rev)
            .takeIf { Files.isRegularFile(it.resolve("model.safetensors")) }
    }

    private class Setup(
        val prompts: List<Triple<String, List<Int>, List<Int>>>,
        val weights: List<FloatArray>,
        val numLayers: Int,
        val poolSize: Int,
        val decodeContext: Int,
        val build: (DecodeGraphKind, Int) -> DxirFunction,
    )

    private fun setup(): Setup {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeTrue(Files.isRegularFile(fixturePath), "fixture not found at $fixturePath")
        val fixture = parseJson(Files.readString(fixturePath)) as JsonObject
        val dir = checkpointDir(fixture)
        assumeTrue(dir != null, "no Qwen/Qwen3-0.6B checkpoint at the fixture's revision")
        val (config, weights) = HfCheckpoint.open(dir!!).use { it.config to HfStagedWeights.stage(it) }
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)
        val prompts = (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }.map { p ->
            fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
            Triple((p["kind"] as JsonString).value, ints("promptTokens"), ints("generatedTokens"))
        }
        return Setup(
            prompts, weights, config.numLayers, numBlocks * blockSize * model.numKvHeads * model.headDim,
            numBlocks * blockSize,
        ) { kind, context -> HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config) }
    }

    private fun greedy(s: Setup, session: PjrtSession, prompt: List<Int>, count: Int): List<Int> {
        fun argmax(v: FloatArray): Int { var b = 0; for (i in v.indices) if (v[i] > v[b]) b = i; return b }
        val decode = s.build(DecodeGraphKind.DECODE, s.decodeContext)
        var pools: List<FloatArray> = List(2 * s.numLayers) { FloatArray(s.poolSize) }
        val context = ((prompt.size + blockSize - 1) / blockSize) * blockSize
        val prefill = s.build(DecodeGraphKind.PREFILL, context)
        val pad = context - prompt.size
        val out = session.runOn(prefill, buildList {
            add(FloatArray(context) { if (it < pad) 0f else prompt[it - pad].toFloat() })
            add(FloatArray(context) { if (it < pad) 0f else (it - pad).toFloat() })
            add(FloatArray(context / blockSize) { it.toFloat() })
            add(floatArrayOf(prompt.size.toFloat()))
            add(FloatArray(context) { if (it < pad) -1f else (it - pad).toFloat() })
            addAll(pools)
            addAll(s.weights)
        })
        pools = out.drop(1)
        val got = arrayListOf(argmax(out[0]))
        while (got.size < count) {
            val pos = prompt.size + got.size - 1
            val step = session.runOn(decode, buildList {
                add(floatArrayOf(got.last().toFloat()))
                add(floatArrayOf(pos.toFloat()))
                add(FloatArray(s.decodeContext / blockSize) { it.toFloat() })
                add(floatArrayOf((pos + 1).toFloat()))
                add(floatArrayOf(pos.toFloat()))
                addAll(pools)
                addAll(s.weights)
            })
            pools = step.drop(1)
            got += argmax(step[0])
        }
        return got
    }

    @Test
    fun qwen3GreedyDecodesTransformersIdsOnTheBackend() {
        val s = setup()
        TestBackend.session().use { session ->
            for ((kind, prompt, want) in s.prompts) {
                val t0 = System.nanoTime()
                val got = greedy(s, session, prompt, want.size)
                println("[pjrt-qwen3:${TestBackend.target}] $kind: ${got.size} tokens in %.0f ms (compile included), ids ${if (got == want) "==" else "!="} transformers".format((System.nanoTime() - t0) / 1e6))
                assertEquals(want, got, "${TestBackend.target} greedy ids for the $kind prompt")
            }
        }
    }

    /**
     * The TPU lowering spells the prefill's f32 dot algorithms as HIGHEST
     * precision. Run on the GPU, that lowering decodes the same ids.
     */
    @Test
    fun theTpuDotSpellingDecodesTheSameIdsOnTheGpu() {
        assumeTrue(!TestBackend.isTpu, "on a TPU the first test already runs this lowering")
        val s = setup()
        val (kind, prompt, want) = s.prompts.first()
        val context = ((prompt.size + blockSize - 1) / blockSize) * blockSize
        val mlir = s.build(DecodeGraphKind.PREFILL, context).toStablehlo("")
        assertTrue("algorithm = <lhs_precision_type = f32" in mlir, "the prefill has no f32 dot algorithm to rewrite")
        assertTrue("algorithm =" !in io.tlaloc.stablehlo.portableF32Dots(mlir))
        TestBackend.session(portableF32Dots = true).use { session ->
            val got = greedy(s, session, prompt, want.size)
            println("[pjrt-qwen3:portable-dots] $kind: ids ${if (got == want) "==" else "!="} transformers")
            assertEquals(want, got)
        }
    }
}
