package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.TestInstance
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Qwen3.5-0.8B, all 24 layers (18 Gated DeltaNet, 6 gated attention), through
 * Tlaloc's decode graph in the reference interpreter, against transformers.
 *
 * The oracle is the committed fixture `qwen3_5_0_8b_greedy.json`
 * (`harness/python/hf_greedy_fixture.py`, float32 on the CPU): a plain prompt
 * and a chat prompt, 16 greedy tokens each. The weights stay in the
 * HuggingFace cache and the test skips without them. They are staged f32 here
 * (the family's default is bf16), so both sides compute in f32 from the same
 * bf16-exact weights.
 *
 * Each prompt is prefilled in one call and decoded one token per call, the
 * linear layers' state carried in slot 0 of their pools. Certified: the ids,
 * and the logits within [LOGIT_REL_TOL] of the largest magnitude. The default
 * run checks [DEFAULT_TOKENS] tokens of each prompt; `TLALOC_QWEN35_FULL=1`
 * checks all 16.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class HfQwen35RealParityTest {

    private val fixture: JsonObject by lazy {
        parseJson(javaClass.getResourceAsStream("qwen3_5_0_8b_greedy.json")!!.readBytes().toString(Charsets.UTF_8)) as JsonObject
    }

    private val maxNew: Int by lazy {
        val all = (fixture["maxNew"] as JsonNumber).value.toInt()
        if (System.getenv("TLALOC_QWEN35_FULL") == "1") all else minOf(DEFAULT_TOKENS, all)
    }

    private var staged: Pair<HfDecoderConfig, List<FloatArray>>? = null

    private fun staged(dir: Path): Pair<HfDecoderConfig, List<FloatArray>> = staged ?: HfCheckpoint.open(dir).use { ckpt ->
        val config = ckpt.config.copy(weightDType = F32)
        config to HfStagedWeights.stage(ckpt, config)
    }.also { staged = it }

    @AfterAll
    fun release() {
        staged = null
    }

    private fun checkpointDir(): Path? {
        System.getenv("TLALOC_QWEN35_CHECKPOINT")?.let { return Path.of(it).takeIf { p -> Files.isDirectory(p) } }
        val rev = (fixture["revision"] as? JsonString)?.value ?: return null
        val dir = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots", rev)
        return dir.takeIf { Files.isRegularFile(it.resolve("config.json")) }
    }

    private class Prompt(
        val kind: String,
        val tokens: IntArray,
        val generated: IntArray,
        val topKIndices: IntArray,
        val topKValues: DoubleArray,
        val chosenLogits: DoubleArray,
    )

    private fun prompts(): List<Prompt> = (fixture["prompts"] as JsonArray).elements.map {
        val o = it as JsonObject
        fun ints(k: String) = (o[k] as JsonArray).elements.map { e -> (e as JsonNumber).value.toInt() }.toIntArray()
        fun dbls(k: String) = (o[k] as JsonArray).elements.map { e -> (e as JsonNumber).value }.toDoubleArray()
        Prompt(
            (o["kind"] as JsonString).value, ints("promptTokens"), ints("generatedTokens"),
            ints("step1TopKIndices"), dbls("step1TopKValues"), dbls("chosenLogits"),
        )
    }

    /** One prefill call, then one decode call per token, choosing the argmax. */
    private class Greedy(val config: HfDecoderConfig, val weights: List<FloatArray>) {
        val blockSize = 8
        val numBlocks = 6
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize, stateSlots = 2)
        val decodeContext = numBlocks * blockSize
        val decode: DxirFunction = build(DecodeGraphKind.DECODE, decodeContext)
        private val prefills = HashMap<Int, DxirFunction>()

        fun build(kind: DecodeGraphKind, context: Int): DxirFunction =
            HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)

        fun freshPools(): List<FloatArray> {
            val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, decodeContext))
            return (0 until config.numLayers).flatMap { l ->
                spec.poolTypesOf(l).toList().map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
            }
        }

        fun run(prompt: IntArray, maxNew: Int, slot: Int = 1): Pair<IntArray, List<FloatArray>> {
            var pools = freshPools()
            val context = ((prompt.size + blockSize - 1) / blockSize) * blockSize
            val prefill = prefills.getOrPut(context) { build(DecodeGraphKind.PREFILL, context) }
            val pad = context - prompt.size
            val out = DxirInterpreter.evalFunction(
                prefill,
                buildList {
                    add(FloatArray(context) { if (it < pad) 0f else prompt[it - pad].toFloat() })
                    add(FloatArray(context) { if (it < pad) 0f else (it - pad).toFloat() })
                    add(FloatArray(context / blockSize) { it.toFloat() })
                    add(floatArrayOf(prompt.size.toFloat()))
                    add(FloatArray(context) { if (it < pad) -1f else (it - pad).toFloat() })
                    add(floatArrayOf(slot.toFloat()))
                    addAll(pools)
                    addAll(weights)
                },
            )
            val logits = arrayListOf(out[0])
            pools = out.drop(1)
            val ids = IntArray(maxNew)
            ids[0] = argmax(out[0])
            for (i in 1 until maxNew) {
                val pos = prompt.size + i - 1
                val step = DxirInterpreter.evalFunction(
                    decode,
                    buildList {
                        add(floatArrayOf(ids[i - 1].toFloat()))
                        add(floatArrayOf(pos.toFloat()))
                        add(FloatArray(decodeContext / blockSize) { it.toFloat() })
                        add(floatArrayOf((pos + 1).toFloat()))
                        add(floatArrayOf(pos.toFloat()))
                        add(floatArrayOf(slot.toFloat()))
                        addAll(pools)
                        addAll(weights)
                    },
                )
                logits += step[0]
                pools = step.drop(1)
                ids[i] = argmax(step[0])
            }
            return ids to logits
        }

        private fun argmax(v: FloatArray): Int {
            var best = 0
            for (i in v.indices) if (v[i] > v[best]) best = i
            return best
        }
    }

    @Test
    fun configReadsTheHybridLayout() {
        val dir = checkpointDir()
        assumeTrue(dir != null, "no Qwen/Qwen3.5-0.8B checkpoint; fetch it with: ~/.local/venvs/vllm/bin/hf download Qwen/Qwen3.5-0.8B")
        HfCheckpoint.open(dir!!).use { ckpt ->
            val c = ckpt.config
            assertEquals(HfModelFamily.Qwen3_5, c.family)
            assertEquals(24, c.numLayers)
            assertEquals((0 until 24).filter { (it + 1) % 4 != 0 }, c.linearLayers)
            assertEquals(64, c.rotaryDim)
            assertTrue(c.tieWordEmbeddings)
            val unmapped = ckpt.verifyInventory()
            assertTrue(unmapped.all { it.startsWith("mtp.") || it.startsWith("model.visual.") }, "unmapped: ${unmapped.take(5)}")
        }
    }

    @Test
    fun qwen35GreedyDecodesHuggingFacesIdsForAPlainAndAChatPrompt() {
        val dir = checkpointDir()
        assumeTrue(dir != null, "no Qwen/Qwen3.5-0.8B checkpoint; fetch it with: ~/.local/venvs/vllm/bin/hf download Qwen/Qwen3.5-0.8B")
        val (config, weights) = staged(dir!!)
        val greedy = Greedy(config, weights)
        var worst = 0.0
        for (p in prompts()) {
            val (ids, logits) = greedy.run(p.tokens, maxNew)
            assertEquals(p.generated.take(maxNew), ids.toList(), "${p.kind} prompt: generated ids differ from transformers'")
            val first = logits[0]
            val denom = max(1.0, first.maxOf { abs(it.toDouble()) })
            for ((j, v) in p.topKIndices.withIndex()) {
                val rel = abs(first[v] - p.topKValues[j]) / denom
                worst = max(worst, rel)
                assertTrue(rel <= LOGIT_REL_TOL, "${p.kind}: step-1 logit[$v] tlaloc=${first[v]} hf=${p.topKValues[j]} rel=$rel")
            }
            for (i in ids.indices) {
                val d = max(1.0, logits[i].maxOf { abs(it.toDouble()) })
                val rel = abs(logits[i][ids[i]] - p.chosenLogits[i]) / d
                worst = max(worst, rel)
                assertTrue(rel <= LOGIT_REL_TOL, "${p.kind}: step ${i + 1} chosen logit rel=$rel")
            }
        }
        println("Qwen3.5-0.8B parity: ${prompts().size} prompts x $maxNew tokens, ids exact, worst relative logit difference $worst")
    }

    private companion object {
        const val LOGIT_REL_TOL = 1e-4
        const val DEFAULT_TOKENS = 4
    }
}
