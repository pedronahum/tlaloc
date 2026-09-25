package io.tlaloc.ir.inference

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
 * Qwen3-0.6B, all 28 layers, through Tlaloc's decode graph in the reference
 * interpreter, against HuggingFace transformers.
 *
 * The oracle is the committed fixture `qwen3_0_6b_greedy.json`, written by
 * `harness/python/hf_greedy_fixture.py`: transformers greedy-decoding 16
 * tokens in float32 on the CPU with eager attention, for a plain prompt and a
 * chat-template prompt. It holds token ids and logits only; the weights stay
 * in the HuggingFace cache and the test skips by name without them.
 *
 * Each prompt is prefilled in one call (a PREFILL entry, right-aligned) and
 * then decoded one token per call, choosing the argmax, exactly as a server
 * does. Certified:
 *
 * - the generated ids equal transformers' ids, for both prompts;
 * - the top-20 logits of the first generated position, and the chosen
 *   token's logit at every position, agree within [LOGIT_REL_TOL] of the
 *   largest logit magnitude.
 *
 * Both sides compute in float32 from bf16-exact weights on the CPU, so what
 * separates them is accumulation order over 28 layers.
 *
 * Cost: about ten seconds per interpreter call on the GB10, so the default
 * run checks the first [DEFAULT_TOKENS] generated tokens of each prompt
 * (about two minutes with staging). `TLALOC_QWEN3_FULL=1` checks all 16 of
 * both prompts (about six minutes).
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class HfQwen3RealParityTest {

    private val maxNew: Int by lazy {
        val all = (fixture["maxNew"] as JsonNumber).value.toInt()
        if (System.getenv("TLALOC_QWEN3_FULL") == "1") all else minOf(DEFAULT_TOKENS, all)
    }

    /** The checkpoint and its staged weights, read once for both tests. */
    private var staged: Pair<HfDecoderConfig, List<FloatArray>>? = null

    private fun staged(dir: Path): Pair<HfDecoderConfig, List<FloatArray>> =
        staged ?: HfCheckpoint.open(dir).use { ckpt -> ckpt.config to HfStagedWeights.stage(ckpt) }
            .also { staged = it }

    @AfterAll
    fun release() {
        staged = null
    }

    private val fixture: JsonObject by lazy {
        val text = javaClass.getResourceAsStream("qwen3_0_6b_greedy.json")!!
            .readBytes().toString(Charsets.UTF_8)
        parseJson(text) as JsonObject
    }

    private fun checkpointDir(): Path? {
        System.getenv("TLALOC_QWEN3_CHECKPOINT")?.let { p ->
            return Path.of(p).takeIf { Files.isRegularFile(it.resolve("model.safetensors")) }
        }
        val snapshots = Path.of(
            System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots",
        )
        val rev = (fixture["revision"] as? JsonString)?.value
        val dir = rev?.let { snapshots.resolve(it) } ?: return null
        return dir.takeIf { Files.isRegularFile(it.resolve("model.safetensors")) }
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

    private val blockSize = 8
    private val numBlocks = 6 // 48 positions: the 24-token chat prompt plus 16 generated fit

    /** A greedy decoder over the interpreter: one prefill call, then one call per token. */
    private class Greedy(
        val config: HfDecoderConfig,
        val weights: List<FloatArray>,
        val blockSize: Int,
        val numBlocks: Int,
    ) {
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)
        val decodeContext = numBlocks * blockSize
        val decode: DxirFunction = build(DecodeGraphKind.DECODE, decodeContext)
        private val prefills = HashMap<Int, DxirFunction>()

        fun build(kind: DecodeGraphKind, context: Int): DxirFunction = HfDecoderGraph.build(
            HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config,
        )

        fun run(prompt: IntArray, maxNew: Int): Pair<IntArray, List<FloatArray>> {
            val poolSize = numBlocks * blockSize * model.numKvHeads * model.headDim
            var pools: List<FloatArray> = List(2 * config.numLayers) { FloatArray(poolSize) }
            // Identity block table: slot = position.
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
                    addAll(pools)
                    addAll(weights)
                },
            )
            val logits = ArrayList<FloatArray>()
            logits += out[0]
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
    fun qwen3GreedyDecodesHuggingFacesIdsForAPlainAndAChatPrompt() {
        val dir = checkpointDir()
        assumeTrue(
            dir != null,
            "no Qwen/Qwen3-0.6B checkpoint in the HuggingFace cache (or TLALOC_QWEN3_CHECKPOINT); " +
                "fetch it with: ~/.local/venvs/vllm/bin/hf download Qwen/Qwen3-0.6B",
        )
        run {
            val (config, weights) = staged(dir!!)
            assertEquals(HfModelFamily.Qwen3, config.family)
            assertEquals((fixture["numHiddenLayers"] as JsonNumber).value.toInt(), config.numLayers)
            val greedy = Greedy(config, weights, blockSize, numBlocks)

            var worst = 0.0
            for (p in prompts()) {
                val (ids, logits) = greedy.run(p.tokens, maxNew)
                assertEquals(
                    p.generated.take(maxNew), ids.toList(),
                    "${p.kind} prompt: generated ids differ from transformers'",
                )
                val first = logits[0]
                val denom = max(1.0, first.maxOf { abs(it.toDouble()) })
                for ((j, v) in p.topKIndices.withIndex()) {
                    val rel = abs(first[v] - p.topKValues[j]) / denom
                    worst = max(worst, rel)
                    assertTrue(
                        rel <= LOGIT_REL_TOL,
                        "${p.kind}: step-1 logit[$v] tlaloc=${first[v]} hf=${p.topKValues[j]} rel=$rel",
                    )
                }
                for (i in ids.indices) {
                    val d = max(1.0, logits[i].maxOf { abs(it.toDouble()) })
                    val rel = abs(logits[i][ids[i]] - p.chosenLogits[i]) / d
                    worst = max(worst, rel)
                    assertTrue(
                        rel <= LOGIT_REL_TOL,
                        "${p.kind}: step ${i + 1} chosen logit tlaloc=${logits[i][ids[i]]} " +
                            "hf=${p.chosenLogits[i]} rel=$rel",
                    )
                }
            }
            println(
                "Qwen3-0.6B parity: ${prompts().size} prompts x $maxNew tokens, ids exact, " +
                    "worst relative logit difference $worst (tolerance $LOGIT_REL_TOL)",
            )
        }
    }

    /**
     * Negative control: the same model with the q/k norms replaced by unit
     * gains (that is, the Llama arithmetic on Qwen3's weights) must not
     * produce transformers' ids. Without this, a graph that skipped the
     * norms could only be caught by the logit tolerance.
     */
    @Test
    fun withoutItsQkNormGainsTheModelDoesNotDecodeHuggingFacesIds() {
        val dir = checkpointDir()
        assumeTrue(dir != null, "no Qwen/Qwen3-0.6B checkpoint in the HuggingFace cache")
        run {
            val (config, staged) = staged(dir!!)
            val roles = HfDecoderGraph.weightRoles(config)
            val weights = staged.mapIndexed { i, w ->
                val r = roles[i]
                val isQkNorm = r is DecoderWeightRole.Layer &&
                    (r.part == DecoderLayerPart.Q_NORM || r.part == DecoderLayerPart.K_NORM)
                if (isQkNorm) FloatArray(w.size) { 1f } else w
            }
            val greedy = Greedy(config, weights, blockSize, numBlocks)
            val p = prompts().first()
            val (ids, _) = greedy.run(p.tokens, 2)
            assertTrue(
                ids.toList() != p.generated.take(2),
                "unit q/k norm gains still gave transformers' first 2 ids ${ids.toList()}",
            )
        }
    }

    private companion object {
        /** Relative to the largest logit magnitude at that position. */
        const val LOGIT_REL_TOL = 1e-4

        /** Generated tokens checked per prompt unless TLALOC_QWEN3_FULL=1. */
        const val DEFAULT_TOKENS = 4
    }
}
