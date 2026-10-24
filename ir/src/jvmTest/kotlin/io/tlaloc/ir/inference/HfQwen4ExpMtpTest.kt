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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The tiny Qwen4-Exp checkpoint's MTP head (`tiny_qwen4_exp/`, its drafts in
 * `mtp.json`: transformers' decoder layer, norms and gated residual wired as
 * vLLM's `Qwen4ExpMultiTokenPredictor`), speculative entries with 3 drafts
 * in the reference interpreter.
 *
 * The head reads the target's four streams before their last mix and passes
 * its own to the next draft; its layer is an indexed attention layer, and the
 * bucket's 48 positions are past what keeps every chunk, so the drafts go
 * through the indexer's selection. A random head almost never drafts the
 * target's token, so some steps verify the target's own continuation as
 * drafts: every step's drafts must equal the oracle's after the position the
 * step ends on, whether it accepted 0, 1 or 3 drafts, and the emitted ids
 * must be transformers' greedy ids.
 */
class HfQwen4ExpMtpTest {

    private val dir: Path = Path.of(javaClass.getResource("tiny_qwen4_exp/config.json")!!.toURI()).parent
    private fun json(name: String) = parseJson(Files.readString(dir.resolve(name))) as JsonObject
    private fun ints(o: JsonObject, k: String) = (o[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }

    private val staged: Pair<HfDecoderConfig, List<FloatArray>> by lazy {
        HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = DRAFTS)
            c to HfStagedWeights.stage(ckpt, c)
        }
    }

    private class Run(val config: HfDecoderConfig, val weights: List<FloatArray>) {
        val k = config.mtpDraftTokens
        val bs = 4
        val ctx = 48
        // Every state is committed once per call: a sequence holds two slots, its base and its target.
        val slots = 2
        val model = config.toDecodeModelShape(numBlocks = ctx / bs, blockSize = bs, stateSlots = slots)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx))
        val verify: DxirFunction = HfDecoderGraph.build(spec, config)
        var pools: List<FloatArray> = spec.inputs.drop(spec.kvPoolInputBase).take(spec.numPools).map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val chained = spec.chainRoles.size
        val tables = FloatArray(ctx / bs) { it.toFloat() }
        // The state commit (DecodeGraphSpec.stateCommitType): base, target, records to apply.
        var commit = intArrayOf(0, 1, 0)
        var position = 0
        var pending = -1
        var drafts = IntArray(0)
        var chain: List<FloatArray>? = null

        private fun call(fn: DxirFunction, ins: List<FloatArray>): List<FloatArray> {
            val out = DxirInterpreter.evalFunction(fn, buildList {
                add(ins[0]); add(ins[1]); add(tables); add(ins[2]); add(ins[3]); add(ins[4]); add(ins[5])
                addAll(pools); addAll(weights)
            })
            pools = out.drop(3 + chained)
            return out
        }

        fun prefill(prompt: IntArray): Pair<Int, IntArray> {
            val c = ((prompt.size + bs - 1) / bs) * bs
            val fn = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx), DecodeGraphKind.PREFILL, prefillChunk = c), config)
            val pad = c - prompt.size
            val out = call(
                fn,
                listOf(
                    FloatArray(c) { if (it < pad) 0f else prompt[it - pad].toFloat() },
                    FloatArray(c) { if (it < pad) 0f else (it - pad).toFloat() },
                    floatArrayOf(prompt.size.toFloat()),
                    FloatArray(c) { if (it < pad) -1f else (it - pad).toFloat() },
                    floatArrayOf(commit[0].toFloat()),
                    FloatArray(3) { commit[it].toFloat() },
                ),
            )
            commit = intArrayOf(commit[0], commit[1], 0)
            position = prompt.size
            pending = out[0][0].toInt()
            drafts = IntArray(k) { out[2][it].toInt() }
            return pending to drafts
        }

        /** One verify step of the pending token and [fed] as its drafts; returns the emitted tokens. */
        fun step(fed: IntArray, fromChain: Boolean): IntArray {
            val t = k + 1
            val tokens = intArrayOf(pending) + fed
            val pos = IntArray(t) { position + it }
            val built = listOf(
                FloatArray(t) { tokens[it].toFloat() },
                FloatArray(t) { pos[it].toFloat() },
                floatArrayOf((position + t).toFloat()),
                FloatArray(t) { pos[it].toFloat() },
                floatArrayOf(commit[0].toFloat()),
                FloatArray(3) { commit[it].toFloat() },
            )
            val c = chain
            val ins = if (fromChain && c != null) {
                for ((i, what) in listOf("tokens", "positions", "lengths", "slot mapping", "state slots", "state commit").withIndex()) {
                    assertEquals(built[i].toList(), c[i].toList(), "the chained $what")
                }
                c
            } else {
                built
            }
            val out = call(verify, ins)
            chain = out.subList(3, 3 + chained)
            val a = out[1][0].toInt()
            val emitted = IntArray(a + 1) { out[0][it].toInt() }
            commit = intArrayOf(commit[1], commit[0], a + 1)
            position += a + 1
            pending = emitted.last()
            drafts = IntArray(k) { out[2][it].toInt() }
            return emitted
        }
    }

    @Test
    fun theCheckpointsMtpHeadIsRead() {
        HfCheckpoint.open(dir).use { ckpt ->
            assertEquals(1, ckpt.config.mtpLayers)
            val spec = ckpt.config.mtpLayerSpec
            assertTrue(spec.gatedResidual && spec.indexer && !spec.ple)
            // With the head's roles every tensor is read but the vision encoder's.
            assertTrue(ckpt.verifyInventory(ckpt.config.copy(mtpDraftTokens = DRAFTS)).all { it.startsWith("model.visual.") })
        }
    }

    @Test
    fun aSpeculativePrefillGivesTheTargetsNextTokenAndTheHeadsDrafts() {
        val (config, weights) = staged
        val greedy = (json("greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }
        val mtp = (json("mtp.json")["prompts"] as JsonArray).elements.map { it as JsonObject }
        for ((g, m) in greedy.zip(mtp)) {
            val (next, drafts) = Run(config, weights).prefill(ints(g, "promptTokens").toIntArray())
            assertEquals(ints(g, "generatedTokens")[0], next, "the target's next token")
            assertEquals(oracle(m)[0], drafts.toList(), "the MTP head's drafts")
        }
    }

    @Test
    fun verifyStepsDraftTheOraclesTokensWhateverTheyAccept() = decode(chained = false)

    /** Each verify step after the first takes the last one's chained inputs; the forced drafts replace the chained tokens' drafts. */
    @Test
    fun chainedVerifyStepsDraftTheSame() = decode(chained = true)

    private fun oracle(m: JsonObject): List<List<Int>> =
        (m["drafts"] as JsonArray).elements.map { r -> (r as JsonArray).elements.map { (it as JsonNumber).value.toInt() } }

    private fun decode(chained: Boolean) {
        val (config, weights) = staged
        val greedy = (json("greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }
        val mtp = (json("mtp.json")["prompts"] as JsonArray).elements.map { it as JsonObject }
        val vocab = config.vocabSize
        for ((g, m) in greedy.zip(mtp)) {
            val prompt = ints(g, "promptTokens")
            val seq = prompt + ints(g, "generatedTokens")
            val want = oracle(m)
            val run = Run(config, weights)
            run.prefill(prompt.toIntArray())
            val accepted = ArrayList<Int>()
            var step = 0
            while (run.position < seq.size - 1) {
                val q = run.position
                // The head's drafts, the target's continuation, or its first token and a wrong one.
                val truth = IntArray(DRAFTS) { seq.getOrElse(q + 1 + it) { -1 } }
                val fed = when {
                    truth.any { it < 0 } || step % 3 == 0 -> run.drafts
                    step % 3 == 1 -> truth
                    else -> intArrayOf(truth[0], (truth[1] + 1) % vocab, truth[2])
                }
                // The chain carries the head's drafts; feeding others takes the host's inputs.
                val emitted = run.step(fed, fromChain = chained && fed === run.drafts)
                for ((i, e) in emitted.withIndex()) if (q + 1 + i < seq.size) assertEquals(seq[q + 1 + i], e, "the target's token at ${q + 1 + i}")
                accepted += emitted.size - 1
                if (run.position < seq.size) {
                    assertEquals(want[run.position - prompt.size], run.drafts.toList(), "drafts after position ${run.position}")
                }
                step++
            }
            assertTrue(accepted.contains(DRAFTS) && accepted.contains(1), "accepted counts $accepted")
            println("[qwen4exp-mtp] prompt of ${prompt.size}: drafts equal the oracle's after every step, accepted per step $accepted")
        }
    }

    private companion object {
        const val DRAFTS = 3
    }
}
