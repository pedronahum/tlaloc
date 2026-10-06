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
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Qwen3.5-0.8B with its MTP head, speculative entries (3 drafts) in the
 * reference interpreter.
 *
 * - A speculative prefill of the prompts of `qwen3_5_0_8b_mtp.json`
 *   (`harness/python/qwen35_mtp_fixture.py`: transformers' decoder layer with
 *   the checkpoint's `mtp.*` tensors) gives the target's next token and the
 *   head's three drafts.
 * - Greedy speculative decoding of the prompts of `qwen3_5_0_8b_greedy.json`
 *   gives transformers' 16 greedy ids whatever the drafts: a verify step
 *   emits the target's own tokens. It prints how many tokens each step
 *   emitted.
 */
class HfQwen35MtpTest {

    private fun json(name: String) =
        parseJson(javaClass.getResourceAsStream(name)!!.readBytes().toString(Charsets.UTF_8)) as JsonObject

    private fun ints(o: JsonObject, k: String) = (o[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }

    private fun checkpointDir(rev: String): Path? {
        val dir = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots", rev)
        return dir.takeIf { Files.isRegularFile(it.resolve("config.json")) }
    }

    private class Spec(val config: HfDecoderConfig, val weights: List<FloatArray>) {
        val k = config.mtpDraftTokens
        val bs = 8
        val numBlocks = 6
        val ctx = numBlocks * bs
        val slots = k + 2
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = bs, stateSlots = slots)
        val verify: DxirFunction = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)), config)
        val prefills = HashMap<Int, DxirFunction>()
        var pools: List<FloatArray> = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)).let { s ->
            s.inputs.drop(s.kvPoolInputBase).take(s.numPools).map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        }
        var readSlot = 0
        var position = 0
        var pending = -1
        var drafts = IntArray(0)

        /** The last verify step's chained inputs (DecodeGraphSpec.chainRoles), or null. */
        var chain: List<FloatArray>? = null
        private val chained = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)).chainRoles.size

        /** Prefill [prompt] in one right-aligned chunk; returns the target's next token and the drafts. */
        fun prefill(prompt: IntArray): Pair<Int, IntArray> {
            val c = ((prompt.size + bs - 1) / bs) * bs
            val fn = prefills.getOrPut(c) {
                // The bucket reaches past the prompt: the head writes its KV at the draft positions.
                HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx), DecodeGraphKind.PREFILL, prefillChunk = c), config)
            }
            val pad = c - prompt.size
            val tokens = IntArray(c) { if (it < pad) 0 else prompt[it - pad] }
            val pos = IntArray(c) { if (it < pad) 0 else it - pad }
            val out = DxirInterpreter.evalFunction(
                fn,
                buildList {
                    add(FloatArray(c) { tokens[it].toFloat() })
                    add(FloatArray(c) { pos[it].toFloat() })
                    add(FloatArray(ctx / bs) { it.toFloat() })
                    add(floatArrayOf(prompt.size.toFloat()))
                    add(FloatArray(c) { if (it < pad) -1f else (it - pad).toFloat() })
                    add(floatArrayOf(readSlot.toFloat()))
                    add(FloatArray(c) { -1f })
                    addAll(pools)
                    addAll(weights)
                },
            )
            pools = out.drop(3 + chained)
            position = prompt.size
            pending = out[0][0].toInt()
            drafts = IntArray(k) { out[2][it].toInt() }
            return pending to drafts
        }

        /**
         * One verify step; returns the tokens it emits. [fromChain]: its inputs are the
         * last verify step's chained ones, which first must equal what the host builds.
         */
        fun step(fromChain: Boolean = false): IntArray {
            val t = k + 1
            val tokens = intArrayOf(pending) + drafts
            val pos = IntArray(t) { position + it }
            var writes = (0 until slots).filter { it != readSlot }.take(t).toIntArray()
            val c = chain
            val built = listOf(
                FloatArray(t) { tokens[it].toFloat() },
                FloatArray(t) { pos[it].toFloat() },
                floatArrayOf((position + t).toFloat()),
                FloatArray(t) { pos[it].toFloat() },
                floatArrayOf(readSlot.toFloat()),
            )
            val ins = if (fromChain && c != null) {
                for ((i, what) in listOf("tokens", "positions", "lengths", "slot mapping", "state slots").withIndex()) {
                    assertEquals(built[i].toList(), c[i].toList(), "the chained $what")
                }
                // The chained write slots: the last step's, the new state slot replaced by the old one.
                val w = c[5].map { it.toInt() }
                assertEquals((0 until slots).toSet() - readSlot, w.toSet(), "the chained state write slots")
                writes = w.toIntArray()
                c
            } else {
                built + listOf(FloatArray(t) { writes[it].toFloat() })
            }
            val out = DxirInterpreter.evalFunction(
                verify,
                buildList {
                    add(ins[0])
                    add(ins[1])
                    add(FloatArray(ctx / bs) { it.toFloat() })
                    add(ins[2])
                    add(ins[3])
                    add(ins[4])
                    add(ins[5])
                    addAll(pools)
                    addAll(weights)
                },
            )
            chain = out.subList(3, 3 + chained)
            pools = out.drop(3 + chained)
            val a = out[1][0].toInt()
            val emitted = IntArray(a + 1) { out[0][it].toInt() }
            readSlot = writes[a]
            position += a + 1
            pending = emitted.last()
            drafts = IntArray(k) { out[2][it].toInt() }
            return emitted
        }
    }

    private fun spec(draftHead: WeightQuant = WeightQuant.NONE): Spec? {
        val rev = (json("qwen3_5_0_8b_mtp.json")["revision"] as JsonString).value
        val dir = checkpointDir(rev) ?: return null
        return HfCheckpoint.open(dir).use { ckpt ->
            val config = ckpt.config.copy(weightDType = F32, mtpDraftTokens = 3, mtpDraftHeadQuant = draftHead)
            Spec(config, HfStagedWeights.stage(ckpt, config))
        }
    }

    @Test
    fun aSpeculativePrefillGivesTheTargetsNextTokenAndTheHeadsDrafts() {
        val s = spec()
        assumeTrue(s != null, "no Qwen/Qwen3.5-0.8B checkpoint")
        for (p in (json("qwen3_5_0_8b_mtp.json")["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            val fresh = Spec(s!!.config, s.weights)
            val (next, drafts) = fresh.prefill(ints(p, "promptTokens").toIntArray())
            assertEquals((p["targetNext"] as JsonNumber).value.toInt(), next, "the target's next token")
            assertEquals(ints(p, "drafts"), drafts.toList(), "the MTP head's drafts")
            println("[qwen35-mtp] prompt ${ints(p, "promptTokens")}: next $next, drafts ${drafts.toList()}")
        }
    }

    @Test
    fun greedySpeculativeDecodingEmitsTheTargetsGreedyIds() = greedy(WeightQuant.NONE)

    /** Each verify step after the first takes the last one's chained inputs. */
    @Test
    fun chainedVerifyStepsEmitTheSameIds() = greedy(WeightQuant.NONE, chained = true)

    /** Drafts through an e4m3fn copy of the head: the same ids, the drafts' acceptance printed. */
    @Test
    fun greedySpeculativeDecodingWithAnFp8DraftHeadEmitsTheSameIds() = greedy(WeightQuant.FP8)

    private fun greedy(draftHead: WeightQuant, chained: Boolean = false) {
        val s = spec(draftHead)
        assumeTrue(s != null, "no Qwen/Qwen3.5-0.8B checkpoint")
        for (p in (json("qwen3_5_0_8b_greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            val want = ints(p, "generatedTokens")
            val run = Spec(s!!.config, s.weights)
            val got = arrayListOf(run.prefill(ints(p, "promptTokens").toIntArray()).first)
            val perStep = ArrayList<Int>()
            while (got.size < want.size) {
                val e = run.step(fromChain = chained)
                perStep += e.size
                got += e.toList()
            }
            assertEquals(want, got.take(want.size), "${(p["kind"] as JsonString).value}: speculative ids")
            println("[qwen35-mtp] draft head ${draftHead.tag}, ${(p["kind"] as JsonString).value}: ${want.size} ids equal transformers' in ${perStep.size} verify steps, tokens per step $perStep")
        }
    }
}
