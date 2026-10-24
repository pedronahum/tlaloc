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
        // Every state is committed once per call: a sequence holds two slots, its base and its target.
        val slots = 2
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = bs, stateSlots = slots)

        /** The verify entry of [width] drafts (DecodeGraphSpec.verifyDrafts below the model's k). */
        private fun verifySpec(width: Int) = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx), verifyDrafts = width.takeIf { it < k })
        private val verifies = HashMap<Int, DxirFunction>()
        fun verify(width: Int): DxirFunction = verifies.getOrPut(width) {
            val s = verifySpec(width)
            // Every width binds the same pools.
            assertEquals(verifySpec(k).inputs.drop(s.kvPoolInputBase).take(s.numPools), s.inputs.drop(s.kvPoolInputBase).take(s.numPools))
            HfDecoderGraph.build(s, config)
        }
        val prefills = HashMap<Int, DxirFunction>()
        var pools: List<FloatArray> = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)).let { s ->
            s.inputs.drop(s.kvPoolInputBase).take(s.numPools).map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        }
        // The state commit (DecodeGraphSpec.stateCommitType): base, target, records to apply.
        var commit = intArrayOf(0, 1, 0)
        var position = 0
        var pending = -1
        var drafts = IntArray(0)

        /** The last verify step's chained inputs (DecodeGraphSpec.chainRoles), or null. */
        var chain: List<FloatArray>? = null

        /** The drafts of the last verify step (its width), or 0 after a prefill. */
        var lastWidth = 0
        private val chained = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)).chainRoles.size

        /**
         * Prefill [prompt] at [position] in one right-aligned chunk; returns the target's next
         * token and the drafts. After verify steps the chunk starts with the pending token.
         */
        fun prefill(prompt: IntArray): Pair<Int, IntArray> {
            val c = ((prompt.size + bs - 1) / bs) * bs
            val fn = prefills.getOrPut(c) {
                // The bucket reaches past the prompt: the head writes its KV at the draft positions.
                HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx), DecodeGraphKind.PREFILL, prefillChunk = c), config)
            }
            val pad = c - prompt.size
            val tokens = IntArray(c) { if (it < pad) 0 else prompt[it - pad] }
            val pos = IntArray(c) { if (it < pad) 0 else position + it - pad }
            val out = DxirInterpreter.evalFunction(
                fn,
                buildList {
                    add(FloatArray(c) { tokens[it].toFloat() })
                    add(FloatArray(c) { pos[it].toFloat() })
                    add(FloatArray(ctx / bs) { it.toFloat() })
                    add(floatArrayOf((position + prompt.size).toFloat()))
                    add(FloatArray(c) { if (it < pad) -1f else pos[it].toFloat() })
                    add(floatArrayOf(commit[0].toFloat()))
                    add(FloatArray(3) { commit[it].toFloat() })
                    addAll(pools)
                    addAll(weights)
                },
            )
            pools = out.drop(3 + chained)
            commit = intArrayOf(commit[0], commit[1], 0)
            position += prompt.size
            pending = out[0][0].toInt()
            drafts = IntArray(k) { out[2][it].toInt() }
            chain = null
            lastWidth = 0
            return pending to drafts
        }

        /**
         * One verify step; returns the tokens it emits. [fromChain]: its inputs are the
         * last verify step's chained ones, which first must equal what the host builds.
         */
        fun step(fromChain: Boolean = false, width: Int = k): IntArray {
            val t = width + 1
            // The step checks the first [width] drafts; after a narrower step the missing ones are
            // the last draft again (a guess the target rejects or accepts as it would any draft).
            val tokens = intArrayOf(pending) + IntArray(width) { drafts.getOrElse(it) { drafts.last() } }
            val pos = IntArray(t) { position + it }
            val c = chain
            val built = listOf(
                FloatArray(t) { tokens[it].toFloat() },
                FloatArray(t) { pos[it].toFloat() },
                floatArrayOf((position + t).toFloat()),
                FloatArray(t) { pos[it].toFloat() },
                floatArrayOf(commit[0].toFloat()),
                FloatArray(3) { commit[it].toFloat() },
            )
            val ins = if (fromChain && c != null && lastWidth == width) {
                for ((i, what) in listOf("tokens", "positions", "lengths", "slot mapping", "state slots", "state commit").withIndex()) {
                    assertEquals(built[i].toList(), c[i].toList(), "the chained $what")
                }
                c
            } else {
                built
            }
            val out = DxirInterpreter.evalFunction(
                verify(width),
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
            commit = intArrayOf(commit[1], commit[0], a + 1)
            position += a + 1
            pending = emitted.last()
            drafts = IntArray(width) { out[2][it].toInt() }
            lastWidth = width
            return emitted
        }
    }

    private fun spec(draftHead: WeightQuant = WeightQuant.NONE, drafts: Int = 3): Spec? {
        val rev = (json("qwen3_5_0_8b_mtp.json")["revision"] as JsonString).value
        val dir = checkpointDir(rev) ?: return null
        return HfCheckpoint.open(dir).use { ckpt ->
            val config = ckpt.config.copy(weightDType = F32, mtpDraftTokens = drafts, mtpDraftHeadQuant = draftHead)
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

    /**
     * A prompt continued after verify steps (a chat's next turn) applies the last step's
     * accepted records in its prefill call: the next token and drafts equal those of one
     * prefill of the whole history.
     */
    @Test
    fun aPrefillAfterVerifyStepsGivesTheTokenAndDraftsOfOnePrefillOfTheHistory() {
        val s = spec()
        assumeTrue(s != null, "no Qwen/Qwen3.5-0.8B checkpoint")
        val p = (json("qwen3_5_0_8b_greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }.first()
        val prompt = ints(p, "promptTokens").toIntArray()
        val turn = prompt.take(3).toIntArray()
        val run = Spec(s!!.config, s.weights)
        val got = arrayListOf(run.prefill(prompt).first)
        repeat(2) { got += run.step().toList() }
        val (next, drafts) = run.prefill(intArrayOf(run.pending) + turn)
        val whole = Spec(s.config, s.weights)
        val (wantNext, wantDrafts) = whole.prefill(prompt + got.toIntArray() + turn)
        assertEquals(wantNext, next, "the next token after the second turn")
        assertEquals(wantDrafts.toList(), drafts.toList(), "the drafts after the second turn")
        println("[qwen35-mtp] a prefill after 2 verify steps (${got.size} tokens) gives one prefill's next token $next and drafts ${drafts.toList()}")
    }

    /**
     * An artifact of 5 drafts with verify entries of 5 and 3: a step of 3 drafts gives, bit for
     * bit, the tokens, accepted drafts and next drafts of an artifact of 3 drafts (its records
     * pools hold 6 records per slot and the step writes 4), and the first 3 of its prefill's 5
     * drafts are the 3-draft prefill's.
     */
    @Test
    fun aThreeDraftStepInAFiveDraftArtifactIsTheThreeDraftArtifactsStep() {
        val three = spec()
        assumeTrue(three != null, "no Qwen/Qwen3.5-0.8B checkpoint")
        val fiveConfig = three!!.config.copy(mtpDraftTokens = 5)
        for (p in (json("qwen3_5_0_8b_greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            val a = Spec(three.config, three.weights)
            val b = Spec(fiveConfig, three.weights)
            val prompt = ints(p, "promptTokens").toIntArray()
            val (na, da) = a.prefill(prompt)
            val (nb, db) = b.prefill(prompt)
            assertEquals(na, nb, "the prefill's next token")
            assertEquals(da.toList(), db.take(3), "the first 3 of the 5 drafts")
            repeat(4) { i ->
                val ea = a.step(fromChain = true)
                val eb = b.step(fromChain = true, width = 3)
                assertEquals(ea.toList(), eb.toList(), "step $i: the emitted tokens")
                assertEquals(a.drafts.toList(), b.drafts.toList(), "step $i: the next drafts")
            }
            println("[qwen35-mtp] ${(p["kind"] as JsonString).value}: 3-draft steps of a 5-draft artifact equal a 3-draft artifact's (5 prefill drafts ${db.toList()})")
        }
    }

    /**
     * Greedy decoding through an artifact of 5 drafts whose verify steps switch between 5 and 3
     * drafts (5, 5, 3, 3, 5, ...; chained where a step has its predecessor's width): transformers'
     * ids.
     */
    @Test
    fun switchingTheDraftWidthMidGenerationEmitsTheTargetsGreedyIds() {
        val s = spec(drafts = 5)
        assumeTrue(s != null, "no Qwen/Qwen3.5-0.8B checkpoint")
        val schedule = intArrayOf(5, 5, 3, 3, 5, 3)
        for (p in (json("qwen3_5_0_8b_greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            val want = ints(p, "generatedTokens")
            val run = Spec(s!!.config, s.weights)
            val got = arrayListOf(run.prefill(ints(p, "promptTokens").toIntArray()).first)
            val perStep = ArrayList<String>()
            var i = 0
            while (got.size < want.size) {
                val w = schedule[i++ % schedule.size]
                val e = run.step(fromChain = true, width = w)
                perStep += "${e.size}/${w + 1}"
                got += e.toList()
            }
            assertEquals(want, got.take(want.size), "${(p["kind"] as JsonString).value}: speculative ids")
            println("[qwen35-mtp] widths 5 and 3, ${(p["kind"] as JsonString).value}: ${want.size} ids equal transformers' in ${perStep.size} verify steps, tokens per step $perStep")
        }
    }

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
