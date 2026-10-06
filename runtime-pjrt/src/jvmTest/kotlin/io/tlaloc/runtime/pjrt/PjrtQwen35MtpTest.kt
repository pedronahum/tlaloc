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
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Qwen3.5-0.8B with its MTP head on the device, speculative entries with
 * [DRAFTS] drafts, f32 weights: the speculative prefill gives the reference
 * drafts of `qwen3_5_0_8b_mtp.json`, and greedy speculative decoding gives
 * transformers' greedy ids (`qwen3_5_0_8b_greedy.json`), printing the tokens
 * each verify step emitted.
 */
class PjrtQwen35MtpTest {

    private val resources = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference")

    private fun json(name: String) = parseJson(Files.readString(resources.resolve(name))) as JsonObject

    private fun ints(o: JsonObject, k: String) = (o[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }

    private class Run(val s: PjrtSession, val config: HfDecoderConfig, val weights: List<FloatArray>) {
        val k = config.mtpDraftTokens
        val bs = 16
        val numBlocks = 9
        val ctx = (numBlocks - 1) * bs
        val slots = k + 2
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = bs, stateSlots = slots)
        val verify: DxirFunction = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)), config)
        val prefills = HashMap<Int, DxirFunction>()
        var pools: List<Any> = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)).let { sp ->
            sp.inputs.drop(sp.kvPoolInputBase).take(sp.numPools).map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        }
        val table = IntArray(ctx / bs) { 1 + it }
        var readSlot = 0
        var position = 0
        var pending = -1
        var drafts = IntArray(0)
        private val chained = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx)).chainRoles.size

        /** The last verify step's chained inputs: checked against the next step's (DecodeGraphSpec.chainRoles). */
        var chain: List<IntArray>? = null

        private fun call(fn: DxirFunction, tokens: IntArray, pos: IntArray, slotMap: IntArray, lens: Int, writes: IntArray): List<Any> {
            val out = s.runOnHost(fn, buildList {
                add(tokens); add(pos); add(table); add(intArrayOf(lens)); add(slotMap)
                add(intArrayOf(readSlot)); add(writes)
                addAll(pools); addAll(weights)
            })
            pools = out.drop(3 + chained)
            chain = if (fn === verify) out.subList(3, 3 + chained).map { it as IntArray } else null
            return out
        }

        private fun slotOf(p: Int) = table[p / bs] * bs + p % bs

        fun prefill(prompt: IntArray): Pair<Int, IntArray> {
            val c = ((prompt.size + bs - 1) / bs) * bs
            val fn = prefills.getOrPut(c) {
                HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx), DecodeGraphKind.PREFILL, prefillChunk = c), config)
            }
            val pad = c - prompt.size
            val out = call(
                fn, IntArray(c) { if (it < pad) 0 else prompt[it - pad] }, IntArray(c) { if (it < pad) 0 else it - pad },
                IntArray(c) { if (it < pad) -1 else slotOf(it - pad) }, prompt.size, IntArray(c) { -1 },
            )
            position = prompt.size
            pending = (out[0] as IntArray)[0]
            drafts = (out[2] as IntArray).copyOf()
            return pending to drafts
        }

        fun step(): IntArray {
            val t = k + 1
            val tokens = intArrayOf(pending) + drafts
            val pos = IntArray(t) { position + it }
            val writes = (0 until slots).filter { it != readSlot }.take(t).toIntArray()
            chain?.let { c ->
                val built = listOf(tokens, pos, intArrayOf(position + t), IntArray(t) { slotOf(pos[it]) }, intArrayOf(readSlot))
                for ((i, what) in listOf("tokens", "positions", "lengths", "slot mapping", "state slots").withIndex()) {
                    assertEquals(built[i].toList(), c[i].toList(), "the chained $what")
                }
                assertEquals(writes.toSet(), c[5].toSet(), "the chained state write slots")
            }
            val out = call(verify, tokens, pos, IntArray(t) { slotOf(pos[it]) }, position + t, writes)
            val a = (out[1] as IntArray)[0]
            val emitted = (out[0] as IntArray).copyOf(a + 1)
            readSlot = writes[a]
            position += a + 1
            pending = emitted.last()
            drafts = (out[2] as IntArray).copyOf()
            return emitted
        }
    }

    private fun staged(rev: String, drafts: Int): Pair<HfDecoderConfig, List<FloatArray>>? {
        val dir = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots", rev)
        if (!Files.isRegularFile(dir.resolve("config.json"))) return null
        return HfCheckpoint.open(dir).use { ckpt ->
            val config = ckpt.config.copy(weightDType = F32, mtpDraftTokens = drafts)
            config to HfStagedWeights.stage(ckpt, config)
        }
    }

    @Test
    fun speculativeDecodingOnTheDeviceMatchesTheReferenceAndTheGreedyIds() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val mtp = json("qwen3_5_0_8b_mtp.json")
        val (config, weights) = staged((mtp["revision"] as JsonString).value, DRAFTS) ?: run {
            assumeTrue(false, "no Qwen/Qwen3.5-0.8B checkpoint")
            return
        }
        TestBackend.session().use { s ->
            for (p in (mtp["prompts"] as JsonArray).elements.map { it as JsonObject }) {
                val (next, drafts) = Run(s, config, weights).prefill(ints(p, "promptTokens").toIntArray())
                assertEquals((p["targetNext"] as JsonNumber).value.toInt(), next, "the target's next token")
                assertEquals(ints(p, "drafts"), drafts.toList(), "the MTP head's drafts")
            }
            for (p in (json("qwen3_5_0_8b_greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }) {
                val want = ints(p, "generatedTokens")
                val run = Run(s, config, weights)
                val got = arrayListOf(run.prefill(ints(p, "promptTokens").toIntArray()).first)
                val perStep = ArrayList<Int>()
                while (got.size < want.size) {
                    run.step().also { perStep += it.size; got += it.toList() }
                }
                assertEquals(want, got.take(want.size), "${(p["kind"] as JsonString).value}: speculative ids")
                println("[pjrt-mtp] ${(p["kind"] as JsonString).value}: ${want.size} ids equal transformers' in ${perStep.size} verify steps, tokens per step $perStep")
            }
        }
    }

    private companion object {
        const val DRAFTS = 3
    }
}
