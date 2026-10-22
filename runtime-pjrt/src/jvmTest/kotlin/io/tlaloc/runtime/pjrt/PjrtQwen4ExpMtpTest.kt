package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
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
import kotlin.test.assertTrue

/**
 * The tiny Qwen4-Exp checkpoint's MTP head on the device, as
 * HfQwen4ExpMtpTest runs it in the interpreter: speculative entries with 3
 * drafts, steps verifying the head's drafts, the target's continuation, or its
 * first token and a wrong one. The target's greedy ids, and after every step
 * the drafts of `mtp.json`. A step verifying the head's drafts checks the last
 * step's chained inputs against the host's.
 */
class PjrtQwen4ExpMtpTest {

    private val dir = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_qwen4_exp")
    private fun json(name: String) = parseJson(Files.readString(dir.resolve(name))) as JsonObject
    private fun ints(o: JsonObject, k: String) = (o[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }

    private class Run(val s: PjrtSession, val config: HfDecoderConfig, val weights: List<FloatArray>) {
        val k = config.mtpDraftTokens
        val bs = 4
        val ctx = 48
        val slots = k + 2
        val model = config.toDecodeModelShape(numBlocks = 1 + ctx / bs, blockSize = bs, stateSlots = slots)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx))
        val verify: DxirFunction = HfDecoderGraph.build(spec, config)
        var pools: List<Any> = spec.inputs.drop(spec.kvPoolInputBase).take(spec.numPools).map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val chained = spec.chainRoles.size
        val table = IntArray(ctx / bs) { 1 + it }
        var readSlot = 0
        var position = 0
        var pending = -1
        var drafts = IntArray(0)
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
            val fn = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx), DecodeGraphKind.PREFILL, prefillChunk = c), config)
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

        /** One verify step of the pending token and [fed]; [checkChain]: the last step's chained inputs must be this one's. */
        fun step(fed: IntArray, checkChain: Boolean): IntArray {
            val t = k + 1
            val tokens = intArrayOf(pending) + fed
            val pos = IntArray(t) { position + it }
            val writes = (0 until slots).filter { it != readSlot }.take(t).toIntArray()
            val c = chain
            if (checkChain && c != null) {
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

    @Test
    fun speculativeDecodingOnTheDeviceDraftsTheOraclesTokens() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = DRAFTS)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val greedy = (json("greedy.json")["prompts"] as JsonArray).elements.map { it as JsonObject }
        val mtp = (json("mtp.json")["prompts"] as JsonArray).elements.map { it as JsonObject }
        TestBackend.session().use { s ->
            for ((g, m) in greedy.zip(mtp)) {
                val prompt = ints(g, "promptTokens")
                val seq = prompt + ints(g, "generatedTokens")
                val want = (m["drafts"] as JsonArray).elements.map { r -> (r as JsonArray).elements.map { (it as JsonNumber).value.toInt() } }
                val run = Run(s, config, weights)
                val (next, first) = run.prefill(prompt.toIntArray())
                assertEquals(seq[prompt.size], next, "the target's next token")
                assertEquals(want[0], first.toList(), "the prefill's drafts")
                val accepted = ArrayList<Int>()
                var step = 0
                while (run.position < seq.size - 1) {
                    val q = run.position
                    val truth = IntArray(DRAFTS) { seq.getOrElse(q + 1 + it) { -1 } }
                    val fed = when {
                        truth.any { it < 0 } || step % 3 == 0 -> run.drafts
                        step % 3 == 1 -> truth
                        else -> intArrayOf(truth[0], (truth[1] + 1) % config.vocabSize, truth[2])
                    }
                    // The chained inputs carry the head's drafts.
                    val emitted = run.step(fed, checkChain = fed === run.drafts)
                    for ((i, e) in emitted.withIndex()) if (q + 1 + i < seq.size) assertEquals(seq[q + 1 + i], e, "the target's token at ${q + 1 + i}")
                    accepted += emitted.size - 1
                    if (run.position < seq.size) assertEquals(want[run.position - prompt.size], run.drafts.toList(), "drafts after position ${run.position}")
                    step++
                }
                assertTrue(accepted.contains(DRAFTS) && accepted.contains(1), "accepted counts $accepted")
                println("[pjrt-qwen4exp-mtp] prompt of ${prompt.size} on ${TestBackend.target}: ids and drafts equal the oracle's, accepted per step $accepted")
            }
        }
    }

    private companion object {
        const val DRAFTS = 3
    }
}
