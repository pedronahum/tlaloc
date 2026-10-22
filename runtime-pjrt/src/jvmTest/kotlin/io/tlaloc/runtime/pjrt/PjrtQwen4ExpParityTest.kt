package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import kotlin.math.abs
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The tiny random Qwen4-Exp checkpoint of HfQwen4ExpParityTest on the device:
 * two sequences in one bucket, each prefilled in calls of 5 tokens and decoded
 * together, through the emitted forms of NGRAM_HASH, INDEXER_TOPK and the
 * masked PAGED_ATTENTION. transformers' greedy ids for both, every logit
 * within 1e-4.
 */
class PjrtQwen4ExpParityTest {

    private val dir = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_qwen4_exp")

    @Test
    fun twoSequencesMatchTransformersOnTheDevice() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val fixture = parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val prompts = (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }
        fun ints(p: JsonObject, k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
        val bs = 4
        val context = 36
        val pages = context / bs
        val model = config.toDecodeModelShape(numBlocks = 1 + 2 * pages, blockSize = bs, stateSlots = 2)
        val bucket = DecodeBucket(2, context)
        val piece = 5
        val prefill = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = piece), config)
        val decode = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket), config)
        val spec = HfDecoderGraph.spec(config, model, bucket)
        val vocab = config.vocabSize
        val toks = prompts.map { ints(it, "promptTokens") }
        val want = prompts.map { ints(it, "generatedTokens") }
        val tables = FloatArray(2 * pages) { (1 + it).toFloat() }
        fun slot(row: Int, pos: Int) = ((1 + row * pages + pos / bs) * bs + pos % bs).toFloat()
        var worst = 0.0
        TestBackend.session().use { s ->
            var pools: List<FloatArray> = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN || it.role == DecodeSlotRole.STATE_POOL_IN }
                .map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
            val calls = toks.maxOf { (it.size + piece - 1) / piece }
            var last = FloatArray(0)
            for (c in 0 until calls) {
                fun cell(row: Int, i: Int): Pair<Int, Int>? {
                    val pr = toks[row]
                    val from = c * piece
                    val n = minOf(piece, pr.size - from)
                    val k = i - (piece - n)
                    return if (n <= 0 || k < 0) null else pr[from + k] to from + k
                }
                val out = s.runOn(prefill, buildList {
                    add(FloatArray(2 * piece) { cell(it / piece, it % piece)?.first?.toFloat() ?: 0f })
                    add(FloatArray(2 * piece) { cell(it / piece, it % piece)?.second?.toFloat() ?: 0f })
                    add(tables)
                    add(FloatArray(2) { 1f })
                    add(FloatArray(2 * piece) { i -> cell(i / piece, i % piece)?.let { slot(i / piece, it.second) } ?: -1f })
                    add(floatArrayOf(0f, 1f))
                    addAll(pools)
                    addAll(weights)
                })
                pools = out.drop(1)
                last = if (c == 0) out[0] else FloatArray(out[0].size) { i ->
                    if ((toks[i / vocab].size + piece - 1) / piece - 1 == c) out[0][i] else last[i]
                }
            }
            fun argmax(v: FloatArray, row: Int) = (0 until vocab).maxByOrNull { v[row * vocab + it] }!!
            val got = List(2) { arrayListOf(argmax(last, it)) }
            val logits = List(2) { row -> arrayListOf(last.copyOfRange(row * vocab, (row + 1) * vocab)) }
            while (got[0].size < want[0].size) {
                val pos = List(2) { toks[it].size + got[it].size - 1 }
                val step = s.runOn(decode, buildList {
                    add(FloatArray(2) { got[it].last().toFloat() })
                    add(FloatArray(2) { pos[it].toFloat() })
                    add(tables)
                    add(FloatArray(2) { (pos[it] + 1).toFloat() })
                    add(FloatArray(2) { slot(it, pos[it]) })
                    add(floatArrayOf(0f, 1f))
                    addAll(pools)
                    addAll(weights)
                })
                pools = step.drop(1)
                for (row in 0 until 2) {
                    got[row] += argmax(step[0], row)
                    logits[row] += step[0].copyOfRange(row * vocab, (row + 1) * vocab)
                }
            }
            for (row in 0 until 2) {
                assertEquals(want[row], got[row], "greedy ids on ${TestBackend.target} for prompt ${toks[row]}")
                val wantLogits = (prompts[row]["logits"] as JsonArray).elements.map { r -> (r as JsonArray).elements.map { (it as JsonNumber).value } }
                for ((st, lg) in logits[row].withIndex()) for (v in lg.indices) worst = maxOf(worst, abs(lg[v] - wantLogits[st][v]))
            }
        }
        println("[pjrt-qwen4exp] ids == transformers, worst |logit difference| $worst")
        assertTrue(worst <= 1e-4, "worst logit difference $worst")
    }
}
