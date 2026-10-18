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
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The qwen4_exp family against transformers on a tiny random checkpoint
 * committed with its oracle (`tiny_qwen4_exp/`, written by
 * `harness/python/tiny_qwen4_exp_fixture.py`): four residual streams, three
 * Gated DeltaNet layers (the second with the n-gram memory) and one attention
 * layer whose indexer keeps 2 chunks of 4 positions, every MLP a mixture of 8
 * experts. Two prompts, the second of 22 tokens with an eos inside: the
 * indexer selects chunks from the 12th position on, and the n-gram history
 * restarts after the eos. The ids are transformers', and every logit of every
 * step is within [TOL] of its. The tolerance is tight because the attention
 * layer moves these logits little: attending to every chunk instead of the
 * indexer's selection is 9.4e-5 away, against 2e-8 with it.
 */
class HfQwen4ExpParityTest {

    private val dir: Path = Path.of(javaClass.getResource("tiny_qwen4_exp/config.json")!!.toURI()).parent
    private val fixture = parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject

    @Test
    fun theTinyCheckpointReadsAsQwen4Exp() {
        HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config
            assertEquals(HfModelFamily.Qwen4Exp, c.family)
            assertEquals(GatedResidualConfig(streams = 4, lowRank = 6), c.gatedResidual)
            assertEquals(IndexerConfig(heads = 4, headDim = 8, budget = 8, chunk = 4), c.indexer)
            assertEquals(listOf(1), c.ple!!.layers)
            assertTrue(c.layer(1).ple && !c.layer(0).ple)
            assertTrue(c.layer(3).indexer && c.layer(3).mixer == TokenMixer.ATTENTION)
            assertTrue(c.linearGateSigmoid)
            // Every tensor is read, the hash constants checked, but the vision encoder's and the MTP head's.
            assertTrue(ckpt.verifyInventory().all { it.startsWith("model.visual.") || it.startsWith("mtp.") })
        }
    }

    @Test
    fun greedyDecodingMatchesTransformersLogitForLogit() {
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs, stateSlots = 2)
        fun build(kind: DecodeGraphKind): DxirFunction =
            HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
        val prefill = build(DecodeGraphKind.PREFILL)
        val decode = build(DecodeGraphKind.DECODE)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context))
        val poolTypes = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN || it.role == DecodeSlotRole.STATE_POOL_IN }.map { it.type }
        var worst = 0.0
        for (p in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
            val prompt = ints("promptTokens")
            val want = ints("generatedTokens")
            val wantLogits = (p["logits"] as JsonArray).elements.map { row ->
                (row as JsonArray).elements.map { (it as JsonNumber).value }
            }
            var pools: List<FloatArray> = poolTypes.map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
            val pad = context - prompt.size
            val tables = FloatArray(context / bs) { (1 + it).toFloat() }
            val out = DxirInterpreter.evalFunction(prefill, buildList {
                add(FloatArray(context) { if (it < pad) 0f else prompt[it - pad].toFloat() })
                add(FloatArray(context) { if (it < pad) 0f else (it - pad).toFloat() })
                add(tables)
                add(floatArrayOf(prompt.size.toFloat()))
                add(FloatArray(context) { if (it < pad) -1f else (bs + it - pad).toFloat() })
                add(floatArrayOf(1f))
                addAll(pools)
                addAll(weights)
            })
            pools = out.drop(1)
            val logits = arrayListOf(out[0])
            fun argmax(v: FloatArray) = v.indices.maxByOrNull { v[it] }!!
            val got = arrayListOf(argmax(out[0]))
            while (got.size < want.size) {
                val pos = prompt.size + got.size - 1
                val step = DxirInterpreter.evalFunction(decode, buildList {
                    add(floatArrayOf(got.last().toFloat()))
                    add(floatArrayOf(pos.toFloat()))
                    add(tables)
                    add(floatArrayOf((pos + 1).toFloat()))
                    add(floatArrayOf((bs + pos).toFloat()))
                    add(floatArrayOf(1f))
                    addAll(pools)
                    addAll(weights)
                })
                pools = step.drop(1)
                logits += step[0]
                got += argmax(step[0])
            }
            assertEquals(want, got, "greedy ids for prompt $prompt")
            for ((s, row) in logits.withIndex()) {
                for (v in row.indices) {
                    val e = abs(row[v] - wantLogits[s][v])
                    worst = maxOf(worst, e)
                    assertTrue(e <= TOL, "step $s logit $v: ${row[v]} vs ${wantLogits[s][v]}")
                }
            }
        }
        println("tiny Qwen4-Exp parity: ids exact, worst |logit difference| $worst")
    }

    /**
     * Both prompts in one bucket of two, each prefilled in calls of 5 tokens
     * (so the indexer's chunks of 4 and the n-gram history cross calls) and
     * decoded together: the same ids and logits as the oracle's.
     */
    @Test
    fun twoSequencesPrefilledInPiecesMatchTransformers() {
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
        var pools: List<FloatArray> = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN || it.role == DecodeSlotRole.STATE_POOL_IN }
            .map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val tables = FloatArray(2 * pages) { (1 + it).toFloat() }
        fun slot(row: Int, pos: Int) = ((1 + row * pages + pos / bs) * bs + pos % bs).toFloat()
        val toks = prompts.map { ints(it, "promptTokens") }
        val calls = toks.maxOf { (it.size + piece - 1) / piece }
        var last = FloatArray(0)
        for (c in 0 until calls) {
            // Each row's piece c, right-aligned; a row with no piece left is padding throughout.
            fun cell(row: Int, i: Int): Triple<Int, Int, Int>? {
                val pr = toks[row]
                val from = c * piece
                val n = minOf(piece, pr.size - from)
                if (n <= 0) return null
                val k = i - (piece - n)
                return if (k < 0) null else Triple(pr[from + k], from + k, 0)
            }
            val out = DxirInterpreter.evalFunction(prefill, buildList {
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
            // A row's logits are its last piece's.
            last = if (c == 0) out[0] else FloatArray(out[0].size) { i ->
                val row = i / config.vocabSize
                if ((toks[row].size + piece - 1) / piece - 1 == c) out[0][i] else last[i]
            }
        }
        fun argmax(v: FloatArray, row: Int) = (0 until config.vocabSize).maxByOrNull { v[row * config.vocabSize + it] }!!
        val got = List(2) { arrayListOf(argmax(last, it)) }
        val logits = List(2) { row -> arrayListOf(last.copyOfRange(row * config.vocabSize, (row + 1) * config.vocabSize)) }
        val want = prompts.map { ints(it, "generatedTokens") }
        while (got[0].size < want[0].size) {
            val pos = List(2) { toks[it].size + got[it].size - 1 }
            val step = DxirInterpreter.evalFunction(decode, buildList {
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
                logits[row] += step[0].copyOfRange(row * config.vocabSize, (row + 1) * config.vocabSize)
            }
        }
        var worst = 0.0
        for (row in 0 until 2) {
            assertEquals(want[row], got[row], "greedy ids for prompt ${toks[row]}")
            val wantLogits = (prompts[row]["logits"] as JsonArray).elements.map { r -> (r as JsonArray).elements.map { (it as JsonNumber).value } }
            for ((st, lg) in logits[row].withIndex()) {
                for (v in lg.indices) worst = maxOf(worst, abs(lg[v] - wantLogits[st][v]))
            }
        }
        println("tiny Qwen4-Exp, two sequences prefilled in pieces of $piece: ids exact, worst |logit difference| $worst")
        assertTrue(worst <= TOL, "worst logit difference $worst")
    }

    private companion object {
        const val TOL = 1e-6
    }
}
