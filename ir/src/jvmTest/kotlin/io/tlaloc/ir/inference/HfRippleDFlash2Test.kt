package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.inference.HfRippleParityTest.Companion.HD
import io.tlaloc.ir.inference.HfRippleParityTest.Companion.HEADS
import io.tlaloc.ir.inference.HfRippleParityTest.Companion.HIDDEN
import io.tlaloc.ir.inference.HfRippleParityTest.Companion.INTER
import io.tlaloc.ir.inference.HfRippleParityTest.Companion.KV
import io.tlaloc.ir.inference.HfRippleParityTest.Companion.THETA
import io.tlaloc.ir.inference.HfRippleParityTest.Companion.VOCAB
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Files
import java.nio.file.Path
import java.util.Random
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Speculative decoding with a DFlash2 draft, in the interpreter, against [DraftReference]:
 * a dense draft in doubles from the arithmetic of DFlash2's published description and vLLM's
 * implementation, on a tiny random target and draft. A verify step's tokens are the target's
 * whatever it is given to verify, and the drafts after it are the reference's from the
 * context the step accepted: all accepted, some, and none.
 */
class HfRippleDFlash2Test {

    private val targetDir: Path = Files.createTempDirectory("tiny-ripple")
    private val draftDir: Path = Files.createTempDirectory("tiny-ripple-draft")
    private val targetWeights = HfRippleParityTest.writeTinyRipple(targetDir)
    private val draftWeights = writeDraft(draftDir).also {
        // TLALOC_KEEP_TINY_RIPPLE=<dir>: write the tiny_ripple and tiny_ripple_dflash2 fixtures there.
        System.getenv("TLALOC_KEEP_TINY_RIPPLE")?.let { keep ->
            HfRippleParityTest.writeTinyRipple(Path.of(keep, "tiny_ripple").also(Files::createDirectories))
            writeDraft(Path.of(keep, "tiny_ripple_dflash2").also(Files::createDirectories))
        }
    }

    @AfterTest
    fun remove() {
        targetDir.toFile().deleteRecursively()
        draftDir.toFile().deleteRecursively()
    }

    private fun ints(v: FloatArray) = v.map { it.toInt() }

    @Test
    fun theDraftConfigIsReadStrictly() {
        HfCheckpoint.open(targetDir, draftDir = draftDir).use { ckpt ->
            val c = ckpt.config.dflash!!
            assertEquals(listOf(1, 4), c.targetLayers)
            assertEquals(listOf(true, true, false), c.layers.map { it.rope })
            assertEquals(listOf(WINDOW, WINDOW, DFlash2Config.TRAINING_CONTEXT), c.layers.map { it.slidingWindow })
            // Every tensor of both checkpoints is read.
            assertEquals(emptyList(), ckpt.verifyInventory(ckpt.config.copy(mtpDraftTokens = K)))
        }
        val target = HfDecoderConfig.parse(HfRippleParityTest.CONFIG)
        fun refused(from: String, to: String, what: String) {
            val bad = DRAFT_CONFIG.replace(from, to)
            assertTrue(bad != DRAFT_CONFIG)
            val m = assertFailsWith<JsonException> { DFlash2Config.parse(bad, target) }.message!!
            assertTrue(what in m, m)
        }
        refused("\"block_size\"", "\"input_embedding_scale\": 2.0, \"block_size\"", "input_embedding_scale")
        refused("\"is_causal\": false", "\"is_causal\": true", "is_causal")
        refused("\"num_target_layers\": 5", "\"num_target_layers\": 6", "num_target_layers")
        refused("\"target_layer_ids\": [1, 4]", "\"target_layer_ids\": [1, 9]", "target_layer_ids")
    }

    @Test
    fun theDraftKeepsItsDtypeUnderAnNvfp4Target() {
        val c = HfCheckpoint.open(targetDir, draftDir = draftDir).use {
            it.config.copy(weightDType = BF16, weightQuant = WeightQuant.NVFP4, attentionQuant = WeightQuant.NONE, mtpDraftTokens = K)
        }
        val model = c.toDecodeModelShape(numBlocks = 16, blockSize = 4)
        val spec = HfDecoderGraph.spec(c, model, DecodeBucket(1, 36))
        HfDecoderGraph.build(spec, c)
        val names = spec.weightSlots.map { it.name }
        assertEquals(names.size, names.toSet().size)
        assertTrue("gateUp0" in names && "gateUpDraft0" in names && "gateUpDraft1" in names, names.toString())
        val draft = spec.weightSlots.filter { "Draft" in it.name }
        assertTrue(draft.isNotEmpty() && draft.all { it.type.dtype == BF16 }, draft.toString())
    }

    @Test
    fun theTinyDraftActuallyDependsOnWhatItIsGiven() {
        val r = DraftReference(draftWeights, targetWeights)
        val base = r.drafts(PROMPT, 7, K)
        assertTrue(base != r.drafts(PROMPT, 8, K) || base != r.drafts(PROMPT + 9, 7, K), "the drafts ignore their inputs")
        // The selector's walk is not simply each row's best logit.
        assertTrue((0 until 12).any { r.drafts(PROMPT + it, it + 3, K) != r.unaryDrafts(PROMPT + it, it + 3, K) }, "the selector never changes a draft")
    }

    @Test
    fun verifyStepsMatchTheReference() {
        val (config, staged) = HfCheckpoint.open(targetDir, draftDir = draftDir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = K)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val bucket = DecodeBucket(1, context)
        val chunk = 8
        val prefillSpec = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk)
        val prefill: DxirFunction = HfDecoderGraph.build(prefillSpec, config)
        val verifySpec = HfDecoderGraph.spec(config, model, bucket)
        val verify: DxirFunction = HfDecoderGraph.build(verifySpec, config)
        assertEquals(setOf("tokenIds", "positions", "blockTables", "seqLens", "slotMapping"), verifySpec.inputs.take(5).map { it.name }.toSet())
        val lead = verifySpec.leadingOutputs
        assertEquals(7, lead, "tokens, accepted, drafts and four chained inputs: no state to commit")
        val poolTypes = verifySpec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { it.type }
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val target = HfRippleParityTest.Reference(targetWeights)
        val draft = DraftReference(draftWeights, targetWeights)

        var pools: List<FloatArray> = poolTypes.map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
        val pad = chunk - PROMPT.size
        val first = DxirInterpreter.evalFunction(prefill, buildList {
            add(FloatArray(chunk) { if (it < pad) 0f else PROMPT[it - pad].toFloat() })
            add(FloatArray(chunk) { if (it < pad) 0f else (it - pad).toFloat() })
            add(tables)
            add(floatArrayOf(PROMPT.size.toFloat()))
            add(FloatArray(chunk) { if (it < pad) -1f else (bs + it - pad).toFloat() })
            addAll(pools)
            addAll(staged)
        })
        pools = first.drop(lead)
        fun best(ids: List<Int>) = target.logits(ids).last().let { l -> l.indices.maxByOrNull { l[it] }!! }
        var context0 = PROMPT
        var pending = ints(first[0])[0]
        assertEquals(best(PROMPT), pending)
        var drafts = ints(first[2])
        assertEquals(draft.drafts(context0, pending, K), drafts, "the prefill's drafts")

        // How many drafts each step is given correct, the rest replaced by wrong ones. -1: the graph's own.
        val accepts = listOf(-1, K, 1, 0, 2)
        val seen = HashSet<Int>()
        for ((s, wantAccepted) in accepts.withIndex()) {
            val given = ArrayList(drafts)
            if (wantAccepted >= 0) {
                var seq = context0 + pending
                for (j in 0 until K) {
                    val truth = best(seq)
                    given[j] = if (j < wantAccepted) truth else (truth + 1) % VOCAB
                    seq = seq + given[j]
                }
            }
            val n = context0.size
            val tokens = listOf(pending) + given
            val step = DxirInterpreter.evalFunction(verify, buildList {
                add(FloatArray(K + 1) { tokens[it].toFloat() })
                add(FloatArray(K + 1) { (n + it).toFloat() })
                add(tables)
                add(floatArrayOf((n + K + 1).toFloat()))
                add(FloatArray(K + 1) { (bs + n + it).toFloat() })
                addAll(pools)
                addAll(staged)
            })
            pools = step.drop(lead)
            val g = ints(step[0])
            val accepted = ints(step[1])[0]
            seen += accepted
            if (wantAccepted >= 0) assertEquals(wantAccepted, accepted, "step $s")
            // Every token the step returns is the target's after what it verified.
            for (j in 0..K) assertEquals(best(context0 + tokens.take(j + 1)), g[j], "step $s, token $j")
            context0 = context0 + tokens.take(accepted + 1)
            pending = g[accepted]
            drafts = ints(step[2])
            assertEquals(draft.drafts(context0, pending, K), drafts, "step $s's drafts, $accepted accepted")
            // The chained inputs are the next step's, the drafts included.
            val next = context0.size
            assertEquals(listOf(pending) + drafts, ints(step[3]), "step $s, chained tokens")
            assertEquals(List(K + 1) { next + it }, ints(step[4]), "step $s, chained positions")
            assertEquals(listOf(next + K + 1), ints(step[5]), "step $s, chained length")
            assertEquals(List(K + 1) { bs + next + it }, ints(step[6]), "step $s, chained slots")
        }
        assertTrue(seen.containsAll(listOf(0, 1, 2, K)), "accepted counts seen: $seen")
    }

    /**
     * The target's sliding layers and the draft's sliding layers on rings
     * ([DecodeModelShape.windowedKv], [DecodeModelShape.draftWindowedKv]) that wrap several times
     * over a sequence of 64 positions: a prompt in calls of at most what every ring takes
     * ([WindowedKvPool.maxTokensPerCall]), then verify steps, with the pages in a shuffled order
     * as a runtime hands them out. The tokens and drafts are the reference's.
     */
    @Test
    fun ringsThatWrapGiveTheReferencesIds() {
        val (config, staged) = HfCheckpoint.open(targetDir, draftDir = draftDir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = K)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 64
        val chunk = 8
        val numBlocks = 1 + context / bs
        val window = config.windowedKvPool(blockSize = bs, maxContext = context, fullNumBlocks = numBlocks, prefillChunk = chunk)!!
        val draftRings = config.draftWindowedKvPools(blockSize = bs, maxContext = context, fullNumBlocks = numBlocks, prefillChunk = chunk)
        // The draft's sliding layers (window 12) get a ring; its full layer (16,384) would not wrap.
        assertEquals(listOf(12), draftRings.map { it.window })
        assertEquals(listOf(0, 1), draftRings.single().layers)
        assertEquals(K + 1, draftRings.single().trailingRows)
        val model = config.toDecodeModelShape(numBlocks = numBlocks, blockSize = bs, windowedKv = window, draftWindowedKv = draftRings)
        val bucket = DecodeBucket(1, context)
        val prefillSpec = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk)
        val prefill = HfDecoderGraph.build(prefillSpec, config)
        val verifySpec = HfDecoderGraph.spec(config, model, bucket)
        val verify = HfDecoderGraph.build(verifySpec, config)
        val ringPools = verifySpec.inputs.filter { it.role == DecodeSlotRole.WINDOW_KV_POOL_IN }
        assertEquals(setOf(0, 1), ringPools.map { it.windowClass }.toSet())
        assertEquals(4, ringPools.count { it.name.startsWith("draft") }, "two draft layers' keys and values on the draft's ring")
        val lead = verifySpec.leadingOutputs
        val maxBlocks = context / bs
        val full = (1..maxBlocks).shuffled(Random(1))
        val rings = model.windowClasses.map { (1..it.ringPages).shuffled(Random(2)) }
        assertTrue(model.windowClasses.all { it.ringPages < maxBlocks }, "every ring wraps: ${model.windowClasses}")
        fun inputs(spec: DecodeGraphSpec, tokens: List<Int>, from: Int, pools: List<FloatArray>): List<FloatArray> {
            val rows = spec.tokensPerSeq
            val pad = rows - tokens.size
            fun pos(i: Int) = from + i - pad
            return spec.inputs.mapNotNull { slot ->
                val ring = if (slot.role == DecodeSlotRole.WINDOW_BLOCK_TABLES || slot.role == DecodeSlotRole.WINDOW_SLOT_MAPPING) rings[slot.windowClass] else null
                when (slot.role) {
                    DecodeSlotRole.TOKEN_IDS -> FloatArray(rows) { if (it < pad) 0f else tokens[it - pad].toFloat() }
                    DecodeSlotRole.POSITIONS -> FloatArray(rows) { if (it < pad) 0f else pos(it).toFloat() }
                    DecodeSlotRole.BLOCK_TABLES -> FloatArray(maxBlocks) { full[it].toFloat() }
                    DecodeSlotRole.SEQ_LENS -> floatArrayOf((from + tokens.size).toFloat())
                    DecodeSlotRole.SLOT_MAPPING -> FloatArray(rows) { if (it < pad) -1f else (full[pos(it) / bs] * bs + pos(it) % bs).toFloat() }
                    DecodeSlotRole.WINDOW_BLOCK_TABLES -> FloatArray(maxBlocks) { ring!![it % ring.size].toFloat() }
                    DecodeSlotRole.WINDOW_SLOT_MAPPING -> FloatArray(rows) {
                        if (it < pad) -1f else (ring!![(pos(it) / bs) % ring.size] * bs + pos(it) % bs).toFloat()
                    }
                    else -> null
                }
            } + pools + staged
        }
        val target = HfRippleParityTest.Reference(targetWeights)
        val draft = DraftReference(draftWeights, targetWeights)
        fun best(ids: List<Int>) = target.logits(ids).last().let { l -> l.indices.maxByOrNull { l[it] }!! }

        val prompt = List(20) { (3 + 5 * it) % 56 }
        var pools: List<FloatArray> = verifySpec.inputs
            .filter { it.role == DecodeSlotRole.KV_POOL_IN || it.role == DecodeSlotRole.WINDOW_KV_POOL_IN }
            .map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        var from = 0
        var last: List<FloatArray> = emptyList()
        var calls = 0
        while (from < prompt.size) {
            val n = minOf(chunk, prompt.size - from, model.windowClasses.minOf { it.maxTokensPerCall(from, bs) })
            last = DxirInterpreter.evalFunction(prefill, inputs(prefillSpec, prompt.subList(from, from + n), from, pools))
            pools = last.drop(lead)
            from += n
            calls++
        }
        assertTrue(calls >= 3, "the prompt runs in $calls calls")
        var seq = prompt
        var pending = ints(last[0])[0]
        assertEquals(best(prompt), pending)
        var drafts = ints(last[2])
        assertEquals(draft.drafts(seq, pending, K), drafts, "the prefill's drafts")
        val accepts = listOf(-1, K, 1, 0, 2, K, -1, K)
        var s = 0
        while (seq.size + 1 + K + 2 * K + 1 <= context) {
            val wantAccepted = accepts[s % accepts.size]
            val given = ArrayList(drafts)
            if (wantAccepted >= 0) {
                var ctx = seq + pending
                for (j in 0 until K) {
                    val truth = best(ctx)
                    given[j] = if (j < wantAccepted) truth else (truth + 1) % 56
                    ctx = ctx + given[j]
                }
            }
            val tokens = listOf(pending) + given
            val step = DxirInterpreter.evalFunction(verify, inputs(verifySpec, tokens, seq.size, pools))
            pools = step.drop(lead)
            val g = ints(step[0])
            val accepted = ints(step[1])[0]
            if (wantAccepted >= 0) assertEquals(wantAccepted, accepted, "step $s")
            for (j in 0..accepted) assertEquals(best(seq + tokens.take(j + 1)), g[j], "step $s at ${seq.size}, token $j")
            seq = seq + tokens.take(accepted + 1)
            pending = g[accepted]
            drafts = ints(step[2])
            assertEquals(draft.drafts(seq, pending, K), drafts, "step $s's drafts at ${seq.size}, $accepted accepted")
            s++
        }
        val draftRing = draftRings.single().ringPages * bs
        assertTrue(seq.size > 2 * draftRing, "the draft's ring of $draftRing positions wrapped: ${seq.size} positions")
    }

    /**
     * Entries that draw their tokens (`-Psampling=true`) with the DFlash2 draft: at temperature 0
     * a prompt continues with the reference's greedy ids; at 0.8 the draw is a function of the
     * seed (the same seed, the same tokens; another seed, other tokens), and every draft the
     * step accepted is the token the target drew at its place.
     */
    @Test
    fun sampledVerifyStepsAreGreedyAtTemperatureZeroAndSeededAbove() {
        val (config, staged) = HfCheckpoint.open(targetDir, draftDir = draftDir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = K)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs, sampling = true)
        val bucket = DecodeBucket(1, context)
        val chunk = 8
        val prefillSpec = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk)
        val prefill = HfDecoderGraph.build(prefillSpec, config)
        val verifySpec = HfDecoderGraph.spec(config, model, bucket)
        val verify = HfDecoderGraph.build(verifySpec, config)
        assertTrue(verify.body.any { it is io.tlaloc.ir.DxirOp && it.op == io.tlaloc.ir.OpKind.SAMPLE_TOKENS }, "the verify entry draws its tokens")
        val lead = verifySpec.leadingOutputs
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        fun decode(temperature: Float, seed: Int, n: Int): List<Int> {
            val sampling = floatArrayOf(Math.round(temperature * 65536f).toFloat(), 16777216f, 0f, seed.toFloat())
            fun inputs(spec: DecodeGraphSpec, tokens: List<Int>, from: Int, rows: Int, pad: Int, pools: List<FloatArray>) = spec.inputs.map { slot ->
                when (slot.role) {
                    DecodeSlotRole.TOKEN_IDS -> FloatArray(rows) { if (it < pad) 0f else tokens[it - pad].toFloat() }
                    DecodeSlotRole.POSITIONS -> FloatArray(rows) { if (it < pad) 0f else (from + it - pad).toFloat() }
                    DecodeSlotRole.BLOCK_TABLES -> tables
                    DecodeSlotRole.SEQ_LENS -> floatArrayOf((from + rows - pad).toFloat())
                    DecodeSlotRole.SLOT_MAPPING -> FloatArray(rows) { if (it < pad) -1f else (bs + from + it - pad).toFloat() }
                    DecodeSlotRole.SAMPLING -> sampling
                    else -> null
                }
            }.filterNotNull() + pools + staged
            var pools: List<FloatArray> = verifySpec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
            val first = DxirInterpreter.evalFunction(prefill, inputs(prefillSpec, PROMPT, 0, chunk, chunk - PROMPT.size, pools))
            pools = first.drop(lead)
            val out = arrayListOf(ints(first[0])[0])
            var drafts = ints(first[2])
            var at = PROMPT.size
            while (out.size < n && at + 2 * (K + 1) < context) {
                val tokens = listOf(out.last()) + drafts
                val step = DxirInterpreter.evalFunction(verify, inputs(verifySpec, tokens, at, K + 1, 0, pools))
                pools = step.drop(lead)
                val g = ints(step[0])
                val accepted = ints(step[1])[0]
                // An accepted draft is what the target drew at its place.
                for (j in 0 until accepted) assertEquals(drafts[j], g[j], "an accepted draft is the drawn token")
                out += g.take(accepted + 1)
                at += accepted + 1
                drafts = ints(step[2])
            }
            return out.take(n)
        }
        val target = HfRippleParityTest.Reference(targetWeights)
        val greedy = ArrayList(PROMPT).also { seq -> repeat(10) { val l = target.logits(seq).last(); seq += l.indices.maxByOrNull { l[it] }!! } }.drop(PROMPT.size)
        assertEquals(greedy, decode(0f, 1, greedy.size), "temperature 0 is greedy")
        val seven = decode(0.8f, 7, 10)
        assertEquals(seven, decode(0.8f, 7, 10), "the same seed, the same draw")
        assertTrue(seven != decode(0.8f, 8, 10) || seven != decode(0.8f, 9, 10), "the seed decides the draw")
    }

    @Test
    fun aVerifyEntryOfFewerDraftsMatchesTheReference() {
        val (config, staged) = HfCheckpoint.open(targetDir, draftDir = draftDir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, mtpDraftTokens = K)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val bucket = DecodeBucket(1, context)
        val chunk = 8
        val few = K - 1
        val prefill = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk), config)
        val spec = HfDecoderGraph.spec(config, model, bucket, verifyDrafts = few)
        val verify = HfDecoderGraph.build(spec, config)
        val lead = spec.leadingOutputs
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val draft = DraftReference(draftWeights, targetWeights)
        var pools = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val pad = chunk - PROMPT.size
        val first = DxirInterpreter.evalFunction(prefill, buildList {
            add(FloatArray(chunk) { if (it < pad) 0f else PROMPT[it - pad].toFloat() })
            add(FloatArray(chunk) { if (it < pad) 0f else (it - pad).toFloat() })
            add(tables)
            add(floatArrayOf(PROMPT.size.toFloat()))
            add(FloatArray(chunk) { if (it < pad) -1f else (bs + it - pad).toFloat() })
            addAll(pools)
            addAll(staged)
        })
        pools = first.drop(lead)
        var seq = PROMPT
        var pending = ints(first[0])[0]
        var drafts = ints(first[2]).take(few)
        repeat(4) { s ->
            val tokens = listOf(pending) + drafts
            val n = seq.size
            val step = DxirInterpreter.evalFunction(verify, buildList {
                add(FloatArray(few + 1) { tokens[it].toFloat() })
                add(FloatArray(few + 1) { (n + it).toFloat() })
                add(tables)
                add(floatArrayOf((n + few + 1).toFloat()))
                add(FloatArray(few + 1) { (bs + n + it).toFloat() })
                addAll(pools)
                addAll(staged)
            })
            pools = step.drop(lead)
            val accepted = ints(step[1])[0]
            seq = seq + tokens.take(accepted + 1)
            pending = ints(step[0])[accepted]
            drafts = ints(step[2])
            assertEquals(draft.drafts(seq, pending, few), drafts, "step $s, $accepted accepted")
        }
    }

    /** The draft from the target's reference: context K/V from the taps, then the query block. */
    private class DraftReference(private val w: Map<String, FloatArray>, private val targetWeights: Map<String, FloatArray>) {
        private val target = HfRippleParityTest.Reference(targetWeights)

        private fun rms(x: DoubleArray, gain: FloatArray): DoubleArray {
            val r = 1.0 / sqrt(x.sumOf { it * it } / x.size + 1e-6)
            return DoubleArray(x.size) { x[it] * r * gain[it] }
        }

        private fun proj(x: DoubleArray, name: String, out: Int, table: Map<String, FloatArray> = w): DoubleArray {
            val m = table.getValue(name)
            return DoubleArray(out) { o -> var s = 0.0; for (i in x.indices) s += x[i] * m[o * x.size + i]; s }
        }

        private fun rope(x: DoubleArray, pos: Int): DoubleArray {
            val half = x.size / 2
            return DoubleArray(x.size) { i ->
                val a = pos / THETA.pow(2.0 * (i % half) / x.size)
                val rot = if (i < half) -x[i + half] else x[i - half]
                x[i] * cos(a) + rot * sin(a)
            }
        }

        private fun heads(v: DoubleArray, count: Int) = List(count) { v.copyOfRange(it * HD, (it + 1) * HD) }

        /** One side of a conv over the query rows. */
        private fun convolve(x: List<DoubleArray>, base: FloatArray, delta: List<DoubleArray>, side: Int): List<DoubleArray> =
            List(x.size) { row ->
                DoubleArray(HIDDEN) { c ->
                    var s = 0.0
                    val group = c / GROUP
                    for (tap in 0 until TAPS) {
                        if (row < tap) continue
                        val coefficient = base[(side * TAPS + tap) * HIDDEN + c] + delta[row][(side * TAPS + tap) * (HIDDEN / GROUP) + group]
                        s += coefficient * x[row - tap][c]
                    }
                    s
                }
            }

        private class Candidates(val ids: List<IntArray>, val unary: List<DoubleArray>, val hidden: List<DoubleArray>)

        private fun candidates(ctx: List<Int>, anchor: Int, k: Int): Candidates {
            target.logits(ctx)
            val n = ctx.size
            val fc = List(n) { t ->
                val cat = DoubleArray(TARGETS.size * HIDDEN) { i -> target.layerOutputs[TARGETS[i / HIDDEN]][t][i % HIDDEN] }
                rms(proj(cat, "fc.weight", HIDDEN), w.getValue("hidden_norm.weight"))
            }
            val table = targetWeights.getValue("model.embed_tokens.weight")
            val tokens = listOf(anchor) + List(k) { MASK }
            var h = List(k + 1) { r -> DoubleArray(HIDDEN) { table[tokens[r] * HIDDEN + it].toDouble() } }
            val positions = List(k + 1) { n + it }
            for (l in 0 until DRAFT_LAYERS) {
                val p = "layers.$l."
                val nope = l == DRAFT_LAYERS - 1
                val window = if (nope) DFlash2Config.TRAINING_CONTEXT else WINDOW
                val x = h.map { rms(it, w.getValue(p + "input_layernorm.weight")) }
                val dA = x.map { proj(it, p + "attention_conv.kernel_projection.weight", 2 * TAPS * (HIDDEN / GROUP)) }
                val baseA = w.getValue(p + "attention_conv.base_kernel")
                val xc = convolve(x, baseA, dA, 0)
                fun keyOf(v: DoubleArray, pos: Int) = heads(proj(v, p + "self_attn.k_proj.weight", KV * HD), KV)
                    .map { rms(it, w.getValue(p + "self_attn.k_norm.weight")) }.map { if (nope) it else rope(it, pos) }
                fun valueOf(v: DoubleArray) = heads(proj(v, p + "self_attn.v_proj.weight", KV * HD), KV)
                val keys = List(n) { keyOf(fc[it], it) } + List(k + 1) { keyOf(xc[it], positions[it]) }
                val values = List(n) { valueOf(fc[it]) } + List(k + 1) { valueOf(xc[it]) }
                val end = n + k + 1
                val from = maxOf(0, end - window)
                val att = List(k + 1) { r ->
                    val q = heads(proj(xc[r], p + "self_attn.q_proj.weight", HEADS * HD), HEADS)
                        .map { rms(it, w.getValue(p + "self_attn.q_norm.weight")) }.map { if (nope) it else rope(it, positions[r]) }
                    val flat = DoubleArray(HEADS * HD)
                    for (head in 0 until HEADS) {
                        val kh = head / (HEADS / KV)
                        val scores = DoubleArray(end - from) { s ->
                            var dot = 0.0
                            for (i in 0 until HD) dot += q[head][i] * keys[from + s][kh][i]
                            dot / sqrt(HD.toDouble())
                        }
                        val top = scores.max()
                        val sum = scores.sumOf { exp(it - top) }
                        for (s in scores.indices) {
                            val a = exp(scores[s] - top) / sum
                            for (i in 0 until HD) flat[head * HD + i] += a * values[from + s][kh][i]
                        }
                    }
                    proj(flat, p + "self_attn.o_proj.weight", HIDDEN)
                }
                val hA = convolve(att, baseA, dA, 1).mapIndexed { r, y -> DoubleArray(HIDDEN) { h[r][it] + y[it] } }
                val y = hA.map { rms(it, w.getValue(p + "post_attention_layernorm.weight")) }
                val dM = y.map { proj(it, p + "mlp_conv.kernel_projection.weight", 2 * TAPS * (HIDDEN / GROUP)) }
                val baseM = w.getValue(p + "mlp_conv.base_kernel")
                val out = convolve(y, baseM, dM, 0).map { m ->
                    val g = proj(m, p + "mlp.gate_proj.weight", INTER)
                    val u = proj(m, p + "mlp.up_proj.weight", INTER)
                    proj(DoubleArray(INTER) { g[it] / (1.0 + exp(-g[it])) * u[it] }, p + "mlp.down_proj.weight", HIDDEN)
                }
                h = convolve(out, baseM, dM, 1).mapIndexed { r, m -> DoubleArray(HIDDEN) { hA[r][it] + m[it] } }
            }
            val rows = h.drop(1).map { rms(it, w.getValue("norm.weight")) }
            val logits = rows.map { proj(it, "lm_head.weight", VOCAB, targetWeights) }
            val ids = logits.map { l -> l.indices.sortedByDescending { l[it] }.take(TOP).toIntArray() }
            return Candidates(ids, logits.mapIndexed { i, l -> DoubleArray(TOP) { l[ids[i][it]] } }, rows)
        }

        fun unaryDrafts(ctx: List<Int>, anchor: Int, k: Int): List<Int> = candidates(ctx, anchor, k).ids.map { it[0] }

        fun drafts(ctx: List<Int>, anchor: Int, k: Int): List<Int> {
            val c = candidates(ctx, anchor, k)
            val pred = w.getValue("candidate_selector.predecessor_codebook")
            val succ = w.getValue("candidate_selector.successor_codebook")
            val out = ArrayList<Int>()
            var previous = anchor
            for (j in 0 until k) {
                val view = proj(c.hidden[j], "candidate_selector.hidden_projection.weight", RANK)
                val scores = DoubleArray(TOP) { i ->
                    var s = c.unary[j][i]
                    for (r in 0 until RANK) s += pred[previous * RANK + r] * view[r] * succ[c.ids[j][i] * RANK + r]
                    s
                }
                previous = c.ids[j][scores.indices.maxByOrNull { scores[it] }!!]
                out += previous
            }
            return out
        }
    }

    private companion object {
        const val K = 3
        const val WINDOW = 12
        const val TAPS = 2
        const val GROUP = 4
        const val TOP = 4
        const val RANK = 5
        const val MASK = 62
        const val DRAFT_LAYERS = 3
        val TARGETS = listOf(1, 4)
        val PROMPT = listOf(3, 5, 9, 12, 33, 7)
        val DRAFT_CONFIG = """
            {"architectures": ["DFlash2DraftModel"], "attention_bias": false, "is_causal": false, "model_type": "qwen3",
             "dflash_config": {"block_size": 4, "conv_group_size": $GROUP, "conv_kernel_size": $TAPS, "mask_token_id": $MASK,
                               "selector_rank": $RANK, "selector_top_k": $TOP, "target_layer_ids": [1, 4]},
             "draft_vocab_size": $VOCAB, "head_dim": $HD, "hidden_act": "silu", "hidden_size": $HIDDEN,
             "intermediate_size": $INTER, "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
             "no_rope_layers": [2], "num_attention_heads": $HEADS, "num_hidden_layers": $DRAFT_LAYERS,
             "num_key_value_heads": $KV, "num_target_layers": 5, "rope_theta": $THETA, "sliding_window": $WINDOW,
             "tie_word_embeddings": false, "vocab_size": $VOCAB}
        """.trimIndent()

        fun writeDraft(dir: Path): Map<String, FloatArray> {
            val rng = Random(20261106)
            val groups = HIDDEN / GROUP
            fun linear(o: Int, i: Int) = FloatArray(o * i) { (0.25 * rng.nextGaussian()).toFloat() }
            fun gain(n: Int) = FloatArray(n) { (1.0 + 0.3 * rng.nextGaussian()).toFloat() }
            val w = LinkedHashMap<String, Pair<IntArray, FloatArray>>()
            fun put(name: String, data: FloatArray, vararg dims: Int) { w[name] = dims to data }
            put("fc.weight", linear(HIDDEN, TARGETS.size * HIDDEN), HIDDEN, TARGETS.size * HIDDEN)
            put("hidden_norm.weight", gain(HIDDEN), HIDDEN)
            put("norm.weight", gain(HIDDEN), HIDDEN)
            put("candidate_selector.hidden_projection.weight", linear(RANK, HIDDEN), RANK, HIDDEN)
            // Large codes, so that the selector's edges outweigh the differences of the logits.
            put("candidate_selector.predecessor_codebook", FloatArray(VOCAB * RANK) { rng.nextGaussian().toFloat() }, VOCAB, RANK)
            put("candidate_selector.successor_codebook", FloatArray(VOCAB * RANK) { rng.nextGaussian().toFloat() }, VOCAB, RANK)
            for (l in 0 until DRAFT_LAYERS) {
                val p = "layers.$l."
                put(p + "self_attn.q_proj.weight", linear(HEADS * HD, HIDDEN), HEADS * HD, HIDDEN)
                put(p + "self_attn.k_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
                put(p + "self_attn.v_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
                put(p + "self_attn.o_proj.weight", linear(HIDDEN, HEADS * HD), HIDDEN, HEADS * HD)
                put(p + "self_attn.q_norm.weight", gain(HD), HD)
                put(p + "self_attn.k_norm.weight", gain(HD), HD)
                put(p + "mlp.gate_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
                put(p + "mlp.up_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
                put(p + "mlp.down_proj.weight", linear(HIDDEN, INTER), HIDDEN, INTER)
                put(p + "input_layernorm.weight", gain(HIDDEN), HIDDEN)
                put(p + "post_attention_layernorm.weight", gain(HIDDEN), HIDDEN)
                for (c in listOf("attention_conv", "mlp_conv")) {
                    put("$p$c.base_kernel", linear(2 * TAPS, HIDDEN), 2, TAPS, HIDDEN)
                    put("$p$c.kernel_projection.weight", linear(2 * TAPS * groups, HIDDEN), 2 * TAPS * groups, HIDDEN)
                }
            }
            SafetensorsFileWriter.write(
                dir.resolve("model.safetensors"),
                w.map { (name, t) -> SafetensorsTensor(name, F32, t.first, HostF32Storage(t.second)) },
            )
            Files.writeString(dir.resolve("config.json"), DRAFT_CONFIG)
            return w.mapValues { it.value.second }
        }
    }
}
