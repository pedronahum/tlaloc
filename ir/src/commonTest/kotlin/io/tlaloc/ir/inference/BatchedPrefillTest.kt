package io.tlaloc.ir.inference

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Prompts prefilled together: a prefill entry of batch > 1 takes the prompts
 * of several sequences in one call, each right-aligned in its own row with its
 * own positions, block table and slots.
 *
 * The claim, in the reference interpreter: every prompt prefilled in a batch
 * gets the last-token logits it gets prefilled alone in a batch-1 entry, BIT
 * FOR BIT, whatever the other rows hold and however many rows are padding;
 * and every sequence then decodes on to the same logits, bit for bit, as
 * after its solo prefill. That holds for full-history pools and for a
 * [WindowedKvPool], where a prompt longer than the ring is prefilled in
 * several rounds, each round one batched call.
 *
 * The host side here does what the Triton backend does: pages from free
 * lists (page 0 is the padding page), prompts started together run in rounds
 * of batched calls, each prompt's call at most
 * [WindowedKvPool.maxTokensPerCall] tokens, through the smallest prefill
 * batch of the ladder that holds the round.
 *
 * Negative controls: padding tokens that write their KV (over position 0 of
 * the sequence in the row before, instead of nowhere), and a row whose tokens
 * are left-aligned; both must change a row's logits.
 *
 * Chunked prefill: entries that take at most `prefillChunk` tokens per
 * sequence (fewer than the context) prefill a longer prompt in several
 * calls and give the same logits, bit for bit, as entries that take the
 * whole context; with a windowed pool the ring is sized so that a whole
 * chunk fits past the window. Negative control: the same chunks through the
 * default ring, one page too short for them, change the logits.
 */
class BatchedPrefillTest {

    private val bs = 4
    private val context = 48

    /** A sliding layer (window 8), a full one, and a sliding one (window 6). */
    private val config = HfDecoderConfig(
        architecture = "LlamaForCausalLM", modelType = "llama",
        hiddenSize = 16, intermediateSize = 24, numLayers = 3,
        numHeads = 4, numKvHeads = 2, headDim = 8, vocabSize = 31,
        rmsNormEps = 1e-6, ropeTheta = 10000.0, maxPositionEmbeddings = 64,
        tieWordEmbeddings = false, attentionBias = false, torchDtype = "float32",
        ropeScalingType = null, family = HfModelFamily.Llama,
        layers = listOf(
            DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 8),
            DecoderLayerSpec(attention = AttentionKind.FULL),
            DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 6),
        ),
    )

    /** Room for four sequences of 48 positions. */
    private val fullBlocks = 1 + 4 * (context / bs)

    private val weights: List<FloatArray> = run {
        val rng = Random(20260926)
        HfDecoderGraph.weightSlots(config).map { slot ->
            val n = slot.type.dims.fold(1) { a, b -> a * b }
            if (slot.type.dims.size == 1) {
                FloatArray(n) { 1f + 0.5f * (rng.nextFloat() - 0.5f) }
            } else {
                FloatArray(n) { 0.9f * (rng.nextFloat() - 0.5f) }
            }
        }
    }

    private fun tokens(seed: Int, n: Int) = Random(seed).let { r -> IntArray(n) { r.nextInt(config.vocabSize) } }

    /** Four prompts: one longer than a ring of 3 pages holds, one of a single page. */
    private val prompts = mapOf(
        "A" to tokens(1, 11), "B" to tokens(2, 5), "C" to tokens(3, 17), "D" to tokens(4, 3),
    )

    private val batchLadder = listOf(1, 2, 4)

    /** What a host gets back: the logits of each sequence's calls, in order. */
    private data class Row(val seq: String, val position: Int, val logits: FloatArray)

    /**
     * The backend's side, in Kotlin. [windowed] gives the sliding layers a
     * windowed pool of the default ring. With [padWrites] the first padding
     * token of the second row writes its KV over position 0 of the first
     * row's sequence instead of nowhere (slot -1), and [leftAligned] names a
     * sequence whose tokens are placed at the start of its row; the controls
     * set them.
     */
    private inner class Host(
        val windowed: Boolean,
        val padWrites: Boolean = false,
        val leftAligned: String? = null,
        val chunk: Int? = null,
        ringPages: Int? = null,
        /** Split calls by the ring (the backend's rule); the ring control turns it off. */
        val splitForRing: Boolean = true,
    ) {
        val pool = if (!windowed) null else config.windowedKvPool(bs, context, fullBlocks, ringPages, prefillChunk = chunk)
        val model = config.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs, windowedKv = pool)
        private val built = HashMap<Pair<DecodeGraphKind, Int>, Pair<DecodeGraphSpec, DxirFunction>>()
        fun graph(kind: DecodeGraphKind, batch: Int) = built.getOrPut(kind to batch) {
            val spec = HfDecoderGraph.spec(
                config, model, DecodeBucket(batch, context), kind,
                prefillChunk = if (kind == DecodeGraphKind.PREFILL) chunk else null,
            )
            spec to HfDecoderGraph.build(spec, config)
        }
        var pools: List<FloatArray> = graph(DecodeGraphKind.DECODE, 1).first.let { spec ->
            (0 until config.numLayers).flatMap { l ->
                val n = spec.poolTypeOf(l).dims.fold(1) { a, b -> a * b }
                listOf(FloatArray(n), FloatArray(n))
            }
        }
        val freeFull = ArrayDeque((1 until fullBlocks).toList())
        val freeWindow = ArrayDeque((1 until (pool?.numBlocks ?: 1)).toList())
        val full = HashMap<String, MutableList<Int>>()
        val ring = HashMap<String, MutableList<Int>>()
        val length = HashMap<String, Int>()
        val rows = ArrayList<Row>()
        val calls = ArrayList<List<String>>()

        fun admit(seq: String, n: Int) {
            val pages = (length.getOrPut(seq) { 0 } + n + bs - 1) / bs
            val f = full.getOrPut(seq) { ArrayList() }
            while (f.size < pages) f += freeFull.removeFirst()
            if (pool != null) {
                val r = ring.getOrPut(seq) { ArrayList() }
                while (r.size < minOf(pool.ringPages, pages)) r += freeWindow.removeFirst()
            }
        }

        fun windowPage(seq: String, block: Int) = ring.getValue(seq)[block % pool!!.ringPages]

        /** Runs [work] (sequence to its call's tokens) as one call of a [kind] entry of [batch] rows. */
        fun call(kind: DecodeGraphKind, batch: Int, work: List<Pair<String, IntArray>>) {
            val (spec, fn) = graph(kind, batch)
            val t = spec.tokensPerSeq
            val m = spec.maxBlocksPerSeq
            val tokenIds = FloatArray(batch * t)
            val positions = FloatArray(batch * t)
            val tables = FloatArray(batch * m)
            val seqLens = FloatArray(batch) { 1f }
            val slots = FloatArray(batch * t) { -1f }
            val wTables = FloatArray(batch * m)
            val wSlots = FloatArray(batch * t) { -1f }
            for ((seq, toks) in work) admit(seq, toks.size)
            for ((r, w) in work.withIndex()) {
                val (seq, toks) = w
                val start = length.getValue(seq)
                val pad = t - toks.size
                if (padWrites && r == 1 && pad > 0) {
                    slots[r * t + if (seq == leftAligned) toks.size else 0] = (full.getValue(work[0].first)[0] * bs).toFloat()
                }
                for (j in toks.indices) {
                    val pos = start + j
                    val k = r * t + if (seq == leftAligned) j else pad + j
                    tokenIds[k] = toks[j].toFloat()
                    positions[k] = pos.toFloat()
                    slots[k] = (full.getValue(seq)[pos / bs] * bs + pos % bs).toFloat()
                    if (pool != null) wSlots[k] = (windowPage(seq, pos / bs) * bs + pos % bs).toFloat()
                }
                for (b in 0 until (start + toks.size + bs - 1) / bs) {
                    tables[r * m + b] = full.getValue(seq)[b].toFloat()
                    if (pool != null) wTables[r * m + b] = windowPage(seq, b).toFloat()
                }
                seqLens[r] = (start + toks.size).toFloat()
            }
            val out = DxirInterpreter.evalFunction(
                fn,
                buildList {
                    add(tokenIds); add(positions); add(tables); add(seqLens); add(slots)
                    if (pool != null) { add(wTables); add(wSlots) }
                    addAll(pools)
                    addAll(weights)
                },
            )
            pools = out.drop(1)
            calls += work.map { it.first }
            val v = config.vocabSize
            for ((r, w) in work.withIndex()) {
                val end = length.getValue(w.first) + w.second.size
                length[w.first] = end
                rows += Row(w.first, end - 1, out[0].copyOfRange(r * v, (r + 1) * v))
            }
        }

        /**
         * Prefills [work] together: in rounds, each round one call holding
         * every prompt with tokens left, each at most as many tokens as its
         * ring allows, through the smallest ladder batch that holds them.
         */
        fun prefillTogether(work: List<Pair<String, IntArray>>) {
            val done = work.associate { it.first to 0 }.toMutableMap()
            while (true) {
                val round = work.mapNotNull { (seq, toks) ->
                    val i = done.getValue(seq)
                    if (i >= toks.size) return@mapNotNull null
                    val ringMost = if (splitForRing) pool?.maxTokensPerCall(length[seq] ?: 0, bs) else null
                    val most = minOf(ringMost ?: toks.size, chunk ?: toks.size)
                    seq to toks.copyOfRange(i, minOf(toks.size, i + most))
                }
                if (round.isEmpty()) break
                call(DecodeGraphKind.PREFILL, batchLadder.first { it >= round.size }, round)
                for ((seq, toks) in round) done[seq] = done.getValue(seq) + toks.size
            }
        }

        fun decode(seq: String, token: Int) = call(DecodeGraphKind.DECODE, 1, listOf(seq to intArrayOf(token)))

        /** Greedy decoding of [steps] tokens per sequence, one sequence at a time. */
        fun continueAll(seqs: List<String>, steps: Int) {
            repeat(steps) {
                for (s in seqs) decode(s, argmax(rows.last { it.seq == s }.logits))
            }
        }

        fun rowsOf(seq: String) = rows.filter { it.seq == seq }
    }

    private fun argmax(v: FloatArray): Int {
        var best = 0
        for (i in v.indices) if (v[i] > v[best]) best = i
        return best
    }

    /** Each sequence prefilled alone, then decoded on: the reference rows. */
    private fun solo(windowed: Boolean, seqs: List<String>, steps: Int): Map<String, List<Row>> =
        seqs.associateWith { s ->
            val h = Host(windowed)
            h.prefillTogether(listOf(s to prompts.getValue(s)))
            h.continueAll(listOf(s), steps)
            h.rowsOf(s)
        }

    /**
     * The rows at the positions both runs returned logits for (a chunked
     * prefill returns them at other chunk ends): the last prompt position and
     * every decode step, at least [common] of them.
     */
    private fun assertSameCommonRows(want: List<Row>, got: List<Row>, common: Int, what: String) {
        val both = want.map { it.position }.intersect(got.map { it.position }.toSet())
        assertEquals(common, both.size, "$what: positions returned by both runs")
        assertSameRows(want.filter { it.position in both }, got.filter { it.position in both }, what)
    }

    private fun assertSameRows(want: List<Row>, got: List<Row>, what: String) {
        assertEquals(want.map { it.position }, got.map { it.position }, "$what: positions")
        for ((a, b) in want.zip(got)) assertContentEquals(a.logits, b.logits, "$what at position ${a.position}")
    }

    @Test
    fun promptsPrefilledTogetherGiveTheirSoloLogitsBitForBit() {
        for (windowed in listOf(false, true)) {
            val seqs = listOf("A", "B", "C", "D")
            val want = solo(windowed, seqs, steps = 6)
            val h = Host(windowed)
            h.prefillTogether(seqs.map { it to prompts.getValue(it) })
            // One call per round: every prompt fits one call without a windowed
            // pool; with one, C's 17 tokens are split 12 + 5 by the ring.
            if (windowed) {
                assertEquals(listOf(seqs, listOf("C")), h.calls)
            } else {
                assertEquals(listOf(seqs), h.calls)
            }
            h.continueAll(seqs, steps = 6)
            for (s in seqs) assertSameRows(want.getValue(s), h.rowsOf(s), "windowed=$windowed, sequence $s")
        }
    }

    @Test
    fun threePromptsInABatchOfFourGiveTheirSoloLogits() {
        // Three prompts run in the batch-4 entry with one padding row.
        val seqs = listOf("B", "A", "D")
        val want = solo(windowed = true, seqs, steps = 3)
        val h = Host(windowed = true)
        h.prefillTogether(seqs.map { it to prompts.getValue(it) })
        assertEquals(listOf(seqs), h.calls)
        h.continueAll(seqs, steps = 3)
        for (s in seqs) assertSameRows(want.getValue(s), h.rowsOf(s), "sequence $s")
    }

    @Test
    fun aSecondChunkPrefilledTogetherContinuesEachSequence() {
        // Two sequences decode a few steps apart, then get a further chunk
        // each in one batched call, at different positions.
        fun run(together: Boolean): Map<String, List<Row>> {
            val results = HashMap<String, List<Row>>()
            val hosts = if (together) listOf(Host(true)) else listOf(Host(true), Host(true))
            val a = hosts.first()
            val b = hosts.last()
            a.prefillTogether(listOf("A" to prompts.getValue("A")))
            b.prefillTogether(listOf("B" to prompts.getValue("B")))
            a.continueAll(listOf("A"), 2)
            b.continueAll(listOf("B"), 4)
            val more = listOf("A" to tokens(5, 9), "B" to tokens(6, 7))
            if (together) {
                a.prefillTogether(more)
            } else {
                a.prefillTogether(more.take(1))
                b.prefillTogether(more.drop(1))
            }
            a.continueAll(listOf("A"), 2)
            b.continueAll(listOf("B"), 2)
            results["A"] = a.rowsOf("A")
            results["B"] = b.rowsOf("B")
            return results
        }
        val want = run(together = false)
        val got = run(together = true)
        for (s in listOf("A", "B")) assertSameRows(want.getValue(s), got.getValue(s), "sequence $s")
    }

    /**
     * Negative control: A's prompt is prefilled in two chunks, the second
     * together with B's prompt, and a padding token of B's row writes its KV
     * over A's position 0 (which that call does not write). A's logits
     * change from then on; with the padding written nowhere they do not.
     */
    @Test
    fun paddingThatWritesItsKvChangesTheLogits() {
        val a = prompts.getValue("A")
        fun run(padWrites: Boolean): Host {
            val h = Host(windowed = false, padWrites = padWrites)
            h.prefillTogether(listOf("A" to a.copyOfRange(0, 6)))
            h.prefillTogether(listOf("A" to a.copyOfRange(6, a.size), "B" to prompts.getValue("B")))
            h.continueAll(listOf("A", "B"), 2)
            return h
        }
        val want = Host(windowed = false).apply {
            prefillTogether(listOf("A" to a.copyOfRange(0, 6)))
            prefillTogether(listOf("A" to a.copyOfRange(6, a.size)))
            continueAll(listOf("A"), 2)
        }.rowsOf("A")
        assertSameRows(want, run(padWrites = false).rowsOf("A"), "padding written nowhere")
        val got = run(padWrites = true).rowsOf("A")
        assertEquals(want.first().logits.toList(), got.first().logits.toList(), "the first chunk ran alone")
        assertTrue(want.drop(1).zip(got.drop(1)).none { (x, y) -> x.logits.contentEquals(y.logits) }, "A read the overwritten position")
    }

    /** Negative control: one row's tokens left-aligned put padding in its last position. */
    @Test
    fun aLeftAlignedRowChangesItsLogits() {
        val seqs = listOf("A", "B", "D")
        val want = solo(windowed = false, seqs, steps = 0)
        val h = Host(windowed = false, leftAligned = "B")
        h.prefillTogether(seqs.map { it to prompts.getValue(it) })
        assertFalse(want.getValue("B").single().logits.contentEquals(h.rowsOf("B").single().logits))
        // The other rows are untouched by it.
        for (s in listOf("A", "D")) assertSameRows(want.getValue(s), h.rowsOf(s), "sequence $s")
    }

    @Test
    fun chunkedPrefillGivesTheWholeContextLogitsBitForBit() {
        for (windowed in listOf(false, true)) {
            val seqs = listOf("A", "C")
            val want = solo(windowed, seqs, steps = 4)
            val h = Host(windowed, chunk = 5)
            // The prefill entries take 5 tokens per sequence, not the context's 48.
            assertEquals(5, h.graph(DecodeGraphKind.PREFILL, 2).first.tokensPerSeq)
            h.prefillTogether(seqs.map { it to prompts.getValue(it) })
            // A (11 tokens: 5 + 5 + 1) and C (17: 5 + 5 + 5 + 2) in 4 rounds.
            assertEquals(listOf(seqs, seqs, seqs, listOf("C")), h.calls)
            h.continueAll(seqs, steps = 4)
            for (s in seqs) {
                assertSameCommonRows(want.getValue(s), h.rowsOf(s), 5, "windowed=$windowed, sequence $s")
            }
        }
    }

    @Test
    fun theRingIsSizedForAChunkPastTheWindow() {
        // Window 8, pages of 4: the default ring is 3 pages (12 positions); a
        // call of 8 tokens past the window needs 7 + 8 = 15, so 4 pages.
        assertEquals(3, config.windowedKvPool(bs, context, fullBlocks)!!.ringPages)
        val pool = config.windowedKvPool(bs, context, fullBlocks, prefillChunk = 8)!!
        assertEquals(4, pool.ringPages)
        assertTrue(pool.maxTokensPerCall(start = 40, blockSize = bs) >= 8)
        // Capped at the context's pages, where a ring never wraps.
        assertEquals(context / bs, config.windowedKvPool(bs, context, fullBlocks, prefillChunk = 1000)!!.ringPages)
        val a = prompts.getValue("C")
        val want = solo(windowed = true, listOf("C"), steps = 2).getValue("C")
        val h = Host(windowed = true, chunk = 8)
        h.prefillTogether(listOf("C" to a))
        // 17 tokens in chunks of 8, none split by the ring.
        assertEquals(listOf(listOf("C"), listOf("C"), listOf("C")), h.calls)
        h.continueAll(listOf("C"), 2)
        assertSameCommonRows(want, h.rowsOf("C"), 3, "ring of 4 pages")
    }

    /**
     * Negative control: the same chunks of 8 through the default ring of 3
     * pages, not split for it: the second chunk writes positions 8..15 over
     * 0..3, which its first rows still read, and the logits change.
     */
    @Test
    fun chunksTooLongForTheRingChangeTheLogits() {
        val a = prompts.getValue("C")
        val want = solo(windowed = true, listOf("C"), steps = 2).getValue("C")
        val h = Host(windowed = true, chunk = 8, ringPages = 3, splitForRing = false)
        h.prefillTogether(listOf("C" to a))
        assertEquals(3, h.calls.size)
        h.continueAll(listOf("C"), 2)
        val common = want.map { it.position }.intersect(h.rowsOf("C").map { it.position }.toSet())
        assertEquals(3, common.size, "the last prompt position and two decode steps")
        val w = want.filter { it.position in common }
        val got = h.rowsOf("C").filter { it.position in common }
        assertTrue(w.zip(got).none { (x, y) -> x.logits.contentEquals(y.logits) }, "the overwritten positions were read")
    }

    @Test
    fun aPrefillChunkIsRefusedOnADecodeEntry() {
        val model = config.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs)
        val e = kotlin.runCatching {
            HfDecoderGraph.spec(config, model, DecodeBucket(1, context), DecodeGraphKind.DECODE, prefillChunk = 4)
        }.exceptionOrNull()
        assertTrue(e?.message?.contains("prefillChunk 4 applies to a prefill entry") == true, "got ${e?.message}")
        val zero = kotlin.runCatching {
            HfDecoderGraph.spec(config, model, DecodeBucket(1, context), DecodeGraphKind.PREFILL, prefillChunk = 0)
        }.exceptionOrNull()
        assertTrue(zero?.message?.contains("must be >= 1") == true, "got ${zero?.message}")
        // A chunk at least the context takes the whole context.
        assertEquals(
            context,
            HfDecoderGraph.spec(config, model, DecodeBucket(1, context), DecodeGraphKind.PREFILL, prefillChunk = 99).tokensPerSeq,
        )
    }

    @Test
    fun theComparisonSeesADifferentPrompt() {
        // The solo rows of two different prompts differ, so equal rows mean something.
        val want = solo(windowed = true, listOf("A", "B"), steps = 1)
        assertTrue(want.getValue("A").zip(want.getValue("B")).none { (a, b) -> a.logits.contentEquals(b.logits) })
    }
}
