package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Windowed KV pools: sliding-window layers whose pages each sequence recycles
 * in a ring.
 *
 * The claim: a model whose sliding layers keep their KV in a
 * [WindowedKvPool] ring computes, in the reference interpreter, logits
 * bit-identical to the same model with full-history pools, at every call of a
 * schedule that runs two sequences several windows long through prefill
 * chunks, batched decode steps and a padded batch. The host side here does
 * what the Triton backend does: pages taken from a free list, a ring per
 * sequence, requests split by [WindowedKvPool.maxTokensPerCall].
 *
 * The negative controls are the two off-by-ones of recycling: a ring one page
 * below [WindowedKvPool.minRingPages] (a page recycled while its last
 * position is still in the window), and a request split into calls one
 * token longer than [WindowedKvPool.maxTokensPerCall] allows (the extra token
 * overwrites a position the call's first row still reads). Both must change
 * the logits.
 */
class WindowedKvPoolTest {

    // ------------------------------------------------------ the ring rules

    /** The slot of position [p] in a ring of [ring] pages of [bs]: page index in the ring, then offset. */
    private fun ringSlotOf(p: Int, ring: Int, bs: Int) = (p / bs) % ring * bs + p % bs

    /** Whether the positions [lo]..[hi] are on distinct slots of the ring. */
    private fun distinct(lo: Int, hi: Int, ring: Int, bs: Int) =
        (lo..hi).map { ringSlotOf(it, ring, bs) }.toSet().size == hi - lo + 1

    @Test
    fun theMinimumRingHoldsOneWindow() {
        // Window 8 in pages of 4: two pages. Window 2048 in pages of 16: 128.
        assertEquals(2, WindowedKvPool.minRingPages(8, 4))
        assertEquals(3, WindowedKvPool.defaultRingPages(8, 4))
        assertEquals(1, WindowedKvPool.minRingPages(1, 4))
        assertEquals(3, WindowedKvPool.minRingPages(9, 4))
        assertEquals(128, WindowedKvPool.minRingPages(2048, 16))
        // Exhaustively: with the minimum, every window a decode step reads is
        // on distinct slots; with one page fewer, some window is not.
        for (w in 1..20) for (bs in 1..6) {
            val min = WindowedKvPool.minRingPages(w, bs)
            for (p in 0..60) assertTrue(distinct(maxOf(0, p - w + 1), p, min, bs), "window $w, pages of $bs, at $p")
            if (min > 1) {
                assertFalse((0..60).all { p -> distinct(maxOf(0, p - w + 1), p, min - 1, bs) }, "window $w, pages of $bs")
            }
        }
    }

    @Test
    fun maxTokensPerCallIsExactlyWhatTheRingHolds() {
        for (w in 1..12) for (bs in 1..5) for (extra in 0..2) {
            val ring = WindowedKvPool.minRingPages(w, bs) + extra
            val pool = WindowedKvPool(window = w, layers = listOf(0), numBlocks = ring + 1, ringPages = ring)
            for (p0 in 0..50) {
                val n = pool.maxTokensPerCall(p0, bs)
                assertTrue(n >= 1, "window $w, pages of $bs, ring $ring, start $p0: $n")
                val lo = maxOf(0, p0 - w + 1)
                assertTrue(distinct(lo, p0 + n - 1, ring, bs), "n fits")
                assertFalse(distinct(lo, p0 + n, ring, bs), "n + 1 does not")
                if (extra >= 1) assertTrue(n >= bs + 1, "a spare page lets every call write more than a page")
            }
        }
    }

    @Test
    fun aSequenceHoldsAtMostTheRing() {
        val pool = WindowedKvPool(window = 8, layers = listOf(0), numBlocks = 9, ringPages = 4)
        assertEquals(listOf(0, 1, 1, 1, 1, 2, 4, 4, 4), listOf(0, 1, 2, 3, 4, 5, 16, 17, 400).map { pool.pagesHeld(it, 4) })
        assertEquals(listOf(0, 1, 2, 3, 0, 1), (0..5).map { pool.ringSlot(it) })
    }

    // ------------------------------------------------------- the contract

    private val bs = 4
    private val context = 48
    private val fullBlocks = 1 + 2 * (context / bs)

    /** Two sliding layers (windows 8 and 6) and a full one; pages of 4, contexts up to 48. */
    private val tiny = HfDecoderConfig(
        architecture = "LlamaForCausalLM", modelType = "llama",
        hiddenSize = 16, intermediateSize = 24, numLayers = 3,
        numHeads = 4, numKvHeads = 2, headDim = 8, vocabSize = 29,
        rmsNormEps = 1e-6, ropeTheta = 10000.0, maxPositionEmbeddings = 64,
        tieWordEmbeddings = false, attentionBias = false, torchDtype = "float32",
        ropeScalingType = null, family = HfModelFamily.Llama,
        layers = listOf(
            DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 8),
            DecoderLayerSpec(attention = AttentionKind.SLIDING, slidingWindow = 6),
            DecoderLayerSpec(attention = AttentionKind.FULL),
        ),
    )

    private fun windowed(ringPages: Int? = null) =
        tiny.windowedKvPool(blockSize = bs, maxContext = context, fullNumBlocks = fullBlocks, ringPages = ringPages)!!

    @Test
    fun theConfigsSlidingLayersFormTheWindowedClass() {
        val w = windowed()
        assertEquals(8, w.window)
        assertEquals(listOf(0, 1), w.layers)
        assertEquals(3, w.ringPages)
        // Two sequences of 48 positions fit the full pool; two rings fit this one.
        assertEquals(1 + 2 * 3, w.numBlocks)
        assertNull(tiny.copy(layers = null).windowedKvPool(bs, context, fullBlocks))
        // A ring as long as the largest context never wraps; none is longer.
        assertEquals(2, tiny.windowedKvPool(bs, 8, fullBlocks)!!.ringPages)
        assertEquals(1, tiny.windowedKvPool(bs, 4, fullBlocks)!!.ringPages)
    }

    @Test
    fun theWindowedSignatureAddsTwoTablesAndRetypesTheSlidingPools() {
        val model = tiny.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs, windowedKv = windowed())
        val spec = HfDecoderGraph.spec(tiny, model, DecodeBucket(2, context))
        assertEquals(
            listOf("tokenIds", "positions", "blockTables", "seqLens", "slotMapping", "windowBlockTables", "windowSlotMapping") +
                (0 until 3).flatMap { listOf("keyCache$it", "valueCache$it") },
            spec.inputs.take(13).map { it.name },
        )
        assertEquals(DecodeSlotRole.WINDOW_BLOCK_TABLES, spec.inputs[5].role)
        assertEquals(DxirType(I32, listOf(2, 12)), spec.inputs[5].type)
        assertEquals(DecodeSlotRole.WINDOW_SLOT_MAPPING, spec.inputs[6].role)
        assertEquals(7, spec.kvPoolInputBase)
        val window = DxirType(F32, listOf(7, bs, 2, 8))
        val full = DxirType(F32, listOf(fullBlocks, bs, 2, 8))
        assertEquals(List(4) { window } + List(2) { full }, spec.inputs.subList(7, 13).map { it.type })
        assertEquals(
            List(4) { DecodeSlotRole.WINDOW_KV_POOL_IN } + List(2) { DecodeSlotRole.KV_POOL_IN },
            spec.inputs.subList(7, 13).map { it.role },
        )
        assertEquals(
            List(4) { DecodeSlotRole.WINDOW_KV_POOL_OUT } + List(2) { DecodeSlotRole.KV_POOL_OUT },
            spec.outputs.drop(1).map { it.role },
        )
        assertEquals((0 until 6).map { (7 + it) to (1 + it) }, spec.donationPairs)
        for ((i, o) in spec.donationPairs) assertEquals(spec.inputs[i].type, spec.outputs[o].type)
        // The artifact without a windowed pool keeps its signature and its key.
        val plain = HfDecoderGraph.spec(tiny, tiny.toDecodeModelShape(fullBlocks, bs), DecodeBucket(2, context))
        assertEquals(5, plain.kvPoolInputBase)
        assertTrue(plain.executableCacheKey("h").endsWith("/c48/t1/dtf32/kvf32"), plain.executableCacheKey("h"))
        assertEquals(plain.executableCacheKey("h") + "/w8r3n7", spec.executableCacheKey("h"))
        HfDecoderGraph.build(spec, tiny)
    }

    @Test
    fun aRingBelowTheMinimumIsRefusedByName() {
        val short = WindowedKvPool(window = 8, layers = listOf(0, 1), numBlocks = 9, ringPages = 1)
        val model = tiny.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs, windowedKv = short)
        val e = assertFailsWith<IllegalArgumentException> { HfDecoderGraph.spec(tiny, model, DecodeBucket(1, context)) }
        assertTrue("a decode step needs 2" in e.message!!, e.message!!)
    }

    @Test
    fun aFullAttentionLayerInTheWindowedPoolIsRefusedByName() {
        val wrong = WindowedKvPool(window = 8, layers = listOf(0, 2), numBlocks = 9, ringPages = 4)
        val model = tiny.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs, windowedKv = wrong)
        val e = assertFailsWith<IllegalArgumentException> {
            HfDecoderGraph.build(HfDecoderGraph.spec(tiny, model, DecodeBucket(1, context)), tiny)
        }
        assertTrue("layer 2 keeps its KV in the windowed pool" in e.message!!, e.message!!)
        val narrow = WindowedKvPool(window = 7, layers = listOf(0, 1), numBlocks = 9, ringPages = 4)
        val e2 = assertFailsWith<IllegalArgumentException> {
            val m = tiny.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs, windowedKv = narrow)
            HfDecoderGraph.build(HfDecoderGraph.spec(tiny, m, DecodeBucket(1, context)), tiny)
        }
        assertTrue("with window 8" in e2.message!!, e2.message!!)
    }

    // ------------------------------------------- the interpreter's logits

    private val weights: List<FloatArray> = run {
        val rng = Random(20260925)
        HfDecoderGraph.weightSlots(tiny).map { slot ->
            val n = slot.type.dims.fold(1) { a, b -> a * b }
            if (slot.type.dims.size == 1) {
                FloatArray(n) { 1f + 0.5f * (rng.nextFloat() - 0.5f) }
            } else {
                FloatArray(n) { 0.9f * (rng.nextFloat() - 0.5f) }
            }
        }
    }

    private val tokensA = Random(1).let { r -> IntArray(41) { r.nextInt(tiny.vocabSize) } }
    private val tokensB = Random(2).let { r -> IntArray(31) { r.nextInt(tiny.vocabSize) } }

    /** One logits row the host got back: which sequence, and the position of its last token. */
    private data class Row(val seq: String, val position: Int, val logits: FloatArray)

    /**
     * What the backend does, in Kotlin: pages from free lists (page 0 is the
     * padding page), a ring of windowed pages per sequence, and requests
     * split into calls of at most [WindowedKvPool.maxTokensPerCall] tokens.
     *
     * [chunkRule] decides the splitting, so the full-history run can follow
     * the windowed run's schedule call for call. [ringUsed] is the ring the
     * host actually keeps, and [chunkSlack] tokens are added to what the rule
     * allows; the controls set them wrong.
     */
    private inner class Host(
        val pool: WindowedKvPool?,
        val chunkRule: WindowedKvPool?,
        val ringUsed: Int? = pool?.ringPages,
        val chunkSlack: Int = 0,
    ) {
        val model = tiny.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs, windowedKv = pool)
        val decode = HfDecoderGraph.build(HfDecoderGraph.spec(tiny, model, DecodeBucket(2, context)), tiny)
        val prefill = HfDecoderGraph.build(
            HfDecoderGraph.spec(tiny, model, DecodeBucket(1, context), DecodeGraphKind.PREFILL), tiny,
        )
        val decodeSpec = HfDecoderGraph.spec(tiny, model, DecodeBucket(2, context))
        var pools: List<FloatArray> = (0 until tiny.numLayers).flatMap { l ->
            val n = decodeSpec.poolTypeOf(l).dims.fold(1) { a, b -> a * b }
            listOf(FloatArray(n), FloatArray(n))
        }
        val freeFull = ArrayDeque((1 until fullBlocks).toList())
        val freeWindow = ArrayDeque((1 until (pool?.numBlocks ?: 1)).toList())
        val full = HashMap<String, MutableList<Int>>()
        val ring = HashMap<String, MutableList<Int>>()
        val length = HashMap<String, Int>()
        val rows = ArrayList<Row>()
        val mostWindowPages = HashMap<String, Int>()

        fun admit(seq: String, n: Int) {
            val len = length.getOrPut(seq) { 0 } + n
            val pages = (len + bs - 1) / bs
            val f = full.getOrPut(seq) { ArrayList() }
            while (f.size < pages) f += freeFull.removeFirst()
            if (pool != null) {
                val r = ring.getOrPut(seq) { ArrayList() }
                while (r.size < minOf(ringUsed!!, pages)) r += freeWindow.removeFirst()
                mostWindowPages[seq] = maxOf(mostWindowPages[seq] ?: 0, r.size)
            }
        }

        fun windowPage(seq: String, block: Int): Int = ring.getValue(seq)[block % ringUsed!!]

        /** Runs [work] (sequence to its new tokens) as one call of [fn]. */
        fun call(fn: io.tlaloc.ir.DxirFunction, batch: Int, t: Int, work: List<Pair<String, IntArray>>) {
            val m = context / bs
            val tokenIds = FloatArray(batch * t)
            val positions = FloatArray(batch * t)
            val tables = FloatArray(batch * m)
            val seqLens = FloatArray(batch) { 1f }
            val slots = FloatArray(batch * t) { -1f }
            val wTables = FloatArray(batch * m)
            val wSlots = FloatArray(batch * t) { -1f }
            for ((r, w) in work.withIndex()) {
                val (seq, toks) = w
                admit(seq, toks.size)
                val start = length.getValue(seq)
                val pad = t - toks.size
                for (j in toks.indices) {
                    val pos = start + j
                    val k = r * t + pad + j
                    tokenIds[k] = toks[j].toFloat()
                    positions[k] = pos.toFloat()
                    slots[k] = (full.getValue(seq)[pos / bs] * bs + pos % bs).toFloat()
                    if (pool != null) wSlots[k] = (windowPage(seq, pos / bs) * bs + pos % bs).toFloat()
                }
                val blocks = (start + toks.size + bs - 1) / bs
                for (b in 0 until blocks) {
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
            val v = tiny.vocabSize
            for ((r, w) in work.withIndex()) {
                val (seq, toks) = w
                val end = length.getValue(seq) + toks.size
                length[seq] = end
                rows += Row(seq, end - 1, out[0].copyOfRange(r * v, (r + 1) * v))
            }
        }

        /** A request of several tokens: prefill calls, each as long as the chunk rule allows. */
        fun request(seq: String, toks: IntArray) {
            var i = 0
            while (i < toks.size) {
                val start = length[seq] ?: 0
                // The slack applies once the window is full, where the extra
                // token lands on the oldest position the call's first row reads
                // (earlier, it would land on one of the call's own tokens).
                val slack = if (chunkRule != null && start >= chunkRule.window - 1) chunkSlack else 0
                val most = maxOf(1, (chunkRule?.maxTokensPerCall(start, bs) ?: toks.size) + slack)
                val n = minOf(toks.size - i, most)
                call(prefill, 1, context, listOf(seq to toks.copyOfRange(i, i + n)))
                i += n
            }
        }

        fun decodeStep(vararg work: Pair<String, Int>) =
            call(decode, 2, 1, work.map { (s, tok) -> s to intArrayOf(tok) })

        /** Two sequences, several windows long, through every kind of call. */
        fun schedule(): List<Row> {
            request("A", tokensA.copyOfRange(0, 13))
            request("B", tokensB.copyOfRange(0, 6))
            for (k in 0 until 7) decodeStep("A" to tokensA[13 + k], "B" to tokensB[6 + k])
            request("A", tokensA.copyOfRange(20, 31))
            for (k in 0 until 9) decodeStep("A" to tokensA[31 + k], "B" to tokensB[13 + k])
            request("B", tokensB.copyOfRange(22, 31))
            decodeStep("A" to tokensA[40])
            return rows
        }
    }

    private fun assertSameRows(want: List<Row>, got: List<Row>) {
        assertEquals(want.map { it.seq to it.position }, got.map { it.seq to it.position })
        for ((a, b) in want.zip(got)) {
            assertContentEquals(a.logits, b.logits, "sequence ${a.seq} at position ${a.position}")
        }
    }

    @Test
    fun windowedRingsGiveTheFullHistoryLogitsBitForBit() {
        val pool = windowed()
        val host = Host(pool, pool)
        val got = host.schedule()
        val want = Host(null, pool).schedule()
        // 41 and 31 positions: five and four windows of 8, rings of 3 pages.
        assertEquals(40, host.length["A"]!! - 1)
        assertEquals(30, host.length["B"]!! - 1)
        assertSameRows(want, got)
        // The ring wrapped, and never grew past its 3 pages; the full layers hold 11 and 8.
        assertEquals(mapOf("A" to 3, "B" to 3), host.mostWindowPages)
        assertEquals(11, host.full.getValue("A").size)
        assertEquals(8, host.full.getValue("B").size)
        // The ring holds 12 positions. A's 13-token prompt is split 12 + 1;
        // once the window is full a call writes 12 - 7 = 5 tokens: A's 11 from
        // position 20 into 5, 5 and 1, B's 9 from 22 into 5 and 4.
        assertEquals(
            listOf(11, 12) + (13..19) + listOf(24, 29, 30) + (31..40),
            host.rows.filter { it.seq == "A" }.map { it.position },
        )
        assertEquals(
            listOf(5) + (6..21) + listOf(26, 30),
            host.rows.filter { it.seq == "B" }.map { it.position },
        )
    }

    @Test
    fun theMinimumRingIsEnough() {
        val pool = windowed(ringPages = WindowedKvPool.minRingPages(8, bs))
        val host = Host(pool, pool)
        assertSameRows(Host(null, pool).schedule(), host.schedule())
        assertEquals(mapOf("A" to 2, "B" to 2), host.mostWindowPages)
    }

    /**
     * Negative control: a ring one page below the minimum, with requests split
     * for that ring, recycles a page while the window still reads it.
     */
    @Test
    fun aRingOnePageShortChangesTheLogits() {
        val min = WindowedKvPool.minRingPages(8, bs)
        val pool = windowed(ringPages = min)
        val short = pool.copy(ringPages = min - 1)
        val want = Host(null, short).schedule()
        val got = Host(pool, short, ringUsed = min - 1).schedule()
        assertEquals(want.map { it.seq to it.position }, got.map { it.seq to it.position })
        val differ = want.zip(got).filter { (a, b) -> !a.logits.contentEquals(b.logits) }
        assertTrue(differ.isNotEmpty(), "a ring one page short went unnoticed")
        // Nothing differs before the one-page ring first wraps, at position 4.
        assertTrue(differ.all { it.first.position >= 4 }, "${differ.map { it.first.position }}")
        assertFailsWith<AssertionError> { assertSameRows(want, got) }
    }

    /**
     * Negative control: a call one token longer than the ring allows writes
     * over a position its first row still reads.
     */
    @Test
    fun aChunkOneTokenTooLongChangesTheLogits() {
        val pool = windowed()
        val want = Host(null, pool, chunkSlack = 1).schedule()
        val got = Host(pool, pool, chunkSlack = 1).schedule()
        assertEquals(want.map { it.seq to it.position }, got.map { it.seq to it.position })
        assertFailsWith<AssertionError> { assertSameRows(want, got) }
    }

    /** The comparison sees the window: full attention in the sliding layers changes the logits. */
    @Test
    fun theComparisonSeesTheWindow() {
        val pool = windowed()
        val slidingRows = Host(null, pool).schedule()
        val fullAttention = tiny.copy(layers = tiny.layers!!.map { it.copy(attention = AttentionKind.FULL, slidingWindow = null) })
        val model = fullAttention.toDecodeModelShape(numBlocks = fullBlocks, blockSize = bs)
        val fn = HfDecoderGraph.build(HfDecoderGraph.spec(fullAttention, model, DecodeBucket(1, context), DecodeGraphKind.PREFILL), fullAttention)
        val m = context / bs
        val out = DxirInterpreter.evalFunction(
            fn,
            buildList {
                add(FloatArray(context) { if (it >= context - 13) tokensA[it - (context - 13)].toFloat() else 0f })
                add(FloatArray(context) { if (it >= context - 13) (it - (context - 13)).toFloat() else 0f })
                add(FloatArray(m) { (it + 1).toFloat() })
                add(floatArrayOf(13f))
                add(FloatArray(context) { if (it >= context - 13) (bs + it - (context - 13)).toFloat() else -1f })
                (0 until 2 * tiny.numLayers).forEach { _ -> add(FloatArray(fullBlocks * bs * 2 * 8)) }
                addAll(weights)
            },
        )
        val first = slidingRows.first { it.seq == "A" && it.position == 12 }
        assertNotEquals(first.logits.toList(), out[0].toList(), "13 tokens with windows of 8 and 6 differ from full attention")
    }
}
