package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.render.KotlinRenderRefusal
import io.tlaloc.ir.render.toKotlinSource
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.465 — Phase H1a: PAGED_ATTENTION, vLLM's decode primitive.
 *
 * The oracle story, strongest first:
 *
 * 1. **Paged-vs-DENSE equivalence.** Build the SAME attention two ways —
 *    the paged form, and a dense contiguous walk over a K/V window that was
 *    materialised by *separately* gathering the pages the block table names —
 *    and pin elementwise equality. This is the strongest oracle available
 *    without a second implementation of paged attention itself: it checks the
 *    softmax/GQA math AND the block-table indexing at once, and it fails if
 *    the walk reads a page the table did not name, or reads a named page at
 *    the wrong offset. Run with an IDENTITY table (pages already contiguous)
 *    and with a PERMUTED table + ragged seqLens, so contiguity is never what
 *    is making the two agree.
 * 2. **A hand-exact tiny case.** 2 sequences, different seqLens, a
 *    partially-filled last block. The keys are chosen so every score ties,
 *    which makes the softmax exactly uniform and the answer the plain MEAN of
 *    the live V rows — arithmetic a reader can check on paper. The slots at
 *    and past seqLen are POISONED with a huge value, so a single over-read
 *    moves the answer by orders of magnitude rather than by epsilon.
 * 3. **The named refusals.** Inference-only by design ⇒ both AD transforms
 *    and the Kotlin renderer refuse BY NAME, naming the training alternative.
 */
class PagedAttentionTest {

    // --- The paged shape under test (small, but nothing degenerate). ---
    private val numSeqs = 2
    private val numHeads = 4
    private val numKvHeads = 2 // GQA group = 2
    private val headDim = 3
    private val blockSize = 2
    private val numBlocks = 6
    private val maxBlocksPerSeq = 3
    private val ctx = maxBlocksPerSeq * blockSize // 6
    private val group = numHeads / numKvHeads
    private val scale = 0.5

    private val qType = DxirType(F32, listOf(numSeqs, numHeads, headDim))
    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val tableType = DxirType(I32, listOf(numSeqs, maxBlocksPerSeq))
    private val lensType = DxirType(I32, listOf(numSeqs))

    private fun pagedFn(scaleAttr: Double = scale): DxirFunction = DxirBuilder.function("paged") {
        val q = param("q", qType)
        val k = param("k", cacheType)
        val v = param("v", cacheType)
        val t = param("t", tableType)
        val l = param("l", lensType)
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to scaleAttr)))
    }

    /** Deterministic spread-out values; an LCG so the test carries its own data. */
    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 2f
        }
    }

    // --- Oracle 1: the dense twin. -------------------------------------

    /**
     * The DENSE side of the equivalence: materialise sequence [s]'s context
     * window by gathering the pages its block table names (a separate,
     * explicit loop), then run plain contiguous attention over the live
     * prefix. No paged indexing appears past the gather.
     */
    private fun denseReference(
        query: FloatArray,
        keyCache: FloatArray,
        valueCache: FloatArray,
        table: IntArray,
        lens: IntArray,
        scaleUsed: Double = scale,
    ): FloatArray {
        val out = FloatArray(numSeqs * numHeads * headDim)
        for (s in 0 until numSeqs) {
            val len = lens[s]
            // Gather: window[t, kv, :] = cache[table[s][t / P], t % P, kv, :]
            val kWin = FloatArray(ctx * numKvHeads * headDim)
            val vWin = FloatArray(ctx * numKvHeads * headDim)
            for (t in 0 until ctx) {
                val blk = table[s * maxBlocksPerSeq + t / blockSize]
                val slot = t % blockSize
                for (kv in 0 until numKvHeads) {
                    for (j in 0 until headDim) {
                        val src = ((blk * blockSize + slot) * numKvHeads + kv) * headDim + j
                        val dst = (t * numKvHeads + kv) * headDim + j
                        kWin[dst] = keyCache[src]
                        vWin[dst] = valueCache[src]
                    }
                }
            }
            // Dense attention over the live prefix, contiguous throughout.
            for (h in 0 until numHeads) {
                val kv = h / group
                val qOff = (s * numHeads + h) * headDim
                val logits = DoubleArray(len)
                for (t in 0 until len) {
                    var dot = 0.0
                    for (j in 0 until headDim) {
                        dot += query[qOff + j].toDouble() * kWin[(t * numKvHeads + kv) * headDim + j]
                    }
                    logits[t] = dot * scaleUsed
                }
                val mx = logits.maxOrNull() ?: 0.0
                var denom = 0.0
                for (t in 0 until len) { logits[t] = exp(logits[t] - mx); denom += logits[t] }
                for (j in 0 until headDim) {
                    var acc = 0.0
                    for (t in 0 until len) {
                        acc += (logits[t] / denom) * vWin[(t * numKvHeads + kv) * headDim + j]
                    }
                    out[qOff + j] = acc.toFloat()
                }
            }
        }
        return out
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, tol: Float = 1e-5f, what: String) {
        kotlin.test.assertEquals(expected.size, actual.size, "$what: size")
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= tol,
                "$what: element $i expected ${expected[i]} but got ${actual[i]} (tol $tol)",
            )
        }
    }

    private fun runPaged(
        query: FloatArray,
        keyCache: FloatArray,
        valueCache: FloatArray,
        table: IntArray,
        lens: IntArray,
        scaleAttr: Double = scale,
    ): FloatArray = DxirInterpreter.evalFunction(
        pagedFn(scaleAttr),
        listOf(
            query, keyCache, valueCache,
            FloatArray(table.size) { table[it].toFloat() },
            FloatArray(lens.size) { lens[it].toFloat() },
        ),
    )[0]

    @Test
    fun pagedMatchesDenseWithAnIdentityBlockTable() {
        val q = pseudo(numSeqs * numHeads * headDim, 7)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 11)
        val v = pseudo(numBlocks * blockSize * numKvHeads * headDim, 13)
        // Sequence s owns pages 3s, 3s+1, 3s+2 in order — the pages ALREADY lie
        // contiguously, so paged and dense are looking at the same memory.
        val table = intArrayOf(0, 1, 2, 3, 4, 5)
        val lens = intArrayOf(ctx, ctx)
        assertClose(denseReference(q, k, v, table, lens), runPaged(q, k, v, table, lens), what = "identity table")
    }

    @Test
    fun pagedMatchesDenseWithAPermutedBlockTableAndRaggedLengths() {
        val q = pseudo(numSeqs * numHeads * headDim, 17)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 19)
        val v = pseudo(numBlocks * blockSize * numKvHeads * headDim, 23)
        // Pages scattered across the pool, interleaved between sequences — the
        // allocator's view. seq1 stops mid-page (3 = one full page + one slot).
        val table = intArrayOf(4, 1, 5, 2, 0, 3)
        val lens = intArrayOf(6, 3)
        assertClose(denseReference(q, k, v, table, lens), runPaged(q, k, v, table, lens), what = "permuted table")
    }

    @Test
    fun pagedMatchesDenseAcrossSeqLenSweep() {
        val q = pseudo(numSeqs * numHeads * headDim, 29)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 31)
        val v = pseudo(numBlocks * blockSize * numKvHeads * headDim, 37)
        val table = intArrayOf(5, 3, 1, 0, 4, 2)
        // Every length from 1 to ctx on seq0, paired against a different one on
        // seq1 — full and partial last blocks, both parities of blockSize.
        for (len0 in 1..ctx) {
            val lens = intArrayOf(len0, ctx - len0 + 1)
            assertClose(
                denseReference(q, k, v, table, lens),
                runPaged(q, k, v, table, lens),
                what = "seqLens ${lens.toList()}",
            )
        }
    }

    // --- Oracle 2: the hand-exact case. --------------------------------

    /**
     * Every key is the ZERO vector, so every score is 0·scale = 0 and the
     * softmax over the live prefix is EXACTLY uniform — the output is the mean
     * of the live V rows, computable on paper. The slots at and past seqLen
     * hold 1000.0, so over-reading even one of them is unmissable.
     */
    @Test
    fun handExactUniformSoftmaxOverPartiallyFilledPages() {
        val q = FloatArray(numSeqs * numHeads * headDim) { 1f }
        val k = FloatArray(numBlocks * blockSize * numKvHeads * headDim) // all zero ⇒ all scores tie
        val poison = 1000f
        val v = FloatArray(numBlocks * blockSize * numKvHeads * headDim) { poison }
        val table = intArrayOf(2, 0, 4, 5, 3, 1)
        val lens = intArrayOf(3, 2) // seq0: 1 full page + 1 slot; seq1: exactly 1 full page

        // Live V rows get 1, 2, 3, ... in logical context order; the rest stay poisoned.
        fun setLive(s: Int, len: Int) {
            for (t in 0 until len) {
                val blk = table[s * maxBlocksPerSeq + t / blockSize]
                val slot = t % blockSize
                for (kv in 0 until numKvHeads) {
                    for (j in 0 until headDim) {
                        v[((blk * blockSize + slot) * numKvHeads + kv) * headDim + j] = (t + 1).toFloat()
                    }
                }
            }
        }
        setLive(0, lens[0]); setLive(1, lens[1])

        val got = runPaged(q, k, v, table, lens)
        // seq0: mean(1,2,3) = 2. seq1: mean(1,2) = 1.5. Same for every head
        // (the ties make GQA grouping irrelevant here — grouping is oracle 1's job).
        for (h in 0 until numHeads) {
            for (j in 0 until headDim) {
                kotlin.test.assertEquals(2f, got[(0 * numHeads + h) * headDim + j], 1e-5f, "seq0 h$h d$j")
                kotlin.test.assertEquals(1.5f, got[(1 * numHeads + h) * headDim + j], 1e-5f, "seq1 h$h d$j")
            }
        }
    }

    @Test
    fun gqaGroupingIsContiguousPerKvHead() {
        // Two query heads that share a kv head must see the SAME K/V; make the
        // two kv heads' caches wildly different and the grouping is observable.
        val q = FloatArray(numSeqs * numHeads * headDim) { 1f }
        val k = FloatArray(numBlocks * blockSize * numKvHeads * headDim)
        val v = FloatArray(numBlocks * blockSize * numKvHeads * headDim) { i ->
            // kv head index is the second-to-last stride.
            val kv = (i / headDim) % numKvHeads
            if (kv == 0) 10f else -10f
        }
        val table = intArrayOf(0, 1, 2, 3, 4, 5)
        val lens = intArrayOf(4, 4)
        val got = runPaged(q, k, v, table, lens)
        for (s in 0 until numSeqs) {
            for (h in 0 until numHeads) {
                val expected = if (h / group == 0) 10f else -10f
                for (j in 0 until headDim) {
                    kotlin.test.assertEquals(expected, got[(s * numHeads + h) * headDim + j], 1e-5f, "s$s h$h")
                }
            }
        }
    }

    // --- Loud validation. ----------------------------------------------

    @Test
    fun outOfRangeBlockIdRefusesByName() {
        val q = pseudo(numSeqs * numHeads * headDim, 3)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 5)
        val table = intArrayOf(0, 1, numBlocks, 3, 4, 5) // page id == numBlocks: off the end
        val ex = assertFailsWith<IllegalArgumentException> {
            runPaged(q, k, k, table, intArrayOf(ctx, ctx))
        }
        assertTrue("blockTables" in (ex.message ?: ""), "must name blockTables; got ${ex.message}")
    }

    @Test
    fun seqLenBeyondTheBlockTableRefusesByName() {
        val q = pseudo(numSeqs * numHeads * headDim, 3)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 5)
        val ex = assertFailsWith<IllegalArgumentException> {
            runPaged(q, k, k, intArrayOf(0, 1, 2, 3, 4, 5), intArrayOf(ctx + 1, 1))
        }
        assertTrue("seqLens" in (ex.message ?: ""), "must name seqLens; got ${ex.message}")
    }

    @Test
    fun dimDerivedAttrsAreRefusedByName() {
        val fn = DxirBuilder.function("paged") {
            val q = param("q", qType)
            val k = param("k", cacheType)
            val v = param("v", cacheType)
            val t = param("t", tableType)
            val l = param("l", lensType)
            listOf(
                op(
                    OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType,
                    mapOf("scale" to scale, "blockSize" to blockSize),
                ),
            )
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(
                    FloatArray(numSeqs * numHeads * headDim),
                    FloatArray(numBlocks * blockSize * numKvHeads * headDim),
                    FloatArray(numBlocks * blockSize * numKvHeads * headDim),
                    FloatArray(numSeqs * maxBlocksPerSeq),
                    FloatArray(numSeqs),
                ),
            )
        }
        assertTrue("sentinel" in (ex.message ?: ""), "must cite the sentinel-dims rule; got ${ex.message}")
    }

    // --- Oracle 3: inference-only ⇒ named refusals. ---------------------

    /**
     * The transform-shaped primal: a SCALAR return, so the refusal fires on
     * the KIND rather than on the reverse transform's scalar-return
     * precondition — a refusal that fired for the wrong reason would prove
     * nothing about the inference-only gate.
     */
    private fun pagedLossFn(): DxirFunction = DxirBuilder.function("pagedLoss") {
        val q = param("q", qType)
        val k = param("k", cacheType)
        val v = param("v", cacheType)
        val t = param("t", tableType)
        val l = param("l", lensType)
        val y = op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to scale))
        listOf(op(OpKind.SUM, listOf(y), DxirType(F32, emptyList())))
    }

    private fun assertInferenceOnlyRefusal(ex: Throwable) {
        val msg = ex.message ?: ""
        assertTrue("PAGED_ATTENTION" in msg, "refusal must name the kind; got: $msg")
        assertTrue("INFERENCE-ONLY" in msg || "inference-only" in msg, "refusal must state the rationale; got: $msg")
        assertTrue("FlashAttention" in msg, "refusal must name the training alternative; got: $msg")
    }

    @Test
    fun reverseTransformRefusesPagedAttentionByName() {
        assertInferenceOnlyRefusal(assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(pagedLossFn()) })
    }

    @Test
    fun forwardTransformRefusesPagedAttentionByName() {
        assertInferenceOnlyRefusal(assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(pagedLossFn()) })
    }

    @Test
    fun kotlinRendererRefusesPagedAttentionByName() {
        val ex = assertFailsWith<KotlinRenderRefusal> { toKotlinSource(pagedFn()) }
        val msg = ex.message ?: ""
        assertTrue("PAGED_ATTENTION" in msg, "render refusal must name the kind; got: $msg")
        assertTrue("inference-only" in msg, "render refusal must state the rationale; got: $msg")
    }
}
