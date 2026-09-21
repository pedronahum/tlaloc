package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.render.KotlinRenderRefusal
import io.tlaloc.ir.render.toKotlinSource
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.466 — Phase H1b: KV_CACHE_WRITE, vLLM's decode deposit (what it calls
 * `reshape_and_cache`).
 *
 * The oracle story, strongest first:
 *
 * 1. **Hand-exact slot-mapping writes.** The pool is pre-filled with a
 *    *position-encoding* pattern — every element equals its own flat index —
 *    so a write that lands one slot, one head or one lane off produces a value
 *    that names where it actually went. The expectation is rebuilt by an
 *    independent loop over (block, offset, kvHead, lane) that consults a
 *    slot→token map, never by re-running the op's own arithmetic. Included on
 *    purpose: TWO TOKENS LANDING IN THE SAME BLOCK at different offsets, and a
 *    write into a PARTIALLY FILLED block whose other slot must survive.
 * 2. **The round-trip.** Write, then attend: a decode graph that deposits K/V
 *    at the slots the block table names and then runs PAGED_ATTENTION with
 *    `seqLens = 1`. A one-element softmax is EXACTLY 1.0, so the attention
 *    output must equal the written V bit for bit — and the slots are POISONED
 *    with 1000.0 beforehand, so a write that did not happen is off by three
 *    orders of magnitude rather than by epsilon. This composes H1a and pins
 *    that the two ops agree about what a flat slot means.
 * 3. **The conventions that are easy to get silently wrong.** Padding
 *    (negative slot ⇒ no write at all), functional semantics (the caller's
 *    array is not mutated), and the loud refusals for a duplicate live slot
 *    and an out-of-range slot.
 * 4. **The named refusals.** Inference-only by design ⇒ both AD transforms and
 *    the Kotlin renderer refuse BY NAME, naming the differentiable alternative.
 */
class KvCacheWriteTest {

    // --- The pool under test (small, but nothing degenerate). ---
    private val numBlocks = 6
    private val blockSize = 2
    private val numKvHeads = 2
    private val headDim = 3
    private val numTokens = 4
    private val numSlots = numBlocks * blockSize // 12
    private val stride = numKvHeads * headDim // 6

    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val tokensType = DxirType(F32, listOf(numTokens, numKvHeads, headDim))
    private val slotsType = DxirType(I32, listOf(numTokens))

    private fun writeFn(attrs: Map<String, Any> = emptyMap()): DxirFunction =
        DxirBuilder.function("kvWrite") {
            val cache = param("cache", cacheType)
            val newKv = param("newKv", tokensType)
            val slots = param("slots", slotsType)
            listOf(op(OpKind.KV_CACHE_WRITE, listOf(cache, newKv, slots), cacheType, attrs))
        }

    /** Position-encoded pool: element `f` holds the value `f`. */
    private fun positionEncodedPool() = FloatArray(numSlots * stride) { it.toFloat() }

    /** Token `i`, head `kv`, lane `j` holds `1000 + 100*i + 10*kv + j`. */
    private fun distinctTokens() = FloatArray(numTokens * stride) { f ->
        val i = f / stride
        val kv = (f % stride) / headDim
        val j = f % headDim
        (1000 + 100 * i + 10 * kv + j).toFloat()
    }

    private fun runWrite(cache: FloatArray, tokens: FloatArray, slots: IntArray): FloatArray =
        DxirInterpreter.evalFunction(
            writeFn(),
            listOf(cache, tokens, FloatArray(slots.size) { slots[it].toFloat() }),
        )[0]

    /**
     * The independent expectation: walk the POOL's coordinates and ask "which
     * token, if any, owns this slot?". Deliberately the inverse direction of
     * the op's own token-major walk, so a shared off-by-one cannot hide.
     */
    private fun expectedPool(cache: FloatArray, tokens: FloatArray, slots: IntArray): FloatArray {
        val owner = HashMap<Int, Int>()
        for (i in slots.indices) if (slots[i] >= 0) owner[slots[i]] = i
        val out = FloatArray(cache.size)
        for (b in 0 until numBlocks) {
            for (o in 0 until blockSize) {
                val slot = b * blockSize + o
                for (kv in 0 until numKvHeads) {
                    for (j in 0 until headDim) {
                        val dst = ((b * blockSize + o) * numKvHeads + kv) * headDim + j
                        val tok = owner[slot]
                        out[dst] = if (tok == null) {
                            cache[dst]
                        } else {
                            tokens[(tok * numKvHeads + kv) * headDim + j]
                        }
                    }
                }
            }
        }
        return out
    }

    private fun assertExact(expected: FloatArray, actual: FloatArray, what: String) {
        kotlin.test.assertEquals(expected.size, actual.size, "$what: size")
        for (i in expected.indices) {
            assertTrue(
                expected[i] == actual[i],
                "$what: element $i expected ${expected[i]} but got ${actual[i]}",
            )
        }
    }

    // --- Oracle 1: hand-exact slot-mapping writes. ----------------------

    /**
     * Slots 5 and 4 are BOTH IN BLOCK 2 (offsets 1 and 0) — the case the flat
     * convention exists to make boring, and the case a block/offset split would
     * be most likely to get wrong. Slots 0 and 11 are the two extremes of the
     * pool. Token order is deliberately not slot order.
     */
    @Test
    fun handExactSlotMappingWritesIncludingTwoTokensInOneBlock() {
        val cache = positionEncodedPool()
        val tokens = distinctTokens()
        val slots = intArrayOf(5, 4, 0, 11)
        val got = runWrite(cache, tokens, slots)
        assertExact(expectedPool(cache, tokens, slots), got, "hand-exact write")

        // Spelled out for a reader, not just computed: block 2 offset 0 is
        // token 1 and block 2 offset 1 is token 0 — the SAME block, and they
        // did not trade places.
        val blk2off0 = ((2 * blockSize + 0) * numKvHeads + 0) * headDim + 0
        val blk2off1 = ((2 * blockSize + 1) * numKvHeads + 0) * headDim + 0
        assertTrue(got[blk2off0] == 1100f, "block 2 offset 0 must hold token 1; got ${got[blk2off0]}")
        assertTrue(got[blk2off1] == 1000f, "block 2 offset 1 must hold token 0; got ${got[blk2off1]}")
    }

    /**
     * A write into a PARTIALLY FILLED block: only offset 0 of block 4 is
     * written, and offset 1 must still hold its original position-encoded
     * contents. This is the steady state of every real decode step — the last
     * page of a sequence is nearly always half empty.
     */
    @Test
    fun writeIntoAPartiallyFilledBlockLeavesTheOtherSlotUntouched() {
        val cache = positionEncodedPool()
        val tokens = distinctTokens()
        val slots = intArrayOf(8, -1, -1, -1) // block 4, offset 0
        val got = runWrite(cache, tokens, slots)
        assertExact(expectedPool(cache, tokens, slots), got, "partially filled block")

        val untouched = ((4 * blockSize + 1) * numKvHeads + 1) * headDim + 2
        assertTrue(
            got[untouched] == untouched.toFloat(),
            "block 4 offset 1 must be untouched (position-encoded ${untouched.toFloat()}); got ${got[untouched]}",
        )
    }

    /** vLLM's `-1` padding convention: an all-padding batch is the identity. */
    @Test
    fun negativeSlotsArePaddingAndWriteNothing() {
        val cache = positionEncodedPool()
        val got = runWrite(cache, distinctTokens(), intArrayOf(-1, -1, -1, -1))
        assertExact(cache, got, "all-padding write")
    }

    /**
     * FUNCTIONAL, not in-place: the caller's pool array comes back unchanged.
     * This is the house value-semantics convention, and the whole reason the op
     * returns an updated pool instead of taking an inout operand. (In
     * deployment XLA's buffer donation makes the copy vanish — an H3 follow-on
     * — but that is a runtime bargain, not an IR one.)
     */
    @Test
    fun theWriteIsFunctionalAndDoesNotMutateTheInputPool() {
        val cache = positionEncodedPool()
        val before = cache.copyOf()
        val got = runWrite(cache, distinctTokens(), intArrayOf(3, 7, -1, 1))
        assertExact(before, cache, "the input pool after the write")
        assertTrue(!got.contentEquals(before), "the write must actually have written something")
    }

    // --- Oracle 2: the round trip through PAGED_ATTENTION. --------------

    /**
     * The decode step, composed: write K and V at the slots the block table
     * names, then attend with `seqLens = 1`. One live context position means
     * the softmax is a single `exp(0)/exp(0) = 1.0`, so the attention output is
     * the written V EXACTLY — no tolerance, no accumulated error.
     *
     * The target slots are POISONED at 1000.0 in the incoming pool, so a write
     * that silently did not land shows up as 1000, not as a rounding wobble.
     * GQA is live here (2 query heads, 1 kv head), so both heads must read the
     * same written row.
     */
    @Test
    fun writeThenPagedAttentionReadsBackExactlyWhatWasWritten() {
        val seqs = 2
        val heads = 2
        val kvHeads = 1
        val dim = 3
        val blocks = 6
        val bs = 2
        val maxBlocks = 3
        val tokens = seqs // one decode token per sequence

        val poolType = DxirType(F32, listOf(blocks, bs, kvHeads, dim))
        val tokType = DxirType(F32, listOf(tokens, kvHeads, dim))
        val slotType = DxirType(I32, listOf(tokens))
        val qType = DxirType(F32, listOf(seqs, heads, dim))
        val tableType = DxirType(I32, listOf(seqs, maxBlocks))
        val lensType = DxirType(I32, listOf(seqs))

        val fn = DxirBuilder.function("decodeStep") {
            val kc = param("kc", poolType)
            val vc = param("vc", poolType)
            val newK = param("newK", tokType)
            val newV = param("newV", tokType)
            val slots = param("slots", slotType)
            val q = param("q", qType)
            val table = param("table", tableType)
            val lens = param("lens", lensType)
            val kc2 = op(OpKind.KV_CACHE_WRITE, listOf(kc, newK, slots), poolType)
            val vc2 = op(OpKind.KV_CACHE_WRITE, listOf(vc, newV, slots), poolType)
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, kc2, vc2, table, lens), qType, mapOf("scale" to 1.0)))
        }

        // Sequence 0 owns page 3, sequence 1 owns page 1 — an allocator's
        // interleaving, not an identity table. Context position 0 of sequence s
        // lives at offset 0 of its first page, so flat slot = page * blockSize.
        val table = intArrayOf(3, 0, 0, 1, 0, 0)
        val lens = intArrayOf(1, 1)
        val slotMapping = intArrayOf(3 * bs, 1 * bs) // 6 and 2

        val poolSize = blocks * bs * kvHeads * dim
        val kPool = FloatArray(poolSize) { 1000f } // poison EVERY slot
        val vPool = FloatArray(poolSize) { 1000f }
        val newK = FloatArray(tokens * kvHeads * dim) { 0.25f * (it + 1) }
        val newV = floatArrayOf(7f, -3f, 0.5f, -11f, 2f, 6.25f)
        val q = FloatArray(seqs * heads * dim) { 0.125f * (it + 1) }

        val got = DxirInterpreter.evalFunction(
            fn,
            listOf(
                kPool, vPool, newK, newV,
                FloatArray(slotMapping.size) { slotMapping[it].toFloat() },
                q,
                FloatArray(table.size) { table[it].toFloat() },
                FloatArray(lens.size) { lens[it].toFloat() },
            ),
        )[0]

        for (s in 0 until seqs) {
            for (h in 0 until heads) {
                for (j in 0 until dim) {
                    val want = newV[s * dim + j] // kvHeads == 1, so head h reads kv head 0
                    val actual = got[(s * heads + h) * dim + j]
                    assertTrue(
                        want == actual,
                        "round trip seq $s head $h lane $j: attention must read back the written " +
                            "V exactly — wanted $want, got $actual",
                    )
                }
            }
        }
    }

    // --- Oracle 3: the loud refusals about slots. -----------------------

    @Test
    fun aRepeatedLiveSlotRefusesByName() {
        val ex = assertFailsWith<IllegalArgumentException> {
            runWrite(positionEncodedPool(), distinctTokens(), intArrayOf(4, 9, 4, -1))
        }
        val msg = ex.message ?: ""
        assertTrue("KV_CACHE_WRITE" in msg, "must name the kind; got $msg")
        assertTrue("DISTINCT" in msg, "must state the distinctness requirement; got $msg")
    }

    @Test
    fun aSlotPastThePoolRefusesByNameAndIsNotTreatedAsPadding() {
        val ex = assertFailsWith<IllegalArgumentException> {
            runWrite(positionEncodedPool(), distinctTokens(), intArrayOf(0, numSlots, -1, -1))
        }
        val msg = ex.message ?: ""
        assertTrue("slot $numSlots" in msg, "must name the offending slot; got $msg")
        assertTrue("NEGATIVE" in msg, "must distinguish padding from an allocator bug; got $msg")
    }

    @Test
    fun aSlotMappingOfTheWrongLengthRefusesByName() {
        val fn = DxirBuilder.function("kvWriteBad") {
            val cache = param("cache", cacheType)
            val newKv = param("newKv", tokensType)
            val slots = param("slots", DxirType(I32, listOf(numTokens + 1)))
            listOf(op(OpKind.KV_CACHE_WRITE, listOf(cache, newKv, slots), cacheType))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(positionEncodedPool(), distinctTokens(), FloatArray(numTokens + 1)),
            )
        }
        assertTrue("one slot per token" in (ex.message ?: ""), "got ${ex.message}")
    }

    @Test
    fun aFloatSlotMappingRefusesByName() {
        val fn = DxirBuilder.function("kvWriteFloatSlots") {
            val cache = param("cache", cacheType)
            val newKv = param("newKv", tokensType)
            val slots = param("slots", DxirType(F32, listOf(numTokens)))
            listOf(op(OpKind.KV_CACHE_WRITE, listOf(cache, newKv, slots), cacheType))
        }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(positionEncodedPool(), distinctTokens(), FloatArray(numTokens)),
            )
        }
        assertTrue("allocator index" in (ex.message ?: ""), "got ${ex.message}")
    }

    @Test
    fun dimDerivedAttrsAreRefusedByName() {
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                writeFn(mapOf("blockSize" to blockSize)),
                listOf(positionEncodedPool(), distinctTokens(), FloatArray(numTokens)),
            )
        }
        assertTrue("sentinel" in (ex.message ?: ""), "must cite the sentinel-dims rule; got ${ex.message}")
    }

    // --- Oracle 4: inference-only ⇒ named refusals. ---------------------

    /** A SCALAR return, so the refusal fires on the KIND and not on the
     *  reverse transform's scalar-return precondition. */
    private fun writeLossFn(): DxirFunction = DxirBuilder.function("kvWriteLoss") {
        val cache = param("cache", cacheType)
        val newKv = param("newKv", tokensType)
        val slots = param("slots", slotsType)
        val y = op(OpKind.KV_CACHE_WRITE, listOf(cache, newKv, slots), cacheType)
        listOf(op(OpKind.SUM, listOf(y), DxirType(F32, emptyList())))
    }

    private fun assertInferenceOnlyRefusal(ex: Throwable) {
        val msg = ex.message ?: ""
        assertTrue("KV_CACHE_WRITE" in msg, "refusal must name the kind; got: $msg")
        assertTrue("INFERENCE-ONLY" in msg || "inference-only" in msg, "refusal must state the rationale; got: $msg")
        assertTrue("SCATTER" in msg, "refusal must name the differentiable alternative; got: $msg")
    }

    @Test
    fun reverseTransformRefusesKvCacheWriteByName() {
        assertInferenceOnlyRefusal(assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(writeLossFn()) })
    }

    @Test
    fun forwardTransformRefusesKvCacheWriteByName() {
        assertInferenceOnlyRefusal(assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(writeLossFn()) })
    }

    @Test
    fun kotlinRendererRefusesKvCacheWriteByName() {
        val ex = assertFailsWith<KotlinRenderRefusal> { toKotlinSource(writeFn()) }
        val msg = ex.message ?: ""
        assertTrue("KV_CACHE_WRITE" in msg, "render refusal must name the kind; got: $msg")
        assertTrue("inference-only" in msg, "render refusal must state the rationale; got: $msg")
    }
}
