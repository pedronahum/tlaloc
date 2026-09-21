package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.467 — Phase H1c: the END-TO-END MINI DECODE GRAPH, and the padding
 * invariant it exists to pin.
 *
 * The graph is H1a and H1b composed into an actual decode step that conforms
 * to [DecodeGraphSpec]:
 *
 * ```
 *   tokenIds → EMBEDDING → h
 *   h → q/k/v projections
 *   KV_CACHE_WRITE(keyCache, newK, slotMapping)     ← H1b
 *   KV_CACHE_WRITE(valueCache, newV, slotMapping)
 *   PAGED_ATTENTION(q, kc', vc', blockTables, seqLens)  ← H1a
 *   → lm head → logits
 * ```
 *
 * The weights are **constants**, not parameters, which is the deployment
 * shape: the `:maestro` manifest is content-addressed over weights+graph, and
 * that is precisely why the executable-cache key carries a model hash.
 *
 * Three oracles, strongest first:
 *
 * 1. **The padding invariant.** A batch of 3 run in a bucket-4 graph gives
 *    BIT-IDENTICAL logits for rows 0–2, and a bit-identical updated pool,
 *    against the same batch run in a bucket-3 graph. Pinned with `==`, not a
 *    tolerance — see [DecodePadding] for why that is the right claim in the
 *    reference interpreter and why the device claim is a tolerance instead.
 * 2. **A hand-checkable value.** With `seqLens = 1` everywhere the softmax is
 *    a single `exp(0)/exp(0) = 1.0`, so attention returns the just-written V
 *    exactly, and the whole step collapses to a chain of small matmuls an
 *    independent walk reproduces without touching the op kinds under test.
 * 3. **The padding row is inert.** Its slot is `-1`, so the pool a
 *    bucket-4 step produces has exactly the three real writes in it and page
 *    0 — the scratch page its block table names — is untouched.
 */
class DecodeGraphPaddingInvariantTest {

    // A small model, but nothing degenerate: GQA is live (2 query heads over
    // 1 kv head) and the pool has more pages than the step uses.
    private val model = DecodeModelShape(
        vocabSize = 5,
        hiddenSize = 6,
        numHeads = 2,
        numKvHeads = 1,
        headDim = 3,
        numLayers = 1,
        numBlocks = 8,
        blockSize = 2,
    )
    private val scale = 1.0 / sqrt(model.headDim.toDouble())

    // --- Weights: deterministic, distinct, and not symmetric. ------------

    private fun weight(rows: Int, cols: Int, seed: Int) = FloatArray(rows * cols) { i ->
        // A small spread of non-round values; no two weights alike.
        (((i * 37 + seed * 11) % 23) - 11).toFloat() / 16f
    }

    private val embedTable = weight(model.vocabSize, model.hiddenSize, seed = 1)
    private val qProj = weight(model.hiddenSize, model.numHeads * model.headDim, seed = 2)
    private val kProj = weight(model.hiddenSize, model.numKvHeads * model.headDim, seed = 3)
    private val vProj = weight(model.hiddenSize, model.numKvHeads * model.headDim, seed = 4)
    private val lmHead = weight(model.numHeads * model.headDim, model.vocabSize, seed = 5)

    /**
     * Build the decode graph for [spec]. Conforms to the contract exactly —
     * [DecodeGraphSpec.verifySignature] is asserted on every graph this test
     * builds, so the contract is not a document that drifts from the code.
     *
     * `positions` is a declared input that this mini graph does not consume:
     * RoPE is H2's business (it needs the real rotary tables), and the
     * contract carries the slot now so the signature does not change shape
     * when H2 fills it in.
     */
    private fun decodeGraph(spec: DecodeGraphSpec): DxirFunction {
        val b = spec.bucket.batch
        val d = model.hiddenSize
        val hd = model.headDim
        val kvWidth = model.numKvHeads * hd
        val qWidth = model.numHeads * hd

        val fn = DxirBuilder.function("decode_${spec.bucket}") {
            val tokenIds = param("tokenIds", spec.tokenIdsType)
            @Suppress("UNUSED_VARIABLE")
            val positions = param("positions", spec.positionsType)
            val blockTables = param("blockTables", spec.blockTablesType)
            val seqLens = param("seqLens", spec.seqLensType)
            val slotMapping = param("slotMapping", spec.slotMappingType)
            val keyCache = param("keyCache0", spec.poolType)
            val valueCache = param("valueCache0", spec.poolType)

            val tbl = const(embedTable, DxirType(F32, listOf(model.vocabSize, d)))
            val emb = op(OpKind.EMBEDDING, listOf(tbl, tokenIds), DxirType(F32, listOf(b, 1, d)))
            val h = op(OpKind.RESHAPE, listOf(emb), DxirType(F32, listOf(b, d)))

            fun project(w: FloatArray, cols: Int) = op(
                OpKind.MATMUL,
                listOf(h, const(w, DxirType(F32, listOf(d, cols)))),
                DxirType(F32, listOf(b, cols)),
            )

            val q = op(
                OpKind.RESHAPE, listOf(project(qProj, qWidth)),
                DxirType(F32, listOf(b, model.numHeads, hd)),
            )
            val newK = op(
                OpKind.RESHAPE, listOf(project(kProj, kvWidth)),
                DxirType(F32, listOf(b, model.numKvHeads, hd)),
            )
            val newV = op(
                OpKind.RESHAPE, listOf(project(vProj, kvWidth)),
                DxirType(F32, listOf(b, model.numKvHeads, hd)),
            )

            val kc = op(OpKind.KV_CACHE_WRITE, listOf(keyCache, newK, slotMapping), spec.poolType)
            val vc = op(OpKind.KV_CACHE_WRITE, listOf(valueCache, newV, slotMapping), spec.poolType)

            val att = op(
                OpKind.PAGED_ATTENTION,
                listOf(q, kc, vc, blockTables, seqLens),
                DxirType(F32, listOf(b, model.numHeads, hd)),
                mapOf("scale" to scale),
            )
            val attFlat = op(OpKind.RESHAPE, listOf(att), DxirType(F32, listOf(b, qWidth)))
            val logits2 = op(
                OpKind.MATMUL,
                listOf(attFlat, const(lmHead, DxirType(F32, listOf(qWidth, model.vocabSize)))),
                DxirType(F32, listOf(b, model.vocabSize)),
            )
            val logits = op(OpKind.RESHAPE, listOf(logits2), spec.logitsType)
            listOf(logits, kc, vc)
        }
        spec.verifySignature(fn, "DecodeGraphPaddingInvariantTest")
        return fn
    }

    /** One decode step's results: logits `[B, 1, V]` and the two pools. */
    private class Step(val logits: FloatArray, val keyPool: FloatArray, val valuePool: FloatArray)

    private fun runStep(
        spec: DecodeGraphSpec,
        tokenIds: IntArray,
        blockTables: IntArray,
        seqLens: IntArray,
        slotMapping: IntArray,
        keyPool: FloatArray,
        valuePool: FloatArray,
    ): Step {
        fun ints(a: IntArray) = FloatArray(a.size) { a[it].toFloat() }
        val out = DxirInterpreter.evalFunction(
            decodeGraph(spec),
            listOf(
                ints(tokenIds),
                ints(IntArray(spec.totalTokens)), // positions — unconsumed here
                ints(blockTables),
                ints(seqLens),
                ints(slotMapping),
                keyPool,
                valuePool,
            ),
        )
        return Step(out[0], out[1], out[2])
    }

    // --- The scenario: three real sequences, allocated out of order. -----

    private val realTokenIds = intArrayOf(3, 1, 4)

    /** Sequence s owns page `pages[s]`; its single decode token goes to
     *  offset 0 of that page, so the flat slot is `page * blockSize`. Page 0
     *  is deliberately left out — it is the scratch page a padded row's block
     *  table names. */
    private val pages = intArrayOf(2, 5, 3)
    private val realSlots = IntArray(3) { pages[it] * model.blockSize }
    private val realSeqLens = intArrayOf(1, 1, 1)

    /** A poisoned pool: every slot holds 1000 + its own flat index, so a write
     *  that did not land, or landed in the wrong slot, names where it went. */
    private fun poisonedPool(seed: Int) = FloatArray(
        model.numBlocks * model.blockSize * model.numKvHeads * model.headDim,
    ) { 1000f + seed * 10_000f + it }

    private fun blockTablesFor(batch: Int, maxBlocks: Int): IntArray {
        val t = IntArray(batch * maxBlocks) { DecodePadding.PADDING_BLOCK }
        for (s in 0 until minOf(batch, pages.size)) t[s * maxBlocks] = pages[s]
        return t
    }

    // --- Oracle 1: the padding invariant. --------------------------------

    /**
     * The claim the whole bucketing story rests on: padding a batch out to its
     * bucket costs the real rows NOTHING — not "almost nothing", not "within
     * tolerance" — in the reference interpreter. Rows 0–2 of the bucket-4 run
     * and both updated pools are compared with `==`.
     */
    @Test
    fun aBatchOfThreeInABucketOfFourIsBitIdenticalForTheRealRows() {
        val tight = DecodeGraphSpec(model, DecodeBucket(batch = 3, maxContext = 4))
        val padded = DecodeGraphSpec(model, DecodeBucket(batch = 4, maxContext = 4))
        assertEquals(tight.maxBlocksPerSeq, padded.maxBlocksPerSeq, "same context bucket, same table width")
        val mb = tight.maxBlocksPerSeq

        val tightRun = runStep(
            tight, realTokenIds, blockTablesFor(3, mb), realSeqLens, realSlots,
            poisonedPool(0), poisonedPool(1),
        )
        val paddedRun = runStep(
            padded,
            DecodePadding.padTo(realTokenIds, 4, DecodePadding.PADDING_TOKEN_ID, "tokenIds"),
            blockTablesFor(4, mb),
            DecodePadding.padTo(realSeqLens, 4, DecodePadding.PADDING_SEQ_LEN, "seqLens"),
            DecodePadding.padTo(realSlots, 4, DecodePadding.PADDING_SLOT, "slotMapping"),
            poisonedPool(0), poisonedPool(1),
        )

        val v = model.vocabSize
        for (row in 0 until 3) {
            for (j in 0 until v) {
                val a = tightRun.logits[row * v + j]
                val b = paddedRun.logits[row * v + j]
                assertTrue(
                    a == b,
                    "row $row logit $j: bucket-3 gave $a, bucket-4 gave $b — the padding " +
                        "invariant says every op in the decode contract is row-independent " +
                        "along the batch axis, so these must be BIT-identical",
                )
            }
        }
        assertExact(tightRun.keyPool, paddedRun.keyPool, "key pool after a padded step")
        assertExact(tightRun.valuePool, paddedRun.valuePool, "value pool after a padded step")

        // And the logits are not trivially all-equal garbage.
        assertTrue(
            tightRun.logits.distinct().size > 3,
            "the mini model produced a degenerate logit vector; the oracle would pass on anything",
        )
    }

    /**
     * The same invariant one bucket further out, and with the *context* bucket
     * widened too: bucket (4, 8) against bucket (3, 4). A wider context bucket
     * is a wider block table whose extra entries are all the scratch page, and
     * the seqLens mask must make them contribute exactly nothing.
     */
    @Test
    fun awiderContextBucketAlsoCostsTheRealRowsNothing() {
        val tight = DecodeGraphSpec(model, DecodeBucket(batch = 3, maxContext = 4))
        val wide = DecodeGraphSpec(model, DecodeBucket(batch = 4, maxContext = 8))

        val tightRun = runStep(
            tight, realTokenIds, blockTablesFor(3, tight.maxBlocksPerSeq), realSeqLens, realSlots,
            poisonedPool(0), poisonedPool(1),
        )
        val wideRun = runStep(
            wide,
            DecodePadding.padTo(realTokenIds, 4, DecodePadding.PADDING_TOKEN_ID, "tokenIds"),
            blockTablesFor(4, wide.maxBlocksPerSeq),
            DecodePadding.padTo(realSeqLens, 4, DecodePadding.PADDING_SEQ_LEN, "seqLens"),
            DecodePadding.padTo(realSlots, 4, DecodePadding.PADDING_SLOT, "slotMapping"),
            poisonedPool(0), poisonedPool(1),
        )

        val v = model.vocabSize
        for (row in 0 until 3) {
            for (j in 0 until v) {
                assertTrue(
                    tightRun.logits[row * v + j] == wideRun.logits[row * v + j],
                    "row $row logit $j diverged when the CONTEXT bucket widened: " +
                        "${tightRun.logits[row * v + j]} vs ${wideRun.logits[row * v + j]}",
                )
            }
        }
        assertExact(tightRun.keyPool, wideRun.keyPool, "key pool across context buckets")
    }

    // --- Oracle 2: a hand-checkable end-to-end value. --------------------

    /**
     * With one live context position per sequence, the softmax is a single
     * `exp(0)/exp(0) = 1.0`, so attention returns the V just written — exactly.
     * The step then collapses to `embed → project → duplicate across the GQA
     * group → lm head`, which this walk reproduces with plain arithmetic and
     * none of the op kinds under test.
     *
     * The pool is poisoned at ±10000, so an attention that read an unwritten
     * slot lands three orders of magnitude away rather than within epsilon.
     */
    @Test
    fun theWholeStepAgreesWithAnIndependentHandWalk() {
        val spec = DecodeGraphSpec(model, DecodeBucket(batch = 3, maxContext = 4))
        val run = runStep(
            spec, realTokenIds, blockTablesFor(3, spec.maxBlocksPerSeq), realSeqLens, realSlots,
            poisonedPool(0), poisonedPool(1),
        )

        val d = model.hiddenSize
        val hd = model.headDim
        val v = model.vocabSize
        for (s in realTokenIds.indices) {
            // 1. embed
            val h = FloatArray(d) { j -> embedTable[realTokenIds[s] * d + j] }
            // 2. the V projection is what attention hands back, because
            //    seqLens = 1 makes the softmax exactly 1.0.
            val vRow = FloatArray(hd) { c ->
                var acc = 0.0
                for (j in 0 until d) acc += h[j].toDouble() * vProj[j * hd + c]
                acc.toFloat()
            }
            // 3. both query heads share the single kv head, so the attention
            //    output is vRow repeated across the GQA group.
            val att = FloatArray(model.numHeads * hd) { i -> vRow[i % hd] }
            // 4. lm head
            for (c in 0 until v) {
                var acc = 0.0
                for (j in att.indices) acc += att[j].toDouble() * lmHead[j * v + c]
                val got = run.logits[s * v + c]
                assertTrue(
                    kotlin.math.abs(acc - got) <= 1e-4,
                    "sequence $s logit $c: hand walk says $acc, the graph says $got",
                )
            }
        }

        // Step 2 above claimed the pool holds the V projection. Pin it
        // directly, so a failure says which half broke: the write, or the
        // read-back through attention.
        for (s in realTokenIds.indices) {
            val h = FloatArray(d) { j -> embedTable[realTokenIds[s] * d + j] }
            val base = realSlots[s] * model.numKvHeads * hd
            for (c in 0 until hd) {
                var acc = 0.0
                for (j in 0 until d) acc += h[j].toDouble() * vProj[j * hd + c]
                assertTrue(
                    kotlin.math.abs(acc - run.valuePool[base + c]) <= 1e-4,
                    "sequence $s slot ${realSlots[s]} lane $c: expected the V projection $acc, " +
                        "pool holds ${run.valuePool[base + c]} (1000-ish means the write never landed)",
                )
            }
        }
    }

    // --- Oracle 3: the padding row is inert. -----------------------------

    /**
     * The padded row writes NOTHING (slot `-1`, H1b's convention) and reads
     * only the scratch page its block table names. So the pool coming out of a
     * bucket-4 step differs from the pool going in at exactly the three real
     * slots, and page 0 is untouched.
     */
    @Test
    fun thePaddingRowWritesNothingAndLeavesTheScratchPageAlone() {
        val spec = DecodeGraphSpec(model, DecodeBucket(batch = 4, maxContext = 4))
        val kIn = poisonedPool(0)
        val run = runStep(
            spec,
            DecodePadding.padTo(realTokenIds, 4, DecodePadding.PADDING_TOKEN_ID, "tokenIds"),
            blockTablesFor(4, spec.maxBlocksPerSeq),
            DecodePadding.padTo(realSeqLens, 4, DecodePadding.PADDING_SEQ_LEN, "seqLens"),
            DecodePadding.padTo(realSlots, 4, DecodePadding.PADDING_SLOT, "slotMapping"),
            kIn, poisonedPool(1),
        )

        val stride = model.numKvHeads * model.headDim
        val changed = (0 until model.numBlocks * model.blockSize).filter { slot ->
            (0 until stride).any { run.keyPool[slot * stride + it] != kIn[slot * stride + it] }
        }
        assertEquals(realSlots.toList().sorted(), changed, "exactly the three real slots changed")

        // Page 0, the scratch page every padded block-table row names, is
        // still poison — the padded row read it and wrote nothing.
        for (off in 0 until model.blockSize * stride) {
            assertTrue(run.keyPool[off] == kIn[off], "scratch page 0 element $off was written")
        }

        // The step is still FUNCTIONAL: the caller's array is untouched.
        assertExact(poisonedPool(0), kIn, "the input pool after a decode step")
    }

    private fun assertExact(expected: FloatArray, actual: FloatArray, what: String) {
        assertEquals(expected.size, actual.size, "$what: size")
        for (i in expected.indices) {
            assertTrue(
                expected[i] == actual[i],
                "$what: element $i expected ${expected[i]} but got ${actual[i]}",
            )
        }
    }
}
