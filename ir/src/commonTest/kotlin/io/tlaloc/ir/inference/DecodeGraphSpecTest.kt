package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.467 — Phase H1c: the decode-graph shape contract.
 *
 * What this pins: the signature a plugin calls (order, names, static shapes),
 * that PREFILL is the SAME contract with a longer token axis, the
 * executable-cache key's discrimination, and that `verifySignature` actually
 * catches a graph that does not match.
 */
class DecodeGraphSpecTest {

    private val model = DecodeModelShape(
        vocabSize = 32_000,
        hiddenSize = 512,
        numHeads = 8,
        numKvHeads = 2,
        headDim = 64,
        numLayers = 3,
        numBlocks = 1024,
        blockSize = 16,
    )

    private val spec = DecodeGraphSpec(model, DecodeBucket(batch = 4, maxContext = 128))

    // --- The signature. ---------------------------------------------------

    @Test
    fun theDecodeSignatureIsFiveSchedulerTensorsThenTwoPoolsPerLayer() {
        assertEquals(1, spec.tokensPerSeq)
        assertEquals(8, spec.maxBlocksPerSeq) // 128 / 16, exact by construction
        assertEquals(4, spec.totalTokens)

        assertEquals(
            listOf("tokenIds", "positions", "blockTables", "seqLens", "slotMapping") +
                (0 until 3).flatMap { listOf("keyCache$it", "valueCache$it") },
            spec.inputs.map { it.name },
        )
        assertEquals(DxirType(I32, listOf(4, 1)), spec.tokenIdsType)
        assertEquals(DxirType(I32, listOf(4, 8)), spec.blockTablesType)
        assertEquals(DxirType(I32, listOf(4)), spec.seqLensType)
        assertEquals(DxirType(I32, listOf(4)), spec.slotMappingType)
        assertEquals(DxirType(F32, listOf(1024, 16, 2, 64)), spec.poolType)
        assertEquals(DxirType(F32, listOf(4, 1, 32_000)), spec.logitsType)

        assertEquals(
            listOf("logits") + (0 until 3).flatMap { listOf("keyCache${it}Out", "valueCache${it}Out") },
            spec.outputs.map { it.name },
        )
        // Every KV pool input has a matching output at the donation offset.
        assertEquals(6, spec.donationPairs.size)
        for ((inIdx, outIdx) in spec.donationPairs) {
            assertEquals(DecodeSlotRole.KV_POOL_IN, spec.inputs[inIdx].role)
            assertEquals(DecodeSlotRole.KV_POOL_OUT, spec.outputs[outIdx].role)
            assertEquals(spec.inputs[inIdx].type, spec.outputs[outIdx].type)
        }
    }

    /**
     * PREFILL is the SAME contract with `T = the bucket's context width`.
     * Everything that is not the token axis is untouched — which is the whole
     * reason it is one contract and not two.
     */
    @Test
    fun prefillIsTheSameContractWithALongerTokenAxis() {
        val pre = DecodeGraphSpec(model, DecodeBucket(4, 128), DecodeGraphKind.PREFILL)
        assertEquals(128, pre.tokensPerSeq)
        assertEquals(DxirType(I32, listOf(4, 128)), pre.tokenIdsType)
        assertEquals(DxirType(I32, listOf(4 * 128)), pre.slotMappingType)
        assertEquals(DxirType(F32, listOf(4, 128, 32_000)), pre.logitsType)

        // The bucket-shaped, model-shaped and pool-shaped parts do not move.
        assertEquals(spec.blockTablesType, pre.blockTablesType)
        assertEquals(spec.seqLensType, pre.seqLensType)
        assertEquals(spec.poolType, pre.poolType)
        assertEquals(spec.inputs.map { it.role }, pre.inputs.map { it.role })
        assertEquals(spec.outputs.map { it.role }, pre.outputs.map { it.role })
    }

    // --- The executable-cache key. ---------------------------------------

    @Test
    fun theCacheKeyDiscriminatesEveryAxisThatChangesTheProgram() {
        val base = spec.executableCacheKey("sha256:abc")
        val keys = listOf(
            base,
            spec.executableCacheKey("sha256:def"), // different weights
            DecodeGraphSpec(model, DecodeBucket(8, 128)).executableCacheKey("sha256:abc"),
            DecodeGraphSpec(model, DecodeBucket(4, 256)).executableCacheKey("sha256:abc"),
            DecodeGraphSpec(model, DecodeBucket(4, 128), DecodeGraphKind.PREFILL)
                .executableCacheKey("sha256:abc"),
            DecodeGraphSpec(model.copy(dtype = BF16), DecodeBucket(4, 128)).executableCacheKey("sha256:abc"),
            DecodeGraphSpec(model.copy(kvDtype = BF16), DecodeBucket(4, 128)).executableCacheKey("sha256:abc"),
        )
        assertEquals(keys.size, keys.distinct().size, "cache keys collided: $keys")

        // And it is STABLE: an independently constructed identical spec hits
        // the same slot. A plugin that rebuilds its spec per request must not
        // miss the cache every time.
        assertEquals(base, DecodeGraphSpec(model.copy(), DecodeBucket(4, 128)).executableCacheKey("sha256:abc"))
        assertTrue(base.startsWith("tlaloc-decode-v1/"), "key was: $base")
    }

    /** The model hash is load-bearing, so a blank one refuses by name rather
     *  than quietly sharing an executable across two sets of weights. */
    @Test
    fun aBlankModelHashIsRefusedByName() {
        val e = assertFailsWith<IllegalArgumentException> { spec.executableCacheKey("  ") }
        assertTrue("different weights must not share" in (e.message ?: ""), "message: ${e.message}")
    }

    // --- verifySignature actually verifies. ------------------------------

    @Test
    fun verifySignatureAcceptsAConformingGraphAndNamesTheSlotOnAMismatch() {
        val small = DecodeModelShape(
            vocabSize = 5, hiddenSize = 6, numHeads = 2, numKvHeads = 1, headDim = 3,
            numLayers = 1, numBlocks = 8, blockSize = 2,
        )
        val s = DecodeGraphSpec(small, DecodeBucket(batch = 2, maxContext = 4))

        val good = DxirBuilder.function("decode") {
            val ps = s.inputs.map { param(it.name, it.type) }
            val logits = const(FloatArray(s.logitsType.elementCount.toInt()), s.logitsType)
            listOf(logits, ps[5], ps[6])
        }
        s.verifySignature(good, "test") // does not throw

        // One param too few.
        val short = DxirBuilder.function("decode") {
            val ps = s.inputs.dropLast(1).map { param(it.name, it.type) }
            listOf(const(FloatArray(s.logitsType.elementCount.toInt()), s.logitsType), ps[5], ps[5])
        }
        val e1 = assertFailsWith<IllegalArgumentException> { s.verifySignature(short, "test") }
        assertTrue("declares 7" in (e1.message ?: ""), "message: ${e1.message}")

        // Right arity, wrong bucket on the block table.
        val wrong = DxirBuilder.function("decode") {
            val ps = s.inputs.mapIndexed { i, slot ->
                param(slot.name, if (i == 2) DxirType(I32, listOf(2, 99)) else slot.type)
            }
            listOf(const(FloatArray(s.logitsType.elementCount.toInt()), s.logitsType), ps[5], ps[6])
        }
        val e2 = assertFailsWith<IllegalArgumentException> { s.verifySignature(wrong, "test") }
        assertTrue("'blockTables'" in (e2.message ?: ""), "message: ${e2.message}")
    }

    /** A model shape that PAGED_ATTENTION could not derive a GQA grouping
     *  from is refused at the contract, not three layers later. */
    @Test
    fun aNonDivisibleGqaGroupingIsRefusedAtTheContract() {
        val e = assertFailsWith<IllegalArgumentException> {
            DecodeModelShape(
                vocabSize = 8, hiddenSize = 6, numHeads = 3, numKvHeads = 2, headDim = 2,
                numLayers = 1, numBlocks = 4, blockSize = 2,
            )
        }
        assertTrue("multiple of numKvHeads" in (e.message ?: ""), "message: ${e.message}")
    }

    /** The padding convention's one genuinely load-bearing constant. */
    @Test
    fun thePaddingSeqLenIsOneAndNeverZero() {
        assertEquals(1, DecodePadding.PADDING_SEQ_LEN)
        assertTrue(
            DecodePadding.PADDING_SEQ_LEN != 0,
            "seqLen 0 masks every context lane to -Inf; the softmax is then 0/0 = NaN in the " +
                "gather-composed emission while the interpreter takes a len==0 early-out and " +
                "leaves zeros. A padding convention must not route through the one shape the " +
                "two disagree about.",
        )
        assertEquals(-1, DecodePadding.PADDING_SLOT)
    }

    @Test
    fun padToRefusesARequestLargerThanItsBucket() {
        assertTrue(
            DecodePadding.padTo(intArrayOf(7, 8), 4, DecodePadding.PADDING_SLOT, "slotMapping")
                .contentEquals(intArrayOf(7, 8, -1, -1)),
        )
        val e = assertFailsWith<IllegalArgumentException> {
            DecodePadding.padTo(intArrayOf(1, 2, 3), 2, 0, "seqLens")
        }
        assertTrue("must be chosen to COVER" in (e.message ?: ""), "message: ${e.message}")
    }

    /** Guard against a silent re-ordering of the signature: the pool base
     *  index is a constant other layers bind against. */
    @Test
    fun thePoolInputBaseIsWhereTheContractSaysItIs() {
        assertEquals(5, DecodeGraphSpec.KV_POOL_INPUT_BASE)
        assertEquals(
            DecodeSlotRole.KV_POOL_IN,
            spec.inputs[DecodeGraphSpec.KV_POOL_INPUT_BASE].role,
        )
        assertTrue(spec.inputs.take(5).none { it.role == DecodeSlotRole.KV_POOL_IN })
    }
}
