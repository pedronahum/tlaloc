package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.465 — Phase H1a: the OFFLINE half of PAGED_ATTENTION's emission
 * certification. The load-bearing oracle is
 * `PjrtPagedAttentionSmokeTest` — it compiles this text under real XLA on the
 * GB10 and pins it against the interpreter's reference walk — but that test
 * self-skips without CUDA, so the structure a reader would look for is pinned
 * here where it always runs: the two page gathers, the seqLens mask, the two
 * dot_generals with their batching dimension numbers, and the GQA reshapes.
 *
 * These are STRUCTURE pins, not golden text: they name the shape of the
 * lowering (gather-composed, masked, batched over sequence AND kv head), which
 * is exactly what Phase H4's fused kernel will replace wholesale.
 */
class PagedAttentionEmitTest {

    private val numSeqs = 2
    private val numHeads = 4
    private val numKvHeads = 2
    private val headDim = 3
    private val blockSize = 2
    private val numBlocks = 6
    private val maxBlocksPerSeq = 3

    private val qType = DxirType(F32, listOf(numSeqs, numHeads, headDim))
    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val tableType = DxirType(I32, listOf(numSeqs, maxBlocksPerSeq))
    private val lensType = DxirType(I32, listOf(numSeqs))

    private fun pagedFn(attrs: Map<String, Any> = mapOf("scale" to 0.5)): DxirFunction =
        DxirBuilder.function("paged") {
            val q = param("q", qType)
            val k = param("k", cacheType)
            val v = param("v", cacheType)
            val t = param("t", tableType)
            val l = param("l", lensType)
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, attrs))
        }

    private fun mlir(): String = pagedFn().toStablehlo()

    @Test
    fun emitsTwoPageGathersWithTheBlockTableAsIndices() {
        val text = mlir()
        val gathers = text.lines().count { "stablehlo.gather" in it }
        kotlin.test.assertEquals(2, gathers, "one page gather per pool (K and V):\n$text")
        // slice_sizes = [1, blockSize, numKvHeads, headDim]: one whole page per
        // table entry, with the page axis collapsed and index_vector_dim == the
        // indices' rank (each table entry is a scalar page id).
        assertTrue(
            "slice_sizes = array<i64: 1, $blockSize, $numKvHeads, $headDim>" in text,
            "gather must slice exactly one page:\n$text",
        )
        assertTrue("collapsed_slice_dims = [0]" in text, "the page axis must collapse:\n$text")
        assertTrue("index_vector_dim = 2" in text, "table entries are scalar page ids:\n$text")
    }

    @Test
    fun emitsTheSeqLensMaskRatherThanASequenceSlice() {
        val text = mlir()
        // iota over the context axis, compared against the broadcast seqLens.
        assertTrue("stablehlo.iota dim = 3" in text, "context-axis iota missing:\n$text")
        assertTrue("stablehlo.compare LT" in text, "the live-position test is `t < seqLens[s]`:\n$text")
        assertTrue("stablehlo.select" in text, "masked lanes must be selected to -Inf:\n$text")
        assertTrue("0xFF800000" in text, "the mask fill must be -Inf (f32 bit pattern):\n$text")
        // Static shapes only: a per-sequence slice would be data-dependent.
        assertTrue(
            "real_dynamic_slice" !in text && "dynamic_slice" !in text,
            "emission must stay statically shaped (mask, don't slice):\n$text",
        )
    }

    @Test
    fun emitsBatchedDotGeneralsOverSequenceAndKvHead() {
        val text = mlir()
        val batched = text.lines().count { "batching_dims = [0, 1] x [0, 2]" in it }
        kotlin.test.assertEquals(
            2, batched,
            "both QK^T and probs·V batch over (sequence, kv head), with the kv axis at " +
                "position 2 in the gathered window:\n$text",
        )
        val ctx = maxBlocksPerSeq * blockSize
        val group = numHeads / numKvHeads
        assertTrue(
            "tensor<${numSeqs}x${numKvHeads}x${group}x${ctx}xf32>" in text,
            "scores must be [numSeqs, numKvHeads, group, maxContextLen]:\n$text",
        )
    }

    @Test
    fun refusesDimDerivedAttrsByName() {
        val ex = assertFailsWith<IllegalArgumentException> {
            pagedFn(mapOf("scale" to 0.5, "numKvHeads" to numKvHeads)).toStablehlo()
        }
        assertTrue("sentinel" in (ex.message ?: ""), "must cite the sentinel-dims rule; got ${ex.message}")
    }

    @Test
    fun refusesAMissingScaleByName() {
        val ex = assertFailsWith<IllegalStateException> { pagedFn(emptyMap()).toStablehlo() }
        assertTrue("scale" in (ex.message ?: ""), "must name the missing attr; got ${ex.message}")
    }
}
