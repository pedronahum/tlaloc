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

    /**
     * Rows that share a table (a prefill chunk): the pages are gathered once
     * per TABLE, into [tables, ctx, Hkv, D], and the dots batch over (table,
     * kv head) with the rows as a free axis. Control: the per-row form (one
     * table per row) gathers per row and batches over (row, kv head).
     */
    @Test
    fun rowsSharingATableGatherEachTableOnce() {
        val rows = 6
        val rowQ = DxirType(F32, listOf(rows, numHeads, headDim))
        fun fn(tableRows: Int) = DxirBuilder.function("shared") {
            val q = param("q", rowQ)
            val k = param("k", cacheType)
            val v = param("v", cacheType)
            val t = param("t", DxirType(I32, listOf(tableRows, maxBlocksPerSeq)))
            val l = param("l", DxirType(I32, listOf(rows)))
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), rowQ, mapOf("scale" to 0.5)))
        }
        val ctx = maxBlocksPerSeq * blockSize
        val shared = fn(2).toStablehlo()
        assertTrue("-> tensor<2x${ctx}x${numKvHeads}x${headDim}xf32>" in shared, "one window per table:\n$shared")
        assertTrue("batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3]" in shared, shared)
        assertTrue("tensor<2x${numKvHeads}x3x${numHeads / numKvHeads}x${ctx}xf32>" in shared, "scores [R, Hkv, Q, G, ctx]:\n$shared")
        assertTrue("stablehlo.transpose" in shared && "dims = [0, 2, 1, 3, 4]" in shared, shared)
        val perRow = fn(rows).toStablehlo()
        assertTrue("-> tensor<${rows}x${ctx}x${numKvHeads}x${headDim}xf32>" in perRow, perRow)
        assertTrue("batching_dims = [0, 2] x [0, 2]" !in perRow && "stablehlo.transpose" !in perRow, perRow)
    }


    @Test
    fun aSlidingWindowAddsTheLowerBoundOfTheMask() {
        val plain = mlir()
        val text = pagedFn(mapOf("scale" to 0.5, "sliding_window" to 3)).toStablehlo()
        // The window's lower bound: t >= seqLens[s] - 3, and-ed with t < seqLens[s].
        assertTrue("stablehlo.constant dense<3> : tensor<i32>" in text, "window constant missing:\n$text")
        assertTrue("stablehlo.compare GE" in text, "the lower bound is `t >= seqLens[s] - window`:\n$text")
        assertTrue("stablehlo.and" in text, "the two bounds must be and-ed:\n$text")
        // Negative control: without the attr the emission has neither.
        assertTrue("stablehlo.compare GE" !in plain && "stablehlo.and" !in plain, plain)
    }

    @Test
    fun bothAttentionDotsAreExactF32Algorithms() {
        // XLA may otherwise run an f32 dot in TF32 on a GPU. The algorithm runs as XLA's own
        // f32 GEMM; `precision = HIGHEST` (what portableF32Dots writes for a TPU) as a SIMT kernel.
        val text = mlir()
        val algorithm = "algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32"
        kotlin.test.assertEquals(2, text.lines().count { "stablehlo.dot_general" in it && algorithm in it }, text)
        assertTrue("HIGHEST" !in text && "tf32" !in text, text)
        kotlin.test.assertEquals(
            2, portableF32Dots(text).lines().count { "stablehlo.dot_general" in it && "precision = [HIGHEST, HIGHEST]" in it },
        )
    }

    /**
     * Rows sharing a table (prefill) spell their dots as the f32 algorithm:
     * f32 operands, products and sums, which XLA may run as its own GEMM.
     * Neither form lets the GPU round an operand to TF32.
     */
    @Test
    fun rowsSharingATableAskForTheF32DotAlgorithm() {
        val rows = 6
        val rowQ = DxirType(F32, listOf(rows, numHeads, headDim))
        val text = DxirBuilder.function("shared") {
            val q = param("q", rowQ)
            val k = param("k", cacheType)
            val v = param("v", cacheType)
            val t = param("t", DxirType(I32, listOf(2, maxBlocksPerSeq)))
            val l = param("l", DxirType(I32, listOf(rows)))
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), rowQ, mapOf("scale" to 0.5)))
        }.toStablehlo()
        val algorithm = "algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32"
        kotlin.test.assertEquals(2, text.lines().count { "stablehlo.dot_general" in it && algorithm in it }, text)
        assertTrue("HIGHEST" !in text && "tf32" !in text, text)
    }

    /**
     * A ring table: the position ring index i holds is `age = (L + N - 1 - i)
     * rem N` behind the row's last one, live when `age < W` and `age < L`.
     * Control: the same op without `ring` has no remainder.
     */
    @Test
    fun aRingMasksByTheAgeOfEachRingSlot() {
        val attrs = mapOf("scale" to 0.5, "sliding_window" to 4)
        val text = pagedFn(attrs + ("ring" to true)).toStablehlo()
        val n = maxBlocksPerSeq * blockSize
        assertTrue("stablehlo.remainder" in text, text)
        assertTrue("stablehlo.constant dense<${n - 1}> : tensor<i32>" in text, "L + N - 1:\n$text")
        assertTrue("stablehlo.constant dense<$n> : tensor<i32>" in text, "rem N:\n$text")
        assertTrue("stablehlo.constant dense<4> : tensor<i32>" in text, "age < W:\n$text")
        assertTrue("stablehlo.remainder" !in pagedFn(attrs).toStablehlo())
    }

    /**
     * Eight rows sharing a table over a context of two key blocks (4,096
     * positions in pages of 16) take the blockwise form: one `stablehlo.while`
     * whose body slices a block of 2,048 keys, with the running max and sum
     * carried; nothing of the width of the context is scored. A context of
     * one key block (2,048) keeps the one-pass form, and so do four rows per
     * table (a speculative verify step) over any context.
     */
    @Test
    fun aContextOfSeveralKeyBlocksIsAttendedBlockByBlock() {
        val rows = 8
        val rowQ = DxirType(F32, listOf(rows, numHeads, headDim))
        fun fn(width: Int) = DxirBuilder.function("long") {
            val pool = DxirType(F32, listOf(width + 1, 16, numKvHeads, headDim))
            val q = param("q", rowQ)
            val k = param("k", pool)
            val v = param("v", pool)
            val t = param("t", DxirType(I32, listOf(1, width)))
            val l = param("l", DxirType(I32, listOf(rows)))
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), rowQ, mapOf("scale" to 0.5)))
        }
        val long = fn(256).toStablehlo()
        kotlin.test.assertEquals(1, long.lines().count { "\"stablehlo.while\"" in it }, long)
        assertTrue("stablehlo.dynamic_slice" in long && "sizes = [1, 2048, $numKvHeads, $headDim]" in long, long)
        val g = numHeads / numKvHeads
        assertTrue("tensor<1x${numKvHeads}x${rows}x${g}x2048xf32>" in long, "scores of one key block:\n$long")
        assertTrue("x4096xf32>" !in long.substringAfter("stablehlo.while"), "a score as wide as the context:\n$long")
        assertTrue("stablehlo.is_finite" in long, long)
        val short = fn(128).toStablehlo()
        assertTrue("stablehlo.while" !in short && "tensor<1x${numKvHeads}x${rows}x${g}x2048xf32>" in short, short)
        val fewRows = DxirBuilder.function("verify") {
            val q4 = DxirType(F32, listOf(4, numHeads, headDim))
            val pool = DxirType(F32, listOf(257, 16, numKvHeads, headDim))
            listOf(
                op(
                    OpKind.PAGED_ATTENTION,
                    listOf(param("q", q4), param("k", pool), param("v", pool), param("t", DxirType(I32, listOf(1, 256))), param("l", DxirType(I32, listOf(4)))),
                    q4, mapOf("scale" to 0.5),
                ),
            )
        }.toStablehlo()
        assertTrue("stablehlo.while" !in fewRows, "four rows per table keep the one-pass form:\n$fewRows")
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
