package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.465 — Phase H1a: PAGED_ATTENTION's gather-composed StableHLO emission
 * compiles under real XLA on the GB10 and agrees with the interpreter's
 * reference walk.
 *
 * This is the emission's load-bearing oracle. The interpreter walks pages
 * directly; the emission instead GATHERS the pages the block table names into
 * a dense [numSeqs, maxBlocksPerSeq*blockSize, numKvHeads, headDim] window and
 * masks everything at or past `seqLens` to -Inf. Those are genuinely different
 * programs — one skips the dead lanes, the other computes and cancels them —
 * so their agreement pins the mask algebra, the gather dimension numbers, and
 * the GQA reshape convention at once. A permuted block table and a
 * partially-filled last page are in the fixture for the same reason they are
 * in the interpreter's oracle: contiguity must never be what makes them agree.
 *
 * `blockTables` and `seqLens` ride as I32 CONSTS rather than params because
 * [PjrtSession.runOn] is an all-F32 param lane (§0.4.354's per-lane single
 * dtype). That is a harness limit, not a shape limit — a mixed-dtype input
 * lane is a named Phase H deferral, and H3's serving path feeds the caches
 * and tables as device buffers through the manifest, not through this test
 * helper.
 *
 * Tolerance: the documented GPU-vs-host float agreement FLOOR (never tighter
 * than 1e-4). The interpreter accumulates in Double and XLA does not; a
 * softmax over a masked window is exactly the place where that shows.
 */
class PjrtPagedAttentionSmokeTest {

    private val numSeqs = 2
    private val numHeads = 4
    private val numKvHeads = 2
    private val headDim = 4
    private val blockSize = 2
    private val numBlocks = 6
    private val maxBlocksPerSeq = 3
    private val scale = 0.5

    private val qType = DxirType(F32, listOf(numSeqs, numHeads, headDim))
    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))

    // A permuted table (pages interleaved between sequences, as an allocator
    // hands them out) and a ragged pair of lengths — seq1 stops MID-PAGE.
    private val table = intArrayOf(4, 1, 5, 2, 0, 3)
    private val lens = intArrayOf(6, 3)

    private fun pagedFn(): DxirFunction = DxirBuilder.function("paged_gpu") {
        val q = param("q", qType)
        val k = param("k", cacheType)
        val v = param("v", cacheType)
        val t = const(
            FloatArray(table.size) { table[it].toFloat() },
            DxirType(I32, listOf(numSeqs, maxBlocksPerSeq)),
        )
        val l = const(
            FloatArray(lens.size) { lens[it].toFloat() },
            DxirType(I32, listOf(numSeqs)),
        )
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, mapOf("scale" to scale)))
    }

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 2f
        }
    }

    @Test
    fun pagedAttentionEmissionMatchesInterpreterOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val q = pseudo(numSeqs * numHeads * headDim, 101)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 103)
        val v = pseudo(numBlocks * blockSize * numKvHeads * headDim, 107)
        val fn = pagedFn()
        val want = DxirInterpreter.evalFunction(fn, listOf(q, k, v))[0]

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val got = session.runOn(fn, listOf(q, k, v)).single()
            kotlin.test.assertEquals(want.size, got.size, "paged attention output size")
            var worst = 0f
            for (i in want.indices) {
                val e = abs(want[i] - got[i])
                if (e > worst) worst = e
                assertTrue(
                    e <= 1e-4f,
                    "paged attention lane $i: GPU ${got[i]} vs interpreter ${want[i]} (|d| = $e)",
                )
            }
            println(
                "[pjrt-paged] gather-composed PAGED_ATTENTION emission agrees with the " +
                    "interpreter on GB10 (permuted block table, partial last page); worst |d| = $worst",
            )
        }
    }
}
