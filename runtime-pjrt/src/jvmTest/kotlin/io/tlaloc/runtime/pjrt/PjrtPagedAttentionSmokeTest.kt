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

    private fun pagedFn(window: Int? = null): DxirFunction = DxirBuilder.function("paged_gpu") {
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
        val attrs = buildMap<String, Any> {
            put("scale", scale)
            if (window != null) put("sliding_window", window)
        }
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, attrs))
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

    /**
     * The sliding-window mask on the GPU: windows 1, 2 and 4 over rows of 6
     * and 3 positions, so a window edge falls inside a page and on one. The
     * windowed outputs must also differ from the full-attention output
     * (negative control), or the mask would be unobserved.
     */
    @Test
    fun slidingWindowEmissionMatchesInterpreterOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val q = pseudo(numSeqs * numHeads * headDim, 109)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 113)
        val v = pseudo(numBlocks * blockSize * numKvHeads * headDim, 127)
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val full = session.runOn(pagedFn(), listOf(q, k, v)).single()
            for (w in listOf(1, 2, 4)) {
                val fn = pagedFn(w)
                val want = DxirInterpreter.evalFunction(fn, listOf(q, k, v))[0]
                val got = session.runOn(fn, listOf(q, k, v)).single()
                var worst = 0f
                for (i in want.indices) worst = maxOf(worst, abs(want[i] - got[i]))
                assertTrue(worst <= 1e-4f, "window $w: GPU vs interpreter worst |d| = $worst")
                val moved = want.indices.maxOf { abs(full[it] - got[it]) }
                assertTrue(moved > 1e-2f, "window $w did not change the output (moved $moved)")
                println("[pjrt-paged] sliding window $w agrees with the interpreter on GB10; worst |d| = $worst")
            }
        }
    }

    /**
     * Rows that share a block table (a prefill chunk of three tokens per
     * sequence, each with its own causal length) through the shared-table
     * lowering: it agrees with the interpreter, with and without a sliding
     * window, and with the per-row lowering of the same rows (the table
     * repeated per row) to the float floor. Control: swapped tables move it.
     */
    @Test
    fun rowsSharingATableMatchTheInterpreterOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val rows = 6
        val rowQ = DxirType(F32, listOf(rows, numHeads, headDim))
        val lensRows = intArrayOf(4, 5, 6, 2, 3, 4)
        fun fn(tables: IntArray, tableRows: Int, window: Int?) = DxirBuilder.function("shared_gpu") {
            val q = param("q", rowQ)
            val k = param("k", cacheType)
            val v = param("v", cacheType)
            val t = const(FloatArray(tables.size) { tables[it].toFloat() }, DxirType(I32, listOf(tableRows, maxBlocksPerSeq)))
            val l = const(FloatArray(rows) { lensRows[it].toFloat() }, DxirType(I32, listOf(rows)))
            val attrs = buildMap<String, Any> {
                put("scale", scale)
                if (window != null) put("sliding_window", window)
            }
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), rowQ, attrs))
        }
        val repeated = IntArray(rows * maxBlocksPerSeq) { table[(it / maxBlocksPerSeq / 3) * maxBlocksPerSeq + it % maxBlocksPerSeq] }
        val swapped = intArrayOf(2, 0, 3, 4, 1, 5)
        val q = pseudo(rows * numHeads * headDim, 131)
        val k = pseudo(numBlocks * blockSize * numKvHeads * headDim, 137)
        val v = pseudo(numBlocks * blockSize * numKvHeads * headDim, 139)
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            for (w in listOf(null, 2)) {
                val shared = fn(table, 2, w)
                val want = DxirInterpreter.evalFunction(shared, listOf(q, k, v))[0]
                val got = session.runOn(shared, listOf(q, k, v)).single()
                val perRow = session.runOn(fn(repeated, rows, w), listOf(q, k, v)).single()
                val worst = want.indices.maxOf { abs(want[it] - got[it]) }
                val vsPerRow = want.indices.maxOf { abs(perRow[it] - got[it]) }
                assertTrue(worst <= 1e-4f, "window $w: GPU vs interpreter worst |d| = $worst")
                assertTrue(vsPerRow <= 1e-4f, "window $w: shared vs per-row lowering worst |d| = $vsPerRow")
                val moved = session.runOn(fn(swapped, 2, w), listOf(q, k, v)).single()
                    .let { o -> want.indices.maxOf { abs(o[it] - got[it]) } }
                assertTrue(moved > 1e-2f, "window $w: swapped tables did not change the output (moved $moved)")
                println(
                    "[pjrt-paged] rows sharing a table (window $w) agree with the interpreter on GB10; " +
                        "worst |d| = $worst, against the per-row lowering $vsPerRow",
                )
            }
        }
    }
}
