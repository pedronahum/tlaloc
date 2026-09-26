package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.PagedAttentionAttrs
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The two long-context lowerings of PAGED_ATTENTION on the GB10, against the
 * interpreter's walk:
 *
 * - rows sharing a table over a context of several key blocks run the
 *   blockwise form (a `stablehlo.while` over blocks of 2,048 keys with a
 *   running max and sum), with full attention and with a sliding window
 *   whose first live block is not block 0;
 * - a ring table (`ring = true`), per row and shared, reads the same
 *   positions a full-width table over the same pages reads.
 *
 * The block tables and lengths are f32 parameters cast to i32 in the graph,
 * so that XLA sees them as runtime values (a constant length would let it
 * fold the loop's trip count).
 *
 * Set TLALOC_PAGED_BENCH=1 to also time one layer of Muse Glimmer's prefill
 * attention (512 rows, 32 heads over 2 kv heads of 128) at context 32,768.
 */
class PjrtBlockwisePagedAttentionTest {

    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            (((s ushr 16) and 0x7fff) / 32768f - 0.5f) * 2f
        }
    }

    private class Shape(
        val rows: Int,
        val tables: Int,
        val heads: Int,
        val kvHeads: Int,
        val headDim: Int,
        val blockSize: Int,
        val numBlocks: Int,
        val tableWidth: Int,
    )

    private fun fn(sh: Shape, scale: Double, window: Int?, ring: Boolean): DxirFunction =
        DxirBuilder.function("paged_long") {
            val qT = DxirType(F32, listOf(sh.rows, sh.heads, sh.headDim))
            val poolT = DxirType(F32, listOf(sh.numBlocks, sh.blockSize, sh.kvHeads, sh.headDim))
            val q = param("q", qT)
            val k = param("k", poolT)
            val v = param("v", poolT)
            val tf = param("tables", DxirType(F32, listOf(sh.tables, sh.tableWidth)))
            val lf = param("lens", DxirType(F32, listOf(sh.rows)))
            val t = op(OpKind.CAST, listOf(tf), DxirType(I32, tf.type.dims))
            val l = op(OpKind.CAST, listOf(lf), DxirType(I32, lf.type.dims))
            val attrs = buildMap<String, Any> {
                put("scale", scale)
                if (window != null) put(PagedAttentionAttrs.SLIDING_WINDOW, window)
                if (ring) put(PagedAttentionAttrs.RING, true)
            }
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qT, attrs))
        }

    private fun worst(a: FloatArray, b: FloatArray): Float = a.indices.maxOf { abs(a[it] - b[it]) }

    /** A permuted page list: sequence j's block b is page 1 + (b * 7 + j * 3) % (numBlocks - 1). */
    private fun pages(sh: Shape, tables: Int, width: Int): FloatArray =
        FloatArray(tables * width) { i ->
            val j = i / width; val b = i % width
            (1 + (b * 7 + j * 3) % (sh.numBlocks - 1)).toFloat()
        }

    @Test
    fun blockwiseRowsSharingATableMatchTheInterpreterOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        // Context 4,096 in pages of 16: two key blocks of 2,048.
        val sh = Shape(rows = 16, tables = 2, heads = 4, kvHeads = 2, headDim = 8, blockSize = 16, numBlocks = 300, tableWidth = 256)
        val poolN = sh.numBlocks * sh.blockSize * sh.kvHeads * sh.headDim
        val q = pseudo(sh.rows * sh.heads * sh.headDim, 11)
        val k = pseudo(poolN, 13)
        val v = pseudo(poolN, 17)
        val tables = pages(sh, sh.tables, sh.tableWidth)
        // Sequence 0 is a chunk across the block boundary at 2,048; sequence 1
        // a chunk inside block 0. Each row's length is its position + 1.
        val lens = FloatArray(sh.rows) { i -> if (i < 8) (2045 + i).toFloat() else (11 + i - 8).toFloat() }
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            for (window in listOf(null, 300, 3000)) {
                val f = fn(sh, 0.35, window, ring = false)
                val mlir = f.toStablehlo()
                assertTrue("stablehlo.while" in mlir, "context 4096 in shared tables takes the blockwise form")
                val inputs = listOf(q, k, v, tables, lens)
                val want = DxirInterpreter.evalFunction(f, inputs)[0]
                val got = session.runOn(f, inputs).single()
                val w = worst(want, got)
                assertTrue(w <= 1e-4f, "window $window: GPU vs interpreter worst |d| = $w")
                // Control: other pages move the output.
                val moved = session.runOn(f, listOf(q, k, v, pages(sh, sh.tables, sh.tableWidth).reversedArray(), lens)).single()
                assertTrue(worst(moved, got) > 1e-2f, "window $window: other pages did not move the output")
                println("[pjrt-paged-long] blockwise, window $window: worst |d| = $w against the interpreter")
            }
            // A window that starts past block 0 for every row: the loop starts at block 1.
            val late = FloatArray(sh.rows) { i -> (3000 + i).toFloat() }
            val f = fn(sh, 0.35, 300, ring = false)
            val want = DxirInterpreter.evalFunction(f, listOf(q, k, v, tables, late))[0]
            val got = session.runOn(f, listOf(q, k, v, tables, late)).single()
            val w = worst(want, got)
            assertTrue(w <= 1e-4f, "a window past block 0: worst |d| = $w")
            println("[pjrt-paged-long] blockwise, window 300 past block 0: worst |d| = $w")
        }
    }

    /**
     * A ring of 5 pages of 4 (20 positions) for a window of 12, filled as a
     * runtime fills it: position t on ring page (t / 4) % 5, the latest
     * write winning. Read as a ring and as a full-width table over the same
     * pages, the interpreter gives the same bits and the GPU agrees with it.
     */
    @Test
    fun ringTablesReadWhatFullWidthTablesReadOnGpu() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val bs = 4; val ringPages = 5; val window = 12; val width = 16
        val heads = 4; val kvHeads = 2; val hd = 8
        val numBlocks = 1 + 3 * ringPages
        val lensDecode = intArrayOf(7, 20, 53)
        val rings = List(3) { s -> IntArray(ringPages) { 1 + s * ringPages + (it * 2 + s) % ringPages } }
        val slot = hd * kvHeads
        val k = FloatArray(numBlocks * bs * slot)
        val v = FloatArray(numBlocks * bs * slot)
        for (s in 0 until 3) {
            val kk = pseudo(lensDecode[s] * slot, 100 + s)
            val vv = pseudo(lensDecode[s] * slot, 200 + s)
            for (t in 0 until lensDecode[s]) {
                val page = rings[s][(t / bs) % ringPages]
                val at = (page * bs + t % bs) * slot
                kk.copyInto(k, at, t * slot, (t + 1) * slot)
                vv.copyInto(v, at, t * slot, (t + 1) * slot)
            }
        }
        fun ringTable(rows: List<Int>) = FloatArray(rows.size * ringPages) { rings[rows[it / ringPages]][it % ringPages].toFloat() }
        fun fullTable(rows: List<Int>) = FloatArray(rows.size * width) { rings[rows[it / width]][(it % width) % ringPages].toFloat() }
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            // Decode: one row per sequence.
            val q = pseudo(3 * heads * hd, 7)
            val lens = FloatArray(3) { lensDecode[it].toFloat() }
            val ringFn = fn(Shape(3, 3, heads, kvHeads, hd, bs, numBlocks, ringPages), 0.4, window, ring = true)
            val fullFn = fn(Shape(3, 3, heads, kvHeads, hd, bs, numBlocks, width), 0.4, window, ring = false)
            val fromRing = DxirInterpreter.evalFunction(ringFn, listOf(q, k, v, ringTable(listOf(0, 1, 2)), lens))[0]
            val fromFull = DxirInterpreter.evalFunction(fullFn, listOf(q, k, v, fullTable(listOf(0, 1, 2)), lens))[0]
            // Rows 0 and 1 never wrapped, row 2 did: the full-width table names
            // its pages the same way, so the positions in its window agree.
            for (i in fromRing.indices) {
                assertEquals(fromFull[i].toRawBits(), fromRing[i].toRawBits(), "decode lane $i: ring vs full-width table")
            }
            val got = session.runOn(ringFn, listOf(q, k, v, ringTable(listOf(0, 1, 2)), lens)).single()
            val w = worst(fromRing, got)
            assertTrue(w <= 1e-4f, "decode through a ring: GPU vs interpreter worst |d| = $w")
            // Control: a ring rotated by one page reads other positions.
            val rotated = FloatArray(3 * ringPages) { rings[it / ringPages][(it % ringPages + 1) % ringPages].toFloat() }
            val moved = session.runOn(ringFn, listOf(q, k, v, rotated, lens)).single()
            assertTrue(worst(moved, got) > 1e-2f, "a rotated ring did not move the output")
            println("[pjrt-paged-long] ring decode: worst |d| = $w; ring and full-width tables agree bit for bit")

            // Prefill: the last 4 rows of sequence 2 (positions 49..52) share its ring.
            val q4 = pseudo(4 * heads * hd, 9)
            val lens4 = FloatArray(4) { (50 + it).toFloat() }
            val ring4 = fn(Shape(4, 1, heads, kvHeads, hd, bs, numBlocks, ringPages), 0.4, window, ring = true)
            val full4 = fn(Shape(4, 1, heads, kvHeads, hd, bs, numBlocks, width), 0.4, window, ring = false)
            val r4 = DxirInterpreter.evalFunction(ring4, listOf(q4, k, v, ringTable(listOf(2)), lens4))[0]
            val f4 = DxirInterpreter.evalFunction(full4, listOf(q4, k, v, fullTable(listOf(2)), lens4))[0]
            for (i in r4.indices) assertEquals(f4[i].toRawBits(), r4[i].toRawBits(), "prefill lane $i: ring vs full-width table")
            val g4 = session.runOn(ring4, listOf(q4, k, v, ringTable(listOf(2)), lens4)).single()
            val w4 = worst(r4, g4)
            assertTrue(w4 <= 1e-4f, "prefill through a ring: GPU vs interpreter worst |d| = $w4")
            println("[pjrt-paged-long] ring prefill rows: worst |d| = $w4")
        }
    }

    @Test
    fun museGlimmerPrefillAttentionTimings() {
        assumeTrue(System.getenv("TLALOC_PAGED_BENCH") == "1", "set TLALOC_PAGED_BENCH=1 to time it")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val rows = 512; val heads = 32; val kvHeads = 2; val hd = 128; val bs = 16
        val ctx = 32768; val width = ctx / bs; val ringPages = 160
        val numBlocks = width + 1
        val scale = 1.0 / kotlin.math.sqrt(hd.toDouble())
        val poolDims = listOf(numBlocks, bs, kvHeads, hd)
        val poolN = numBlocks * bs * kvHeads * hd
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val q = session.bufferFromHostF32(pseudo(rows * heads * hd, 3), listOf(rows, heads, hd))
            val k = session.bufferFromHostF32(pseudo(poolN, 5).also { a -> for (i in a.indices) a[i] *= 3f }, poolDims)
            val v = session.bufferFromHostF32(pseudo(poolN, 7), poolDims)
            val fullTable = session.bufferFromHostF32(FloatArray(width) { (1 + it).toFloat() }, listOf(1, width))
            val ringTable = session.bufferFromHostF32(FloatArray(ringPages) { (1 + it).toFloat() }, listOf(1, ringPages))
            fun time(label: String, f: DxirFunction, table: io.tlaloc.runtime.pjrt.ffm.PjrtBuffer, start: Int) {
                val lens = session.bufferFromHostF32(FloatArray(rows) { (start + 1 + it).toFloat() }, listOf(rows))
                val ins = listOf(q, k, v, table, lens)
                session.executeOn(f, ins).forEach { it.close() } // compile + warm
                val ms = (0 until 5).map {
                    val t0 = System.nanoTime()
                    session.executeOn(f, ins).forEach { it.close() }
                    (System.nanoTime() - t0) / 1e6
                }.sorted()
                println("[paged-bench] $label, chunk at $start: median ${"%.1f".format(ms[2])} ms (${ms.joinToString { "%.1f".format(it) }})")
                lens.close()
            }
            val full = fn(Shape(rows, 1, heads, kvHeads, hd, bs, numBlocks, width), scale, null, ring = false)
            val sliding = fn(Shape(rows, 1, heads, kvHeads, hd, bs, numBlocks, width), scale, 2048, ring = false)
            val ring = fn(Shape(rows, 1, heads, kvHeads, hd, bs, numBlocks, ringPages), scale, 2048, ring = true)
            for (start in listOf(1536, 16384, 32256)) {
                time("full attention, table of $width pages", full, fullTable, start)
                time("sliding 2048, table of $width pages", sliding, fullTable, start)
                time("sliding 2048, ring of $ringPages pages", ring, ringTable, start)
            }
            listOf(q, k, v, fullTable, ringTable).forEach { it.close() }
        }
    }
}
