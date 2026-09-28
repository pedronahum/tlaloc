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
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
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
        TestBackend.session().use { session ->
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
     * The blockwise form at its edges, against the interpreter and against
     * the per-row form (one table per row, the one-pass softmax over the
     * whole context), on a context of three key blocks (6,144 positions):
     *
     * - lengths at a block boundary (2,048, 4,096 and the whole context) and
     *   one past it (2,049, where the last block has one live position);
     * - lengths of 1, and rows of length 0 next to live rows (0 out);
     * - every row of length 0 (the loop runs no block);
     * - a window of 2,048 whose start crosses the boundary at 2,048 inside
     *   one chunk, and one that starts exactly on it;
     * - blocks that are fully masked for a row: a row live only in block 0
     *   next to a row live only in block 2, with block 1 masked for both.
     */
    @Test
    fun blockwiseEdgesMatchTheInterpreterAndThePerRowFormOnGpu() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val rows = 8
        val tables = 2
        val sh = Shape(rows = rows, tables = tables, heads = 4, kvHeads = 2, headDim = 8, blockSize = 16, numBlocks = 400, tableWidth = 384)
        val poolN = sh.numBlocks * sh.blockSize * sh.kvHeads * sh.headDim
        val q = pseudo(rows * sh.heads * sh.headDim, 21)
        val k = pseudo(poolN, 23).also { a -> for (i in a.indices) a[i] *= 4f }
        val v = pseudo(poolN, 29)
        val shared = pages(sh, tables, sh.tableWidth)
        // The per-row form: each row gets its own copy of its table.
        val perRowTables = FloatArray(rows * sh.tableWidth) { i ->
            val r = i / sh.tableWidth
            shared[(r / (rows / tables)) * sh.tableWidth + i % sh.tableWidth]
        }
        val perRowShape = Shape(rows, rows, sh.heads, sh.kvHeads, sh.headDim, sh.blockSize, sh.numBlocks, sh.tableWidth)
        val cases = listOf(
            Triple("boundaries", null, intArrayOf(2047, 2048, 2049, 4096, 4095, 4097, 6143, 6144)),
            Triple("lengths 1 and 0", null, intArrayOf(1, 0, 1, 2, 0, 1, 2049, 1)),
            Triple("all empty", null, IntArray(rows)),
            Triple("window crossing 2048", 2048, intArrayOf(4090, 4093, 4096, 4097, 4094, 4095, 4096, 4100)),
            Triple("window starting on 2048", 2048, intArrayOf(4096, 4096, 4096, 4096, 2048, 2048, 2049, 2050)),
            Triple("fully masked blocks", 300, intArrayOf(10, 11, 12, 13, 6000, 6001, 6002, 6003)),
            Triple("fully masked, no window", null, intArrayOf(10, 11, 12, 13, 6000, 6001, 6002, 6003)),
        )
        TestBackend.session().use { session ->
            for ((label, window, lensI) in cases) {
                val lens = FloatArray(rows) { lensI[it].toFloat() }
                val f = fn(sh, 0.35, window, ring = false)
                assertTrue("stablehlo.while" in f.toStablehlo(), "$label: context 6144 in shared tables takes the blockwise form")
                val inputs = listOf(q, k, v, shared, lens)
                val want = DxirInterpreter.evalFunction(f, inputs)[0]
                val got = session.runOn(f, inputs).single()
                assertTrue(got.all { it.isFinite() }, "$label: the blockwise form produced a non-finite value")
                val w = worst(want, got)
                assertTrue(w <= 1e-4f, "$label: GPU vs interpreter worst |d| = $w")
                val lane = sh.heads * sh.headDim
                for (r in 0 until rows) {
                    if (lensI[r] != 0) continue
                    for (j in 0 until lane) assertEquals(0f, got[r * lane + j], "$label: row $r of length 0 is not 0")
                }
                // The per-row form over the same pages (rows of length 0 left
                // out: its one-pass softmax has nothing to divide by there).
                val old = session.runOn(fn(perRowShape, 0.35, window, ring = false), listOf(q, k, v, perRowTables, lens)).single()
                var wOld = 0f
                for (r in 0 until rows) {
                    if (lensI[r] == 0) continue
                    for (j in 0 until lane) wOld = maxOf(wOld, abs(old[r * lane + j] - got[r * lane + j]))
                }
                assertTrue(wOld <= 1e-4f, "$label: blockwise vs per-row form worst |d| = $wOld")
                println("[pjrt-paged-long] blockwise edge '$label': worst |d| = $w against the interpreter, $wOld against the per-row form")
            }
            // Control: moving one row's length by one position moves that row only.
            val base = intArrayOf(2047, 2048, 2049, 4096, 4095, 4097, 6143, 6144)
            val f = fn(sh, 0.35, null, ring = false)
            val a = session.runOn(f, listOf(q, k, v, shared, FloatArray(rows) { base[it].toFloat() })).single()
            val b = session.runOn(f, listOf(q, k, v, shared, FloatArray(rows) { (base[it] - if (it == 2) 1 else 0).toFloat() })).single()
            val lane = sh.heads * sh.headDim
            val moved = (0 until rows).filter { r -> (0 until lane).any { j -> a[r * lane + j] != b[r * lane + j] } }
            assertEquals(listOf(2), moved, "a length one shorter must move exactly its own row")
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
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
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
        TestBackend.session().use { session ->
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

    /**
     * Ring decode at the ring's edges: lengths of 1, the window (12), the
     * ring (20), twice the ring (40, the ring's column 0 just rewritten) and
     * one past it (41), each row's ring filled as a runtime fills it. The
     * interpreter's ring walk must give the bits of a full-width table over
     * the same pages, and the GPU must agree with it.
     */
    @Test
    fun ringDecodeAtTheRingsEdgesOnGpu() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val bs = 4; val ringPages = 5; val window = 12; val width = 16
        val heads = 4; val kvHeads = 2; val hd = 8
        val lensI = intArrayOf(1, 12, 20, 40, 41)
        val n = lensI.size
        val numBlocks = 1 + n * ringPages
        val rings = List(n) { s -> IntArray(ringPages) { 1 + s * ringPages + (it * 3 + s) % ringPages } }
        val slot = hd * kvHeads
        val k = FloatArray(numBlocks * bs * slot)
        val v = FloatArray(numBlocks * bs * slot)
        for (s in 0 until n) {
            val kk = pseudo(lensI[s] * slot, 300 + s)
            val vv = pseudo(lensI[s] * slot, 400 + s)
            for (t in 0 until lensI[s]) {
                val at = (rings[s][(t / bs) % ringPages] * bs + t % bs) * slot
                kk.copyInto(k, at, t * slot, (t + 1) * slot)
                vv.copyInto(v, at, t * slot, (t + 1) * slot)
            }
        }
        val ringT = FloatArray(n * ringPages) { rings[it / ringPages][it % ringPages].toFloat() }
        val fullT = FloatArray(n * width) { rings[it / width][(it % width) % ringPages].toFloat() }
        val q = pseudo(n * heads * hd, 31)
        val lens = FloatArray(n) { lensI[it].toFloat() }
        val ringFn = fn(Shape(n, n, heads, kvHeads, hd, bs, numBlocks, ringPages), 0.4, window, ring = true)
        val fullFn = fn(Shape(n, n, heads, kvHeads, hd, bs, numBlocks, width), 0.4, window, ring = false)
        val fromRing = DxirInterpreter.evalFunction(ringFn, listOf(q, k, v, ringT, lens))[0]
        val fromFull = DxirInterpreter.evalFunction(fullFn, listOf(q, k, v, fullT, lens))[0]
        for (i in fromRing.indices) assertEquals(fromFull[i].toRawBits(), fromRing[i].toRawBits(), "lane $i: ring vs full-width table")
        TestBackend.session().use { session ->
            val got = session.runOn(ringFn, listOf(q, k, v, ringT, lens)).single()
            val w = worst(fromRing, got)
            assertTrue(w <= 1e-4f, "ring decode at the edges: GPU vs interpreter worst |d| = $w")
            // Control: a window of the whole ring (20) reads 8 more
            // positions for the rows longer than 12, and moves them.
            val wider = fn(Shape(n, n, heads, kvHeads, hd, bs, numBlocks, ringPages), 0.4, window + 8, ring = true)
            val moved = session.runOn(wider, listOf(q, k, v, ringT, lens)).single()
            assertTrue(worst(moved, got) > 1e-3f, "a wider window did not move the output")
            println("[pjrt-paged-long] ring decode at lengths ${lensI.toList()}: worst |d| = $w")
        }
    }

    @Test
    fun museGlimmerPrefillAttentionTimings() {
        assumeTrue(System.getenv("TLALOC_PAGED_BENCH") == "1", "set TLALOC_PAGED_BENCH=1 to time it")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val rows = 512; val heads = 32; val kvHeads = 2; val hd = 128; val bs = 16
        val ctx = 32768; val width = ctx / bs; val ringPages = 160
        val numBlocks = width + 1
        val scale = 1.0 / kotlin.math.sqrt(hd.toDouble())
        val poolDims = listOf(numBlocks, bs, kvHeads, hd)
        val poolN = numBlocks * bs * kvHeads * hd
        TestBackend.session().use { session ->
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
