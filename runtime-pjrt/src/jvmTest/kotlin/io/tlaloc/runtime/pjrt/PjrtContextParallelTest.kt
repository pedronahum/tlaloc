package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.PagedAttentionAttrs
import io.tlaloc.ir.inference.ContextParallel
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Decode context parallelism's arithmetic on one device: each sequence's pages striped over two
 * slices (even and odd pages, as the devices of a context-parallel group hold them), each slice's
 * PAGED_ATTENTION with `return_lse`, merged by [ContextParallel.merge], against PAGED_ATTENTION
 * over the whole context. The three StableHLO forms: one row per table, rows sharing tables (a
 * verify step) and the blockwise loop (rows sharing a long context), and a row with no position.
 * The interpreter and XLA:GPU, within f32 rounding of the largest output.
 */
class PjrtContextParallelTest {

    private val heads = 4
    private val kvHeads = 2
    private val d = 16

    /** PAGED_ATTENTION over [tables] (`[T, m]`), rows of [lens]; with [lse] also its log-sum-exp. */
    private fun attention(rows: Int, tables: Int, m: Int, blocks: Int, bs: Int, lse: Boolean) =
        DxirBuilder.function("pa") {
            val q = param("q", DxirType(F32, listOf(rows, heads, d)))
            val k = param("k", DxirType(F32, listOf(blocks, bs, kvHeads, d)))
            val v = param("v", DxirType(F32, listOf(blocks, bs, kvHeads, d)))
            val t = param("tables", DxirType(I32, listOf(tables, m)))
            val l = param("lens", DxirType(I32, listOf(rows)))
            val attrs = mapOf<String, Any>("scale" to 0.25) + if (lse) mapOf(PagedAttentionAttrs.RETURN_LSE to true) else emptyMap()
            val types = listOf(DxirType(F32, listOf(rows, heads, d))) + if (lse) listOf(DxirType(F32, listOf(rows, heads))) else emptyList()
            val pa = opMulti(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), types, attrs)
            if (lse) listOf(DxirOpResult(pa, 0), DxirOpResult(pa, 1)) else listOf(pa)
        }

    /** Two slices' attention and their merge, one function. */
    private fun merged(rows: Int, tables: Int, m: Int, blocks: Int, bs: Int): DxirFunction = DxirBuilder.function("cp") {
        val q = param("q", DxirType(F32, listOf(rows, heads, d)))
        val k = param("k", DxirType(F32, listOf(blocks, bs, kvHeads, d)))
        val v = param("v", DxirType(F32, listOf(blocks, bs, kvHeads, d)))
        val parts = (0..1).map { s ->
            val t = param("tables$s", DxirType(I32, listOf(tables, m / 2)))
            val l = param("lens$s", DxirType(I32, listOf(rows)))
            val pa = opMulti(
                OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l),
                listOf(DxirType(F32, listOf(rows, heads, d)), DxirType(F32, listOf(rows, heads))),
                mapOf("scale" to 0.25, PagedAttentionAttrs.RETURN_LSE to true),
            )
            DxirOpResult(pa, 0) to DxirOpResult(pa, 1)
        }
        val (out, _) = ContextParallel.merge(this, parts.map { it.first }, parts.map { it.second })
        listOf(out)
    }

    private fun case(s: PjrtSession, label: String, tables: Int, perTable: Int, m: Int, bs: Int, lensOf: (Int) -> Int, wantWhile: Boolean) {
        val rows = tables * perTable
        val blocks = tables * m
        val rnd = Random(rows * 31 + m)
        val q = FloatArray(rows * heads * d) { rnd.nextFloat() * 2 - 1 }
        val k = FloatArray(blocks * bs * kvHeads * d) { rnd.nextFloat() * 2 - 1 }
        val v = FloatArray(blocks * bs * kvHeads * d) { rnd.nextFloat() * 2 - 1 }
        val lens = IntArray(rows) { lensOf(it) }
        val full = IntArray(tables * m) { (it / m) * m + it % m }
        // Slice sl holds the table's pages j with j % 2 == sl, in order; its row length is how many
        // of the row's positions those pages hold.
        val sliceTables = (0..1).map { sl -> IntArray(tables * m / 2) { (it / (m / 2)) * m + 2 * (it % (m / 2)) + sl } }
        val sliceLens = (0..1).map { sl ->
            IntArray(rows) { r -> (0 until lens[r]).count { ((it / bs) % 2) == sl } }
        }
        val whole = attention(rows, tables, m, blocks, bs, lse = false)
        val cp = merged(rows, tables, m, blocks, bs)
        assertTrue(("stablehlo.while" in whole.toStablehlo()) == wantWhile, "$label: the expected form")
        fun f(a: IntArray) = FloatArray(a.size) { a[it].toFloat() }
        val want = DxirInterpreter.evalFunction(whole, listOf(q, k, v, f(full), f(lens)))[0]
        val cpIns = listOf(q, k, v, f(sliceTables[0]), f(sliceLens[0]), f(sliceTables[1]), f(sliceLens[1]))
        val interp = DxirInterpreter.evalFunction(cp, cpIns)[0]
        val dev = s.runOnHost(whole, listOf<Any>(q, k, v, full, lens)).single() as FloatArray
        val devCp = s.runOnHost(cp, listOf<Any>(q, k, v, sliceTables[0], sliceLens[0], sliceTables[1], sliceLens[1])).single() as FloatArray
        val mag = want.maxOf { abs(it) }
        val di = want.indices.maxOf { abs(interp[it] - want[it]) }
        // The merged slices against the interpreter everywhere (a row with no position is 0); the
        // whole op on the device only where a row has positions (its one-pass form gives NaN for
        // an empty row, which serving never sends: a padding row has length 1).
        val live = want.indices.filter { lens[it / (heads * d)] > 0 }
        val dd = maxOf(want.indices.maxOf { abs(devCp[it] - want[it]) }, live.maxOf { abs(dev[it] - want[it]) })
        println("[context-parallel] $label: merged slices against the whole context, interpreter $di, XLA:GPU $dd (largest |out| $mag)")
        assertTrue(di <= 1e-5f * mag && dd <= 1e-5f * mag, "$label: $di / $dd of $mag")
    }

    @Test
    fun twoSlicesOfTheContextMergeToTheWholeAttention() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session().use { s ->
            case(s, "one row per table, an empty row", tables = 3, perTable = 1, m = 8, bs = 4, lensOf = { listOf(13, 0, 32)[it] }, wantWhile = false)
            case(s, "rows sharing tables", tables = 2, perTable = 4, m = 8, bs = 4, lensOf = { r -> listOf(21, 30)[r / 4] - (3 - r % 4) }, wantWhile = false)
            case(s, "blockwise", tables = 1, perTable = 8, m = 512, bs = 16, lensOf = { r -> 5000 + r }, wantWhile = true)
        }
    }
}
