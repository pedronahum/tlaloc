package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.I32
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.PagedAttentionAttrs
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * PAGED_ATTENTION as the fused CUDA kernel (`tlaloc_paged_attention`, built
 * into triton/backends/tlaloc/libtlaloc_kernels.so by triton/build_backend.sh)
 * against XLA's form of the same op and against the interpreter: decode rows
 * (one per table), verify rows (four per table) and prefill rows (more than
 * 64 query vectors per table and KV head: the kernel's prefill form), f32 and
 * e4m3fn pools, lengths across slice and tile boundaries. TLALOC_ATTN_BENCH=1 also times
 * both forms at Qwen3.8-27B's layout over a 32K bucket.
 */
class PjrtFusedPagedAttentionTest {

    private val library = Path.of("..", "triton", "backends", "tlaloc", "libtlaloc_kernels.so").toAbsolutePath().normalize()

    private fun registered(): Boolean {
        if (!TestBackend.deviceAvailable || TestBackend.isTpu || !Files.isRegularFile(library)) return false
        PjrtFfiRegistry.registerNativeLibrary(PjrtBinaries.requireCudaPlugin("PjrtFusedPagedAttentionTest"), library)
        return true
    }

    private class Case(
        val rows: Int, val tables: Int, val heads: Int, val kvHeads: Int, val dim: Int,
        val bs: Int, val width: Int, val numBlocks: Int, val pool: DType,
    )

    private fun fn(c: Case, fused: Boolean): DxirFunction = DxirBuilder.function("attn") {
        val qT = DxirType(F32, listOf(c.rows, c.heads, c.dim))
        val poolT = DxirType(c.pool, listOf(c.numBlocks, c.bs, c.kvHeads, c.dim))
        val q = param("q", qT)
        val k = param("k", poolT)
        val v = param("v", poolT)
        val t = param("t", DxirType(I32, listOf(c.tables, c.width)))
        val l = param("l", DxirType(I32, listOf(c.rows)))
        val attrs = buildMap<String, Any> {
            put("scale", 0.0625)
            if (fused) put(PagedAttentionAttrs.FUSED_KERNEL, true)
        }
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qT, attrs))
    }

    private fun check(c: Case, lens: IntArray, seed: Int) {
        val rnd = Random(seed)
        val n = c.numBlocks * c.bs * c.kvHeads * c.dim
        val q = FloatArray(c.rows * c.heads * c.dim) { rnd.nextFloat() * 2 - 1 }
        val kCodes = ByteArray(n) { (rnd.nextInt(0x60) or (rnd.nextInt(2) shl 7)).toByte() }
        val vCodes = ByteArray(n) { (rnd.nextInt(0x60) or (rnd.nextInt(2) shl 7)).toByte() }
        val kF = FloatArray(n) { if (c.pool == F32) rnd.nextFloat() * 4 - 2 else f8e4m3fnToFloat(kCodes[it]) }
        val vF = FloatArray(n) { if (c.pool == F32) rnd.nextFloat() * 2 - 1 else f8e4m3fnToFloat(vCodes[it]) }
        // Each table names distinct pages, spread over the pool.
        val table = IntArray(c.tables * c.width) { 1 + (it * 7) % (c.numBlocks - 1) }
        val fused = fn(c, true)
        val plain = fn(c, false)
        assertTrue("tlaloc_paged_attention" in fused.toStablehlo(), "the fused form is emitted")
        val want = DxirInterpreter.evalFunction(
            plain, listOf(q, kF, vF, FloatArray(table.size) { table[it].toFloat() }, FloatArray(lens.size) { lens[it].toFloat() }),
        )[0]
        TestBackend.session().use { s ->
            fun pools(): List<Any> = if (c.pool == F32) listOf(kF, vF) else listOf(kCodes, vCodes)
            val got = s.runOnHost(fused, listOf<Any>(q) + pools() + listOf(table, lens)).single() as FloatArray
            val xla = s.runOnHost(plain, listOf<Any>(q) + pools() + listOf(table, lens)).single() as FloatArray
            var worst = 0f
            var worstXla = 0f
            var mag = 0f
            for (i in want.indices) {
                worst = maxOf(worst, abs(got[i] - want[i]))
                worstXla = maxOf(worstXla, abs(got[i] - xla[i]))
                mag = maxOf(mag, abs(want[i]))
            }
            // f32 sums over a few hundred positions of values up to ~30: 1e-4, or 1e-5 of the largest output.
            val tol = maxOf(1e-4f, 1e-5f * mag)
            println(
                "[fused-paged] rows=${c.rows} tables=${c.tables} heads=${c.heads}/${c.kvHeads} D=${c.dim} pool=${c.pool.name} " +
                    "lens=${lens.toList()}: worst |d| $worst against the interpreter, $worstXla against XLA's form (largest |y| $mag)",
            )
            assertTrue(worst <= tol && worstXla <= tol, "fused kernel differs: $worst / $worstXla of $mag")
        }
    }

    @Test
    fun theFusedKernelAttendsAsTheInterpreterAndXlasForm() {
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so (triton/build_backend.sh)")
        // Decode: one row per table, a context of 3 slices of 256.
        val decode = Case(rows = 3, tables = 3, heads = 6, kvHeads = 2, dim = 16, bs = 16, width = 48, numBlocks = 200, pool = F32)
        check(decode, intArrayOf(1, 300, 768), 1)
        check(Case(3, 3, 6, 2, 16, 16, 48, 200, F8E4M3FN), intArrayOf(31, 257, 700), 2)
        // Verify: four rows per table, causal lengths.
        val verify = Case(rows = 8, tables = 2, heads = 8, kvHeads = 2, dim = 32, bs = 16, width = 40, numBlocks = 120, pool = F8E4M3FN)
        check(verify, intArrayOf(500, 501, 502, 503, 30, 31, 32, 33), 3)
        // Qwen3.5's head dim and the 27B's grouping, two verify rows per table.
        check(Case(rows = 4, tables = 2, heads = 24, kvHeads = 4, dim = 256, bs = 16, width = 32, numBlocks = 80, pool = F8E4M3FN), intArrayOf(400, 401, 64, 65), 4)
        // Prefill: 40 consecutive tokens per table at the 35B's grouping (tiles of
        // 8 tokens), and 23 at the 27B's (tiles of 10, the last one partial).
        check(
            Case(rows = 80, tables = 2, heads = 16, kvHeads = 2, dim = 256, bs = 16, width = 40, numBlocks = 100, pool = F8E4M3FN),
            IntArray(80) { if (it < 40) 560 + it + 1 else 20 + (it - 40) + 1 },
            5,
        )
        check(
            Case(rows = 23, tables = 1, heads = 24, kvHeads = 4, dim = 256, bs = 16, width = 20, numBlocks = 40, pool = F8E4M3FN),
            IntArray(23) { 290 + it + 1 },
            6,
        )
    }

    @Test
    fun fusedAgainstXlaTimings() {
        assumeTrue(System.getenv("TLALOC_ATTN_BENCH") == "1", "set TLALOC_ATTN_BENCH=1")
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so")
        val bs = 16; val d = 256; val nb = 8200; val ctx = 32768; val width = ctx / bs
        val rnd = Random(9)
        TestBackend.session().use { s ->
            for ((model, heads, kvHeads) in listOf(Triple("27B", 24, 4), Triple("35B", 16, 2))) {
                val poolDims = listOf(nb, bs, kvHeads, d)
                val k = s.bufferFromHostBytes(ByteArray(nb * bs * kvHeads * d) { rnd.nextInt(0x50).toByte() }, poolDims, F8E4M3FN)
                val v = s.bufferFromHostBytes(ByteArray(nb * bs * kvHeads * d) { rnd.nextInt(0x50).toByte() }, poolDims, F8E4M3FN)
                for ((label, rowsPerTable) in listOf("decode, 4 sequences" to 1, "verify, 4 sequences x 4 rows" to 4)) {
                    val tables = 4
                    val rows = tables * rowsPerTable
                    val c = Case(rows, tables, heads, kvHeads, d, bs, width, nb, F8E4M3FN)
                    val q = s.bufferFromHostF32(FloatArray(rows * heads * d) { rnd.nextFloat() - 0.5f }, listOf(rows, heads, d))
                    val t = s.bufferFromHostI32(IntArray(tables * width) { 1 + (it * 7) % (nb - 1) }, listOf(tables, width))
                    val l = s.bufferFromHostI32(IntArray(rows) { ctx - 30 + it % rowsPerTable }, listOf(rows))
                    for (fused in listOf(false, true)) {
                        val f = fn(c, fused)
                        val ins = listOf(q, k, v, t, l)
                        repeat(3) { s.executeOn(f, ins).forEach { it.close() } }
                        val iters = 20
                        val t0 = System.nanoTime()
                        repeat(iters) { s.executeOn(f, ins).forEach { it.close() } }
                        val ms = (System.nanoTime() - t0) / 1e6 / iters
                        println("[fused-paged-bench] $model $label at 32K: ${if (fused) "fused kernel" else "XLA form"} %.3f ms".format(ms))
                    }
                    listOf(q, t, l).forEach { it.close() }
                }
                listOf(k, v).forEach { it.close() }
            }
        }
    }
}
