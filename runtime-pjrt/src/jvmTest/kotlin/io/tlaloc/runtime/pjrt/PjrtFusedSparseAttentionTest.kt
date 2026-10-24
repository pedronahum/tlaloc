package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.I32
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.IndexerTopkAttrs
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
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * Query-selective sparse attention as the CUDA kernels of libtlaloc_kernels.so:
 * INDEXER_TOPK as `tlaloc_indexer_topk` gives the reference's keep masks (ties at zero
 * scores, rows with fewer complete chunks than top_k, incomplete tails, a radix select
 * over 2,048 chunks), and PAGED_ATTENTION with a keep operand as `tlaloc_paged_attention`
 * attends as the interpreter and XLA's form in each of the kernel's forms at
 * Qwen3.8-Flash-Next's layout (24 heads, 2 KV heads, head dim 256): decode rows, verify
 * rows of 4 and 5 tokens, prefill rows exact and in f16.
 */
class PjrtFusedSparseAttentionTest {

    private val library = Path.of("..", "triton", "backends", "tlaloc", "libtlaloc_kernels.so").toAbsolutePath().normalize()

    private fun registered(): Boolean {
        if (!TestBackend.deviceAvailable || TestBackend.isTpu || !Files.isRegularFile(library)) return false
        PjrtFfiRegistry.registerNativeLibrary(PjrtBinaries.requireCudaPlugin("PjrtFusedSparseAttentionTest"), library)
        return true
    }

    private fun topkFn(rows: Int, heads: Int, d: Int, numBlocks: Int, bs: Int, tables: Int, width: Int, chunk: Int, topK: Int, fused: Boolean): DxirFunction =
        DxirBuilder.function("topk") {
            val q = param("q", DxirType(F32, listOf(rows, heads, d)))
            val k = param("keys", DxirType(F32, listOf(numBlocks, bs, 1, d)))
            val t = param("tables", DxirType(I32, listOf(tables, width)))
            val l = param("lens", DxirType(I32, listOf(rows)))
            val attrs = buildMap<String, Any> {
                put("chunk", chunk); put("top_k", topK); put("scale", 0.088)
                if (fused) put(IndexerTopkAttrs.FUSED_KERNEL, true)
            }
            listOf(op(OpKind.INDEXER_TOPK, listOf(q, k, t, l), DxirType(I32, listOf(rows, width * bs / chunk)), attrs))
        }

    private fun topk(rows: Int, tables: Int, width: Int, topK: Int, lens: IntArray, seed: Int, zeros: Boolean) {
        val heads = 4; val d = 128; val bs = 16; val chunk = 4
        val numBlocks = 1 + tables * width
        val fused = topkFn(rows, heads, d, numBlocks, bs, tables, width, chunk, topK, true)
        val plain = topkFn(rows, heads, d, numBlocks, bs, tables, width, chunk, topK, false)
        assertTrue("tlaloc_indexer_topk" in fused.toStablehlo())
        val rnd = Random(seed)
        // With zeros, positive keys and queries leaning negative: most chunks score relu(...) = 0
        // and tie, fewer than top_k score above 0, and the ties are broken by chunk order.
        val q = FloatArray(rows * heads * d) { rnd.nextFloat() - if (zeros) 0.53f else 0.5f }
        val keys = FloatArray(numBlocks * bs * d) { if (zeros) rnd.nextFloat() else rnd.nextFloat() - 0.5f }
        val table = IntArray(tables * width) { 1 + (it * 7) % (numBlocks - 1) }
        val want = IndexerTopkAttrs.reference(IndexerTopkAttrs.parse(plain.returns[0] as io.tlaloc.ir.DxirOp, "test"), q, keys, table, lens)
        TestBackend.session().use { s ->
            val got = s.runOnHost(fused, listOf(q, keys, table, lens))[0] as IntArray
            val xla = s.runOnHost(plain, listOf(q, keys, table, lens))[0] as IntArray
            val nc = width * bs / chunk
            val kept = (0 until rows).map { r -> (0 until nc).count { want[r * nc + it] == 1 } }
            val positive = (0 until rows).map { r ->
                (0 until minOf(nc, lens[r] / chunk)).count { c ->
                    val pos = c * chunk + chunk - 1
                    val page = table[(r / (rows / tables)) * width + pos / bs]
                    (0 until heads).any { h -> (0 until d).sumOf { j -> q[(r * heads + h) * d + j].toDouble() * keys[(page * bs + pos % bs) * d + j] } > 0 }
                }
            }
            println("[fused-indexer] rows $rows, tables $tables, $nc chunks, top_k $topK: kept per row $kept, scoring above 0 ${positive.take(8)}")
            assertContentEquals(want, xla, "XLA's form")
            assertContentEquals(want, got, "the kernel")
        }
    }

    @Test
    fun theIndexerKernelKeepsTheReferencesChunks() {
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so (triton/build_backend.sh)")
        // Short and long rows, partial tails, an empty one.
        topk(rows = 8, tables = 2, width = 16, topK = 20, lens = intArrayOf(0, 7, 79, 80, 81, 200, 255, 256), seed = 1, zeros = false)
        topk(rows = 8, tables = 2, width = 16, topK = 20, lens = intArrayOf(3, 70, 81, 150, 199, 210, 250, 256), seed = 2, zeros = true)
        // A radix select over 2,048 chunks keeping 512, verify rows of 4 per table.
        topk(rows = 8, tables = 2, width = 512, topK = 512, lens = IntArray(8) { if (it < 4) 8186 + it else 3000 + it }, seed = 3, zeros = false)
        topk(rows = 8, tables = 2, width = 512, topK = 512, lens = IntArray(8) { if (it < 4) 8186 + it else 3000 + it }, seed = 4, zeros = true)
        // Prefill rows: 64 consecutive positions of one table.
        topk(rows = 64, tables = 1, width = 256, topK = 300, lens = IntArray(64) { 3000 + it }, seed = 5, zeros = false)
    }

    private fun attnFn(rows: Int, tables: Int, heads: Int, kvHeads: Int, d: Int, bs: Int, width: Int, numBlocks: Int, nc: Int, pool: DType, fused: Boolean, f16: Boolean): DxirFunction =
        DxirBuilder.function("attn") {
            val qT = DxirType(F32, listOf(rows, heads, d))
            val poolT = DxirType(pool, listOf(numBlocks, bs, kvHeads, d))
            val q = param("q", qT)
            val k = param("k", poolT)
            val v = param("v", poolT)
            val t = param("t", DxirType(I32, listOf(tables, width)))
            val l = param("l", DxirType(I32, listOf(rows)))
            val keep = param("keep", DxirType(I32, listOf(rows, nc)))
            val attrs = buildMap<String, Any> {
                put("scale", 0.0625)
                if (fused) put(PagedAttentionAttrs.FUSED_KERNEL, true)
                if (f16) put(PagedAttentionAttrs.PREFILL_F16, true)
            }
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l, keep), qT, attrs))
        }

    private fun attn(rows: Int, tables: Int, lens: IntArray, pool: DType, seed: Int, f16: Boolean = false, heads: Int = 24, kvHeads: Int = 2, d: Int = 256) {
        val bs = 16; val width = (lens.max() + bs - 1) / bs + 2; val numBlocks = 1 + tables * width; val chunk = 4
        val nc = width * bs / chunk
        val fused = attnFn(rows, tables, heads, kvHeads, d, bs, width, numBlocks, nc, pool, true, f16)
        val plain = attnFn(rows, tables, heads, kvHeads, d, bs, width, numBlocks, nc, pool, false, false)
        assertTrue("tlaloc_paged_attention" in fused.toStablehlo(), "the fused form is emitted")
        val rnd = Random(seed)
        val n = numBlocks * bs * kvHeads * d
        val q = FloatArray(rows * heads * d) { rnd.nextFloat() * 2 - 1 }
        val kCodes = ByteArray(n) { (rnd.nextInt(0x60) or (rnd.nextInt(2) shl 7)).toByte() }
        val vCodes = ByteArray(n) { (rnd.nextInt(0x60) or (rnd.nextInt(2) shl 7)).toByte() }
        val kF = FloatArray(n) { if (pool == F32) rnd.nextFloat() * 4 - 2 else f8e4m3fnToFloat(kCodes[it]) }
        val vF = FloatArray(n) { if (pool == F32) rnd.nextFloat() * 2 - 1 else f8e4m3fnToFloat(vCodes[it]) }
        val table = IntArray(tables * width) { 1 + (it * 7) % (numBlocks - 1) }
        // Runs of kept chunks (as a selection keeps neighbours), a few kept alone, and the chunk of each row's last position.
        val keep = IntArray(rows * nc)
        for (r in 0 until rows) {
            var c = 0
            while (c < nc) {
                if (rnd.nextInt(10) == 0) { val run = 1 + rnd.nextInt(24); for (j in c until minOf(nc, c + run)) keep[r * nc + j] = 1; c += run } else c++
            }
            if (lens[r] > 0) keep[r * nc + (lens[r] - 1) / chunk] = 1
        }
        val ins = listOf(q, kF, vF, FloatArray(table.size) { table[it].toFloat() }, FloatArray(rows) { lens[it].toFloat() }, FloatArray(keep.size) { keep[it].toFloat() })
        val want = DxirInterpreter.evalFunction(plain, ins)[0]
        TestBackend.session().use { s ->
            fun pools(): List<Any> = if (pool == F32) listOf(kF, vF) else listOf(kCodes, vCodes)
            val got = s.runOnHost(fused, listOf<Any>(q) + pools() + listOf(table, lens, keep)).single() as FloatArray
            val xla = s.runOnHost(plain, listOf<Any>(q) + pools() + listOf(table, lens, keep)).single() as FloatArray
            var worst = 0f; var worstXla = 0f; var mag = 0f
            for (i in want.indices) {
                worst = maxOf(worst, abs(got[i] - want[i])); worstXla = maxOf(worstXla, abs(got[i] - xla[i])); mag = maxOf(mag, abs(want[i]))
            }
            val tol = if (f16) 3e-3f * mag else maxOf(1e-4f, 1e-5f * mag)
            val share = keep.count { it == 1 } * 100 / keep.size
            println(
                "[fused-sparse] rows=$rows tables=$tables heads=$heads/$kvHeads D=$d pool=${pool.name}${if (f16) " f16" else ""} " +
                    "lens ${lens.min()}..${lens.max()}, $share% of chunks kept: worst |d| $worst against the interpreter, $worstXla against XLA's form (largest |y| $mag)",
            )
            assertTrue(worst <= tol && worstXla <= tol, "fused kernel differs: $worst / $worstXla of $mag")
        }
    }

    @Test
    fun maskedAttentionKernelsAttendAsTheInterpreter() {
        assumeTrue(registered(), "no CUDA device or no libtlaloc_kernels.so (triton/build_backend.sh)")
        // Decode rows: the CUDA-core slice kernel (f32 pools) and the tensor-core one (e4m3fn).
        attn(rows = 3, tables = 3, lens = intArrayOf(5, 2100, 3001), pool = F32, seed = 1, heads = 6, d = 64)
        attn(rows = 3, tables = 3, lens = intArrayOf(9, 2100, 3001), pool = F8E4M3FN, seed = 2)
        // Verify rows: 4 tokens per table (48 queries per KV head) and the MTP head's 5 (60), the split form.
        attn(rows = 8, tables = 2, lens = intArrayOf(3000, 3001, 3002, 3003, 2500, 2501, 2502, 2503), pool = F8E4M3FN, seed = 3)
        attn(rows = 10, tables = 2, lens = IntArray(10) { if (it < 5) 4000 + it else 2100 + it }, pool = F8E4M3FN, seed = 4)
        // Prefill rows: 40 consecutive tokens per table, exact and f16.
        val pre = IntArray(80) { if (it < 40) 2900 + it + 1 else 2060 + (it - 40) + 1 }
        attn(rows = 80, tables = 2, lens = pre, pool = F8E4M3FN, seed = 5)
        attn(rows = 80, tables = 2, lens = pre, pool = F8E4M3FN, seed = 5, f16 = true)
    }
}
