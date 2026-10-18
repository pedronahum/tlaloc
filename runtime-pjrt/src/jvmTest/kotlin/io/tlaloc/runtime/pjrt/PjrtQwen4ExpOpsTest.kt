package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.IndexerTopkAttrs
import io.tlaloc.ir.NgramHashAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.inference.PleConfig
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * The emitted forms of Qwen4-Exp's ops on the device against their references:
 * NGRAM_HASH at Qwen3.8-Flash-Next's own hash constants (ids up to 3.2e8, past
 * what the interpreter's floats hold), INDEXER_TOPK's keep masks, and
 * PAGED_ATTENTION with a keep operand in its per-row and shared-table forms.
 */
class PjrtQwen4ExpOpsTest {

    @Test
    fun ngramHashMatchesTheReferenceAtFlashNextsConstants() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val ple = PleConfig(
            layers = listOf(1), embedDim = 2560, convKernel = 4, ngramSize = 3, headsPerNgram = 8, eosTokenId = 248044,
            vocabBase = 20_000_000L, vocabDivisor = 128, seed = 1234L, unigramVocab = 248320,
        )
        val n = 256
        val attrs = mapOf("multipliers" to ple.multipliers(0), "orders" to ple.orders, "moduli" to ple.moduli(0), "offsets" to ple.offsets(0))
        val fn = DxirBuilder.function("hash") {
            val t = param("tokens", DxirType(I32, listOf(n, 3)))
            listOf(op(OpKind.NGRAM_HASH, listOf(t), DxirType(I32, listOf(n, ple.heads)), attrs))
        }
        val rnd = Random(7)
        val tokens = IntArray(3 * n) { rnd.nextInt(248320) }
        val want = NgramHashAttrs.reference(NgramHashAttrs.parse(fn.returns[0] as io.tlaloc.ir.DxirOp, "test"), tokens)
        val got = TestBackend.session().use { s -> s.runOnHost(fn, listOf(tokens))[0] as IntArray }
        assertContentEquals(want, got)
        assertTrue(want.max() > (1 shl 24), "the ids reach past 2^24: ${want.max()}")
        println("[pjrt-ngram-hash] $n rows x ${ple.heads} heads == reference, ids up to ${want.max()}")
    }

    @Test
    fun indexerTopkKeepsTheReferencesChunks() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val rows = 8; val heads = 4; val d = 16; val bs = 8; val pages = 5; val tablesN = 2; val chunk = 4; val topK = 3
        val numBlocks = 1 + tablesN * pages
        val ctx = pages * bs
        val fn = DxirBuilder.function("topk") {
            val q = param("q", DxirType(F32, listOf(rows, heads, d)))
            val k = param("keys", DxirType(F32, listOf(numBlocks, bs, 1, d)))
            val t = param("tables", DxirType(I32, listOf(tablesN, pages)))
            val l = param("lens", DxirType(I32, listOf(rows)))
            listOf(
                op(
                    OpKind.INDEXER_TOPK, listOf(q, k, t, l), DxirType(I32, listOf(rows, ctx / chunk)),
                    mapOf("chunk" to chunk, "top_k" to topK, "scale" to 0.25),
                ),
            )
        }
        val rnd = Random(11)
        val q = FloatArray(rows * heads * d) { rnd.nextFloat() - 0.5f }
        val keys = FloatArray(numBlocks * bs * d) { rnd.nextFloat() - 0.5f }
        val tables = IntArray(tablesN * pages) { 1 + it }
        // Short rows (fewer complete chunks than top_k, an empty one), long ones, partial tails.
        val lens = intArrayOf(1, 7, 12, 13, 22, 29, 36, 40)
        val want = IndexerTopkAttrs.reference(IndexerTopkAttrs.parse(fn.returns[0] as io.tlaloc.ir.DxirOp, "test"), q, keys, tables, lens)
        val interp = DxirInterpreter.evalFunction(fn, listOf(q, keys, FloatArray(tables.size) { tables[it].toFloat() }, FloatArray(rows) { lens[it].toFloat() }))[0]
        assertContentEquals(want, IntArray(interp.size) { interp[it].toInt() })
        val got = TestBackend.session().use { s -> s.runOnHost(fn, listOf(q, keys, tables, lens))[0] as IntArray }
        assertContentEquals(want, got)
        println("[pjrt-indexer-topk] keep masks == reference for lengths ${lens.toList()}")
    }

    private fun maskedAttention(rows: Int, tablesN: Int) {
        val heads = 4; val kvHeads = 2; val d = 16; val bs = 4; val pages = 6; val nc = pages * bs / 4
        val numBlocks = 1 + tablesN * pages
        val fn = DxirBuilder.function("att") {
            val q = param("q", DxirType(F32, listOf(rows, heads, d)))
            val k = param("k", DxirType(F32, listOf(numBlocks, bs, kvHeads, d)))
            val v = param("v", DxirType(F32, listOf(numBlocks, bs, kvHeads, d)))
            val t = param("tables", DxirType(I32, listOf(tablesN, pages)))
            val l = param("lens", DxirType(I32, listOf(rows)))
            val keep = param("keep", DxirType(I32, listOf(rows, nc)))
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l, keep), DxirType(F32, listOf(rows, heads, d)), mapOf("scale" to 0.25)))
        }
        val rnd = Random(rows)
        val lens = IntArray(rows) { 1 + rnd.nextInt(pages * bs) }
        // Keep a random half of the chunks and always the one holding a row's last position.
        val keep = IntArray(rows * nc) { i -> if ((i % nc) == (lens[i / nc] - 1) / 4 || rnd.nextBoolean()) 1 else 0 }
        val ins = listOf(
            FloatArray(rows * heads * d) { rnd.nextFloat() - 0.5f },
            FloatArray(numBlocks * bs * kvHeads * d) { rnd.nextFloat() - 0.5f },
            FloatArray(numBlocks * bs * kvHeads * d) { rnd.nextFloat() - 0.5f },
            FloatArray(tablesN * pages) { (1 + it).toFloat() },
            FloatArray(rows) { lens[it].toFloat() },
            FloatArray(rows * nc) { keep[it].toFloat() },
        )
        val want = DxirInterpreter.evalFunction(fn, ins)[0]
        // Dropping the keep operand changes the rows: the mask is not a no-op on these inputs.
        val unmasked = DxirInterpreter.evalFunction(
            DxirBuilder.function("att") {
                val ps = fn.params.dropLast(1).map { param(it.name, it.type) }
                listOf(op(OpKind.PAGED_ATTENTION, ps, DxirType(F32, listOf(rows, heads, d)), mapOf("scale" to 0.25)))
            },
            ins.dropLast(1),
        )[0]
        assertTrue(want.indices.any { abs(want[it] - unmasked[it]) > 1e-3f })
        val got = TestBackend.session().use { s -> s.runOn(fn, ins)[0] }
        val worst = want.indices.maxOf { abs(got[it] - want[it]) }
        println("[pjrt-masked-attention] rows $rows, tables $tablesN: worst |difference| $worst")
        assertTrue(worst <= 1e-5f, "worst $worst")
    }

    @Test
    fun pagedAttentionWithAKeepOperandMatchesTheInterpreter() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        maskedAttention(rows = 6, tablesN = 6)
        maskedAttention(rows = 12, tablesN = 2)
    }
}
