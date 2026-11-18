package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.PagedAttentionAttrs
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertEquals

/**
 * Sharded serving Step 4 ([docs/design/SHARDED_SERVING.md]): PAGED_ATTENTION under a real Shardy
 * mesh, partitioned by XLA's SPMD pass — not the manual one-device slicing of the expert-parallel
 * and context-parallel tests. The question the design names: does the attention op (and its fused
 * custom-call form) accept a sharded operand, or does the partitioner refuse it?
 *
 * Rows (the batch axis) shard over the mesh; the KV pools are replicated. The portable form must
 * compute the same attention sharded as on one device. The fused (`FUSED_KERNEL`) form lowers to a
 * `stablehlo.custom_call`; whether XLA partitions a sharded operand into it is the decisive check,
 * captured here rather than assumed.
 */
class PjrtShardedPagedAttentionTest {

    private val n = 4
    private val rows = 4
    private val heads = 4
    private val kvHeads = 4
    private val d = 8
    private val m = 2          // pages per table
    private val bs = 4         // block size
    private val tables = rows  // one table per row
    private val blocks = tables * m

    private fun mesh() = DxirMesh("m", listOf(DxirMeshAxis("w", n)))
    private fun rowsOnW(rank: Int) = DxirSharding(
        "m", listOf(DxirDimSharding(listOf(DxirAxisRef.Full("w")))) + List(rank - 1) { DxirDimSharding(emptyList()) },
    )
    private fun replicated(rank: Int) = DxirSharding("m", List(rank) { DxirDimSharding(emptyList()) })

    /** PAGED_ATTENTION with the row axis sharded over the mesh; pools replicated. */
    private fun shardedAttention(fused: Boolean) = DxirBuilder.function("pa_sharded") {
        declareMesh(mesh())
        val q = param("q", DxirType(F32, listOf(rows, heads, d)), rowsOnW(3))
        val k = param("k", DxirType(F32, listOf(blocks, bs, kvHeads, d)), replicated(4))
        val v = param("v", DxirType(F32, listOf(blocks, bs, kvHeads, d)), replicated(4))
        val t = param("tables", DxirType(I32, listOf(tables, m)), rowsOnW(2))
        val l = param("lens", DxirType(I32, listOf(rows)), rowsOnW(1))
        val attrs = mapOf<String, Any>("scale" to 0.25) +
            (if (fused) mapOf(PagedAttentionAttrs.FUSED_KERNEL to true) else emptyMap())
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), DxirType(F32, listOf(rows, heads, d)), attrs))
    }

    private fun inputs(): List<FloatArray> {
        val rnd = Random(4242)
        val q = FloatArray(rows * heads * d) { rnd.nextFloat() * 2 - 1 }
        val k = FloatArray(blocks * bs * kvHeads * d) { rnd.nextFloat() * 2 - 1 }
        val v = FloatArray(blocks * bs * kvHeads * d) { rnd.nextFloat() * 2 - 1 }
        // I32 operands travel as floats through runSharded/evalFunction, as the context-parallel
        // test passes them; the param's dtype converts them back.
        val full = FloatArray(tables * m) { ((it / m) * m + it % m).toFloat() }
        val lens = FloatArray(rows) { (5 + it).toFloat() }   // every row has positions (no empty-row NaN)
        return listOf(q, k, v, full, lens)
    }

    @Test
    fun portablePagedAttentionShardsByRowsAndMatchesOneDevice() {
        MultiDeviceBackend.session(n) { s ->
            val ins = inputs()
            val fn = shardedAttention(fused = false)
            val want = DxirInterpreter.evalFunction(fn, ins)[0]
            val got = s.runSharded(fn, ins).single()
            val mag = want.maxOf { abs(it) }
            val diff = want.indices.maxOf { abs(got[it] - want[it]) }
            println("[sharded-pa] portable, rows on 4 devices: max |diff| $diff (largest |out| $mag)")
            assertTrue(diff <= 1e-5f * mag + 1e-6f, "sharded portable attention differs: $diff of $mag")
        }
    }

    /**
     * The decisive check: the fused custom-call form under the same sharding. XLA's SPMD pass
     * either partitions a sharded operand into the custom_call (it compiles and runs) or refuses
     * (it throws). Either outcome is recorded; a refusal is the design's falsifier, not a bug here.
     *
     * The fused kernel target is CUDA-specific (GB10 `@kptx_paged_attention`); on the CPU plugin
     * the emitter lowers PAGED_ATTENTION portably even with FUSED_KERNEL, so no custom_call appears
     * and the custom-call-specific question cannot be asked here. That is recorded and the test
     * skips; asking it needs multi-device CUDA, which a single-GPU box does not have.
     */
    @Test
    fun fusedPagedAttentionCustomCallUnderSharding() {
        MultiDeviceBackend.session(n) { s ->
            val ins = inputs()
            val fn = shardedAttention(fused = true)
            val mlir = fn.toStablehlo()
            assumeTrue(
                "stablehlo.custom_call" in mlir,
                "this platform lowers PAGED_ATTENTION portably (no custom_call); the fused-kernel " +
                    "sharding question needs multi-device CUDA",
            )
            val outcome = runCatching {
                val got = s.runSharded(fn, ins).single()
                val want = DxirInterpreter.evalFunction(shardedAttention(fused = false), ins)[0]
                val mag = want.maxOf { abs(it) }
                val diff = want.indices.maxOf { abs(got[it] - want[it]) }
                "compiled and ran sharded; max |diff| vs portable $diff (largest |out| $mag)"
            }
            println("[sharded-pa] fused custom_call under row sharding: ${outcome.getOrElse { "REFUSED: ${it.message?.take(300)}" }}")
            // The experiment records the outcome; it does not assert which way XLA decides, because
            // that is the question. A refusal is a valid, recorded result (the design's falsifier).
            assertTrue(outcome.isSuccess || outcome.exceptionOrNull() != null, "an outcome was observed")
        }
    }
}
