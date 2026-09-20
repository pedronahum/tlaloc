package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.400 — EMBEDDING_GRAD's scatter+add-region emission certified against
 * real XLA: the first `stablehlo.scatter` with a genuine add region compiles
 * and executes on the GB10, and the collision-summing dTable agrees with the
 * interpreter. Since `stablehlo-translate` is not on this machine's PATH (the
 * round-trip suite skips), this is the emission's live parsing oracle — the
 * §0.4.396 flip-smoke convention. The index vector rides as an I32 const, so
 * the §0.4.400 integer dense-literal spelling is exercised too.
 */
class PjrtEmbeddingSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun embeddingForwardAndGradientRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        // loss = Σ embedding(table, idx) ⊙ w, idx = [0, 2, 0, 1]: slot 0 is
        // selected twice so the scatter's add region is load-bearing (a replace
        // body would silently drop one contribution).
        val fn = DxirBuilder.function("embed_loss") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(4, 2)))
            val idx = const(floatArrayOf(0f, 2f, 0f, 1f), DxirType(I32, listOf(4)))
            val emb = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 2)))
            val p = op(OpKind.MUL, listOf(emb, w), DxirType(F32, listOf(4, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }

        val table = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 1.1f, 0.9f, -0.3f, 0.5f)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val grad = DxirReverseTransform.apply(fn)
            val wantLoss = DxirInterpreter.evalFunction(fn, listOf(table, w)).single().single()
            val wantGrads = DxirInterpreter.evalFunction(grad, listOf(table, w))
            val gotLoss = session.runOn(fn, listOf(table, w)).single().single()
            val gotGrads = session.runOn(grad, listOf(table, w))

            val lossDiff = abs(gotLoss - wantLoss) / maxOf(1f, abs(wantLoss))
            var gradDiff = 0f
            for (r in wantGrads.indices) {
                for (i in wantGrads[r].indices) {
                    gradDiff = maxOf(gradDiff, abs(gotGrads[r][i] - wantGrads[r][i]))
                }
            }
            println("[pjrt-embedding] loss rel|diff|=$lossDiff, grad max|diff|=$gradDiff on GB10")
            assertTrue(lossDiff <= 1e-5f, "embedding loss diverges: $lossDiff")
            assertTrue(gradDiff <= 1e-5f, "embedding gradients diverge from interpreter: $gradDiff")
        }
    }

    @Test
    fun paddedEmbeddingGradientRunsOnGpuAndMatchesInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        // §0.4.409 — padding_index = 1 with idx = [0, 1, 0, 2]: XLA runs the
        // gather + compare/select mask on the primal AND the pre-scatter update
        // mask on the adjoint. The padded vocab row's gradient must come back
        // EXACTLY zero (a scatter of masked-to-zero rows over a splat-zero
        // base), and the slot-0 collision must still sum.
        val fn = DxirBuilder.function("embed_pad_loss") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(4, 2)))
            val idx = const(floatArrayOf(0f, 1f, 0f, 2f), DxirType(I32, listOf(4)))
            val emb = op(
                OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 2)),
                attrs = mapOf("padding_index" to 1),
            )
            val p = op(OpKind.MUL, listOf(emb, w), DxirType(F32, listOf(4, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }

        val table = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 1.1f, 0.9f, -0.3f, 0.5f)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val grad = DxirReverseTransform.apply(fn)
            val wantLoss = DxirInterpreter.evalFunction(fn, listOf(table, w)).single().single()
            val wantGrads = DxirInterpreter.evalFunction(grad, listOf(table, w))
            val gotLoss = session.runOn(fn, listOf(table, w)).single().single()
            val gotGrads = session.runOn(grad, listOf(table, w))

            val lossDiff = abs(gotLoss - wantLoss) / maxOf(1f, abs(wantLoss))
            var gradDiff = 0f
            for (r in wantGrads.indices) {
                for (i in wantGrads[r].indices) {
                    gradDiff = maxOf(gradDiff, abs(gotGrads[r][i] - wantGrads[r][i]))
                }
            }
            println("[pjrt-embedding-padded] loss rel|diff|=$lossDiff, grad max|diff|=$gradDiff on GB10")
            assertTrue(lossDiff <= 1e-5f, "padded embedding loss diverges: $lossDiff")
            assertTrue(gradDiff <= 1e-5f, "padded embedding gradients diverge from interpreter: $gradDiff")
            // The padded vocab row (row 1 of dTable) must be EXACT zeros on XLA too.
            assertTrue(
                gotGrads[0][2] == 0f && gotGrads[0][3] == 0f,
                "padded vocab row must be exactly zero on the GPU, got " +
                    "[${gotGrads[0][2]}, ${gotGrads[0][3]}]",
            )
        }
    }
}
