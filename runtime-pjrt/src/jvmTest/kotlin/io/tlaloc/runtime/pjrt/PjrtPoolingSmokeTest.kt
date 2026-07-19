package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
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
 * §0.4.363 — pooling certified against real XLA: MAXPOOL2D/AVGPOOL2D
 * forwards (first `stablehlo.reduce_window` emissions — max/−∞ and
 * add/0 + count_include_pad divide) and both gradient graphs — the
 * avgpool adjoint's splat-kernel transposed conv and the maxpool
 * adjoint's rank-6 upsample → EQ-mask → WHERE — compile and execute on
 * the GB10, agreeing with the interpreter.
 */
class PjrtPoolingSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun poolingForwardAndGradientsRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val xT = DxirType(F32, listOf(2, 3, 6, 6))
        val avgYT = DxirType(F32, listOf(2, 3, 3, 3))
        val avgLoss = DxirBuilder.function("avg_loss") {
            val x = param("x", xT)
            val y = op(
                OpKind.AVGPOOL2D, listOf(x), avgYT,
                attrs = mapOf(
                    "window" to listOf(3, 3),
                    "window_strides" to listOf(2, 2),
                    "padding" to listOf(listOf(1, 0), listOf(1, 0)),
                ),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), avgYT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
        val maxLoss = DxirBuilder.function("max_loss") {
            val x = param("x", xT)
            val y = op(OpKind.MAXPOOL2D, listOf(x), avgYT, attrs = mapOf("window" to listOf(2, 2)))
            val y2 = op(OpKind.MUL, listOf(y, y), avgYT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }

        // Distinct well-separated values: keeps the maxpool argmax unambiguous
        // so interpreter-vs-XLA agreement is exact rather than tie-dependent.
        val n = 2 * 3 * 6 * 6
        val perm = (0 until n).shuffled(kotlin.random.Random(9))
        val x = FloatArray(n) { perm[it] * 0.07f - 0.035f * n }

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            for (fn in listOf(avgLoss, maxLoss)) {
                val grad = DxirReverseTransform.apply(fn)
                val wantLoss = DxirInterpreter.evalFunction(fn, listOf(x)).single().single()
                val wantGrad = DxirInterpreter.evalFunction(grad, listOf(x)).single()
                val gotLoss = session.runOn(fn, listOf(x)).single().single()
                val gotGrad = session.runOn(grad, listOf(x)).single()

                val lossDiff = abs(gotLoss - wantLoss) / maxOf(1f, abs(wantLoss))
                var gradDiff = 0f
                for (i in wantGrad.indices) gradDiff = maxOf(gradDiff, abs(gotGrad[i] - wantGrad[i]))
                println("[pjrt-pool] ${fn.name}: loss rel|diff|=$lossDiff, grad max|diff|=$gradDiff on GB10")
                assertTrue(lossDiff <= 1e-5f, "${fn.name} loss diverges: $lossDiff")
                assertTrue(gradDiff <= 1e-3f, "${fn.name} gradient diverges from interpreter: $gradDiff")
            }
        }
    }
}
