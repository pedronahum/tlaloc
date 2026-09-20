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
 * §0.4.405 — general POLYGAMMA(n) certified against real XLA: the order-splat
 * `"chlo.polygamma"(splat n.0, x)` emission compiles and executes on the GB10
 * at orders beyond the §0.4.402 trigamma spelling's n = 1 — the forward graphs
 * carry orders 1 (TRIGAMMA) and 2, and the gradient graphs climb to 2 and 3
 * through TrigammaRule/PolygammaRule — agreeing with the interpreter's
 * shared-Double-kernel arms. XLA legalizes polygamma via its own zeta-based
 * scheme, an implementation INDEPENDENT of the Bernoulli/reflection kernel in
 * `:core/SpecialFunctions.kt`, so the f32 agreement asserted here is a real
 * two-implementation cross-check, not a round-trip.
 */
class PjrtPolygammaSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun trigammaAndPolygammaForwardAndGradientsRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val xT = DxirType(F32, listOf(2, 3))
        fun loss(name: String, attrs: Map<String, Any>, kind: OpKind) = DxirBuilder.function(name) {
            val x = param("x", xT)
            val y = op(kind, listOf(x), xT, attrs = attrs)
            val y2 = op(OpKind.MUL, listOf(y, y), xT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }

        // Positive axis, off the poles (§0.4.395 discipline): near a pole the
        // higher-order ψ⁽ⁿ⁾ explode and f32-GPU vs double-through-interpreter
        // diverges for rounding reasons, not emission ones.
        val x = floatArrayOf(0.6f, 1.3f, 2.7f, 4.1f, 0.9f, 3.3f)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val cases = listOf(
                loss("trigamma_loss", emptyMap(), OpKind.TRIGAMMA),
                loss("polygamma2_loss", mapOf("order" to 2), OpKind.POLYGAMMA),
            )
            for (fn in cases) {
                val grad = DxirReverseTransform.apply(fn)
                val wantLoss = DxirInterpreter.evalFunction(fn, listOf(x)).single().single()
                val wantGrad = DxirInterpreter.evalFunction(grad, listOf(x)).single()
                val gotLoss = session.runOn(fn, listOf(x)).single().single()
                val gotGrad = session.runOn(grad, listOf(x)).single()

                val lossDiff = abs(gotLoss - wantLoss) / maxOf(1f, abs(wantLoss))
                var gradDiff = 0f
                for (i in wantGrad.indices) {
                    gradDiff = maxOf(gradDiff, abs(gotGrad[i] - wantGrad[i]) / maxOf(1f, abs(wantGrad[i])))
                }
                println("[pjrt-polygamma] ${fn.name}: loss rel|diff|=$lossDiff, grad max rel|diff|=$gradDiff on GB10")
                assertTrue(lossDiff <= 1e-4f, "${fn.name} loss diverges: $lossDiff")
                assertTrue(gradDiff <= 2e-3f, "${fn.name} gradient diverges from interpreter: $gradDiff")
            }
        }
    }
}
