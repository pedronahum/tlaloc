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
 * §0.4.395 — the Phase C2 trig tails certified against real XLA: the first
 * `stablehlo.tan` and `stablehlo.atan2` (the atan(x) = atan2(x, 1) spelling)
 * emissions compile and execute on the GB10, forwards AND gradient graphs
 * (TanRule's `1 + tan²` recompute, AtanRule's `1/(1 + x²)`), agreeing with
 * the interpreter. This is the validation the emitter arm's `stablehlo.tan`
 * choice rests on — if a future plugin rejects the op, this test is the
 * tripwire.
 */
class PjrtTanAtanSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun tanAtanForwardAndGradientsRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val xT = DxirType(F32, listOf(2, 3))
        fun loss(name: String, kind: OpKind) = DxirBuilder.function(name) {
            val x = param("x", xT)
            val y = op(kind, listOf(x), xT)
            val y2 = op(OpKind.MUL, listOf(y, y), xT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }

        // tan probes stay inside (−π/2, π/2) — near a pole the f32 GPU result
        // and the double-accumulating interpreter would diverge for reasons
        // that are rounding, not emission.
        val x = floatArrayOf(0.3f, -0.7f, 1.1f, 0.0f, -1.3f, 0.9f)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            for (fn in listOf(loss("tan_loss", OpKind.TAN), loss("atan_loss", OpKind.ATAN))) {
                val grad = DxirReverseTransform.apply(fn)
                val wantLoss = DxirInterpreter.evalFunction(fn, listOf(x)).single().single()
                val wantGrad = DxirInterpreter.evalFunction(grad, listOf(x)).single()
                val gotLoss = session.runOn(fn, listOf(x)).single().single()
                val gotGrad = session.runOn(grad, listOf(x)).single()

                val lossDiff = abs(gotLoss - wantLoss) / maxOf(1f, abs(wantLoss))
                var gradDiff = 0f
                for (i in wantGrad.indices) gradDiff = maxOf(gradDiff, abs(gotGrad[i] - wantGrad[i]))
                println("[pjrt-trig] ${fn.name}: loss rel|diff|=$lossDiff, grad max|diff|=$gradDiff on GB10")
                assertTrue(lossDiff <= 1e-5f, "${fn.name} loss diverges: $lossDiff")
                assertTrue(gradDiff <= 1e-3f, "${fn.name} gradient diverges from interpreter: $gradDiff")
            }
        }
    }
}
