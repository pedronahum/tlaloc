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
 * §0.4.402 — the Phase C1 special functions certified against real XLA: the
 * first CHLO emissions (`chlo.lgamma`, `chlo.digamma`, and — inside the
 * digamma GRADIENT graph — `"chlo.polygamma"(splat 1.0, x)` for the internal
 * TRIGAMMA op) compile and execute on the GB10, forwards AND gradient graphs,
 * agreeing with the interpreter's shared-Double-kernel arms. This is the
 * validation the emitter's CHLO choice rests on: XLA's PJRT compile path
 * parses the CHLO dialect and legalizes it before StableHLO→HLO — if a future
 * plugin build drops that, this test is the tripwire (fallback then: a
 * host-side Lanczos/series decomposition or a loud emit-time refusal).
 */
class PjrtLgammaDigammaSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun lgammaDigammaForwardAndGradientsRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val xT = DxirType(F32, listOf(2, 3))
        fun loss(name: String, kind: OpKind) = DxirBuilder.function(name) {
            val x = param("x", xT)
            val y = op(kind, listOf(x), xT)
            val y2 = op(OpKind.MUL, listOf(y, y), xT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }

        // Probes stay on the positive axis, off the poles at 0, −1, −2, … —
        // near a pole the adjoint explodes and f32 GPU vs double-through
        // interpreter would diverge for rounding reasons, not emission ones.
        val x = floatArrayOf(0.5f, 1.3f, 2.7f, 4.1f, 0.9f, 3.3f)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            for (fn in listOf(loss("lgamma_loss", OpKind.LGAMMA), loss("digamma_loss", OpKind.DIGAMMA))) {
                val grad = DxirReverseTransform.apply(fn)
                val wantLoss = DxirInterpreter.evalFunction(fn, listOf(x)).single().single()
                val wantGrad = DxirInterpreter.evalFunction(grad, listOf(x)).single()
                val gotLoss = session.runOn(fn, listOf(x)).single().single()
                val gotGrad = session.runOn(grad, listOf(x)).single()

                val lossDiff = abs(gotLoss - wantLoss) / maxOf(1f, abs(wantLoss))
                var gradDiff = 0f
                for (i in wantGrad.indices) gradDiff = maxOf(gradDiff, abs(gotGrad[i] - wantGrad[i]))
                println("[pjrt-special-fns] ${fn.name}: loss rel|diff|=$lossDiff, grad max|diff|=$gradDiff on GB10")
                assertTrue(lossDiff <= 1e-4f, "${fn.name} loss diverges: $lossDiff")
                assertTrue(gradDiff <= 2e-3f, "${fn.name} gradient diverges from interpreter: $gradDiff")
            }
        }
    }
}
