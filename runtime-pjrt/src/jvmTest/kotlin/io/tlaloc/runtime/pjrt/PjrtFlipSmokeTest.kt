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
 * §0.4.396 — REVERSE (flip, Phase C3) certified against real XLA: the first
 * `stablehlo.reverse` emission compiles and executes on the GB10, forwards
 * AND gradient graph (the self-adjoint REVERSE-of-the-upstream), agreeing
 * with the interpreter. Since `stablehlo-translate` is not on this machine's
 * PATH (the round-trip suite skips), this is the emitter arm's live parsing
 * oracle — if a future plugin rejects the pretty `dims = […]` spelling, this
 * test is the tripwire.
 */
class PjrtFlipSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun flipForwardAndGradientRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val xT = DxirType(F32, listOf(2, 3))
        // loss = Σ (flip(x, [0,1]) ⊙ w)² with a baked non-uniform w so the
        // gradient's REVERSE actually has to move different values around.
        val fn = DxirBuilder.function("flip_loss") {
            val x = param("x", xT)
            val w = const(floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f, -1.1f, 0.6f), xT)
            val f = op(OpKind.REVERSE, listOf(x), xT, attrs = mapOf("dimensions" to listOf(0, 1)))
            val p = op(OpKind.MUL, listOf(f, w), xT)
            val p2 = op(OpKind.MUL, listOf(p, p), xT)
            listOf(op(OpKind.SUM, listOf(p2), scalar))
        }

        val x = floatArrayOf(0.3f, -0.7f, 1.1f, 0.0f, -1.3f, 0.9f)

        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            val grad = DxirReverseTransform.apply(fn)
            val wantLoss = DxirInterpreter.evalFunction(fn, listOf(x)).single().single()
            val wantGrad = DxirInterpreter.evalFunction(grad, listOf(x)).single()
            val gotLoss = session.runOn(fn, listOf(x)).single().single()
            val gotGrad = session.runOn(grad, listOf(x)).single()

            val lossDiff = abs(gotLoss - wantLoss) / maxOf(1f, abs(wantLoss))
            var gradDiff = 0f
            for (i in wantGrad.indices) gradDiff = maxOf(gradDiff, abs(gotGrad[i] - wantGrad[i]))
            println("[pjrt-flip] loss rel|diff|=$lossDiff, grad max|diff|=$gradDiff on GB10")
            assertTrue(lossDiff <= 1e-5f, "flip loss diverges: $lossDiff")
            assertTrue(gradDiff <= 1e-5f, "flip gradient diverges from interpreter: $gradDiff")
        }
    }
}
