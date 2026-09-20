package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.362 — CONV2D certified against real XLA: the general-attr conv
 * loss (strides + asymmetric padding + rhs_dilation) and its full
 * [Conv2dRule] gradient graph — CONV_TRANSPOSE2D with `window_reversal`
 * and `lhs_dilation` (first use of the emitter's `reverse` window
 * field), batch↔feature transposes, remainder-cropping negative
 * padding — compile through StableHLO and execute on the GB10,
 * agreeing with the interpreter. XLA is the independent oracle for
 * both the interpreter's conv semantics and the emitter's spelling.
 */
class PjrtConvSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    private fun convLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(2, 2, 5, 4))
        val wT = DxirType(F32, listOf(3, 2, 3, 2))
        val yT = DxirType(F32, listOf(2, 3, 2, 4))
        return DxirBuilder.function("conv_loss") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(
                OpKind.CONV2D, listOf(x, w), yT,
                attrs = mapOf(
                    "window_strides" to listOf(2, 1),
                    "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                    "rhs_dilation" to listOf(1, 2),
                ),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    @Test
    fun convLossAndGradientRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val fn = convLossFn()
        val grad = DxirReverseTransform.apply(fn)

        val rng = java.util.Random(42)
        val x = FloatArray(2 * 2 * 5 * 4) { rng.nextFloat() - 0.5f }
        val w = FloatArray(3 * 2 * 3 * 2) { rng.nextFloat() - 0.5f }

        val wantLoss = DxirInterpreter.evalFunction(fn, listOf(x, w)).single().single()
        val wantGrads = DxirInterpreter.evalFunction(grad, listOf(x, w))

        val gotLoss: Float
        val gotGrads: List<FloatArray>
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            gotLoss = session.runOn(fn, listOf(x, w)).single().single()
            gotGrads = session.runOn(grad, listOf(x, w))
        }

        val lossDiff = abs(gotLoss - wantLoss)
        var gradDiff = 0f
        for (r in wantGrads.indices) {
            for (i in wantGrads[r].indices) {
                gradDiff = maxOf(gradDiff, abs(gotGrads[r][i] - wantGrads[r][i]))
            }
        }
        println("[pjrt-conv] loss |diff|=$lossDiff, grads max|diff|=$gradDiff on GB10 vs interpreter")
        assertTrue(lossDiff <= 1e-4f * maxOf(1f, abs(wantLoss)), "conv loss diverges: $lossDiff")
        assertTrue(gradDiff <= 1e-4f, "conv gradients diverge from interpreter: $gradDiff")
    }

    /**
     * §0.4.429 — grouped conv against real XLA: `feature_group_count = 2` rides
     * onto the emitted convolution, and XLA is the independent oracle for the
     * interpreter's new per-group semantics. The FORWARD-mode tangent graph runs
     * too — every op in it is a primal grouped convolution (the bilinear product
     * rule), which is exactly the emission surface; reverse mode stays off the
     * GPU here because grouped ADJOINT emission is that section's named deferral.
     */
    @Test
    fun groupedConvForwardAndJvpRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val xT = DxirType(F32, listOf(1, 4, 5, 4))
        val wT = DxirType(F32, listOf(6, 2, 3, 2))
        val yT = DxirType(F32, listOf(1, 6, 2, 4))
        val fn = DxirBuilder.function("grouped_conv_loss") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(
                OpKind.CONV2D, listOf(x, w), yT,
                attrs = mapOf(
                    "window_strides" to listOf(2, 1),
                    "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                    "rhs_dilation" to listOf(1, 2),
                    "feature_group_count" to 2,
                ),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
        val fwd = io.tlaloc.ir.passes.DxirForwardTransform.apply(fn)

        val rng = java.util.Random(429)
        val x = FloatArray(1 * 4 * 5 * 4) { rng.nextFloat() - 0.5f }
        val w = FloatArray(6 * 2 * 3 * 2) { rng.nextFloat() - 0.5f }
        val vx = FloatArray(x.size) { rng.nextFloat() - 0.5f }
        val vw = FloatArray(w.size) { rng.nextFloat() - 0.5f }

        val wantLoss = DxirInterpreter.evalFunction(fn, listOf(x, w)).single().single()
        val wantTangent =
            DxirInterpreter.evalFunction(fwd, listOf(x, w, vx, vw)).last().single()

        val gotLoss: Float
        val gotTangent: Float
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            gotLoss = session.runOn(fn, listOf(x, w)).single().single()
            gotTangent = session.runOn(fwd, listOf(x, w, vx, vw)).last().single()
        }

        val lossDiff = abs(gotLoss - wantLoss)
        val tanDiff = abs(gotTangent - wantTangent)
        println("[pjrt-grouped-conv] loss |diff|=$lossDiff, jvp |diff|=$tanDiff on GB10 vs interpreter")
        assertTrue(lossDiff <= 1e-4f * maxOf(1f, abs(wantLoss)), "grouped conv loss diverges: $lossDiff")
        assertTrue(
            tanDiff <= 1e-3f * maxOf(1f, abs(wantTangent)),
            "grouped conv jvp diverges from interpreter: $tanDiff",
        )
    }
}
