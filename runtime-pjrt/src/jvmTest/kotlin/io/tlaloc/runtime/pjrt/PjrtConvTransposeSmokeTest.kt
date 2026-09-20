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
 * §0.4.393 — the two emitter gaps that were left, certified against real XLA on the
 * GB10 with the interpreter as the oracle.
 *
 * 1. The fused TRANSPOSED-conv adjoints. The interpreter and host twins invert the
 *    primal's tap equation per element, which StableHLO has no primitive for, so the
 *    emitter goes through the identity `convT(x, w) ≡ conv(dilate(x, L), swap01(w))`
 *    and composes pieces that already existed: an interior `stablehlo.pad` to dilate,
 *    the §0.4.385 conv-adjoint expansions, a strided `stablehlo.slice` to undilate,
 *    and channel swaps. That is a lot of machinery assembled from an algebraic
 *    identity — exactly the kind of thing only a real backend can confirm, since a
 *    wrong padding solve or a transposed slice still produces well-typed MLIR.
 *
 * 2. `SIGN`, which was the one user-reachable kind with no emitter arm: `ops.sign`
 *    is in the FIR unary map AND the MAX/MIN reduction rule builds its extremum
 *    indicator as `1 - sign(y - x)`, so `grad { x.max(1).sum() }` lowered and
 *    synthesised fine but could not be emitted. Pinned here through the reduction
 *    gradient rather than a bare `sign` call, because that is the path a user hits.
 */
class PjrtConvTransposeSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    /**
     * A transposed conv with the attrs that make the identity non-trivial:
     * `lhs_dilation` 2 (so `dilate`/`undilate` actually do something), symmetric
     * padding, x [1,2,3,3] against an IOHW kernel [2,3,2,2].
     * hDil = (3−1)·2+1 = 5, kEff = 2, so y = (5+1+1−2)/1+1 = 6 → [1,3,6,6].
     * `Σ y²` rather than `Σ y` so the adjoint's upstream is non-uniform.
     */
    private fun convTransposeLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(1, 2, 3, 3))
        val wT = DxirType(F32, listOf(2, 3, 2, 2))
        val yT = DxirType(F32, listOf(1, 3, 6, 6))
        return DxirBuilder.function("convT_loss") {
            val x = param("x", xT)
            val w = param("w", wT)
            val y = op(
                OpKind.CONV_TRANSPOSE2D, listOf(x, w), yT,
                attrs = mapOf(
                    "window_strides" to listOf(1, 1),
                    "lhs_dilation" to listOf(2, 2),
                    "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                ),
            )
            val y2 = op(OpKind.MUL, listOf(y, y), yT)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }
    }

    /** `Σ max(x, axis=1)` — its gradient is the SIGN-bearing extremum indicator. */
    private fun maxReductionLossFn(): DxirFunction {
        val xT = DxirType(F32, listOf(2, 3))
        val yT = DxirType(F32, listOf(2, 1))
        return DxirBuilder.function("max_loss") {
            val x = param("x", xT)
            val y = op(OpKind.MAX, listOf(x), yT, attrs = mapOf("reduction_dims" to listOf(1)))
            listOf(op(OpKind.SUM, listOf(y), scalar))
        }
    }

    private fun assertGpuMatchesInterpreter(fn: DxirFunction, inputs: List<FloatArray>, tag: String) {
        val grad = DxirReverseTransform.apply(fn)
        val wantLoss = DxirInterpreter.evalFunction(fn, inputs).single().single()
        val wantGrads = DxirInterpreter.evalFunction(grad, inputs)

        val gotLoss: Float
        val gotGrads: List<FloatArray>
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            gotLoss = session.runOn(fn, inputs).single().single()
            gotGrads = session.runOn(grad, inputs)
        }

        val lossDiff = abs(gotLoss - wantLoss)
        var gradDiff = 0f
        for (r in wantGrads.indices) {
            for (i in wantGrads[r].indices) {
                gradDiff = maxOf(gradDiff, abs(gotGrads[r][i] - wantGrads[r][i]))
            }
        }
        println("[$tag] loss |diff|=$lossDiff, grads max|diff|=$gradDiff on GB10 vs interpreter")
        assertTrue(lossDiff <= 1e-4f * maxOf(1f, abs(wantLoss)), "$tag loss diverges: $lossDiff")
        assertTrue(gradDiff <= 1e-4f, "$tag gradients diverge from interpreter: $gradDiff")
    }

    @Test
    fun convTransposeLossAndGradientRunOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val rng = java.util.Random(7)
        assertGpuMatchesInterpreter(
            convTransposeLossFn(),
            listOf(
                FloatArray(1 * 2 * 3 * 3) { rng.nextFloat() - 0.5f },
                FloatArray(2 * 3 * 2 * 2) { rng.nextFloat() - 0.5f },
            ),
            "pjrt-convT",
        )
    }

    @Test
    fun maxReductionGradientUsingSignRunsOnGpuAndMatchInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        // Distinct values per row, so the argmax is unique and the indicator is
        // unambiguous — this test is about SIGN emitting at all, not about tie policy
        // (which `DxirHostConvParityTest` pins for pooling).
        assertGpuMatchesInterpreter(
            maxReductionLossFn(),
            listOf(floatArrayOf(0.25f, -1.5f, 0.75f, 2.0f, -0.5f, 1.25f)),
            "pjrt-sign",
        )
    }
}
