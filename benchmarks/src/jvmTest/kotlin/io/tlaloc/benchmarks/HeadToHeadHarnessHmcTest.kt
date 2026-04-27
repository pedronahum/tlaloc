package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.226 — head-to-head harness Phase 1 fifth slice test. Validates the
 * [HmcLogisticRegressionHarness] inhabitant — the **fourth paper benchmark**
 * in the harness suite and the **first** to exercise EXP and LOG ops.
 *
 * **What this test pins**:
 *  - Forward value matches the Kotlin reference within 1% relative tolerance
 *    (straight-line dxir + EXP/LOG; the f32 noise comes mainly from EXP/LOG
 *    rounding in the Kotlin Math vs the dxir interpreter's path).
 *  - Per-input gradient FD-validated against central-difference on the Kotlin
 *    reference. Tighter tolerance than HookeanSpring because there's no
 *    compounded recurrence — straight-line chain of EXP/LOG/arithmetic.
 *  - Cross-input consistency at multiple β values.
 *  - Timing statistics positive and well-ordered.
 *
 * **The discriminator**: chain rule through EXP and LOG with shared operand
 * (the `xb` value flows into both `(y-1)*xb` AND `log(1 + exp(-xb))`). A
 * buggy AD that double-counted or dropped a partial would surface here.
 */
class HeadToHeadHarnessHmcTest {

    @Test
    fun hmcHarnessRunsAndProducesBaseline() {
        val result = HmcLogisticRegressionHarness.runBaseline(warmup = 50, measured = 100)

        assertEquals("hmc-logistic-regression-n4-d2", result.benchmark)
        assertEquals(50, result.warmupIterations)
        assertEquals(100, result.measuredIterations)

        val b0 = 0.5f
        val b1 = 0.3f

        // Forward: Kotlin reference.
        val expectedForward = BenchmarkPrimals.hmcLogisticRegressionReference(b0, b1)
        val forwardTol = abs(expectedForward) * 1e-2f + 1e-3f
        assertTrue(
            abs(result.forwardValue - expectedForward) < forwardTol,
            "expected forward ≈ $expectedForward, got ${result.forwardValue} (tol=$forwardTol)",
        )

        // 2 inputs → 2 gradients.
        assertEquals(2, result.gradientValues.size)

        // Per-input FD validation. Straight-line dxir, no compounded recurrence,
        // so 1.5% relative tolerance + 1e-3 absolute floor is appropriate
        // (slightly looser than BGDHyperOpt's 1% to accommodate EXP/LOG f32
        // path differences between Kotlin Math and dxir interpreter).
        val h = 1e-3f
        val fdGrads = floatArrayOf(
            (BenchmarkPrimals.hmcLogisticRegressionReference(b0 + h, b1)
                - BenchmarkPrimals.hmcLogisticRegressionReference(b0 - h, b1)) / (2f * h),
            (BenchmarkPrimals.hmcLogisticRegressionReference(b0, b1 + h)
                - BenchmarkPrimals.hmcLogisticRegressionReference(b0, b1 - h)) / (2f * h),
        )
        val labels = listOf("df/db0", "df/db1")
        for (i in 0 until 2) {
            val analGrad = result.gradientValues[i]
            val fdGrad = fdGrads[i]
            val tol = abs(fdGrad) * 0.015f + 1e-3f
            assertTrue(
                abs(analGrad - fdGrad) < tol,
                "${labels[i]}: analytical=$analGrad, FD=$fdGrad (tol=$tol)",
            )
        }

        // Both gradients should be non-zero at non-trivial β.
        assertTrue(abs(result.gradientValues[0]) > 1e-4f, "df/db0 should be non-zero")
        assertTrue(abs(result.gradientValues[1]) > 1e-4f, "df/db1 should be non-zero")

        // Timing sanity, mirrors prior harness tests.
        assertTrue(result.minNanos > 0)
        assertTrue(result.medianNanos >= result.minNanos)
        assertTrue(result.p99Nanos >= result.medianNanos)
        assertTrue(
            result.medianNanos < result.minNanos * 100,
            "median ${result.medianNanos}ns suspiciously larger than 100× min ${result.minNanos}ns",
        )

        println("[harness ${result.benchmark}] forward=${result.forwardValue} (ref=$expectedForward)")
        println("[harness ${result.benchmark}] gradients=${result.gradientValues}")
        println("[harness ${result.benchmark}] FD grads=${fdGrads.toList()}")
        println(
            "[harness ${result.benchmark}] grad eval timing (n=${result.measuredIterations}): " +
                "median=${result.medianNanos}ns min=${result.minNanos}ns p99=${result.p99Nanos}ns",
        )
    }

    @Test
    fun hmcForwardMatchesReferenceAcrossMultipleBeta() {
        // Cross-β consistency: at multiple (b0, b1) values, harness forward eval
        // matches the Kotlin reference. Catches input-routing or per-iteration
        // dataset-baking bugs.
        val primal = BenchmarkPrimals.hmcLogisticRegressionPrimal()

        val testCases = listOf(
            0.5f to 0.3f,       // baseline
            0.0f to 0.0f,       // origin: sum1 = 0 (since xb = 0); sum2 = 4·log(2);
                                //   prior = 0; result = 0 - 4·log(2) - 0 = -4·log(2)
            1.0f to -1.0f,      // perturbed sign-mixed
            -0.2f to 0.5f,      // small negative b0
        )

        for ((b0, b1) in testCases) {
            val expected = BenchmarkPrimals.hmcLogisticRegressionReference(b0, b1)
            val out = DxirInterpreter.evalFunction(
                primal,
                listOf(floatArrayOf(b0), floatArrayOf(b1)),
            )
            val tol = abs(expected) * 1e-2f + 1e-3f
            assertTrue(
                abs(out[0][0] - expected) < tol,
                "at (b0=$b0, b1=$b1): expected=$expected got=${out[0][0]} (tol=$tol)",
            )
        }
    }
}
