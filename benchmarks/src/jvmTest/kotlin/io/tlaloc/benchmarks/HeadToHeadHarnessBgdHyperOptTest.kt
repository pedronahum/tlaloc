package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.223 — head-to-head harness Phase 1 second slice test. Validates the
 * [BgdHyperOptHarness] inhabitant — the first **paper benchmark** in the
 * harness suite (BGDHyperOpt is OOPSLA 2021 paper Fig. 6).
 *
 * **What this test pins**:
 *  - Forward value matches the closed-form Kotlin reference within f32 tolerance.
 *  - Gradient w.r.t. `r` (the hyperparameter being optimised) agrees with
 *    central-difference FD on the Kotlin reference within 1% relative tolerance.
 *  - Other gradients (Sxy, Sx2, M) are non-zero (BGDHyperOpt's outer loop
 *    depends on all four inputs).
 *  - Timing statistics positive and well-ordered.
 *
 * **The discriminator**: ∂w/∂r is the gradient that BGDHyperOpt's
 * meta-gradient-descent uses to optimise the learning rate. If Tlaloc's
 * gradient is wrong here, the meta-optimisation would diverge from the
 * paper's reported behaviour.
 */
class HeadToHeadHarnessBgdHyperOptTest {

    @Test
    fun bgdHyperOptHarnessRunsAndProducesBaseline() {
        val result = BgdHyperOptHarness.runBaseline(warmup = 50, measured = 100)

        assertEquals("bgd-hyperopt-outer-loop-K3", result.benchmark)
        assertEquals(50, result.warmupIterations)
        assertEquals(100, result.measuredIterations)

        // Forward value: closed-form reference at (r=0.01, Sxy=111.2, Sx2=55, M=5, K=3).
        val expectedForward = BenchmarkPrimals.bgdHyperOptOuterLoopReference(
            r = 0.01f, Sxy = 111.2f, Sx2 = 55.0f, M = 5.0f, K = 3,
        )
        // 1% relative tolerance + 1e-4 absolute floor — accounts for f32 path
        // differences between the dxir interpreter (depth-3 unrolled MUL+ADD chain)
        // and the Kotlin reference (loop with f32 accumulation).
        val forwardTol = abs(expectedForward) * 1e-2f + 1e-4f
        assertTrue(
            abs(result.forwardValue - expectedForward) < forwardTol,
            "expected forward ≈ $expectedForward, got ${result.forwardValue} (tol=$forwardTol)",
        )

        // 4 inputs → 4 gradient values.
        assertEquals(4, result.gradientValues.size)

        // ∂w/∂r FD validation. The meta-gradient-descent's headline gradient.
        val r = 0.01f
        val Sxy = 111.2f
        val Sx2 = 55.0f
        val M = 5.0f
        val K = 3
        val h = 1e-4f
        val fdDdr = (BenchmarkPrimals.bgdHyperOptOuterLoopReference(r + h, Sxy, Sx2, M, K) -
            BenchmarkPrimals.bgdHyperOptOuterLoopReference(r - h, Sxy, Sx2, M, K)) / (2f * h)
        val analyticalDdr = result.gradientValues[0]
        val ddrTol = abs(fdDdr) * 1e-2f + 1e-3f
        assertTrue(
            abs(analyticalDdr - fdDdr) < ddrTol,
            "df/dr: analytical=$analyticalDdr, FD=$fdDdr (tol=$ddrTol)",
        )

        // Non-zero gradient sanity: the outer loop's value depends on all four
        // inputs, so each partial should be non-zero at our chosen point.
        // (Strict zero would indicate a routing bug rather than a numerical
        // coincidence at this parameter setting.)
        val nonZeroFloor = 1e-6f
        for (i in 0 until 4) {
            assertTrue(
                abs(result.gradientValues[i]) > nonZeroFloor,
                "gradient[$i] = ${result.gradientValues[i]} is suspiciously close to zero " +
                    "(expected non-zero at chosen input point)",
            )
        }

        // Timing sanity, mirrors HeadToHeadHarnessQwopTest's pattern.
        assertTrue(result.minNanos > 0)
        assertTrue(result.medianNanos >= result.minNanos)
        assertTrue(result.p99Nanos >= result.medianNanos)
        assertTrue(
            result.medianNanos < result.minNanos * 100,
            "median ${result.medianNanos}ns suspiciously larger than 100× min ${result.minNanos}ns " +
                "— GC pauses dominating the timing window?",
        )

        // Side-channel print for /loop spot-checks.
        println("[harness ${result.benchmark}] forward=${result.forwardValue} (ref=$expectedForward)")
        println("[harness ${result.benchmark}] gradients=${result.gradientValues}")
        println("[harness ${result.benchmark}] df/dr: tlaloc=$analyticalDdr, FD=$fdDdr")
        println(
            "[harness ${result.benchmark}] grad eval timing (n=${result.measuredIterations}): " +
                "median=${result.medianNanos}ns min=${result.minNanos}ns p99=${result.p99Nanos}ns",
        )
    }

    @Test
    fun bgdHyperOptForwardMatchesReferenceAcrossMultipleR() {
        // Cross-r consistency pin: at multiple r values, harness forward should
        // match the Kotlin reference. Catches any input-routing bug that affects
        // the harness's evaluation path differently from the reference.
        val Sxy = 111.2f
        val Sx2 = 55.0f
        val M = 5.0f
        val K = 3
        val primal = BenchmarkPrimals.bgdHyperOptOuterLoopPrimal(K)

        for (r in listOf(0.001f, 0.01f, 0.05f, 0.1f)) {
            val expected = BenchmarkPrimals.bgdHyperOptOuterLoopReference(r, Sxy, Sx2, M, K)
            val out = DxirInterpreter.evalFunction(
                primal,
                listOf(floatArrayOf(r), floatArrayOf(Sxy), floatArrayOf(Sx2), floatArrayOf(M)),
            )
            val tol = abs(expected) * 1e-2f + 1e-4f
            assertTrue(
                abs(out[0][0] - expected) < tol,
                "at r=$r: expected=$expected got=${out[0][0]} (tol=$tol)",
            )
        }
    }
}
