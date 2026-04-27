package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.225 — head-to-head harness Phase 1 fourth slice test. Validates the
 * [BrachistochroneHarness] inhabitant — the third **paper benchmark** in the
 * harness suite (after BGDHyperOpt and HookeanSpring).
 *
 * **What this test pins**:
 *  - Forward value matches the closed-form `(1 + y)^N` exactly in f32.
 *  - Gradient matches the closed-form `N · (1 + y)^(N-1)` within tight tolerance.
 *  - Cross-y consistency: at multiple y values, harness forward eval matches
 *    the Kotlin recurrence reference.
 *  - Gradient sign + sign-symmetry: at y > 0 gradient is positive; at y < 0
 *    (above -1) gradient sign tracks the closed form.
 *  - Timing statistics positive and well-ordered.
 *
 * **The discriminator**: this primal has TWO independent reference paths:
 * (a) the recurrence step-by-step (`brachistochroneCompoundVelocityReference`),
 * (b) the closed form `(1 + y)^N` via `kotlin.math.pow`. Pinning Tlaloc against
 * BOTH catches bugs that affect just one path (e.g., a primal misrouting that
 * produces a different recurrence shape but still happens to coincide with
 * the closed form at one point).
 */
class HeadToHeadHarnessBrachistochroneTest {

    @Test
    fun brachistochroneHarnessRunsAndProducesBaseline() {
        val result = BrachistochroneHarness.runBaseline(warmup = 50, measured = 100)

        assertEquals("brachistochrone-compound-velocity-N5", result.benchmark)
        assertEquals(50, result.warmupIterations)
        assertEquals(100, result.measuredIterations)

        val y = 0.5f
        val N = 5

        // Forward: closed-form (1 + y)^N. Exact in f32 at y=0.5, N=5: 1.5^5 = 7.59375.
        val expectedForward = (1.0 + y).pow(N).toFloat()
        val recurrenceRef = BenchmarkPrimals.brachistochroneCompoundVelocityReference(y, N)
        // Both reference paths should match each other (sanity).
        assertTrue(
            abs(expectedForward - recurrenceRef) < 1e-4f,
            "Closed-form $expectedForward and recurrence $recurrenceRef should agree",
        )
        // Tlaloc's forward should match both within tight tolerance.
        assertTrue(
            abs(result.forwardValue - expectedForward) < 1e-3f,
            "expected forward = $expectedForward, got ${result.forwardValue}",
        )

        // Gradient: closed-form N · (1 + y)^(N-1). At y=0.5, N=5: 5 × 1.5^4 = 25.3125.
        val expectedGrad = N.toFloat() * (1.0 + y).pow(N - 1).toFloat()
        assertEquals(1, result.gradientValues.size)
        val analyticalGrad = result.gradientValues[0]
        assertTrue(
            abs(analyticalGrad - expectedGrad) < 1e-2f,
            "expected df/dy = $expectedGrad (N · (1+y)^(N-1) = 5 × 1.5^4), got $analyticalGrad",
        )

        // Sign sanity: at y > -1 the gradient is strictly positive (every iter
        // multiplies by (1+y) > 0).
        assertTrue(analyticalGrad > 0f, "df/dy should be positive at y > -1, got $analyticalGrad")

        // Timing sanity, mirrors prior harness tests.
        assertTrue(result.minNanos > 0)
        assertTrue(result.medianNanos >= result.minNanos)
        assertTrue(result.p99Nanos >= result.medianNanos)
        assertTrue(
            result.medianNanos < result.minNanos * 100,
            "median ${result.medianNanos}ns suspiciously larger than 100× min ${result.minNanos}ns",
        )

        println("[harness ${result.benchmark}] forward=${result.forwardValue} (closed-form=$expectedForward)")
        println("[harness ${result.benchmark}] gradient=$analyticalGrad (closed-form=$expectedGrad)")
        println(
            "[harness ${result.benchmark}] grad eval timing (n=${result.measuredIterations}): " +
                "median=${result.medianNanos}ns min=${result.minNanos}ns p99=${result.p99Nanos}ns",
        )
    }

    @Test
    fun brachistochroneForwardAndGradientAcrossMultipleY() {
        // Cross-y consistency: at multiple y values, both forward and gradient
        // match closed-form. Includes:
        //   - y=0: forward = 1, gradient = 5 (recurrence stays constant; gradient
        //     is N at rest configuration).
        //   - y=-0.5: forward = 0.5^5 = 0.03125; gradient = 5·0.5^4 = 0.3125.
        //   - y=1: forward = 2^5 = 32; gradient = 5·2^4 = 80.
        val N = 5
        val primal = BenchmarkPrimals.brachistochroneCompoundVelocityPrimal(N)
        val coarsened = io.tlaloc.ir.passes.PhiCalculus.apply(primal)
        val grad = io.tlaloc.ir.passes.DxirReverseTransform.apply(coarsened)

        for (y in listOf(0.0f, 0.25f, 0.5f, -0.5f, 1.0f)) {
            // Forward.
            val expectedFwd = (1.0 + y).pow(N).toFloat()
            val outFwd = DxirInterpreter.evalFunction(primal, listOf(floatArrayOf(y)))
            val fwdTol = abs(expectedFwd) * 1e-3f + 1e-4f
            assertTrue(
                abs(outFwd[0][0] - expectedFwd) < fwdTol,
                "forward at y=$y: expected=$expectedFwd, got=${outFwd[0][0]}",
            )

            // Gradient.
            val expectedGrad = N.toFloat() * (1.0 + y).pow(N - 1).toFloat()
            val outGrad = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(y)))
            val gradTol = abs(expectedGrad) * 1e-3f + 1e-3f
            assertTrue(
                abs(outGrad[0][0] - expectedGrad) < gradTol,
                "gradient at y=$y: expected=$expectedGrad, got=${outGrad[0][0]}",
            )
        }
    }
}
