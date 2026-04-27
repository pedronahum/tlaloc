package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.224 — head-to-head harness Phase 1 third slice test. Validates the
 * [HookeanSpringHarness] inhabitant — the second **paper benchmark** in the
 * harness suite (after §0.4.223's BGDHyperOpt).
 *
 * **What this test pins**:
 *  - Forward value matches the Kotlin recurrence reference within f32 tolerance.
 *  - Per-input gradient (`pInit`, `vInit`, `kSpring`) FD-validated against
 *    central-difference on the Kotlin reference.
 *  - Cross-input gradient sanity: at the displaced-from-rest configuration,
 *    `df/dpInit` is non-zero (final pos depends on initial displacement),
 *    `df/dkSpring` is non-zero (stiffness affects trajectory).
 *  - Timing statistics positive and well-ordered.
 *
 * **The discriminator**: HookeanSpring's recurrence couples position and
 * velocity tightly — every iteration's update of one depends on the other.
 * A buggy AD that decoupled them (e.g., dropped the `dt * force` contribution
 * to vel's gradient) would produce a wrong `df/dkSpring`. FD validation
 * catches this.
 */
class HeadToHeadHarnessHookeanSpringTest {

    @Test
    fun hookeanSpringHarnessRunsAndProducesBaseline() {
        val result = HookeanSpringHarness.runBaseline(warmup = 50, measured = 100)

        assertEquals("hookean-spring-scalar-N10", result.benchmark)
        assertEquals(50, result.warmupIterations)
        assertEquals(100, result.measuredIterations)

        val pInit = 1.0f
        val vInit = 0.0f
        val kSpring = 1.0f
        val N = 10
        val dt = 0.1f

        // Forward value: Kotlin recurrence reference.
        val expectedForward = BenchmarkPrimals.hookeanSpringReference(pInit, vInit, kSpring, N, dt)
        val forwardTol = abs(expectedForward) * 1e-2f + 1e-4f
        assertTrue(
            abs(result.forwardValue - expectedForward) < forwardTol,
            "expected forward ≈ $expectedForward (Kotlin reference), got ${result.forwardValue}",
        )

        // 3 inputs → 3 gradient values.
        assertEquals(3, result.gradientValues.size)

        // Per-input FD validation. The recurrence's gradient is non-trivial —
        // every input affects the final position via the coupled (pos, vel)
        // recurrence over 10 steps. Use a wider FD step and looser tolerance
        // than BGDHyperOpt because: (a) symplectic Euler accumulates more
        // f32 noise than the BGDHyperOpt closed-form, (b) 10-step recurrence
        // has more compounding rounding than 3-step.
        val h = 1e-3f
        val fdGrads = floatArrayOf(
            (BenchmarkPrimals.hookeanSpringReference(pInit + h, vInit, kSpring, N, dt)
                - BenchmarkPrimals.hookeanSpringReference(pInit - h, vInit, kSpring, N, dt)) / (2f * h),
            (BenchmarkPrimals.hookeanSpringReference(pInit, vInit + h, kSpring, N, dt)
                - BenchmarkPrimals.hookeanSpringReference(pInit, vInit - h, kSpring, N, dt)) / (2f * h),
            (BenchmarkPrimals.hookeanSpringReference(pInit, vInit, kSpring + h, N, dt)
                - BenchmarkPrimals.hookeanSpringReference(pInit, vInit, kSpring - h, N, dt)) / (2f * h),
        )
        val labels = listOf("df/dpInit", "df/dvInit", "df/dkSpring")
        for (i in 0 until 3) {
            val analGrad = result.gradientValues[i]
            val fdGrad = fdGrads[i]
            // 3% relative tolerance + 1e-2 absolute floor, accounting for the
            // 10-step recurrence's compounded f32 noise.
            val tol = abs(fdGrad) * 0.03f + 1e-2f
            assertTrue(
                abs(analGrad - fdGrad) < tol,
                "${labels[i]}: analytical=$analGrad, FD=$fdGrad (tol=$tol)",
            )
        }

        // Sanity: df/dpInit and df/dkSpring should be non-zero at this
        // displaced-from-rest configuration (the trajectory genuinely depends
        // on both). df/dvInit may be small but non-zero too.
        assertTrue(
            abs(result.gradientValues[0]) > 1e-3f,
            "df/dpInit should be non-zero at pInit=1, got ${result.gradientValues[0]}",
        )
        assertTrue(
            abs(result.gradientValues[2]) > 1e-3f,
            "df/dkSpring should be non-zero at kSpring=1, got ${result.gradientValues[2]}",
        )

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
    fun hookeanSpringForwardMatchesReferenceAcrossMultipleInputSets() {
        // Cross-input consistency pin: at multiple input configurations, harness
        // forward eval matches the Kotlin recurrence reference. Catches any
        // input-routing bug. Includes a "rest" configuration (pInit=vInit=0
        // → trajectory stays at zero regardless of kSpring).
        val N = 10
        val dt = 0.1f
        val primal = BenchmarkPrimals.hookeanSpringPrimal(N, dt)

        val testCases = listOf(
            Triple(1.0f, 0.0f, 1.0f),      // baseline
            Triple(0.0f, 0.0f, 5.0f),      // rest configuration: stays at 0
            Triple(2.0f, -0.5f, 0.5f),     // perturbed initial conditions
            Triple(-1.0f, 1.0f, 2.0f),     // mirror-symmetric input
        )

        for ((pInit, vInit, kSpring) in testCases) {
            val expected = BenchmarkPrimals.hookeanSpringReference(pInit, vInit, kSpring, N, dt)
            val out = DxirInterpreter.evalFunction(
                primal,
                listOf(floatArrayOf(pInit), floatArrayOf(vInit), floatArrayOf(kSpring)),
            )
            val tol = abs(expected) * 1e-2f + 1e-4f
            assertTrue(
                abs(out[0][0] - expected) < tol,
                "at (pInit=$pInit, vInit=$vInit, kSpring=$kSpring): expected=$expected got=${out[0][0]} (tol=$tol)",
            )
        }
    }
}
