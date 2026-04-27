package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.227 — head-to-head harness Phase 1 sixth slice test. Validates the
 * [CartPolePhase1Harness] inhabitant — the **fifth and final paper benchmark**
 * for full M9 coverage. First harness inhabitant to exercise SIN, COS, ABS,
 * and IF (clip-at-zero).
 *
 * **What this test pins**:
 *  - Forward value matches Kotlin reference within tight tolerance.
 *  - Per-input gradient FD-validated. Looser tolerance (3% rel) accommodates
 *    sin/cos/abs f32 path differences AND the IF branch eval.
 *  - **The structural discriminator**: `df/dat ≈ 0` because `at` only flows
 *    into the unused `pt` chain (rt → qt → pt; pt isn't in the return).
 *    Any DCE bug or misrouted chain rule that incorrectly accumulated grad
 *    through the unused branch would surface as a non-zero `df/dat`.
 *  - Cross-input consistency at multiple cfg values, including a clipped
 *    case where `maxArg < 0` triggers the IF's else-branch.
 *  - Timing statistics positive and well-ordered.
 */
class HeadToHeadHarnessCartPoleTest {

    @Test
    fun cartPoleHarnessRunsAndProducesBaseline() {
        val result = CartPolePhase1Harness.runBaseline(warmup = 50, measured = 100)

        assertEquals("cartpole-phase1-onestep", result.benchmark)
        assertEquals(50, result.warmupIterations)
        assertEquals(100, result.measuredIterations)

        val at = 0.5f
        val x0 = 0.0f
        val x1 = 0.1f
        val x2 = 0.05f
        val x3 = 0.02f

        // Forward: Kotlin reference.
        val expectedForward = BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1, x2, x3)
        val forwardTol = abs(expectedForward) * 1e-2f + 1e-4f
        assertTrue(
            abs(result.forwardValue - expectedForward) < forwardTol,
            "expected forward ≈ $expectedForward (Kotlin reference), got ${result.forwardValue}",
        )

        // 5 inputs → 5 gradient values.
        assertEquals(5, result.gradientValues.size)

        // **The headline pin**: df/dat ≈ 0. The action `at` only feeds into
        // the unused `pt` chain (rt → qt → pt; pt isn't in the return value).
        // A DCE-aware reverse-mode AD should produce zero here. Any
        // misrouted chain rule that accumulated grad through the unused
        // chain would surface a non-zero value.
        val dfDat = result.gradientValues[0]
        assertTrue(
            abs(dfDat) < 1e-4f,
            "df/dat should be ~0 (at flows only into unused pt chain), got $dfDat",
        )

        // Per-input FD validation for the live inputs (x0, x1, x2, x3).
        val h = 1e-3f
        val fdGrads = floatArrayOf(
            0f,   // df/dat — covered by the strict zero pin above
            (BenchmarkPrimals.cartPolePhase1Reference(at, x0 + h, x1, x2, x3)
                - BenchmarkPrimals.cartPolePhase1Reference(at, x0 - h, x1, x2, x3)) / (2f * h),
            (BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1 + h, x2, x3)
                - BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1 - h, x2, x3)) / (2f * h),
            (BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1, x2 + h, x3)
                - BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1, x2 - h, x3)) / (2f * h),
            (BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1, x2, x3 + h)
                - BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1, x2, x3 - h)) / (2f * h),
        )
        val labels = listOf("df/dat", "df/dx0", "df/dx1", "df/dx2", "df/dx3")
        for (i in 1 until 5) {
            val analGrad = result.gradientValues[i]
            val fdGrad = fdGrads[i]
            // 3% relative tolerance + 1e-3 absolute floor — accommodates
            // sin/cos/abs f32 path differences plus the IF branch eval.
            val tol = abs(fdGrad) * 0.03f + 1e-3f
            assertTrue(
                abs(analGrad - fdGrad) < tol,
                "${labels[i]}: analytical=$analGrad, FD=$fdGrad (tol=$tol)",
            )
        }

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
    fun cartPoleForwardMatchesReferenceAcrossMultipleConfigs() {
        // Cross-cfg consistency: at multiple state configurations, harness
        // forward eval matches the Kotlin reference. Includes a CLIPPED case
        // where maxArg < 0 triggers the IF's else-branch (clipped → 0,
        // term = 0.5, term² = 0.25).
        val primal = BenchmarkPrimals.cartPolePhase1Primal()

        // Configs as (at, x0, x1, x2, x3):
        val testCases = listOf(
            floatArrayOf(0.5f, 0.0f, 0.1f, 0.05f, 0.02f),    // baseline
            floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),       // origin
            // Clipped case: |xn0| > 2.4 → first factor < 0 → maxArg < 0 →
            // clipped = 0 → term = 0.5 → result = 0.25.
            floatArrayOf(0.5f, 3.0f, 0.0f, 0.0f, 0.0f),       // x0=3, |xn0|>2.4
            // Mirror-symmetric (negative inputs).
            floatArrayOf(-0.5f, -0.1f, 0.0f, -0.05f, 0.0f),
        )

        for (cfg in testCases) {
            val (at, x0, x1, x2, x3) = listOf(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4])
            val expected = BenchmarkPrimals.cartPolePhase1Reference(at, x0, x1, x2, x3)
            val out = DxirInterpreter.evalFunction(
                primal,
                listOf(
                    floatArrayOf(at), floatArrayOf(x0), floatArrayOf(x1),
                    floatArrayOf(x2), floatArrayOf(x3),
                ),
            )
            val tol = abs(expected) * 1e-2f + 1e-4f
            assertTrue(
                abs(out[0][0] - expected) < tol,
                "at cfg=${cfg.toList()}: expected=$expected got=${out[0][0]} (tol=$tol)",
            )
        }
    }
}
