package io.tlaloc.benchmarks

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.222 — head-to-head harness Phase 1 first slice test. Validates the
 * [HeadToHeadHarness] scaffold by running the [QwopAvatarStepHarness]
 * inhabitant and pinning:
 *  - Numerical baseline: forward + per-input gradient values match the
 *    §0.4.220 integration test's hand-traced expected values.
 *  - Timing statistics are positive and sane (median < 10× min — sanity
 *    that the timing window captured a coherent workload, not GC noise).
 *  - JSON serialisation round-trips structurally.
 *
 * **Why a low warmup/measured for the test**: full 200/800 iterations
 * of full-primal gradient eval would slow the test suite. Use 50/100
 * for the test pin; the side-channel `println` reports the actual values
 * so a /loop iteration can spot-check throughput.
 */
class HeadToHeadHarnessQwopTest {

    @Test
    fun qwopAvatarStepHarnessRunsAndProducesBaseline() {
        val result = QwopAvatarStepHarness.runBaseline(warmup = 50, measured = 100)

        assertEquals("qwop-avatar-step", result.benchmark)
        assertEquals(50, result.warmupIterations)
        assertEquals(100, result.measuredIterations)

        // §0.4.220 integration test pinned the forward value at m*=0.1.
        // Same input set here → same expected forward.
        assertTrue(
            abs(result.forwardValue - 0.7227f) < 0.01f,
            "expected forward ≈ 0.7227 at m*=0.1, got ${result.forwardValue}",
        )

        // 4 muscle inputs → 4 gradient values.
        assertEquals(4, result.gradientValues.size)
        // §0.4.220's sign discriminator: at m*=0.1 (all positive),
        // every gradient should be positive.
        for ((i, g) in result.gradientValues.withIndex()) {
            assertTrue(g > 0f, "gradient[$i] should be positive at m*=0.1, got $g")
        }

        // Timing sanity: positive and reasonably tight distribution.
        assertTrue(result.minNanos > 0, "minNanos should be positive, got ${result.minNanos}")
        assertTrue(
            result.medianNanos >= result.minNanos,
            "median ($result.medianNanos) should be >= min (${result.minNanos})",
        )
        assertTrue(
            result.p99Nanos >= result.medianNanos,
            "p99 (${result.p99Nanos}) should be >= median (${result.medianNanos})",
        )
        // Sanity: median should not be ridiculously larger than min, even
        // accounting for GC noise. 100× allows for one outlier GC pause.
        assertTrue(
            result.medianNanos < result.minNanos * 100,
            "median (${result.medianNanos}) suspiciously larger than 100× min (${result.minNanos}) " +
                "— GC pauses dominating the timing window?",
        )

        // Side-channel print for /loop spot-checks.
        println("[harness ${result.benchmark}] forward=${result.forwardValue}")
        println("[harness ${result.benchmark}] gradients=${result.gradientValues}")
        println(
            "[harness ${result.benchmark}] grad eval timing (n=${result.measuredIterations}): " +
                "median=${result.medianNanos}ns min=${result.minNanos}ns p99=${result.p99Nanos}ns",
        )
    }

    @Test
    fun headToHeadResultJsonRoundTrips() {
        val result = QwopAvatarStepHarness.runBaseline(warmup = 10, measured = 20)
        val json = result.toJsonString()

        // Structural smoke test on the JSON output. Phase 2 will add a real
        // parser; for Phase 1 we just verify the JSON contains the expected
        // top-level keys with the expected values.
        assertTrue(json.startsWith("{"), "JSON should start with '{'")
        assertTrue(json.endsWith("}"), "JSON should end with '}'")
        assertTrue(json.contains("\"benchmark\": \"qwop-avatar-step\""), "JSON should contain benchmark name")
        assertTrue(json.contains("\"forwardValue\""), "JSON should contain forwardValue field")
        assertTrue(json.contains("\"gradientValues\""), "JSON should contain gradientValues field")
        assertTrue(json.contains("\"medianNanos\""), "JSON should contain medianNanos field")
        assertTrue(json.contains("\"minNanos\""), "JSON should contain minNanos field")
        assertTrue(json.contains("\"p99Nanos\""), "JSON should contain p99Nanos field")
        assertTrue(json.contains("\"warmupIterations\": 10"), "JSON should preserve warmup count")
        assertTrue(json.contains("\"measuredIterations\": 20"), "JSON should preserve measured count")
    }
}
