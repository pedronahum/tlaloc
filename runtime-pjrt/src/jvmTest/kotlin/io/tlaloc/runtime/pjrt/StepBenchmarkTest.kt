package io.tlaloc.runtime.pjrt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class StepBenchmarkTest {

    @Test
    fun warmupStopsOnceCompilingStops() {
        var compiles = 0
        var calls = 0
        // Compiles on the first three steps, then never again.
        val r = StepBenchmark(minWarmup = 2, maxWarmup = 10, timedSteps = 5).run(compileCount = { compiles }) {
            if (calls++ < 3) compiles++
        }
        assertEquals(3, r.compiles)
        assertEquals(3, r.warmupSteps, "warm-up ends when two consecutive steps leave the count unchanged")
        assertTrue(r.p10Ms <= r.medianMs && r.medianMs <= r.p90Ms)
        assertTrue("\"compiles\": 3" in r.toJson("x"))
    }

    @Test
    fun aCompileDuringTheTimedStepsFails() {
        var compiles = 0
        var calls = 0
        assertFailsWith<StepBenchmark.RecompileDuringTimingException> {
            StepBenchmark(minWarmup = 2, maxWarmup = 4, timedSteps = 5).run(compileCount = { compiles }) {
                if (++calls == 6) compiles++
            }
        }
    }

    @Test
    fun aStepThatNeverStopsCompilingFails() {
        var compiles = 0
        assertFailsWith<IllegalStateException> {
            StepBenchmark(minWarmup = 2, maxWarmup = 4, timedSteps = 1).run(compileCount = { compiles }) { compiles++ }
        }
    }
}
