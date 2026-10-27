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

    @Test
    fun utilizationIsWorkOverTheMedianStepAgainstThePeaks() {
        val r = StepBenchmark.Result(
            firstStepMs = 100.0, warmupSteps = 10, warmupOverheadMs = 0.0, compiles = 1,
            medianMs = 2.0, p10Ms = 2.0, p90Ms = 2.0, meanMs = 2.0, minMs = 2.0, maxMs = 2.0, peakDeviceBytes = null,
        )
        val v6e = io.tlaloc.ir.recognizer.cost.DeviceDescriptors.TPU_V6E
        // 918 GFLOP in 2 ms is 459 TFLOP/s: half of v6e's 918 TFLOP/s; 3.276 GB in 2 ms is its bandwidth.
        val u = r.utilization(v6e, flopsPerStep = 918e9, bytesPerStep = 3.276e9)
        kotlin.test.assertEquals(459e12, u.flopsPerSec, 1e3)
        kotlin.test.assertEquals(0.5, u.mfu, 1e-12)
        kotlin.test.assertEquals(1.0, u.mbu!!, 1e-12)
        val json = r.toJson("step", u)
        kotlin.test.assertTrue("\"mfu\": 0.5000" in json && "\"mbu\": 1.0000" in json && "\"device\": \"tpu_v6e\"" in json, json)
        kotlin.test.assertTrue("\"mbu\": null" in r.toJson("step", r.utilization(v6e, 1e9)))
    }
}
