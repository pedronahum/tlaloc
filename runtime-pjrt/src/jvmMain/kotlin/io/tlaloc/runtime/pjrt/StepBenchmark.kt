package io.tlaloc.runtime.pjrt

/**
 * Times a repeated step the way TorchTPU's perflab does (`perflab/measure.py`), with medians and
 * spread where perflab reports a plain mean:
 *
 *   - the first step is timed on its own (it compiles);
 *   - warm-up runs until the compile count is the same on two consecutive steps, at least
 *     [minWarmup] and at most [maxWarmup] steps, and fails if compiling never stops;
 *   - [timedSteps] steps follow, each synchronized ([sync]) and timed alone; a compile among
 *     them fails the measurement ([RecompileDuringTimingException]), because the number would
 *     include it;
 *   - the result keeps the first-step time, the warm-up overhead (`Σ warm-up − n·last`), the
 *     median, the 10th and 90th percentiles, the mean, the minimum and maximum, and the device's
 *     peak bytes when [peakBytes] reports one.
 *
 * Latency is measured, not throughput: every timed step is synchronized before the next starts.
 */
class StepBenchmark(
    val minWarmup: Int = 10,
    val maxWarmup: Int = 20,
    val timedSteps: Int = 10,
) {
    init {
        require(minWarmup >= 1 && maxWarmup >= minWarmup) { "StepBenchmark: need 1 <= minWarmup <= maxWarmup" }
        require(timedSteps >= 1) { "StepBenchmark: timedSteps must be at least 1" }
    }

    class RecompileDuringTimingException(message: String) : IllegalStateException(message)

    data class Result(
        val firstStepMs: Double,
        val warmupSteps: Int,
        val warmupOverheadMs: Double,
        val compiles: Int,
        val medianMs: Double,
        val p10Ms: Double,
        val p90Ms: Double,
        val meanMs: Double,
        val minMs: Double,
        val maxMs: Double,
        val peakDeviceBytes: Long?,
    ) {
        /** One line of JSON with every field, for appending to a results file. */
        fun toJson(label: String): String =
            "{\"label\": \"$label\", \"first_step_ms\": ${f(firstStepMs)}, \"warmup_steps\": $warmupSteps, " +
                "\"warmup_overhead_ms\": ${f(warmupOverheadMs)}, \"compiles\": $compiles, \"median_ms\": ${f(medianMs)}, " +
                "\"p10_ms\": ${f(p10Ms)}, \"p90_ms\": ${f(p90Ms)}, \"mean_ms\": ${f(meanMs)}, \"min_ms\": ${f(minMs)}, " +
                "\"max_ms\": ${f(maxMs)}, \"peak_device_bytes\": ${peakDeviceBytes ?: "null"}}"

        private fun f(x: Double) = "%.3f".format(x)
    }

    /**
     * Measures [step]. [compileCount] is the number of programs compiled so far (for a
     * [PjrtSession], [PjrtSession.cacheSize]); [sync] waits for a step's result, and receives what
     * [step] returned.
     */
    fun <T> run(
        compileCount: () -> Int,
        peakBytes: () -> Long? = { null },
        sync: (T) -> Unit = {},
        step: () -> T,
    ): Result {
        fun timed(): Double {
            val t0 = System.nanoTime()
            sync(step())
            return (System.nanoTime() - t0) / 1e6
        }
        val first = timed()
        val warm = ArrayList<Double>()
        var before = compileCount()
        while (true) {
            warm += timed()
            val after = compileCount()
            val stable = after == before
            before = after
            if (warm.size >= minWarmup && stable) break
            check(warm.size < maxWarmup) {
                "StepBenchmark: still compiling after $maxWarmup warm-up steps (compile count $after)"
            }
        }
        val compiled = compileCount()
        val times = DoubleArray(timedSteps) {
            val t = timed()
            val now = compileCount()
            if (now != compiled) {
                throw RecompileDuringTimingException(
                    "StepBenchmark: timed step $it compiled a program (compile count $compiled -> $now)",
                )
            }
            t
        }
        val sorted = times.sorted()
        fun pct(p: Double): Double {
            val x = p * (sorted.size - 1)
            val lo = x.toInt()
            val hi = minOf(lo + 1, sorted.size - 1)
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo)
        }
        return Result(
            firstStepMs = first,
            warmupSteps = warm.size,
            warmupOverheadMs = warm.sum() - warm.last() * warm.size,
            compiles = compiled,
            medianMs = pct(0.5),
            p10Ms = pct(0.1),
            p90Ms = pct(0.9),
            meanMs = times.average(),
            minMs = sorted.first(),
            maxMs = sorted.last(),
            peakDeviceBytes = peakBytes(),
        )
    }
}
