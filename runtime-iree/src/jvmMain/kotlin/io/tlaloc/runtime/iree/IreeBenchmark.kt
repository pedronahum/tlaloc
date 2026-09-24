package io.tlaloc.runtime.iree

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * Stats from a single `iree-benchmark-module` run: per-repetition real_time
 * values converted to nanoseconds, plus the iteration count each repetition
 * looped through internally. Use [medianNanos] / [minNanos] / [p99Nanos] for
 * cross-framework comparison; the reps count tells you how many measurements
 * back the percentiles.
 */
data class IreeBenchmarkStats(
    val repetitionRealTimesNs: List<Long>,
    val iterationsPerRepetition: Long,
) {
    val medianNanos: Long get() = sortedReps[sortedReps.size / 2]
    val minNanos: Long get() = sortedReps.first()
    val p99Nanos: Long get() = sortedReps[((sortedReps.size - 1) * 99) / 100]
    private val sortedReps: List<Long> by lazy { repetitionRealTimesNs.sorted() }
}

/**
 * Subprocess facade over `iree-benchmark-module`. iree-benchmark-module runs
 * the warmup + measurement loop inside its own process, so per-iteration
 * timing is dominated by the actual compute (not the JVM↔subprocess RTT).
 * Timing through `iree-run-module` pays ~30 ms subprocess overhead per
 * dispatch, which swamps sub-ms ops; this tool pays one subprocess per
 * benchmark instead.
 *
 * Usage: compile via [IreeRuntime.compile] (cached VMFB), pass the resulting
 * [IreeModule] in here along with the same input strings you would have given
 * [IreeRuntime.invoke]. The helper produces [IreeBenchmarkStats] suitable for
 * a [io.tlaloc.benchmarks.HeadToHeadResult].
 */
object IreeBenchmark {

    private const val DEFAULT_TIMEOUT_SECONDS: Long = 600L

    fun run(
        module: IreeModule,
        function: String = "main",
        inputs: List<String> = emptyList(),
        repetitions: Int = 5,
        minTimePerRep: String = "0.5s",
        timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS,
    ): IreeBenchmarkStats {
        val bin = IreeBinaries.requireTool("iree-benchmark-module", IreeBinaries.ireeBenchmarkModule)

        // Reuse the §0.4.288 flagfile-for-inputs pattern so giant input lists
        // (LlamaDecoder is 13 weights at ~196 KB textual each) don't hit the OS
        // argv limit.
        val flagfile = Files.createTempFile("tlaloc-iree-bench-flagfile-", ".txt")
        val stdoutFile = Files.createTempFile("tlaloc-iree-bench-stdout-", ".csv")
        val stderrFile = Files.createTempFile("tlaloc-iree-bench-stderr-", ".log")
        try {
            Files.writeString(
                flagfile,
                buildString {
                    for (input in inputs) {
                        append("--input=").append(input).append('\n')
                    }
                },
            )

            val args = listOf(
                bin,
                "--module=${module.vmfbPath}",
                "--device=${module.target.device}",
                "--function=$function",
                "--benchmark_format=csv",
                "--benchmark_min_time=$minTimePerRep",
                "--benchmark_repetitions=$repetitions",
                "--flagfile=$flagfile",
            )
            val pb = ProcessBuilder(args)
                .redirectOutput(stdoutFile.toFile())
                .redirectError(stderrFile.toFile())
            val p = pb.start()
            p.outputStream.close()
            val finished = p.waitFor(timeoutSeconds, TimeUnit.SECONDS)
            if (!finished) {
                p.destroyForcibly()
                error("iree-benchmark-module timed out after ${timeoutSeconds}s")
            }
            if (p.exitValue() != 0) {
                error(
                    "iree-benchmark-module failed (exit=${p.exitValue()}); stderr:\n" +
                        Files.readString(stderrFile).take(4000),
                )
            }
            return parse(Files.readString(stdoutFile))
        } finally {
            Files.deleteIfExists(flagfile)
            Files.deleteIfExists(stdoutFile)
            Files.deleteIfExists(stderrFile)
        }
    }

    /**
     * Parse `iree-benchmark-module --benchmark_format=csv` output. Layout:
     *
     * ```
     * name,iterations,real_time,cpu_time,time_unit,bytes_per_second,items_per_second,…
     * "BM_<fn>/process_time/real_time",1011,0.234,0.897,ms,…
     * "BM_<fn>/process_time/real_time",1011,0.218,0.891,ms,…
     * …
     * "BM_<fn>/process_time/real_time_mean",N,0.223,0.885,ms,…
     * "BM_<fn>/process_time/real_time_median",N,0.220,0.881,ms,…
     * "BM_<fn>/process_time/real_time_stddev",…
     * "BM_<fn>/process_time/real_time_cv",…
     * ```
     *
     * The per-rep rows use the bare `…/real_time` suffix; aggregate rows
     * append `_mean` / `_median` / `_stddev` / `_cv`. We parse only the
     * per-rep rows (what we want for our own median/min/p99 stats).
     */
    internal fun parse(stdout: String): IreeBenchmarkStats {
        val lines = stdout.lines().filter { it.isNotBlank() }
        require(lines.isNotEmpty()) { "iree-benchmark-module stdout was empty" }
        val header = lines.first { it.startsWith("name,") || it.contains("\"name\"") }
        val cols = header.split(',')
        val realTimeIdx = cols.indexOf("real_time")
        val iterationsIdx = cols.indexOf("iterations")
        val timeUnitIdx = cols.indexOf("time_unit")
        require(realTimeIdx >= 0 && iterationsIdx >= 0 && timeUnitIdx >= 0) {
            "iree-benchmark-module CSV header missing one of {real_time, iterations, time_unit}: $header"
        }

        val perRepRegex = Regex("""^"BM_[^"]*/real_time"""")
        val timesNs = mutableListOf<Long>()
        var iterations = 0L
        for (line in lines) {
            if (!perRepRegex.containsMatchIn(line)) continue
            val parts = splitCsvLine(line)
            val rt = parts[realTimeIdx].toDoubleOrNull()
                ?: error("could not parse real_time from CSV row: $line")
            val unit = parts[timeUnitIdx]
            val rtNs = when (unit) {
                "ns" -> rt
                "us" -> rt * 1_000.0
                "ms" -> rt * 1_000_000.0
                "s" -> rt * 1_000_000_000.0
                else -> error("unsupported real_time unit '$unit' in row: $line")
            }
            timesNs.add(rtNs.toLong())
            iterations = parts[iterationsIdx].toLong()
        }
        require(timesNs.isNotEmpty()) {
            "iree-benchmark-module produced no per-rep rows; raw stdout (first 2 KB):\n${stdout.take(2000)}"
        }
        return IreeBenchmarkStats(repetitionRealTimesNs = timesNs, iterationsPerRepetition = iterations)
    }

    /** Lightweight CSV row splitter handling double-quoted fields. */
    private fun splitCsvLine(line: String): List<String> {
        val out = mutableListOf<String>()
        var i = 0
        val n = line.length
        val sb = StringBuilder()
        var inQuotes = false
        while (i < n) {
            val c = line[i]
            when {
                c == '"' -> inQuotes = !inQuotes
                c == ',' && !inQuotes -> { out += sb.toString(); sb.clear() }
                else -> sb.append(c)
            }
            i++
        }
        out += sb.toString()
        return out
    }
}
