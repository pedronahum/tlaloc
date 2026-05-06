package io.tlaloc.benchmarks

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File
import kotlin.test.Test

/**
 * §0.4.298 — final step of the four-step comparison plan. Reads the
 * per-framework `harness-results-*.json` files dumped by §0.4.294 / §0.4.295
 * (Tlaloc-CPU + Tlaloc-CUDA), §0.4.296 (PyTorch-CPU), §0.4.297 (JAX-GPU)
 * and joins them into a single comparison matrix:
 *
 *  - one row per `(benchmark, framework)` pair,
 *  - speedup ratios computed against `tlaloc-iree-cuda` so the matrix
 *    answers "how much slower / faster is each framework than Tlaloc on
 *    this Blackwell host?",
 *  - markdown + CSV + JSON dumps under `benchmarks/build/`.
 *
 * Self-skips when zero source files are found (i.e. none of the prior
 * benches has been run yet on this host). Otherwise produces a partial
 * matrix with `—` for rows whose source is missing — useful even when
 * one framework is unavailable.
 *
 * Invocation order to fill the matrix completely (run the per-framework
 * benches first, then this aggregator):
 *
 *     ./gradlew :benchmarks:jvmTest --tests \
 *       "io.tlaloc.benchmarks.LlamaDecoderIreeMediumBenchmarkTest" \
 *       "io.tlaloc.benchmarks.LlamaDecoderIreeMediumBackwardBenchmarkTest" \
 *       "io.tlaloc.benchmarks.LlamaDecoderPytorchBenchTest" \
 *       "io.tlaloc.benchmarks.LlamaDecoderJaxBenchTest" \
 *       "io.tlaloc.benchmarks.LlamaDecoderComparisonMatrixTest"
 */
class LlamaDecoderComparisonMatrixTest {

    private data class Row(
        val benchmark: String,
        val framework: String,
        val medianNs: Long,
        val minNs: Long,
        val p99Ns: Long,
        val nIterations: Int,
    )

    private fun parseJsonRows(text: String): List<Row> {
        // Files are arrays of HeadToHeadResult JSON objects (multi-line
        // pretty-printed). Tokenise on `}` to split entries; then pull
        // the relevant fields with simple substring scans.
        val rows = mutableListOf<Row>()
        var i = 0
        while (i < text.length) {
            val open = text.indexOf('{', i)
            if (open < 0) break
            val close = text.indexOf('}', open)
            if (close < 0) break
            val obj = text.substring(open, close + 1)
            i = close + 1
            fun field(name: String): String? {
                val k = "\"$name\""
                val ki = obj.indexOf(k)
                if (ki < 0) return null
                val colon = obj.indexOf(':', ki + k.length)
                var end = colon + 1
                while (end < obj.length && obj[end] != ',' && obj[end] != '\n') end++
                return obj.substring(colon + 1, end).trim().removeSurrounding("\"")
            }
            val benchmark = field("benchmark") ?: continue
            val framework = field("framework") ?: continue
            val medianNs = field("medianNanos")?.toLongOrNull() ?: continue
            val minNs = field("minNanos")?.toLongOrNull() ?: continue
            val p99Ns = field("p99Nanos")?.toLongOrNull() ?: continue
            val nIters = field("measuredIterations")?.toIntOrNull() ?: continue
            rows += Row(benchmark, framework, medianNs, minNs, p99Ns, nIters)
        }
        return rows
    }

    @Test
    fun aggregatesLlamaDecoderMediumComparisonMatrix() {
        val outputDir = File("build")
        outputDir.mkdirs()
        val sources = outputDir.listFiles { f ->
            f.isFile && f.name.startsWith("harness-results-") && f.name.endsWith(".json") &&
                "summary" !in f.name && "comparison" !in f.name
        }?.toList().orEmpty()
        assumeTrue(
            sources.isNotEmpty(),
            "no per-framework `harness-results-*.json` files found under $outputDir — " +
                "run the §0.4.294 / §0.4.295 / §0.4.296 / §0.4.297 benches first.",
        )

        val rows = sources.flatMap { parseJsonRows(it.readText()) }
        val mediumRows = rows.filter { it.benchmark.contains("llama-decoder-medium") }
        assumeTrue(
            mediumRows.isNotEmpty(),
            "no `llama-decoder-medium-*` rows in $sources — nothing to aggregate.",
        )

        // Pivot: matrix[mode][framework] = Row
        val frameworks = listOf(
            "pytorch-cpu",
            "tlaloc-iree-cpu",
            "tlaloc-iree-cuda",
            "jax-gpu",
        )
        val modes = listOf("forward", "backward")
        val matrix: Map<String, Map<String, Row?>> = modes.associateWith { mode ->
            frameworks.associateWith { fw ->
                mediumRows.firstOrNull {
                    it.framework == fw && it.benchmark.endsWith(mode)
                }
            }
        }

        // Speedup baseline: Tlaloc-CUDA wherever available, else first
        // available row in the mode's column. Speedup = baseline / row
        // (so > 1.0 means Tlaloc is faster, < 1.0 means slower).
        fun pickBaseline(mode: String): Row? {
            return matrix[mode]?.get("tlaloc-iree-cuda")
                ?: matrix[mode]?.values?.firstOrNull { it != null }
        }

        val md = buildString {
            append("# LlamaDecoder medium-config comparison\n\n")
            append("Inputs: 256 tok × 512 dModel × 2048 vocab. Same Kotlin-seed-42 inputs (.npy)\n")
            append("driving every framework — apples-to-apples by construction.\n\n")
            append("Speedup column = `tlaloc-iree-cuda median / framework median`.\n")
            append("Values > 1.0 = Tlaloc-CUDA faster; < 1.0 = framework faster.\n\n")
            for (mode in modes) {
                append("## $mode\n\n")
                append("| framework | median (µs) | min (µs) | p99 (µs) | iters/rep | speedup vs Tlaloc-CUDA |\n")
                append("|---|---:|---:|---:|---:|---:|\n")
                val baseline = pickBaseline(mode)
                for (fw in frameworks) {
                    val r = matrix[mode]?.get(fw)
                    if (r == null) {
                        append("| $fw | — | — | — | — | — |\n")
                    } else {
                        val ratio = if (baseline != null && r.medianNs > 0)
                            "%.2fx".format(baseline.medianNs.toDouble() / r.medianNs)
                        else "—"
                        append("| $fw | ${r.medianNs / 1_000} | ${r.minNs / 1_000} | ${r.p99Ns / 1_000} | ${r.nIterations} | $ratio |\n")
                    }
                }
                append("\n")
            }
            append("## Full training step (forward + backward, median)\n\n")
            append("| framework | step (ms) | speedup vs Tlaloc-CUDA |\n")
            append("|---|---:|---:|\n")
            val baselineStep = run {
                val f = matrix["forward"]?.get("tlaloc-iree-cuda")
                val b = matrix["backward"]?.get("tlaloc-iree-cuda")
                if (f != null && b != null) f.medianNs + b.medianNs else null
            }
            for (fw in frameworks) {
                val f = matrix["forward"]?.get(fw)
                val b = matrix["backward"]?.get(fw)
                if (f == null || b == null) {
                    append("| $fw | — | — |\n")
                } else {
                    val step = f.medianNs + b.medianNs
                    val ratio = if (baselineStep != null && step > 0)
                        "%.2fx".format(baselineStep.toDouble() / step)
                    else "—"
                    append("| $fw | %.3f | $ratio |\n".format(step / 1_000_000.0))
                }
            }
        }

        // Dump MD + CSV + JSON.
        File(outputDir, "harness-results-llama-medium-comparison.md").writeText(md)
        File(outputDir, "harness-results-llama-medium-comparison.csv").writeText(
            buildString {
                append("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns\n")
                for (mode in modes) {
                    for (fw in frameworks) {
                        val r = matrix[mode]?.get(fw) ?: continue
                        append(r.benchmark); append(',')
                        append(r.framework); append(',')
                        append(r.nIterations); append(',')
                        append(r.medianNs); append(',')
                        append(r.minNs); append(',')
                        append(r.p99Ns); append('\n')
                    }
                }
            },
        )
        File(outputDir, "harness-results-llama-medium-comparison.json").writeText(
            buildString {
                append("[\n")
                val all = modes.flatMap { mode ->
                    frameworks.mapNotNull { fw -> matrix[mode]?.get(fw) }
                }
                for ((idx, r) in all.withIndex()) {
                    if (idx > 0) append(",\n")
                    append("  {\n")
                    append("    \"benchmark\": \"${r.benchmark}\",\n")
                    append("    \"framework\": \"${r.framework}\",\n")
                    append("    \"medianNanos\": ${r.medianNs},\n")
                    append("    \"minNanos\": ${r.minNs},\n")
                    append("    \"p99Nanos\": ${r.p99Ns},\n")
                    append("    \"measuredIterations\": ${r.nIterations}\n")
                    append("  }")
                }
                append("\n]\n")
            },
        )
        println(md)
    }
}
