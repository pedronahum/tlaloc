package io.tlaloc.benchmarks

import io.tlaloc.runtime.iree.IreeBenchmark
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.IreeRuntime
import io.tlaloc.runtime.iree.IreeTarget
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.293 — performance harness extension. Times the LlamaDecoder forward
 * pass through Tlaloc-IREE's CPU and CUDA targets via `iree-benchmark-module`,
 * which runs the warmup + measurement loop inside its own process so per-
 * iteration timing reflects actual compute (not JVM↔subprocess RTT).
 *
 * The test produces [HeadToHeadResult]-shaped rows with framework labels
 * `tlaloc-iree-cpu` and `tlaloc-iree-cuda`, and dumps them under
 * `benchmarks/build/harness-results-tlaloc-iree-{cpu,cuda}.{csv,json}` for
 * the cross-framework aggregator. Closes M9's 4-row comparison story:
 * `tlaloc-interpreter`, `tlaloc-iree-cpu`, `tlaloc-iree-cuda`, and the
 * Python-reference rows from the existing `harness/python/run_pytorch.py`
 * pipeline.
 *
 * Runtime is dominated by iree-benchmark-module's internal repetitions
 * (5 reps × ~0.5 s min-time = ~2.5 s per benchmark). Self-skips when IREE
 * binaries / CUDA device unavailable.
 */
class LlamaDecoderIreeBenchmarkTest {

    private fun benchmarkLlamaForward(target: IreeTarget): HeadToHeadResult {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val coarsened = io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns(raw, io.tlaloc.ir.recognizer.recognizeAll(raw))
        val decomposed = io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened(coarsened)
        val mlir = decomposed.toStablehlo("")
        val module = IreeRuntime.compile(mlir, target)

        val inputs = llamaSynthesizeInputs(seed = 42L, decomposed)
        val textualInputs = decomposed.params.zip(inputs).map { (p, arr) ->
            val shapePrefix = if (p.type.dims.isEmpty()) "" else p.type.dims.joinToString("x") + "x"
            shapePrefix + "f32=" + arr.joinToString(",")
        }

        val stats = IreeBenchmark.run(
            module = module,
            function = decomposed.name,
            inputs = textualInputs,
            repetitions = 5,
            minTimePerRep = "0.5s",
        )

        val frameworkLabel = when (target) {
            IreeTarget.LlvmCpu -> "tlaloc-iree-cpu"
            IreeTarget.Cuda -> "tlaloc-iree-cuda"
        }

        // Sanity: get the forward loss from a single dispatch so the result
        // row carries the same loss the §0.4.289 numerical-correctness test
        // pinned (-177.062). Costs one extra subprocess but ensures the row
        // surfaces the loss alongside the timing.
        val outs = io.tlaloc.runtime.iree.runOnIree(decomposed, inputs, target)
        val forwardValue = outs.single().single()

        return HeadToHeadResult(
            benchmark = "llama-decoder-tiny-forward",
            forwardValue = forwardValue,
            gradientValues = emptyList(),
            warmupIterations = 0,
            measuredIterations = stats.iterationsPerRepetition.toInt(),
            medianNanos = stats.medianNanos,
            minNanos = stats.minNanos,
            p99Nanos = stats.p99Nanos,
            framework = frameworkLabel,
        )
    }

    private fun dump(result: HeadToHeadResult, suffix: String) {
        val outputDir = java.io.File("build")
        outputDir.mkdirs()
        java.io.File(outputDir, "harness-results-tlaloc-iree-$suffix.csv").writeText(
            buildString {
                append("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns\n")
                append(result.benchmark); append(',')
                append(result.framework); append(',')
                append(result.measuredIterations); append(',')
                append(result.medianNanos); append(',')
                append(result.minNanos); append(',')
                append(result.p99Nanos); append('\n')
            },
        )
        java.io.File(outputDir, "harness-results-tlaloc-iree-$suffix.json").writeText(
            "[\n  ${result.toJsonString().lines().joinToString("\n  ")}\n]\n",
        )
    }

    @Test
    fun benchmarksLlamaForwardOnIreeCpu() {
        assumeTrue(
            IreeBinaries.available && IreeBinaries.ireeBenchmarkModule != null,
            "iree-compile / iree-run-module / iree-benchmark-module not resolved — skipping.",
        )
        val r = benchmarkLlamaForward(IreeTarget.LlvmCpu)
        println(
            "[llama-iree-cpu] median=${r.medianNanos / 1_000} us min=${r.minNanos / 1_000} us " +
                "p99=${r.p99Nanos / 1_000} us iters/rep=${r.measuredIterations} loss=${r.forwardValue}",
        )
        assertTrue(r.medianNanos > 0, "median timing must be positive")
        assertTrue(r.minNanos <= r.medianNanos, "min must be ≤ median")
        assertTrue(r.medianNanos <= r.p99Nanos, "median must be ≤ p99")
        dump(r, "cpu")
    }

    @Test
    fun benchmarksLlamaForwardOnIreeCuda() {
        assumeTrue(
            IreeBinaries.available && IreeBinaries.ireeBenchmarkModule != null,
            "iree-compile / iree-benchmark-module not resolved — skipping.",
        )
        assumeTrue(
            IreeBinaries.cudaAvailable,
            "no CUDA device detected via `nvidia-smi -L` — skipping CUDA benchmark.",
        )
        val r = benchmarkLlamaForward(IreeTarget.Cuda)
        println(
            "[llama-iree-cuda] median=${r.medianNanos / 1_000} us min=${r.minNanos / 1_000} us " +
                "p99=${r.p99Nanos / 1_000} us iters/rep=${r.measuredIterations} loss=${r.forwardValue}",
        )
        assertTrue(r.medianNanos > 0, "median timing must be positive")
        assertTrue(r.minNanos <= r.medianNanos, "min must be ≤ median")
        assertTrue(r.medianNanos <= r.p99Nanos, "median must be ≤ p99")
        dump(r, "cuda")
    }
}
