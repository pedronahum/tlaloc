package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.runtime.iree.IreeBenchmark
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.IreeRuntime
import io.tlaloc.runtime.iree.IreeTarget
import io.tlaloc.runtime.iree.NpyWriter
import io.tlaloc.runtime.iree.runOnIree
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.295 — backward timing for the medium LlamaDecoder. Builds the same
 * primal §0.4.294 timed for forward, then runs `recognize → coarsen →
 * DxirReverseTransform.apply → decomposeCoarsened` to produce the 13-output
 * gradient function (one ∂loss/∂param per param). Times CPU and CUDA dispatch
 * via `iree-benchmark-module` against `.npy` inputs.
 *
 * The backward function recomputes forward primitives inside coarsener
 * gradient bodies (RmsNorm recomputes mean(x²); CrossEntropy recomputes
 * softmax) — so wall-clock backward is typically 2–3× wall-clock forward,
 * not 1× (no value-and-grad fusion in this baseline pipeline).
 *
 * Self-skips when IREE binaries / iree-benchmark-module / CUDA missing.
 */
class LlamaDecoderIreeMediumBackwardBenchmarkTest {

    private fun stage(target: IreeTarget, workDir: java.nio.file.Path): Triple<
        io.tlaloc.ir.DxirFunction,
        io.tlaloc.runtime.iree.IreeModule,
        List<String>,
    > {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val grad = DxirReverseTransform.apply(coarsened)
        val gradDecomposed = decomposeCoarsened(grad)
        val mlir = gradDecomposed.toStablehlo("")
        val module = IreeRuntime.compile(mlir, target, timeoutSeconds = 600)

        // The gradient fn carries the same params as the primal (same names,
        // same types) — synthesizeInputs keys off param names so it works
        // unchanged.
        val inputs = llamaSynthesizeInputs(seed = 42L, gradDecomposed)
        val npyArgs = gradDecomposed.params.zip(inputs).mapIndexed { i, (p, arr) ->
            val target = workDir.resolve("input_${"%03d".format(i)}_${p.name}.npy")
            NpyWriter.writeFloat32(target, arr, p.type.dims)
            target.toFile().deleteOnExit()
            "@$target"
        }
        return Triple(gradDecomposed, module, npyArgs)
    }

    private fun benchmark(target: IreeTarget): HeadToHeadResult {
        val workDir = Files.createTempDirectory("tlaloc-llama-medium-grad-")
        workDir.toFile().deleteOnExit()
        val (fn, module, npyArgs) = stage(target, workDir)

        // Sanity: one full backward dispatch via runOnIree (with .npy
        // marshalling) proves the gradient function compiles and dispatches
        // on this target. iree-benchmark-module would also surface a runtime
        // failure but at coarser resolution.
        val inputs = llamaSynthesizeInputs(seed = 42L, fn)
        val outs = runOnIree(fn, inputs, target = target, timeoutSeconds = 600, useNpyInputs = true)
        assertEquals(13, outs.size, "AD function must return 13 gradient tensors")
        for ((i, g) in outs.withIndex()) {
            for ((j, v) in g.withIndex()) {
                assertTrue(
                    v.isFinite(),
                    "gradient[${fn.params[i].name}][$j] must be finite; got $v on $target",
                )
            }
        }

        val stats = IreeBenchmark.run(
            module = module,
            function = fn.name,
            inputs = npyArgs,
            repetitions = 3,
            // Medium-CPU backward is ~30–50 ms; medium-CUDA backward is
            // ~10–15 ms. 0.5 s min-time gives ≥10 reps even on the slow path.
            minTimePerRep = "0.5s",
            timeoutSeconds = 600,
        )

        val frameworkLabel = when (target) {
            IreeTarget.LlvmCpu -> "tlaloc-iree-cpu"
            IreeTarget.Cuda -> "tlaloc-iree-cuda"
        }
        // The backward fn returns 13 tensors (not a scalar loss), so we don't
        // populate forwardValue with anything meaningful. Set to NaN so any
        // downstream consumer that misreads this row's `forwardValue` knows.
        return HeadToHeadResult(
            benchmark = "llama-decoder-medium-backward",
            forwardValue = Float.NaN,
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
        java.io.File(outputDir, "harness-results-tlaloc-iree-medium-backward-$suffix.csv").writeText(
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
        java.io.File(outputDir, "harness-results-tlaloc-iree-medium-backward-$suffix.json").writeText(
            "[\n  ${result.toJsonString().lines().joinToString("\n  ")}\n]\n",
        )
    }

    @Test
    fun benchmarksLlamaMediumBackwardOnIreeCpu() {
        assumeTrue(
            IreeBinaries.available && IreeBinaries.ireeBenchmarkModule != null,
            "IREE binaries / iree-benchmark-module not resolved — skipping.",
        )
        val r = benchmark(IreeTarget.LlvmCpu)
        println(
            "[llama-medium-grad-iree-cpu] median=${r.medianNanos / 1_000} us min=${r.minNanos / 1_000} us " +
                "p99=${r.p99Nanos / 1_000} us iters/rep=${r.measuredIterations}",
        )
        assertTrue(r.medianNanos > 0)
        dump(r, "cpu")
    }

    @Test
    fun benchmarksLlamaMediumBackwardOnIreeCuda() {
        assumeTrue(
            IreeBinaries.available && IreeBinaries.ireeBenchmarkModule != null,
            "IREE binaries / iree-benchmark-module not resolved — skipping.",
        )
        assumeTrue(
            IreeBinaries.cudaAvailable,
            "no CUDA device detected via `nvidia-smi -L` — skipping CUDA backward benchmark.",
        )
        val r = benchmark(IreeTarget.Cuda)
        println(
            "[llama-medium-grad-iree-cuda] median=${r.medianNanos / 1_000} us min=${r.minNanos / 1_000} us " +
                "p99=${r.p99Nanos / 1_000} us iters/rep=${r.measuredIterations}",
        )
        assertTrue(r.medianNanos > 0)
        dump(r, "cuda")
    }
}
