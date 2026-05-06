package io.tlaloc.benchmarks

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
 * §0.4.294 — performance harness scale-up. Times the LlamaDecoder forward pass
 * at the [LlamaDecoderConfig.medium] config (256 tokens × 512 dModel × 2048
 * vocab, ~1.3 GFLOPs/forward) on both IREE-CPU and IREE-CUDA. This is the
 * smallest LlamaDecoder size where CUDA's compute throughput wins over its
 * kernel-launch overhead on the GB10 Blackwell — at tiny config (§0.4.293)
 * CUDA *lost* to CPU because the workload was dominated by launch latency.
 *
 * Inputs go through `.npy` files (`--input=@<path>`) instead of the textual
 * `<shape>xf32=v0,v1,…` form used at tiny config. The medium inputs total
 * ~24 MB of raw f32; the textual form would balloon to ~240 MB, which is too
 * large for either argv or a single-line flagfile read.
 *
 * Self-skips when IREE binaries / iree-benchmark-module / CUDA are missing.
 */
class LlamaDecoderIreeMediumBenchmarkTest {

    /**
     * Build + lower the medium config to MLIR, compile via IREE for [target],
     * write the seeded inputs as `.npy` files under [workDir], and return the
     * (decomposedFunction, compiledModule, npyInputArgs) triple ready for
     * dispatch / benchmarking. Each `npyInputArgs[i]` is the `@<path>` string
     * the IREE tools accept.
     */
    private fun stage(target: IreeTarget, workDir: java.nio.file.Path): Triple<
        io.tlaloc.ir.DxirFunction,
        io.tlaloc.runtime.iree.IreeModule,
        List<String>,
    > {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val decomposed = decomposeCoarsened(coarsened)
        val mlir = decomposed.toStablehlo("")
        val module = IreeRuntime.compile(mlir, target, timeoutSeconds = 600)

        val inputs = llamaSynthesizeInputs(seed = 42L, decomposed)
        val npyArgs = decomposed.params.zip(inputs).mapIndexed { i, (p, arr) ->
            val target = workDir.resolve("input_${"%02d".format(i)}_${p.name}.npy")
            NpyWriter.writeFloat32(target, arr, p.type.dims)
            target.toFile().deleteOnExit()
            "@$target"
        }
        return Triple(decomposed, module, npyArgs)
    }

    private fun benchmark(target: IreeTarget): HeadToHeadResult {
        val workDir = Files.createTempDirectory("tlaloc-llama-medium-")
        workDir.toFile().deleteOnExit()
        val (fn, module, npyArgs) = stage(target, workDir)

        // Sanity: one dispatch via runOnIree (with .npy marshalling for the medium-
        // size inputs) proves the config compiles + runs end-to-end. The benchmark
        // tool would also surface a runtime failure but at coarser resolution.
        val inputs = llamaSynthesizeInputs(seed = 42L, fn)
        val outs = runOnIree(fn, inputs, target = target, timeoutSeconds = 600, useNpyInputs = true)
        val forwardValue = outs.single().single()
        assertTrue(forwardValue.isFinite(), "medium forward loss must be finite; got $forwardValue")

        val stats = IreeBenchmark.run(
            module = module,
            function = fn.name,
            inputs = npyArgs,
            repetitions = 3,
            // Medium CPU forward is ~10 ms; 0.5 s min-time → ~50 iterations
            // per rep, 3 reps → 150 measurements total. Plenty for stable
            // percentiles without dragging out the test run.
            minTimePerRep = "0.5s",
            timeoutSeconds = 600,
        )

        val frameworkLabel = when (target) {
            IreeTarget.LlvmCpu -> "tlaloc-iree-cpu"
            IreeTarget.Cuda -> "tlaloc-iree-cuda"
        }
        return HeadToHeadResult(
            benchmark = "llama-decoder-medium-forward",
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
        java.io.File(outputDir, "harness-results-tlaloc-iree-medium-$suffix.csv").writeText(
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
        java.io.File(outputDir, "harness-results-tlaloc-iree-medium-$suffix.json").writeText(
            "[\n  ${result.toJsonString().lines().joinToString("\n  ")}\n]\n",
        )
    }

    @Test
    fun benchmarksLlamaMediumForwardOnIreeCpu() {
        assumeTrue(
            IreeBinaries.available && IreeBinaries.ireeBenchmarkModule != null,
            "IREE binaries / iree-benchmark-module not resolved — skipping.",
        )
        val r = benchmark(IreeTarget.LlvmCpu)
        println(
            "[llama-medium-iree-cpu] median=${r.medianNanos / 1_000} us min=${r.minNanos / 1_000} us " +
                "p99=${r.p99Nanos / 1_000} us iters/rep=${r.measuredIterations} loss=${r.forwardValue}",
        )
        assertTrue(r.medianNanos > 0)
        dump(r, "cpu")
    }

    @Test
    fun benchmarksLlamaMediumForwardOnIreeCuda() {
        assumeTrue(
            IreeBinaries.available && IreeBinaries.ireeBenchmarkModule != null,
            "IREE binaries / iree-benchmark-module not resolved — skipping.",
        )
        assumeTrue(
            IreeBinaries.cudaAvailable,
            "no CUDA device detected via `nvidia-smi -L` — skipping CUDA benchmark.",
        )
        val r = benchmark(IreeTarget.Cuda)
        println(
            "[llama-medium-iree-cuda] median=${r.medianNanos / 1_000} us min=${r.minNanos / 1_000} us " +
                "p99=${r.p99Nanos / 1_000} us iters/rep=${r.measuredIterations} loss=${r.forwardValue}",
        )
        assertTrue(r.medianNanos > 0)
        dump(r, "cuda")
    }

    @Test
    fun cpuAndCudaForwardLossesAgreeAtMediumConfig() {
        // §0.4.291 pinned bit-identical loss across CPU and CUDA on tiny config;
        // re-check at medium where the larger reduction trees might diverge by
        // a few FP32 ulps. PyTorch-allclose-style tolerance.
        assumeTrue(IreeBinaries.available, "IREE binaries not resolved — skipping.")
        assumeTrue(IreeBinaries.cudaAvailable, "no CUDA device — skipping.")
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val decomposed = decomposeCoarsened(coarsened)
        val inputs = llamaSynthesizeInputs(seed = 42L, decomposed)

        val lossCpu = runOnIree(decomposed, inputs, IreeTarget.LlvmCpu, timeoutSeconds = 600, useNpyInputs = true)
            .single().single()
        val lossCuda = runOnIree(decomposed, inputs, IreeTarget.Cuda, timeoutSeconds = 600, useNpyInputs = true)
            .single().single()
        val absDiff = kotlin.math.abs(lossCpu - lossCuda)
        val relDiff = absDiff / kotlin.math.max(1.0f, kotlin.math.max(kotlin.math.abs(lossCpu), kotlin.math.abs(lossCuda)))
        println(
            "[llama-medium-cpu-vs-cuda] cpu=$lossCpu cuda=$lossCuda absDiff=$absDiff relDiff=$relDiff",
        )
        // Looser than tiny-config (where the two were bit-identical) since
        // medium has more reductions; FP32 reduction-order divergence between
        // LLVM-CPU SIMD and CUDA warp-shuffle reductions should still stay
        // sub-1e-3 on a relative basis at this size.
        assertTrue(
            relDiff < 1e-3f,
            "CPU vs CUDA loss disagreement at medium config: cpu=$lossCpu cuda=$lossCuda relDiff=$relDiff",
        )
        assertTrue(kotlin.math.abs(lossCpu) < 100_000f, "loss wildly out of band; got $lossCpu")
        assertEquals(true, lossCpu.isFinite() && lossCuda.isFinite())
    }
}
