package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.NpyWriter
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.299 — orchestrates `harness/python/run_pjrt_xla_llama_bench.py`. The
 * spike question this answers: when Tlaloc's *exact* StableHLO MLIR
 * (forward + backward) is fed to the same PJRT-XLA backend JAX uses
 * internally, does it land at JAX-GPU's ~670 µs forward, or at
 * Tlaloc-IREE-CUDA's ~3 750 µs?
 *
 * If PJRT-XLA-on-Tlaloc-MLIR ≈ JAX-GPU → the gap is in IREE's GPU codegen
 * (lever: tune IREE flags, re-engage kernel_descriptor for CUDA).
 * If it lands close to IREE-CUDA → the gap is in Tlaloc's IR shape (lever:
 * fused-coarsener emit, kernel_descriptor metadata).
 *
 * Self-skips when no CUDA, no python, no jax, or no bench script.
 */
class LlamaDecoderPjrtXlaBenchTest {

    private fun resolvePythonBinary(): String? {
        System.getenv("TLALOC_TORCH_PYTHON")?.let { p ->
            if (Files.isExecutable(Path.of(p))) return p
        }
        val home = System.getProperty("user.home")
        if (home != null) {
            val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
            if (Files.isExecutable(venv)) return venv.toString()
        }
        return null
    }

    private fun pythonHasJax(python: String): Boolean {
        val pb = ProcessBuilder(python, "-c", "import jax")
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(20, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun jaxSeesGpu(python: String): Boolean {
        val pb = ProcessBuilder(
            python, "-c",
            "import jax,sys; sys.exit(0 if any(d.platform in ('cuda','gpu') for d in jax.devices()) else 1)",
        )
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(30, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun resolveScriptPath(): Path =
        Path.of("..", "harness", "python", "run_pjrt_xla_llama_bench.py").toAbsolutePath().normalize()

    @Test
    fun benchmarksLlamaMediumPjrtXlaCuda() {
        assumeTrue(IreeBinaries.cudaAvailable, "no CUDA device — skipping.")
        val python = resolvePythonBinary()
        assumeTrue(python != null, "no python interpreter resolved — skipping.")
        assumeTrue(pythonHasJax(python!!), "python at $python cannot `import jax` — skipping.")
        assumeTrue(jaxSeesGpu(python), "JAX cannot see a CUDA / GPU device — skipping.")
        val script = resolveScriptPath()
        assumeTrue(Files.exists(script), "spike script not found at $script — skipping.")

        val workDir = Files.createTempDirectory("tlaloc-llama-medium-pjrt-")
        workDir.toFile().deleteOnExit()

        // Build forward + backward DxirFunctions, dump their StableHLO emit.
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val fwd = decomposeCoarsened(coarsened)
        val grad = DxirReverseTransform.apply(coarsened)
        val bwd = decomposeCoarsened(grad)

        val fwdMlirPath = workDir.resolve("forward.mlir")
        val bwdMlirPath = workDir.resolve("backward.mlir")
        Files.writeString(fwdMlirPath, fwd.toStablehlo(""))
        Files.writeString(bwdMlirPath, bwd.toStablehlo(""))

        // Same Kotlin-seed-42 inputs the §0.4.296 / §0.4.297 benches use → npy.
        val inputs = llamaSynthesizeInputs(seed = 42L, fwd)
        for ((i, p) in fwd.params.withIndex()) {
            val target = workDir.resolve("${p.name}.npy")
            NpyWriter.writeFloat32(target, inputs[i], p.type.dims)
            target.toFile().deleteOnExit()
        }

        val outputJson = workDir.resolve("pjrt-bench.json")
        outputJson.toFile().deleteOnExit()
        val pb = ProcessBuilder(
            python,
            script.toString(),
            "--inputs-dir", workDir.toString(),
            "--forward-mlir", fwdMlirPath.toString(),
            "--backward-mlir", bwdMlirPath.toString(),
            "--output", outputJson.toString(),
            "--config-label", "medium",
            "--min-time-seconds", "1.5",
        )
        val stderrFile = Files.createTempFile("tlaloc-pjrt-stderr-", ".log")
        stderrFile.toFile().deleteOnExit()
        pb.redirectError(stderrFile.toFile())
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT)
        val proc = pb.start()
        val finished = proc.waitFor(900, TimeUnit.SECONDS)
        if (!finished) {
            proc.destroyForcibly()
            error("PJRT-XLA spike timed out after 900 s")
        }
        val stderr = Files.readString(stderrFile)
        if (proc.exitValue() != 0) {
            error("PJRT-XLA spike exited ${proc.exitValue()}; stderr:\n${stderr.take(4000)}")
        }
        stderr.lines().lastOrNull { it.contains("[run_pjrt_xla_llama_bench]") }
            ?.let { println(it) }

        val raw2 = Files.readString(outputJson)
        fun extractField(section: String, name: String): Long {
            val sIdx = raw2.indexOf("\"$section\"")
            require(sIdx >= 0) { "missing section '$section': $raw2" }
            val open = raw2.indexOf('{', sIdx)
            val close = raw2.indexOf('}', open)
            val body = raw2.substring(open, close)
            val k = "\"$name\""
            val ki = body.indexOf(k)
            val colon = body.indexOf(':', ki + k.length)
            var end = colon + 1
            while (end < body.length && body[end] != ',' && body[end] != '}') end++
            return body.substring(colon + 1, end).trim().toLong()
        }
        val fwdMedian = extractField("forward", "median_ns")
        val fwdMin = extractField("forward", "min_ns")
        val fwdP99 = extractField("forward", "p99_ns")
        val fwdIters = extractField("forward", "n_iterations").toInt()
        val bwdMedian = extractField("backward", "median_ns")
        val bwdMin = extractField("backward", "min_ns")
        val bwdP99 = extractField("backward", "p99_ns")
        val bwdIters = extractField("backward", "n_iterations").toInt()

        println(
            "[llama-medium-pjrt-xla-cuda] forward median=${fwdMedian / 1_000} us iters=$fwdIters " +
                "| backward median=${bwdMedian / 1_000} us iters=$bwdIters",
        )
        assertTrue(fwdMedian > 0)
        assertTrue(bwdMedian > 0)
        assertTrue(bwdMedian > fwdMedian, "backward should be slower than forward")

        val outputDir = java.io.File("build")
        outputDir.mkdirs()
        java.io.File(outputDir, "harness-results-tlaloc-pjrt-xla-cuda-medium.csv").writeText(
            buildString {
                append("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns\n")
                append("llama-decoder-medium-forward,tlaloc-pjrt-xla-cuda,$fwdIters,$fwdMedian,$fwdMin,$fwdP99\n")
                append("llama-decoder-medium-backward,tlaloc-pjrt-xla-cuda,$bwdIters,$bwdMedian,$bwdMin,$bwdP99\n")
            },
        )
        // JSON in the same shape the aggregator reads.
        java.io.File(outputDir, "harness-results-tlaloc-pjrt-xla-cuda-medium.json").writeText(
            """[
  {
    "benchmark": "llama-decoder-medium-forward",
    "framework": "tlaloc-pjrt-xla-cuda",
    "forwardValue": NaN,
    "gradientValues": [],
    "warmupIterations": 0,
    "measuredIterations": $fwdIters,
    "medianNanos": $fwdMedian,
    "minNanos": $fwdMin,
    "p99Nanos": $fwdP99
  },
  {
    "benchmark": "llama-decoder-medium-backward",
    "framework": "tlaloc-pjrt-xla-cuda",
    "forwardValue": NaN,
    "gradientValues": [],
    "warmupIterations": 0,
    "measuredIterations": $bwdIters,
    "medianNanos": $bwdMedian,
    "minNanos": $bwdMin,
    "p99Nanos": $bwdP99
  }
]
""",
        )
    }
}
