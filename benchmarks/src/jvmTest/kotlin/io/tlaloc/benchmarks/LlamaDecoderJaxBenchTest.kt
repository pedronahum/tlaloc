package io.tlaloc.benchmarks

import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.NpyWriter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.297 — orchestrates `harness/python/run_jax_llama_bench.py` to produce
 * JAX-GPU forward + backward timings on the LlamaDecoder medium config.
 * JAX/XLA is the closest neighbour to Tlaloc's IREE-CUDA path (both go
 * through the StableHLO/XLA compiler family), so this row is the most
 * directly meaningful comparison in the four-row matrix.
 *
 * Self-skips when:
 *   - the venv'd python is missing,
 *   - python can't `import jax`,
 *   - JAX can't see a CUDA / GPU device,
 *   - the bench script isn't where we expect.
 *
 * The Kotlin side reuses the §0.4.296 `.npy` input layout so the JAX run
 * sees byte-identical inputs to the PyTorch + Tlaloc rows.
 */
class LlamaDecoderJaxBenchTest {

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
        Path.of("..", "harness", "python", "run_jax_llama_bench.py").toAbsolutePath().normalize()

    private fun stageInputs(workDir: Path): Path {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val inputs = llamaSynthesizeInputs(seed = 42L, raw)
        for ((i, p) in raw.params.withIndex()) {
            val target = workDir.resolve("${p.name}.npy")
            NpyWriter.writeFloat32(target, inputs[i], p.type.dims)
            target.toFile().deleteOnExit()
        }
        return workDir
    }

    private fun parseResults(raw: String, framework: String): Pair<HeadToHeadResult, HeadToHeadResult> {
        fun extract(section: String): HeadToHeadResult {
            val key = "\"$section\""
            val si = raw.indexOf(key)
            require(si >= 0) { "JAX bench JSON missing section '$section': $raw" }
            val open = raw.indexOf('{', si)
            val close = raw.indexOf('}', open)
            require(open >= 0 && close > open) { "malformed JSON section '$section': $raw" }
            val body = raw.substring(open + 1, close)
            fun field(name: String): Long {
                val k = "\"$name\""
                val i = body.indexOf(k)
                require(i >= 0) { "section '$section' missing '$name': $body" }
                val colon = body.indexOf(':', i + k.length)
                var end = colon + 1
                while (end < body.length && body[end] != ',' && body[end] != '}') end++
                return body.substring(colon + 1, end).trim().toLong()
            }
            return HeadToHeadResult(
                benchmark = "llama-decoder-medium-$section",
                forwardValue = Float.NaN,
                gradientValues = emptyList(),
                warmupIterations = 0,
                measuredIterations = field("n_iterations").toInt(),
                medianNanos = field("median_ns"),
                minNanos = field("min_ns"),
                p99Nanos = field("p99_ns"),
                framework = framework,
            )
        }
        return extract("forward") to extract("backward")
    }

    private fun dump(forward: HeadToHeadResult, backward: HeadToHeadResult, suffix: String) {
        val outputDir = java.io.File("build")
        outputDir.mkdirs()
        java.io.File(outputDir, "harness-results-$suffix.csv").writeText(
            buildString {
                append("benchmark,framework,n_iterations,median_ns,min_ns,p99_ns\n")
                for (r in listOf(forward, backward)) {
                    append(r.benchmark); append(',')
                    append(r.framework); append(',')
                    append(r.measuredIterations); append(',')
                    append(r.medianNanos); append(',')
                    append(r.minNanos); append(',')
                    append(r.p99Nanos); append('\n')
                }
            },
        )
        java.io.File(outputDir, "harness-results-$suffix.json").writeText(
            "[\n" +
                listOf(forward, backward).joinToString(",\n") { r ->
                    "  ${r.toJsonString().lines().joinToString("\n  ")}"
                } +
                "\n]\n",
        )
    }

    @Test
    fun benchmarksLlamaMediumJaxGpu() {
        // CUDA detection mirrors the Tlaloc-side IREE-CUDA tests so both
        // self-skip uniformly on hosts without an NVIDIA GPU.
        assumeTrue(
            IreeBinaries.cudaAvailable,
            "no CUDA device detected via `nvidia-smi -L` — skipping JAX-GPU bench.",
        )
        val python = resolvePythonBinary()
        assumeTrue(python != null, "no python interpreter resolved — skipping.")
        assumeTrue(pythonHasJax(python!!), "python at $python cannot `import jax` — skipping.")
        assumeTrue(
            jaxSeesGpu(python),
            "JAX is installed but cannot see a CUDA / GPU device — skipping.",
        )
        val script = resolveScriptPath()
        assumeTrue(
            Files.exists(script),
            "JAX bench script not found at $script — expected at harness/python/run_jax_llama_bench.py.",
        )

        val workDir = Files.createTempDirectory("tlaloc-llama-medium-jax-")
        workDir.toFile().deleteOnExit()
        val inputsDir = stageInputs(workDir)
        val outputJson = workDir.resolve("jax-bench.json")
        outputJson.toFile().deleteOnExit()

        val pb = ProcessBuilder(
            python,
            script.toString(),
            "--inputs-dir", inputsDir.toString(),
            "--output", outputJson.toString(),
            "--config-label", "medium",
            "--min-time-seconds", "1.5",
        )
        val stderrFile = Files.createTempFile("tlaloc-jax-stderr-", ".log")
        stderrFile.toFile().deleteOnExit()
        pb.redirectError(stderrFile.toFile())
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT)
        val proc = pb.start()
        val finished = proc.waitFor(600, TimeUnit.SECONDS)
        if (!finished) {
            proc.destroyForcibly()
            error("JAX bench subprocess timed out after 600 s")
        }
        val stderr = Files.readString(stderrFile)
        if (proc.exitValue() != 0) {
            error("JAX bench exited ${proc.exitValue()}; stderr:\n${stderr.take(4000)}")
        }
        stderr.lines().lastOrNull { it.contains("[run_jax_llama_bench]") }
            ?.let { println(it) }

        val (forward, backward) = parseResults(Files.readString(outputJson), framework = "jax-gpu")
        println(
            "[llama-medium-jax-gpu] forward median=${forward.medianNanos / 1_000} us " +
                "iters=${forward.measuredIterations} | " +
                "backward median=${backward.medianNanos / 1_000} us iters=${backward.measuredIterations}",
        )
        assertTrue(forward.medianNanos > 0)
        assertTrue(backward.medianNanos > 0)
        assertTrue(backward.medianNanos > forward.medianNanos, "backward should be slower than forward")

        dump(forward, backward, "jax-gpu-medium")
    }
}
