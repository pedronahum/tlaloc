package io.tlaloc.benchmarks

import io.tlaloc.runtime.iree.NpyWriter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.296 — orchestrates `harness/python/run_pytorch_llama_bench.py` to
 * produce PyTorch-CPU forward + backward timings on the LlamaDecoder medium
 * config. Mirrors the Tlaloc-side §0.4.294 / §0.4.295 measurement protocol
 * (same seed, same .npy inputs, same medium config) so the numbers compare
 * apples-to-apples against `tlaloc-iree-{cpu,cuda}` rows.
 *
 * Self-skips when the venv'd python or torch is missing, or the bench
 * script isn't on disk where we expect.
 */
class LlamaDecoderPytorchBenchTest {

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

    private fun pythonHasTorch(python: String): Boolean {
        val pb = ProcessBuilder(python, "-c", "import torch")
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(15, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun resolveScriptPath(): Path =
        Path.of("..", "harness", "python", "run_pytorch_llama_bench.py").toAbsolutePath().normalize()

    /**
     * Writes inputs as `.npy` under [workDir] (one file per param, keyed by
     * name — same layout the §0.4.289 cross-language tests use). Returns the
     * inputs dir for the Python script.
     */
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

    /**
     * Parse the bench script's JSON output into two [HeadToHeadResult]s
     * (one for forward, one for backward). Trivial regex-based parser since
     * the JSON format is fixed by `run_pytorch_llama_bench.py`.
     */
    private fun parseResults(raw: String, framework: String): Pair<HeadToHeadResult, HeadToHeadResult> {
        fun extract(section: String): HeadToHeadResult {
            // Find `"<section>": {…}`; pick out median_ns, min_ns, p99_ns, n_iterations.
            val key = "\"$section\""
            val si = raw.indexOf(key)
            require(si >= 0) { "PyTorch bench JSON missing section '$section': $raw" }
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
            val mode = section
            return HeadToHeadResult(
                benchmark = "llama-decoder-medium-$mode",
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
    fun benchmarksLlamaMediumPytorchCpu() {
        val python = resolvePythonBinary()
        assumeTrue(python != null, "no python interpreter resolved — skipping.")
        assumeTrue(
            pythonHasTorch(python!!),
            "python at $python cannot `import torch` — skipping.",
        )
        val script = resolveScriptPath()
        assumeTrue(
            Files.exists(script),
            "PyTorch bench script not found at $script — expected at harness/python/run_pytorch_llama_bench.py.",
        )

        val workDir = Files.createTempDirectory("tlaloc-llama-medium-pytorch-")
        workDir.toFile().deleteOnExit()
        val inputsDir = stageInputs(workDir)
        val outputJson = workDir.resolve("pytorch-bench.json")
        outputJson.toFile().deleteOnExit()

        val pb = ProcessBuilder(
            python,
            script.toString(),
            "--inputs-dir", inputsDir.toString(),
            "--output", outputJson.toString(),
            "--config-label", "medium",
            "--min-time-seconds", "1.5",
        )
        // Redirect to file (avoid the §0.4.292 pipe-deadlock pattern even though
        // PyTorch's stderr output here is small; harmless and consistent).
        val stderrFile = Files.createTempFile("tlaloc-pytorch-stderr-", ".log")
        stderrFile.toFile().deleteOnExit()
        pb.redirectError(stderrFile.toFile())
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT)
        val proc = pb.start()
        val finished = proc.waitFor(600, TimeUnit.SECONDS)
        if (!finished) {
            proc.destroyForcibly()
            error("PyTorch bench subprocess timed out after 600 s")
        }
        val stderr = Files.readString(stderrFile)
        if (proc.exitValue() != 0) {
            error("PyTorch bench exited ${proc.exitValue()}; stderr:\n${stderr.take(4000)}")
        }
        // Echo the script's last status line to the test log for triage.
        stderr.lines().lastOrNull { it.contains("[run_pytorch_llama_bench]") }
            ?.let { println(it) }

        val (forward, backward) = parseResults(Files.readString(outputJson), framework = "pytorch-cpu")
        println(
            "[llama-medium-pytorch-cpu] forward median=${forward.medianNanos / 1_000} us " +
                "iters=${forward.measuredIterations} | " +
                "backward median=${backward.medianNanos / 1_000} us iters=${backward.measuredIterations}",
        )
        assertTrue(forward.medianNanos > 0)
        assertTrue(backward.medianNanos > 0)
        assertTrue(backward.medianNanos > forward.medianNanos, "backward should be slower than forward")

        dump(forward, backward, "pytorch-cpu-medium")
    }
}
