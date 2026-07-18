package io.tlaloc.benchmarks

import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.325 — **the negative "JAX can't" pin.**
 *
 * Counterpart to [LlamaDecoderCustomCallEmitTest]: feed Tlaloc's
 * GB10-target MLIR (containing `stablehlo.custom_call @flash_attn_v3`)
 * to JAX/PJRT-XLA's compile path and pin that compilation fails because
 * `@flash_attn_v3` isn't a registered custom-call symbol in JAX/XLA.
 *
 * As a positive control, the same test compiles Tlaloc's CPU_GENERIC
 * MLIR (zero custom_calls) through the same pipeline and pins that it
 * succeeds — proving the failure is specific to the unrecognized kernel
 * symbol, not a generic Tlaloc-MLIR-vs-JAX-parser issue.
 *
 * Self-skips when CUDA / Python / JAX is unavailable.
 *
 * # Why this is the evidence
 *
 * "JAX doesn't pick across kernel families per-device" is structural,
 * not opinion. JAX / XLA ships its own attention codegen; it does not
 * dispatch on a `flash_attn_v3` symbol. When Tlaloc emits MLIR that
 * names that symbol, JAX literally cannot compile it. Same Kotlin
 * source compiled for `CPU_GENERIC` produces MLIR that *does* compile,
 * because no custom-call symbol is named.
 *
 * Phase 2 of the custom_call rollout (PJRT custom-call symbol
 * registration via FFM) will close the loop: at that point Tlaloc-PJRT
 * dispatches the same MLIR that this test pins as JAX-rejected.
 */
class JaxRejectsTlalocCustomCallMlirTest {

    private fun resolvePythonBinary(): String? {
        System.getenv("TLALOC_TORCH_PYTHON")?.let { p ->
            if (Files.isExecutable(Path.of(p))) return p
        }
        val home = System.getProperty("user.home") ?: return null
        val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
        return if (Files.isExecutable(venv)) venv.toString() else null
    }

    /**
     * GB10 is unified-memory: JAX's default 75% preallocation would
     * claim ~90 GB of system RAM per probe subprocess. Applied to every
     * python invocation this test spawns.
     */
    private fun ProcessBuilder.withJaxNoPrealloc(): ProcessBuilder = apply {
        environment()["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    }

    private fun pythonHasJax(python: String): Boolean {
        val pb = ProcessBuilder(python, "-c", "import jax").withJaxNoPrealloc()
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
        ).withJaxNoPrealloc()
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(30, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun resolveScriptPath(): Path =
        Path.of("..", "harness", "python", "run_jax_compile_check.py").toAbsolutePath().normalize()

    private data class CheckResult(
        val exitCode: Int,
        val compiled: Boolean,
        val stage: String,
        val error: String?,
        val stdout: String,
    )

    private fun runCheck(python: String, script: Path, mlir: String, workDir: Path): CheckResult {
        val mlirFile = workDir.resolve("input.mlir")
        Files.writeString(mlirFile, mlir)
        val outFile = workDir.resolve("result.json")

        val pb = ProcessBuilder(
            python,
            script.toString(),
            "--mlir", mlirFile.toString(),
            "--output", outFile.toString(),
        ).withJaxNoPrealloc()
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val stdout = proc.inputStream.bufferedReader().readText()
        val finished = proc.waitFor(120, TimeUnit.SECONDS)
        require(finished) { "run_jax_compile_check timed out\nstdout:\n$stdout" }

        // The script always writes its JSON, even on exit=1.
        val json = Files.readString(outFile)
        val compiled = "\"compiled\": true" in json
        val stage = Regex("\"stage\":\\s*\"([^\"]*)\"").find(json)?.groupValues?.get(1) ?: "unknown"
        val error = Regex("\"error\":\\s*(?:\"((?:[^\"\\\\]|\\\\.)*)\"|null)")
            .find(json)?.groupValues?.get(1)?.takeIf { it.isNotEmpty() }
        return CheckResult(proc.exitValue(), compiled, stage, error, stdout)
    }

    @Test
    fun jaxRejectsGb10ArtifactWithFlashAttnCustomCall() {
        assumeTrue(IreeBinaries.cudaAvailable, "no CUDA device — skipping.")
        val python = resolvePythonBinary()
        assumeTrue(python != null, "no python interpreter resolved — skipping.")
        assumeTrue(pythonHasJax(python!!), "python at $python cannot `import jax` — skipping.")
        assumeTrue(jaxSeesGpu(python), "JAX cannot see a CUDA / GPU device — skipping.")
        val script = resolveScriptPath()
        assumeTrue(Files.exists(script), "compile-check script not found at $script — skipping.")

        val workDir = Files.createTempDirectory("tlaloc-jax-rejects-")
        workDir.toFile().deleteOnExit()

        // Generate Tlaloc's GB10 artifact — contains `@flash_attn_v3`.
        val gb10Mlir = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.NVIDIA_GB10,
        ).toStablehlo("")
        require("stablehlo.custom_call @flash_attn_v3" in gb10Mlir) {
            "GB10 pipeline must emit a flash_attn_v3 custom_call for this test to be meaningful"
        }
        val gb10Result = runCheck(python, script, gb10Mlir, workDir)

        // Skip (don't fail) if JAX failed for an environmental reason
        // — keeps the test honest in CI environments where JAX is
        // available but not actually capable of compile (driver mismatch
        // etc.).
        assumeTrue(
            gb10Result.stage !in setOf("import", "device", "input"),
            "JAX environment broken (stage=${gb10Result.stage}, error=${gb10Result.error}) — skipping.",
        )

        assertEquals(
            1, gb10Result.exitCode,
            "JAX compile of Tlaloc's GB10 MLIR must fail (exit=1); got exit=${gb10Result.exitCode}, " +
                "stage=${gb10Result.stage}, error=${gb10Result.error}",
        )
        assertTrue(
            !gb10Result.compiled,
            "JAX must NOT successfully compile MLIR naming `@flash_attn_v3` " +
                "(JAX/XLA does not ship that custom-call symbol)",
        )
        assertTrue(
            gb10Result.stage == "compile" || gb10Result.stage == "parse",
            "expected failure at parse or compile stage; got stage=${gb10Result.stage}",
        )
        // The error string from XLA mentions the unregistered call_target_name.
        // Soft check — XLA's exact wording can change across versions; we just
        // confirm the error is non-trivial.
        assertTrue(
            gb10Result.error != null && gb10Result.error.isNotBlank(),
            "expected a non-empty XLA error message",
        )

        println(
            "[jax-rejects-tlaloc-mlir] GB10 artifact rejected as expected: " +
                "stage=${gb10Result.stage}, error excerpt=${gb10Result.error.take(200)}",
        )

        // Positive control: same source, CPU_GENERIC target → no custom_call → JAX accepts.
        val cpuMlir = llamaKernelLoweredForwardPipeline(
            LlamaDecoderConfig.medium, KernelTarget.CPU_GENERIC,
        ).toStablehlo("")
        require("stablehlo.custom_call" !in cpuMlir) {
            "CPU pipeline must emit no custom_calls for the positive control to be meaningful"
        }
        val cpuResult = runCheck(python, script, cpuMlir, workDir)
        assertEquals(
            0, cpuResult.exitCode,
            "JAX must successfully compile Tlaloc's CPU_GENERIC MLIR (no custom_calls); " +
                "got exit=${cpuResult.exitCode}, stage=${cpuResult.stage}, error=${cpuResult.error}",
        )
        assertTrue(
            cpuResult.compiled,
            "JAX should accept the CPU artifact — same Kotlin source, decomposed primitives only",
        )

        println(
            "[jax-rejects-tlaloc-mlir] CPU artifact accepted as expected: " +
                "stage=${cpuResult.stage} — proves the rejection is specific to the kernel symbol, " +
                "not a generic Tlaloc-MLIR vs JAX-parser issue.",
        )
    }
}
