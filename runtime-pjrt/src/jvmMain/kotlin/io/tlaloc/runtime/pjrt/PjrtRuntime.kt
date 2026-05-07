package io.tlaloc.runtime.pjrt

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * Compile + dispatch target for PjrtRuntime. Mirrors `IreeTarget` but for the
 * PJRT-XLA backend.
 *
 *   - [LlvmCpu] — XLA's `cpu` backend (LLVM-based codegen). Cheap to compile,
 *     useful for correctness checks / no-GPU hosts.
 *   - [Cuda]    — XLA's `cuda` backend on NVIDIA GPUs. Verified end-to-end on
 *     the GB10 Blackwell host in §0.4.299 (Tlaloc's emit lands at JAX-GPU
 *     quality through this path).
 */
enum class PjrtTarget(val device: String) {
    LlvmCpu(device = "cpu"),
    Cuda(device = "cuda"),
}

/**
 * Handle to a compiled PJRT module. v1 stores only the StableHLO MLIR text on
 * disk; the actual `LoadedExecutable` lives ephemerally inside each Python
 * subprocess invocation. A future JNI-backed PJRT runtime will replace this
 * with a long-lived in-process handle.
 */
class PjrtModule internal constructor(val mlirPath: Path, val target: PjrtTarget)

class PjrtCompileException(val process: PjrtProcessResult) :
    RuntimeException("pjrt_dispatch.py compile-or-dispatch failed (exit=${process.exitCode}):\nstderr: ${process.stderr.take(2000)}")

data class PjrtProcessResult(
    val exitCode: Int,
    val stdout: String,
    val stderr: String,
) {
    val ok: Boolean get() = exitCode == 0
}

/**
 * Subprocess facade over `harness/python/pjrt_dispatch.py`. Same architectural
 * shape as the §0.4.284 `IreeRuntime` facade but the underlying PJRT compile
 * + dispatch live inside one Python invocation per `invoke()` call.
 *
 * v1 limitations:
 *   - Per-call subprocess startup cost (~30 ms for python + jax import).
 *     Acceptable for correctness tests / one-shot dispatch; per-iteration
 *     timing should use a dedicated benchmark wrapper that invokes the
 *     timing loop inside the subprocess (mirrors `IreeBenchmark`).
 *   - Compile is not separated from invoke yet — every dispatch recompiles.
 *     A long-lived Python helper that caches `LoadedExecutable` is the
 *     natural follow-up.
 *   - F32 only. Multi-dtype lowering follows the §0.4.287 plan.
 */
object PjrtRuntime {

    private const val DEFAULT_TIMEOUT_SECONDS: Long = 300L

    /**
     * Stage the MLIR on disk for [invoke]. Compile is implicit in [invoke]
     * for v1 (each subprocess re-compiles); the [PjrtModule] handle just
     * carries the path + target for the dispatcher to consume.
     */
    fun compile(stablehloMlir: String, target: PjrtTarget = PjrtTarget.Cuda): PjrtModule {
        val workDir = Files.createTempDirectory("tlaloc-pjrt-")
        workDir.toFile().deleteOnExit()
        val mlirPath = workDir.resolve("module.mlir")
        Files.writeString(mlirPath, stablehloMlir)
        mlirPath.toFile().deleteOnExit()
        return PjrtModule(mlirPath, target)
    }

    /**
     * Dispatch [module] with [inputs] (.npy files on disk, paths in order) and
     * return the output .npy paths the dispatcher wrote.
     */
    fun invoke(
        module: PjrtModule,
        function: String = "main",
        inputs: List<Path>,
        nOutputs: Int,
        timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS,
    ): List<Path> {
        val python = PjrtBinaries.python
            ?: error("PJRT python interpreter not resolved; set TLALOC_PJRT_PYTHON or `pip install jax[cuda12]` into ~/.local/venvs/iree")

        // Build inputs / outputs list-files. The dispatcher reads these so the
        // OS argv / flagfile-size limits we hit on big llama runs don't bite.
        val workDir = Files.createTempDirectory("tlaloc-pjrt-dispatch-")
        workDir.toFile().deleteOnExit()
        val inputsList = workDir.resolve("inputs.txt")
        val outputsList = workDir.resolve("outputs.txt")
        val outputPaths = (0 until nOutputs).map { i ->
            val p = workDir.resolve("output_${"%03d".format(i)}.npy")
            p.toFile().deleteOnExit()
            p
        }
        Files.writeString(inputsList, inputs.joinToString("\n") { it.toString() } + "\n")
        Files.writeString(outputsList, outputPaths.joinToString("\n") { it.toString() } + "\n")

        val script = scriptPath()
        require(Files.exists(script)) {
            "PJRT dispatcher script not found at $script — expected at harness/python/pjrt_dispatch.py"
        }

        val args = listOf(
            python,
            script.toString(),
            "--mlir", module.mlirPath.toString(),
            "--function", function,
            "--device", module.target.device,
            "--inputs-list", inputsList.toString(),
            "--outputs-list", outputsList.toString(),
        )
        val result = runProcess(args, timeoutSeconds)
        if (!result.ok) throw PjrtCompileException(result)
        return outputPaths
    }

    /**
     * Locates `harness/python/pjrt_dispatch.py` relative to the JVM's working
     * directory. Gradle's default test cwd for `:runtime-pjrt:jvmTest` is
     * `<repo>/runtime-pjrt/`; from there the script is at
     * `../harness/python/pjrt_dispatch.py`. The §0.4.296 / §0.4.297 / §0.4.299
     * tests use the same convention from `:benchmarks:jvmTest`.
     */
    private fun scriptPath(): Path =
        Path.of("..", "harness", "python", "pjrt_dispatch.py").toAbsolutePath().normalize()

    private fun runProcess(args: List<String>, timeoutSeconds: Long): PjrtProcessResult {
        val stdoutFile = Files.createTempFile("tlaloc-pjrt-stdout-", ".log")
        val stderrFile = Files.createTempFile("tlaloc-pjrt-stderr-", ".log")
        stdoutFile.toFile().deleteOnExit()
        stderrFile.toFile().deleteOnExit()

        val pb = ProcessBuilder(args)
            .redirectOutput(stdoutFile.toFile())
            .redirectError(stderrFile.toFile())
        val p = pb.start()
        p.outputStream.close()
        val done = p.waitFor(timeoutSeconds, TimeUnit.SECONDS)
        if (!done) {
            p.destroyForcibly()
            error("pjrt_dispatch.py timed out after ${timeoutSeconds}s")
        }
        return PjrtProcessResult(
            exitCode = p.exitValue(),
            stdout = Files.readString(stdoutFile),
            stderr = Files.readString(stderrFile),
        )
    }
}
