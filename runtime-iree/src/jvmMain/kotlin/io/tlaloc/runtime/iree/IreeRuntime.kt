package io.tlaloc.runtime.iree

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

data class IreeProcessResult(
    val exitCode: Int,
    val stdout: String,
    val stderr: String,
) {
    val ok: Boolean get() = exitCode == 0
}

class IreeCompileException(
    val mlirSource: String,
    val process: IreeProcessResult,
) : RuntimeException(
    "iree-compile failed (exit=${process.exitCode}):\n" +
        "stderr: ${process.stderr.take(2000)}",
)

class IreeRunException(
    val function: String,
    val inputs: List<String>,
    val process: IreeProcessResult,
) : RuntimeException(
    "iree-run-module failed (exit=${process.exitCode}) for @$function with inputs $inputs:\n" +
        "stderr: ${process.stderr.take(2000)}",
)

/**
 * Compile + dispatch target. Carries both the `iree-compile` flags that select
 * the codegen backend and the `iree-run-module` device string that drives
 * dispatch. New targets land here as enum entries — the bridge code stays
 * target-agnostic.
 *
 *   - [LlvmCpu] — local CPU dispatch via the `llvm-cpu` codegen backend and
 *     the `local-task` task system. The default.
 *   - [Cuda] — NVIDIA GPU dispatch via the `cuda` backend. Verified to compile
 *     and dispatch the LlamaDecoder MLIR on GB10 Blackwell out of the box —
 *     no sm_100-specific flags needed for the wheel-shipped IREE 3.11.0.
 */
enum class IreeTarget(val compileFlags: List<String>, val device: String) {
    LlvmCpu(
        compileFlags = listOf(
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ),
        device = "local-task",
    ),
    Cuda(
        compileFlags = listOf("--iree-hal-target-device=cuda"),
        device = "cuda",
    ),
}

/**
 * Handle to a compiled VMFB in its own temporary directory. Reuse one instance for
 * repeated invocations rather than recompiling, and [close] it when done: that
 * deletes the directory. A module never closed is removed at JVM exit. The handle
 * carries its [target] so [IreeRuntime.invoke] knows which device string to pass —
 * an [IreeModule] compiled for one backend can't be dispatched on another.
 */
class IreeModule internal constructor(val vmfbPath: Path, val target: IreeTarget) : AutoCloseable {
    /** Deletes the VMFB and its directory. Idempotent. */
    override fun close() {
        IreeRuntime.deleteRecursively(vmfbPath.parent)
    }
}

/**
 * Subprocess-based facade over `iree-compile` and `iree-run-module`. Phase 1 of the
 * IREE port plan (§0.4.230) chose ProcessBuilder over JNI to keep the first slice small;
 * §0.4.284 landed that slice for CPU dispatch, §0.4.290 broadens it to CUDA via [IreeTarget].
 *
 * Inputs and outputs are passed as IREE's textual `--input='f32=…'` form (per-input
 * marshalling lives in [IreeBridge]'s `runOnIree`).
 */
object IreeRuntime {

    private const val DEFAULT_TIMEOUT_SECONDS: Long = 60L

    fun compile(
        stablehloMlir: String,
        target: IreeTarget = IreeTarget.LlvmCpu,
        timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS,
        extraCompileFlags: List<String> = emptyList(),
    ): IreeModule {
        val bin = IreeBinaries.requireTool("iree-compile", IreeBinaries.ireeCompile)

        val workDir = Files.createTempDirectory("tlaloc-iree-")
        val mlirPath = workDir.resolve("module.mlir")
        val vmfbPath = workDir.resolve("module.vmfb")
        var ok = false
        try {
            // §0.4.300 — `extraCompileFlags` lets callers tune fusion / codegen
            // knobs (e.g. `--iree-dispatch-creation-enable-aggressive-fusion=true`)
            // without forking the IreeTarget enum. They land *after* the target's
            // base flags so a caller can also override target-default flags if
            // they really want to (last writer wins in iree-compile's flag parser).
            val args = buildList {
                add(bin)
                addAll(target.compileFlags)
                addAll(extraCompileFlags)
                add("--iree-input-type=stablehlo")
                add(mlirPath.toString())
                add("-o"); add(vmfbPath.toString())
            }
            Files.writeString(mlirPath, stablehloMlir)
            val result = runProcess(args, stdin = null, timeoutSeconds = timeoutSeconds)
            if (!result.ok) throw IreeCompileException(stablehloMlir, result)
            Files.deleteIfExists(mlirPath)
            // Backstop for a module its caller never closes. Registered directory
            // first, file second: deleteOnExit runs in reverse order.
            workDir.toFile().deleteOnExit()
            vmfbPath.toFile().deleteOnExit()
            ok = true
            return IreeModule(vmfbPath, target)
        } finally {
            if (!ok) deleteRecursively(workDir)
        }
    }

    /**
     * Invokes `function` on `module` with `inputs` formatted as IREE textual values
     * (e.g. `"f32=2.0"`, `"4xf32=1.0,2.0,3.0,4.0"`). Returns the raw `result[i]:` lines'
     * type-and-value tail (e.g. `"f32=3"`).
     *
     * Device selection is read from `module.target.device` — the module knows what it
     * was compiled for and can't be dispatched on another backend.
     *
     * Inputs are passed via `--flagfile=<temp>` rather than inline `--input=...` args,
     * so a llama-shaped invocation with ~100K floats per weight matrix doesn't bust the
     * OS argv limit (`error=7, Argument list too long`). The flagfile is a temp
     * file with one `--input=<value>` line per input, deleted when the call returns.
     */
    fun invoke(
        module: IreeModule,
        function: String = "main",
        inputs: List<String> = emptyList(),
        timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS,
    ): List<String> {
        val bin = IreeBinaries.requireTool("iree-run-module", IreeBinaries.ireeRunModule)

        val flagfile = Files.createTempFile("tlaloc-iree-flagfile-", ".txt")
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
                // §0.4.292 — disable the default 1024-element output truncation. Gradient
                // tensors hit ~16 K elements; without this, parseOutput would see truncated
                // (`...`-terminated) lines and fail with a precise "expected N got M"
                // size mismatch — but on success the data would be silently wrong.
                "--output_max_element_count=2147483647",
                "--flagfile=$flagfile",
            )
            val result = runProcess(args, stdin = null, timeoutSeconds = timeoutSeconds)
            if (!result.ok) throw IreeRunException(function, inputs, result)
            return parseOutputs(result.stdout)
        } finally {
            Files.deleteIfExists(flagfile)
        }
    }

    private fun runProcess(args: List<String>, stdin: String?, timeoutSeconds: Long): IreeProcessResult {
        // Redirect stdout/stderr to temp files rather than reading them after waitFor.
        // Without this, when a subprocess writes more than the OS pipe buffer (~64 KB
        // on Linux) without anything draining the JVM-side pipe, the subprocess blocks
        // on write and waitFor deadlocks. iree-run-module's textual gradient output for
        // the LlamaDecoder backward (~116 KB across 13 tensors) hit this exact pattern
        // (§0.4.292). Files bypass the pipe entirely.
        val stdoutFile = Files.createTempFile("tlaloc-iree-stdout-", ".log")
        val stderrFile = Files.createTempFile("tlaloc-iree-stderr-", ".log")
        try {
            val pb = ProcessBuilder(args)
                .redirectOutput(stdoutFile.toFile())
                .redirectError(stderrFile.toFile())
            val p = pb.start()
            if (stdin != null) p.outputStream.bufferedWriter().use { it.write(stdin) } else p.outputStream.close()
            val done = p.waitFor(timeoutSeconds, TimeUnit.SECONDS)
            if (!done) {
                p.destroyForcibly()
                error("${args.first()} timed out after ${timeoutSeconds}s")
            }
            val stdout = Files.readString(stdoutFile)
            val stderr = Files.readString(stderrFile)
            return IreeProcessResult(p.exitValue(), stdout, stderr)
        } finally {
            Files.deleteIfExists(stdoutFile)
            Files.deleteIfExists(stderrFile)
        }
    }

    /** Deletes [dir] and everything under it; missing paths are ignored. */
    internal fun deleteRecursively(dir: Path) {
        if (!Files.exists(dir)) return
        Files.walk(dir).use { paths ->
            paths.sorted(Comparator.reverseOrder()).forEach { runCatching { Files.deleteIfExists(it) } }
        }
    }

    private val resultLineRegex = Regex("""^result\[\d+]:\s*hal\.buffer_view\s*\R(.+)$""", RegexOption.MULTILINE)

    private fun parseOutputs(stdout: String): List<String> =
        resultLineRegex.findAll(stdout).map { it.groupValues[1].trim() }.toList()
}
