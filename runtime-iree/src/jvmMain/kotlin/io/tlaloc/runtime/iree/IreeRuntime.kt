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
 * In-memory handle to a compiled VMFB. The artifact is materialised under the JVM temp dir
 * and removed on JVM exit; reuse the same instance for repeated invocations rather than
 * recompiling.
 */
class IreeModule internal constructor(val vmfbPath: Path)

/**
 * Subprocess-based facade over `iree-compile` and `iree-run-module`. Phase 1 of the
 * IREE port plan (§0.4.230) chose ProcessBuilder over JNI to keep the first slice small;
 * this commit lands that slice for CPU dispatch (`llvm-cpu` target, `local-task` device).
 *
 * Inputs and outputs are passed as IREE's textual `--input='f32=…'` form for now —
 * dxir tensor marshalling is the next phase.
 */
object IreeRuntime {

    private const val DEFAULT_TIMEOUT_SECONDS: Long = 60L

    fun compile(stablehloMlir: String, timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS): IreeModule {
        val bin = IreeBinaries.ireeCompile
            ?: error("iree-compile not resolved; set TLALOC_IREE_BIN or install via the dual-track plan's path A")

        val workDir = Files.createTempDirectory("tlaloc-iree-")
        workDir.toFile().deleteOnExit()
        val mlirPath = workDir.resolve("module.mlir")
        val vmfbPath = workDir.resolve("module.vmfb")
        Files.writeString(mlirPath, stablehloMlir)
        mlirPath.toFile().deleteOnExit()

        val args = listOf(
            bin,
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-input-type=stablehlo",
            mlirPath.toString(),
            "-o", vmfbPath.toString(),
        )
        val result = runProcess(args, stdin = null, timeoutSeconds = timeoutSeconds)
        if (!result.ok) throw IreeCompileException(stablehloMlir, result)
        vmfbPath.toFile().deleteOnExit()
        return IreeModule(vmfbPath)
    }

    /**
     * Invokes `function` on `module` with `inputs` formatted as IREE textual values
     * (e.g. `"f32=2.0"`, `"4xf32=1.0,2.0,3.0,4.0"`). Returns the raw `result[i]:` lines'
     * type-and-value tail (e.g. `"f32=3"`).
     */
    fun invoke(
        module: IreeModule,
        function: String = "main",
        inputs: List<String> = emptyList(),
        timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS,
    ): List<String> {
        val bin = IreeBinaries.ireeRunModule
            ?: error("iree-run-module not resolved; set TLALOC_IREE_BIN or install via the dual-track plan's path A")

        val args = buildList {
            add(bin)
            add("--module=${module.vmfbPath}")
            add("--device=local-task")
            add("--function=$function")
            inputs.forEach { add("--input=$it") }
        }
        val result = runProcess(args, stdin = null, timeoutSeconds = timeoutSeconds)
        if (!result.ok) throw IreeRunException(function, inputs, result)
        return parseOutputs(result.stdout)
    }

    private fun runProcess(args: List<String>, stdin: String?, timeoutSeconds: Long): IreeProcessResult {
        val pb = ProcessBuilder(args)
        pb.redirectErrorStream(false)
        val p = pb.start()
        if (stdin != null) p.outputStream.bufferedWriter().use { it.write(stdin) } else p.outputStream.close()
        val done = p.waitFor(timeoutSeconds, TimeUnit.SECONDS)
        if (!done) {
            p.destroyForcibly()
            error("${args.first()} timed out after ${timeoutSeconds}s")
        }
        val stdout = p.inputStream.bufferedReader().readText()
        val stderr = p.errorStream.bufferedReader().readText()
        return IreeProcessResult(p.exitValue(), stdout, stderr)
    }

    private val resultLineRegex = Regex("""^result\[\d+]:\s*hal\.buffer_view\s*\R(.+)$""", RegexOption.MULTILINE)

    private fun parseOutputs(stdout: String): List<String> =
        resultLineRegex.findAll(stdout).map { it.groupValues[1].trim() }.toList()
}
