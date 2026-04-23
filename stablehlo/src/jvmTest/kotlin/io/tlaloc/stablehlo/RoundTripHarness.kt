package io.tlaloc.stablehlo

import java.util.concurrent.TimeUnit

internal data class TranslateResult(
    val exitCode: Int,
    val stdout: String,
    val stderr: String,
) {
    val ok: Boolean get() = exitCode == 0
}

private fun resolveOnPath(tool: String): String? {
    val pb = ProcessBuilder("/usr/bin/which", tool).redirectErrorStream(true)
    return runCatching {
        val p = pb.start()
        p.waitFor(2, TimeUnit.SECONDS)
        val out = p.inputStream.bufferedReader().readText().trim()
        if (p.exitValue() == 0 && out.isNotEmpty()) out else null
    }.getOrNull()
}

private fun runTool(bin: String, input: String, args: List<String>, timeoutSeconds: Long): TranslateResult {
    val pb = ProcessBuilder(listOf(bin) + args)
    pb.redirectErrorStream(false)
    val p = pb.start()
    p.outputStream.bufferedWriter().use { it.write(input) }
    val done = p.waitFor(timeoutSeconds, TimeUnit.SECONDS)
    if (!done) {
        p.destroyForcibly()
        error("$bin timed out after ${timeoutSeconds}s")
    }
    val stdout = p.inputStream.bufferedReader().readText()
    val stderr = p.errorStream.bufferedReader().readText()
    return TranslateResult(p.exitValue(), stdout, stderr)
}

internal object StablehloTranslate {
    val binary: String? by lazy { resolveOnPath("stablehlo-translate") }
    val available: Boolean get() = binary != null

    fun run(input: String, vararg args: String, timeoutSeconds: Long = 15): TranslateResult {
        val bin = binary ?: error("stablehlo-translate not on PATH")
        return runTool(bin, input, args.toList(), timeoutSeconds)
    }
}

/**
 * Round-trip validator for the SDY dialect. `sdy-opt` parses + verifies + prints MLIR;
 * a non-zero exit or empty output means the input was rejected. Reads from stdin when
 * invoked with `-`.
 */
internal object SdyOpt {
    val binary: String? by lazy { resolveOnPath("sdy-opt") }
    val available: Boolean get() = binary != null

    fun run(input: String, vararg args: String, timeoutSeconds: Long = 15): TranslateResult {
        val bin = binary ?: error("sdy-opt not on PATH")
        return runTool(bin, input, args.toList() + "-", timeoutSeconds)
    }
}
