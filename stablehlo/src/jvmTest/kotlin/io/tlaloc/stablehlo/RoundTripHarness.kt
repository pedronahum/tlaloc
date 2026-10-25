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

/**
 * `stablehlo-translate --serialize --target=V`: parse, verify and serialize a module as a
 * portable artifact. Three ways, in order:
 *  - the `stablehlo-translate` binary on PATH;
 *  - jaxlib's `stablehlo.serialize_portable_artifact_str` (the same StableHLO library, in
 *    the oracle venv), which also records each module that passes by its SHA-256
 *    (test-support Goldens, set `roundtrip/verified`) with `TLALOC_RECORD_GOLDENS=1`;
 *  - without either, that recorded set: a module passes when the exact text was verified
 *    before, and fails when it was not (emitted text changed: record again).
 * [available] is false only when there is none of the three.
 */
internal object StablehloTranslate {
    val binary: String? by lazy { resolveOnPath("stablehlo-translate") }
    private val jaxlib: String? by lazy { io.tlaloc.testing.PythonOracle.python("jaxlib.mlir.dialects.stablehlo") }
    private val recorded: Boolean by lazy { io.tlaloc.testing.Goldens.read("roundtrip", "verified") != null }
    val available: Boolean get() = binary != null || jaxlib != null || recorded

    private const val SERIALIZE = """
import sys
from jaxlib.mlir.dialects import stablehlo
try:
    b = stablehlo.serialize_portable_artifact_str(sys.stdin.read(), sys.argv[1])
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
sys.stdout.write(b.hex())
"""

    fun run(input: String, vararg args: String, timeoutSeconds: Long = 15): TranslateResult {
        binary?.let { return runTool(it, input, args.toList(), timeoutSeconds) }
        val target = args.firstOrNull { it.startsWith("--target=") }?.removePrefix("--target=")
        require(args.contains("--serialize") && target != null) {
            "StablehloTranslate: without the binary only --serialize --target=V is supported (got ${args.toList()})"
        }
        var live: TranslateResult? = null
        val ok = io.tlaloc.testing.Goldens.verified("roundtrip", "verified", "$target ${io.tlaloc.testing.Goldens.sha256Of(input)}") {
            jaxlib?.let { py ->
                runTool(py, input, listOf("-I", "-c", SERIALIZE, target), timeoutSeconds * 4).also { live = it }.ok
            }
        }
        return live ?: when (ok) {
            true -> TranslateResult(0, "recorded", "")
            else -> TranslateResult(
                1, "",
                "this module text was never verified (no stablehlo-translate or jaxlib here); run the test where " +
                    "the oracle venv is installed with TLALOC_RECORD_GOLDENS=1",
            )
        }
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
