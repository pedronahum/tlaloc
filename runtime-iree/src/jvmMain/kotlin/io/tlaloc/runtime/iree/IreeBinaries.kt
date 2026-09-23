package io.tlaloc.runtime.iree

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * Locates the IREE command-line tools. Each tool resolves independently, from
 * the first of these that holds an executable file of that name:
 *   1. `$TLALOC_IREE_BIN` — a directory holding `iree-compile`, `iree-run-module`
 *      and (optionally) `iree-benchmark-module`.
 *   2. `$VIRTUAL_ENV/bin` — an activated venv with the IREE wheels installed.
 *   3. `~/.local/venvs/<any>/bin`, in name order.
 *   4. Every directory on `PATH`, in order.
 *
 * [available] is true only when `iree-compile` and `iree-run-module` both
 * resolve; tests self-skip on it. [searchReport] lists every place looked.
 */
object IreeBinaries {

    val ireeCompile: String? by lazy { resolve("iree-compile") }
    val ireeRunModule: String? by lazy { resolve("iree-run-module") }
    val ireeBenchmarkModule: String? by lazy { resolve("iree-benchmark-module") }

    val available: Boolean
        get() = ireeCompile != null && ireeRunModule != null

    /**
     * Best-effort detector for whether NVIDIA CUDA dispatch is plausible on this host.
     * `nvidia-smi -L` returning zero with a non-empty device listing is sufficient
     * evidence to attempt CUDA compile + dispatch via [IreeTarget.Cuda]; if the
     * subsequent `iree-run-module --device=cuda` invocation actually fails (driver
     * mismatch, etc.) that surfaces as a real test failure rather than a silent skip.
     */
    val cudaAvailable: Boolean by lazy {
        runCatching {
            val pb = ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true)
            val p = pb.start()
            val finished = p.waitFor(5, TimeUnit.SECONDS)
            if (!finished) {
                p.destroyForcibly(); false
            } else {
                val out = p.inputStream.bufferedReader().readText().trim()
                p.exitValue() == 0 && out.isNotEmpty()
            }
        }.getOrElse { false }
    }

    /** The environment variable naming a directory that holds the IREE tools. */
    const val BIN_ENV: String = "TLALOC_IREE_BIN"

    private fun resolve(tool: String): String? =
        resolveTool(tool, searchDirs(System.getenv(BIN_ENV), System.getenv("VIRTUAL_ENV"),
            System.getProperty("user.home"), System.getenv("PATH")))

    /** The directories searched, in order, with every input a parameter. */
    internal fun searchDirs(binEnv: String?, virtualEnv: String?, home: String?, pathEnv: String?): List<Path> =
        buildList {
            binEnv?.takeIf { it.isNotBlank() }?.let { add(Path.of(it)) }
            virtualEnv?.takeIf { it.isNotBlank() }?.let { add(Path.of(it, "bin")) }
            if (home != null) {
                val venvs = Path.of(home, ".local", "venvs")
                if (Files.isDirectory(venvs)) {
                    runCatching {
                        Files.list(venvs).use { s -> s.toList().sortedBy { it.fileName.toString() } }
                    }.getOrDefault(emptyList()).forEach { add(it.resolve("bin")) }
                }
            }
            pathEnv?.split(java.io.File.pathSeparatorChar)
                ?.filter { it.isNotBlank() }
                ?.forEach { add(Path.of(it)) }
        }.distinct()

    internal fun resolveTool(tool: String, dirs: List<Path>): String? =
        dirs.asSequence().map { it.resolve(tool) }
            .firstOrNull { Files.isRegularFile(it) && Files.isExecutable(it) }
            ?.toString()

    /** Where each IREE tool was looked for and what was found, for error messages and skip reasons. */
    val searchReport: String
        get() = describeSearch(System.getenv(BIN_ENV), System.getenv("VIRTUAL_ENV"),
            System.getProperty("user.home"), System.getenv("PATH"))

    internal fun describeSearch(binEnv: String?, virtualEnv: String?, home: String?, pathEnv: String?): String {
        val dirs = searchDirs(binEnv, virtualEnv, home, pathEnv)
        val b = StringBuilder("IREE tool resolution — where Tlaloc looked:\n")
        b.appendLine("  \$$BIN_ENV: ${binEnv?.takeIf { it.isNotBlank() } ?: "not set"}")
        b.appendLine("  \$VIRTUAL_ENV: ${virtualEnv?.takeIf { it.isNotBlank() } ?: "not set"}")
        b.appendLine("  directories, in order: ${dirs.joinToString(", ")}")
        var missing = false
        for (tool in listOf("iree-compile", "iree-run-module", "iree-benchmark-module")) {
            val found = resolveTool(tool, dirs)
            if (found == null) missing = true
            b.appendLine("  $tool: ${found ?: "not found"}")
        }
        if (missing) {
            b.append(
                "Fix: `pip install iree-base-compiler iree-base-runtime` into a venv under " +
                    "~/.local/venvs/ (or activate it), or export $BIN_ENV=<directory holding the tools>.",
            )
        }
        return b.toString().trimEnd()
    }

    /** The resolved [tool], or an [IllegalStateException] carrying [searchReport]. */
    internal fun requireTool(tool: String, resolved: String?): String =
        resolved ?: throw IllegalStateException("$tool not found.\n$searchReport")
}
