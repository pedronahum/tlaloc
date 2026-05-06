package io.tlaloc.runtime.iree

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * Resolution order:
 *   1. `TLALOC_IREE_BIN` env var — directory containing `iree-compile` and `iree-run-module`.
 *   2. `~/.local/venvs/iree/bin/` — the path produced by the dual-track plan's pip-install step.
 *   3. PATH lookup via `which`.
 *
 * `available` is true only when both binaries resolve. Tests should self-skip via
 * `Assumptions.assumeTrue(IreeBinaries.available, …)` so build hosts without IREE stay green.
 */
object IreeBinaries {

    val ireeCompile: String? by lazy { resolve("iree-compile") }
    val ireeRunModule: String? by lazy { resolve("iree-run-module") }

    val available: Boolean
        get() = ireeCompile != null && ireeRunModule != null

    private fun resolve(tool: String): String? {
        System.getenv("TLALOC_IREE_BIN")?.let { dir ->
            val candidate = Path.of(dir, tool)
            if (Files.isExecutable(candidate)) return candidate.toString()
        }
        val home = System.getProperty("user.home")
        if (home != null) {
            val candidate = Path.of(home, ".local", "venvs", "iree", "bin", tool)
            if (Files.isExecutable(candidate)) return candidate.toString()
        }
        return whichOnPath(tool)
    }

    private fun whichOnPath(tool: String): String? {
        val pb = ProcessBuilder("/usr/bin/which", tool).redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(2, TimeUnit.SECONDS)
            val out = p.inputStream.bufferedReader().readText().trim()
            if (p.exitValue() == 0 && out.isNotEmpty()) out else null
        }.getOrNull()
    }
}
