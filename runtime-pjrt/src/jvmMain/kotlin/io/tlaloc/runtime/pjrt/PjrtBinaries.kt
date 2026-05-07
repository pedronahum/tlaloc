package io.tlaloc.runtime.pjrt

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * Resolves the toolchain bits PjrtRuntime needs. Mirrors the §0.4.284 IreeBinaries
 * shape but for Python+JAX (the v1 PJRT facade goes through a Python subprocess
 * that uses jaxlib's MLIR bindings + PJRT-XLA backend; a JNI-direct path to
 * libpjrt_c_api.so is the long-term replacement).
 *
 * Resolution order for the python interpreter:
 *   1. `TLALOC_PJRT_PYTHON` env var.
 *   2. `TLALOC_TORCH_PYTHON` env var (kept compatible with the §0.4.296+
 *      orchestrator naming so the same venv resolves both PyTorch and JAX).
 *   3. `~/.local/venvs/iree/bin/python` (the venv §0.4.297 installed JAX into).
 *   4. PATH lookup.
 */
object PjrtBinaries {

    val python: String? by lazy {
        System.getenv("TLALOC_PJRT_PYTHON")?.takeIf { Files.isExecutable(Path.of(it)) }
            ?: System.getenv("TLALOC_TORCH_PYTHON")?.takeIf { Files.isExecutable(Path.of(it)) }
            ?: run {
                val home = System.getProperty("user.home") ?: return@run null
                val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
                if (Files.isExecutable(venv)) venv.toString() else null
            }
            ?: whichOnPath("python3")
            ?: whichOnPath("python")
    }

    val available: Boolean
        get() = python != null && pythonHasJax()

    /**
     * True when `import jax` succeeds AND JAX sees at least one CUDA / GPU
     * device. v1 PJRT-CPU dispatch is technically possible but not exercised
     * yet; treat "PJRT available" as "PJRT-CUDA available" until §0.4.303
     * adds the CPU target.
     */
    val cudaAvailable: Boolean by lazy {
        val py = python ?: return@lazy false
        val pb = ProcessBuilder(
            py, "-c",
            "import jax,sys; sys.exit(0 if any(d.platform in ('cuda','gpu') for d in jax.devices()) else 1)",
        )
        pb.redirectErrorStream(true)
        runCatching {
            val p = pb.start()
            p.waitFor(30, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun pythonHasJax(): Boolean {
        val py = python ?: return false
        val pb = ProcessBuilder(py, "-c", "import jax")
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(20, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
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
