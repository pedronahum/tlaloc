package io.tlaloc.runtime.pjrt

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit

/**
 * §0.4.305 — resolves the PJRT plugin path. Replaces the §0.4.302 python-based
 * resolver — `:runtime-pjrt` no longer goes through a Python subprocess; the
 * §0.4.303/§0.4.304 FFM bindings load the plugin directly via
 * [java.lang.foreign.SymbolLookup.libraryLookup].
 *
 * Plugin resolution order:
 *   1. `TLALOC_PJRT_PLUGIN_PATH` env var.
 *   2. `~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`
 *      (the §0.4.297 JAX install — the only PJRT plugin currently distributed
 *      to this host). A future §0.4.306 commit will add a fetch-script /
 *      bundled-resource path so deployment doesn't need a JAX install.
 *
 * `cudaAvailable` mirrors `IreeBinaries.cudaAvailable` — uses `nvidia-smi -L`
 * so tests can self-skip on hosts without an NVIDIA GPU.
 */
object PjrtBinaries {

    val pluginPath: Path? by lazy {
        System.getenv("TLALOC_PJRT_PLUGIN_PATH")?.let { p ->
            val path = Path.of(p)
            if (Files.exists(path)) return@lazy path
        }
        val home = System.getProperty("user.home") ?: return@lazy null
        val jaxBundled = Path.of(
            home, ".local", "venvs", "iree", "lib", "python3.12",
            "site-packages", "jax_plugins", "xla_cuda12", "xla_cuda_plugin.so",
        )
        jaxBundled.takeIf { Files.exists(it) }
    }

    val available: Boolean
        get() = pluginPath != null

    /**
     * Best-effort detector for whether NVIDIA CUDA dispatch is plausible on this
     * host. Mirrors the §0.4.290 [io.tlaloc.runtime.iree.IreeBinaries.cudaAvailable]
     * implementation — `nvidia-smi -L` returning zero with non-empty output is
     * sufficient evidence to attempt PJRT-CUDA. Actual driver / plugin failures
     * surface as real test failures (not silent skips).
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
}
