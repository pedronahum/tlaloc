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
     * §0.4.459 (G2a) — TPU PJRT plugin resolution. Resolution order:
     *
     *   1. `TLALOC_PJRT_PLUGIN_PATH`, honoured for the TPU lane **only when
     *      the file name is tpu-shaped** (contains "tpu": `libtpu.so`,
     *      `pjrt_c_api_tpu_plugin.so`). The env var is generic — on CUDA
     *      hosts it legitimately names `xla_cuda_plugin.so`, and handing
     *      that to the TPU lane would create a CUDA client under a TPU
     *      target. The name gate keeps the two lanes from crossing; a
     *      deliberate misnaming defeats it, and the smoke suite's
     *      platform-name assertion (`"tpu"`) is the backstop.
     *   2. The libtpu default install locations, documented from the PyPI
     *      `libtpu` wheel layout and the Cloud TPU VM images:
     *        - `$VIRTUAL_ENV/lib/python3.N/site-packages/libtpu/libtpu.so`
     *          (the wheel ships exactly one .so, named `libtpu.so`, inside
     *          the `libtpu` package directory)
     *        - `~/.local/lib/python3.N/site-packages/libtpu/libtpu.so`
     *          (`pip install --user libtpu`)
     *        - `/lib/libtpu.so`, `/usr/lib/libtpu.so` (TPU VM system
     *          images; older images also distribute the plugin as
     *          `pjrt_c_api_tpu_plugin.so` — reach those via the env var)
     *
     * On this GB10 none of these exist, so [tpuAvailable] is false and the
     * TPU smoke suite self-skips — the designed local behaviour.
     */
    val tpuPluginPath: Path? by lazy {
        resolveTpuPlugin(
            envValue = System.getenv("TLALOC_PJRT_PLUGIN_PATH"),
            virtualEnv = System.getenv("VIRTUAL_ENV"),
            home = System.getProperty("user.home"),
        )
    }

    val tpuAvailable: Boolean
        get() = tpuPluginPath != null

    /** The tpu-shaped-name gate, extracted so the resolution rules are
     * pinned by a GPU-less unit test. */
    internal fun isTpuShapedPluginName(fileName: String): Boolean =
        fileName.contains("tpu")

    /** Pure-ish resolution core (filesystem reads only; all environment
     * inputs are parameters) — unit-testable without touching real env. */
    internal fun resolveTpuPlugin(envValue: String?, virtualEnv: String?, home: String?): Path? {
        envValue?.let { p ->
            val path = Path.of(p)
            if (Files.exists(path) && isTpuShapedPluginName(path.fileName.toString())) return path
        }
        val candidates = buildList {
            virtualEnv?.let { addAll(libtpuSitePackages(Path.of(it))) }
            home?.let { addAll(libtpuSitePackages(Path.of(it, ".local"))) }
            add(Path.of("/lib/libtpu.so"))
            add(Path.of("/usr/lib/libtpu.so"))
        }
        return candidates.firstOrNull { Files.exists(it) }
    }

    /** `<prefix>/lib/python3.N/site-packages/libtpu/libtpu.so` for every
     * python3.* directory present — the PyPI wheel layout. */
    private fun libtpuSitePackages(prefix: Path): List<Path> = runCatching {
        val lib = prefix.resolve("lib")
        if (!Files.isDirectory(lib)) return@runCatching emptyList()
        Files.list(lib).use { stream ->
            stream.filter { it.fileName.toString().startsWith("python3") }
                .map { it.resolve("site-packages").resolve("libtpu").resolve("libtpu.so") }
                .toList()
        }
    }.getOrElse { emptyList() }

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
