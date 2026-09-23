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
 * Plugin resolution order — §0.4.503 (Tier 3, item 4) generalised this:
 *   1. `TLALOC_PJRT_PLUGIN_PATH` env var, when it names an existing file.
 *   2. A search over the places a JAX CUDA PJRT plugin actually gets installed,
 *      with the venv name, the Python minor version and the plugin package all
 *      globbed rather than spelled — see [cudaPluginCandidates].
 *
 * What was here until §0.4.503, and why it had to go: step 2 was the single
 * literal path
 * `~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`.
 * That is one developer's machine written into the library — one venv NAME
 * (`iree`, which is not even the name of the thing it holds), one Python MINOR
 * version, one plugin PACKAGE. A user who ran `pip install jax[cuda12]` into any
 * venv of their own got `available == false` and, before this commit, no way to
 * find out where Tlaloc had looked. [pluginSearchReport] now names every location,
 * in order, and says what it found at each.
 *
 * The GB10 path that certifies this repository's GPU claims still resolves — by
 * the glob, not by the literal — and `PjrtBinariesResolutionTest` pins exactly
 * that shape so a future tidy-up cannot quietly drop it.
 *
 * `cudaAvailable` mirrors `IreeBinaries.cudaAvailable` — uses `nvidia-smi -L`
 * so tests can self-skip on hosts without an NVIDIA GPU.
 */
object PjrtBinaries {

    /** The environment variable a deployment sets to name its own plugin `.so`. */
    const val PLUGIN_PATH_ENV: String = "TLALOC_PJRT_PLUGIN_PATH"

    val pluginPath: Path? by lazy {
        resolveCudaPlugin(
            envValue = System.getenv(PLUGIN_PATH_ENV),
            virtualEnv = System.getenv("VIRTUAL_ENV"),
            home = System.getProperty("user.home"),
        )
    }

    val available: Boolean
        get() = pluginPath != null

    /**
     * §0.4.503 — every place [pluginPath] looked, in order, and what was there.
     * Meant to be pasted verbatim into a skip reason or an error: a resolver that
     * fails without saying where it searched makes the user guess, and the thing
     * they are guessing about is a path inside a Python installation.
     */
    val pluginSearchReport: String
        get() = describeCudaPluginSearch(
            envValue = System.getenv(PLUGIN_PATH_ENV),
            virtualEnv = System.getenv("VIRTUAL_ENV"),
            home = System.getProperty("user.home"),
        )

    /**
     * The resolved CUDA plugin, or an [IllegalStateException] that says why there
     * is none: on an operating system other than Linux, that JAX publishes its CUDA
     * PJRT plugin for Linux only; on Linux, the whole [pluginSearchReport].
     * [caller] names the entry point in the message.
     */
    fun requireCudaPlugin(caller: String): Path =
        pluginPath ?: throw IllegalStateException(
            missingCudaPluginMessage(caller, System.getProperty("os.name").orEmpty(), pluginSearchReport),
        )

    /** The text behind [requireCudaPlugin], with its inputs as parameters. */
    internal fun missingCudaPluginMessage(caller: String, osName: String, searchReport: String): String {
        val header = "$caller: no PJRT CUDA plugin found."
        return if (!osName.lowercase().startsWith("linux")) {
            "$header The JAX CUDA PJRT plugin (xla_cuda_plugin.so) is published for Linux only, and " +
                "this JVM runs on $osName. Run the CUDA lane on a Linux host with an NVIDIA GPU, or set " +
                "$PLUGIN_PATH_ENV to a PJRT plugin built for this platform.\n$searchReport"
        } else {
            "$header\n$searchReport"
        }
    }

    /**
     * Pure-ish resolution core (filesystem reads only; all environment inputs are
     * parameters), mirroring [resolveTpuPlugin] so both lanes are unit-testable on
     * a host with no plugin at all.
     */
    internal fun resolveCudaPlugin(envValue: String?, virtualEnv: String?, home: String?): Path? {
        envValue?.takeIf { it.isNotBlank() }?.let { p ->
            val path = Path.of(p)
            // An env var that names a file WINS, whatever the file is called: a
            // deployment shipping its own plugin has already made the decision, and
            // second-guessing its file name is how §0.4.459's TPU-lane name gate
            // earned its long comment. The CUDA-shape filter below applies only to
            // paths Tlaloc GUESSED.
            if (Files.exists(path)) return path
        }
        return cudaPluginCandidates(virtualEnv, home).firstOrNull { Files.exists(it) }
    }

    /**
     * Where a JAX CUDA PJRT plugin lives, in priority order, with every part that
     * varies between machines globbed:
     *
     *  - the ROOT: `$VIRTUAL_ENV` first (an activated venv is the user telling us
     *    which Python they mean, and [resolveTpuPlugin] already honours it), then
     *    every directory under `~/.local/venvs/` — this is the shape the GB10 that
     *    certifies Tlaloc's GPU claims uses — then `~/.venv`, `~/venv`, `~/.local`
     *    (a `pip install --user`), then `/usr/local` and `/usr` for a system install.
     *  - the PYTHON VERSION: `lib/python3.*` and `lib64/python3.*`, not `python3.12`.
     *  - the SITE DIRECTORY: `site-packages` (pip/venv) and `dist-packages` (Debian).
     *  - the PLUGIN PACKAGE: any directory under `jax_plugins/` whose name mentions
     *    cuda — `xla_cuda12` today, `xla_cuda13` the day it ships — and any `.so`
     *    inside it, rather than the literal `xla_cuda_plugin.so`.
     *
     * Deterministic: every directory listing is sorted, so two runs on one machine
     * resolve the same plugin, and a machine with two CUDA venvs gets a stable
     * answer instead of a filesystem-order one.
     */
    internal fun cudaPluginCandidates(virtualEnv: String?, home: String?): List<Path> {
        val roots = buildList {
            virtualEnv?.takeIf { it.isNotBlank() }?.let { add(Path.of(it)) }
            if (home != null) {
                addAll(sortedChildren(Path.of(home, ".local", "venvs")))
                add(Path.of(home, ".venv"))
                add(Path.of(home, "venv"))
                add(Path.of(home, ".local"))
            }
            add(Path.of("/usr/local"))
            add(Path.of("/usr"))
        }
        return roots.flatMap { root ->
            sequenceOf("lib", "lib64").flatMap { libDir ->
                sortedChildren(root.resolve(libDir))
                    .asSequence()
                    .filter { it.fileName.toString().startsWith("python3") }
                    .flatMap { pyDir ->
                        sequenceOf("site-packages", "dist-packages").map { pyDir.resolve(it) }
                    }
            }.flatMap { siteDir ->
                sortedChildren(siteDir.resolve("jax_plugins"))
                    .asSequence()
                    .filter { "cuda" in it.fileName.toString().lowercase() }
                    .flatMap { pkg ->
                        sortedChildren(pkg).asSequence()
                            .filter { it.fileName.toString().endsWith(".so") }
                    }
            }.toList()
        }
    }

    /** Sorted directory listing, or empty when the path is not a readable directory. */
    private fun sortedChildren(dir: Path): List<Path> = runCatching {
        if (!Files.isDirectory(dir)) return@runCatching emptyList()
        Files.list(dir).use { stream -> stream.toList().sortedBy { it.fileName.toString() } }
    }.getOrElse { emptyList() }

    /**
     * The human-readable search report. A separate function from
     * [resolveCudaPlugin] and driven by the same inputs, so the report cannot
     * describe a search different from the one that ran.
     */
    internal fun describeCudaPluginSearch(
        envValue: String?,
        virtualEnv: String?,
        home: String?,
    ): String {
        val b = StringBuilder()
        b.appendLine("PJRT CUDA plugin resolution — where Tlaloc looked:")
        when {
            envValue.isNullOrBlank() ->
                b.appendLine("  1. \$$PLUGIN_PATH_ENV: not set")
            Files.exists(Path.of(envValue)) ->
                b.appendLine("  1. \$$PLUGIN_PATH_ENV = $envValue — FOUND")
            else ->
                b.appendLine("  1. \$$PLUGIN_PATH_ENV = $envValue — no file there")
        }
        val candidates = cudaPluginCandidates(virtualEnv, home)
        if (candidates.isEmpty()) {
            b.appendLine(
                "  2. no jax_plugins/*cuda*/*.so under any searched root. Roots searched: " +
                    "\$VIRTUAL_ENV${if (virtualEnv.isNullOrBlank()) " (not set)" else " = $virtualEnv"}, " +
                    "~/.local/venvs/*, ~/.venv, ~/venv, ~/.local, /usr/local, /usr — each at " +
                    "lib{,64}/python3.*/{site,dist}-packages/jax_plugins/*cuda*/*.so",
            )
        } else {
            b.appendLine("  2. candidates found by the glob, in resolution order:")
            candidates.forEach { c ->
                b.appendLine("       $c${if (Files.exists(c)) " — FOUND" else " — missing"}")
            }
        }
        // The "how to fix it" line is emitted only when nothing resolved. Printing
        // advice underneath a successful resolution is how a log teaches its reader to
        // stop reading it.
        val resolved = resolveCudaPlugin(envValue, virtualEnv, home)
        if (resolved == null) {
            b.append(
                "Fix: export $PLUGIN_PATH_ENV=/path/to/xla_cuda_plugin.so, or " +
                    "`pip install jax[cuda12]` into a venv under ~/.local/venvs/ (or activate " +
                    "it, so \$VIRTUAL_ENV points at it).",
            )
        } else {
            b.append("Resolved: $resolved")
        }
        return b.toString()
    }

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
