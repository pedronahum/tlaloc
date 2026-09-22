package io.tlaloc.runtime.pjrt

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.503 (Tier 3, item 4) — the CUDA PJRT plugin resolver, certified on a
 * synthetic filesystem.
 *
 * What was wrong: after `TLALOC_PJRT_PLUGIN_PATH`, `PjrtBinaries.pluginPath` fell
 * back to ONE literal string —
 * `~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`
 * — which is one developer's venv name, one Python minor version and one plugin
 * package. Every other machine on earth got `available == false` with no
 * explanation of where Tlaloc had looked.
 *
 * Every test here builds a FAKE install tree under a temp directory and passes it
 * in as `home` / `virtualEnv`, so nothing depends on this host — they pass on a
 * laptop with no GPU and on the GB10 alike. The one test that IS about this host
 * is [theGb10InstallShapeStillResolvesHere], and it self-skips when the host has no
 * plugin rather than asserting about a machine it is not running on.
 */
class PjrtCudaPluginResolutionTest {

    /** Build `<root>/lib/<python>/<site>/jax_plugins/<pkg>/<so>` and return the .so. */
    private fun fakePlugin(
        root: Path,
        python: String = "python3.12",
        site: String = "site-packages",
        lib: String = "lib",
        pkg: String = "xla_cuda12",
        so: String = "xla_cuda_plugin.so",
    ): Path {
        val dir = root.resolve(lib).resolve(python).resolve(site).resolve("jax_plugins").resolve(pkg)
        Files.createDirectories(dir)
        val file = dir.resolve(so)
        Files.createFile(file)
        return file
    }

    private inline fun withTemp(block: (Path) -> Unit) {
        val dir = Files.createTempDirectory("tlaloc-pjrt-res")
        try {
            block(dir)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    // ---------------- the env var still wins ----------------

    @Test
    fun anExistingEnvPathWinsWhateverItIsCalled() {
        withTemp { dir ->
            val own = dir.resolve("my_deployment_plugin.so")
            Files.createFile(own)
            // A deployment that ships its own plugin has already decided. Note the
            // contrast with the TPU lane, which DOES filter the env value by name —
            // there the filter stops a CUDA plugin being used under a TPU target.
            assertEquals(
                own,
                PjrtBinaries.resolveCudaPlugin(own.toString(), virtualEnv = null, home = null),
            )
        }
    }

    @Test
    fun anEnvPathThatDoesNotExistFallsThroughToTheSearch() {
        withTemp { dir ->
            val home = dir.resolve("home").also { Files.createDirectories(it) }
            val real = fakePlugin(home.resolve(".local").resolve("venvs").resolve("jaxenv"))
            assertEquals(
                real,
                PjrtBinaries.resolveCudaPlugin(
                    envValue = dir.resolve("nope.so").toString(),
                    virtualEnv = null,
                    home = home.toString(),
                ),
            )
        }
    }

    @Test
    fun aBlankEnvPathIsNotTreatedAsAPath() {
        withTemp { dir ->
            val home = dir.resolve("home").also { Files.createDirectories(it) }
            assertNull(
                PjrtBinaries.resolveCudaPlugin(envValue = "  ", virtualEnv = null, home = home.toString()),
            )
        }
    }

    // ---------------- the parts that used to be literals ----------------

    @Test
    fun anyVenvNameUnderLocalVenvsResolves() {
        // The old code hard-coded the venv name `iree`. This is the defect a user
        // hits first: they made a venv called something sensible.
        withTemp { dir ->
            for (name in listOf("jax", "tlaloc", "gpu", "my-env")) {
                val home = dir.resolve("h-$name").also { Files.createDirectories(it) }
                val so = fakePlugin(home.resolve(".local").resolve("venvs").resolve(name))
                assertEquals(
                    so,
                    PjrtBinaries.resolveCudaPlugin(null, virtualEnv = null, home = home.toString()),
                    "a venv named '$name' must resolve; the old resolver only knew 'iree'",
                )
            }
        }
    }

    @Test
    fun anyPython3MinorVersionResolves() {
        withTemp { dir ->
            for (py in listOf("python3.10", "python3.11", "python3.13", "python3.14")) {
                val home = dir.resolve("h-$py").also { Files.createDirectories(it) }
                val so = fakePlugin(home.resolve(".local").resolve("venvs").resolve("v"), python = py)
                assertEquals(
                    so,
                    PjrtBinaries.resolveCudaPlugin(null, virtualEnv = null, home = home.toString()),
                    "$py must resolve; the old resolver was pinned to python3.12",
                )
            }
        }
    }

    @Test
    fun anyCudaPluginPackageAndSoNameResolves() {
        withTemp { dir ->
            // xla_cuda13 is the package name that does not exist yet and will.
            val home = dir.resolve("h").also { Files.createDirectories(it) }
            val so = fakePlugin(
                home.resolve(".local").resolve("venvs").resolve("v"),
                pkg = "xla_cuda13",
                so = "pjrt_c_api_gpu_plugin.so",
            )
            assertEquals(so, PjrtBinaries.resolveCudaPlugin(null, null, home.toString()))
        }
    }

    @Test
    fun debianDistPackagesAndLib64Resolve() {
        withTemp { dir ->
            val a = dir.resolve("a").also { Files.createDirectories(it) }
            val distSo = fakePlugin(a, site = "dist-packages")
            assertEquals(distSo, PjrtBinaries.resolveCudaPlugin(null, a.toString(), null))

            val b = dir.resolve("b").also { Files.createDirectories(it) }
            val lib64So = fakePlugin(b, lib = "lib64")
            assertEquals(lib64So, PjrtBinaries.resolveCudaPlugin(null, b.toString(), null))
        }
    }

    @Test
    fun anActivatedVirtualEnvIsSearchedAndIsSearchedFirst() {
        withTemp { dir ->
            val home = dir.resolve("home").also { Files.createDirectories(it) }
            val fromHome = fakePlugin(home.resolve(".local").resolve("venvs").resolve("aaa"))
            val venv = dir.resolve("active").also { Files.createDirectories(it) }
            val fromVenv = fakePlugin(venv)
            assertTrue(Files.exists(fromHome))
            assertEquals(
                fromVenv,
                PjrtBinaries.resolveCudaPlugin(null, venv.toString(), home.toString()),
                "\$VIRTUAL_ENV is the user naming the Python they mean; it must outrank a " +
                    "venv found by scanning",
            )
        }
    }

    @Test
    fun aNonCudaJaxPluginIsNotOfferedToTheCudaLane() {
        withTemp { dir ->
            val venv = dir.resolve("v").also { Files.createDirectories(it) }
            fakePlugin(venv, pkg = "xla_tpu", so = "libtpu.so")
            assertNull(
                PjrtBinaries.resolveCudaPlugin(null, venv.toString(), null),
                "a jax_plugins package with no 'cuda' in its name must not resolve as the CUDA " +
                    "plugin — that is how a TPU .so would end up under a CUDA target",
            )
        }
    }

    @Test
    fun resolutionIsDeterministicWhenTwoVenvsBothHaveAPlugin() {
        withTemp { dir ->
            val home = dir.resolve("home").also { Files.createDirectories(it) }
            val venvs = home.resolve(".local").resolve("venvs")
            val zz = fakePlugin(venvs.resolve("zz"))
            val aa = fakePlugin(venvs.resolve("aa"))
            assertTrue(Files.exists(zz))
            // Sorted, not filesystem order: two runs on one machine must pick the same
            // plugin, or a benchmark number means nothing.
            repeat(3) {
                assertEquals(aa, PjrtBinaries.resolveCudaPlugin(null, null, home.toString()))
            }
        }
    }

    @Test
    fun nothingResolvesOnAnEmptyHome() {
        withTemp { dir ->
            val home = dir.resolve("empty").also { Files.createDirectories(it) }
            // /usr and /usr/local are also searched and, on this host, hold no jax
            // plugin; if that ever changes this assertion is the thing that notices.
            assertNull(PjrtBinaries.resolveCudaPlugin(null, null, home.toString()))
        }
    }

    // ---------------- the report names every place it looked ----------------

    @Test
    fun theReportNamesTheEnvVarAndEveryRootWhenNothingIsFound() {
        withTemp { dir ->
            val home = dir.resolve("empty").also { Files.createDirectories(it) }
            val report = PjrtBinaries.describeCudaPluginSearch(null, null, home.toString())
            assertTrue(PjrtBinaries.PLUGIN_PATH_ENV in report, report)
            assertTrue("not set" in report, report)
            for (root in listOf("~/.local/venvs/*", "~/.venv", "~/venv", "~/.local", "/usr/local", "/usr")) {
                assertTrue(root in report, "the report must name the root $root:\n$report")
            }
            assertTrue("python3.*" in report, "…and that the python version is globbed:\n$report")
            assertTrue("jax[cuda12]" in report, "…and how to fix it:\n$report")
        }
    }

    @Test
    fun theReportSaysWhenTheEnvVarPointsAtNothing() {
        withTemp { dir ->
            val missing = dir.resolve("gone.so").toString()
            val report = PjrtBinaries.describeCudaPluginSearch(missing, null, dir.toString())
            assertTrue(missing in report, report)
            assertTrue("no file there" in report, "the report must say the env path is empty:\n$report")
        }
    }

    @Test
    fun theReportListsTheCandidatesItFound() {
        withTemp { dir ->
            val home = dir.resolve("home").also { Files.createDirectories(it) }
            val so = fakePlugin(home.resolve(".local").resolve("venvs").resolve("v"))
            val report = PjrtBinaries.describeCudaPluginSearch(null, null, home.toString())
            assertTrue(so.toString() in report, "the resolved candidate must appear:\n$report")
            assertTrue("FOUND" in report, report)
            assertTrue(
                "Resolved: $so" in report,
                "a successful search says what it resolved, and does NOT print fix-it advice " +
                    "underneath it:\n$report",
            )
            assertTrue("jax[cuda12]" !in report, "no advice on success:\n$report")
        }
    }

    // ---------------- this host: the path that certifies the GPU claims ----------------

    @Test
    fun theGb10InstallShapeStillResolvesHere() {
        val resolved = PjrtBinaries.pluginPath
        if (resolved == null) {
            // Not a failure: most hosts have no plugin. The report is the deliverable.
            assertTrue(
                PjrtBinaries.PLUGIN_PATH_ENV in PjrtBinaries.pluginSearchReport,
                "with no plugin resolved, the report is all the user gets; it must at least " +
                    "name the env var:\n${PjrtBinaries.pluginSearchReport}",
            )
            return
        }
        assertTrue(
            Files.exists(resolved),
            "pluginPath resolved to $resolved, which does not exist",
        )
        assertTrue(
            resolved.toString().endsWith(".so"),
            "a PJRT plugin is a shared object; got $resolved",
        )
    }
}
