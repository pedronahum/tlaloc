package io.tlaloc.runtime.pjrt.ffm

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.lang.foreign.Arena
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.303 — pure-FFM smoke. Loads the PJRT plugin (xla_cuda_plugin.so),
 * calls GetPjrtApi → Client_Create → Client_PlatformName → Client_Destroy
 * via [PjrtFfm] / [PjrtApi] / [PjrtClient]. No Python in the dispatch
 * path — proves the JNI-replacement we discussed actually works.
 *
 * The plugin lives where JAX put it (the §0.4.297 install). A future
 * §0.4.306 commit will add a `TLALOC_PJRT_PLUGIN_PATH` env override + a
 * `scripts/fetch-pjrt-plugin.sh` so deployment doesn't depend on a JAX
 * install at all.
 */
class PjrtFfmSmokeTest {

    private fun resolvePluginPath(): Path? {
        System.getenv("TLALOC_PJRT_PLUGIN_PATH")?.let { p ->
            val path = Path.of(p)
            if (Files.exists(path)) return path
        }
        val home = System.getProperty("user.home") ?: return null
        val jaxBundled = Path.of(
            home, ".local", "venvs", "iree", "lib", "python3.12",
            "site-packages", "jax_plugins", "xla_cuda12", "xla_cuda_plugin.so",
        )
        return jaxBundled.takeIf { Files.exists(it) }
    }

    @Test
    fun loadsPjrtCudaPluginAndReadsPlatformName() {
        val plugin = resolvePluginPath()
        assumeTrue(
            plugin != null,
            "PJRT plugin (xla_cuda_plugin.so) not found — set TLALOC_PJRT_PLUGIN_PATH " +
                "or `pip install jax[cuda12]` into ~/.local/venvs/iree.",
        )

        Arena.ofShared().use { arena ->
            val api = PjrtFfm.load(plugin!!, arena)
            api.createClient().use { client ->
                val name = client.platformName()
                println("[pjrt-ffm-smoke] platform_name='$name'")
                assertTrue(
                    name == "cuda" || name == "gpu",
                    "expected platform_name in {cuda,gpu}; got '$name'",
                )
            }
        }
    }
}
