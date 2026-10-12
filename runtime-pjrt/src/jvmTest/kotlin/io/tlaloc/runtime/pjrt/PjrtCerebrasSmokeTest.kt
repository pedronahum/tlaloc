package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The CerebrasStableHLO lane ([PjrtTarget.Cerebras]): plugin resolution, pinned
 * on any host, and a dispatch through `libPJRTCerebrasStableHLO.dylib` checked
 * against [DxirInterpreter]. The device test runs in whatever mode the plugin
 * is in — its host interpreter by default, the wafer simulator with
 * `CEREBRASSTABLEHLO_BACKEND=simulator` — and skips by name when no plugin
 * resolves.
 */
class PjrtCerebrasSmokeTest {

    private fun touch(p: Path): Path = p.also { Files.createDirectories(it.parent); Files.createFile(it) }

    @Test
    fun resolvesSiblingCheckoutFromAModuleDirectory() {
        val root = Files.createTempDirectory("cerebras-res")
        val lib = touch(root.resolve("CerebrasStableHLO/build/lib/PJRT/${PjrtBinaries.CEREBRAS_PLUGIN_FILE}"))
        val moduleDir = Files.createDirectories(root.resolve("tlaloc/runtime-pjrt"))
        assertEquals(lib, PjrtBinaries.resolveCerebrasPlugin(null, null, moduleDir.toString()))
    }

    @Test
    fun genericEnvVarIsHonouredOnlyForACerebrasShapedName() {
        val dir = Files.createTempDirectory("cerebras-res")
        val cuda = touch(dir.resolve("xla_cuda_plugin.so"))
        val cerebras = touch(dir.resolve(PjrtBinaries.CEREBRAS_PLUGIN_FILE))
        val empty = Files.createTempDirectory("cerebras-empty").toString()
        assertNull(PjrtBinaries.resolveCerebrasPlugin(cuda.toString(), null, empty))
        assertEquals(cerebras, PjrtBinaries.resolveCerebrasPlugin(cerebras.toString(), null, empty))
    }

    @Test
    fun cerebrasEnvVarWinsWhateverTheFileIsCalled() {
        val custom = touch(Files.createTempDirectory("cerebras-res").resolve("plugin.dylib"))
        val empty = Files.createTempDirectory("cerebras-empty").toString()
        assertEquals(custom, PjrtBinaries.resolveCerebrasPlugin(null, custom.toString(), empty))
    }

    @Test
    fun cerebrasTargetDecomposesAndTakesNoClientOptions() {
        assertEquals(io.tlaloc.ir.recognizer.kernel.KernelTarget.CPU_GENERIC, PjrtTarget.Cerebras.kernelTarget)
        assertTrue(PjrtTarget.Cerebras.takesNoClientOptions)
    }

    @Test
    fun matmulPlusAddAgreesWithTheInterpreter() {
        val plugin = PjrtBinaries.cerebrasPluginPath
        assumeTrue(plugin != null, "no CerebrasStableHLO plugin resolved — skipping.")
        val a = DxirType(F32, listOf(4, 8))
        val b = DxirType(F32, listOf(8, 4))
        val c = DxirType(F32, listOf(4, 4))
        val fn = DxirBuilder.function("mlp_layer") {
            val x = param("x", a)
            val w = param("w", b)
            val y = param("y", c)
            listOf(op(OpKind.ADD, listOf(op(OpKind.MATMUL, listOf(x, w), c), y), c))
        }
        val inputs = listOf(
            FloatArray(32) { (it % 7 - 3) * 0.1f },
            FloatArray(32) { (it % 5 - 2) * 0.2f },
            FloatArray(16) { it * 0.05f },
        )
        PjrtSession(plugin = plugin!!, target = PjrtTarget.Cerebras).use { session ->
            assertEquals("cerebrasstablehlo", session.platformName())
            val want = DxirInterpreter.evalFunction(fn, inputs)[0]
            val got = session.runOn(fn, inputs)[0]
            for (i in want.indices) assertTrue(abs(want[i] - got[i]) <= 1e-4f, "[$i] ${got[i]} vs ${want[i]}")
        }
    }
}
