package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The MetalHLO lane ([PjrtTarget.Metal]): plugin resolution, pinned on any host,
 * and dispatch through `libPJRTMetalHLO.dylib` on Apple silicon, checked against
 * [DxirInterpreter]. The device tests skip by name when no MetalHLO plugin
 * resolves — build one with `swift build -c release --product PJRTMetalHLO` in a
 * MetalHLO checkout next to this one.
 */
class PjrtMetalSmokeTest {

    private val scalar = DxirType(F32, emptyList())

    // ---- Resolution (filesystem only; runs everywhere) ----

    private fun touch(p: Path): Path = p.also { Files.createDirectories(it.parent); Files.createFile(it) }

    @Test
    fun resolvesSiblingCheckoutFromAModuleDirectory() {
        val root = Files.createTempDirectory("metal-res")
        val dylib = touch(root.resolve("MetalHLO/.build/release/${PjrtBinaries.METALHLO_PLUGIN_FILE}"))
        val moduleDir = Files.createDirectories(root.resolve("tlaloc/runtime-pjrt"))
        assertEquals(
            dylib,
            PjrtBinaries.resolveMetalPlugin(null, null, null, null, moduleDir.toString()),
        )
    }

    @Test
    fun releaseBuildWinsOverDebug() {
        val root = Files.createTempDirectory("metal-res")
        touch(root.resolve("MetalHLO/.build/debug/${PjrtBinaries.METALHLO_PLUGIN_FILE}"))
        val release = touch(root.resolve("MetalHLO/.build/release/${PjrtBinaries.METALHLO_PLUGIN_FILE}"))
        assertEquals(release, PjrtBinaries.resolveMetalPlugin(null, null, null, null, root.toString()))
    }

    @Test
    fun genericEnvVarIsHonouredOnlyForAMetalShapedName() {
        val dir = Files.createTempDirectory("metal-res")
        val cuda = touch(dir.resolve("xla_cuda_plugin.so"))
        val metal = touch(dir.resolve("libPJRTMetalHLO.dylib"))
        val empty = Files.createTempDirectory("metal-empty").toString()
        assertNull(PjrtBinaries.resolveMetalPlugin(cuda.toString(), null, null, null, empty))
        assertEquals(metal, PjrtBinaries.resolveMetalPlugin(metal.toString(), null, null, null, empty))
    }

    @Test
    fun metalHloEnvVarWinsWhateverTheFileIsCalled() {
        val dir = Files.createTempDirectory("metal-res")
        val custom = touch(dir.resolve("plugin.dylib"))
        val empty = Files.createTempDirectory("metal-empty").toString()
        assertEquals(custom, PjrtBinaries.resolveMetalPlugin(null, custom.toString(), null, null, empty))
    }

    @Test
    fun resolvesAPipInstalledPluginUnderTheVirtualEnv() {
        val venv = Files.createTempDirectory("metal-venv")
        val dylib = touch(
            venv.resolve("lib/python3.12/site-packages/jax_plugins/metalhlo/${PjrtBinaries.METALHLO_PLUGIN_FILE}"),
        )
        val empty = Files.createTempDirectory("metal-empty").toString()
        assertEquals(dylib, PjrtBinaries.resolveMetalPlugin(null, null, venv.toString(), null, empty))
    }

    @Test
    fun appleSiliconDetection() {
        assertTrue(PjrtBinaries.isAppleSilicon("Mac OS X", "aarch64"))
        assertTrue(!PjrtBinaries.isAppleSilicon("Linux", "aarch64"))
        assertTrue(!PjrtBinaries.isAppleSilicon("Mac OS X", "x86_64"))
    }

    @Test
    fun metalTargetDecomposesAndTakesNoClientOptions() {
        assertEquals(io.tlaloc.ir.recognizer.kernel.KernelTarget.CPU_GENERIC, PjrtTarget.Metal.kernelTarget)
        assertTrue(PjrtTarget.Metal.takesNoClientOptions)
        assertTrue(!PjrtTarget.Cuda.takesNoClientOptions)
    }

    // ---- Device (Apple silicon with a resolved plugin) ----

    private fun requireMetal() {
        assumeTrue(PjrtBinaries.metalPluginPath != null, "no MetalHLO plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.metalAvailable, "not Apple silicon — skipping.")
    }

    private fun metalSession() = PjrtSession(plugin = PjrtBinaries.metalPluginPath!!, target = PjrtTarget.Metal)

    private fun assertMatchesInterpreter(fn: DxirFunction, inputs: List<FloatArray>, actual: List<FloatArray>, tol: Float = 1e-4f) {
        val expected = DxirInterpreter.evalFunction(fn, inputs)
        assertEquals(expected.size, actual.size, "result count")
        for (i in expected.indices) {
            assertEquals(expected[i].size, actual[i].size, "result[$i] size")
            for (j in expected[i].indices) {
                val e = expected[i][j]
                val diff = abs(e - actual[i][j])
                assertTrue(
                    diff <= tol * max(1f, abs(e)),
                    "result[$i][$j]: interpreter=$e metal=${actual[i][j]} diff=$diff",
                )
            }
        }
    }

    @Test
    fun platformNameIsMetalHlo() {
        requireMetal()
        metalSession().use { assertEquals("metalhlo", it.platformName()) }
    }

    @Test
    fun scalarAffineAgreesWithInterpreter() {
        requireMetal()
        val fn = DxirBuilder.function("scalar_affine") {
            val x = param("x", scalar)
            val mul = op(OpKind.MUL, listOf(x, const(2.0f, scalar)), scalar)
            listOf(op(OpKind.ADD, listOf(mul, const(1.0f, scalar)), scalar))
        }
        val inputs = listOf(floatArrayOf(7.0f))
        metalSession().use { assertMatchesInterpreter(fn, inputs, it.runOn(fn, inputs)) }
    }

    @Test
    fun matmulTanhSumAgreesWithInterpreter() {
        requireMetal()
        val a = DxirType(F32, listOf(4, 8))
        val b = DxirType(F32, listOf(8, 3))
        val c = DxirType(F32, listOf(4, 3))
        val fn = DxirBuilder.function("mlp_head") {
            val x = param("x", a)
            val w = param("w", b)
            val h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(x, w), c)), c)
            listOf(h, op(OpKind.SUM, listOf(h), scalar))
        }
        val inputs = listOf(
            FloatArray(32) { (it % 7 - 3) * 0.1f },
            FloatArray(24) { (it % 5 - 2) * 0.2f },
        )
        metalSession().use { session ->
            assertMatchesInterpreter(fn, inputs, session.runOn(fn, inputs))
            session.runOn(fn, inputs)
            assertEquals(1, session.cacheSize, "the second call reuses the compiled executable")
        }
    }

    @Test
    fun oneShotRunOnPjrtUsesTheMetalPlugin() {
        requireMetal()
        val v = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("exp_plus") {
            val x = param("x", v)
            listOf(op(OpKind.ADD, listOf(op(OpKind.EXP, listOf(x), v), x), v))
        }
        val inputs = listOf(floatArrayOf(-1f, 0f, 0.5f, 2f))
        assertMatchesInterpreter(fn, inputs, runOnPjrt(fn, inputs, PjrtTarget.Metal))
    }
}
