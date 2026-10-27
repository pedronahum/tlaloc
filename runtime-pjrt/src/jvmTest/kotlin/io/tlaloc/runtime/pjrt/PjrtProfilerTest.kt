package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The plugin's profiler around a few steps: an XSpace with the GPU's plane and its kernels. */
class PjrtProfilerTest {

    @Test
    fun aProfiledRunWritesAnXSpaceWithTheDevicesActivity(@TempDir dir: Path) {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val t = DxirType(F32, listOf(256, 256))
        val fn = DxirBuilder.function("profiled_mm") {
            val a = param("a", t)
            listOf(op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(a, a), t)), t))
        }
        val x = FloatArray(256 * 256) { (it % 7) * 0.01f }
        val trace = dir.resolve("run.xplane.pb")
        PjrtSession(target = PjrtTarget.Cuda, diskCache = null).use { s ->
            s.runOn(fn, listOf(x))
            val out = s.profile(trace) { List(3) { s.runOn(fn, listOf(x)) } }
            assertEquals(3, out.size)
        }
        val bytes = Files.readAllBytes(trace)
        val text = String(bytes, Charsets.ISO_8859_1)
        println("PjrtProfilerTest: ${bytes.size} bytes of XSpace; names: " + Regex("/[a-z]+:[A-Za-z_]+[:0-9]*").findAll(text).map { it.value }.toSet())
        assertTrue(bytes.size > 1000, "a trace of ${bytes.size} bytes")
        assertTrue("/device:GPU:0" in text, "no GPU plane in the trace")
        assertTrue("/host:CPU" in text, "no host plane in the trace")
        assertTrue("fusion" in text, "no XLA kernel in the trace")
    }
}
