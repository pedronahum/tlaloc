package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.name
import kotlin.io.path.readText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [TpuProgramRecorder], and the programs of the kernel fixtures, which only run on a TPU and so
 * are recorded here rather than by their tests (`scripts/tpu/aot-check.sh`).
 */
class TpuProgramRecorderTest {

    @Test
    fun recordsEachProgramOnceWithItsOptions() {
        val dir = Files.createTempDirectory("tlaloc-tpu-programs")
        try {
            val fn = DxirBuilder.function("double") {
                val x = param("x", DxirType(F32, listOf(4)))
                listOf(op(OpKind.ADD, listOf(x, x), x.type))
            }
            val mlir = PjrtSession.stablehloFor(fn, PjrtTarget.Tpu)
            repeat(2) { TpuProgramRecorder.record("double", mlir, CompileOptions.spmd(listOf(0L, 1L)), into = dir) }
            val files = dir.listDirectoryEntries().map { it.name }.sorted()
            assertEquals(2, files.size, "$files")
            assertTrue(files[0].startsWith("double-") && files[0].endsWith(".mlir"), "$files")
            assertEquals(mlir, dir.resolve(files[0]).readText())
            assertEquals(
                "replicas=1\npartitions=2\nspmd=true\nshardy=true\n",
                dir.resolve(files[1]).readText(),
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun recordsTheKernelFixturePrograms() {
        assumeTrue(TpuProgramRecorder.dir != null, "TLALOC_TPU_PROGRAMS_DIR not set")
        val fixtures = Paths.get(javaClass.getResource("/tpu-kernels")!!.toURI()).listDirectoryEntries().map { it.name }.sorted()
        for (name in fixtures) {
            val program = TpuKernelFixtures.load(name).program(referenceFallback = true)
            val mlir = PjrtSession.stablehloFor(program, PjrtTarget.Tpu)
            assertTrue("tpu_custom_call" in mlir, "$name: no tpu_custom_call in its TPU form")
            TpuProgramRecorder.record("kernel_$name", mlir, CompileOptions.DEFAULT)
        }
    }
}
