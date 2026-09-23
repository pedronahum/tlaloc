package io.tlaloc.runtime.iree

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Temporary files of the IREE subprocess facade are removed when each call
 * returns, not at JVM exit, and the IREE tools are found in any venv, on
 * PATH, or through TLALOC_IREE_BIN.
 */
class IreeHousekeepingTest {

    private val tmp: Path = Path.of(System.getProperty("java.io.tmpdir"))

    private fun ireeTempEntries(): Set<String> =
        Files.list(tmp).use { s -> s.map { it.fileName.toString() }.filter { it.startsWith("tlaloc-iree") }.toList().toSet() }

    private fun assumeIree() = assumeTrue(IreeBinaries.available, "IREE tools not resolved — skipping.\n${IreeBinaries.searchReport}")

    private val v3 = DxirType(F32, listOf(3))

    private fun double() = DxirBuilder.function("double_it") {
        val x = param("x", v3)
        listOf(op(OpKind.ADD, listOf(x, x), v3))
    }

    @Test
    fun runOnIreeLeavesNoTemporaryFilesBehind() {
        assumeIree()
        val before = ireeTempEntries()
        for (npy in listOf(false, true)) {
            val out = runOnIree(double(), listOf(floatArrayOf(1f, 2f, 3f)), useNpyInputs = npy)
            assertEquals(listOf(2f, 4f, 6f), out.single().toList())
        }
        assertEquals(emptySet(), ireeTempEntries() - before, "temporary IREE files left in $tmp")
    }

    @Test
    fun aFailedCompileRemovesItsDirectory() {
        assumeIree()
        val before = ireeTempEntries()
        assertFailsWith<IreeCompileException> { IreeRuntime.compile("this is not MLIR") }
        assertEquals(emptySet(), ireeTempEntries() - before, "a failed compile left files in $tmp")
    }

    @Test
    fun closingAModuleDeletesItsDirectory() {
        assumeIree()
        val module = IreeRuntime.compile(
            """
            module {
              func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
                return %arg0 : tensor<f32>
              }
            }
            """.trimIndent(),
        )
        val dir = module.vmfbPath.parent
        assertTrue(Files.exists(module.vmfbPath))
        assertEquals(listOf("module.vmfb"), Files.list(dir).use { s -> s.map { it.fileName.toString() }.toList() },
            "only the VMFB stays after compile; the MLIR input is removed")
        module.close()
        assertFalse(Files.exists(dir))
        module.close()
    }

    // ---------------------------------------------------------------- every host

    private fun executable(dir: Path, name: String): Path {
        Files.createDirectories(dir)
        val f = dir.resolve(name)
        Files.writeString(f, "#!/bin/sh\n")
        Files.setPosixFilePermissions(f, PosixFilePermissions.fromString("rwxr-xr-x"))
        return f
    }

    @Test
    fun theToolsAreSearchedForInEnvVenvAnyLocalVenvThenPath() {
        val root = Files.createTempDirectory("iree-bins-")
        try {
            val home = root.resolve("home")
            val pathA = root.resolve("path-a")
            val dirs = IreeBinaries.searchDirs(
                binEnv = root.resolve("explicit").toString(),
                virtualEnv = root.resolve("active").toString(),
                home = home.toString(),
                pathEnv = "$pathA:${root.resolve("path-b")}",
            )
            Files.createDirectories(home.resolve(".local/venvs/zeta"))
            Files.createDirectories(home.resolve(".local/venvs/alpha"))
            val withVenvs = IreeBinaries.searchDirs(root.resolve("explicit").toString(), root.resolve("active").toString(), home.toString(), "$pathA")
            assertEquals(
                listOf(
                    root.resolve("explicit"),
                    root.resolve("active/bin"),
                    home.resolve(".local/venvs/alpha/bin"),
                    home.resolve(".local/venvs/zeta/bin"),
                    pathA,
                ),
                withVenvs,
            )
            assertEquals(root.resolve("path-b"), dirs.last())

            assertNull(IreeBinaries.resolveTool("iree-compile", withVenvs))
            val onPath = executable(pathA, "iree-compile")
            assertEquals(onPath.toString(), IreeBinaries.resolveTool("iree-compile", withVenvs))
            val inVenv = executable(home.resolve(".local/venvs/zeta/bin"), "iree-compile")
            assertEquals(inVenv.toString(), IreeBinaries.resolveTool("iree-compile", withVenvs))
            val notExecutable = home.resolve(".local/venvs/alpha/bin/iree-compile")
            Files.createDirectories(notExecutable.parent)
            Files.writeString(notExecutable, "")
            assertEquals(inVenv.toString(), IreeBinaries.resolveTool("iree-compile", withVenvs), "a non-executable file is skipped")
        } finally {
            IreeRuntime.deleteRecursively(root)
        }
    }

    @Test
    fun theSearchReportNamesEveryDirectoryAndTheFix() {
        val root = Files.createTempDirectory("iree-bins-")
        try {
            val report = IreeBinaries.describeSearch(null, null, root.toString(), root.resolve("p").toString())
            assertTrue("\$TLALOC_IREE_BIN: not set" in report, report)
            assertTrue(root.resolve("p").toString() in report, report)
            assertTrue("iree-compile: not found" in report && "Fix:" in report, report)
            val e = assertFailsWith<IllegalStateException> { IreeBinaries.requireTool("iree-compile", null) }
            assertTrue(e.message!!.startsWith("iree-compile not found.\nIREE tool resolution"), e.message)
        } finally {
            IreeRuntime.deleteRecursively(root)
        }
    }
}
