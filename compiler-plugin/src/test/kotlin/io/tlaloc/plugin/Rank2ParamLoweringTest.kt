package io.tlaloc.plugin

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import java.net.URLClassLoader
import java.nio.file.Files
import org.jetbrains.kotlin.cli.common.arguments.K2JVMCompilerArguments
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import org.jetbrains.kotlin.config.Services
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.185 + §0.4.186 — Phase 0c slices (a) + (b). FIR-side recognises Rank2/3 DTensor
 * parameters; synthesis-side accepts rank-1/2/3 F32 in the grad function's nodes and
 * routes BROADCAST ops through `broadcastLike<S>(v, template)` for any rank.
 *
 * Slice (b) closes the rank-2 BROADCAST gap by widening the synthesise() gate +
 * `irBroadcast` from rank-1-only to rank-1/2/3. The grad function for
 * `grad { a: DTensor<Rank2<R, C>, F32> -> 0.0f }` emits `BROADCAST(const 0.0,
 * a's-rank-2-shape)` (the zero-tensor gradient when the body doesn't reference `a`),
 * which now lowers cleanly to `broadcastLike<Rank2<R, C>>(0.0f, a)` in IR.
 *
 * The test asserts end-to-end success: the K2-synthesised gradient produces a real
 * rank-2 zero-tensor instead of falling back to the broken-stub sentinel.
 */
class Rank2ParamLoweringTest {

    @Test
    fun `rank2 DTensor param grad zero through synthesised path`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> 0.0f }
                val input = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
                val gradOut = g(input)
                val flat = gradOut.hostF32()
                println(flat.size)
                for (v in flat) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK2_TO_FLOAT, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")

        // Sanity: synthesis succeeded — no "kept original call" warning.
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back unexpectedly; rank-2 BROADCAST should now lower per §0.4.186. " +
                "Warnings: ${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // Output: flat size = 2*3 = 6 zeros (the grad of a constant body is the zero tensor).
        val lines = result.stdout.trim().lines()
        assertEquals(2, lines.size, "expected 2 stdout lines (size + values), got: ${result.stdout}")
        assertEquals("6", lines[0], "rank-2 grad output should have size 6 (2×3)")
        val values = lines[1].trim().split(" ").map { it.toFloat() }
        assertEquals(6, values.size)
        for ((i, v) in values.withIndex()) {
            // grad of a constant body is zero everywhere; sentinel-defeated.
            assertTrue(
                abs(v) < 1e-6f,
                "slot $i = $v expected 0.0 (grad of constant body)",
            )
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "slot $i = $v matches broken-stub sentinel; synthesis didn't fire",
            )
        }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-rank2-probe").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(severity: CompilerMessageSeverity, message: String, location: CompilerMessageSourceLocation?) {
                    collected += CompileMessage(severity, message)
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                pluginClasspaths = pluginClasspath()
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            if (exitCode != 0) return RunResult(exitCode, collected, "")
            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val capturedOut = PrintStream(baos, true, Charsets.UTF_8)
            val urls = arrayOf(outDir.toURI().toURL())
            val loader = URLClassLoader(urls, javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                val mainCls = loader.loadClass("MainKt")
                val mainMethod = mainCls.getMethod("main")
                mainMethod.invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        private val AUTOGRAD_STUB_BROKEN_RANK2_TO_FLOAT = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Sym>, F32>) -> DTensor<Rank2<Sym, Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32) }
        """.trimIndent()
    }
}
