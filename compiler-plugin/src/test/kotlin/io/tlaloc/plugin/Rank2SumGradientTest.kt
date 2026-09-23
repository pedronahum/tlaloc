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
 * §0.4.188 — DTensor → Float bridge + `:core.ops.sum` plugin recognition.
 * Closes the last gap before active-gradient lambda bodies on rank-2 inputs.
 *
 * `grad { a: DTensor<Rank2<R, C>, F32> -> a.sum().toFloat() }` — the gradient
 * of `sum(a)` w.r.t. `a` is a tensor of ones with `a`'s shape (because
 * `∂(Σa_ij)/∂a_kl = 1` for every (k, l)). SumRule emits `BROADCAST(1.0, a.shape)`
 * which §0.4.186's synthesis-side widening lowers via `broadcastLike(1.0, a)`.
 *
 * This is the first ACTIVE rank-2 gradient through the K2 plugin — previous
 * §0.4.185/186 tests verified only the zero-gradient (constant body) case.
 */
class Rank2SumGradientTest {

    @Test
    fun `grad of sum on rank2 input is rank2 ones tensor`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> a.sum().toFloat() }
                val input = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
                val gradOut = g(input)
                val flat = gradOut.hostF32()
                println(flat.size)
                for (v in flat) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK2_2x3, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")

        // Confirm synthesis succeeded — no fallback.
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected rank-2 sum gradient to lower per §0.4.186 + §0.4.188. " +
                "Warnings: ${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // Output: 2×3 = 6 elements, all 1.0 (∂(Σa_ij)/∂a_kl = 1 for every (k, l)).
        val lines = result.stdout.trim().lines()
        assertEquals(2, lines.size, "expected 2 stdout lines (size + values), got: ${result.stdout}")
        assertEquals("6", lines[0], "rank-2 grad output should have size 6 (2×3)")
        val values = lines[1].trim().split(" ").map { it.toFloat() }
        assertEquals(6, values.size)
        for ((i, v) in values.withIndex()) {
            assertTrue(
                abs(v - 1.0f) < 1e-6f,
                "slot $i = $v expected 1.0 (∂Σa/∂a_kl = 1)",
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
        val tempDir = Files.createTempDirectory("tlaloc-rank2-sum-test").toFile()
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
        private val AUTOGRAD_STUB_BROKEN_RANK2_2x3 = """
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
