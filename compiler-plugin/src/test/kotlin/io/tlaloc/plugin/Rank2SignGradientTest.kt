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
 * §0.4.204 — CartPole Phase 3 sixth slice. First end-to-end gradient through
 * tensor `sign()`. Tests SignRule's zero-gradient adjoint emission + the
 * `irSign` synthesis arm.
 *
 * Primal: `grad { (X, W) -> (X matmul W).tanh().sign().sum().toFloat() }`
 * with `X: Rank2<Sym, Lit<Int>>, W: Rank2<Lit<Int>, Lit<Long>>`.
 *
 * Since `sign()` has identically-zero gradient (per `SignRule`), both ∂X and
 * ∂W are zero tensors regardless of input values. Test asserts ∂X = zeros(2, 3)
 * and ∂W = zeros(3, 4). This is the design contract for sign in CartPole's
 * `a = sign(tanh(...) - ε)` discretisation: gradients stop at the action
 * boundary; downstream RL training uses policy-gradient mechanisms.
 */
class Rank2SignGradientTest {

    @Test
    fun `grad through sign is zero-tensor regardless of input`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.matmul
            import io.tlaloc.core.ops.sign
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.tanh
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Lit<Int>>, F32>, w: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32> ->
                    (x matmul w).tanh().sign().sum().toFloat()
                }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3,
                    floatArrayOf(0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f))
                val W = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(3, 4,
                    floatArrayOf(0.7f, -0.7f, 0.8f, -0.8f, 0.9f, -0.9f, 1.0f, -1.0f, 1.1f, -1.1f, 1.2f, -1.2f))
                val (dX, dW) = g(X, W)
                val flatX = dX.hostF32()
                val flatW = dW.hostF32()
                println("dX ${'$'}{flatX.size}")
                for (v in flatX) print("${'$'}v ")
                println()
                println("dW ${'$'}{flatW.size}")
                for (v in flatW) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_RECTANGULAR, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected end-to-end sign gradient to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")

        val valuesX = lines[1].trim().split(" ").map { it.toFloat() }
        val valuesW = lines[3].trim().split(" ").map { it.toFloat() }

        assertEquals(6, valuesX.size, "dX size")
        assertEquals(12, valuesW.size, "dW size")
        for ((i, v) in valuesX.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dX slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(abs(v) < 1e-3f, "dX slot $i = $v expected ~0 (sign has zero gradient)")
        }
        for ((i, v) in valuesW.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dW slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(abs(v) < 1e-3f, "dW slot $i = $v expected ~0 (sign has zero gradient)")
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
        val tempDir = Files.createTempDirectory("tlaloc-sign-test").toFile()
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
            } catch (t: Throwable) {
                RunResult(
                    2,
                    collected + CompileMessage(
                        CompilerMessageSeverity.ERROR,
                        "RUN FAILURE: ${t.cause ?: t}",
                    ),
                    baos.toString(Charsets.UTF_8),
                )
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        private val AUTOGRAD_STUB_RECTANGULAR = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>) ->
                        Pair<DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32),
                    DTensor(HostF32Storage(FloatArray(12) { -1.0f }), intArrayOf(3, 4), F32),
                ) }
        """.trimIndent()
    }
}
