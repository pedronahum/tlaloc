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
 * §0.4.364 — the shape-plumbing user surface reaches `grad {}` lambdas
 * (the deferred half of DiffKT-gap item 2): `where` + comparison masks
 * differentiate end-to-end through the K2 plugin.
 *
 * Primal: `grad { a, b -> where(a gt b, a * a, b).sum().toFloat() }`.
 * The FIR lowering turns `gt` into COMPARE(GT):Bool + CAST and `where`
 * into COMPARE(≠0) + WHERE; the synthesis maps them back onto the
 * `:core/ops` host runtime (Bool tensors ride as 0/1 F32 masks).
 *
 * Analytic gradients through WhereRule's mask routing:
 * `da = 2a` where `a > b`, else 0; `db = 1` where `a ≤ b`, else 0.
 */
class WhereCompareGradientTest {

    @Test
    fun `grad through where and gt routes by comparison mask`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.gt
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            import io.tlaloc.core.ops.where
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    where(a gt b, a * a, b).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2,
                    floatArrayOf(1.0f, -2.0f, 3.0f, 0.5f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2,
                    floatArrayOf(0.5f, 1.0f, -1.0f, 2.0f))
                val (dA, dB) = g(A, B)
                val flatA = dA.hostF32()
                val flatB = dB.hostF32()
                println("dA " + flatA.size)
                for (v in flatA) print("" + v + " ")
                println()
                println("dB " + flatB.size)
                for (v in flatB) print("" + v + " ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected end-to-end where/gt gradient to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")

        val valuesA = lines[1].trim().split(" ").map { it.toFloat() }
        val valuesB = lines[3].trim().split(" ").map { it.toFloat() }

        // a = [1, -2, 3, 0.5], b = [0.5, 1, -1, 2] → a>b = [T, F, T, F]:
        // da = 2a⊙mask = [2, 0, 6, 0]; db = 1−mask = [0, 1, 0, 1].
        val wantA = listOf(2.0f, 0.0f, 6.0f, 0.0f)
        val wantB = listOf(0.0f, 1.0f, 0.0f, 1.0f)
        assertEquals(4, valuesA.size, "dA size")
        assertEquals(4, valuesB.size, "dB size")
        assertTrue(
            valuesA[0] != -1.0f || valuesA[1] != -1.0f,
            "stub sentinel returned — rewrite never fired. All compiler messages:\n" +
                result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
        )
        for (i in 0 until 4) {
            assertTrue(
                abs(valuesA[i] - wantA[i]) < 1e-5f,
                "dA[$i] = ${valuesA[i]}, want ${wantA[i]} (mask routing broken)",
            )
            assertTrue(
                abs(valuesB[i] - wantB[i]) < 1e-5f,
                "dB[$i] = ${valuesB[i]}, want ${wantB[i]} (complement routing broken)",
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
        val tempDir = Files.createTempDirectory("tlaloc-where-test").toFile()
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
        private val AUTOGRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) ->
                        Pair<DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                ) }
        """.trimIndent()
    }
}
