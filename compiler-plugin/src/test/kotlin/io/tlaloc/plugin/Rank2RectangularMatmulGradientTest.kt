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
 * §0.4.197 — first end-to-end rectangular MATMUL gradient test (R ≠ K ≠ C).
 * Closes the Phase 0c-rectangular arc opened in §0.4.191.
 *
 * Primal: `grad { (a, b) -> (a matmul b).sum().toFloat() }` with `a: Rank2<Sym, Lit<Int>>`
 * and `b: Rank2<Lit<Int>, Lit<Long>>` — three distinct ShapeAtoms (Sym, Lit<Int>,
 * Lit<Long>) at the IrType level. Concrete shape (2, 3, 4) at runtime.
 *
 * For A = [[1,2,3],[4,5,6]] (2×3) and B = all-ones (3×4):
 *   - ∂Σ(AB)/∂a_ik = Σ_j b_kj. With B all-ones, each row sum is 4.
 *     ∂A = 2×3 matrix of 4s = [[4,4,4],[4,4,4]].
 *   - ∂Σ(AB)/∂b_kj = Σ_i a_ik. ∂b_0j = 1+4 = 5, ∂b_1j = 2+5 = 7, ∂b_2j = 3+6 = 9.
 *     ∂B = [[5,5,5,5],[7,7,7,7],[9,9,9,9]].
 */
class Rank2RectangularMatmulGradientTest {

    @Test
    fun `2-arg grad of sum of A matmul B with rectangular shapes matches analytic`() {
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
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32> ->
                    (a matmul b).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
                val B = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(3, 4, FloatArray(12) { 1.0f })
                val (dA, dB) = g(A, B)
                val flatA = dA.hostF32()
                val flatB = dB.hostF32()
                println("${'$'}{flatA.size} ${'$'}{dA.dims[0]} ${'$'}{dA.dims[1]}")
                for (v in flatA) print("${'$'}v ")
                println()
                println("${'$'}{flatB.size} ${'$'}{dB.dims[0]} ${'$'}{dB.dims[1]}")
                for (v in flatB) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_RECTANGULAR, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected end-to-end rectangular MATMUL gradient to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")
        val headerA = lines[0].trim().split(" ").map { it.toInt() }
        assertEquals(listOf(6, 2, 3), headerA, "dA should be 2x3")
        val valuesA = lines[1].trim().split(" ").map { it.toFloat() }
        val headerB = lines[2].trim().split(" ").map { it.toInt() }
        assertEquals(listOf(12, 3, 4), headerB, "dB should be 3x4")
        val valuesB = lines[3].trim().split(" ").map { it.toFloat() }

        val expectedA = floatArrayOf(4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f)
        val expectedB = floatArrayOf(5.0f, 5.0f, 5.0f, 5.0f, 7.0f, 7.0f, 7.0f, 7.0f, 9.0f, 9.0f, 9.0f, 9.0f)
        for ((i, v) in valuesA.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dA slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(
                abs(v - expectedA[i]) < 1e-3f,
                "dA slot $i = $v expected ${expectedA[i]}",
            )
        }
        for ((i, v) in valuesB.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dB slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(
                abs(v - expectedB[i]) < 1e-3f,
                "dB slot $i = $v expected ${expectedB[i]}",
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
        val tempDir = Files.createTempDirectory("tlaloc-rank2-rect-test").toFile()
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
