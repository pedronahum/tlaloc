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
 * §0.4.203 — CartPole Phase 3 fifth slice. First 4-grad-output gradient through
 * the K2 plugin. Tests:
 *  - `synthesise()` cap widened from 3 to 4 returns (uses
 *    `io.tlaloc.autograd.Quadruple` boxing introduced in §0.4.134).
 *  - 3-layer NN forward `(((X · W1).relu() · W2).relu() · W3).sum()` with four
 *    rectangular weight tensors (X, W1, W2, W3) and FIVE distinct shape atoms.
 *
 * Five atoms used to keep all matmul shapes structurally distinct:
 *   Sym (= B), Lit<Int> (= I), Lit<Long> (= H1), Lit<Short> (= H2), Lit<Byte> (= O).
 *
 * For X = ones(2, 3), W1 = ones(3, 4), W2 = ones(4, 5), W3 = ones(5, 2):
 *   - y1 = X · W1 = `[[3] × 4 × 2]` (2×4)
 *   - y1r = relu(y1) = same as y1 (all positive)
 *   - y2 = y1r · W2 = `[[12] × 5 × 2]` (2×5; each row entry = 4 columns × 3 = 12)
 *   - y2r = relu(y2) = same as y2 (all positive)
 *   - y3 = y2r · W3 = `[[60] × 2 × 2]` (2×2; each row entry = 5 columns × 12 = 60)
 *   - L = sum(y3) = 240
 *
 * Analytic gradients (ones × W^T chains; positive-mask relu is identity):
 *   - ∂L/∂y3 = ones(2, 2)
 *   - ∂L/∂y2r = ones(2, 2) · W3^T = each entry = 2. (2, 5) of 2s.
 *   - ∂L/∂W3 = y2r^T · ones(2, 2) = each entry = 24 (2 × 12). (5, 2) of 24s.
 *   - ∂L/∂y2 = ∂L/∂y2r * step(y2) = (2, 5) of 2s (step(y2)=ones since y2>0).
 *   - ∂L/∂y1r = ∂L/∂y2 · W2^T = each entry = 10 (sum of 5 cols × 2). (2, 4) of 10s.
 *   - ∂L/∂W2 = y1r^T · ∂L/∂y2. y1r^T is (4, 2) of 3s, ∂L/∂y2 is (2, 5) of 2s.
 *     Each entry = sum_k 3 × 2 over k=0..1 = 12. (4, 5) of 12s.
 *   - ∂L/∂y1 = ∂L/∂y1r * step(y1) = (2, 4) of 10s.
 *   - ∂L/∂X = ∂L/∂y1 · W1^T = each entry = 40 (sum of 4 cols × 10). (2, 3) of 40s.
 *   - ∂L/∂W1 = X^T · ∂L/∂y1 = each entry = 20 (2 batch × 10). (3, 4) of 20s.
 */
class Rank2ThreeLayerNNGradientTest {

    @Test
    fun `grad of 3-layer relu net with 4 params matches analytic`() {
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
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Lit<Int>>, F32>,
                                w1: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                                w2: DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
                                w3: DTensor<Rank2<Lit<Short>, Lit<Byte>>, F32> ->
                    (((x matmul w1).relu() matmul w2).relu() matmul w3).sum().toFloat()
                }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, FloatArray(6) { 1.0f })
                val W1 = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(3, 4, FloatArray(12) { 1.0f })
                val W2 = Tensors.f32Matrix<Lit<Long>, Lit<Short>>(4, 5, FloatArray(20) { 1.0f })
                val W3 = Tensors.f32Matrix<Lit<Short>, Lit<Byte>>(5, 2, FloatArray(10) { 1.0f })
                val q = g(X, W1, W2, W3)
                val flatX = q.first.hostF32()
                val flatW1 = q.second.hostF32()
                val flatW2 = q.third.hostF32()
                val flatW3 = q.fourth.hostF32()
                println("dX ${'$'}{flatX.size} ${'$'}{q.first.dims[0]} ${'$'}{q.first.dims[1]}")
                for (v in flatX) print("${'$'}v ")
                println()
                println("dW1 ${'$'}{flatW1.size} ${'$'}{q.second.dims[0]} ${'$'}{q.second.dims[1]}")
                for (v in flatW1) print("${'$'}v ")
                println()
                println("dW2 ${'$'}{flatW2.size} ${'$'}{q.third.dims[0]} ${'$'}{q.third.dims[1]}")
                for (v in flatW2) print("${'$'}v ")
                println()
                println("dW3 ${'$'}{flatW3.size} ${'$'}{q.fourth.dims[0]} ${'$'}{q.fourth.dims[1]}")
                for (v in flatW3) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_4ARG, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected 3-layer NN gradient with 4 params to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(8, lines.size, "expected 8 stdout lines, got: ${result.stdout}")

        fun parseValuesLine(line: String): FloatArray =
            line.trim().split(" ").map { it.toFloat() }.toFloatArray()

        val headerX = lines[0].trim().split(" ").drop(1).map { it.toInt() }
        assertEquals(listOf(6, 2, 3), headerX, "dX header")
        val valuesX = parseValuesLine(lines[1])
        val headerW1 = lines[2].trim().split(" ").drop(1).map { it.toInt() }
        assertEquals(listOf(12, 3, 4), headerW1, "dW1 header")
        val valuesW1 = parseValuesLine(lines[3])
        val headerW2 = lines[4].trim().split(" ").drop(1).map { it.toInt() }
        assertEquals(listOf(20, 4, 5), headerW2, "dW2 header")
        val valuesW2 = parseValuesLine(lines[5])
        val headerW3 = lines[6].trim().split(" ").drop(1).map { it.toInt() }
        assertEquals(listOf(10, 5, 2), headerW3, "dW3 header")
        val valuesW3 = parseValuesLine(lines[7])

        val expectedX = FloatArray(6) { 40f }
        val expectedW1 = FloatArray(12) { 20f }
        val expectedW2 = FloatArray(20) { 12f }
        val expectedW3 = FloatArray(10) { 24f }

        fun assertCloseAll(name: String, actual: FloatArray, expected: FloatArray) {
            assertEquals(expected.size, actual.size, "$name size")
            for (i in expected.indices) {
                assertTrue(
                    abs(actual[i] + 1.0f) > 1e-6f,
                    "$name slot $i = ${actual[i]} matches broken-stub sentinel",
                )
                assertTrue(
                    abs(actual[i] - expected[i]) < 1e-3f,
                    "$name slot $i = ${actual[i]} expected ${expected[i]}",
                )
            }
        }
        assertCloseAll("dX", valuesX, expectedX)
        assertCloseAll("dW1", valuesW1, expectedW1)
        assertCloseAll("dW2", valuesW2, expectedW2)
        assertCloseAll("dW3", valuesW3, expectedW3)
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-3layer-test").toFile()
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
        private val AUTOGRAD_STUB_4ARG = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>,
                          DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                          DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
                          DTensor<Rank2<Lit<Short>, Lit<Byte>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>,
                     DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                     DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
                     DTensor<Rank2<Lit<Short>, Lit<Byte>>, F32>) ->
                        Quadruple<DTensor<Rank2<Sym, Lit<Int>>, F32>,
                                   DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                                   DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
                                   DTensor<Rank2<Lit<Short>, Lit<Byte>>, F32>> =
                { _, _, _, _ -> Quadruple(
                    DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32),
                    DTensor(HostF32Storage(FloatArray(12) { -1.0f }), intArrayOf(3, 4), F32),
                    DTensor(HostF32Storage(FloatArray(20) { -1.0f }), intArrayOf(4, 5), F32),
                    DTensor(HostF32Storage(FloatArray(10) { -1.0f }), intArrayOf(5, 2), F32),
                ) }
        """.trimIndent()
    }
}
