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
 * §0.4.205 — CartPole Phase 3 seventh slice. Full primitive composition test for
 * the CartPole NN forward chain `a = sign(tanh(relu(relu(X · W1) · W2) · W3))`.
 * Combines every Phase 3 primitive shipped in §0.4.198–§0.4.204:
 *  - 3 rectangular MATMULs (Phase 0c-rectangular surface, §0.4.197)
 *  - 2 tensor RELUs (§0.4.199 + forward elementwise propagation)
 *  - 1 tensor TANH (§0.4.200)
 *  - 1 tensor SIGN (§0.4.204)
 *  - 4-grad-param surface with Quadruple return (§0.4.203)
 *  - 5 distinct ShapeAtoms threaded through the matmul chain (Sym, Lit<Int>,
 *    Lit<Long>, Lit<Short>, Lit<Byte>)
 *
 * Gradient is identically zero through `sign` (per `SignRule`), so the test
 * asserts:
 *  - synthesis didn't fall back ("kept original call" warning absent)
 *  - all four gradient outputs (∂X, ∂W1, ∂W2, ∂W3) are zero-tensors
 *
 * The forward path exercises the full chain composition; if any primitive
 * surfaces a bug in chain composition (rather than in isolated unit tests),
 * this firing catches it.
 *
 * This test mirrors the structure CartPole's NN uses, sans the scalar `- ε`
 * subtraction step (which requires explicit `broadcastLike` since DTensor
 * doesn't have a `minus(Float)` overload). Adding the epsilon shift would
 * be additive — same gradient (zero) since sign blocks it.
 */
class CartPoleNNChainTest {

    @Test
    fun `full sign-tanh-relu-relu-matmul3 chain lowers and gradient is zero`() {
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
            import io.tlaloc.core.ops.sign
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.tanh
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Lit<Int>>, F32>,
                                w1: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                                w2: DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
                                w3: DTensor<Rank2<Lit<Short>, Lit<Byte>>, F32> ->
                    (((x matmul w1).relu() matmul w2).relu() matmul w3).tanh().sign().sum().toFloat()
                }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3,
                    floatArrayOf(0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f))
                val W1 = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(3, 4, FloatArray(12) { 0.5f })
                val W2 = Tensors.f32Matrix<Lit<Long>, Lit<Short>>(4, 5, FloatArray(20) { 0.5f })
                val W3 = Tensors.f32Matrix<Lit<Short>, Lit<Byte>>(5, 2, FloatArray(10) { 0.5f })
                val q = g(X, W1, W2, W3)
                val flatX = q.first.hostF32()
                val flatW1 = q.second.hostF32()
                val flatW2 = q.third.hostF32()
                val flatW3 = q.fourth.hostF32()
                println("dX ${'$'}{flatX.size}")
                for (v in flatX) print("${'$'}v ")
                println()
                println("dW1 ${'$'}{flatW1.size}")
                for (v in flatW1) print("${'$'}v ")
                println()
                println("dW2 ${'$'}{flatW2.size}")
                for (v in flatW2) print("${'$'}v ")
                println()
                println("dW3 ${'$'}{flatW3.size}")
                for (v in flatW3) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_4ARG, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected full sign-tanh-relu-relu-matmul3 chain to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(8, lines.size, "expected 8 stdout lines, got: ${result.stdout}")

        fun parseValuesLine(line: String): FloatArray =
            line.trim().split(" ").map { it.toFloat() }.toFloatArray()

        val valuesX = parseValuesLine(lines[1])
        val valuesW1 = parseValuesLine(lines[3])
        val valuesW2 = parseValuesLine(lines[5])
        val valuesW3 = parseValuesLine(lines[7])

        assertEquals(6, valuesX.size, "dX size")
        assertEquals(12, valuesW1.size, "dW1 size")
        assertEquals(20, valuesW2.size, "dW2 size")
        assertEquals(10, valuesW3.size, "dW3 size")

        // All gradients should be zero — sign blocks gradient flow per SignRule.
        // Also verify they're not the broken-stub sentinel (-1.0).
        fun assertZeroAll(name: String, values: FloatArray) {
            for ((i, v) in values.withIndex()) {
                assertTrue(
                    abs(v + 1.0f) > 1e-6f,
                    "$name slot $i = $v matches broken-stub sentinel",
                )
                assertTrue(
                    abs(v) < 1e-3f,
                    "$name slot $i = $v expected ~0 (sign blocks gradient flow)",
                )
            }
        }
        assertZeroAll("dX", valuesX)
        assertZeroAll("dW1", valuesW1)
        assertZeroAll("dW2", valuesW2)
        assertZeroAll("dW3", valuesW3)
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-cartpole-chain-test").toFile()
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
