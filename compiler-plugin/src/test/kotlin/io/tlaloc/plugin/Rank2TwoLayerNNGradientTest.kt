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
 * §0.4.199 — CartPole Phase 3 second slice. First end-to-end gradient through a
 * 2-layer NN forward (`relu(X · W1) · W2`).sum()`) with two rectangular MATMULs
 * and a tensor RELU between them. Tests:
 *  - 3-param `grad` lambda with `Triple` return.
 *  - Tensor RELU on the synthesis side (forward path; preserved in the gradient
 *    body because `MatmulRule` reads `relu(matmul1)` as the LHS primal for the
 *    inner matmul's adjoint).
 *  - Two MATMUL gradients chained through a RELU's STEP mask.
 *
 * Primal: `grad { (X, W1, W2) -> (relu(X matmul W1) matmul W2).sum().toFloat() }`
 * with `X: Rank2<Sym, Lit<Int>>`, `W1: Rank2<Lit<Int>, Lit<Long>>`,
 * `W2: Rank2<Lit<Long>, Lit<Short>>` — four distinct shape atoms across the chain.
 * Concrete shape (B=2, I=3, H=4, O=2) at runtime.
 *
 * For X = ones(2,3), W1 = `[[1,-1,1,-1] × 3]` (3×4), W2 = ones(4,2):
 *   - y1 = X · W1 = `[[3,-3,3,-3] × 2]` (2×4)
 *   - y1r = relu(y1) = `[[3,0,3,0] × 2]` (2×4)
 *   - y2 = y1r · W2 = `[[6,6],[6,6]]` (2×2; each col gets two 3s)
 *   - L = sum(y2) = 24
 *
 * Analytic gradients:
 *   - ∂L/∂y2 = ones(2,2)
 *   - ∂L/∂y1r = ones(2,2) · W2^T = (each entry = 2) → `[[2,2,2,2] × 2]` (2×4)
 *   - ∂L/∂W2 = y1r^T · ones(2,2). y1r^T row j = [3,3] when j∈{0,2} else [0,0].
 *     `[[6,6],[0,0],[6,6],[0,0]]` (4×2)
 *   - ∂L/∂y1 = ∂L/∂y1r * step(y1) = `[[2,0,2,0] × 2]` (2×4)
 *   - ∂L/∂X = ∂L/∂y1 · W1^T. Each row of ∂L/∂y1 hits W1^T's positive cols (0, 2)
 *     contributing 2+2=4. ∂L/∂X = `[[4,4,4],[4,4,4]]` (2×3)
 *   - ∂L/∂W1 = X^T · ∂L/∂y1 = `[[4,0,4,0] × 3]` (3×4; each row of X^T is ones)
 */
class Rank2TwoLayerNNGradientTest {

    @Test
    fun `grad of 2-layer relu net matches analytic`() {
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
                                w2: DTensor<Rank2<Lit<Long>, Lit<Short>>, F32> ->
                    ((x matmul w1).relu() matmul w2).sum().toFloat()
                }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, FloatArray(6) { 1.0f })
                val W1 = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(
                    3, 4,
                    floatArrayOf(1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f),
                )
                val W2 = Tensors.f32Matrix<Lit<Long>, Lit<Short>>(4, 2, FloatArray(8) { 1.0f })
                val (dX, dW1, dW2) = g(X, W1, W2)
                val flatX = dX.hostF32()
                val flatW1 = dW1.hostF32()
                val flatW2 = dW2.hostF32()
                println("${'$'}{flatX.size} ${'$'}{dX.dims[0]} ${'$'}{dX.dims[1]}")
                for (v in flatX) print("${'$'}v ")
                println()
                println("${'$'}{flatW1.size} ${'$'}{dW1.dims[0]} ${'$'}{dW1.dims[1]}")
                for (v in flatW1) print("${'$'}v ")
                println()
                println("${'$'}{flatW2.size} ${'$'}{dW2.dims[0]} ${'$'}{dW2.dims[1]}")
                for (v in flatW2) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_3ARG, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected 2-layer NN gradient to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(6, lines.size, "expected 6 stdout lines, got: ${result.stdout}")
        val headerX = lines[0].trim().split(" ").map { it.toInt() }
        assertEquals(listOf(6, 2, 3), headerX, "dX should be 2x3")
        val valuesX = lines[1].trim().split(" ").map { it.toFloat() }
        val headerW1 = lines[2].trim().split(" ").map { it.toInt() }
        assertEquals(listOf(12, 3, 4), headerW1, "dW1 should be 3x4")
        val valuesW1 = lines[3].trim().split(" ").map { it.toFloat() }
        val headerW2 = lines[4].trim().split(" ").map { it.toInt() }
        assertEquals(listOf(8, 4, 2), headerW2, "dW2 should be 4x2")
        val valuesW2 = lines[5].trim().split(" ").map { it.toFloat() }

        val expectedX = floatArrayOf(4f, 4f, 4f, 4f, 4f, 4f)
        val expectedW1 = floatArrayOf(4f, 0f, 4f, 0f, 4f, 0f, 4f, 0f, 4f, 0f, 4f, 0f)
        val expectedW2 = floatArrayOf(6f, 6f, 0f, 0f, 6f, 6f, 0f, 0f)
        for ((i, v) in valuesX.withIndex()) {
            assertTrue(abs(v + 1.0f) > 1e-6f, "dX slot $i = $v matches broken-stub sentinel")
            assertTrue(abs(v - expectedX[i]) < 1e-3f, "dX slot $i = $v expected ${expectedX[i]}")
        }
        for ((i, v) in valuesW1.withIndex()) {
            assertTrue(abs(v + 1.0f) > 1e-6f, "dW1 slot $i = $v matches broken-stub sentinel")
            assertTrue(abs(v - expectedW1[i]) < 1e-3f, "dW1 slot $i = $v expected ${expectedW1[i]}")
        }
        for ((i, v) in valuesW2.withIndex()) {
            assertTrue(abs(v + 1.0f) > 1e-6f, "dW2 slot $i = $v matches broken-stub sentinel")
            assertTrue(abs(v - expectedW2[i]) < 1e-3f, "dW2 slot $i = $v expected ${expectedW2[i]}")
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
        val tempDir = Files.createTempDirectory("tlaloc-2layer-test").toFile()
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
        private val AUTOGRAD_STUB_3ARG = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>,
                          DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                          DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>,
                     DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                     DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>) ->
                        Triple<DTensor<Rank2<Sym, Lit<Int>>, F32>,
                               DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                               DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>> =
                { _, _, _ -> Triple(
                    DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32),
                    DTensor(HostF32Storage(FloatArray(12) { -1.0f }), intArrayOf(3, 4), F32),
                    DTensor(HostF32Storage(FloatArray(8) { -1.0f }), intArrayOf(4, 2), F32),
                ) }
        """.trimIndent()
    }
}
