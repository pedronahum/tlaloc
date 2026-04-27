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
import kotlin.math.tanh
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.200 — CartPole Phase 3 third slice. First end-to-end gradient through a
 * tensor TANH activation: `grad { (X, W) -> tanh(X matmul W).sum().toFloat() }`
 * with `X: Rank2<Sym, Lit<Int>>, W: Rank2<Lit<Int>, Lit<Long>>` (rectangular
 * shapes, three distinct shape atoms). Concrete shape (2, 3, 4) at runtime.
 *
 * For X = ones(2, 3) and W = `[[0.5, -0.5, 1.0, -1.0] × 3]` (3×4):
 *   - y = X · W = `[[1.5, -1.5, 3.0, -3.0] × 2]` (2×4)
 *   - tanh_y = tanh(y) = `[[T(1.5), -T(1.5), T(3.0), -T(3.0)] × 2]`
 *   - L = sum(tanh_y) = 2 * (T(1.5) - T(1.5) + T(3.0) - T(3.0)) = 0
 *
 * Analytic gradient (TanhRule's adjoint: `(1 - tanh(y)²) * upstream`):
 *   - ∂L/∂tanh_y = ones(2, 4)
 *   - ∂L/∂y = (1 - tanh_y²) * ones = `[[s(1.5), s(1.5), s(3.0), s(3.0)] × 2]`
 *     where `s(z) = 1 - tanh(z)²`. (Same value for ±z since tanh is odd, square is even.)
 *   - ∂L/∂X = ∂L/∂y · W^T (2×3)
 *     - row j of ∂L/∂X = sum_k ∂L/∂y[j,k] * W[*,k]^T row entry
 *     - W^T = `[[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]` (4×3)
 *     - ∂L/∂X[0,c] = s(1.5)*0.5 + s(1.5)*-0.5 + s(3.0)*1.0 + s(3.0)*-1.0 = 0
 *   - ∂L/∂X = zeros(2,3) (the symmetric (+, -, +, -) layout of W cancels exactly).
 *   - ∂L/∂W = X^T · ∂L/∂y. X^T = (3,2) of ones.
 *     - row r of ∂L/∂W = sum_j ones * ∂L/∂y[j,*] = 2 * `[s(1.5), s(1.5), s(3.0), s(3.0)]`
 *     - ∂L/∂W = `[[2*s(1.5), 2*s(1.5), 2*s(3.0), 2*s(3.0)] × 3]` (3×4)
 *
 * The dX = zero outcome is a clean numerical test of the (1-tanh²) chain;
 * dW exercises both the smaller derivative s(3.0) ≈ 0.0099 and the bigger
 * s(1.5) ≈ 0.181.
 */
class Rank2TanhGradientTest {

    @Test
    fun `grad of tanh of A matmul B sum matches analytic`() {
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
            import io.tlaloc.core.ops.tanh
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Lit<Int>>, F32>, w: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32> ->
                    (x matmul w).tanh().sum().toFloat()
                }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, FloatArray(6) { 1.0f })
                val W = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(
                    3, 4,
                    floatArrayOf(0.5f, -0.5f, 1.0f, -1.0f, 0.5f, -0.5f, 1.0f, -1.0f, 0.5f, -0.5f, 1.0f, -1.0f),
                )
                val (dX, dW) = g(X, W)
                val flatX = dX.hostF32()
                val flatW = dW.hostF32()
                println("${'$'}{flatX.size} ${'$'}{dX.dims[0]} ${'$'}{dX.dims[1]}")
                for (v in flatX) print("${'$'}v ")
                println()
                println("${'$'}{flatW.size} ${'$'}{dW.dims[0]} ${'$'}{dW.dims[1]}")
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
            "synthesis fell back; expected end-to-end tanh(matmul) gradient to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")
        val headerX = lines[0].trim().split(" ").map { it.toInt() }
        assertEquals(listOf(6, 2, 3), headerX, "dX should be 2x3")
        val valuesX = lines[1].trim().split(" ").map { it.toFloat() }
        val headerW = lines[2].trim().split(" ").map { it.toInt() }
        assertEquals(listOf(12, 3, 4), headerW, "dW should be 3x4")
        val valuesW = lines[3].trim().split(" ").map { it.toFloat() }

        // s(z) = 1 - tanh(z)^2.
        val s15 = 1.0 - tanh(1.5).let { it * it }
        val s30 = 1.0 - tanh(3.0).let { it * it }
        val expectedX = FloatArray(6) { 0.0f }  // dX cancels exactly due to W's symmetric ± pattern.
        val expectedW = FloatArray(12) { idx ->
            val col = idx % 4
            val s = if (col < 2) s15 else s30  // cols 0,1 use s(1.5); cols 2,3 use s(3.0)
            (2.0 * s).toFloat()
        }
        for ((i, v) in valuesX.withIndex()) {
            assertTrue(abs(v) < 1e-3f, "dX slot $i = $v expected ~0 (tanh symmetry should cancel)")
        }
        for ((i, v) in valuesW.withIndex()) {
            assertTrue(abs(v + 1.0f) > 1e-6f, "dW slot $i = $v matches broken-stub sentinel")
            assertTrue(abs(v - expectedW[i]) < 1e-3f, "dW slot $i = $v expected ${expectedW[i]}")
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
        val tempDir = Files.createTempDirectory("tlaloc-tanh-test").toFile()
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
