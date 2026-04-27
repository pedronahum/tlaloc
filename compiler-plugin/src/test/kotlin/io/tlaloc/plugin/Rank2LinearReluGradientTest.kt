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
 * §0.4.198 — Phase 3 first slice. First end-to-end gradient through a tiny
 * 1-layer NN forward (`relu(X · W).sum()`) with rectangular MATMUL + tensor
 * RELU. Closes the smallest gradient surface that combines two of the new
 * Phase 0c-rectangular pieces (rectangular matmul, BROADCAST template via
 * runtime dims) with the new tensor `STEP` from §0.4.198.
 *
 * Primal: `grad { (X, W) -> (X matmul W).relu().sum().toFloat() }` with
 * `X: Rank2<Sym, Lit<Int>>` and `W: Rank2<Lit<Int>, Lit<Long>>` — three
 * distinct ShapeAtoms. Concrete shape `(2, 3, 4)` at runtime.
 *
 * For X = all-ones (2×3) and W = `[[1,-1,1,-1],[1,-1,1,-1],[1,-1,1,-1]]` (3×4):
 *   - Y = X · W = `[[3,-3,3,-3],[3,-3,3,-3]]` (2×4)
 *   - step(Y) = `[[1,0,1,0],[1,0,1,0]]`  (= ∂L/∂Y after ones-broadcast)
 *   - ∂L/∂X = step(Y) · W^T = `[[2,2,2],[2,2,2]]` (2×3, since each row hits
 *     two W-rows where the relu mask is 1, both contributing 1+1=2)
 *   - ∂L/∂W = X^T · step(Y) = `[[2,0,2,0],[2,0,2,0],[2,0,2,0]]` (3×4)
 */
class Rank2LinearReluGradientTest {

    @Test
    fun `grad of relu of A matmul B sum matches analytic`() {
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
                val g = grad { x: DTensor<Rank2<Sym, Lit<Int>>, F32>, w: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32> ->
                    (x matmul w).relu().sum().toFloat()
                }
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, FloatArray(6) { 1.0f })
                val W = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(
                    3, 4,
                    floatArrayOf(1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f),
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
            "synthesis fell back; expected end-to-end relu(matmul) gradient to lower. " +
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

        val expectedX = floatArrayOf(2f, 2f, 2f, 2f, 2f, 2f)
        val expectedW = floatArrayOf(2f, 0f, 2f, 0f, 2f, 0f, 2f, 0f, 2f, 0f, 2f, 0f)
        for ((i, v) in valuesX.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dX slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(
                abs(v - expectedX[i]) < 1e-3f,
                "dX slot $i = $v expected ${expectedX[i]}",
            )
        }
        for ((i, v) in valuesW.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dW slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(
                abs(v - expectedW[i]) < 1e-3f,
                "dW slot $i = $v expected ${expectedW[i]}",
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
        val tempDir = Files.createTempDirectory("tlaloc-rank2-relu-test").toFile()
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
