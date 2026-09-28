package io.tlaloc.plugin

import io.tlaloc.core.LinalgKernels
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
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The linear-algebra surface of `:core/ops/Linalg.kt` under `grad {}`, `grad2 {}`
 * and `jvp {}`, through the real K2 plugin. Each gradient must synthesize (no
 * "kept original call" fallback) and match central finite differences of the
 * same function computed in Double on [LinalgKernels].
 *
 * Tolerance: the synthesized gradient runs the F32 host twins, each of which
 * computes in Double and rounds once, so every op contributes about 6e-8 relative;
 * with condition numbers below 30 and under ten ops in any gradient body the error
 * stays below 1e-5 of the largest entry. The Double finite differences (h = 1e-5)
 * are good to about 1e-10. The bound, 1e-4 of the largest entry, leaves a factor
 * of ten; a wrong rule moves entries by O(1).
 */
class LinalgGradientTest {

    private val n = 4
    private val spd = doubleArrayOf(
        6.2, 1.1, -0.7, 0.4,
        1.3, 5.1, 0.9, -1.2,
        -0.5, 0.9, 4.8, 0.6,
        0.4, -1.0, 0.6, 5.5,
    )
    private val tri = doubleArrayOf(
        2.0, 9.0, -7.0, 5.0,
        0.6, 1.7, 8.0, -3.0,
        -0.4, 0.3, 2.4, 4.0,
        0.9, -0.8, 0.5, 1.9,
    )
    private val rhs = doubleArrayOf(0.7, -1.2, 0.4, 2.1, -0.3, 0.8, 1.5, -0.6)
    private val dir = DoubleArray(16) { ((it * 7) % 11 - 5) / 5.0 }

    private fun lit(a: DoubleArray) = a.joinToString(", ") { "${it.toFloat()}f" }

    private fun cubeSum(x: DoubleArray) = x.sumOf { it * it * it }

    private fun fdGrad(x: DoubleArray, loss: (DoubleArray) -> Double): DoubleArray {
        val h = 1e-5
        return DoubleArray(x.size) { i ->
            (loss(x.copyOf().also { it[i] += h }) - loss(x.copyOf().also { it[i] -= h })) / (2 * h)
        }
    }

    private fun assertClose(want: DoubleArray, got: List<Float>, what: String, relTol: Double = 1e-4) {
        assertEquals(want.size, got.size, "$what size (got $got)")
        val scale = max(1e-12, want.maxOf { abs(it) })
        for (i in want.indices) {
            assertTrue(abs(want[i] - got[i]) <= relTol * scale, "$what[$i] = ${got[i]}, want ${want[i]}")
        }
    }

    @Test
    fun `cholesky triangularSolve tril and triu differentiate through the plugin`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.jvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.cholesky
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            import io.tlaloc.core.ops.triangularSolve
            import io.tlaloc.core.ops.tril
            import io.tlaloc.core.ops.triu
            fun show(name: String, t: DTensor<*, F32>) = println(name + " " + t.hostF32().joinToString(","))
            fun main() {
                val a = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(spd)}))
                val t = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(tri)}))
                val b = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(${lit(rhs)}))
                val v = Tensors.f32Matrix<Sym, Sym>(4, 4, floatArrayOf(${lit(dir)}))

                val gChol = grad { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val l = x.cholesky()
                    (l * l * l).sum().toFloat()
                }
                show("chol", gChol(a))

                val jChol = jvp { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val l = x.cholesky()
                    (l * l * l).sum().toFloat()
                }
                println("jchol " + jChol(a, v))

                val gSolve = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.triangularSolve(y, true)
                    (s * s * s).sum().toFloat()
                }
                val (dA, dB) = gSolve(t, b)
                show("solveA", dA)
                show("solveB", dB)

                val gSolveT = grad2 { x: DTensor<Rank2<Sym, Sym>, F32>, y: DTensor<Rank2<Sym, Sym>, F32> ->
                    val s = x.triangularSolve(y, false, true, false)
                    (s * s * s).sum().toFloat()
                }
                show("solveUT", gSolveT(t.triu(), b).first)

                val gTri = grad { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val m = x.tril() + x.triu()
                    (m * m * m).sum().toFloat()
                }
                show("tri", gTri(t))
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0, result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}",
        )
        val fellBack = result.messages.filter { "kept original call" in it.message }
        assertTrue(fellBack.isEmpty(), "synthesis fell back:\n${fellBack.joinToString("\n--\n") { it.message }}")
        val rows = result.stdout.trim().lines().associate { line ->
            val (k, v) = line.split(" ", limit = 2)
            k to v.split(",").map { it.toFloat() }
        }

        val choleskyLoss = { x: DoubleArray -> cubeSum(LinalgKernels.cholesky(x, n)) }
        val gChol = fdGrad(spd, choleskyLoss)
        assertClose(gChol, rows.getValue("chol"), "d cholesky")
        assertClose(doubleArrayOf(gChol.indices.sumOf { gChol[it] * dir[it] }), rows.getValue("jchol"), "jvp cholesky")

        fun solve(a: DoubleArray, b: DoubleArray, lower: Boolean, transposeA: Boolean) =
            cubeSum(LinalgKernels.triangularSolve(a, b, n, 2, lower, transposeA, false))
        assertClose(fdGrad(tri) { solve(it, rhs, true, false) }, rows.getValue("solveA"), "d solve / dA")
        assertClose(fdGrad(rhs) { solve(tri, it, true, false) }, rows.getValue("solveB"), "d solve / dB")
        val triU = LinalgKernels.triangle(tri, n, n, 0.0, 1.0, 1.0)
        assertClose(fdGrad(triU) { solve(it, rhs, false, true) }, rows.getValue("solveUT"), "d solve(upper, transposed) / dA")

        val triLoss = { x: DoubleArray ->
            val m = LinalgKernels.triangle(x, n, n, 1.0, 2.0, 1.0)
            cubeSum(m)
        }
        assertClose(fdGrad(tri, triLoss), rows.getValue("tri"), "d (tril + triu)")
    }

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)

    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private fun compileAndRun(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-linalg-test").toFile()
        try {
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
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                RunResult(
                    2,
                    collected + CompileMessage(CompilerMessageSeverity.ERROR, "RUN FAILURE: ${t.cause ?: t}"),
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
}
