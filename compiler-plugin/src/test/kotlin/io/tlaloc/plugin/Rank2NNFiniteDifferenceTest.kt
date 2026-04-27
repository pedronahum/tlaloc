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
 * §0.4.201 — CartPole Phase 3 fourth slice. First FD-validated CartPole-style
 * gradient test: a 1-hidden-layer NN forward `((X · W1).tanh() · W2).sum()`
 * with rectangular weights, tanh activation, and central-difference
 * finite-difference validation of the analytic gradient.
 *
 * The test computes both the analytic gradient (via `grad { ... }`) and a
 * finite-difference approximation (perturbing each parameter element by
 * `±EPS = 1e-3` and computing `(f(+) - f(-)) / (2·EPS)`), then asserts the
 * two agree within 5e-2. The relatively loose tolerance accommodates f32's
 * limited precision in the central-difference truncation error.
 *
 * This is the inflection point from "synthesis primitives wired up" to
 * "real CartPole-style verification methodology". Subsequent CartPole port
 * slices can reuse the FD-comparison utility.
 *
 * Shapes (4 distinct ShapeAtoms):
 *   X: Rank2<Sym, Lit<Int>>      (2, 3)
 *   W1: Rank2<Lit<Int>, Lit<Long>>   (3, 4)
 *   W2: Rank2<Lit<Long>, Lit<Short>> (4, 2)
 */
class Rank2NNFiniteDifferenceTest {

    @Test
    fun `1-hidden-layer NN gradient agrees with finite-differencing`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.matmul
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.tanh
            import io.tlaloc.core.ops.toFloat

            // Forward function used for finite-difference perturbations. Inlined
            // copy of the grad lambda's body. We re-create the DTensors at each
            // call (no in-place mutation) to keep semantics explicit.
            fun forward(
                x: DTensor<Rank2<Sym, Lit<Int>>, F32>,
                w1: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                w2: DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
            ): Float = ((x matmul w1).tanh() matmul w2).sum().toFloat()

            fun perturb(
                base: DTensor<*, F32>,
                idx: Int,
                delta: Float,
            ): FloatArray {
                val src = base.hostF32()
                val out = FloatArray(src.size) { src[it] }
                out[idx] = out[idx] + delta
                return out
            }

            fun mkX(values: FloatArray, dims: IntArray) =
                Tensors.f32Matrix<Sym, Lit<Int>>(dims[0], dims[1], values)
            fun mkW1(values: FloatArray, dims: IntArray) =
                Tensors.f32Matrix<Lit<Int>, Lit<Long>>(dims[0], dims[1], values)
            fun mkW2(values: FloatArray, dims: IntArray) =
                Tensors.f32Matrix<Lit<Long>, Lit<Short>>(dims[0], dims[1], values)

            fun finiteDiff(
                xs: FloatArray, w1s: FloatArray, w2s: FloatArray,
                xDims: IntArray, w1Dims: IntArray, w2Dims: IntArray,
            ): Triple<FloatArray, FloatArray, FloatArray> {
                val EPS = 1e-3f
                val dX = FloatArray(xs.size)
                val dW1 = FloatArray(w1s.size)
                val dW2 = FloatArray(w2s.size)

                for (i in xs.indices) {
                    val plus = perturb(mkX(xs, xDims), i, EPS)
                    val minus = perturb(mkX(xs, xDims), i, -EPS)
                    val fp = forward(mkX(plus, xDims), mkW1(w1s, w1Dims), mkW2(w2s, w2Dims))
                    val fm = forward(mkX(minus, xDims), mkW1(w1s, w1Dims), mkW2(w2s, w2Dims))
                    dX[i] = (fp - fm) / (2f * EPS)
                }
                for (i in w1s.indices) {
                    val plus = perturb(mkW1(w1s, w1Dims), i, EPS)
                    val minus = perturb(mkW1(w1s, w1Dims), i, -EPS)
                    val fp = forward(mkX(xs, xDims), mkW1(plus, w1Dims), mkW2(w2s, w2Dims))
                    val fm = forward(mkX(xs, xDims), mkW1(minus, w1Dims), mkW2(w2s, w2Dims))
                    dW1[i] = (fp - fm) / (2f * EPS)
                }
                for (i in w2s.indices) {
                    val plus = perturb(mkW2(w2s, w2Dims), i, EPS)
                    val minus = perturb(mkW2(w2s, w2Dims), i, -EPS)
                    val fp = forward(mkX(xs, xDims), mkW1(w1s, w1Dims), mkW2(plus, w2Dims))
                    val fm = forward(mkX(xs, xDims), mkW1(w1s, w1Dims), mkW2(minus, w2Dims))
                    dW2[i] = (fp - fm) / (2f * EPS)
                }
                return Triple(dX, dW1, dW2)
            }

            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Lit<Int>>, F32>,
                                w1: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                                w2: DTensor<Rank2<Lit<Long>, Lit<Short>>, F32> ->
                    ((x matmul w1).tanh() matmul w2).sum().toFloat()
                }

                // Use NON-symmetric values so cancellations don't hide gradient bugs.
                val xs = floatArrayOf(0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f)
                val w1s = floatArrayOf(
                    0.1f, -0.1f, 0.2f, -0.2f,
                    0.3f, -0.3f, 0.4f, -0.4f,
                    0.5f, -0.5f, 0.6f, -0.6f,
                )
                val w2s = floatArrayOf(
                    0.7f, -0.7f, 0.8f, -0.8f,
                    0.9f, -0.9f, 1.0f, -1.0f,
                )
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, xs)
                val W1 = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(3, 4, w1s)
                val W2 = Tensors.f32Matrix<Lit<Long>, Lit<Short>>(4, 2, w2s)

                val (dXAna, dW1Ana, dW2Ana) = g(X, W1, W2)
                val (dXFd, dW1Fd, dW2Fd) = finiteDiff(
                    xs, w1s, w2s, intArrayOf(2, 3), intArrayOf(3, 4), intArrayOf(4, 2),
                )

                val flatXAna = dXAna.hostF32()
                val flatW1Ana = dW1Ana.hostF32()
                val flatW2Ana = dW2Ana.hostF32()
                println("ANA_X ${'$'}{flatXAna.size}")
                for (v in flatXAna) print("${'$'}v ")
                println()
                println("FD_X ${'$'}{dXFd.size}")
                for (v in dXFd) print("${'$'}v ")
                println()
                println("ANA_W1 ${'$'}{flatW1Ana.size}")
                for (v in flatW1Ana) print("${'$'}v ")
                println()
                println("FD_W1 ${'$'}{dW1Fd.size}")
                for (v in dW1Fd) print("${'$'}v ")
                println()
                println("ANA_W2 ${'$'}{flatW2Ana.size}")
                for (v in flatW2Ana) print("${'$'}v ")
                println()
                println("FD_W2 ${'$'}{dW2Fd.size}")
                for (v in dW2Fd) print("${'$'}v ")
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
            "synthesis fell back; expected end-to-end FD-validated NN gradient to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(12, lines.size, "expected 12 stdout lines (6 ANA + 6 FD interleaved), got: ${result.stdout}")

        fun parseFloatLine(line: String): FloatArray =
            line.trim().split(" ").map { it.toFloat() }.toFloatArray()

        val anaX = parseFloatLine(lines[1])
        val fdX = parseFloatLine(lines[3])
        val anaW1 = parseFloatLine(lines[5])
        val fdW1 = parseFloatLine(lines[7])
        val anaW2 = parseFloatLine(lines[9])
        val fdW2 = parseFloatLine(lines[11])

        // Tolerance 5e-2 — central-difference + f32 + tanh's nonlinearity gives
        // truncation error around 1e-3..1e-2 for this regime; picking 5e-2 as
        // a generous tolerance to make the test robust without losing
        // diagnostic value.
        val TOL = 5e-2f

        fun assertCloseAll(name: String, ana: FloatArray, fd: FloatArray) {
            assertEquals(ana.size, fd.size, "$name analytic vs FD size mismatch")
            for (i in ana.indices) {
                val diff = abs(ana[i] - fd[i])
                assertTrue(
                    diff < TOL,
                    "$name slot $i: analytic=${ana[i]} fd=${fd[i]} diff=$diff > tol=$TOL",
                )
            }
        }
        assertCloseAll("dX", anaX, fdX)
        assertCloseAll("dW1", anaW1, fdW1)
        assertCloseAll("dW2", anaW2, fdW2)
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-fd-test").toFile()
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
