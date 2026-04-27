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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.206 — CartPole Phase 3 eighth slice (FINAL Phase 3 piece). First
 * end-to-end gradient-descent training loop using the K2 plugin's synthesised
 * gradient. Mirrors CartPole's outer training loop structure (`while (loss
 * > threshold) { ... apply gradient updates ... }`) but with a simpler primal
 * (3-layer NN with tanh output, no sign — sign would block all gradient flow).
 *
 * The test:
 *  1. Builds a 3-layer NN gradient via `grad { (X, W1, W2, W3) -> ((X · W1).relu()
 *     · W2).relu() · W3).tanh().sum().toFloat() }` (the same chain shape as
 *     §0.4.205 minus the sign, since sign blocks gradient).
 *  2. Runs N gradient-descent steps: `Wi ← Wi - lr · dWi` using the new
 *     §0.4.206 `DTensor * Float` operator.
 *  3. Verifies loss DECREASES from initial to final iteration (the simplest
 *     "training works" smoke test).
 *
 * **Closes CartPole Phase 3.** Per `docs/CARTPOLE_PORT_PLAN.md`, Phase 3's
 * deliverable was "complete CartPole forward pass including the neural net
 * AND the outer `while (loss > threshold)` training loop". §0.4.197 closed
 * Phase 0c-rectangular; §0.4.198–§0.4.205 shipped the 7 NN-forward slices;
 * §0.4.206 closes the outer-loop slice. CartPole-side surface is now:
 *  - 3 phases CLOSED (per `docs/CARTPOLE_PORT_PLAN.md`'s ship-state table)
 *  - + Phase 0c-rectangular CLOSED (§0.4.197)
 *  - + Phase 3 CLOSED (this firing)
 */
class CartPoleTrainingLoopTest {

    @Test
    fun `gradient-descent training loop on 3-layer NN decreases loss`() {
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
            import io.tlaloc.core.ops.minus
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.tanh
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat

            // Forward function used for the loss check at start / end of training.
            fun forward(
                x: DTensor<Rank2<Sym, Lit<Int>>, F32>,
                w1: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                w2: DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
                w3: DTensor<Rank2<Lit<Short>, Lit<Byte>>, F32>,
            ): Float = (((x matmul w1).relu() matmul w2).relu() matmul w3).tanh().sum().toFloat()

            fun main() {
                val g = grad { x: DTensor<Rank2<Sym, Lit<Int>>, F32>,
                                w1: DTensor<Rank2<Lit<Int>, Lit<Long>>, F32>,
                                w2: DTensor<Rank2<Lit<Long>, Lit<Short>>, F32>,
                                w3: DTensor<Rank2<Lit<Short>, Lit<Byte>>, F32> ->
                    (((x matmul w1).relu() matmul w2).relu() matmul w3).tanh().sum().toFloat()
                }

                // Initial weights with non-trivial values (small but distinguishable).
                val X = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3,
                    floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f))
                var W1 = Tensors.f32Matrix<Lit<Int>, Lit<Long>>(3, 4, FloatArray(12) { 0.3f })
                var W2 = Tensors.f32Matrix<Lit<Long>, Lit<Short>>(4, 5, FloatArray(20) { 0.3f })
                var W3 = Tensors.f32Matrix<Lit<Short>, Lit<Byte>>(5, 2, FloatArray(10) { 0.3f })

                val initialLoss = forward(X, W1, W2, W3)
                println("initial_loss ${'$'}initialLoss")

                // Gradient-descent: Wi ← Wi - lr · dWi (minimise loss → push weights
                // in the negative-gradient direction). Five steps is plenty for
                // a smoke test; the loss should monotonically decrease.
                val lr = 0.5f
                for (step in 0 until 5) {
                    val q = g(X, W1, W2, W3)
                    W1 = W1 - (q.second * lr)
                    W2 = W2 - (q.third * lr)
                    W3 = W3 - (q.fourth * lr)
                    val stepLoss = forward(X, W1, W2, W3)
                    println("step ${'$'}step loss ${'$'}stepLoss")
                }

                val finalLoss = forward(X, W1, W2, W3)
                println("final_loss ${'$'}finalLoss")
                println("decreased ${'$'}{finalLoss < initialLoss}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_4ARG, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected training loop to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // Pull the final-loss < initial-loss check from stdout.
        val out = result.stdout.trim().lines()
        val initialLossLine = out.firstOrNull { it.startsWith("initial_loss ") }
            ?: error("missing initial_loss line in:\n${result.stdout}")
        val finalLossLine = out.firstOrNull { it.startsWith("final_loss ") }
            ?: error("missing final_loss line in:\n${result.stdout}")
        val decreasedLine = out.firstOrNull { it.startsWith("decreased ") }
            ?: error("missing decreased line in:\n${result.stdout}")

        val initialLoss = initialLossLine.substringAfter("initial_loss ").toFloat()
        val finalLoss = finalLossLine.substringAfter("final_loss ").toFloat()
        val decreased = decreasedLine.substringAfter("decreased ").trim() == "true"

        assertTrue(
            decreased,
            "loss did not decrease across 5 GD steps: initial=$initialLoss → final=$finalLoss\n" +
                "stdout:\n${result.stdout}",
        )
        assertTrue(
            finalLoss < initialLoss,
            "final loss ($finalLoss) >= initial loss ($initialLoss)",
        )
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-cartpole-train-test").toFile()
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
