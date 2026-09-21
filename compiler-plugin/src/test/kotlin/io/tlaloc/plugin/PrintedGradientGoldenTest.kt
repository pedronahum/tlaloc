package io.tlaloc.plugin

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.embedding
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.relu
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.render.toKotlinSource
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
 * §0.4.449 — the readable-reverse GOLDEN oracle, the beyond-Tangent claim made
 * executable: a gradient function produced by the F1 capture route
 * (`captureN` → `Tape.toDxirFunction` → `DxirReverseTransform`) is printed as
 * Kotlin source by `toKotlinSource`, COMPILED with the in-process
 * [K2JVMCompiler] (the §0.4.58 harness pattern, `:core` on the classpath —
 * no plugin: the printed source is plain user-grade Kotlin), RUN via
 * reflection, and its outputs pinned BIT-IDENTICAL (`Float.toRawBits`)
 * against [DxirInterpreter] evaluating the SAME `DxirFunction` on the SAME
 * inputs. Tangent prints readable derivative source; this proves the printed
 * derivative is also a working, certified program.
 *
 *  g1 (Dense-like): loss = mean(relu(x·W + b)²) — exercises MATMUL, the
 *      bias-row BROADCAST, RELU, MUL, MEAN, and on the reverse side the
 *      splat consts, RESHAPE-to-keepdims, stretch-BROADCAST, STEP masks,
 *      TRANSPOSE/MATMUL adjoints and the SUM un-broadcast of the bias.
 *  g2 (embedding): loss = sum(embedding(table, idx)²) — exercises the I32
 *      param rendering, EMBEDDING/EMBEDDING_GRAD and the ZEROS_LIKE
 *      structural zero (printed as `intZerosLike`).
 */
class PrintedGradientGoldenTest {

    @Test
    fun `printed dense-like gradient compiles, runs, and matches the interpreter bit-for-bit`() {
        val x = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(0.5f, -1.25f, 2.0f, 0.75f, -0.25f, 1.5f))
        val w = Tensors.f32Matrix<Sym, Sym>(3, 2, floatArrayOf(0.3f, -0.7f, 1.1f, 0.2f, -0.4f, 0.9f))
        val b = Tensors.f32Vector<Sym>(floatArrayOf(0.1f, -0.2f))

        @Suppress("UNCHECKED_CAST")
        val primal = captureN(listOf(x, w, b), name = "denseGolden") { leaves ->
            val xt = leaves[0] as Tracer<Rank2<Sym, Sym>>
            val wt = leaves[1] as Tracer<Rank2<Sym, Sym>>
            val bt = leaves[2] as Tracer<Rank1<Sym>>
            val y = (xt matmul wt) + bt
            val a = y.relu()
            (a * a).mean() as Tracer<Shape>
        }
        val gradient = DxirReverseTransform.apply(primal, includeForward = true)
        val source = gradient.toKotlinSource()

        // Readability pins: the printed gradient reads as host-twin user code.
        assertTrue("matmul" in source, "printed source must spell matmul; got:\n$source")
        assertTrue(".relu()" in source || ".step()" in source, "printed source must spell the relu/step mask; got:\n$source")
        assertTrue("fun denseGolden" in source, "printed source must be a named function; got:\n$source")

        val inputs = listOf(
            floatArrayOf(0.5f, -1.25f, 2.0f, 0.75f, -0.25f, 1.5f),
            floatArrayOf(0.3f, -0.7f, 1.1f, 0.2f, -0.4f, 0.9f),
            floatArrayOf(0.1f, -0.2f),
        )
        val expected = DxirInterpreter.evalFunction(gradient, inputs)
        assertEquals(4, expected.size, "includeForward gradient must return (loss, dx, dw, db)")

        val fnName = printedFunctionName(source)
        val driver = """
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val x = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(0.5f, -1.25f, 2.0f, 0.75f, -0.25f, 1.5f))
                val w = Tensors.f32Matrix<Sym, Sym>(3, 2, floatArrayOf(0.3f, -0.7f, 1.1f, 0.2f, -0.4f, 0.9f))
                val b = Tensors.f32Vector<Sym>(floatArrayOf(0.1f, -0.2f))
                val outs = $fnName(x, w, b)
                for (t in outs) {
                    @Suppress("UNCHECKED_CAST")
                    val f = (t as DTensor<*, F32>).hostF32()
                    println(f.joinToString(" ") { it.toRawBits().toString() })
                }
            }
        """.trimIndent()

        val result = compileAndRun(source, driver)
        assertEquals(0, result.exitCode, "printed source failed to compile/run:\n${result.messages}\n--- source ---\n$source")

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 output lines, got: ${result.stdout}")
        for (i in expected.indices) {
            val got = lines[i].trim().split(" ").map { it.toInt() }
            val want = expected[i].map { it.toRawBits() }
            assertEquals(
                want, got,
                "output $i of the printed gradient must be BIT-IDENTICAL to the interpreter " +
                    "(floats: printed=${got.map { Float.fromBits(it) }} interpreter=${expected[i].toList()})",
            )
        }
    }

    @Test
    fun `printed embedding gradient renders I32 params and the fused scatter, bit-identical`() {
        val table = Tensors.f32Matrix<Sym, Sym>(
            5, 3,
            floatArrayOf(
                0.1f, -0.2f, 0.3f,
                1.0f, 0.5f, -0.5f,
                -1.5f, 2.0f, 0.25f,
                0.75f, -0.75f, 1.25f,
                0.0f, 0.6f, -0.9f,
            ),
        )
        val idx = Tensors.i32Vector<Sym>(intArrayOf(3, 0, 3, 1))

        @Suppress("UNCHECKED_CAST")
        val primal = captureN(listOf(table, idx), name = "embedGolden") { leaves ->
            val e = leaves[0].embedding<Shape>(leaves[1])
            (e * e).sum() as Tracer<Shape>
        }
        val gradient = DxirReverseTransform.apply(primal, includeForward = true)
        val source = gradient.toKotlinSource()

        assertTrue("embeddingGrad" in source, "printed source must spell embeddingGrad; got:\n$source")
        assertTrue("intZerosLike" in source, "the I32 structural zero must print as intZerosLike; got:\n$source")
        assertTrue("DTensor<Rank1<Sym>, I32>" in source, "the index param must be I32-typed; got:\n$source")

        val inputs = listOf(
            floatArrayOf(
                0.1f, -0.2f, 0.3f,
                1.0f, 0.5f, -0.5f,
                -1.5f, 2.0f, 0.25f,
                0.75f, -0.75f, 1.25f,
                0.0f, 0.6f, -0.9f,
            ),
            floatArrayOf(3f, 0f, 3f, 1f), // the interpreter's float-encoded I32 convention
        )
        val expected = DxirInterpreter.evalFunction(gradient, inputs)
        assertEquals(3, expected.size, "includeForward gradient must return (loss, dTable, dIdx)")

        val fnName = printedFunctionName(source)
        val driver = """
            import io.tlaloc.core.*
            import io.tlaloc.core.ops.*
            fun main() {
                val table = Tensors.f32Matrix<Sym, Sym>(5, 3, floatArrayOf(
                    0.1f, -0.2f, 0.3f,
                    1.0f, 0.5f, -0.5f,
                    -1.5f, 2.0f, 0.25f,
                    0.75f, -0.75f, 1.25f,
                    0.0f, 0.6f, -0.9f,
                ))
                val idx = Tensors.i32Vector<Sym>(intArrayOf(3, 0, 3, 1))
                val (loss, dTable, dIdx) = $fnName(table, idx)
                println(loss.hostF32().joinToString(" ") { it.toRawBits().toString() })
                println(dTable.hostF32().joinToString(" ") { it.toRawBits().toString() })
                println(dIdx.hostI32().joinToString(" ") { it.toString() })
            }
        """.trimIndent()

        val result = compileAndRun(source, driver)
        assertEquals(0, result.exitCode, "printed source failed to compile/run:\n${result.messages}\n--- source ---\n$source")

        val lines = result.stdout.trim().lines()
        assertEquals(3, lines.size, "expected 3 output lines, got: ${result.stdout}")
        // F32 slots: raw-bit identity.
        for (i in 0..1) {
            val got = lines[i].trim().split(" ").map { it.toInt() }
            val want = expected[i].map { it.toRawBits() }
            assertEquals(want, got, "output $i must be BIT-IDENTICAL to the interpreter")
        }
        // The I32 structural-zero slot: the interpreter's float-encoded zeros
        // must equal the printed function's IntArray zeros, value for value.
        val gotIdx = lines[2].trim().split(" ").map { it.toInt() }
        val wantIdx = expected[2].map { it.toInt() }
        assertEquals(wantIdx, gotIdx, "the structural-zero index gradient must match")
        assertTrue(gotIdx.all { it == 0 }, "the index gradient is a structural zero")
    }

    // ------------------------------------------------------------ harness

    private fun printedFunctionName(source: String): String {
        val m = Regex("""fun (\w+)\(""").find(source)
            ?: error("printed source carries no function declaration:\n$source")
        return m.groupValues[1]
    }

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    /**
     * The RngGradientTest K2JVMCompiler pattern, minus the plugin: the printed
     * source is plain Kotlin over `:core`, so the ONLY classpath it needs is
     * the test JVM's own (which carries the `:core` jvm jar).
     */
    private fun compileAndRun(printed: String, driver: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-printed-grad").toFile()
        try {
            File(tempDir, "Printed.kt").writeText(printed)
            File(tempDir, "Main.kt").writeText(driver)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
                    collected += CompileMessage(severity, message)
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
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
                val mainCls = loader.loadClass("MainKt")
                mainCls.getMethod("main").invoke(null)
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
