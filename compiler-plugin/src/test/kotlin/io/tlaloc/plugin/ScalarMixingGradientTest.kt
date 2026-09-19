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
 * Phase A5 E2E: scalar × tensor mixing inside `grad {}` — the everyday
 * `a * 2.0f` / `3.0f - a` / `b / 2.0f` spellings DiffKT accepts everywhere
 * (`timesScalar` + `broadcast(S1, S2)` on every binary op) and Tlaloc rejects
 * today: the tensor `times(Float)` overload shares the `io.tlaloc.core.ops.times`
 * FQN with the tensor⊙tensor one, so BINARY_OP_MAP lowers `MUL(a, 2.0f)` with a
 * rank-2 lhs and a rank-0 rhs — a shape mismatch.
 *
 * The fix lowers the mixed-rank case to the IR's existing splat: the rank-0
 * operand becomes `BROADCAST(const, tensorOperand.type)` with an EMPTY
 * `broadcast_dimensions` (the §0.4.359 scalar-seed polymorphism the §0.4.371
 * generalization preserved), so both MUL operands share a type and the whole
 * adjoint falls out of rules that already ship — MulRule for the product,
 * BroadcastRule's splat adjoint (a full reduce, emitted as the runtime-extent
 * `SUM_TO` of §0.4.373) for a differentiable scalar side. No new op kind, no
 * new VjpRule, and nothing reads a -1 sentinel dim.
 *
 *   test 1  loss = Σ (a ⊙ 2.0f) ⊙ b               a, b are [2,2]
 *             da = 2b            db = 2a
 *   test 2  loss = Σ (a + 1.0f) ⊙ b
 *             da = b             db = a + 1
 *   test 3  loss = Σ (3.0f − a) ⊙ (b / 2.0f)
 *             da = −b/2          db = (3 − a)/2
 *   test 4  loss = Σ a ⊙ (Σb)     the COMPUTED scalar side (`b.sum().toFloat()`),
 *             da = Σb            db = Σa   differentiable: its splat adjoint is the
 *             full reduce BroadcastRule emits as SUM_TO, re-broadcast by SumRule
 *
 * Test 3 also pins the tensor-NEG synthesis fix: SubRule's `NEG(upstream)` and
 * DivRule's `NEG(mul)` had no tensor arm, so `findUnaryOp` resolved
 * `kotlin.Float.unaryMinus` for a DTensor-typed NEG and the gradient threw a
 * ClassCastException at run time.
 */
class ScalarMixingGradientTest {

    @Test
    fun `grad through tensor times Float literal`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    ((a * 2.0f) * b).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        assertGradient(
            src,
            want = mapOf(
                // da = 2b
                "da" to listOf(20f, 40f, 60f, 80f),
                // db = 2a
                "db" to listOf(2f, 4f, 6f, 8f),
            ),
        )
    }

    @Test
    fun `grad through tensor plus Float literal`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    (((a + 1.0f) * b).sum().toFloat())
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        assertGradient(
            src,
            want = mapOf(
                // da = b
                "da" to listOf(10f, 20f, 30f, 40f),
                // db = a + 1
                "db" to listOf(2f, 3f, 4f, 5f),
            ),
        )
    }

    @Test
    fun `grad through scalar-on-the-left minus and tensor div literal`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.div
            import io.tlaloc.core.ops.minus
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    ((3.0f - a) * (b / 2.0f)).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        assertGradient(
            src,
            want = mapOf(
                // da = -b/2
                "da" to listOf(-5f, -10f, -15f, -20f),
                // db = (3 - a)/2
                "db" to listOf(1f, 0.5f, 0f, -0.5f),
            ),
        )
    }

    @Test
    fun `grad through a computed differentiable scalar factor`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    (a * b.sum().toFloat()).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        // s = Σb = 100, Σa = 10; loss = Σ(a·s) = s·Σa
        //   da_i = s = 100        db_j = Σa = 10  (the scalar side's adjoint is the
        //   full reduce BroadcastRule emits as SUM_TO, then re-broadcast by SumRule)
        assertGradient(
            src,
            want = mapOf(
                "da" to listOf(100f, 100f, 100f, 100f),
                "db" to listOf(10f, 10f, 10f, 10f),
            ),
        )
    }

    private fun assertGradient(src: String, want: Map<String, List<Float>>) {
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(
            0,
            result.exitCode,
            "compile/run failed:\n" + result.messages
                .filter {
                    it.severity == CompilerMessageSeverity.ERROR || it.severity == CompilerMessageSeverity.WARNING
                }
                .joinToString("\n") { "${it.severity}: ${it.message}" },
        )

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the scalar-mixing gradient to lower. Warnings:\n${
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }
            }",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name'")
            assertEquals(expect.size, values.size, "$name size")
            assertTrue(
                values.any { it != -1.0f },
                "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                    result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
            )
            for (j in expect.indices) {
                assertTrue(
                    abs(values[j] - expect[j]) < 1e-5f,
                    "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                )
            }
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
        val tempDir = Files.createTempDirectory("tlaloc-scalar-mixing-test").toFile()
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
                val cause = t.cause ?: t
                RunResult(
                    2,
                    collected + CompileMessage(
                        CompilerMessageSeverity.ERROR,
                        "RUN FAILURE: $cause\n" +
                            cause.stackTrace.take(12).joinToString("\n") { "    at $it" },
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
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) ->
                        Pair<DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                ) }
        """.trimIndent()
    }
}
