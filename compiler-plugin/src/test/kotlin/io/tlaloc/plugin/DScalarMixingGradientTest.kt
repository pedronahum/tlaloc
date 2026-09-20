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
 * §0.4.397 — Phase A5c-3(iv): the differentiable-scalar-PARAM half of
 * `DScalar × DTensor` mixing (DiffKT's `timesScalar`, its one scalar-mixing
 * `Operations` primitive).
 *
 * Test 1 pins the spelling that WORKS end-to-end: a `Float` scalar param.
 * `a * s` resolves the A5a `DTensor.times(Float)` overload, the mixed-rank FIR
 * arm splats `s` through the templated BROADCAST, and the scalar side being a
 * PARAM is the strongest form of the A5a "computed scalar" case — its gradient
 * `ds = Σa` is BroadcastRule's full-reduce adjoint, returned as the pair's
 * plain-Float second element.
 *
 * Test 2 — §0.4.414 flipped the §0.4.397 pin: the `FloatScalar`-typed spelling
 * now synthesises end-to-end. The `:core` `DTensor.times(DScalar)` overload
 * lowers to the IDENTICAL dxir the Float spelling gets (`%1: f32`, splat, mul,
 * sum); synthesis materialises the param AS `FloatScalar` (so the function
 * type matches the call site), unwraps it once through `.toFloat()` at body
 * start, and boxes the rank-0 gradient back via the value-class constructor —
 * no tape fallback, gradients matching the Float-param twin exactly.
 *
 * Test 3 certifies the same surface through `valueAndGrad2`: the boxed
 * gradient rides the Triple's third slot while the plain-Float value and the
 * tensor gradient stay untouched.
 */
class DScalarMixingGradientTest {

    @Test
    fun `grad through DTensor times Float scalar param`() {
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
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, s: Float ->
                    (a * s).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                val (da, ds) = g(A, 2.5f)
                println("da")
                for (v in da.hostF32()) print("" + v + " ")
                println()
                println("ds")
                println(ds)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_FLOAT, src)
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
            "synthesis fell back; expected the Float-scalar-param gradient to lower. Warnings:\n${
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }
            }",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")
        val da = lines[1].trim().split(" ").map { it.toFloat() }
        val ds = lines[3].trim().toFloat()

        // loss = Σ (a ⊙ s) → da_i = s = 2.5; ds = Σa = 10.
        assertEquals(4, da.size, "da size")
        assertTrue(
            da.any { it != -1.0f } || ds != -1.0f,
            "stub sentinel returned — rewrite never fired. Messages:\n" +
                result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
        )
        for (i in da.indices) {
            assertTrue(abs(da[i] - 2.5f) < 1e-5f, "da[$i] = ${da[i]}, want 2.5. Stdout:\n${result.stdout}")
        }
        assertTrue(abs(ds - 10f) < 1e-5f, "ds = $ds, want 10 (= Σa). Stdout:\n${result.stdout}")
    }

    @Test
    fun `grad through DTensor times FloatScalar param synthesises with boxing`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.FloatScalar
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, s: FloatScalar ->
                    (a * s).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                val (da, ds) = g(A, FloatScalar(2.5f))
                println("da")
                for (v in da.hostF32()) print("" + v + " ")
                println()
                println("ds")
                println(ds.v)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_FLOATSCALAR, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        // §0.4.414 — no tape fallback: the boxed-scalar param synthesises.
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the FloatScalar-param gradient to lower. Warnings:\n${
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }
            }",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines, got: ${result.stdout}")
        val da = lines[1].trim().split(" ").map { it.toFloat() }
        val ds = lines[3].trim().toFloat()

        // Identical analytics to the Float-param twin: da_i = s = 2.5; ds = Σa = 10,
        // boxed back as FloatScalar(10) whose .v the harness prints.
        assertEquals(4, da.size, "da size")
        assertTrue(
            da.any { it != -1.0f } || ds != -1.0f,
            "stub sentinel returned — rewrite never fired. Messages:\n" +
                result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
        )
        for (i in da.indices) {
            assertTrue(abs(da[i] - 2.5f) < 1e-5f, "da[$i] = ${da[i]}, want 2.5. Stdout:\n${result.stdout}")
        }
        assertTrue(abs(ds - 10f) < 1e-5f, "ds.v = $ds, want 10 (= Σa). Stdout:\n${result.stdout}")
    }

    @Test
    fun `valueAndGrad2 through DTensor times FloatScalar param boxes the gradient slot`() {
        val src = """
            import io.tlaloc.autograd.valueAndGrad2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.FloatScalar
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = valueAndGrad2 { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, s: FloatScalar ->
                    (a * s).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                val (value, da, ds) = g(A, FloatScalar(2.5f))
                println("value")
                println(value)
                println("da")
                for (v in da.hostF32()) print("" + v + " ")
                println()
                println("ds")
                println(ds.v)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_VALUEANDGRAD2, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the valueAndGrad2 FloatScalar surface to lower. Warnings:\n${
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }
            }",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(6, lines.size, "expected 6 stdout lines, got: ${result.stdout}")
        val value = lines[1].trim().toFloat()
        val da = lines[3].trim().split(" ").map { it.toFloat() }
        val ds = lines[5].trim().toFloat()

        // value = Σ (a ⊙ s) = 2.5 · 10 = 25 (the plain-Float first slot);
        // da_i = 2.5; ds = Σa = 10 boxed as FloatScalar in the third slot.
        assertTrue(abs(value - 25f) < 1e-5f, "value = $value, want 25. Stdout:\n${result.stdout}")
        assertEquals(4, da.size, "da size")
        for (i in da.indices) {
            assertTrue(abs(da[i] - 2.5f) < 1e-5f, "da[$i] = ${da[i]}, want 2.5. Stdout:\n${result.stdout}")
        }
        assertTrue(abs(ds - 10f) < 1e-5f, "ds.v = $ds, want 10 (= Σa). Stdout:\n${result.stdout}")
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-dscalar-mixing-test").toFile()
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
        private val AUTOGRAD_STUB_FLOAT = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>, Float) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, Float) ->
                        Pair<DTensor<Rank2<Sym, Lit<Int>>, F32>, Float> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                    -1.0f,
                ) }
        """.trimIndent()

        private val AUTOGRAD_STUB_FLOATSCALAR = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.FloatScalar
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>, FloatScalar) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, FloatScalar) ->
                        Pair<DTensor<Rank2<Sym, Lit<Int>>, F32>, FloatScalar> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                    FloatScalar(-1.0f),
                ) }
        """.trimIndent()

        private val AUTOGRAD_STUB_VALUEANDGRAD2 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.FloatScalar
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun valueAndGrad2(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>, FloatScalar) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, FloatScalar) ->
                        Triple<Float, DTensor<Rank2<Sym, Lit<Int>>, F32>, FloatScalar> =
                { _, _ -> Triple(
                    -1.0f,
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                    FloatScalar(-1.0f),
                ) }
        """.trimIndent()
    }
}
