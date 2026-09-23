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
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase A5b — `pow` reaches user code (DiffKT parity: the audit's orphaned
 * `pow(Float/Int/tensor-exponent)` item).
 *
 * POW has been ruled below the surface since Stage B.3 — `PowRule` (both partials,
 * including the I32/I64-exponent CAST), the interpreter arm, `stablehlo.power`
 * emission, the forward-mode tangent, and synthesis's scalar `kotlin.math.pow` arm
 * (§0.4.52, landed for C6's closed-form geometric sum) — but nothing lowered TO it,
 * so no user expression could produce a POW node. Two wiring gaps closed:
 *
 *  - `:core/ops` grows the host `pow` (tensor exponent, Float exponent, Int
 *    exponent); `BINARY_OP_MAP` maps both `io.tlaloc.core.ops.pow` and
 *    `kotlin.math.pow` to `OpKind.POW`, and POW joins `ELEMENTWISE_BINARY_KINDS` so
 *    the Phase A5a mixed-rank arm splats a literal exponent to the base's shape —
 *    the IR then always sees the uniform two-operand POW PowRule expects, with no
 *    new op kind and no sentinel-dim reading.
 *  - synthesis dispatches a TENSOR POW through the same generic tensor-binary path
 *    as ADD/SUB/MUL/DIV (`findTensorBinaryOp("pow")`, whose overload filter picks
 *    the tensor-exponent form); the scalar path is unchanged. `deriveResultIrType`
 *    and the backward IrType solver both gained POW in their elementwise-binary arms.
 *
 * Four tests through the real K2 plugin, each certified to synthesise with no
 * tape fallback:
 *  1. `a.pow(2.0f)` — Float literal exponent:   da = 2a
 *  2. `a.pow(3)`    — Int literal exponent:     da = 3a²
 *  3. `a.pow(b)`    — tensor exponent:          da = b·a^(b−1), db = a^b·ln(a)
 *  4. `x.pow(2.0f)` — scalar `kotlin.math.pow`: dx = 2x
 */
class PowGradientTest {

    @Test
    fun `grad through tensor pow with a Float literal exponent`() {
        // loss = Σ a^2 → da = 2a
        assertTensorGradient(
            stub = ONE_PARAM_STUB,
            body = "a.pow(2.0f).sum().toFloat()",
            extraImports = listOf("io.tlaloc.core.ops.pow"),
            a = floatArrayOf(1f, 2f, 3f, 4f),
            want = mapOf("da" to listOf(2f, 4f, 6f, 8f)),
        )
    }

    @Test
    fun `grad through tensor pow with an Int literal exponent`() {
        // loss = Σ a^3 → da = 3a^2
        assertTensorGradient(
            stub = ONE_PARAM_STUB,
            body = "a.pow(3).sum().toFloat()",
            extraImports = listOf("io.tlaloc.core.ops.pow"),
            a = floatArrayOf(1f, 2f, 3f, 4f),
            want = mapOf("da" to listOf(3f, 12f, 27f, 48f)),
        )
    }

    @Test
    fun `grad through tensor pow with a tensor exponent`() {
        // loss = Σ a^b → da = b·a^(b−1), db = a^b·ln(a)
        val a = floatArrayOf(1f, 2f, 3f, 4f)
        val b = floatArrayOf(2f, 2f, 3f, 3f)
        assertTensorGradient(
            stub = TWO_PARAM_STUB,
            body = "a.pow(b).sum().toFloat()",
            extraImports = listOf("io.tlaloc.core.ops.pow"),
            a = a,
            b = b,
            want = mapOf(
                "da" to a.indices.map { i -> b[i] * a[i].toDouble().pow(b[i] - 1.0).toFloat() },
                "db" to a.indices.map { i ->
                    a[i].toDouble().pow(b[i].toDouble()).toFloat() * ln(a[i].toDouble()).toFloat()
                },
            ),
        )
    }

    @Test
    fun `grad through scalar pow`() {
        // f(x) = x^2 → dx = 2x
        val src = """
            import io.tlaloc.autograd.grad
            import kotlin.math.pow
            fun main() {
                val g = grad { x: Float -> x.pow(2.0f) }
                println(listOf(3.0f, -2.0f, 0.5f).map { g(it) }.joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(SCALAR_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.errorsAndWarnings()}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, parts.size, "expected one gradient per probe point: ${result.stdout}")
        for ((i, x) in listOf(3.0f, -2.0f, 0.5f).withIndex()) {
            assertTrue(
                abs(parts[i] + 1.0f) > 1e-3f,
                "slot $i = ${parts[i]} matches the broken-stub sentinel; the rewrite did not fire",
            )
            assertTrue(
                abs(parts[i] - 2.0f * x) < 1e-4f,
                "d/dx x^2 at x=$x: got ${parts[i]}, want ${2.0f * x}",
            )
        }
    }

    private fun assertTensorGradient(
        stub: String,
        body: String,
        extraImports: List<String>,
        a: FloatArray,
        b: FloatArray? = null,
        want: Map<String, List<Float>>,
    ) {
        val params = if (b == null) {
            "a: DTensor<Rank2<Sym, Lit<Int>>, F32>"
        } else {
            "a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32>"
        }
        val call = if (b == null) "g(A)" else "g(A, B)"
        val binds = if (b == null) {
            ""
        } else {
            "\n                val B = Tensors.f32Matrix<Sym, Lit<Int>>(1, ${b.size}, floatArrayOf(${b.joinToString(", ") { "${it}f" }}))"
        }
        val dumps = want.keys.joinToString("; ") { "dump(\"$it\", $it)" }
        val destructure = if (b == null) "val da = $call" else "val (da, db) = $call"
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
            import io.tlaloc.core.ops.toFloat
            ${extraImports.joinToString("\n            ") { "import $it" }}
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { $params ->
                    $body
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(1, ${a.size}, floatArrayOf(${a.joinToString(", ") { "${it}f" }}))$binds
                $destructure
                $dumps
            }
        """.trimIndent()
        val result = compileAndRun(stub, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.errorsAndWarnings()}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the pow gradient to lower. Warnings:\n${result.errorsAndWarnings()}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(2 * want.size, lines.size, "expected ${2 * want.size} stdout lines: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name'")
            assertEquals(expect.size, values.size, "$name size")
            assertTrue(
                values.any { it != -1.0f } || expect.all { it == -1.0f },
                "$name: stub sentinel returned — rewrite never fired.\n${result.errorsAndWarnings()}",
            )
            for (j in expect.indices) {
                // F32 arithmetic on values up to ~1e2: relative tolerance.
                val tol = 1e-4f * max(1.0f, abs(expect[j]))
                assertTrue(
                    abs(values[j] - expect[j]) < tol,
                    "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                )
            }
        }
    }

    private fun RunResult.errorsAndWarnings(): String = messages
        .filter {
            it.severity == CompilerMessageSeverity.ERROR || it.severity == CompilerMessageSeverity.WARNING
        }
        .joinToString("\n") { "${it.severity}: ${it.message}" }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-pow-gradient-test").toFile()
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
        private const val TENSOR_TYPE = "DTensor<Rank2<Sym, Lit<Int>>, F32>"

        private val ONE_PARAM_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: ($TENSOR_TYPE) -> Float): ($TENSOR_TYPE) -> $TENSOR_TYPE =
                { _ -> DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(1, 4), F32) }
        """.trimIndent()

        private val TWO_PARAM_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: ($TENSOR_TYPE, $TENSOR_TYPE) -> Float): ($TENSOR_TYPE, $TENSOR_TYPE) -> Pair<$TENSOR_TYPE, $TENSOR_TYPE> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(1, 4), F32),
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(1, 4), F32),
                ) }
        """.trimIndent()

        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()
    }
}
