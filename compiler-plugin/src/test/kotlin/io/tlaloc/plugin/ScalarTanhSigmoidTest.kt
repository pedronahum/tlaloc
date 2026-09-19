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
import kotlin.math.exp
import kotlin.math.tanh
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase A5b — the scalar `tanh` / `sigmoid` surface (DiffKT parity: the audit's
 * orphaned-lowerings item).
 *
 * Both ops were fully ruled at the IR level from Stage A/B on — `TanhRule`
 * (`d/dx tanh = 1 − tanh²`), `SigmoidRule` (`σ′ = σ(1−σ)`), interpreter arms,
 * `stablehlo.tanh` / sigmoid-composition emission, forward-mode tangents — and the
 * TENSOR spellings `io.tlaloc.core.ops.tanh` / `.sigmoid` have lowered since
 * §0.4.200. What was missing was the scalar half: `:core/DScalar.kt` declared no
 * scalar host fns at all, so `UNARY_OP_MAP` had nothing to map and
 * `grad { x: Float -> x.tanh() }` never reached the rules.
 *
 * Wiring: the host fns land next to the §0.4.158 exp/log entries (five overloads
 * each: Float, Double, FloatScalar, DoubleScalar, DScalar); `UNARY_OP_MAP` maps
 * `io.tlaloc.core.tanh` → TANH and `io.tlaloc.core.sigmoid` → SIGMOID. Synthesis
 * needed no change for TANH (the scalar path already routes to `kotlin.math.tanh`
 * via [DxirToIrSynthesis.irTanh]) but scalar SIGMOID was explicitly rejected in
 * §0.4.200 — there is no `kotlin.math.sigmoid` — so `irSigmoid` now falls through
 * to `irCoreScalarCall`, which resolves the `io.tlaloc.core` extension whose
 * receiver matches the op's primitive dtype.
 *
 * Three tests, all scalar `grad {}` through the real K2 plugin:
 *  1. `x.tanh()` — d/dx = 1 − tanh²(x)
 *  2. `x.sigmoid()` — d/dx = σ(x)·(1 − σ(x))
 *  3. `2·tanh(x) + sigmoid(x)` — the two rules in one body, with a scalar literal
 *     factor on the tanh term
 */
class ScalarTanhSigmoidTest {

    @Test
    fun `scalar tanh gradient matches analytic`() {
        // d/dx tanh(x) = 1 - tanh(x)^2
        assertScalarGradient(
            body = "x.tanh()",
            imports = listOf("io.tlaloc.core.tanh"),
            xs = listOf(0.5f, -1.0f, 0.0f),
        ) { x -> 1.0f - tanh(x.toDouble()).toFloat() * tanh(x.toDouble()).toFloat() }
    }

    @Test
    fun `scalar sigmoid gradient matches analytic`() {
        // d/dx sigmoid(x) = s * (1 - s), s = 1 / (1 + exp(-x))
        assertScalarGradient(
            body = "x.sigmoid()",
            imports = listOf("io.tlaloc.core.sigmoid"),
            xs = listOf(0.5f, -1.0f, 0.0f),
        ) { x ->
            val s = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
            s * (1.0f - s)
        }
    }

    @Test
    fun `scalar tanh and sigmoid compose in one body`() {
        // f(x) = 2*tanh(x) + sigmoid(x); f'(x) = 2*(1 - tanh^2) + s*(1 - s)
        assertScalarGradient(
            body = "2.0f * x.tanh() + x.sigmoid()",
            imports = listOf("io.tlaloc.core.sigmoid", "io.tlaloc.core.tanh"),
            xs = listOf(0.25f, 1.5f),
        ) { x ->
            val t = tanh(x.toDouble()).toFloat()
            val s = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
            2.0f * (1.0f - t * t) + s * (1.0f - s)
        }
    }

    private fun assertScalarGradient(
        body: String,
        imports: List<String>,
        xs: List<Float>,
        analytic: (Float) -> Float,
    ) {
        val src = """
            import io.tlaloc.autograd.grad
            ${imports.joinToString("\n            ") { "import $it" }}
            fun main() {
                val g = grad { x: Float -> $body }
                println(listOf(${xs.joinToString(", ") { "${it}f" }}).map { g(it) }.joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(
            0,
            result.exitCode,
            "compile/run failed:\n" + result.messages
                .filter {
                    it.severity == CompilerMessageSeverity.ERROR || it.severity == CompilerMessageSeverity.WARNING
                }
                .joinToString("\n") { "${it.severity}: ${it.message}" },
        )
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(xs.size, parts.size, "expected one gradient per probe point: ${result.stdout}")
        for ((i, x) in xs.withIndex()) {
            assertTrue(
                abs(parts[i] + 1.0f) > 1e-3f,
                "slot $i = ${parts[i]} matches the broken-stub sentinel; the IR transform did not fire",
            )
            val want = analytic(x)
            assertTrue(
                abs(parts[i] - want) < 1e-4f,
                "d/dx($body) at x=$x: got ${parts[i]}, want $want",
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
        val tempDir = Files.createTempDirectory("tlaloc-scalar-tanh-sigmoid-test").toFile()
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
        private val AUTOGRAD_STUB_BROKEN = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()
    }
}
