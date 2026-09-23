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
 * §0.4.426 — Phase C5 tail: the `grad {}`-integrable integral through the
 * OPEN B5 gate, used exactly as a USER would (the point of the gate — the
 * §0.4.411 record names the integral "exactly a custom-derivative citizen").
 *
 * The user spelling for FIXED LITERAL BOUNDS: the primal is a fixed-node
 * quadrature written in lowerable ops (Simpson over {0, ½, 1} here — exact
 * for integrands of degree ≤ 3 in x, so the x²-family pins are analytic, the
 * quarter-integer-grid discipline), and the adjoint is the LEIBNIZ RULE
 * d/dp ∫ₐᵇ f(x; p) dx = ∫ₐᵇ ∂f/∂p dx, attached verbatim via `customVjp`
 * (reverse) / `customVjpJvp` (both modes — the JVP⇄VJP cross-identity cert).
 * The host-level twin of the same wiring is `integralWithParamGrad`
 * (:core/Integral.kt), pinned against the same closed forms in IntegralTest.
 *
 * What this class deliberately does NOT claim: a reusable
 * `integral(a, b) { f }` library spelling. B5 v1 requires lambda literals at
 * the customVjp call site inside the differentiated body (no escape, no
 * wrapping helper), so the region-op spelling — Romberg over the interpreted
 * lowered body, Leibniz adjoint as a quadrature over the body's own VJP —
 * stays the plan's named C5 deferral.
 */
class IntegralGradientTest {

    @Test
    fun `grad through a customVjp integral - Leibniz adjoint, analytic pins`() {
        val src = """
            import io.tlaloc.autograd.customVjp
            import io.tlaloc.autograd.grad
            fun main() {
                // g(a) = ∫₀¹ a·x² dx: primal = Simpson over {0, 1/2, 1}
                // (exact for x²), adjoint = Leibniz d/da = ∫₀¹ x² dx = 1/3.
                val g1 = grad { a: Float ->
                    val integ = customVjp(
                        { p: Float ->
                            (p * 0.0f + 4.0f * (p * 0.25f) + p * 1.0f) * 0.16666667f
                        },
                        { u: Float, p: Float -> u * 0.33333334f },
                    )
                    integ(a)
                }
                println("linear " + g1(2.0f))
                // g(a) = ∫₀¹ (a·x)² dx = a²/3: adjoint = 2a/3 — the Leibniz
                // quadrature ∫₀¹ 2a·x² dx spelled in closed form.
                val g2 = grad { a: Float ->
                    val integ = customVjp(
                        { p: Float ->
                            ((p * 0.0f) * (p * 0.0f) + 4.0f * ((p * 0.5f) * (p * 0.5f)) + (p * 1.0f) * (p * 1.0f)) * 0.16666667f
                        },
                        { u: Float, p: Float -> u * (p * 0.6666667f) },
                    )
                    integ(a)
                }
                println("nonlinear " + g2(1.5f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val lines = result.stdout.trim().lines()
        assertEquals(2, lines.size, "stdout: ${result.stdout}")
        val linear = lines[0].removePrefix("linear ").toFloat()
        val nonlinear = lines[1].removePrefix("nonlinear ").toFloat()
        assertTrue(linear != -1.0f, "stub sentinel returned — the rewrite never fired")
        // d/da ∫₀¹ a·x² dx = 1/3, independent of a.
        assertTrue(abs(linear - 1.0f / 3.0f) < 1e-6f, "linear grad = $linear, want 1/3")
        // d/da ∫₀¹ (a·x)² dx = 2a/3 = 1 at a = 3/2.
        assertTrue(abs(nonlinear - 1.0f) < 1e-5f, "nonlinear grad = $nonlinear, want 1.0")
    }

    @Test
    fun `customVjpJvp integral - JVP-VJP cross-identity at the analytic value`() {
        val src = """
            import io.tlaloc.autograd.customVjpJvp
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.valueAndJvp
            fun main() {
                // g(a) = ∫₀¹ (a·x)² dx = a²/3 with BOTH user derivatives:
                // vjpFn = u·2a/3, jvpFn = da·2a/3 — the same Leibniz number,
                // two independent user encodings.
                val vj = valueAndJvp { a: Float ->
                    val integ = customVjpJvp(
                        { p: Float ->
                            ((p * 0.0f) * (p * 0.0f) + 4.0f * ((p * 0.5f) * (p * 0.5f)) + (p * 1.0f) * (p * 1.0f)) * 0.16666667f
                        },
                        { u: Float, p: Float -> u * (p * 0.6666667f) },
                        { p: Float, dp: Float -> dp * (p * 0.6666667f) },
                    )
                    integ(a)
                }
                val (y, dy) = vj(1.5f, 2.0f)
                println("value " + y)
                println("tangent " + dy)
                val g = grad { a: Float ->
                    val integ = customVjpJvp(
                        { p: Float ->
                            ((p * 0.0f) * (p * 0.0f) + 4.0f * ((p * 0.5f) * (p * 0.5f)) + (p * 1.0f) * (p * 1.0f)) * 0.16666667f
                        },
                        { u: Float, p: Float -> u * (p * 0.6666667f) },
                        { p: Float, dp: Float -> dp * (p * 0.6666667f) },
                    )
                    integ(a)
                }
                println("grad " + g(1.5f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val lines = result.stdout.trim().lines()
        assertEquals(3, lines.size, "stdout: ${result.stdout}")
        val value = lines[0].removePrefix("value ").toFloat()
        val tangent = lines[1].removePrefix("tangent ").toFloat()
        val gradOut = lines[2].removePrefix("grad ").toFloat()
        assertTrue(value != -1.0f && gradOut != -1.0f, "stub sentinel returned — a rewrite never fired")
        // Primal: (3/2)²/3 = 3/4 — the Simpson tableau is exact for x².
        assertTrue(abs(value - 0.75f) < 1e-6f, "value = $value, want a²/3 = 0.75")
        // Forward: dy = dp·2a/3 = 2·1 = 2. Reverse: ∇ = 2a/3 = 1.
        assertTrue(abs(tangent - 2.0f) < 1e-5f, "tangent = $tangent, want 2.0")
        assertTrue(abs(gradOut - 1.0f) < 1e-5f, "grad = $gradOut, want 1.0")
        // The scalar JVP⇄VJP cross-identity: ⟨∇g(a), dp⟩ = jvp_g(a, dp).
        assertTrue(
            abs(gradOut * 2.0f - tangent) < 1e-5f,
            "JVP⇄VJP cross-identity failed: grad·dp = ${gradOut * 2.0f} vs tangent $tangent",
        )
    }

    private fun assertNoFallback(result: RunResult) {
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the customVjp integral to lower:\n${result.messages.render()}",
        )
    }

    private fun List<CompileMessage>.render(): String =
        joinToString("\n--\n") { "${it.severity}: ${it.message}" }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-integral-test").toFile()
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
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
            fun valueAndJvp(f: (Float) -> Float): (Float, Float) -> Pair<Float, Float> =
                { _, _ -> Pair(-1.0f, -1.0f) }
            fun <A, R> customVjp(f: (A) -> R, vjpFn: (R, A) -> A): (A) -> R = f
            fun <A, R> customVjpJvp(f: (A) -> R, vjpFn: (R, A) -> A, jvpFn: (A, A) -> R): (A) -> R = f
        """.trimIndent()
    }
}
