package io.tlaloc.plugin

import io.tlaloc.core.digamma
import io.tlaloc.core.trigamma
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
 * §0.4.402 — Phase C1 special functions (DiffKT parity): LGAMMA and DIGAMMA
 * reach user code and differentiate in `grad {}` through the real K2 plugin,
 * in every spelling the FIR maps:
 *
 *  1. scalar `x.lgamma()` (`io.tlaloc.core.lgamma`) — d/dx = ψ(x); the
 *     synthesised gradient body calls the `io.tlaloc.core.digamma` scalar
 *     extension via the `irCoreScalarCall` path (no `kotlin.math` equivalent
 *     exists for ANY of these — the §0.4.377 sigmoid precedent)
 *  2. scalar `x.digamma()` — d/dx = ψ₁(x); the gradient body contains the
 *     INTERNAL TRIGAMMA op, so this pins that trigamma synthesises even
 *     though it has no FIR entry of its own
 *  3. tensor `x.lgamma().sum()` (`io.tlaloc.core.ops.lgamma`) — dxᵢ = ψ(xᵢ),
 *     asserted to synthesise with NO tape fallback
 *  4. the tensor digamma twin — dxᵢ = ψ₁(xᵢ)
 *
 * Probes stay on the positive axis away from the poles at 0, −1, −2, ….
 */
class LgammaDigammaGradientTest {

    @Test
    fun `scalar lgamma gradient matches analytic`() {
        assertScalarGradient(
            body = "x.lgamma()",
            imports = listOf("io.tlaloc.core.lgamma"),
            xs = listOf(0.5f, 1.5f, 3.2f),
        ) { x -> x.toDouble().digamma().toFloat() }
    }

    @Test
    fun `scalar digamma gradient matches analytic`() {
        assertScalarGradient(
            body = "x.digamma()",
            imports = listOf("io.tlaloc.core.digamma"),
            xs = listOf(0.5f, 1.5f, 3.2f),
        ) { x -> x.toDouble().trigamma().toFloat() }
    }

    @Test
    fun `tensor lgamma gradient matches analytic`() {
        assertTensorGradient(body = "x.lgamma().sum().toFloat()", op = "lgamma") { x ->
            x.toDouble().digamma().toFloat()
        }
    }

    @Test
    fun `tensor digamma gradient matches analytic`() {
        assertTensorGradient(body = "x.digamma().sum().toFloat()", op = "digamma") { x ->
            x.toDouble().trigamma().toFloat()
        }
    }

    private val tensorProbe = floatArrayOf(0.7f, 1.8f, 4.2f)

    private fun assertTensorGradient(body: String, op: String, analytic: (Float) -> Float) {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.$op
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank1<Sym>, F32> -> $body }
                val x = Tensors.f32Vector<Sym>(floatArrayOf(${tensorProbe.joinToString(", ") { "${it}f" }}))
                println(g(x).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1, src)
        assertEquals(
            0,
            result.exitCode,
            "compile/run failed:\n" + result.messages.joinToString("\n") { "${it.severity}: ${it.message}" },
        )
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the tensor $op gradient to lower. Warnings:\n${
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }
            }",
        )
        val got = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(tensorProbe.size, got.size, "expected one gradient slot per element: ${result.stdout}")
        for ((i, x) in tensorProbe.withIndex()) {
            assertTrue(
                abs(got[i] + 1.0f) > 1e-3f,
                "slot $i = ${got[i]} matches the broken-stub sentinel; the IR transform did not fire",
            )
            val want = analytic(x)
            assertTrue(abs(got[i] - want) < 1e-4f, "d($body)[$i] at x=$x: got ${got[i]}, want $want")
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
        val tempDir = Files.createTempDirectory("tlaloc-lgamma-test").toFile()
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

        /** Sentinel stub sized for the 3-element tensor probe. */
        private val AUTOGRAD_STUB_BROKEN_RANK1 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32) }
        """.trimIndent()
    }
}
