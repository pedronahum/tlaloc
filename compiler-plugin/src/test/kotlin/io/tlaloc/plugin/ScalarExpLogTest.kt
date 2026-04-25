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
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.158 — scalar `Float.exp()` / `Float.log()` plugin lowering.
 *
 * Mirrors the §0.4.38 scalar-sqrt pattern (added by [BrachistochroneTest]):
 * scalar exp/log extensions live in `:core/DScalar.kt` at FQNs
 * `io.tlaloc.core.exp` and `io.tlaloc.core.log`; `FirLambdaToDxirLowering.UNARY_OP_MAP`
 * maps each to `OpKind.EXP` / `OpKind.LOG`; `DxirToIrSynthesis.irExp` / `irLog`
 * (already shipping since §0.4.53) emit the IrCalls back to `kotlin.math.exp` /
 * `kotlin.math.ln`. ExpRule / LogRule cover the gradient side at the SCT path.
 *
 * Why this matters: `docs/HMC_PORT_PLAN.md` Phase 1's gap analysis flagged scalar
 * exp/log as the missing primitive blocking the HMC port. The full HMC kernel
 * computes `log(1 + exp(-Xβ))` per record; scalar/loop-form ports of that compute
 * `(1.0f + (-xb).exp()).log()` element-by-element.
 *
 * Three tests:
 *  1. `Float.exp()` standalone — d/dx exp(x) = exp(x).
 *  2. `Float.log()` standalone — d/dx log(x) = 1/x.
 *  3. The HMC per-record pattern — `(1 + exp(-x)).log()` — d/dx = -1 / (1 + exp(x)).
 */
class ScalarExpLogTest {

    @Test
    fun `scalar exp gradient matches analytic`() {
        // f(x) = exp(x); df/dx = exp(x).
        // At x = 0.5: exp(0.5) = 1.6487...
        // At x = -1.0: exp(-1.0) = 0.3679...
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.exp
            fun main() {
                val g = grad { x: Float -> x.exp() }
                println("${'$'}{g(0.5f)},${'$'}{g(-1.0f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(2, parts.size)
        assertTrue(
            abs(parts[0] + 1.0f) > 1e-3f,
            "first slot = ${parts[0]} matches broken-stub sentinel; IR transform did not fire",
        )
        assertTrue(
            abs(parts[0] - exp(0.5).toFloat()) < 1e-4f,
            "expected df/dx(0.5) = exp(0.5) ≈ 1.6487, got ${parts[0]}",
        )
        assertTrue(
            abs(parts[1] - exp(-1.0).toFloat()) < 1e-4f,
            "expected df/dx(-1.0) = exp(-1.0) ≈ 0.3679, got ${parts[1]}",
        )
    }

    @Test
    fun `scalar log gradient matches analytic`() {
        // f(x) = log(x); df/dx = 1/x.
        // At x = 2.0: 1/2 = 0.5.
        // At x = 0.5: 1/0.5 = 2.0.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.log
            fun main() {
                val g = grad { x: Float -> x.log() }
                println("${'$'}{g(2.0f)},${'$'}{g(0.5f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(2, parts.size)
        assertTrue(
            abs(parts[0] + 1.0f) > 1e-3f,
            "first slot = ${parts[0]} matches broken-stub sentinel",
        )
        assertTrue(
            abs(parts[0] - 0.5f) < 1e-4f,
            "expected df/dx(2.0) = 1/2 = 0.5, got ${parts[0]}",
        )
        assertTrue(
            abs(parts[1] - 2.0f) < 1e-4f,
            "expected df/dx(0.5) = 1/0.5 = 2.0, got ${parts[1]}",
        )
    }

    @Test
    fun `HMC per-record term gradient matches analytic`() {
        // f(x) = log(1 + exp(-x))
        // df/dx = -exp(-x) / (1 + exp(-x)) = -1 / (1 + exp(x))
        // At x = 0.0:  -1 / (1 + 1)         = -0.5
        // At x = 2.0:  -1 / (1 + exp(2.0))  ≈ -0.1192
        // At x = -1.0: -1 / (1 + exp(-1.0)) ≈ -0.7311
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.exp
            import io.tlaloc.core.log
            fun main() {
                val g = grad { x: Float -> (1.0f + (-x).exp()).log() }
                println("${'$'}{g(0.0f)},${'$'}{g(2.0f)},${'$'}{g(-1.0f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, parts.size)
        val expected0 = (-1.0 / (1.0 + exp(0.0))).toFloat()
        val expected1 = (-1.0 / (1.0 + exp(2.0))).toFloat()
        val expected2 = (-1.0 / (1.0 + exp(-1.0))).toFloat()
        assertTrue(
            abs(parts[0] + 1.0f) > 1e-3f,
            "first slot = ${parts[0]} matches broken-stub sentinel",
        )
        assertTrue(abs(parts[0] - expected0) < 1e-4f, "x=0: expected $expected0, got ${parts[0]}")
        assertTrue(abs(parts[1] - expected1) < 1e-4f, "x=2: expected $expected1, got ${parts[1]}")
        assertTrue(abs(parts[2] - expected2) < 1e-4f, "x=-1: expected $expected2, got ${parts[2]}")
    }

    // --------- Harness (file-local; mirrors BrachistochroneTest's helpers) ---------

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
    )

    private data class RunResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
        val stdout: String,
    )

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-explog-run").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }

            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean =
                    collected.any { it.severity == CompilerMessageSeverity.ERROR }
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
            val capturedOut = PrintStream(baos, /* autoFlush = */ true, Charsets.UTF_8)
            val urls = arrayOf(outDir.toURI().toURL())
            val loader = URLClassLoader(urls, javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                val mainCls = loader.loadClass("MainKt")
                val mainMethod = mainCls.getMethod("main")
                mainMethod.invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        /** Sentinel stub: matching gradient values prove the IR transform fired. */
        private val AUTOGRAD_STUB_BROKEN = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()
    }
}
