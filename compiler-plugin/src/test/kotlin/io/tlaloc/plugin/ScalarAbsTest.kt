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
 * §0.4.167 — scalar `Float.abs()` plugin lowering.
 *
 * Mirrors §0.4.166's scalar sin/cos pattern. `Float.abs()` resolves at FQN
 * `io.tlaloc.core.abs`; `FirLambdaToDxirLowering.UNARY_OP_MAP` maps it to
 * `OpKind.ABS`; `DxirToIrSynthesis.irAbs` emits `IrCall` to `kotlin.math.abs`.
 * `AbsRule` covers the gradient side via `STEP(x) - STEP(-x)` (= sign(x), with
 * sign(0) = 0).
 *
 * Why this matters: `docs/CARTPOLE_PORT_PLAN.md` Phase 0a-2 named scalar `abs`
 * as the next CartPole prerequisite. The full CartPole loss uses
 * `(2.4 - |xt+1,0|) · (0.21 - |xt+1,2|)` for the clipping term.
 *
 * Three tests:
 *  1. Standalone `Float.abs()` — `df/dx |x| = sign(x)` at positive, negative,
 *     and zero arguments. The sign(0) = 0 convention is the standard AD choice
 *     (PyTorch / JAX agree).
 *  2. CartPole-pattern term — `(2.4 - |x|) * (0.21 - |y|)` differentiated
 *     w.r.t. x at multiple sample points.
 *  3. Composite gradient — `(|x| - 1)²` differentiated w.r.t. x. Pinned at
 *     three sample points covering both sign branches.
 */
class ScalarAbsTest {

    @Test
    fun `scalar abs gradient matches sign convention`() {
        // f(x) = |x|; df/dx = sign(x) with sign(0) = 0.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.abs
            fun main() {
                val g = grad { x: Float -> x.abs() }
                println("${'$'}{g(2.0f)},${'$'}{g(-3.0f)},${'$'}{g(0.0f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, parts.size)
        assertTrue(
            abs(parts[0] + 1.0f) > 1e-3f,
            "first slot = ${parts[0]} matches broken-stub sentinel; IR transform did not fire",
        )
        // x = 2: sign(2) = +1.
        assertTrue(abs(parts[0] - 1.0f) < 1e-4f, "df/dx(2.0) expected +1, got ${parts[0]}")
        // x = -3: sign(-3) = -1.
        assertTrue(abs(parts[1] - (-1.0f)) < 1e-4f, "df/dx(-3.0) expected -1, got ${parts[1]}")
        // x = 0: sign(0) = 0 (Tlaloc convention via STEP(x) - STEP(-x)).
        assertTrue(abs(parts[2] - 0.0f) < 1e-4f, "df/dx(0.0) expected 0, got ${parts[2]}")
    }

    @Test
    fun `CartPole loss-clip term gradient matches analytic`() {
        // f(x) = (2.4 - |x|) * (0.21 - |y|);  y hardcoded to 0.5 (so |y| = 0.5).
        // Note (0.21 - 0.5) = -0.29 (negative, as in the actual CartPole when state
        // exceeds the 0.21 angular threshold).
        // df/dx = (-sign(x)) * (0.21 - |y|) = -sign(x) * (-0.29) = 0.29 * sign(x).
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.abs
            fun main() {
                val g = grad { x: Float ->
                    val y = 0.5f
                    (2.4f - x.abs()) * (0.21f - y.abs())
                }
                println("${'$'}{g(1.0f)},${'$'}{g(-1.5f)},${'$'}{g(2.0f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, parts.size)
        for ((i, v) in parts.withIndex()) {
            assertTrue(abs(v + 1.0f) > 1e-3f, "slot $i = $v matches broken-stub sentinel")
        }
        // Expected: 0.29 * sign(x_i).
        // At x = 1.0 (positive): 0.29.
        assertTrue(abs(parts[0] - 0.29f) < 1e-4f, "expected 0.29 at x=1.0, got ${parts[0]}")
        // At x = -1.5 (negative): -0.29.
        assertTrue(abs(parts[1] - (-0.29f)) < 1e-4f, "expected -0.29 at x=-1.5, got ${parts[1]}")
        // At x = 2.0 (positive): 0.29.
        assertTrue(abs(parts[2] - 0.29f) < 1e-4f, "expected 0.29 at x=2.0, got ${parts[2]}")
    }

    @Test
    fun `composite abs gradient matches analytic`() {
        // f(x) = (|x| - 1)²;  df/dx = 2*(|x| - 1)*sign(x).
        // At x = 3: 2*(3-1)*1 = 4.
        // At x = -2: 2*(2-1)*(-1) = -2.
        // At x = 0.5: 2*(0.5-1)*1 = -1.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.abs
            fun main() {
                val g = grad { x: Float ->
                    val a = x.abs() - 1.0f
                    a * a
                }
                println("${'$'}{g(3.0f)},${'$'}{g(-2.0f)},${'$'}{g(0.5f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, parts.size)
        assertTrue(abs(parts[0] - 4.0f) < 1e-4f, "expected 4 at x=3, got ${parts[0]}")
        assertTrue(abs(parts[1] - (-2.0f)) < 1e-4f, "expected -2 at x=-2, got ${parts[1]}")
        assertTrue(abs(parts[2] - (-1.0f)) < 1e-4f, "expected -1 at x=0.5, got ${parts[2]}")
    }

    // --------- Harness (file-local; mirrors ScalarSinCosTest's helpers) ---------

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
        val tempDir = Files.createTempDirectory("tlaloc-abs-run").toFile()
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
        private val AUTOGRAD_STUB_BROKEN = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()
    }
}
