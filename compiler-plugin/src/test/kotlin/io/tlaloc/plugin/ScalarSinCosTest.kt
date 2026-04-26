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
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.166 — scalar `Float.sin()` / `Float.cos()` plugin lowering.
 *
 * Direct mirror of §0.4.158's `ScalarExpLogTest.kt` pattern: scalar trig extensions
 * live in `:core/DScalar.kt` at FQNs `io.tlaloc.core.sin` / `io.tlaloc.core.cos`;
 * `FirLambdaToDxirLowering.UNARY_OP_MAP` maps each to `OpKind.SIN` / `OpKind.COS`;
 * `DxirToIrSynthesis.irSin` / `irCos` emit `IrCall` to `kotlin.math.sin` /
 * `kotlin.math.cos`. `SinRule` (`d/dx sin = cos`) and `CosRule` (`d/dx cos = -sin`)
 * cover the gradient side; their mutual cross-references make landing them as
 * a pair the natural shape.
 *
 * Why this matters: `docs/CARTPOLE_PORT_PLAN.md` Phase 0a's gap analysis flagged
 * scalar sin/cos as missing primitives blocking the CartPole port. The full
 * CartPole physics step uses `sin(x_t,2)` and `cos(x_t,2)` for the pole's
 * angular state.
 *
 * Three tests:
 *  1. `Float.sin()` standalone — d/dx sin(x) = cos(x).
 *  2. `Float.cos()` standalone — d/dx cos(x) = -sin(x).
 *  3. CartPole-pattern term — `9·a + 0.045·x²·sin(θ)` for the inertial term `r_t`,
 *     differentiated w.r.t. `a`. d/da = 9 (sanity: x and θ shouldn't appear).
 */
class ScalarSinCosTest {

    @Test
    fun `scalar sin gradient matches analytic`() {
        // f(x) = sin(x); df/dx = cos(x).
        // At x = 0.5: cos(0.5) ≈ 0.8776
        // At x = π/4 ≈ 0.7854: cos(π/4) ≈ 0.7071
        // At x = -1.0: cos(-1.0) ≈ 0.5403
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sin
            fun main() {
                val g = grad { x: Float -> x.sin() }
                println("${'$'}{g(0.5f)},${'$'}{g(0.7854f)},${'$'}{g(-1.0f)}")
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
        assertTrue(
            abs(parts[0] - cos(0.5).toFloat()) < 1e-4f,
            "expected df/dx(0.5) = cos(0.5) ≈ 0.8776, got ${parts[0]}",
        )
        assertTrue(
            abs(parts[1] - cos(0.7854).toFloat()) < 1e-4f,
            "expected df/dx(π/4) = cos(π/4) ≈ 0.7071, got ${parts[1]}",
        )
        assertTrue(
            abs(parts[2] - cos(-1.0).toFloat()) < 1e-4f,
            "expected df/dx(-1.0) = cos(-1.0) ≈ 0.5403, got ${parts[2]}",
        )
    }

    @Test
    fun `scalar cos gradient matches analytic`() {
        // f(x) = cos(x); df/dx = -sin(x).
        // At x = 0.5: -sin(0.5) ≈ -0.4794
        // At x = 0.0: -sin(0.0) = 0.0
        // At x = -1.0: -sin(-1.0) ≈ 0.8415
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.cos
            fun main() {
                val g = grad { x: Float -> x.cos() }
                println("${'$'}{g(0.5f)},${'$'}{g(0.0f)},${'$'}{g(-1.0f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, parts.size)
        assertTrue(
            abs(parts[0] + 1.0f) > 1e-3f,
            "first slot = ${parts[0]} matches broken-stub sentinel",
        )
        assertTrue(
            abs(parts[0] - (-sin(0.5).toFloat())) < 1e-4f,
            "expected df/dx(0.5) = -sin(0.5) ≈ -0.4794, got ${parts[0]}",
        )
        assertTrue(
            abs(parts[1] - 0.0f) < 1e-4f,
            "expected df/dx(0.0) = 0, got ${parts[1]}",
        )
        assertTrue(
            abs(parts[2] - (-sin(-1.0).toFloat())) < 1e-4f,
            "expected df/dx(-1.0) = -sin(-1.0) ≈ 0.8415, got ${parts[2]}",
        )
    }

    @Test
    fun `CartPole inertial term gradient matches analytic`() {
        // CartPole physics: `r_t = 9·a + 0.045·x²·sin(θ)`. Differentiate w.r.t. `a` —
        // the action is the only differentiation target this test pins, so the result
        // should be exactly 9 regardless of x or θ.
        // (Multi-input grad would be the broader test; this one verifies the sin
        //  term is correctly handled when it's a constant w.r.t. the diff variable.)
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.sin
            fun main() {
                val g = grad { a: Float ->
                    // x = 1.5, theta = 0.3 hard-coded as Float literals.
                    val r = 9.0f * a + 0.045f * 1.5f * 1.5f * (0.3f).sin()
                    r
                }
                println("${'$'}{g(0.5f)},${'$'}{g(-2.0f)},${'$'}{g(0.0f)}")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(",").map { it.toFloat() }
        assertEquals(3, parts.size)
        for ((i, v) in parts.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-3f,
                "slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(
                abs(v - 9.0f) < 1e-4f,
                "expected df/da = 9 (a-coefficient), got $v at slot $i",
            )
        }
    }

    // --------- Harness (file-local; mirrors ScalarExpLogTest's helpers) ---------

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
        val tempDir = Files.createTempDirectory("tlaloc-sincos-run").toFile()
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
