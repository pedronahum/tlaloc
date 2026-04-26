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
 * §0.4.175 — CartPole Phase 1 port (per-time-step physics, hard-coded action).
 *
 * Closes the multi-firing port that opened with §0.4.165 (planning), passed through
 * §0.4.166 (sin/cos), §0.4.167 (abs), §0.4.168 (first attempt + downstream gate),
 * §0.4.169–§0.4.173 (diagnostics), §0.4.174 (lift pass + first end-to-end compile,
 * but with a 2.6× incorrect gradient), and lands at §0.4.175 with the deep-clone
 * fix that closes the cross-IF leak.
 *
 * Verifies the K2-plugin-synthesised gradient against finite-differencing of the
 * SAME source — so the comparison isolates SCT-side accounting from forward
 * arithmetic.
 */
class CartPolePhase1Test {

    @Test
    fun `cartpole phase1 gradient matches finite difference`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.abs
            import io.tlaloc.core.cos
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.sin
            import io.tlaloc.core.ops.get
            fun primal(packed: DTensor<Rank1<Sym>, F32>): Float {
                val at = packed[0]
                val x0 = packed[1]
                val x1 = packed[2]
                val x2 = packed[3]
                val x3 = packed[4]
                val rt = 9.0f * at + 0.045f * x3 * x3 * x2.sin()
                val cosX2 = x2.cos()
                val qt = (9.8f * x2.sin() - rt * cosX2) / (0.65f - 0.4f * cosX2 * cosX2)
                val pt = rt - 0.045f * qt * cosX2
                val xn0 = x0 + 0.02f * x1
                val xn2 = x2 + 0.02f * x3
                val maxArg = (2.4f - xn0.abs()) * (0.21f - xn2.abs())
                val clipped = if (maxArg > 0.0f) maxArg else 0.0f
                val term = 0.5f - clipped
                return term * term
            }
            fun main() {
                val g = grad { packed: DTensor<Rank1<Sym>, F32> ->
                    val at = packed[0]
                    val x0 = packed[1]
                    val x1 = packed[2]
                    val x2 = packed[3]
                    val x3 = packed[4]
                    val rt = 9.0f * at + 0.045f * x3 * x3 * x2.sin()
                    val cosX2 = x2.cos()
                    val qt = (9.8f * x2.sin() - rt * cosX2) / (0.65f - 0.4f * cosX2 * cosX2)
                    val pt = rt - 0.045f * qt * cosX2
                    val xn0 = x0 + 0.02f * x1
                    val xn2 = x2 + 0.02f * x3
                    val maxArg = (2.4f - xn0.abs()) * (0.21f - xn2.abs())
                    val clipped = if (maxArg > 0.0f) maxArg else 0.0f
                    val term = 0.5f - clipped
                    term * term
                }
                val cfg = floatArrayOf(0.5f, 0.0f, 0.1f, 0.05f, 0.02f)
                val input = Tensors.f32Vector<Sym>(cfg)
                val analytic = g(input).hostF32()
                val eps = 1.0e-3f
                val fd = FloatArray(5)
                for (k in 0 until 5) {
                    val plus = cfg.copyOf().also { it[k] += eps }
                    val minus = cfg.copyOf().also { it[k] -= eps }
                    fd[k] = (primal(Tensors.f32Vector<Sym>(plus)) - primal(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                }
                for (k in 0 until 5) {
                    print("${'$'}{analytic[k]}=${'$'}{fd[k]};")
                }
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_5, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // Sanity: confirm the IR-side synthesis fired (no "kept original call" warning).
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING &&
                "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "K2 plugin failed to synthesise CartPole Phase 1 — runtime tape fallback fired",
        )
        val parts = result.stdout.trim().trimEnd(';').split(";")
        assertEquals(5, parts.size, "expected 5 slots, got: ${result.stdout}")
        for ((k, part) in parts.withIndex()) {
            val (analyticStr, fdStr) = part.split("=", limit = 2)
            val analytic = analyticStr.toFloat()
            val fd = fdStr.toFloat()
            // Sentinel check: any analytic = -1.0 collapses the fallback case unambiguously.
            assertTrue(
                abs(analytic + 1.0f) > 1e-6f || abs(fd + 1.0f) > 1e-6f,
                "slot $k: analytic=$analytic fd=$fd both equal -1.0 sentinel; IR transform didn't fire",
            )
            val absErr = abs(analytic - fd)
            val relErr = absErr / (abs(fd) + 1.0e-7f)
            // Mixed tolerance — abs+sin/cos+IF composition accumulates modest f32 noise.
            assertTrue(
                absErr < 1e-3f || relErr < 5e-3f,
                "slot $k: analytic=$analytic fd=$fd absErr=$absErr relErr=$relErr — both tolerances exceeded",
            )
        }
    }

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
        val tempDir = Files.createTempDirectory("tlaloc-cartpole-run").toFile()
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
        private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT_5 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(floatArrayOf(-1.0f, -1.0f, -1.0f, -1.0f, -1.0f)), intArrayOf(5), F32) }
        """.trimIndent()
    }
}
