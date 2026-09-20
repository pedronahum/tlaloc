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
 * §0.4.407 — the IF direct forward arm, E2E. A `jvp {}` over a Kotlin if/else
 * lowers to a genuine [io.tlaloc.ir.OpKind.IF] (the §0.4.24 lowerWhen path —
 * scalar conditionals are true IF regions, not the §0.4.364 tensor `where`
 * surface), which §0.4.403's forward branch left to the "fall back otherwise"
 * tail: PhiCalculus's F-rules keep genuine two-branch conditionals, and
 * DxirForwardTransform refused every region-bearing op — so this exact body
 * warned "kept original call" and hit the stub at runtime. Now the transform's
 * IF arm lowers it: the tangent is a SECOND if/else over the same condition.
 *
 * The test pins (a) no fallback, (b) the IF genuinely SURVIVED to synthesis —
 * the lowered-dxir dump must contain an IF op, so a PhiCalculus rewrite that
 * silently closed the conditional could not turn this into a vacuous pass —
 * and (c) values on both sides of the branch against the analytic derivative
 * and a Double central difference, with broken-sentinel stubs proving the
 * rewrite fired.
 */
class JvpIfIntrinsicTest {

    @Test
    fun `jvp and valueAndJvp lower a kotlin if-else body through the IF forward arm`() {
        val src = """
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.valueAndJvp
            fun main() {
                val j = jvp { x: Float -> if (x > 0f) x * x else -x }
                val vj = valueAndJvp { x: Float -> if (x > 0f) x * x else -x }
                println("dyThen " + j(2.0f, 1.0f))
                println("dyScaled " + j(2.0f, -2.0f))
                println("dyElse " + j(-3.0f, 1.0f))
                val (y, dy) = vj(2.0f, 1.0f)
                println("y " + y)
                println("dy " + dy)
            }
        """.trimIndent()
        val result = compileAndRun(STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the if/else body to lower through the IF forward arm. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // The IF must SURVIVE to the synthesised function — if a rewrite closed
        // the conditional into straight-line arithmetic, this E2E would not be
        // exercising the new arm at all.
        val loweredWithIf = result.messages.any {
            "lowered 'jvp' to forward-mode dxir" in it.message && "= if(" in it.message
        }
        assertTrue(
            loweredWithIf,
            "expected the lowered forward-mode dxir to carry an IF op. Messages:\n" +
                result.messages.filter { "forward-mode dxir" in it.message }
                    .joinToString("\n--\n") { it.message },
        )

        // f(x) = if (x > 0) x² else -x;  f'(x) = if (x > 0) 2x else -1.
        val wantDyThen = 4.0f
        val wantDyScaled = -8.0f
        val wantDyElse = -1.0f
        val wantY = 4.0f

        // Central-difference cross-check on both sides (Double, eps=1e-6).
        fun prim(x: Double): Double = if (x > 0) x * x else -x
        val eps = 1e-6
        val fdThen = (prim(2.0 + eps) - prim(2.0 - eps)) / (2 * eps)
        val fdElse = (prim(-3.0 + eps) - prim(-3.0 - eps)) / (2 * eps)
        assertTrue(abs(fdThen - wantDyThen) < 1e-4, "oracle drift: fdThen=$fdThen")
        assertTrue(abs(fdElse - wantDyElse) < 1e-4, "oracle drift: fdElse=$fdElse")

        val vals = result.stdout.trim().lines().associate {
            val (k, num) = it.trim().split(" ", limit = 2)
            k to num.toFloat()
        }
        assertTrue(
            abs(vals.getValue("dyThen") + 1f) > 1e-3f,
            "sentinel -1 returned — the rewrite never fired. stdout:\n${result.stdout}",
        )
        assertTrue(abs(vals.getValue("dyThen") - wantDyThen) < 1e-4f, "dyThen=${vals["dyThen"]} want $wantDyThen")
        assertTrue(abs(vals.getValue("dyScaled") - wantDyScaled) < 1e-4f, "dyScaled=${vals["dyScaled"]} want $wantDyScaled")
        assertTrue(abs(vals.getValue("dyElse") - wantDyElse) < 1e-4f, "dyElse=${vals["dyElse"]} want $wantDyElse")
        assertTrue(abs(vals.getValue("y") - wantY) < 1e-4f, "y=${vals["y"]} want $wantY")
        assertTrue(abs(vals.getValue("dy") - wantDyThen) < 1e-4f, "dy=${vals["dy"]} want $wantDyThen")
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-jvp-if-test").toFile()
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
        /** Broken-sentinel stubs (the BrachistochroneTest convention): if the
         * plugin rewrite does not fire, the program prints -1.0 and the test
         * fails on the sentinel check rather than passing by accident. */
        private val STUB = """
            package io.tlaloc.autograd
            fun jvp(f: (Float) -> Float): (Float, Float) -> Float = { _, _ -> -1.0f }
            fun valueAndJvp(f: (Float) -> Float): (Float, Float) -> Pair<Float, Float> =
                { _, _ -> Pair(-1.0f, -1.0f) }
        """.trimIndent()
    }
}
