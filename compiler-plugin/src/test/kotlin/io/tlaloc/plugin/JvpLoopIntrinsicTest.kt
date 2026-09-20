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
 * §0.4.403 — Phase B3 E2E: forward mode through a LOOP-bearing body. Before
 * this slice the plugin's forward branch skipped the coarsening pipeline
 * entirely ("forward v1 is straight-line", §0.4.372), so a `jvp {}` over a
 * for-loop warned "kept original call" and threw `pluginMissing` at first
 * invocation. Now the forward branch runs PhiCalculus.apply (+ the region
 * lift) exactly as the reverse branch does, and the WHILE closes into a
 * straight-line body the forward transform handles.
 *
 * Surface: the Brachistochrone compound-velocity kernel (BrachistochroneTest
 * precedent) — f(y) = (1+y)^5 via `v = v + v·y` over 5 iterations.
 * Analytic: jvp(y, dy) = 5·(1+y)^4·dy; y = 1.5^5 = 7.59375 at y=0.5.
 * Cross-checked against a Double central difference in the test body.
 */
class JvpLoopIntrinsicTest {

    @Test
    fun `jvp and valueAndJvp lower a for-loop body through the coarsening pipeline`() {
        val src = """
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.valueAndJvp
            fun main() {
                val j = jvp { y: Float ->
                    var v = 1.0f
                    for (i in 0 until 5) {
                        v = v + v * y
                    }
                    v
                }
                val vj = valueAndJvp { y: Float ->
                    var v = 1.0f
                    for (i in 0 until 5) {
                        v = v + v * y
                    }
                    v
                }
                println("dy0 " + j(0.0f, 1.0f))
                println("dyH " + j(0.5f, 1.0f))
                println("dyS " + j(0.5f, -2.0f))
                val (y2, dy2) = vj(0.5f, 1.0f)
                println("y2 " + y2)
                println("dy2 " + dy2)
            }
        """.trimIndent()
        val result = compileAndRun(STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the loop body to coarsen and lower forward-mode. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // Analytic references: f(y) = (1+y)^5, f'(y) = 5·(1+y)^4.
        val wantDy0 = 5.0f                                    // 5·1^4·1
        val wantDyH = 5.0f * 1.5f * 1.5f * 1.5f * 1.5f        // 25.3125
        val wantDyS = wantDyH * -2.0f                         // -50.625
        val wantY2 = 1.5f * 1.5f * 1.5f * 1.5f * 1.5f         // 7.59375

        // Central-difference cross-check (Double, eps=1e-6): the analytic
        // closed form and the FD oracle must agree before we pin the plugin
        // output against either.
        fun prim(y: Double): Double {
            var v = 1.0
            repeat(5) { v += v * y }
            return v
        }
        val eps = 1e-6
        val fd = (prim(0.5 + eps) - prim(0.5 - eps)) / (2 * eps)
        assertTrue(
            abs(fd - wantDyH) / wantDyH < 1e-4,
            "test-internal oracle drift: fd=$fd vs analytic=$wantDyH",
        )

        val vals = result.stdout.trim().lines().associate {
            val (k, num) = it.trim().split(" ", limit = 2)
            k to num.toFloat()
        }
        assertTrue(
            abs(vals.getValue("dy0") + 1f) > 1e-3f,
            "sentinel -1 returned — the rewrite never fired. stdout:\n${result.stdout}",
        )
        assertTrue(abs(vals.getValue("dy0") - wantDy0) < 1e-3f, "dy0=${vals["dy0"]} want $wantDy0")
        assertTrue(abs(vals.getValue("dyH") - wantDyH) < 1e-2f, "dyH=${vals["dyH"]} want $wantDyH")
        assertTrue(abs(vals.getValue("dyS") - wantDyS) < 2e-2f, "dyS=${vals["dyS"]} want $wantDyS")
        assertTrue(abs(vals.getValue("y2") - wantY2) < 1e-3f, "y2=${vals["y2"]} want $wantY2")
        assertTrue(abs(vals.getValue("dy2") - wantDyH) < 1e-2f, "dy2=${vals["dy2"]} want $wantDyH")
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-jvp-loop-test").toFile()
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
