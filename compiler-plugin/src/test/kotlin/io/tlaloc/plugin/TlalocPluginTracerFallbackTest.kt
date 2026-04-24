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
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.58 — D.3ii-tape-tracer-integration.
 *
 * End-to-end proof that a user's `grad { x: Tracer<ScalarShape> -> ... }` with a
 * break-bearing `while` loop compiles through the Tlaloc plugin, falls back cleanly
 * when the plugin can't specialise the shape, and produces the correct gradient via
 * the runtime tape in `:autograd`. Unlike [TlalocPluginDiagnosticTest], which uses
 * `AUTOGRAD_STUB_BROKEN` sentinels to prove the plugin path didn't fire, this test
 * uses the REAL `:autograd` on the test classpath (added as a `testImplementation`
 * dependency in `compiler-plugin/build.gradle.kts`) so the observable output is the
 * real gradient.
 *
 * Splits from [TlalocPluginDiagnosticTest] because:
 *   - No stub is injected — the real autograd's `grad` / Tracer surface resolves
 *     against the classpath jar added via `testImplementation(project(":autograd"))`.
 *   - The test runs the compiled bytecode in a classloader that sees the real
 *     `io.tlaloc.autograd` package, so observed stdout reflects real arithmetic.
 */
class TlalocPluginTracerFallbackTest {

    @Test
    fun `break-bearing while on Tracer surface falls back and runtime tape produces correct gradient`() {
        // f(x) = doubling until d > 10. For x = 0.5:
        //   5 doublings → df/dx = 2^5 = 32.
        // The plugin CANNOT emit this closed-form (data-dependent break;
        // §0.4.55 deferred closure). It either lowers the Tracer surface and
        // hits DxirReverseTransform's gate → tryReverseTransform returns null
        // → original call kept; OR it rejects at FIR (Tracer params are out-of-
        // scope) → LAMBDA_UNSUPPORTED warning; either way the call remains
        // unmodified and the runtime tape in :autograd produces the gradient.
        val src = """
            import io.tlaloc.autograd.*
            import io.tlaloc.core.ScalarShape
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            fun main() {
                // §0.4.59 — `d.peek()` replaces the §0.4.58-era `d.toDTensor().hostF32()[0]`
                // polling idiom. No per-iteration FloatArray copy; the same fallback +
                // runtime-tape behaviour is invariant.
                val g = grad { x: Tracer<ScalarShape> ->
                    var d = x
                    while (d.peek() <= 10.0f) {
                        d = d + d
                    }
                    d
                }
                val out = g(Tensors.f32Scalar(0.5f)).hostF32()[0]
                println(out)
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages.joinToString("\n") { "[${it.severity}] ${it.message}" }}")
        assertEquals(
            "32.0",
            result.stdout.trim(),
            "expected runtime-tape gradient 32.0 (2^5) for x=0.5 doubling kernel; " +
                "got '${result.stdout.trim()}'",
        )
    }

    // --- test harness ------------------------------------------------------

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
    )

    private data class RunResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
        val stdout: String,
    )

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    /**
     * Compile [user] with the Tlaloc plugin enabled, no injected autograd stub — the
     * real `:autograd` is resolved via the host JVM's `java.class.path`. On successful
     * compile, execute `MainKt.main()` in a fresh [URLClassLoader] with stdout captured.
     */
    private fun compileAndRun(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-plugin-tracer-run").toFile()
        try {
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
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            try {
                System.setOut(PrintStream(baos))
                val mainKt = loader.loadClass("MainKt")
                val main = mainKt.getMethod("main")
                main.invoke(null)
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
            return RunResult(exitCode, collected, baos.toString())
        } finally {
            tempDir.deleteRecursively()
        }
    }
}
