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
import kotlin.math.max
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Compiles a `Main.kt` with the real `:autograd` intrinsics and the Tlaloc plugin, runs
 * `main()` and returns what it printed. Shared by the F64 tests.
 */
internal object F64TestHarness {

    data class Message(val severity: CompilerMessageSeverity, val message: String, val line: Int, val column: Int)

    data class Result(val exitCode: Int, val messages: List<Message>, val stdout: String) {
        fun describe(): String =
            messages.joinToString("\n") { "${it.severity} ${it.line}:${it.column} ${it.message}" } +
                "\nstdout:\n$stdout"

        /** The line printed as `name v1,v2,...`, parsed as Doubles. */
        fun values(name: String): DoubleArray {
            val line = stdout.lines().firstOrNull { it.startsWith("$name ") }
                ?: error("no line '$name' in stdout:\n$stdout")
            return line.removePrefix("$name ").split(",").map { it.trim().toDouble() }.toDoubleArray()
        }

        fun value(name: String): Double = values(name).single()
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    fun compileAndRun(user: String, pluginOptions: Array<String> = emptyArray()): Result {
        val tempDir = Files.createTempDirectory("tlaloc-f64-test").toFile()
        try {
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<Message>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors() = collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
                    collected += Message(severity, message, location?.line ?: -1, location?.column ?: -1)
                }
            }
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                pluginClasspaths = pluginClasspath()
                if (pluginOptions.isNotEmpty()) this.pluginOptions = pluginOptions
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector, Services.EMPTY, args).code
            if (exitCode != 0) return Result(exitCode, collected, "")
            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(PrintStream(baos, true, Charsets.UTF_8))
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                Result(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                val cause = t.cause ?: t
                Result(
                    2,
                    collected + Message(
                        CompilerMessageSeverity.ERROR,
                        "RUN FAILURE: $cause\n" + cause.stackTrace.take(15).joinToString("\n") { "    at $it" },
                        -1,
                        -1,
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

    /** Compiles and runs [user], failing the test with every compiler message if either step fails. */
    fun run(user: String, pluginOptions: Array<String> = emptyArray()): Result {
        val r = compileAndRun(user, pluginOptions)
        assertEquals(0, r.exitCode, "compile/run failed:\n${r.describe()}")
        assertTrue(
            r.messages.none { it.severity == CompilerMessageSeverity.WARNING || it.severity == CompilerMessageSeverity.STRONG_WARNING },
            "unexpected warnings:\n${r.describe()}",
        )
        return r
    }

    /** Central differences of [f] at [x], step [h] per coordinate. */
    fun fd(x: DoubleArray, h: Double, f: (DoubleArray) -> Double): DoubleArray = DoubleArray(x.size) { i ->
        (f(x.copyOf().also { it[i] += h }) - f(x.copyOf().also { it[i] -= h })) / (2 * h)
    }

    /** Every entry of [got] within [relTol] of the largest magnitude in [want]. */
    fun assertClose(want: DoubleArray, got: DoubleArray, relTol: Double, what: String) {
        assertEquals(want.size, got.size, "$what: size (got ${got.toList()})")
        val scale = max(1e-300, want.maxOf { abs(it) })
        for (i in want.indices) {
            assertTrue(
                abs(want[i] - got[i]) <= relTol * scale,
                "$what[$i] = ${got[i]}, want ${want[i]} (difference ${abs(want[i] - got[i]) / scale} of the " +
                    "largest entry, tolerance $relTol)",
            )
        }
    }

    /** Literal DoubleArray source for a Kotlin snippet, every digit kept. */
    fun lit(a: DoubleArray): String = a.joinToString(", ") { it.toString() }
}
