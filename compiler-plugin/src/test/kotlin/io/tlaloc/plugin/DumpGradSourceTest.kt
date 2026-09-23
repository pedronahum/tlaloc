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
import kotlin.test.assertTrue

/**
 * §0.4.450 — the readable-reverse dump E2E (docs/AD_SINGLE_ENGINE_AUDIT.md,
 * surface 2 — the north star's compile-time half): with the plugin CLI option
 * `-P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>` on, every `grad {}`
 * lambda the plugin successfully synthesises also dumps its
 * REVERSE-TRANSFORMED gradient dxir rendered as Kotlin source (§0.4.449's
 * `toKotlinSource`) — an INFO message headed by the lambda's source location,
 * plus a `.kt` file named after that location. The dump happens at the dxir
 * level BEFORE synthesis, so the gradient the user reads is the SAME function
 * the synthesis compiles.
 *
 * The STRONG pin (test 1): compile a scalar `grad {}` program with the option
 * on, extract the dumped `.kt` file, compile IT standalone with the in-process
 * [K2JVMCompiler] (NO plugin — the dumped source is plain user Kotlin over
 * `:core`), run it, and match the plugin-compiled gradient's output
 * RAW-BIT-IDENTICAL (`Float.toRawBits`, no epsilon). Both routes evaluate the
 * same dxir — the synthesised bytecode with Float primitives, the printed
 * source with the §0.4.447 bit-equal host twins — so identity is free by
 * construction, and the pin keeps it that way.
 *
 * The honest-refusal pin (test 2): a TENSOR `grad {}` lambda synthesises fine
 * but its dxir carries the -1 SENTINEL dims, which the §0.4.449 renderer
 * refuses by design (a ranked-literal rendering would bake sentinel-derived
 * garbage — the house landmine). The dump must say so LOUDLY (a SKIPPED
 * message naming the sentinel refusal), write NO file, and never guess.
 */
class DumpGradSourceTest {

    @Test
    fun `scalar grad dump appears, names the location, and the dumped source runs bit-identical`() {
        val dumpDir = Files.createTempDirectory("tlaloc-grad-dump").toFile()
        try {
            val src = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x * x + 2.5f * x }
                    println(g(1.75f).toRawBits())
                    println(g(-0.5f).toRawBits())
                }
            """.trimIndent()
            val result = compileAndRun(SCALAR_STUB, src, dumpDir)
            assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")
            assertTrue(
                result.messages.none {
                    "kept original call" in it.message
                },
                "the scalar lambda must synthesise (no fallback); warnings:\n" +
                    result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                        .joinToString("\n--\n") { it.message },
            )

            // 1. The INFO dump appears, headed by the lambda's source location.
            val dump = result.messages.singleOrNull {
                it.severity == CompilerMessageSeverity.INFO && "Tlaloc grad source for 'grad'" in it.message
            } ?: error(
                "no grad-source INFO dump emitted; messages:\n" +
                    result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
            )
            assertTrue(
                "Main.kt:3:" in dump.message,
                "the dump header must carry the lambda's file:line location; got:\n${dump.message}",
            )

            // 2. The dump reads as host-twin Kotlin for the known lambda: ranked
            //    scalar types, Tensors.* consts, the DTensor operators — and it is
            //    a complete named function.
            for (needle in listOf(
                "fun ", "DTensor<ScalarShape, F32>", "Tensors.f32Scalar(", " * ", " + ",
            )) {
                assertTrue(
                    needle in dump.message,
                    "dump must contain the host-twin spelling '$needle'; got:\n${dump.message}",
                )
            }

            // 3. The dir form wrote the same rendering as a location-named .kt file.
            val dumped = dumpDir.listFiles { f -> f.name.endsWith(".kt") }?.toList().orEmpty()
            assertEquals(1, dumped.size, "expected exactly one dumped .kt file, got $dumped")
            assertTrue(
                dumped[0].name.startsWith("Main_kt_3_"),
                "the dump file is named after the lambda's location; got ${dumped[0].name}",
            )
            val dumpedSource = dumped[0].readText()
            assertTrue(
                dumpedSource.trimEnd() in dump.message,
                "the .kt file must carry the same source as the INFO dump",
            )

            // 4. THE STRONG PIN — the dumped source compiles standalone (no
            //    plugin), runs, and matches the plugin-compiled gradient's own
            //    outputs raw-bit-for-bit.
            val pluginBits = result.stdout.trim().lines().map { it.trim().toInt() }
            assertEquals(2, pluginBits.size, "plugin-compiled program must print 2 raw-bit lines")
            assertTrue(
                pluginBits.none { it == (-1.0f).toRawBits() },
                "stub sentinel returned — the plugin rewrite never fired",
            )

            val fnName = Regex("""fun (\w+)\(""").find(dumpedSource)?.groupValues?.get(1)
                ?: error("dumped source carries no function declaration:\n$dumpedSource")
            val driver = """
                import io.tlaloc.core.*
                import io.tlaloc.core.ops.*
                fun main() {
                    println($fnName(Tensors.f32Scalar(1.75f)).hostF32()[0].toRawBits())
                    println($fnName(Tensors.f32Scalar(-0.5f)).hostF32()[0].toRawBits())
                }
            """.trimIndent()
            val standalone = compilePlainAndRun(dumpedSource, driver)
            assertEquals(
                0, standalone.exitCode,
                "dumped source failed to compile/run standalone:\n${standalone.messages}" +
                    "\n--- dumped source ---\n$dumpedSource",
            )
            val standaloneBits = standalone.stdout.trim().lines().map { it.trim().toInt() }
            assertEquals(
                pluginBits, standaloneBits,
                "the dumped gradient must be RAW-BIT-IDENTICAL to the plugin-compiled " +
                    "gradient (dumped=${standaloneBits.map { Float.fromBits(it) }} " +
                    "plugin=${pluginBits.map { Float.fromBits(it) }})",
            )
        } finally {
            dumpDir.deleteRecursively()
        }
    }

    @Test
    fun `a sentinel-dim tensor grad dumps a loud named refusal and writes no file`() {
        val dumpDir = Files.createTempDirectory("tlaloc-grad-dump-refuse").toFile()
        try {
            val src = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Rank1
                import io.tlaloc.core.Sym
                import io.tlaloc.core.Tensors
                import io.tlaloc.core.hostF32
                import io.tlaloc.core.ops.sum
                import io.tlaloc.core.ops.times
                import io.tlaloc.core.ops.toFloat
                fun main() {
                    val g = grad { x: DTensor<Rank1<Sym>, F32> -> (x * x).sum().toFloat() }
                    val dx = g(Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)))
                    println(dx.hostF32().toList())
                }
            """.trimIndent()
            val result = compileAndRun(RANK1_STUB, src, dumpDir)
            assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")
            assertTrue(
                result.messages.none {
                    "kept original call" in it.message
                },
                "the tensor lambda must still synthesise (the dump is a window, not a gate)",
            )
            assertTrue(
                "[-1.0, -1.0, -1.0]" !in result.stdout,
                "stub sentinel returned — the plugin rewrite never fired",
            )
            val skipped = result.messages.filter {
                it.severity == CompilerMessageSeverity.INFO &&
                    "Tlaloc grad source dump SKIPPED for 'grad'" in it.message
            }
            assertEquals(
                1, skipped.size,
                "the sentinel gradient must dump exactly one SKIPPED refusal; messages:\n" +
                    result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
            )
            assertTrue(
                "sentinel" in skipped[0].message,
                "the refusal must NAME the sentinel-dim reason; got:\n${skipped[0].message}",
            )
            assertTrue(
                result.messages.none {
                    it.severity == CompilerMessageSeverity.INFO && "Tlaloc grad source for 'grad'" in it.message
                },
                "no source dump may accompany a refusal — never plausible wrong source",
            )
            val dumped = dumpDir.listFiles { f -> f.name.endsWith(".kt") }?.toList().orEmpty()
            assertTrue(dumped.isEmpty(), "a refused rendering must write no file; got $dumped")
        } finally {
            dumpDir.deleteRecursively()
        }
    }

    // --------- Harness (the RngGradientTest pattern + plugin CLI options) ---------

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun collector(into: MutableList<CompileMessage>): MessageCollector = object : MessageCollector {
        override fun clear() {}
        override fun hasErrors() = into.any { it.severity == CompilerMessageSeverity.ERROR }
        override fun report(
            severity: CompilerMessageSeverity,
            message: String,
            location: CompilerMessageSourceLocation?,
        ) {
            into += CompileMessage(severity, message)
        }
    }

    private fun runMain(outDir: File, collected: List<CompileMessage>): RunResult {
        val originalOut = System.out
        val baos = ByteArrayOutputStream()
        val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
        return try {
            System.setOut(PrintStream(baos, true, Charsets.UTF_8))
            loader.loadClass("MainKt").getMethod("main").invoke(null)
            RunResult(0, collected, baos.toString(Charsets.UTF_8))
        } catch (t: Throwable) {
            RunResult(
                2,
                collected + CompileMessage(CompilerMessageSeverity.ERROR, "RUN FAILURE: ${t.cause ?: t}"),
                baos.toString(Charsets.UTF_8),
            )
        } finally {
            System.setOut(originalOut)
            loader.close()
        }
    }

    /** Compile [user] + [stub] WITH the plugin and the dump-dir option; run MainKt. */
    private fun compileAndRun(stub: String, user: String, dumpDir: File): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-dump-test").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                pluginClasspaths = pluginClasspath()
                pluginOptions = arrayOf(
                    "plugin:io.tlaloc.plugin:dumpGradSourceDir=${dumpDir.absolutePath}",
                )
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector(collected), Services.EMPTY, args).code
            if (exitCode != 0) return RunResult(exitCode, collected, "")
            return runMain(outDir, collected)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    /** Compile [printed] + [driver] with NO plugin (plain user Kotlin over `:core`); run MainKt. */
    private fun compilePlainAndRun(printed: String, driver: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-dump-standalone").toFile()
        try {
            File(tempDir, "Printed.kt").writeText(printed)
            File(tempDir, "Main.kt").writeText(driver)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector(collected), Services.EMPTY, args).code
            if (exitCode != 0) return RunResult(exitCode, collected, "")
            return runMain(outDir, collected)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        /** Sentinel stub: matching gradient values prove the IR transform fired. */
        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()

        private val RANK1_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(
                f: (DTensor<Rank1<Sym>, F32>) -> Float,
            ): (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32) }
        """.trimIndent()
    }
}
