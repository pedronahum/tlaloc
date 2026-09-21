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
 * §0.4.446 — the ONE-ENGINE claim made executable. After the runtime value-tape's
 * deletion, the Tracer-convenience API (`grad { x: Tracer<...> -> ... }`, trace at
 * runtime) and the `grad {}` intrinsic (`grad { x: DTensor<...> -> ... }`, rewrite
 * at compile time) are two ENTRY POINTS into the same engine: capture →
 * `DxirReverseTransform` over `VjpRegistry` → execution. This test compiles ONE
 * program through the real K2 plugin with the REAL `:autograd` on the classpath
 * (no stub — the [TlalocPluginTracerFallbackTest] harness), writes the same
 * mathematical function on both surfaces, and asserts the gradients are
 * IDENTICAL — `==` on every float, no tolerance. Any epsilon here would mean two
 * engines again.
 *
 * Coverage: a 1-arg elementwise chain with a RELU kink (`(relu(x)·x + x).sum()`,
 * mixed-sign input so the kink is exercised on both sides) and a 2-arg MATMUL
 * composite via `grad2`. The intrinsic side must actually LOWER (no fallback
 * warning) — otherwise the kept original call would throw `pluginMissing` and
 * the run would fail loudly, so a green run also certifies which route ran.
 */
class OneEngineParityTest {

    @Test
    fun `Tracer route and intrinsic route produce identical gradients`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.Tracer
            import io.tlaloc.autograd.matmul
            import io.tlaloc.autograd.plus
            import io.tlaloc.autograd.relu
            import io.tlaloc.autograd.sum
            import io.tlaloc.autograd.times
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.matmul
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.relu
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, arr: FloatArray) {
                println(name)
                println(arr.joinToString(" ") { it.toRawBits().toString() })
            }
            fun main() {
                // f(x) = sum(relu(x)*x + x) — same math, two surfaces.
                val gi = grad { x: DTensor<Rank1<Sym>, F32> ->
                    (x.relu() * x + x).sum().toFloat()
                }
                val gt = grad { x: Tracer<Rank1<Sym>> ->
                    (x.relu() * x + x).sum()
                }
                val xIn = floatArrayOf(0.5f, -1.25f, 2.0f, -3.5f, 0.0f)
                dump("i1", gi(Tensors.f32Vector<Sym>(xIn)).hostF32())
                dump("t1", gt(Tensors.f32Vector<Sym>(xIn)).hostF32())

                // g(a, b) = sum(a @ b) — the 2-arg surfaces.
                val g2i = grad2 { a: DTensor<Rank2<Sym, Sym>, F32>, b: DTensor<Rank2<Sym, Sym>, F32> ->
                    (a matmul b).sum().toFloat()
                }
                val g2t = grad2 { a: Tracer<Rank2<Sym, Sym>>, b: Tracer<Rank2<Sym, Sym>> ->
                    (a matmul b).sum()
                }
                val aIn = floatArrayOf(1f, 2f, 3f, 4f)
                val bIn = floatArrayOf(0.5f, -6f, 7f, 0.125f)
                val (dAi, dBi) = g2i(
                    Tensors.f32Matrix<Sym, Sym>(2, 2, aIn),
                    Tensors.f32Matrix<Sym, Sym>(2, 2, bIn),
                )
                val (dAt, dBt) = g2t(
                    Tensors.f32Matrix<Sym, Sym>(2, 2, aIn),
                    Tensors.f32Matrix<Sym, Sym>(2, 2, bIn),
                )
                dump("i2a", dAi.hostF32()); dump("i2b", dBi.hostF32())
                dump("t2a", dAt.hostF32()); dump("t2b", dBt.hostF32())
            }
        """.trimIndent()
        val result = compileAndRun(src)
        assertEquals(
            0,
            result.exitCode,
            "compile/run failed:\n${result.messages.joinToString("\n") { "[${it.severity}] ${it.message}" }}",
        )
        // The intrinsic side must have LOWERED — a fallback would leave the
        // DTensor-lambda call as the pluginMissing thrower, and the run above
        // would have exited non-zero; still, pin the absence of the warning so
        // a future silent-success fallback cannot sneak past.
        // The Tracer-typed lambdas are SUPPOSED to be refused by the plugin
        // ("unsupported type io.tlaloc.autograd.Tracer" — the §0.4.424 dual-name
        // design: they trace at runtime instead). Only a fallback of the
        // DTensor-typed intrinsic lambdas breaks the pin.
        val fellBack = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING &&
                ("kept original call" in it.message ||
                    ("could not lower lambda" in it.message &&
                        "unsupported type io.tlaloc.autograd.Tracer" !in it.message))
        }
        assertTrue(
            !fellBack,
            "intrinsic side fell back — the parity pin must compare the compile-time route, " +
                "not pluginMissing. Warnings:\n${
                    result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                        .joinToString("\n--\n") { it.message }
                }",
        )
        val lines = result.stdout.trim().lines()
        assertEquals(12, lines.size, "expected 12 stdout lines, got: ${result.stdout}")
        val sections = (lines.indices step 2).associate { i ->
            lines[i].trim() to lines[i + 1].trim().split(" ").map { Float.fromBits(it.toInt()) }
        }
        // IDENTICAL — raw-bit equality, not a tolerance. One engine, one answer.
        assertEquals(
            sections.getValue("i1").map { it.toRawBits() },
            sections.getValue("t1").map { it.toRawBits() },
            "grad: intrinsic ${sections.getValue("i1")} vs Tracer ${sections.getValue("t1")}",
        )
        assertEquals(
            sections.getValue("i2a").map { it.toRawBits() },
            sections.getValue("t2a").map { it.toRawBits() },
            "grad2 dA: intrinsic ${sections.getValue("i2a")} vs Tracer ${sections.getValue("t2a")}",
        )
        assertEquals(
            sections.getValue("i2b").map { it.toRawBits() },
            sections.getValue("t2b").map { it.toRawBits() },
            "grad2 dB: intrinsic ${sections.getValue("i2b")} vs Tracer ${sections.getValue("t2b")}",
        )
        // Sanity: the relu-kink gradient really is the analytic 2x+1 / 1 split,
        // so the pin isn't vacuously comparing two wrong answers.
        val want = floatArrayOf(2f, 1f, 5f, 1f, 1f)  // x>0: 2x+1; x<=0: 1
        val t1 = sections.getValue("t1")
        for (i in want.indices) {
            assertTrue(
                t1[i] == want[i],
                "analytic anchor: d[$i] expected ${want[i]}, got ${t1[i]}",
            )
        }
    }

    // --- test harness (the no-stub variant: real :autograd on the classpath) ---

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private fun compileAndRun(user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-one-engine-parity").toFile()
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
                mainKt.getMethod("main").invoke(null)
            } catch (t: Throwable) {
                return RunResult(
                    2,
                    collected + CompileMessage(
                        CompilerMessageSeverity.ERROR,
                        "runtime failure: ${t.cause ?: t}",
                    ),
                    baos.toString(),
                )
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
