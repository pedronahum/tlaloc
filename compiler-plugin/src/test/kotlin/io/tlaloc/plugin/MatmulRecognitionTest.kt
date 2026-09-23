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
 * §0.4.187 — Phase 0c slice (c): plugin recognition for `infix fun matmul` (in
 * `:core/ops/HostOps.kt`). With slices (a) + (b) shipped (§0.4.185 + §0.4.186) the
 * substrate accepts rank-2 inputs end-to-end; this test confirms the FIR-side
 * recognition wires through to the `OpKind.MATMUL` op kind in dxir.
 *
 * Test shape: `grad { a: DTensor<Rank2<R, R>, F32> -> val z = a matmul a; 0.0f }`.
 * The matmul is dead-code from the loss's perspective (the lambda returns 0.0f
 * regardless), so the gradient w.r.t. `a` is zero — same as §0.4.186's pattern.
 * The point is to confirm the FIR-side BINARY_OP_MAP[matmul] lookup fires; the
 * "saw handoff" warning's dxir dump should contain `matmul(...)`.
 *
 * End-to-end gradient correctness for ACTIVE matmul (where the gradient flows
 * through MATMUL via MatmulRule's TRANSPOSE + MATMUL emissions) is gated on a
 * DTensor → Float bridge in the K2 plugin (so the lambda body can produce Float
 * from a matmul result). That's a separate slice; this test confirms only the
 * recognition.
 */
class MatmulRecognitionTest {

    @Test
    fun `matmul in lambda body lowers to dxir MATMUL op`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.matmul
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> ->
                    val z = a matmul a
                    0.0f
                }
                val input = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                val gradOut = g(input)
                val flat = gradOut.hostF32()
                println(flat.size)
                for (v in flat) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK2x2, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")

        // Assert the FIR-side dxir contains a `matmul(...)` op — confirms BINARY_OP_MAP
        // wired through.
        // §0.4.499 — the handoff dump is an INFO now (and opt-in; the harness turns
        // `dumpLoweredIr` on). It used to be an unconditional WARNING in every
        // consumer's build log.
        val handoffWarning = result.messages
            .filter { it.severity == CompilerMessageSeverity.INFO }
            .firstOrNull { "saw handoff" in it.message }
        assertTrue(
            handoffWarning != null,
            "no 'saw handoff' dump — FirLambdaToDxirLowering didn't emit dxir dump",
        )
        assertTrue(
            handoffWarning!!.message.contains("matmul("),
            "dxir dump didn't show matmul(...) op; got first 30 lines:\n" +
                handoffWarning.message.lines().take(30).joinToString("\n"),
        )

        // End-to-end: gradient is rank-2 zero tensor (since lambda returns 0.0f
        // independent of `a`). 4 elements (2×2 input).
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected rank-2 zero-grad to lower per §0.4.186. " +
                "Warnings: ${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )
        val lines = result.stdout.trim().lines()
        assertEquals(2, lines.size, "expected 2 stdout lines (size + values), got: ${result.stdout}")
        assertEquals("4", lines[0], "rank-2 grad output should have size 4 (2×2)")
        val values = lines[1].trim().split(" ").map { it.toFloat() }
        assertEquals(4, values.size)
        for ((i, v) in values.withIndex()) {
            assertTrue(abs(v) < 1e-6f, "slot $i = $v expected 0.0 (grad of constant body)")
            assertTrue(abs(v + 1.0f) > 1e-6f, "slot $i = $v matches broken-stub sentinel")
        }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-matmul-test").toFile()
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
                // §0.4.499 — this harness READS the lowered-dxir dump, which is off
                // by default now; and (where listed) it exercises the pre-alpha
                // tape-fallback path, which is a compile error by default.
                pluginOptions = arrayOf(
                    "plugin:io.tlaloc.plugin:dumpLoweredIr=true",
                )
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
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        private val AUTOGRAD_STUB_BROKEN_RANK2x2 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Sym>, F32>) -> DTensor<Rank2<Sym, Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32) }
        """.trimIndent()
    }
}
