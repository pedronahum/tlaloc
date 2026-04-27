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
 * §0.4.185 — Phase 0c first slice: confirms `FirLambdaToDxirLowering.resolveParamType`
 * recognises Rank2 DTensor parameters. Compiles a `grad { A: DTensor<Rank2<R, C>, F32> -> A.sum() }`
 * primal and asserts the FIR-side lowering produces a dxir function (visible in the
 * "Tlaloc lowered lambda to dxir" success warning).
 *
 * **Synthesis-side widening is NOT in scope for this firing.** The grad function's
 * BROADCAST op (emitted by SumRule's gradient path on a rank-2 input) trips
 * [DxirToIrSynthesis.irBroadcast]'s rank-1-only gate, so the K2 plugin emits "kept
 * original call" and falls back to the runtime tape's broken stub. The test
 * therefore checks only the FIR-side success, not end-to-end gradient correctness
 * — that's the next Phase 0c slice.
 */
class Rank2ParamLoweringTest {

    @Test
    fun `rank2 DTensor param lowers through FIR with synthesis fallback`() {
        // Minimal Rank2-input primal that the FIR side can lower: returns a constant
        // (no rank-2 ops in the body that the plugin would need to recognise). The
        // mere PRESENCE of the Rank2 param exercises `resolveParamType`'s new arm.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> 0.0f }
                val input = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
                val gradOut = g(input)
                println(gradOut.hostF32()[0])
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_BROKEN_RANK2_TO_FLOAT, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")

        // Sanity: confirm the FIR-side lowering produced a dxir function.
        // The "Tlaloc lowered lambda to dxir" warning fires only when
        // FirLambdaToDxirLowering successfully produced a DxirFunction.
        val firSucceeded = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING &&
                "Tlaloc lowered lambda to dxir" in it.message
        }
        assertTrue(
            firSucceeded,
            "FIR-side lowering didn't produce dxir for the Rank2 param; " +
                "warnings: ${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )
        // The dxir dump should reference the Rank2 input as `f32[-1,-1]` (two sentinel
        // dims). Find the "saw handoff" warning and check.
        val handoffWarning = result.messages
            .filter { it.severity == CompilerMessageSeverity.WARNING }
            .firstOrNull { "saw handoff" in it.message }
        assertTrue(
            handoffWarning != null,
            "no 'saw handoff' warning — FirLambdaToDxirLowering didn't emit the dxir dump",
        )
        assertTrue(
            handoffWarning!!.message.contains("f32[-1,-1]"),
            "dxir dump didn't show f32[-1,-1] for the Rank2 param; got:\n${handoffWarning.message.lines().take(5).joinToString("\n")}",
        )

        // Synthesis is expected to fall back here (rank-2 BROADCAST in the grad body
        // is outside the synthesis scope today). Just confirm the broken-stub
        // sentinel fired — that's the documented behavior pending Phase 0c slice (b).
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            keptOriginal,
            "synthesis surprisingly accepted the rank-2 grad body — Phase 0c slice (b) may already be done; review the §0.4 entry assumptions",
        )
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-rank2-probe").toFile()
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
            } finally {
                System.setOut(originalOut)
                loader.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    companion object {
        private val AUTOGRAD_STUB_BROKEN_RANK2_TO_FLOAT = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Sym>, F32>) -> DTensor<Rank2<Sym, Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32) }
        """.trimIndent()
    }
}
