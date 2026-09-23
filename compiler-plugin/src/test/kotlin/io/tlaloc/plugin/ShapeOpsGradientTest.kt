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
 * §0.4.367 — Phase A2a E2E: the RESHAPE family (`reshape(dims)` /
 * `unsqueeze` / `squeeze`) and permutation `transpose` differentiate
 * end-to-end through the K2 plugin. Three lambdas over a, b ∈ ℝ^{2×2}:
 *
 *  g1 = ∇ Σ (aᵀ ⊙ b^{id})            → dA = Bᵀ, dB = Aᵀ
 *       (`transpose(1,0)` + the identity perm `transpose(0,1)` — the
 *        identity forwards, the swap takes the classic rank-2 path)
 *  g2 = ∇ Σ (a.reshape(4,1) ⊙ b.reshape(4,1)) → dA = B, dB = A
 *       (concrete-literal reshape; the adjoint reshapes back to the
 *        SENTINEL param shape via matched runtime dims)
 *  g3 = ∇ Σ (a.unsqueeze(2).squeeze(2) ⊙ b.unsqueeze(0).squeeze(0))
 *                                      → dA = B, dB = A
 *       (rank-3 round-trips; adjoints exercise both squeezeAxes and
 *        unsqueezeAxes synthesis arms at trailing AND leading axes)
 */
class ShapeOpsGradientTest {

    @Test
    fun `grad through transpose reshape squeeze unsqueeze`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.reshape
            import io.tlaloc.core.ops.squeeze
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            import io.tlaloc.core.ops.transpose
            import io.tlaloc.core.ops.unsqueeze
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    (a.transpose(1, 0) * b.transpose(0, 1)).sum().toFloat()
                }
                val g2 = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    (a.reshape(4, 1) * b.reshape(4, 1)).sum().toFloat()
                }
                val g3 = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    (a.unsqueeze(2).squeeze(2) * b.unsqueeze(0).squeeze(0)).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1.0f, 3.0f, 5.0f, 2.0f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(0.5f, 1.0f, 2.0f, 0.25f))
                val (d1a, d1b) = g1(A, B)
                dump("g1a", d1a); dump("g1b", d1b)
                val (d2a, d2b) = g2(A, B)
                dump("g2a", d2a); dump("g2b", d2b)
                val (d3a, d3b) = g3(A, B)
                dump("g3a", d3a); dump("g3b", d3b)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected all three shape-op gradients to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // A = [[1,3],[5,2]], B = [[0.5,1],[2,0.25]]:
        // g1: loss = Σ_{ij} a[j,i]·b[i,j] ⇒ dA = Bᵀ, dB = Aᵀ.
        val want = mapOf(
            "g1a" to listOf(0.5f, 2.0f, 1.0f, 0.25f),   // Bᵀ
            "g1b" to listOf(1.0f, 5.0f, 3.0f, 2.0f),    // Aᵀ
            "g2a" to listOf(0.5f, 1.0f, 2.0f, 0.25f),   // B
            "g2b" to listOf(1.0f, 3.0f, 5.0f, 2.0f),    // A
            "g3a" to listOf(0.5f, 1.0f, 2.0f, 0.25f),   // B
            "g3b" to listOf(1.0f, 3.0f, 5.0f, 2.0f),    // A
        )
        val lines = result.stdout.trim().lines()
        assertEquals(12, lines.size, "expected 12 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name'")
            assertEquals(4, values.size, "$name size")
            assertTrue(
                values.any { it != -1.0f },
                "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                    result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
            )
            for (j in 0 until 4) {
                assertTrue(
                    abs(values[j] - expect[j]) < 1e-5f,
                    "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                )
            }
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
        val tempDir = Files.createTempDirectory("tlaloc-shapeops-test").toFile()
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
        private val AUTOGRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>) ->
                        Pair<DTensor<Rank2<Sym, Lit<Int>>, F32>, DTensor<Rank2<Sym, Lit<Int>>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                ) }
        """.trimIndent()
    }
}
