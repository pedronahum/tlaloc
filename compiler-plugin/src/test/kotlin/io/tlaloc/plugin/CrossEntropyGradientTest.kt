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
import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.370 — Phase A3b E2E: `crossEntropyLoss` and `nllLoss` differentiate
 * end-to-end through the K2 plugin. Both compose onto existing fully-ruled ops
 * (NEG/SUM/MUL/LOG/SOFTMAX) — no new VjpRule — so the point is that the FIR
 * composition synthesises with no fallback and matches the analytic gradients.
 *
 *  g1 = ∇ crossEntropyLoss(a, b)  → da = softmax(a)·Σb − b (per row), db = −logSoftmax(a)
 *  g2 = ∇ nllLoss(a, b)           → da = −b, db = −a
 */
class CrossEntropyGradientTest {

    @Test
    fun `grad through crossEntropyLoss and nllLoss`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.crossEntropyLoss
            import io.tlaloc.core.ops.nllLoss
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    crossEntropyLoss(a, b).toFloat()
                }
                val g2 = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    nllLoss(a, b).toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1.0f, 3.0f, 5.0f, 2.0f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(0.5f, 1.0f, 2.0f, 0.25f))
                val (d1a, d1b) = g1(A, B)
                dump("g1a", d1a); dump("g1b", d1b)
                val (d2a, d2b) = g2(A, B)
                dump("g2a", d2a); dump("g2b", d2b)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected crossEntropyLoss + nllLoss gradients to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val a = floatArrayOf(1f, 3f, 5f, 2f)
        val b = floatArrayOf(0.5f, 1f, 2f, 0.25f)
        val want = references(a, b)

        val lines = result.stdout.trim().lines()
        assertEquals(8, lines.size, "expected 8 stdout lines, got: ${result.stdout}")
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
                    abs(values[j] - expect[j]) < 1e-4f,
                    "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                )
            }
        }
    }

    /** Analytic references for CE/NLL over axis 1 of a 2×2 matrix. */
    private fun references(a: FloatArray, b: FloatArray): Map<String, List<Float>> {
        val g1a = FloatArray(4); val g1b = FloatArray(4)
        val g2a = FloatArray(4); val g2b = FloatArray(4)
        for (row in 0 until 2) {
            val base = row * 2
            val mx = maxOf(a[base].toDouble(), a[base + 1].toDouble())
            val e = doubleArrayOf(exp(a[base] - mx), exp(a[base + 1] - mx))
            val s = e[0] + e[1]
            val y = doubleArrayOf(e[0] / s, e[1] / s)
            val logS = ln(s)
            val bSum = b[base].toDouble() + b[base + 1]
            for (j in 0 until 2) {
                // CE: da_j = y_j·Σb − b_j; db_j = −logSoftmax_j.
                g1a[base + j] = (y[j] * bSum - b[base + j]).toFloat()
                g1b[base + j] = (-(a[base + j] - mx - logS)).toFloat()
                // NLL(a as logProbs, b as oneHot): da_j = −b_j; db_j = −a_j.
                g2a[base + j] = -b[base + j]
                g2b[base + j] = -a[base + j]
            }
        }
        return mapOf(
            "g1a" to g1a.toList(), "g1b" to g1b.toList(),
            "g2a" to g2a.toList(), "g2b" to g2b.toList(),
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
        val tempDir = Files.createTempDirectory("tlaloc-ce-test").toFile()
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
