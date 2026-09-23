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
 * §0.4.193 — first multi-param plugin gradient test. Exercises the K2 plugin's
 * 2-arg `grad { (a, b) -> ... }` Pair-return path on a SQUARE MATMUL where both
 * params share a single shape `Rank2<Sym, Sym>`. Closes a coverage gap between
 * §0.4.189's single-param MATMUL test and the eventual rectangular variant
 * (R ≠ K ≠ C) gated on Phase 0c-rectangular slice 3 (op-result IrType derivation
 * + BROADCAST template selection).
 *
 * Primal: `grad { (a, b) -> (a matmul b).sum().toFloat() }` returning Pair(dA, dB).
 * For A = B = [[1,2],[3,4]] (same shape, distinct DTensors):
 *   - ∂Σ(AB)/∂A = ones(2,2) · B^T → row sum of B replicated across rows.
 *     B^T = [[1,3],[2,4]], 1·B^T = [[3, 7], [3, 7]] (sum of each B-row, broadcast over A's rows).
 *     Wait — Σ(AB) = Σ_ij Σ_k a_ik b_kj. ∂/∂a_ik = Σ_j b_kj. For B=[[1,2],[3,4]]:
 *       ∂a_00 = Σ_j b_0j = 1+2 = 3
 *       ∂a_01 = Σ_j b_1j = 3+4 = 7
 *       ∂a_10 = ∂a_00 = 3
 *       ∂a_11 = ∂a_01 = 7
 *     So ∂A = [[3, 7], [3, 7]].
 *   - ∂Σ(AB)/∂B = A^T · ones(2,2) → column sum of A replicated across cols.
 *     ∂/∂b_kj = Σ_i a_ik. For A=[[1,2],[3,4]]:
 *       ∂b_00 = Σ_i a_i0 = 1+3 = 4
 *       ∂b_10 = Σ_i a_i1 = 2+4 = 6
 *       ∂b_01 = ∂b_00 = 4
 *       ∂b_11 = ∂b_10 = 6
 *     So ∂B = [[4, 4], [6, 6]].
 */
class Rank2MatmulTwoParamGradientTest {

    @Test
    fun `2-arg grad of sum of A matmul B with shared shape matches analytic`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.matmul
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Sym>, F32>, b: DTensor<Rank2<Sym, Sym>, F32> ->
                    (a matmul b).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                val B = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                val (dA, dB) = g(A, B)
                val flatA = dA.hostF32()
                val flatB = dB.hostF32()
                println(flatA.size)
                for (v in flatA) print("${'$'}v ")
                println()
                println(flatB.size)
                for (v in flatB) print("${'$'}v ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB_2ARG_SQUARE_2x2, src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected end-to-end 2-arg MATMUL gradient to lower. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "expected 4 stdout lines (sizeA + valuesA + sizeB + valuesB), got: ${result.stdout}")
        assertEquals("4", lines[0])
        val valuesA = lines[1].trim().split(" ").map { it.toFloat() }
        assertEquals("4", lines[2])
        val valuesB = lines[3].trim().split(" ").map { it.toFloat() }

        val expectedA = floatArrayOf(3.0f, 7.0f, 3.0f, 7.0f)
        val expectedB = floatArrayOf(4.0f, 4.0f, 6.0f, 6.0f)
        for ((i, v) in valuesA.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dA slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(
                abs(v - expectedA[i]) < 1e-3f,
                "dA slot $i = $v expected ${expectedA[i]}",
            )
        }
        for ((i, v) in valuesB.withIndex()) {
            assertTrue(
                abs(v + 1.0f) > 1e-6f,
                "dB slot $i = $v matches broken-stub sentinel",
            )
            assertTrue(
                abs(v - expectedB[i]) < 1e-3f,
                "dB slot $i = $v expected ${expectedB[i]}",
            )
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
        val tempDir = Files.createTempDirectory("tlaloc-rank2-matmul-2arg-test").toFile()
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
        private val AUTOGRAD_STUB_2ARG_SQUARE_2x2 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, F32>) ->
                        Pair<DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                ) }
        """.trimIndent()
    }
}
