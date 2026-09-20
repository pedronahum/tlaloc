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
 * §0.4.420 — Phase E1c E2E: the `grad {}` sparse surface — the GNN-shaped
 * certification the whole ratified sparse arc builds to. `sparseMatmul(v,
 * colIdx, rowPtr, B)` differentiates end-to-end through the K2 plugin over a
 * FOUR-param lambda: values [nnz] F32 (differentiable), colIdx [nnz] +
 * rowPtr [N+1] I32 (the two-integer-param shape §0.4.419's ZEROS_LIKE
 * exists to admit — this lambda was structurally impossible before E1c-pre),
 * dense B [C, D] F32 (differentiable). The §0.4.418 SparseMatmulRule below
 * the surface emits the fused SDDMM values-adjoint and the
 * execution-time-transposed `Aᵀ · upstream`, which the synthesis lowers to
 * the host twins `sparseMatmulValuesAdjoint` / `sparseMatmulTransposed`.
 *
 * The CSR pattern carries the E1a/E1b landmines on purpose: an EMPTY row
 * (row 1), a skewed row (row 2 holds 3 of 5 entries), and an explicit STORED
 * ZERO at (2,2) — which must still RECEIVE a gradient (d_values[k] is a
 * property of the position, not the value). All data on the quarter-integer
 * grid so every contraction is exact in F32.
 *
 *  A [3,4] CSR: row0 = {(0,0)=1.5, (0,2)=-0.75}, row1 = {}, row2 =
 *  {(2,1)=2.25, (2,2)=0.0*, (2,3)=0.5}; B [4,2].
 *
 *  g1 = ∇ Σ A·B      → d_v[k] = Σ_j B[col(k), j]  (SDDMM, upstream ones)
 *                      d_B    = Aᵀ · 1           (column sums of values)
 *  g2 = ∇ Σ (A·B)²   → d_v[k] = Σ_j 2y[row(k), j]·B[col(k), j] — the body
 *                      must RECOMPUTE y = A·B (the plain-form irSparseMatmul
 *                      arm) before the SDDMM and the transposed product
 *  both: d_colIdx = I32 zeros [5], d_rowPtr = I32 zeros [4] — each at ITS
 *  OWN param's extent (nnz=5 ≠ N+1=4), the §0.4.419 addressing E2E.
 */
class SparseMatmulGradientTest {

    @Test
    fun `grad through sparseMatmul lowers with no fallback`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.I32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.hostI32
            import io.tlaloc.core.ops.sparseMatmul
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dumpF(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun dumpI(name: String, t: DTensor<*, I32>) {
                println(name)
                for (v in t.hostI32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { v: DTensor<Rank1<Sym>, F32>,
                                ci: DTensor<Rank1<Sym>, I32>,
                                rp: DTensor<Rank1<Sym>, I32>,
                                b: DTensor<Rank2<Sym, Sym>, F32> ->
                    sparseMatmul(v, ci, rp, b).sum().toFloat()
                }
                val g2 = grad { v: DTensor<Rank1<Sym>, F32>,
                                ci: DTensor<Rank1<Sym>, I32>,
                                rp: DTensor<Rank1<Sym>, I32>,
                                b: DTensor<Rank2<Sym, Sym>, F32> ->
                    val y = sparseMatmul(v, ci, rp, b)
                    (y * y).sum().toFloat()
                }
                val V = Tensors.f32Vector<Sym>(floatArrayOf(1.5f, -0.75f, 2.25f, 0.0f, 0.5f))
                val CI = Tensors.i32Vector<Sym>(intArrayOf(0, 2, 1, 2, 3))
                val RP = Tensors.i32Vector<Sym>(intArrayOf(0, 2, 2, 5))
                val B = Tensors.f32Matrix<Sym, Sym>(4, 2, floatArrayOf(
                    1.0f, -0.5f, 0.25f, 2.0f, -1.5f, 0.75f, 0.5f, 1.25f,
                ))
                val (d1v, d1ci, d1rp, d1b) = g1(V, CI, RP, B)
                dumpF("g1v", d1v); dumpI("g1ci", d1ci); dumpI("g1rp", d1rp); dumpF("g1b", d1b)
                val (d2v, d2ci, d2rp, d2b) = g2(V, CI, RP, B)
                dumpF("g2v", d2v); dumpI("g2ci", d2ci); dumpI("g2rp", d2rp); dumpF("g2b", d2b)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; the sparseMatmul gradients must lower (Phase E1c). " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // Hand oracle on the quarter-integer grid (exact in F32):
        //   y = A·B = [[2.625, -1.3125], [0, 0], [0.8125, 5.125]], 2y as below.
        val wantG1v = listOf(0.5f, -0.75f, 2.25f, -0.75f, 1.75f)
        val wantG1b = listOf(1.5f, 1.5f, 2.25f, 2.25f, -0.75f, -0.75f, 0.5f, 0.5f)
        val wantG2v = listOf(6.5625f, -9.84375f, 20.90625f, 5.25f, 13.625f)
        val wantG2b = listOf(7.875f, -3.9375f, 3.65625f, 23.0625f, -3.9375f, 1.96875f, 0.8125f, 5.125f)
        val wantF = mapOf("g1v" to wantG1v, "g1b" to wantG1b, "g2v" to wantG2v, "g2b" to wantG2b)
        val wantZeroSize = mapOf("g1ci" to 5, "g2ci" to 5, "g1rp" to 4, "g2rp" to 4)

        val lines = result.stdout.trim().lines()
        assertEquals(16, lines.size, "expected 16 stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            when (name) {
                "g1v", "g1b", "g2v", "g2b" -> {
                    val expect = wantF.getValue(name)
                    assertEquals(expect.size, values.size, "$name size")
                    assertTrue(
                        values.any { it != -1.0f },
                        "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                            result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
                    )
                    for (j in values.indices) {
                        assertTrue(
                            abs(values[j] - expect[j]) < 1e-5f,
                            "$name[$j] = ${values[j]}, want ${expect[j]}. Full stdout:\n${result.stdout}",
                        )
                    }
                }
                "g1ci", "g2ci", "g1rp", "g2rp" -> {
                    // Each structural zero at ITS OWN param's extent — nnz=5
                    // for colIdx, N+1=4 for rowPtr — the §0.4.419 addressing.
                    assertEquals(wantZeroSize.getValue(name), values.size, "$name size")
                    assertTrue(
                        values.all { it == 0f },
                        "$name: integer CSR component gradient must be the structural zero, got $values",
                    )
                }
                else -> error("unexpected section '$name'")
            }
        }
        // The explicit stored zero at k=3 (position (2,2), value 0.0) must
        // RECEIVE a gradient in both losses — it is in the pattern.
        val g1v = result.stdout.trim().lines()[1].trim().split(" ").map { it.toFloat() }
        assertTrue(g1v[3] != 0f, "the explicit stored zero must receive a gradient, got ${g1v[3]}")
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-sparse-matmul-test").toFile()
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
        /**
         * The 4-param stub: `(Rank1 F32, Rank1 I32, Rank1 I32, Rank2 F32) →
         * Float`, returning the real :autograd `Quadruple` (on the test
         * classpath since §0.4.134) of stub sentinels.
         */
        private val AUTOGRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.HostI32Storage
            import io.tlaloc.core.I32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(
                f: (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank2<Sym, Sym>, F32>) -> Float,
            ): (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank2<Sym, Sym>, F32>) ->
                    Quadruple<DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank1<Sym>, I32>, DTensor<Rank2<Sym, Sym>, F32>> =
                { _, _, _, _ -> Quadruple(
                    DTensor(HostF32Storage(FloatArray(5) { -1.0f }), intArrayOf(5), F32),
                    DTensor(HostI32Storage(IntArray(5) { -1 }), intArrayOf(5), I32),
                    DTensor(HostI32Storage(IntArray(4) { -1 }), intArrayOf(4), I32),
                    DTensor(HostF32Storage(FloatArray(8) { -1.0f }), intArrayOf(4, 2), F32),
                ) }
        """.trimIndent()
    }
}
