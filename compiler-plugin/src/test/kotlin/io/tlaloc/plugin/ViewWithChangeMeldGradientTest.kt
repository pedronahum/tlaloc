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
 * §0.4.428 — Phase A2 tails E2E: the DiffKT `view`/`withChange`/`meld` sugar
 * surfaces differentiate end-to-end through the K2 plugin. No new IR anywhere:
 *
 *   view(range)  = the §0.4.374 SLICE verbatim (PAD_TO adjoint);
 *   view(index)  = SLICE of the unit window + the axis-dropping RESHAPE;
 *   withChange   = x + PAD_TO(replacement − slice(x), template = x) — the
 *                  §0.4.399 PAD_TO ⇄ SLICE_AT closure used in a PRIMAL for the
 *                  first time: d_x = upstream with the window zeroed,
 *                  d_replacement = the upstream's window;
 *   meld         = flatten-per-operand + the binary-CONCAT fold (SLICE_LIKE
 *                  windows reshaped back by the RESHAPE adjoint).
 *
 * All oracles are analytic (routing is exact): quarter-integer grids, no FD.
 */
class ViewWithChangeMeldGradientTest {

    @Test
    fun `grad through view of a contiguous range`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.view
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    // a:[4,2] viewed at rows 1..2; b:[2,2] full-viewed so both
                    // erase to Shape and the elementwise product unifies.
                    (a.view(1..2, 0) * b.view(0..1, 0)).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(4, 2, floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        // da = b zero-padded into rows 1..2 of a; db = a's rows 1..2.
        assertGradients(
            compileAndRun(AUTOGRAD_STUB_R2_R2, src),
            mapOf(
                "da" to listOf(0f, 0f, 10f, 20f, 30f, 40f, 0f, 0f),
                "db" to listOf(2f, 3f, 4f, 5f),
            ),
        )
    }

    @Test
    fun `grad through single-index view drops the axis`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.flatten
            import io.tlaloc.core.ops.view
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    // view(1, 0) is rank-1 (the axis is DROPPED); b flattens to
                    // erase Rank1 so the product unifies.
                    (a.view(1, 0) * b.flatten()).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(3, 2, floatArrayOf(0.5f, 1f, 1.5f, 2f, 2.5f, 3f))
                val B = Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        // da = b routed into row 1 of a, zeros elsewhere; db = a's row 1.
        assertGradients(
            compileAndRun(AUTOGRAD_STUB_R2_R1, src),
            mapOf(
                "da" to listOf(0f, 0f, 10f, 20f, 0f, 0f),
                "db" to listOf(1.5f, 2f),
            ),
        )
    }

    @Test
    fun `grad through withChange of a row`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.withChange
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, r: DTensor<Rank1<Sym>, F32> ->
                    val y = a.withChange(1, 0, r)
                    (y * y).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(3, 2, floatArrayOf(0.5f, 1f, 2f, 3f, 4f, 5f))
                val R = Tensors.f32Vector<Sym>(floatArrayOf(10f, 20f))
                val (da, dr) = g(A, R)
                dump("da", da); dump("dr", dr)
            }
        """.trimIndent()
        // y = [[0.5,1],[10,20],[4,5]]; loss = Σ y². da = 2y OUTSIDE the
        // replaced row and exactly 0 inside it (a's row 1 never reaches the
        // loss); dr = 2y in the window = 2r.
        assertGradients(
            compileAndRun(AUTOGRAD_STUB_R2_R1, src),
            mapOf(
                "da" to listOf(1f, 2f, 0f, 0f, 8f, 10f),
                "dr" to listOf(20f, 40f),
            ),
        )
    }

    @Test
    fun `grad through withChange of a row range`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.withChange
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, r: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    val y = a.withChange(1..2, 0, r)
                    (y * y).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(4, 2, floatArrayOf(0.5f, 1f, 1.5f, 2f, 2.5f, 3f, 3.5f, 4f))
                val R = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(10f, 20f, 30f, 40f))
                val (da, dr) = g(A, R)
                dump("da", da); dump("dr", dr)
            }
        """.trimIndent()
        // y = [[0.5,1],[10,20],[30,40],[3.5,4]]; da = 2y outside rows 1..2,
        // zero inside; dr = 2r.
        assertGradients(
            compileAndRun(AUTOGRAD_STUB_R2_R2, src),
            mapOf(
                "da" to listOf(1f, 2f, 0f, 0f, 0f, 0f, 7f, 8f),
                "dr" to listOf(20f, 40f, 60f, 80f),
            ),
        )
    }

    @Test
    fun `grad through meld routes each operand its window`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.meld
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    val y = meld(a, b)
                    (y * y).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(0.5f, 1f, 1.5f, 2f))
                val B = Tensors.f32Vector<Sym>(floatArrayOf(2.5f, 3f, 3.5f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        // loss = Σ meld(a,b)² = Σa² + Σb²: da = 2a RESHAPED BACK to [2,2]
        // (rank-2 operand rides the flatten's reshape adjoint), db = 2b
        // (rank-1 operand skips the reshape — element routing pins both).
        assertGradients(
            compileAndRun(AUTOGRAD_STUB_R2_R1, src),
            mapOf(
                "da" to listOf(1f, 2f, 3f, 4f),
                "db" to listOf(5f, 6f, 7f),
            ),
        )
    }

    private fun assertGradients(result: RunResult, want: Map<String, List<Float>>) {
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")
        val keptOriginal = result.messages.any {
            "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the gradient to lower. Warnings:\n${
                result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }
            }",
        )
        val lines = result.stdout.trim().lines()
        assertEquals(2 * want.size, lines.size, "expected ${2 * want.size} stdout lines, got: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name'")
            assertEquals(expect.size, values.size, "$name size")
            assertTrue(
                values.any { it != -1.0f },
                "$name: stub sentinel returned — rewrite never fired. Messages:\n" +
                    result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" },
            )
            for (j in expect.indices) {
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
        val tempDir = Files.createTempDirectory("tlaloc-view-withchange-meld-grad-test").toFile()
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
        private const val R2 = "DTensor<Rank2<Sym, Lit<Int>>, F32>"
        private const val R1 = "DTensor<Rank1<Sym>, F32>"

        private val AUTOGRAD_STUB_R2_R2 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: ($R2, $R2) -> Float): ($R2, $R2) -> Pair<$R2, $R2> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(8) { -1.0f }), intArrayOf(4, 2), F32),
                    DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32),
                ) }
        """.trimIndent()

        private val AUTOGRAD_STUB_R2_R1 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: ($R2, $R1) -> Float): ($R2, $R1) -> Pair<$R2, $R1> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(3, 2), F32),
                    DTensor(HostF32Storage(FloatArray(2) { -1.0f }), intArrayOf(2), F32),
                ) }
        """.trimIndent()
    }
}
