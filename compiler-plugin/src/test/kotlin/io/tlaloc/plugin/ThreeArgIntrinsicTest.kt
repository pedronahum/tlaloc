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
 * §0.4.424 — the 3-arg intrinsic tails, E2E through the REAL generic
 * `:autograd` declarations (no monomorphic stubs — the §0.4.412 harness
 * shape, "kept original call" a hard failure).
 *
 * Part 1 — `grad3` / `valueAndGrad3`: the reverse transform and synthesis
 * were arity-agnostic all along (§0.4.420's 4-param sparse `grad {}` is the
 * standing proof), so the slice is the declared surface plus the plugin's
 * name gates. Oracle: `f(a, b, c) = Σ(a ⊙ b ⊙ c)` on a quarter-integer
 * grid — `∇a = b ⊙ c`, `∇b = a ⊙ c`, `∇c = a ⊙ b`, all exact in F32.
 *
 * Part 2 — `jacobianReverse2`: the §0.4.412 reverse assembly at
 * `jacobian2`'s arity — each output-basis cotangent pass yields row `i` of
 * BOTH per-argument blocks from ONE seeded reverse pullback
 * `vjp2_f(x, w, ȳ) → (x̄, w̄)`:
 *
 *  jr1 = jacobianReverse2 { x, w -> x ⊙ w }      → J_x = diag(w), J_w = diag(x)
 *  jf1 = jacobian2        { x, w -> x ⊙ w }      → the SAME blocks, forward-assembled
 *  jr2 = jacobianReverse2 { x, w -> Σ(x ⊙ w) }   → the two [1, n] gradient rows
 *  jr3 = jacobianReverse2 { x, w -> concat(x, w) } → J_x = [I₂; 0] [5, 2],
 *                                                    J_w = [0; I₃] [5, 3]
 *
 * jr1 vs jf1 is the cross-assembly oracle (different seeded transforms,
 * same matrices). jr2 is the scalar-R degenerate (Float unit cotangent
 * through the 2-arg pullback). jr3 is genuinely TALL and RECTANGULAR
 * (m = 5 > nx = 2, nw = 3): the stacked identity blocks pin per-argument
 * row indexing through ConcatRule's symbolic SLICE_LIKE adjoints.
 */
class ThreeArgIntrinsicTest {

    @Test
    fun `grad3 and valueAndGrad3 lower through the plugin`() {
        val src = """
            import io.tlaloc.autograd.grad3
            import io.tlaloc.autograd.valueAndGrad3
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, *>) {
                val data = (t.storage as HostF32Storage).data
                println(name + " " + t.dims.joinToString("x") + " " + data.joinToString(","))
            }
            fun main() {
                val g = grad3 { a: DTensor<Rank1<Sym>, F32>,
                                b: DTensor<Rank1<Sym>, F32>,
                                c: DTensor<Rank1<Sym>, F32> ->
                    (a * b * c).sum().toFloat()
                }
                val vg = valueAndGrad3 { a: DTensor<Rank1<Sym>, F32>,
                                         b: DTensor<Rank1<Sym>, F32>,
                                         c: DTensor<Rank1<Sym>, F32> ->
                    (a * b * c).sum().toFloat()
                }
                val A = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                val B = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, 1.5f, 2.5f))
                val C = Tensors.f32Vector<Sym>(floatArrayOf(2.0f, 0.25f, 1.0f))
                val (da, db, dc) = g(A, B, C)
                show("da", da)
                show("db", db)
                show("dc", dc)
                val q = vg(A, B, C)
                println("value " + q.first)
                show("vda", q.second)
                show("vdb", q.third)
                show("vdc", q.fourth)
            }
        """.trimIndent()
        val result = compileAndRun(src, "grad3")
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")
        assertNoFallback(result)

        val rows = tensorRows(result.stdout)
        // A = [1, 2, 3], B = [0.5, 1.5, 2.5], C = [2, 0.25, 1].
        val wantDa = listOf(1.0f, 0.375f, 2.5f) // b ⊙ c
        val wantDb = listOf(2.0f, 0.5f, 3.0f) // a ⊙ c
        val wantDc = listOf(0.5f, 3.0f, 7.5f) // a ⊙ b
        check(rows, result.stdout, "da", "3", wantDa)
        check(rows, result.stdout, "db", "3", wantDb)
        check(rows, result.stdout, "dc", "3", wantDc)
        check(rows, result.stdout, "vda", "3", wantDa)
        check(rows, result.stdout, "vdb", "3", wantDb)
        check(rows, result.stdout, "vdc", "3", wantDc)
        // Σ a⊙b⊙c = 1·0.5·2 + 2·1.5·0.25 + 3·2.5·1 = 1 + 0.75 + 7.5 = 9.25 exact.
        val valueLine = result.stdout.lines().firstOrNull { it.startsWith("value ") }
            ?: error("no 'value' line in stdout:\n${result.stdout}")
        assertEquals(9.25f, valueLine.removePrefix("value ").toFloat(), 1e-6f, "primal value")
    }

    @Test
    fun `jacobianReverse2 assembles per-argument rows from seeded two-arg pullbacks`() {
        val src = """
            import io.tlaloc.autograd.jacobian2
            import io.tlaloc.autograd.jacobianReverse2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.concat
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun show(name: String, t: DTensor<*, *>) {
                val data = (t.storage as HostF32Storage).data
                println(name + " " + t.dims.joinToString("x") + " " + data.joinToString(","))
            }
            fun main() {
                val jr1 = jacobianReverse2 { x: DTensor<Rank1<Sym>, F32>, w: DTensor<Rank1<Sym>, F32> -> x * w }
                val jf1 = jacobian2 { x: DTensor<Rank1<Sym>, F32>, w: DTensor<Rank1<Sym>, F32> -> x * w }
                val jr2 = jacobianReverse2 { x: DTensor<Rank1<Sym>, F32>, w: DTensor<Rank1<Sym>, F32> ->
                    (x * w).sum().toFloat()
                }
                val jr3 = jacobianReverse2 { x: DTensor<Rank1<Sym>, F32>, w: DTensor<Rank1<Sym>, F32> ->
                    concat(0, x, w)
                }
                val X = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))
                val W = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, 1.5f, 2.5f))
                val (j1x, j1w) = jr1(X, W)
                show("jr1x", j1x); show("jr1w", j1w)
                val (f1x, f1w) = jf1(X, W)
                show("jf1x", f1x); show("jf1w", f1w)
                val (j2x, j2w) = jr2(X, W)
                show("jr2x", j2x); show("jr2w", j2w)
                val X2 = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f))
                val W3 = Tensors.f32Vector<Sym>(floatArrayOf(3.0f, 4.0f, 5.0f))
                val (j3x, j3w) = jr3(X2, W3)
                show("jr3x", j3x); show("jr3w", j3w)
            }
        """.trimIndent()
        val result = compileAndRun(src, "jacobian-reverse2")
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.joinToString("\n") { it.message }}\nstdout:\n${result.stdout}")
        assertNoFallback(result)

        val rows = tensorRows(result.stdout)
        // X = [1, 2, 3], W = [0.5, 1.5, 2.5].
        check(rows, result.stdout, "jr1x", "3x3", listOf(0.5f, 0f, 0f, 0f, 1.5f, 0f, 0f, 0f, 2.5f))
        check(rows, result.stdout, "jr1w", "3x3", listOf(1f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 3f))
        // The cross-assembly oracle: forward-assembled columns must agree
        // ENTRYWISE with reverse-assembled rows over the same body.
        for (pair in listOf("jr1x" to "jf1x", "jr1w" to "jf1w")) {
            val rev = rows[pair.first]!!.second
            val fwd = rows[pair.second]!!.second
            assertEquals(rev.size, fwd.size, "${pair.first}/${pair.second} size")
            for (i in rev.indices) {
                assertTrue(
                    abs(rev[i] - fwd[i]) < 1e-6f,
                    "cross-assembly disagreement at ${pair.first}[$i]: reverse=${rev[i]} forward=${fwd[i]}",
                )
            }
        }
        // Scalar-R degenerate: the two [1, n] gradient rows from a Float unit cotangent.
        check(rows, result.stdout, "jr2x", "1x3", listOf(0.5f, 1.5f, 2.5f))
        check(rows, result.stdout, "jr2w", "1x3", listOf(1f, 2f, 3f))
        // Tall rectangular: concat(x, w) with nx = 2, nw = 3 has m = 5;
        // J_x = [I₂; 0] and J_w = [0; I₃] stacked.
        check(
            rows, result.stdout, "jr3x", "5x2",
            listOf(
                1f, 0f,
                0f, 1f,
                0f, 0f,
                0f, 0f,
                0f, 0f,
            ),
        )
        check(
            rows, result.stdout, "jr3w", "5x3",
            listOf(
                0f, 0f, 0f,
                0f, 0f, 0f,
                1f, 0f, 0f,
                0f, 1f, 0f,
                0f, 0f, 1f,
            ),
        )
    }

    private fun assertNoFallback(result: RunResult) {
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; these intrinsics have no tape path so this is a hard failure. " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )
    }

    private fun tensorRows(stdout: String): Map<String, Pair<String, List<Float>>> =
        stdout.trim().lines().filter { !it.startsWith("value ") }.associate {
            val parts = it.trim().split(" ", limit = 3)
            parts[0] to (parts[1] to parts[2].split(",").map { s -> s.toFloat() })
        }

    private fun check(
        rows: Map<String, Pair<String, List<Float>>>,
        stdout: String,
        name: String,
        wantDims: String,
        want: List<Float>,
        tol: Float = 1e-4f,
    ) {
        val (dims, got) = rows[name] ?: error("no '$name' row in stdout:\n$stdout")
        assertEquals(wantDims, dims, "$name dims")
        assertEquals(want.size, got.size, "$name size")
        for (i in want.indices) {
            assertTrue(abs(got[i] - want[i]) < tol, "$name[$i]=${got[i]} want ${want[i]} (got $got)")
        }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(user: String, tag: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-$tag-test").toFile()
        try {
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
}
