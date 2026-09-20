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
 * Phase A2b — `concat` / `stack` differentiable through `grad {}` (DiffKT parity).
 *
 * The op was already ruled at the IR level (§0.4.360: interpreter, variadic
 * `stablehlo.concatenate`, forward tangent, `ConcatRule`) but had no user surface at
 * all — no `:core` host op, no FIR entry, no synthesis arm — and `ConcatRule` baked
 * its window offsets out of static dims, which under `grad {}` are -1 sentinels.
 * §0.4.381 fixed the rule with `SLICE_LIKE` (runtime-extent windows read off
 * shape-only templates); this is the surface that reaches it.
 *
 * Two design points the tests pin:
 *
 *  - **The FIR folds an n-ary concat into BINARY CONCATs.** Synthesis cannot build
 *    an `IrVararg` (the documented reason for the whole `…RankN` host-shim family),
 *    and the primal concat node DOES reach the gradient body for the ordinary
 *    `concat(…).sum()` loss tail — so an n-ary node would be unsynthesizable. Concat
 *    is associative along the axis, so the fold is semantics-preserving and unbounded
 *    in n. Test 2 runs three operands, which is two binary nodes and, for the third
 *    operand, a `SLICE_LIKE` whose prior template is the first concat's result.
 *  - **`stack` is sugar**: unsqueeze each operand at the axis, then concat along it.
 *    Test 3 lands on a rank-3 result from rank-2 operands.
 *
 *   test 1  Σ concat(1, a⊙2, b⊙3)     a [2,2], b [2,3] → da = 2s [2,2], db = 3s [2,3]
 *   test 2  Σ concat(1, a, b, a)      a appears TWICE, so its two windows accumulate
 *                                       → da = 2s [2,2], db = 1s [2,3]
 *   test 3  Σ stack(0, a⊙2, b⊙3)      a, b [2,2] → [2,2,2] → da = 2s, db = 3s
 *   test 4  Σ concat(1, a⊙2, b⊙3, a⊙5, b⊙7, a⊙11)   FIVE operands (§0.4.425): four
 *                                       binary CONCAT nodes, windows of widths
 *                                       2,3,2,3,2 → da = 18s [2,2], db = 10s [2,3]
 */
class ConcatGradientTest {

    @Test
    fun `grad through concat of two differently-shaped operands`() {
        val src = body(
            "concat(1, a * 2.0f, b * 3.0f).sum().toFloat()",
            "concat",
            intArrayOf(2, 3),
        )
        assertGradient(
            src,
            want = mapOf(
                "da" to Grad(intArrayOf(2, 2), listOf(2f, 2f, 2f, 2f)),
                "db" to Grad(intArrayOf(2, 3), listOf(3f, 3f, 3f, 3f, 3f, 3f)),
            ),
        )
    }

    @Test
    fun `grad through a three-operand concat with a repeated operand`() {
        // a feeds windows 0 and 2, so its two SLICE_LIKE contributions accumulate;
        // the third operand's window starts after the FIRST CONCAT's result, i.e. its
        // prior template is an intermediate rather than a param.
        val src = body("concat(1, a, b, a).sum().toFloat()", "concat", intArrayOf(2, 3))
        assertGradient(
            src,
            want = mapOf(
                "da" to Grad(intArrayOf(2, 2), listOf(2f, 2f, 2f, 2f)),
                "db" to Grad(intArrayOf(2, 3), listOf(1f, 1f, 1f, 1f, 1f, 1f)),
            ),
        )
    }

    @Test
    fun `grad through a five-operand concat with mixed runtime extents`() {
        // §0.4.425 — the fold-to-binary is ARITY-GENERIC: five operands become four
        // binary CONCAT nodes, so every SLICE_LIKE in the gradient body has at most
        // ONE prior template no matter how wide the user concat is — the old
        // "bounded at 4 operands" ceiling lived only in the sliceLikeAfter{N} twin
        // family, which user code never reaches. Windows have DIFFERENT runtime
        // extents (widths 2,3,2,3,2 along axis 1) and distinct prime coefficients,
        // so a window returned to the wrong operand or at the wrong offset shows up
        // in the per-operand analytic sums: da = (2+5+11)s, db = (3+7)s.
        val src = body(
            "concat(1, a * 2.0f, b * 3.0f, a * 5.0f, b * 7.0f, a * 11.0f).sum().toFloat()",
            "concat",
            intArrayOf(2, 3),
        )
        assertGradient(
            src,
            want = mapOf(
                "da" to Grad(intArrayOf(2, 2), List(4) { 18f }),
                "db" to Grad(intArrayOf(2, 3), List(6) { 10f }),
            ),
        )
    }

    @Test
    fun `grad through stack`() {
        val src = body("stack(0, a * 2.0f, b * 3.0f).sum().toFloat()", "stack", intArrayOf(2, 2))
        assertGradient(
            src,
            want = mapOf(
                "da" to Grad(intArrayOf(2, 2), listOf(2f, 2f, 2f, 2f)),
                "db" to Grad(intArrayOf(2, 2), listOf(3f, 3f, 3f, 3f)),
            ),
        )
    }

    /** The user source, with [op] imported and B bound at [bDims] (A is always [2,2]). */
    private fun body(loss: String, op: String, bDims: IntArray) = """
        import io.tlaloc.autograd.grad
        import io.tlaloc.core.DTensor
        import io.tlaloc.core.F32
        import io.tlaloc.core.Lit
        import io.tlaloc.core.Rank2
        import io.tlaloc.core.Sym
        import io.tlaloc.core.Tensors
        import io.tlaloc.core.hostF32
        import io.tlaloc.core.ops.$op
        import io.tlaloc.core.ops.sum
        import io.tlaloc.core.ops.times
        import io.tlaloc.core.ops.toFloat
        fun dump(name: String, t: DTensor<*, F32>) {
            println(name)
            println(t.dims.toList().joinToString("x"))
            for (v in t.hostF32()) print("" + v + " ")
            println()
        }
        fun main() {
            val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                $loss
            }
            val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
            val B = Tensors.f32Matrix<Sym, Lit<Int>>(${bDims[0]}, ${bDims[1]},
                floatArrayOf(${(0 until bDims[0] * bDims[1]).joinToString(", ") { "${10f * (it + 1)}f" }}))
            val (da, db) = g(A, B)
            dump("da", da); dump("db", db)
        }
    """.trimIndent()

    /** An expected gradient: its runtime shape (the window contract) and its values. */
    private data class Grad(val dims: IntArray, val values: List<Float>) {
        override fun equals(other: Any?): Boolean =
            other is Grad && other.dims.contentEquals(dims) && other.values == values

        override fun hashCode(): Int = 31 * dims.contentHashCode() + values.hashCode()
    }

    private fun assertGradient(src: String, want: Map<String, Grad>) {
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.errorsAndWarnings()}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the concat gradient to lower.\n${result.errorsAndWarnings()}",
        )

        val lines = result.stdout.trim().lines()
        assertEquals(3 * want.size, lines.size, "expected ${3 * want.size} stdout lines:\n${result.stdout}")
        for (i in lines.indices step 3) {
            val name = lines[i].trim()
            val dims = lines[i + 1].trim().split("x").map { it.toInt() }.toIntArray()
            val values = lines[i + 2].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name' in:\n${result.stdout}")
            assertTrue(
                dims.contentEquals(expect.dims),
                "$name shape = ${dims.toList()}, want ${expect.dims.toList()} — the concat window was " +
                    "not sliced back to the operand's own shape.\n${result.stdout}",
            )
            assertEquals(expect.values.size, values.size, "$name size")
            assertTrue(
                values.any { it != -1.0f } || expect.values.all { it == -1.0f },
                "$name: stub sentinel returned — rewrite never fired.\n${result.errorsAndWarnings()}",
            )
            for (j in expect.values.indices) {
                assertTrue(
                    abs(values[j] - expect.values[j]) < 1e-4f,
                    "$name[$j] = ${values[j]}, want ${expect.values[j]}. Full stdout:\n${result.stdout}",
                )
            }
        }
    }

    private fun RunResult.errorsAndWarnings(): String = messages
        .filter {
            it.severity == CompilerMessageSeverity.ERROR || it.severity == CompilerMessageSeverity.WARNING
        }
        .joinToString("\n") { "${it.severity}: ${it.message}" }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-concat-gradient-test").toFile()
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
                val cause = t.cause ?: t
                RunResult(
                    2,
                    collected + CompileMessage(
                        CompilerMessageSeverity.ERROR,
                        "RUN FAILURE: $cause\n" +
                            cause.stackTrace.take(12).joinToString("\n") { "    at $it" },
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

        private val AUTOGRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: ($R2, $R2) -> Float): ($R2, $R2) -> Pair<$R2, $R2> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32),
                    DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32),
                ) }
        """.trimIndent()
    }
}
