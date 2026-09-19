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
 * Phase A5c-2 — implicit tensor × tensor broadcasting through `grad {}`
 * (DiffKT parity: `broadcast(S1, S2)` under every binary op). A5c-1 made the IR
 * broadcast; this is the user surface that lets a program say so.
 *
 * Three things had to line up:
 *
 *  - **`:core`**: the shape-preserving operators now delegate to the broadcasting
 *    walk, because a shared STATIC shape type does not imply shared runtime dims —
 *    `[N,1]` and `[N,C]` are both `Rank2<Sym, Lit<Int>>`, and that is exactly what a
 *    `grad {}` body sees under -1 sentinel dims. The `<S1, S2>` overloads in
 *    BroadcastOps.kt cover operands whose static types genuinely differ
 *    (rank extension), and live in a separate file only because generics erase: two
 *    `plus(DTensor, DTensor)` extensions in one facade class is a platform
 *    declaration clash.
 *  - **FIR**: an elementwise binary's result type is the NumPy broadcast of its
 *    operand types, not `lhs.type` — rank `max(rank_a, rank_b)` (exact, since ranks
 *    come from the call-site type even when dims are sentinels) and an extent that
 *    stays a sentinel unless both aligned extents are concrete.
 *  - **Synthesis**: tensor ADD/SUB/MUL/DIV call the broadcasting host ops, threading
 *    the result-shape witness from the derived IrType, which now propagates from the
 *    WIDER operand (and only backward-propagates to same-rank operands).
 *
 * The adjoints need no new machinery: A5c-1's `SUM_TO` un-broadcast reads its target
 * extents from the operand's runtime shape, so a `[N,C]` contribution onto an `[N,1]`
 * param reduces the stretched axis and keeps it size-1 — which is what test 1 pins.
 *
 *   test 1  a is [2,1], b is [2,3], BOTH statically `Rank2<Sym, Lit<Int>>`
 *             loss = Σ (a ⊙ b)      da = rowsum(b) SHAPE [2,1]   db = broadcast(a)
 *   test 2  v is [3] (Rank1), m is [2,3] (Rank2) — ranks genuinely differ
 *             loss = Σ (v ⊙ m)      dv = colsum(m) SHAPE [3]     dm = broadcast(v)
 *   test 3  a concrete size-1 axis from `sum(1, keepDims = true)`
 *             loss = Σ (rowsum(a) ⊙ b)   da = broadcast(rowsum(b))  db = broadcast(rowsum(a))
 */
class BroadcastBinaryGradientTest {

    @Test
    fun `grad through same-static-type operands with different runtime dims`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
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
                    (a * b).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 1, floatArrayOf(1f, 2f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        // a = [[1],[2]] broadcasts over b's 3 columns.
        //   da[i,0] = Σ_j b[i,j] → [60, 150], SHAPE [2,1] (the stretched axis stays size-1)
        //   db[i,j] = a[i]       → [[1,1,1],[2,2,2]], SHAPE [2,3]
        assertGradient(
            STUB_R2_R2,
            src,
            want = mapOf(
                "da" to Grad(intArrayOf(2, 1), listOf(60f, 150f)),
                "db" to Grad(intArrayOf(2, 3), listOf(1f, 1f, 1f, 2f, 2f, 2f)),
            ),
        )
    }

    @Test
    fun `grad through rank-differing operands`() {
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
                val g = grad { v: DTensor<Rank1<Sym>, F32>, m: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    (v * m).sum().toFloat()
                }
                val V = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
                val M = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f))
                val (dv, dm) = g(V, M)
                dump("dv", dv); dump("dm", dm)
            }
        """.trimIndent()
        // v = [1,2,3] gains a replicated leading axis over m's 2 rows.
        //   dv[j]   = Σ_k m[k,j] → [50, 70, 90], SHAPE [3] (rank-1, NOT [1,3])
        //   dm[k,j] = v[j]       → [[1,2,3],[1,2,3]], SHAPE [2,3]
        assertGradient(
            STUB_R1_R2,
            src,
            want = mapOf(
                "dv" to Grad(intArrayOf(3), listOf(50f, 70f, 90f)),
                "dm" to Grad(intArrayOf(2, 3), listOf(1f, 2f, 3f, 1f, 2f, 3f)),
            ),
        )
    }

    @Test
    fun `grad through a keepdims reduction broadcast against a full tensor`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
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
                    (a.sum(1, keepDims = true) * b).sum().toFloat()
                }
                val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
                val B = Tensors.f32Matrix<Sym, Lit<Int>>(2, 3, floatArrayOf(1f, 1f, 1f, 2f, 2f, 2f))
                val (da, db) = g(A, B)
                dump("da", da); dump("db", db)
            }
        """.trimIndent()
        // rowsum(a) = [[6],[15]] ([2,1], a CONCRETE size-1 axis) ⊙ b ([2,3]).
        //   da[i,j] = Σ_j' b[i,j']  → [[3,3,3],[6,6,6]]   (the row sum flows back to every column)
        //   db[i,j] = rowsum(a)[i]  → [[6,6,6],[15,15,15]]
        assertGradient(
            STUB_R2_R2,
            src,
            want = mapOf(
                "da" to Grad(intArrayOf(2, 3), listOf(3f, 3f, 3f, 6f, 6f, 6f)),
                "db" to Grad(intArrayOf(2, 3), listOf(6f, 6f, 6f, 15f, 15f, 15f)),
            ),
        )
    }

    /** An expected gradient: its runtime shape (the un-broadcast contract) and its values. */
    private data class Grad(val dims: IntArray, val values: List<Float>) {
        override fun equals(other: Any?): Boolean =
            other is Grad && other.dims.contentEquals(dims) && other.values == values

        override fun hashCode(): Int = 31 * dims.contentHashCode() + values.hashCode()
    }

    private fun assertGradient(stub: String, src: String, want: Map<String, Grad>) {
        val result = compileAndRun(stub, src)
        assertEquals(
            0,
            result.exitCode,
            "compile/run failed:\n${result.errorsAndWarnings()}",
        )

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the broadcasting gradient to lower.\n${result.errorsAndWarnings()}",
        )

        // Each section is three lines: name, "RxC" dims, values.
        val lines = result.stdout.trim().lines()
        assertEquals(3 * want.size, lines.size, "expected ${3 * want.size} stdout lines:\n${result.stdout}")
        for (i in lines.indices step 3) {
            val name = lines[i].trim()
            val dims = lines[i + 1].trim().split("x").map { it.toInt() }.toIntArray()
            val values = lines[i + 2].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name' in:\n${result.stdout}")
            assertTrue(
                dims.contentEquals(expect.dims),
                "$name shape = ${dims.toList()}, want ${expect.dims.toList()} — the adjoint was not " +
                    "un-broadcast to the operand's own shape.\n${result.stdout}",
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
        val tempDir = Files.createTempDirectory("tlaloc-broadcast-binary-test").toFile()
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
        private const val R1 = "DTensor<Rank1<Sym>, F32>"

        // Two `grad` overloads in one stub file would clash on the JVM (both erase to
        // `grad(Function2)Function2`), so each test gets a stub declaring only the
        // signature its lambda needs.
        private val STUB_R2_R2 = """
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

        private val STUB_R1_R2 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: ($R1, $R2) -> Float): ($R1, $R2) -> Pair<$R1, $R2> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32),
                    DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32),
                ) }
        """.trimIndent()
    }
}
