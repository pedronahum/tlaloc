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
import kotlin.math.expm1
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `expm1` / `log1p` and `softCrossEntropyLoss` differentiate in `grad {}` through the
 * K2 plugin, with no fallback:
 *
 *  - scalar `x.expm1() + x.log1p()` and the `kotlin.math` spelling `expm1(x) + ln1p(x)`:
 *    d/dx = exp(x) + 1/(1 + x)
 *  - tensor `x.expm1().sum()` and `x.log1p().sum()`
 *  - `softCrossEntropyLoss(a, b)`: da = softmax(a) − softmax(b) per row (from the logit
 *    difference, arXiv 2610.31642), db = −p ⊙ (log q − Σ p log q)
 */
class SoftCrossEntropyGradientTest {

    /** Each test runs at F32 (as written) and at F64 (the same stub and program through [F64Source.of]). */
    private var precision = Precision.F32

    private fun at(p: Precision, body: () -> Unit) {
        precision = p
        body()
    }

    @Test
    fun `scalar expm1 and log1p gradients`() = at(Precision.F32) { `scalar expm1 and log1p gradients (body)`() }

    @Test
    fun `scalar expm1 and log1p gradients, F64`() = at(Precision.F64) { `scalar expm1 and log1p gradients (body)`() }

    private fun `scalar expm1 and log1p gradients (body)`() {
        for ((body, imports) in listOf(
            "x.expm1() + x.log1p()" to listOf("io.tlaloc.core.expm1", "io.tlaloc.core.log1p"),
            "expm1(x) + ln1p(x)" to listOf("kotlin.math.expm1", "kotlin.math.ln1p"),
        )) {
            val xs = listOf(0.3f, -0.5f, 1e-6f)
            val src = """
                import io.tlaloc.autograd.grad
                ${imports.joinToString("\n                ") { "import $it" }}
                fun main() {
                    val g = grad { x: Float -> $body }
                    println(listOf(${xs.joinToString(", ") { "${it}f" }}).map { g(it) }.joinToString(","))
                }
            """.trimIndent()
            val result = compileAndRun(SCALAR_STUB, src)
            assertEquals(0, result.exitCode, "compile/run failed:\n" + result.report())
            val got = result.stdout.trim().split(",").map { it.toDouble() }
            for ((i, x) in xs.withIndex()) {
                val want = exp(x.toDouble()) + 1.0 / (1.0 + x)
                assertTrue(abs(got[i] - want) < 1e-5, "d/dx($body) at $x: ${got[i]}, want $want")
            }
        }
    }

    @Test
    fun `tensor expm1 and log1p gradients`() = at(Precision.F32) { `tensor expm1 and log1p gradients (body)`() }

    @Test
    fun `tensor expm1 and log1p gradients, F64`() = at(Precision.F64) { `tensor expm1 and log1p gradients (body)`() }

    private fun `tensor expm1 and log1p gradients (body)`() {
        val xs = floatArrayOf(0.3f, -0.7f, 1.1f)
        for ((op, d) in listOf("expm1" to { x: Double -> exp(x) }, "log1p" to { x: Double -> 1.0 / (1.0 + x) })) {
            val src = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Rank1
                import io.tlaloc.core.Sym
                import io.tlaloc.core.Tensors
                import io.tlaloc.core.hostF32
                import io.tlaloc.core.ops.$op
                import io.tlaloc.core.ops.sum
                import io.tlaloc.core.ops.toFloat
                fun main() {
                    val g = grad { x: DTensor<Rank1<Sym>, F32> -> x.$op().sum().toFloat() }
                    val x = Tensors.f32Vector<Sym>(floatArrayOf(${xs.joinToString(", ") { "${it}f" }}))
                    println(g(x).hostF32().joinToString(","))
                }
            """.trimIndent()
            val result = compileAndRun(RANK1_STUB, src)
            assertEquals(0, result.exitCode, "compile/run failed:\n" + result.report())
            assertTrue(result.messages.none { "kept original call" in it.message }, "synthesis fell back:\n" + result.report())
            val got = result.stdout.trim().split(",").map { it.toDouble() }
            for ((i, x) in xs.withIndex()) {
                val want = d(x.toDouble())
                assertTrue(abs(got[i] - want) < 1e-5, "d $op at $x: ${got[i]}, want $want")
            }
        }
    }

    @Test
    fun `grad through softCrossEntropyLoss`() = at(Precision.F32) { `grad through softCrossEntropyLoss (body)`() }

    @Test
    fun `grad through softCrossEntropyLoss, F64`() = at(Precision.F64) { `grad through softCrossEntropyLoss (body)`() }

    private fun `grad through softCrossEntropyLoss (body)`() {
        // Row 0 ordinary; row 1 a student 1e-3 from its teacher.
        val a = floatArrayOf(1f, 3f, 0.5f, 0.501f)
        val b = floatArrayOf(2f, 0.5f, 0.5f, 0.5f)
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.softCrossEntropyLoss
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32>, b: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                    softCrossEntropyLoss(a, b).toFloat()
                }
                val a = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(${a.joinToString(", ") { "${it}f" }}))
                val b = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(${b.joinToString(", ") { "${it}f" }}))
                val (da, db) = g(a, b)
                println(da.hostF32().joinToString(" "))
                println(db.hostF32().joinToString(" "))
            }
        """.trimIndent()
        val result = compileAndRun(PAIR_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n" + result.report())
        assertTrue(result.messages.none { "kept original call" in it.message }, "synthesis fell back:\n" + result.report())
        val (da, db) = result.stdout.trim().lines().map { line -> line.trim().split(" ").map { it.toDouble() } }
        for (r in 0 until 2) {
            val s = listOf(a[2 * r].toDouble(), a[2 * r + 1].toDouble())
            val t = listOf(b[2 * r].toDouble(), b[2 * r + 1].toDouble())
            fun logSoftmax(x: List<Double>): List<Double> {
                val m = maxOf(x[0], x[1])
                return x.map { it - m - ln1p(exp(minOf(x[0], x[1]) - m)) }
            }
            val lq = logSoftmax(s)
            val lp = logSoftmax(t)
            val p = lp.map { exp(it) }
            // Two classes: q_0 − p_0 = p_0·expm1(s_0) with s_0 = log q_0 − log p_0.
            val g0 = p[0] * expm1(lq[0] - lp[0])
            val wantA = listOf(g0, -g0)
            val mean = p[0] * lq[0] + p[1] * lq[1]
            val wantB = listOf(-p[0] * (lq[0] - mean), -p[1] * (lq[1] - mean))
            for (k in 0..1) {
                assertTrue(abs(da[2 * r + k] - wantA[k]) <= 1e-4 * abs(wantA[k]), "da[$r, $k] = ${da[2 * r + k]}, want ${wantA[k]}")
                assertTrue(abs(db[2 * r + k] - wantB[k]) <= 1e-4 * abs(wantB[k]) + 1e-7, "db[$r, $k] = ${db[2 * r + k]}, want ${wantB[k]}")
            }
        }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String) {
        fun report() = messages
            .filter { it.severity == CompilerMessageSeverity.ERROR || it.severity == CompilerMessageSeverity.WARNING }
            .joinToString("\n") { "${it.severity}: ${it.message}" }
    }

    private fun compileAndRun(stubAtF32: String, userAtF32: String): RunResult {
        val stub = if (precision == Precision.F64) F64Source.of(stubAtF32) else stubAtF32
        val user = if (precision == Precision.F64) F64Source.of(userAtF32) else userAtF32
        val tempDir = Files.createTempDirectory("tlaloc-soft-ce-test").toFile()
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
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                loader.loadClass("MainKt").getMethod("main").invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
            } catch (t: Throwable) {
                val cause = t.cause ?: t
                RunResult(
                    2,
                    collected + CompileMessage(
                        CompilerMessageSeverity.ERROR,
                        "RUN FAILURE: $cause\n" + cause.stackTrace.take(12).joinToString("\n") { "    at $it" },
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
        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()

        private val RANK1_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32) }
        """.trimIndent()

        private val PAIR_STUB = """
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
