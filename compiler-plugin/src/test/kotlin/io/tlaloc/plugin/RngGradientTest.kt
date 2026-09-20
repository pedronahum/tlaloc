package io.tlaloc.plugin

import io.tlaloc.core.RandomKey
import io.tlaloc.core.cauchyFloats
import io.tlaloc.core.normalFloats
import io.tlaloc.core.uniformFloats
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
 * §0.4.421 — Phase D2 tail E2E: draws inside `grad {}` lambdas. The §0.4.413
 * IR arms certified that draws differentiate as constants and that a cloned
 * draw re-draws the SAME stream; this is the user-surface half — the existing
 * host spellings (`RandomKey(k0, k1).normalVector<Sym>(n)` and siblings)
 * lower to the zero-operand RNG ops inside `grad {}`, and the synthesis
 * replays the literal attrs through the `rng*` host twins. The oracle
 * discipline is the strongest available: this JUnit code calls the SAME
 * `:core/Random.kt` kernels the lowered program does, so ε agrees bit-for-bit
 * and the uniform pin (`d x = u` for a linear loss) asserts EXACT equality.
 *
 *  g1 (reparameterized): ∇ Σ (loc + scale ⊙ ε)² with ε = normal(42, 7)[5]
 *      → d loc = 2(loc + scale⊙ε), d scale = 2(loc + scale⊙ε)⊙ε — the
 *      gradient body CLONES the draw (MulRule reads ε), same key → same ε.
 *  g2 (linear, uniform, rank-2): ∇ Σ (u ⊙ x) with u = uniform(3, 9)[2,3]
 *      → d x = u BIT-EXACT.
 *  Determinism: g1 evaluated twice is bit-identical.
 *
 * The negative pin (own test): v1 is LITERAL-ONLY by recorded design — a
 * RandomKey held in a val is not a direct literal constructor receiver, so
 * the lambda falls back loudly ("kept original call") instead of guessing.
 */
class RngGradientTest {

    @Test
    fun `grad through literal-key draws lowers with no fallback`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.RandomKey
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.normalVector
            import io.tlaloc.core.uniformMatrix
            import io.tlaloc.core.ops.plus
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dumpF(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { loc: DTensor<Rank1<Sym>, F32>, scale: DTensor<Rank1<Sym>, F32> ->
                    val eps = RandomKey(42, 7).normalVector<Sym>(5)
                    val z = loc + scale * eps
                    (z * z).sum().toFloat()
                }
                val g2 = grad { x: DTensor<Rank2<Sym, Sym>, F32> ->
                    val u = RandomKey(3, 9).uniformMatrix<Sym, Sym>(2, 3)
                    (u * x).sum().toFloat()
                }
                val loc = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, -1.25f, 2.0f, 0.75f, -0.25f))
                val scale = Tensors.f32Vector<Sym>(floatArrayOf(1.5f, 0.5f, -0.75f, 1.0f, 2.25f))
                val x = Tensors.f32Matrix<Sym, Sym>(2, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
                val (dLoc, dScale) = g1(loc, scale)
                dumpF("dloc", dLoc); dumpF("dscale", dScale)
                val (dLoc2, dScale2) = g1(loc, scale)
                dumpF("dloc2", dLoc2); dumpF("dscale2", dScale2)
                dumpF("dx", g2(x))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")

        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; literal-key draws must lower (Phase D2 tail). " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )

        // The SAME kernels the lowered program calls — ε agrees bit-for-bit.
        val eps = normalFloats(RandomKey(42, 7), 5)
        val u = uniformFloats(RandomKey(3, 9), 6)
        val loc = floatArrayOf(0.5f, -1.25f, 2.0f, 0.75f, -0.25f)
        val scale = floatArrayOf(1.5f, 0.5f, -0.75f, 1.0f, 2.25f)
        val z = FloatArray(5) { loc[it] + scale[it] * eps[it] }
        val wantDLoc = FloatArray(5) { 2f * z[it] }
        val wantDScale = FloatArray(5) { 2f * z[it] * eps[it] }

        val lines = result.stdout.trim().lines()
        assertEquals(10, lines.size, "expected 10 stdout lines, got: ${result.stdout}")
        val sections = (lines.indices step 2).associate { i ->
            lines[i].trim() to lines[i + 1].trim().split(" ").map { it.toFloat() }
        }

        val dloc = sections.getValue("dloc")
        val dscale = sections.getValue("dscale")
        assertTrue(dloc.any { it != -1.0f }, "stub sentinel returned — rewrite never fired")
        for (i in 0 until 5) {
            assertTrue(
                abs(dloc[i] - wantDLoc[i]) < 1e-5f,
                "dloc[$i] = ${dloc[i]}, want ${wantDLoc[i]} (ε=${eps[i]})",
            )
            assertTrue(
                abs(dscale[i] - wantDScale[i]) < 1e-5f,
                "dscale[$i] = ${dscale[i]}, want ${wantDScale[i]} (ε=${eps[i]})",
            )
        }
        // Determinism: the cloned draw re-draws the SAME stream — bit-identical.
        assertEquals(dloc, sections.getValue("dloc2"), "gradient must be deterministic across calls")
        assertEquals(dscale, sections.getValue("dscale2"), "gradient must be deterministic across calls")

        // Linear uniform pin: d x = u EXACTLY (the §0.4.413 IR pin's E2E twin).
        val dx = sections.getValue("dx")
        assertEquals(6, dx.size, "dx size")
        for (i in 0 until 6) {
            assertTrue(dx[i] == u[i], "dx[$i] = ${dx[i]} must be BIT-EXACT u[$i] = ${u[i]}")
        }
    }

    @Test
    fun `grad through a literal-key cauchy draw lowers compositionally`() {
        // §0.4.431 — the Phase D distribution design E2E: `cauchyVector`
        // lowers as RNG_UNIFORM → SUB ½ → MUL π → TAN (no new OpKind), the
        // draw-then-transform graph differentiates as a constant, and the
        // linear-loss gradient hands back the draw itself. The oracle is the
        // strongest available: `:core`'s `cauchyFloats` mirrors the lowered
        // arms operation for operation on the same JVM, so d x = c BIT-EXACT.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.RandomKey
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.cauchyVector
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank1<Sym>, F32> ->
                    val c = RandomKey(11, 5).cauchyVector<Sym>(3)
                    (c * x).sum().toFloat()
                }
                val dx = g(Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)))
                println("dx")
                for (v in dx.hostF32()) print("" + v + " ")
                println()
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING &&
                ("kept original call" in it.message || "could not lower lambda" in it.message)
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; the literal-key cauchy draw must lower (Phase D distributions). " +
                "Warnings:\n${result.messages.filter { it.severity == CompilerMessageSeverity.WARNING }
                    .joinToString("\n--\n") { it.message }}",
        )
        val want = cauchyFloats(RandomKey(11, 5), 3)
        val lines = result.stdout.trim().lines()
        assertEquals(2, lines.size, "expected 2 stdout lines, got: ${result.stdout}")
        assertEquals("dx", lines[0].trim())
        val dx = lines[1].trim().split(" ").map { it.toFloat() }
        assertEquals(3, dx.size, "dx size")
        assertTrue(dx.any { it != -1.0f }, "stub sentinel returned — rewrite never fired")
        for (i in 0 until 3) {
            assertTrue(
                dx[i] == want[i],
                "dx[$i] = ${dx[i]} must be BIT-EXACT cauchyFloats[$i] = ${want[i]}",
            )
        }
    }

    @Test
    fun `a non-literal RandomKey receiver falls back loudly`() {
        // v1 is literal-only by recorded design: the receiver must be a
        // direct RandomKey(k0, k1) constructor call. A key held in a val is
        // a property access — the lowering refuses (naming the restriction)
        // and the lambda keeps the original call instead of guessing.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.RandomKey
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.normalVector
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank1<Sym>, F32> ->
                    val k = RandomKey(1, 2)
                    (k.normalVector<Sym>(3) * x).sum().toFloat()
                }
                val out = g(Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f)))
                println(out.hostF32().toList())
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages}")
        // Two fallback gates can catch this shape: the FIR walker refuses the
        // standalone `RandomKey(1, 2)` constructor statement ("could not
        // lower lambda: unsupported call"), or — were the val inlined — the
        // rng arm's own named v1 refusal lands in the same warning. Either
        // way the lambda must FALL BACK, never guess.
        val fellBack = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING &&
                ("could not lower lambda" in it.message || "kept original call" in it.message)
        }
        assertTrue(
            fellBack,
            "a non-literal RandomKey receiver must fall back loudly; " +
                "messages:\n${result.messages.joinToString("\n--\n") { "${it.severity}: ${it.message}" }}",
        )
        assertTrue(
            "[-1.0, -1.0, -1.0]" in result.stdout,
            "the stub sentinel must come back on fallback; stdout: ${result.stdout}",
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
        val tempDir = Files.createTempDirectory("tlaloc-rng-test").toFile()
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
         * Three `grad` overloads: the 2-param Rank1 pair (the reparam loss),
         * the 1-param Rank2 (the uniform pin), and the 1-param Rank1 (the
         * negative fallback pin). Sentinels everywhere.
         */
        private val AUTOGRAD_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(
                f: (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, F32>) -> Float,
            ): (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, F32>) ->
                    Pair<DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(5) { -1.0f }), intArrayOf(5), F32),
                    DTensor(HostF32Storage(FloatArray(5) { -1.0f }), intArrayOf(5), F32),
                ) }
            fun grad(
                f: (DTensor<Rank2<Sym, Sym>, F32>) -> Float,
            ): (DTensor<Rank2<Sym, Sym>, F32>) -> DTensor<Rank2<Sym, Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(6) { -1.0f }), intArrayOf(2, 3), F32) }
            @JvmName("gradRank1")
            fun grad(
                f: (DTensor<Rank1<Sym>, F32>) -> Float,
            ): (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32) }
        """.trimIndent()
    }
}
