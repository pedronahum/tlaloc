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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.416 — Phase B5 E2E, the forward side: `customJvp(f, jvpFn)` and
 * `customVjpJvp(f, vjpFn, jvpFn)` differentiate end-to-end through the K2
 * plugin. The certs are the slice's own list:
 *
 *  - a deliberately NON-mathematical jvpFn (`3·dt` where the truth for
 *    `f = exp(t)` is `exp(t)·dt`) — the ONLY way 3 comes out of `jvp {}` is
 *    the USER tangent running verbatim, no fallback — with `valueAndJvp`
 *    showing the semantic fork in ONE call: value = exp(t), tangent = 3·dt;
 *  - `customJvp2`'s (a, b, da, db) convention through `jvp2`;
 *  - `customVjpJvp` with CONSISTENT bodies passing the JVP⇄VJP
 *    cross-identity end to end (⟨∇f(x), v⟩ = jvp_f(x, v), the §0.4.394-style
 *    inner-product check, on the tensor path through the full vertical);
 *  - `customVjpJvp` with INCONSISTENT bodies still running each mode's OWN
 *    body (grad → the vjpFn's 5, jvp → the jvpFn's 3 — the design doc's
 *    semantic-fork principle pinned explicitly);
 *  - the reverse-mode refusal (`grad {}` over a customJvp-only body is a
 *    compile-time error naming `customJvp` — the mirror of §0.4.415's
 *    forward refusal).
 */
class CustomJvpGradientTest {

    @Test
    fun `customJvp user tangent runs in jvp - nonmath 3 where truth is exp, jvp2 convention`() {
        val src = """
            import io.tlaloc.autograd.customJvp
            import io.tlaloc.autograd.customJvp2
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.jvp2
            import io.tlaloc.autograd.valueAndJvp
            import io.tlaloc.core.exp
            fun main() {
                val j = jvp { x: Float ->
                    val e = customJvp({ t: Float -> t.exp() }, { t: Float, dt: Float -> dt * 3.0f })
                    e(x)
                }
                println("jvp " + j(2.0f, 1.0f))
                val vj = valueAndJvp { x: Float ->
                    customJvp({ t: Float -> t.exp() }, { t: Float, dt: Float -> dt * 3.0f })(x)
                }
                val (y, dy) = vj(2.0f, 1.0f)
                println("value " + y)
                println("tangent " + dy)
                val j2 = jvp2 { a: Float, b: Float ->
                    val f2 = customJvp2(
                        { p: Float, q: Float -> p * q },
                        { p: Float, q: Float, dp: Float, dq: Float -> dp * 7.0f + dq * 11.0f },
                    )
                    f2(a, b)
                }
                println("pair " + j2(2.0f, 3.0f, 1.0f, 0.0f) + " " + j2(2.0f, 3.0f, 0.0f, 1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "stdout: ${result.stdout}")
        // The truth would be exp(2)·1 ≈ 7.389; only the USER tangent gives 3.
        assertEquals("jvp 3.0", lines[0])
        // The semantic fork in one call: the VALUE stream still runs f…
        val y = lines[1].removePrefix("value ").toFloat()
        assertTrue(abs(y - exp(2.0f)) < 1e-3f, "value must be exp(2); got $y")
        // …while the tangent stream runs the user body.
        assertEquals("tangent 3.0", lines[2])
        // Truth would be (q·da, p·db) = (3, 2); the user body gives (7, 11).
        assertEquals("pair 7.0 11.0", lines[3])
    }

    @Test
    fun `customVjpJvp consistent bodies pass the JVP-VJP cross-identity E2E`() {
        val src = """
            import io.tlaloc.autograd.customVjpJvp
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.jvp
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad { x: DTensor<Rank1<Sym>, F32> ->
                    val sq = customVjpJvp(
                        { t: DTensor<Rank1<Sym>, F32> -> t * t },
                        { u: DTensor<Rank1<Sym>, F32>, t: DTensor<Rank1<Sym>, F32> -> u * t * 2.0f },
                        { t: DTensor<Rank1<Sym>, F32>, dt: DTensor<Rank1<Sym>, F32> -> t * dt * 2.0f },
                    )
                    sq(x).sum().toFloat()
                }
                val j = jvp { x: DTensor<Rank1<Sym>, F32> ->
                    val sq = customVjpJvp(
                        { t: DTensor<Rank1<Sym>, F32> -> t * t },
                        { u: DTensor<Rank1<Sym>, F32>, t: DTensor<Rank1<Sym>, F32> -> u * t * 2.0f },
                        { t: DTensor<Rank1<Sym>, F32>, dt: DTensor<Rank1<Sym>, F32> -> t * dt * 2.0f },
                    )
                    sq(x).sum().toFloat()
                }
                val x = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, -2.0f, 3.0f))
                val v = Tensors.f32Vector<Sym>(floatArrayOf(0.5f, 1.0f, -1.0f))
                val gx = g(x).hostF32()
                var inner = 0.0f
                val vh = v.hostF32()
                for (i in gx.indices) inner += gx[i] * vh[i]
                println("grad " + gx.joinToString(" "))
                println("inner " + inner)
                println("jvp " + j(x, v))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val lines = result.stdout.trim().lines()
        assertEquals(3, lines.size, "stdout: ${result.stdout}")
        val gx = lines[0].removePrefix("grad ").split(" ").map { it.toFloat() }
        val inner = lines[1].removePrefix("inner ").toFloat()
        val dy = lines[2].removePrefix("jvp ").toFloat()
        // ∇ Σ x² = 2x, from the USER vjpFn.
        val x = floatArrayOf(1f, -2f, 3f)
        val v = floatArrayOf(0.5f, 1f, -1f)
        for (i in x.indices) {
            assertTrue(abs(gx[i] - 2f * x[i]) < 1e-5f, "grad[$i] = ${gx[i]}, want ${2f * x[i]}")
        }
        assertTrue(gx.any { it != -1.0f }, "stub sentinel returned — the grad rewrite never fired")
        // The cross-identity: ⟨∇f(x), v⟩ (reverse, user vjpFn) must equal
        // jvp_f(x, v) (forward, user jvpFn) — two independent user encodings
        // of the same derivative producing one number.
        assertTrue(
            abs(inner - dy) < 1e-4f,
            "JVP⇄VJP cross-identity failed E2E: ⟨grad, v⟩=$inner vs jvp=$dy",
        )
        // …and both match the analytic 2Σxᵢvᵢ.
        val want = 2f * x.indices.sumOf { (x[it] * v[it]).toDouble() }.toFloat()
        assertTrue(abs(dy - want) < 1e-4f, "jvp=$dy, want analytic $want")
    }

    @Test
    fun `customVjpJvp inconsistent bodies - each mode runs its own body`() {
        // vjpFn = 5·upstream, jvpFn = 3·dt, and the primal's own math says
        // 2x — three DIFFERENT answers, so any fallback or cross-derivation
        // is caught. The design doc's semantic-fork principle: each mode
        // honours ITS body.
        val src = """
            import io.tlaloc.autograd.customVjpJvp
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.jvp
            fun main() {
                val g = grad { x: Float ->
                    val f = customVjpJvp(
                        { t: Float -> t * t },
                        { u: Float, t: Float -> u * 5.0f },
                        { t: Float, dt: Float -> dt * 3.0f },
                    )
                    f(x)
                }
                println("grad " + g(4.0f))
                val j = jvp { x: Float ->
                    val f = customVjpJvp(
                        { t: Float -> t * t },
                        { u: Float, t: Float -> u * 5.0f },
                        { t: Float, dt: Float -> dt * 3.0f },
                    )
                    f(x)
                }
                println("jvp " + j(4.0f, 1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val lines = result.stdout.trim().lines()
        assertEquals(2, lines.size, "stdout: ${result.stdout}")
        // Reverse runs the vjpFn (5) — not the jvpFn (3), not the math (8).
        assertEquals("grad 5.0", lines[0])
        // Forward runs the jvpFn (3) — not the vjpFn (5), not the math (8).
        assertEquals("jvp 3.0", lines[1])
    }

    @Test
    fun `hessian composes forward-over-reverse through a customVjpJvp node E2E`() {
        // No stub: the REAL io.tlaloc.autograd resolves off the test
        // classpath (`hessian` needs its runtime assembleHessianForward, and
        // the customVjpJvp declaration is the shipped one). The seeded body
        // is forward-OVER-reverse: the reverse pass splices the user vjpFn
        // (and clones the COARSENED for value recomputation under the
        // sentinel templates), then the forward pass meets that clone
        // mid-body and splices the user jvpFn — the tangent_body splice
        // inside a reverse-produced body, E2E. Consistent bodies:
        // f(x) = Σ x² gives H = 2·I.
        val src = """
            import io.tlaloc.autograd.customVjpJvp
            import io.tlaloc.autograd.hessian
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val h = hessian { x: DTensor<Rank1<Sym>, F32> ->
                    val sq = customVjpJvp(
                        { t: DTensor<Rank1<Sym>, F32> -> t * t },
                        { u: DTensor<Rank1<Sym>, F32>, t: DTensor<Rank1<Sym>, F32> -> u * t * 2.0f },
                        { t: DTensor<Rank1<Sym>, F32>, dt: DTensor<Rank1<Sym>, F32> -> t * dt * 2.0f },
                    )
                    sq(x).sum().toFloat()
                }
                val hx = h(Tensors.f32Vector<Sym>(floatArrayOf(1.0f, -2.0f, 3.0f)))
                val data = (hx.storage as HostF32Storage).data
                println(hx.dims.joinToString("x") + " " + data.joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(null, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val line = result.stdout.trim()
        val (dims, csv) = line.split(" ", limit = 2)
        assertEquals("3x3", dims, "H must assemble [3, 3]; stdout: $line")
        val values = csv.split(",").map { it.toFloat() }
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                val want = if (i == j) 2f else 0f
                assertTrue(
                    abs(values[i * 3 + j] - want) < 1e-4f,
                    "H[$i,$j] = ${values[i * 3 + j]}, want $want (H = 2I). Stdout: $line",
                )
            }
        }
    }

    @Test
    fun `grad over a customJvp-only body is a compile-time error naming customJvp`() {
        val src = """
            import io.tlaloc.autograd.customJvp
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    customJvp({ t: Float -> t * t }, { t: Float, dt: Float -> dt * 3.0f })(x)
                }
                println(g(2.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertTrue(
            result.exitCode != 0,
            "reverse mode over a customJvp-only node must refuse at compile time; got exit 0:\n${result.messages.render()}",
        )
        assertTrue(
            result.messages.any { "customJvp" in it.message },
            "the refusal must name customJvp:\n${result.messages.render()}",
        )
    }

    private fun assertNoFallback(result: RunResult) {
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the custom derivative to lower:\n${result.messages.render()}",
        )
    }

    private fun List<CompileMessage>.render(): String =
        joinToString("\n--\n") { "${it.severity}: ${it.message}" }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)
    private data class RunResult(val exitCode: Int, val messages: List<CompileMessage>, val stdout: String)

    /** [stub] shadows the classpath `io.tlaloc.autograd` (broken-sentinel
     * declarations proving the rewrite fired); null compiles against the
     * REAL classpath module instead (the hessian path needs its runtime
     * assembly helper). */
    private fun compileAndRun(stub: String?, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-customjvp-test").toFile()
        try {
            if (stub != null) File(tempDir, "Stub.kt").writeText(stub)
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
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
            @JvmName("gradTensor1")
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32) }
            fun jvp(f: (Float) -> Float): (Float, Float) -> Float = { _, _ -> -1.0f }
            @JvmName("jvpTensor1")
            fun jvp(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, F32>) -> Float =
                { _, _ -> -1.0f }
            fun valueAndJvp(f: (Float) -> Float): (Float, Float) -> Pair<Float, Float> =
                { _, _ -> Pair(-1.0f, -1.0f) }
            fun jvp2(f: (Float, Float) -> Float): (Float, Float, Float, Float) -> Float =
                { _, _, _, _ -> -1.0f }
            fun <A, R> customJvp(f: (A) -> R, jvpFn: (A, A) -> R): (A) -> R = f
            fun <A, B, R> customJvp2(f: (A, B) -> R, jvpFn: (A, B, A, B) -> R): (A, B) -> R = f
            fun <A, R> customVjpJvp(f: (A) -> R, vjpFn: (R, A) -> A, jvpFn: (A, A) -> R): (A) -> R = f
            fun <A, B, R> customVjpJvp2(
                f: (A, B) -> R,
                vjpFn: (R, A, B) -> Pair<A, B>,
                jvpFn: (A, B, A, B) -> R,
            ): (A, B) -> R = f
        """.trimIndent()
    }
}
