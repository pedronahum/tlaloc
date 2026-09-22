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
 * §0.4.415 — Phase B5 E2E: the `customVjp(f, vjpFn)` call-form (ratified
 * Candidate A, docs/CUSTOM_DERIVATIVES_DESIGN.md) differentiates end-to-end
 * through the K2 plugin. The certs are the design doc's own list:
 *
 *  - a deliberately NON-mathematical vjpFn (`3·upstream` where composing the
 *    primal `t·t` gives `2t·upstream`) — the ONLY way 3 comes out is the USER
 *    body running verbatim in `grad {}`, no fallback;
 *  - a captured outer literal `val`;
 *  - `customVjp2`'s Pair-returning adjoint (again non-mathematical: 7 and 11);
 *  - the tensor path, whose sentinel dims exercise the FULL new vertical —
 *    the mid-body COARSENED surviving into the gradient function, the
 *    decompose-before-synthesis step, and the CHECK_SHAPE_LIKE runtime assert
 *    synthesised to the host `checkShapeLike`;
 *  - `stopGradient` sugar (`f = identity, vjpFn = zeros`): ∇ Σ x·sg(x) = x,
 *    not 2x;
 *  - the ESCAPE refusal (a customVjp val re-bound instead of applied);
 *  - the forward-mode refusal (`jvp {}` over a customVjp body is a
 *    compile-time error naming `user_gradient`).
 */
class CustomVjpGradientTest {

    @Test
    fun `scalar customVjp runs the user adjoint - nonmath, captured val, customVjp2`() {
        val src = """
            import io.tlaloc.autograd.customVjp
            import io.tlaloc.autograd.customVjp2
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.grad2
            fun main() {
                val g1 = grad { x: Float ->
                    val sq = customVjp({ t: Float -> t * t }, { u: Float, t: Float -> u * 3.0f })
                    sq(x)
                }
                println("nonmath " + g1(5.0f))
                val g2 = grad { x: Float ->
                    val c = 4.0f
                    val f = customVjp({ t: Float -> t * c }, { u: Float, t: Float -> u * c })
                    f(x)
                }
                println("captured " + g2(9.0f))
                val g3 = grad2 { a: Float, b: Float ->
                    val f2 = customVjp2(
                        { p: Float, q: Float -> p * q },
                        { u: Float, p: Float, q: Float -> Pair(u * 7.0f, u * 11.0f) },
                    )
                    f2(a, b)
                }
                val (da, db) = g3(2.0f, 3.0f)
                println("pair " + da + " " + db)
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val lines = result.stdout.trim().lines()
        assertEquals(3, lines.size, "stdout: ${result.stdout}")
        // Composition would give 2x = 10.0 at x=5; only the USER body gives 3.
        assertEquals("nonmath 3.0", lines[0])
        assertEquals("captured 4.0", lines[1])
        // Composition would give (b, a) = (3, 2); the user body gives (7, 11).
        assertEquals("pair 7.0 11.0", lines[2])
    }

    @Test
    fun `tensor customVjp - nonmath adjoint and stopGradient sugar`() {
        val src = """
            import io.tlaloc.autograd.customVjp
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun dump(name: String, t: DTensor<*, F32>) {
                println(name)
                for (v in t.hostF32()) print("" + v + " ")
                println()
            }
            fun main() {
                val g1 = grad { x: DTensor<Rank1<Sym>, F32> ->
                    val sq = customVjp(
                        { t: DTensor<Rank1<Sym>, F32> -> t * t },
                        { u: DTensor<Rank1<Sym>, F32>, t: DTensor<Rank1<Sym>, F32> -> u * 3.0f },
                    )
                    sq(x).sum().toFloat()
                }
                dump("nonmath", g1(Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f))))
                val g2 = grad { x: DTensor<Rank1<Sym>, F32> ->
                    val sg = customVjp(
                        { t: DTensor<Rank1<Sym>, F32> -> t },
                        { u: DTensor<Rank1<Sym>, F32>, t: DTensor<Rank1<Sym>, F32> -> u * 0.0f },
                    )
                    (x * sg(x)).sum().toFloat()
                }
                dump("stopgrad", g2(Tensors.f32Vector<Sym>(floatArrayOf(1.5f, -2.0f, 4.0f))))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "compile/run failed:\n${result.messages.render()}")
        assertNoFallback(result)
        val want = mapOf(
            // Composition would give 2x = [2, 4, 6]; the user body gives 3s.
            "nonmath" to listOf(3.0f, 3.0f, 3.0f),
            // ∇ Σ x·stopGrad(x) = stopGrad(x) = x — NOT the 2x a see-through
            // gradient would produce.
            "stopgrad" to listOf(1.5f, -2.0f, 4.0f),
        )
        val lines = result.stdout.trim().lines()
        assertEquals(4, lines.size, "stdout: ${result.stdout}")
        for (i in lines.indices step 2) {
            val name = lines[i].trim()
            val values = lines[i + 1].trim().split(" ").map { it.toFloat() }
            val expect = want[name] ?: error("unexpected section '$name'")
            assertTrue(
                values.any { it != -1.0f },
                "$name: stub sentinel returned — the rewrite never fired:\n${result.messages.render()}",
            )
            assertEquals(3, values.size, "$name size")
            for (j in values.indices) {
                assertTrue(
                    abs(values[j] - expect[j]) < 1e-5f,
                    "$name[$j] = ${values[j]}, want ${expect[j]}. Stdout:\n${result.stdout}",
                )
            }
        }
    }

    @Test
    fun `escaping the customVjp result refuses loudly by name`() {
        val src = """
            import io.tlaloc.autograd.customVjp
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    val f = customVjp({ t: Float -> t * t }, { u: Float, t: Float -> u })
                    val alias = f
                    alias(x)
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertEquals(0, result.exitCode, "escape is a lowering refusal (tape fallback), not a compile error:\n${result.messages.render()}")
        assertTrue(
            result.messages.any { "escapes the lambda" in it.message },
            "expected the named escape refusal in the diagnostics:\n${result.messages.render()}",
        )
    }

    @Test
    fun `runtime shape-contract violation fails loudly through checkShapeLike`() {
        // The mismatch the STATIC types cannot see: both operands are
        // `Rank1<Sym>` but carry DIFFERENT runtime extents ([3] and [2]), and
        // the vjpFn SWAPS them — d_a := b, d_b := a. Undecidable under
        // `grad {}`'s sentinels at compile time, so the CHECK_SHAPE_LIKE wrap
        // must catch it AT RUNTIME with the pinned host message instead of
        // accumulating a silently wrong-shaped gradient.
        val src = """
            import io.tlaloc.autograd.customVjp2
            import io.tlaloc.autograd.grad2
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.ops.sum
            import io.tlaloc.core.ops.times
            import io.tlaloc.core.ops.toFloat
            fun main() {
                val g = grad2 { a: DTensor<Rank1<Sym>, F32>, b: DTensor<Rank1<Sym>, F32> ->
                    val bad = customVjp2(
                        { p: DTensor<Rank1<Sym>, F32>, q: DTensor<Rank1<Sym>, F32> -> p * p },
                        { u: DTensor<Rank1<Sym>, F32>, p: DTensor<Rank1<Sym>, F32>, q: DTensor<Rank1<Sym>, F32> ->
                            Pair(q, p)
                        },
                    )
                    bad(a, b).sum().toFloat()
                }
                g(
                    Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f)),
                    Tensors.f32Vector<Sym>(floatArrayOf(4.0f, 5.0f)),
                )
                println("survived")
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        if (result.exitCode == 0) {
            // The rewrite must have fired AND the run must have died in the
            // shape assert — "survived" reaching stdout means a silently
            // wrong-shaped gradient was accepted.
            error("expected the runtime shape assert to fire; stdout: ${result.stdout}\n${result.messages.render()}")
        }
        assertNoFallback(result)
        assertTrue(
            result.messages.any { "checkShapeLike" in it.message && "violates the VJP shape contract" in it.message },
            "expected the pinned checkShapeLike failure:\n${result.messages.render()}",
        )
    }

    @Test
    fun `jvp over a customVjp body is a compile-time error naming user_gradient`() {
        val src = """
            import io.tlaloc.autograd.customVjp
            import io.tlaloc.autograd.jvp
            fun main() {
                val j = jvp { x: Float ->
                    customVjp({ t: Float -> t * t }, { u: Float, t: Float -> u * 3.0f })(x)
                }
                println(j(2.0f, 1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(AUTOGRAD_STUB, src)
        assertTrue(
            result.exitCode != 0,
            "forward mode over a user gradient must refuse at compile time; got exit 0:\n${result.messages.render()}",
        )
        assertTrue(
            result.messages.any { "user_gradient" in it.message },
            "the refusal must name user_gradient:\n${result.messages.render()}",
        )
    }

    private fun assertNoFallback(result: RunResult) {
        val keptOriginal = result.messages.any {
            it.severity == CompilerMessageSeverity.WARNING && "kept original call" in it.message
        }
        assertTrue(
            !keptOriginal,
            "synthesis fell back; expected the customVjp gradient to lower:\n${result.messages.render()}",
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

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-customvjp-test").toFile()
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
                // §0.4.499 — the customVjp ESCAPE refusal is a lowering failure, so §0.4.499 makes it a
                // compile error by default; this file pins the opt-out path — the refusal is
                // still named, and the tape fallback still runs.
                pluginOptions = arrayOf("plugin:io.tlaloc.plugin:strictLowering=false")
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
            fun grad2(f: (Float, Float) -> Float): (Float, Float) -> Pair<Float, Float> =
                { _, _ -> Pair(-1.0f, -1.0f) }
            @JvmName("grad2Tensor1")
            fun grad2(f: (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, F32>) ->
                        Pair<DTensor<Rank1<Sym>, F32>, DTensor<Rank1<Sym>, F32>> =
                { _, _ -> Pair(
                    DTensor(HostF32Storage(FloatArray(3) { -1.0f }), intArrayOf(3), F32),
                    DTensor(HostF32Storage(FloatArray(2) { -1.0f }), intArrayOf(2), F32),
                ) }
            fun jvp(f: (Float) -> Float): (Float, Float) -> Float = { _, _ -> -1.0f }
            fun <A, R> customVjp(f: (A) -> R, vjpFn: (R, A) -> A): (A) -> R = f
            fun <A, B, R> customVjp2(f: (A, B) -> R, vjpFn: (R, A, B) -> Pair<A, B>): (A, B) -> R = f
        """.trimIndent()
    }
}
