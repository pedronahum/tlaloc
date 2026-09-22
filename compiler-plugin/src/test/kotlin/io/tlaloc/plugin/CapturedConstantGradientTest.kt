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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.500 — captured compile-time constants, slice 1 of the capture arc.
 *
 * Until this section a `grad { }` lambda could reference NOTHING declared outside
 * itself. `FirLambdaToDxirLowering.lookupReference` resolved a property access
 * against the lowering's `env` alone — lambda value-parameters and lambda-local
 * `val`s — and every other reference, including a top-level `const val`, threw
 * `reference to symbol outside the lowering scope`. That is why
 * `examples/differentiable-physics` had a README section titled "A limitation you
 * will meet immediately" and why every number in its simulator was an inlined
 * literal with the constant's name in a trailing comment.
 *
 * **The oracle here is EQUIVALENCE, not compilation.** Each positive test compiles
 * and runs the SAME program twice — once with the constant captured by name, once
 * with its value inlined as a literal — and requires the two gradients to print
 * byte-identical output. "It compiled" would not be a claim worth making: the
 * whole design of the fold is that [FirLambdaToDxirLowering.constFromLiteral] is
 * the single emission path for both spellings, so the lowered DXIR, the reverse
 * transform's output, the coarsening and the synthesized bytecode are the same.
 * If the fold ever diverges — a wrong dtype, a lost sign, an `Int` where an `F32`
 * belonged — these tests fail on the numbers, not on a compiler message.
 *
 * The negative half matters just as much. A capture that is NOT compile-time
 * resolvable (a `var`, a computed `val`, a parameter of the enclosing function, a
 * non-`const` member property) must keep refusing BY NAME, and with wording that
 * distinguishes "not a compile-time constant — captured runtime values are not yet
 * supported" from the old generic out-of-scope text. That distinction is the
 * contract slice 2 (runtime capture) builds on: nothing here silently degrades.
 */
class CapturedConstantGradientTest {

    // ---------------- positive: the fold, checked against the inlined literal ----

    @Test
    fun `a captured const val in scalar arithmetic gives the gradient the literal gives`() {
        // f(x) = (x + B) * A ; df/dx = A. Two constants, two positions, one of them
        // multiplied by the parameter and one added to it.
        assertSameAsInlined(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                const val A = 3.5f
                const val B = 0.25f
                fun main() {
                    val g = grad { x: Float -> (x + B) * A }
                    println("" + g(2.0f) + " " + g(-7.5f))
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> (x + 0.25f) * 3.5f }
                    println("" + g(2.0f) + " " + g(-7.5f))
                }
            """.trimIndent(),
            // df/dx = A = 3.5 at every point.
            expected = "3.5 3.5",
        )
    }

    @Test
    fun `a const val whose initializer is itself an expression folds`() {
        // The constant evaluator's path, not the literal-initializer one: `HALF_DT`
        // has a FirFunctionCall initializer (`div`), so the fold only happens because
        // the lowering hands FIR's own `FirExpressionEvaluator` the session.
        assertSameAsInlined(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                const val DT = 0.5f
                const val HALF_DT = DT / 2.0f
                fun main() {
                    val g = grad { x: Float -> x * x * HALF_DT }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x * 0.25f }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            // d/dx (x² · 0.25) = 0.5x  ->  1.0 at x=2, -1.5 at x=-3.
            expected = "1.0 -1.5",
        )
    }

    @Test
    fun `a captured Double const folds at f64, not at f32`() {
        // Pins the dtype half of the fold: `constFromLiteral` must read the literal's
        // KIND, not its boxed runtime class (FIR stores every integer literal's value
        // as a kotlin.Long). A Double constant that folded to f32 would change the
        // dtype of the whole expression tree.
        //
        // WRITING THIS TEST FOUND A CRASH THAT HAD NOTHING TO DO WITH CAPTURES. The
        // inlined control — `grad { x: Double -> x * 1.5 }`, no capture anywhere —
        // killed the compiler with a bare
        //   IrGenerationExtensionException: class java.lang.Float cannot be cast to
        //   class java.lang.Double
        // and no Tlaloc diagnostic at all, because `DxirReverseTransform`'s scalar
        // constant folding built every folded constant from a Float projection even
        // for an F64 node, and `DxirToIrSynthesis` then did `v as Double`. The f64
        // arm of that fold is §0.4.500's; this test is its regression pin as much as
        // it is the capture's, which is why both spellings are run.
        assertSameAsInlined(
            stub = DOUBLE_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                const val K = 1.5
                const val OFF = 0.25
                fun main() {
                    val g = grad { x: Double -> (x + OFF) * K }
                    println("" + g(2.0))
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Double -> (x + 0.25) * 1.5 }
                    println("" + g(2.0))
                }
            """.trimIndent(),
            expected = "1.5",
        )
    }

    @Test
    fun `a captured const val is a legal operand to a core tensor op`() {
        // The mixed-rank splat path (`splatScalarTo` -> `splatLiteral`), which is only
        // taken because the folded node IS a `DxirConst`: a non-constant rank-0 operand
        // would have gone through BROADCAST instead and produced a different body.
        assertSameAsInlined(
            stub = TENSOR_STUB,
            captured = """
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
                const val SCALE = 2.5f
                fun main() {
                    val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                        ((a * SCALE) * a).sum().toFloat()
                    }
                    val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                    for (v in g(A).hostF32()) print("" + v + " ")
                    println()
                }
            """.trimIndent(),
            inlined = """
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
                fun main() {
                    val g = grad { a: DTensor<Rank2<Sym, Lit<Int>>, F32> ->
                        ((a * 2.5f) * a).sum().toFloat()
                    }
                    val A = Tensors.f32Matrix<Sym, Lit<Int>>(2, 2, floatArrayOf(1f, 2f, 3f, 4f))
                    for (v in g(A).hostF32()) print("" + v + " ")
                    println()
                }
            """.trimIndent(),
            // d/da Σ 2.5·a² = 5·a  ->  5, 10, 15, 20.
            expected = "5.0 10.0 15.0 20.0",
        )
    }

    @Test
    fun `a captured Int const val as a for-loop trip count takes the same unroll as the literal`() {
        // Two folds at once, and the trip-count one is the load-bearing case: a
        // `ForLoopBound.Expression` would have been lowered to PhiCalculus's C6
        // SYMBOLIC trip-count shape, so `0 until STEPS` and `0 until 6` would have
        // produced two different gradient bodies for the same program. The equality
        // below is what says they do not.
        assertSameAsInlined(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                const val STEPS = 6
                const val DT = 0.25f
                const val DRAG = 0.3f
                fun main() {
                    val g = grad { v: Float ->
                        var x = 0.0f
                        var vv = v
                        for (i in 0 until STEPS) {
                            vv = vv - DRAG * vv * DT
                            x = x + vv * DT
                        }
                        x * x
                    }
                    println("" + g(2.0f) + " " + g(-1.25f))
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { v: Float ->
                        var x = 0.0f
                        var vv = v
                        for (i in 0 until 6) {
                            vv = vv - 0.3f * vv * 0.25f
                            x = x + vv * 0.25f
                        }
                        x * x
                    }
                    println("" + g(2.0f) + " " + g(-1.25f))
                }
            """.trimIndent(),
        )
    }

    @Test
    fun `a captured const val inside both branches of an if lowers`() {
        assertSameAsInlined(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                const val THRESH = 1.0f
                const val HI = 3.0f
                fun main() {
                    val g = grad { x: Float -> if (x > THRESH) HI * x else x * x }
                    println("" + g(4.0f) + " " + g(0.5f))
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> if (x > 1.0f) 3.0f * x else x * x }
                    println("" + g(4.0f) + " " + g(0.5f))
                }
            """.trimIndent(),
            // x=4 takes the HI branch (d = 3), x=0.5 the square branch (d = 2x = 1).
            expected = "3.0 1.0",
        )
    }

    @Test
    fun `a top-level val with a literal initializer folds, not only a const val`() {
        // A top-level `val` is single-assignment with a fixed initializer, so the value
        // the lambda would read at run time is the value folded here — but it is NOT a
        // `const val`, so it exercises the second arm of the fold's gate.
        assertSameAsInlined(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                val GAIN = 2.0f
                fun main() {
                    val g = grad { x: Float -> x * x * GAIN }
                    println("" + g(3.0f))
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x * 2.0f }
                    println("" + g(3.0f))
                }
            """.trimIndent(),
            // d/dx 2x² = 4x = 12 at x=3.
            expected = "12.0",
        )
    }

    // ---------------- negative: what still refuses, and how it says so -----------

    @Test
    fun `a captured var refuses by name as a runtime value, not as an out-of-scope symbol`() {
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                var gain = 2.0f
                fun main() {
                    val g = grad { x: Float -> x * gain }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'gain'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "is not a compile-time constant" in err && "`var`" in err,
            "the refusal must say WHY it is not foldable; got:\n$err",
        )
        assertTrue(
            "captured RUNTIME values are not yet supported" in err,
            "the refusal must be the runtime-capture wording, which is what slice 2 turns " +
                "on — not the pre-§0.4.500 generic out-of-scope text; got:\n$err",
        )
        assertTrue(
            "outside the lowering scope" !in err,
            "the old text must not survive for a case it no longer describes; got:\n$err",
        )
    }

    @Test
    fun `a captured val with a computed initializer refuses by name`() {
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                fun readIt(): Float = 2.0f
                val GAIN = readIt()
                fun main() {
                    val g = grad { x: Float -> x * GAIN }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'GAIN'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "its initializer is not a compile-time constant" in err,
            "the refusal must say that the INITIALIZER is what could not be folded; got:\n$err",
        )
    }

    @Test
    fun `a captured parameter of the enclosing function refuses by name`() {
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                fun build(gain: Float): (Float) -> Float = grad { x: Float -> x * gain }
                fun main() {
                    println(build(2.0f)(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'gain'" in err, "the refusal must name the captured parameter; got:\n$err")
        assertTrue(
            "parameter of the enclosing function" in err,
            "the refusal must name the construct, not just the symbol; got:\n$err",
        )
        assertTrue(
            "captured RUNTIME values are not yet supported" in err,
            "an enclosing parameter is the canonical slice-2 case; got:\n$err",
        )
    }

    @Test
    fun `a captured non-const member val refuses by name`() {
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                class Box { val gain: Float = 2.0f }
                fun main() {
                    val b = Box()
                    val g = grad { x: Float -> x * b.gain }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'gain'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "member property of a class and not `const`" in err,
            "an instance property's value is not a compile-time constant even with a literal " +
                "initializer (an override may supply another); got:\n$err",
        )
    }

    @Test
    fun `a capture chain refuses at the first link the compiler cannot fold`() {
        // `LABEL` IS a `const val`, but `LABEL.length` is a read of String's member
        // property, and that is the reference the lowering sees. It refuses naming
        // `length` — the actual thing it could not fold — rather than the constant
        // that happens to be underneath it. Recorded because it is the shape a user
        // will hit while reaching for a captured constant: the refusal names the LINK
        // that failed, not the expression it was part of.
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                const val LABEL = "gain"
                fun main() {
                    val g = grad { x: Float -> x * LABEL.length.toFloat() }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'length'" in err, "the refusal must name the link it could not fold; got:\n$err")
        assertTrue(
            "captured RUNTIME values are not yet supported" in err,
            "and it is still the runtime-capture refusal, not a crash; got:\n$err",
        )
    }

    // ---------------- harness ----------------------------------------------------

    /**
     * Compile and run [captured] and [inlined], require both to be green, and require
     * their stdout to be IDENTICAL. [expected] (when given) additionally pins the
     * analytic value, so a fold that broke BOTH spellings the same way is still caught.
     */
    private fun assertSameAsInlined(
        stub: String,
        captured: String,
        inlined: String,
        expected: String? = null,
    ) {
        // The inlined control goes first, deliberately: if the shape is one the plugin
        // cannot handle AT ALL, the control fails and says so, instead of the capture
        // being blamed for a gap that has nothing to do with it.
        val withLiteral = compileAndRun(stub, inlined)
        assertEquals(
            0, withLiteral.exitCode,
            "the inlined-literal control must compile and run; got:\n${withLiteral.render()}",
        )
        val withConst = compileAndRun(stub, captured)
        assertEquals(
            0, withConst.exitCode,
            "the captured-constant program must compile and run; got:\n${withConst.render()}",
        )
        assertTrue(
            withConst.stdout.trim().split(Regex("\\s+")).any { it != SENTINEL },
            "the broken-stub sentinel came back — the rewrite never fired, so this test would " +
                "have passed on two identically-wrong answers. Messages:\n${withConst.render()}",
        )
        assertEquals(
            withLiteral.stdout.trim(), withConst.stdout.trim(),
            "capturing the constant changed the gradient. This is the whole claim of §0.4.500: " +
                "a folded `const val` must be indistinguishable from the literal.\n" +
                "  literal:  ${withLiteral.stdout.trim()}\n" +
                "  captured: ${withConst.stdout.trim()}",
        )
        if (expected != null) {
            assertEquals(
                expected, withConst.stdout.trim(),
                "both spellings agree, but not with the analytic gradient",
            )
        }
    }

    /** Compile [user] expecting the §0.4.499 compile-time refusal; return its text. */
    private fun assertRefusal(stub: String, user: String): String {
        val result = compileAndRun(stub, user)
        assertTrue(
            result.exitCode != 0,
            "this capture is not compile-time resolvable, so the build must fail rather than " +
                "degrade silently; got:\n${result.render()}",
        )
        val errors = result.messages.filter { it.severity == CompilerMessageSeverity.ERROR }
        assertTrue(errors.isNotEmpty(), "expected a compile ERROR; got:\n${result.render()}")
        return errors.joinToString("\n") { it.message }
    }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private data class CompileMessage(val severity: CompilerMessageSeverity, val message: String)

    private data class RunResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
        val stdout: String,
    ) {
        fun render(): String =
            messages.joinToString("\n") { "[${it.severity}] ${it.message}" } +
                (if (stdout.isNotBlank()) "\n--- stdout ---\n$stdout" else "")
    }

    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-captured-const").toFile()
        try {
            File(tempDir, "Stub.kt").writeText(stub)
            File(tempDir, "Main.kt").writeText(user)
            val outDir = File(tempDir, "out").apply { mkdirs() }

            val collected = mutableListOf<CompileMessage>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean =
                    collected.any { it.severity == CompilerMessageSeverity.ERROR }
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
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

    private companion object {
        /** What every stub's un-rewritten fallback returns. A gradient that came back
         * as nothing but this value means the IR rewrite never fired, and the test
         * would otherwise pass by comparing two identically-broken answers. */
        private const val SENTINEL = "-1.0"

        /** The recognised `grad` FQN as a source stub, returning a sentinel so a test
         * that passes without the rewrite firing is impossible. */
        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
        """.trimIndent()

        private val DOUBLE_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Double) -> Double): (Double) -> Double = { _ -> -1.0 }
        """.trimIndent()

        private val TENSOR_STUB = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Lit
            import io.tlaloc.core.Rank2
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank2<Sym, Lit<Int>>, F32>) -> Float):
                    (DTensor<Rank2<Sym, Lit<Int>>, F32>) -> DTensor<Rank2<Sym, Lit<Int>>, F32> =
                { _ -> DTensor(HostF32Storage(FloatArray(4) { -1.0f }), intArrayOf(2, 2), F32) }
        """.trimIndent()
    }
}
