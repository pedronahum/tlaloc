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
 * §0.4.501 — captured RUNTIME values, slice 2 of the capture arc.
 *
 * §0.4.500 taught the lowering to FOLD a captured compile-time constant into the
 * `DxirConst` an inline literal produces. A value the compiler cannot fold — the
 * result of a call, a parameter of the enclosing function — kept refusing by name.
 * This section turns those on: the captured value becomes a TRAILING parameter of
 * the lowered `DxirFunction`, and the IR phase binds that parameter at the call
 * site to an `irGet` of the very declaration the user's own lambda closed over, so
 * the synthesized gradient closes over it the same way.
 *
 * **Two oracles, both equivalence.**
 *
 * 1. THE EXPLICIT-PARAMETER SPELLING. The same body with the captured value passed
 *    in as a lambda parameter instead — `grad2 { x, s -> x * x * s }.first` against
 *    `grad { x -> x * x * scale }`. This is the oracle that matters, because the
 *    lowering is BUILT to make the two identical: a capture is re-lowered with its
 *    param declared up front (see `FirLambdaToDxirLowering.lower`), so the dxir the
 *    capture produces is the dxir the trailing-parameter spelling produces, and the
 *    reverse transform, the φ-calculus coarsening and the synthesized bytecode
 *    cannot tell them apart.
 * 2. THE INLINED VALUE. The same body with the number written in, which the tests
 *    know because they chose it. That one catches a fold/bind that agreed with
 *    itself while being wrong about the value.
 *
 * **The arity is the thing that could break silently.** The captured parameter is an
 * INPUT, never a differentiation target: the user asked for the gradient with
 * respect to the lambda's declared parameters. If that were wrong, every `grad {}`
 * in the repo would start returning a different number of gradients. It is pinned
 * three ways here — `grad` returning a bare Float while its explicit-parameter twin
 * returns a `Pair`, `grad2` still returning a two-component `Pair` with a capture in
 * the body, `valueAndGrad` still returning (value, one gradient) — and by the stub
 * signatures themselves, which would not type-check if the synthesized lambda's
 * arity or return type moved.
 *
 * **And what still refuses.** A `var` (no single value to bind), a top-level or
 * member property (a getter call, not a value declaration), a type outside
 * Float/Double/Int/Long, and any intrinsic other than the reverse-mode `grad`
 * family (whose own parameter lists are rebuilt from the lowered one). Each names
 * itself and what to do instead; none silently degrades.
 */
class CapturedRuntimeValueGradientTest {

    // ---------------- positive: the capture, against both oracles ----------------

    @Test
    fun `a captured runtime Float gives the gradient the explicit parameter gives`() {
        // f(x) = x² · scale ; df/dx = 2·x·scale, scale = 3 at run time.
        assertSameGradient(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad { x: Float -> x * x * scale }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad2
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad2 { x: Float, s: Float -> x * x * s }
                    println("" + g(2.0f, scale).first + " " + g(-3.0f, scale).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x * 3.0f }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            expected = "12.0 -18.0",
        )
    }

    @Test
    fun `a captured runtime Double is bound at f64`() {
        // The dtype half: a captured Double must reach the body as an F64 param, not
        // a widened F32 one. 0.1 · 0.3 is the discriminator §0.4.500 used for the
        // same question about constants.
        assertSameGradient(
            stub = DOUBLE_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE_D
                fun main() {
                    val scale = runtimeScale(0.15)
                    val g = grad { x: Double -> x * x * scale }
                    println("" + g(0.1) + " " + g(-3.0))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad2
                $RUNTIME_SCALE_D
                fun main() {
                    val scale = runtimeScale(0.15)
                    val g = grad2 { x: Double, s: Double -> x * x * s }
                    println("" + g(0.1, scale).first + " " + g(-3.0, scale).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Double -> x * x * 0.3 }
                    println("" + g(0.1) + " " + g(-3.0))
                }
            """.trimIndent(),
        )
    }

    @Test
    fun `a captured runtime Int reaches the body through the same cast a param does`() {
        // The other two dtypes the capture surface admits. An integer capture is
        // non-differentiable — the reverse transform emits a typed zero for it, which
        // is then DROPPED, so `grad` keeps returning one Float. The explicit-parameter
        // control's return type is `Pair<Float, Int>`: that zero is what the captured
        // spelling does not return.
        assertSameGradient(
            stub = INT_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                fun runtimeCount(base: Int): Int = base + 1
                fun main() {
                    val n = runtimeCount(2)
                    val g = grad { x: Float -> x * x * n.toFloat() }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad2
                fun runtimeCount(base: Int): Int = base + 1
                fun main() {
                    val n = runtimeCount(2)
                    val g = grad2 { x: Float, k: Int -> x * x * k.toFloat() }
                    println("" + g(2.0f, n).first + " " + g(-3.0f, n).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x * 3.toFloat() }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            expected = "12.0 -18.0",
        )
    }

    @Test
    fun `a captured runtime Long reaches an f64 body the same way`() {
        assertSameGradient(
            stub = LONG_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                fun runtimeCount(base: Long): Long = base + 1L
                fun main() {
                    val n = runtimeCount(2L)
                    val g = grad { x: Double -> x * x * n.toDouble() }
                    println("" + g(2.0) + " " + g(-3.0))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad2
                fun runtimeCount(base: Long): Long = base + 1L
                fun main() {
                    val n = runtimeCount(2L)
                    val g = grad2 { x: Double, k: Long -> x * x * k.toDouble() }
                    println("" + g(2.0, n).first + " " + g(-3.0, n).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Double -> x * x * 3L.toDouble() }
                    println("" + g(2.0) + " " + g(-3.0))
                }
            """.trimIndent(),
            expected = "12.0 -18.0",
        )
    }

    @Test
    fun `a capture read twice is ONE parameter, read twice`() {
        // `x * scale + scale * x` — df/dx = 2·scale = 6. The numbers alone would not
        // catch a duplicated capture (two params bound to the same declaration
        // compute the same answer), so this test also reads the lowered dxir: the
        // signature must carry exactly two params, the user's and one capture.
        assertSameGradient(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad { x: Float -> x * scale + scale * x }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad2
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad2 { x: Float, s: Float -> x * s + s * x }
                    println("" + g(2.0f, scale).first + " " + g(-3.0f, scale).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * 3.0f + 3.0f * x }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            expected = "6.0 6.0",
        )

        val lowered = compileAndRun(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad { x: Float -> x * scale + scale * x }
                    println(g(2.0f))
                }
            """.trimIndent(),
            pluginOptions = arrayOf("plugin:io.tlaloc.plugin:dumpLoweredIr=true"),
        )
        assertEquals(0, lowered.exitCode, "compile/run failed:\n${lowered.render()}")
        val dump = lowered.messages.firstOrNull { "fn grad_body(" in it.message }
            ?: error("no lowered-dxir dump; messages:\n${lowered.render()}")
        assertTrue(
            "fn grad_body(%0: f32, %1: f32) -> f32" in dump.message,
            "a value captured twice must become ONE trailing param (the user's %0 plus one " +
                "capture %1), not two; got:\n${dump.message.lines().first()}",
        )
    }

    @Test
    fun `a captured parameter of the enclosing function is read at the call, not baked in`() {
        // The same `grad { }` call site, two closures, two different captured values.
        // If the capture were folded, cached or bound once per call SITE rather than
        // per call, both would print the same number.
        assertSameGradient(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                fun build(scale: Float): (Float) -> Float = grad { x: Float -> x * x * scale }
                fun main() {
                    val g2 = build(2.0f)
                    val g5 = build(5.0f)
                    println("" + g2(1.0f) + " " + g5(1.0f) + " " + g2(3.0f))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad2
                fun main() {
                    val g = grad2 { x: Float, s: Float -> x * x * s }
                    println("" + g(1.0f, 2.0f).first + " " + g(1.0f, 5.0f).first +
                        " " + g(3.0f, 2.0f).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g2 = grad { x: Float -> x * x * 2.0f }
                    val g5 = grad { x: Float -> x * x * 5.0f }
                    println("" + g2(1.0f) + " " + g5(1.0f) + " " + g2(3.0f))
                }
            """.trimIndent(),
            expected = "4.0 10.0 12.0",
        )
    }

    @Test
    fun `two captures in one body keep their first-reference order`() {
        // f(x) = (x + a) · b ; df/dx = b. If the two captures were bound in the wrong
        // order the gradient would come back as `a` (2.0) instead of `b` (4.0) — this
        // is the test that a mis-ordered binding cannot pass.
        assertSameGradient(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                fun main() {
                    val a = runtimeScale(1.0f)
                    val b = runtimeScale(3.0f)
                    val g = grad { x: Float -> (x + a) * b }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad3
                $RUNTIME_SCALE
                fun main() {
                    val a = runtimeScale(1.0f)
                    val b = runtimeScale(3.0f)
                    val g = grad3 { x: Float, aa: Float, bb: Float -> (x + aa) * bb }
                    println("" + g(2.0f, a, b).first + " " + g(-3.0f, a, b).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> (x + 2.0f) * 4.0f }
                    println("" + g(2.0f) + " " + g(-3.0f))
                }
            """.trimIndent(),
            expected = "4.0 4.0",
        )
    }

    @Test
    fun `grad2 with a capture still returns exactly two gradients`() {
        // THE ARITY PIN. f(x, y) = x·y·scale ; (df/dx, df/dy) = (y·scale, x·scale).
        // The stub's declared return type is Pair<Float, Float>: a synthesized lambda
        // that also returned the capture's gradient would be a Triple and the
        // type-match guard in the IR extension would drop the rewrite, which the
        // sentinel check below then catches.
        assertSameGradient(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad2
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad2 { x: Float, y: Float -> x * y * scale }
                    val (dx, dy) = g(2.0f, 5.0f)
                    println("" + dx + " " + dy)
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad3
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad3 { x: Float, y: Float, s: Float -> x * y * s }
                    val t = g(2.0f, 5.0f, scale)
                    println("" + t.first + " " + t.second)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad2
                fun main() {
                    val g = grad2 { x: Float, y: Float -> x * y * 3.0f }
                    val (dx, dy) = g(2.0f, 5.0f)
                    println("" + dx + " " + dy)
                }
            """.trimIndent(),
            expected = "15.0 6.0",
        )
    }

    @Test
    fun `valueAndGrad with a capture still returns the value and one gradient`() {
        // f(x) = x² · scale at x = 3, scale = 3 → value 27, gradient 18.
        assertSameGradient(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.valueAndGrad
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = valueAndGrad { x: Float -> x * x * scale }
                    val (v, d) = g(3.0f)
                    println("" + v + " " + d)
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.valueAndGrad2
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = valueAndGrad2 { x: Float, s: Float -> x * x * s }
                    val t = g(3.0f, scale)
                    println("" + t.first + " " + t.second)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.valueAndGrad
                fun main() {
                    val g = valueAndGrad { x: Float -> x * x * 3.0f }
                    val (v, d) = g(3.0f)
                    println("" + v + " " + d)
                }
            """.trimIndent(),
            expected = "27.0 18.0",
        )
    }

    @Test
    fun `a capture inside a for loop survives the coarsening`() {
        // The shape `examples/differentiable-physics` is made of: a constant-trip-count
        // loop, which PhiCalculus unrolls (C5) before the reverse transform runs. The
        // captured param has to travel through the coarsening intact.
        // d = x·scale³ ; dd/dx = scale³ = 27.
        assertSameGradient(
            stub = SCALAR_STUB,
            captured = """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad { x: Float ->
                        var d = x
                        for (i in 0 until 3) { d = d * scale }
                        d
                    }
                    println("" + g(2.0f) + " " + g(-1.0f))
                }
            """.trimIndent(),
            asExplicitParameter = """
                import io.tlaloc.autograd.grad2
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val g = grad2 { x: Float, s: Float ->
                        var d = x
                        for (i in 0 until 3) { d = d * s }
                        d
                    }
                    println("" + g(2.0f, scale).first + " " + g(-1.0f, scale).first)
                }
            """.trimIndent(),
            inlined = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float ->
                        var d = x
                        for (i in 0 until 3) { d = d * 3.0f }
                        d
                    }
                    println("" + g(2.0f) + " " + g(-1.0f))
                }
            """.trimIndent(),
            expected = "27.0 27.0",
        )
    }

    @Test
    fun `the printed gradient carries the capture as a parameter and runs standalone`() {
        // docs/READABLE_REVERSE.md's certified property, under a capture: the Kotlin
        // the compiler prints for the gradient must COMPILE AND RUN, and agree with
        // the bytecode it compiled. A captured value appears as a trailing parameter
        // of the printed function — which is the honest rendering, since that is
        // exactly what it is in the dxir — so the printed function is called with the
        // captured value in that slot and must answer bit-for-bit.
        val dumpDir = Files.createTempDirectory("tlaloc-capture-dump").toFile()
        try {
            val result = compileAndRun(
                SCALAR_STUB,
                """
                    import io.tlaloc.autograd.grad
                    $RUNTIME_SCALE
                    fun main() {
                        val scale = runtimeScale(2.0f)
                        val g = grad { x: Float -> x * x * scale }
                        println(g(1.75f).toRawBits())
                        println(g(-0.5f).toRawBits())
                    }
                """.trimIndent(),
                pluginOptions = arrayOf(
                    "plugin:io.tlaloc.plugin:dumpGradSourceDir=${dumpDir.absolutePath}",
                ),
            )
            assertEquals(0, result.exitCode, "compile/run failed:\n${result.render()}")
            assertTrue(
                result.messages.none { "kept original call" in it.message },
                "the captured lambda must synthesise, not fall back:\n${result.render()}",
            )
            val dumped = dumpDir.listFiles { f -> f.name.endsWith(".kt") }?.toList().orEmpty()
            assertEquals(1, dumped.size, "expected exactly one dumped .kt file, got $dumped")
            val printed = dumped[0].readText()
            val signature = printed.lines().first { it.startsWith("fun ") }
            assertTrue(
                "scale: DTensor<ScalarShape, F32>" in signature,
                "the printed gradient must name the captured value as a parameter — that is " +
                    "what it is in the dxir, and a printed function that hid it would not run; " +
                    "got:\n$signature",
            )
            assertTrue(
                "Pair<" !in signature && "Triple<" !in signature,
                "the printed gradient must return ONE gradient (the user's), not one per " +
                    "parameter; got:\n$signature",
            )
            val fnName = Regex("""fun (\w+)\(""").find(printed)?.groupValues?.get(1)
                ?: error("dumped source carries no function declaration:\n$printed")
            val driver = """
                import io.tlaloc.core.*
                import io.tlaloc.core.ops.*
                fun main() {
                    val s = Tensors.f32Scalar(3.0f)
                    println($fnName(Tensors.f32Scalar(1.75f), s).hostF32()[0].toRawBits())
                    println($fnName(Tensors.f32Scalar(-0.5f), s).hostF32()[0].toRawBits())
                }
            """.trimIndent()
            val standalone = compilePlainAndRun(printed, driver)
            assertEquals(
                0, standalone.exitCode,
                "the printed gradient did not compile/run standalone:\n${standalone.render()}" +
                    "\n--- printed ---\n$printed",
            )
            assertEquals(
                result.stdout.trim(), standalone.stdout.trim(),
                "the printed gradient must be RAW-BIT-IDENTICAL to the compiled one",
            )
        } finally {
            dumpDir.deleteRecursively()
        }
    }

    // ---------------- negative: what still refuses, by name ----------------------

    @Test
    fun `a captured var refuses by name and says it must be immutable`() {
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                fun main() {
                    var gain = runtimeScale(1.0f)
                    val g = grad { x: Float -> x * gain }
                    gain = 9.0f
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'gain'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "must be immutable" in err,
            "and say WHY a `var` cannot be bound (the value at the gradient call may differ " +
                "from the value where the lambda was written); got:\n$err",
        )
        assertTrue(
            "reference to symbol outside the lowering scope" !in err,
            "the pre-§0.4.500 generic sentence must not come back; got:\n$err",
        )
    }

    @Test
    fun `a captured top-level val with a runtime initializer refuses as a property`() {
        // Not foldable (its initializer is a call) and not a local: reading it is a
        // getter call, which is not a value declaration the IR phase can bind to.
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                val GAIN = runtimeScale(1.0f)
                fun main() {
                    val g = grad { x: Float -> x * GAIN }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'GAIN'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "top-level or member property" in err && "getter" in err,
            "and name the reason — a property read is a getter CALL, not a local read; got:\n$err",
        )
    }

    @Test
    fun `a captured member property with a runtime initializer refuses as a property`() {
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                $RUNTIME_SCALE
                class Box { val gain = runtimeScale(1.0f) }
                fun main() {
                    val b = Box()
                    val g = grad { x: Float -> x * b.gain }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'gain'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "top-level or member property" in err,
            "an instance property is a getter call on a receiver the lowering never sees; got:\n$err",
        )
    }

    @Test
    fun `a captured value whose type is outside the surface refuses naming the surface`() {
        // A `:core` FloatScalar box. The FIR side deliberately admits a NARROWER type
        // surface for captures than for declared params: a value-class scalar param
        // enters synthesis through the §0.4.414 call-site unwrap, and a captured param
        // has no call-site slot to read the box from.
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.FloatScalar
                fun runtimeBox(base: Float): FloatScalar = FloatScalar(base + 1.0f)
                fun main() {
                    val boxed = runtimeBox(2.0f)
                    val g = grad { x: Float -> x * boxed.toFloat() }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'boxed'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "`Float`, `Double`, `Int` or `Long`" in err,
            "and name the type surface a captured value may have; got:\n$err",
        )
    }

    @Test
    fun `a value declared INSIDE the lambda is not a capture and refuses as one is not`() {
        // THE GATE THIS SLICE WAS MISSING UNTIL THE SUITE FOUND IT. A reference that
        // misses the lowering's `env` is not necessarily a capture: a `val` declared
        // in a WHILE body and read from the trailing `if (cond) break` is lowered in
        // the CONDITION region, where the body's bindings do not exist, and it lands
        // in exactly the same place a capture does. It is declared INSIDE the lambda,
        // so binding it at the call site would point the synthesized gradient at an
        // `IrVariable` that is not in scope there. Both halves of the plugin now
        // check: FIR against the lambda's own source range, and the IR phase against
        // the call's (a declaration it can close over is written before the call).
        val err = assertRefusal(
            SCALAR_STUB,
            """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float ->
                        var d = x
                        var k = 0
                        while (k < 100) {
                            d = d * 2.0f
                            k = k + 1
                            val delta = d - 50.0f
                            if (delta > 0.0f) break
                        }
                        d
                    }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'delta'" in err, "the refusal must name the value; got:\n$err")
        assertTrue(
            "declared INSIDE this lambda" in err,
            "and say that it is not a capture at all, which is the distinction that keeps the " +
                "gradient from binding an out-of-scope declaration; got:\n$err",
        )
    }

    @Test
    fun `a capture in a forward-mode intrinsic refuses and names the grad family`() {
        // `jvp` builds its own parameter list out of the lowered one (primals ++
        // tangents), so a trailing capture param there would change what the returned
        // function takes. Refused by name rather than half-supported.
        val err = assertRefusal(
            JVP_STUB,
            """
                import io.tlaloc.autograd.jvp
                $RUNTIME_SCALE
                fun main() {
                    val scale = runtimeScale(2.0f)
                    val j = jvp { x: Float -> x * x * scale }
                    println(j(2.0f, 1.0f))
                }
            """.trimIndent(),
        )
        assertTrue("'scale'" in err, "the refusal must name the capture; got:\n$err")
        assertTrue(
            "does not accept a captured runtime value" in err && "`grad2`" in err,
            "and name the intrinsics that DO carry one; got:\n$err",
        )
    }

    // ---------------- harness ----------------------------------------------------

    /**
     * Compile and run all three spellings and require IDENTICAL stdout: the capture,
     * the same body with that value as an explicit trailing lambda parameter, and the
     * same body with the value inlined. [expected] additionally pins the analytic
     * answer, so three spellings that agreed on a wrong number are still caught.
     */
    private fun assertSameGradient(
        stub: String,
        captured: String,
        asExplicitParameter: String,
        inlined: String,
        expected: String? = null,
    ) {
        // The two controls go first: if the shape is one the plugin cannot handle at
        // all, they fail and say so instead of the capture being blamed for it.
        val explicit = compileAndRun(stub, asExplicitParameter)
        assertEquals(
            0, explicit.exitCode,
            "the explicit-parameter control must compile and run; got:\n${explicit.render()}",
        )
        val withLiteral = compileAndRun(stub, inlined)
        assertEquals(
            0, withLiteral.exitCode,
            "the inlined control must compile and run; got:\n${withLiteral.render()}",
        )
        val withCapture = compileAndRun(stub, captured)
        assertEquals(
            0, withCapture.exitCode,
            "the capturing program must compile and run; got:\n${withCapture.render()}",
        )
        for ((label, r) in listOf("capture" to withCapture, "explicit" to explicit)) {
            assertTrue(
                r.messages.none { "kept original call" in it.message },
                "the $label spelling fell back to the runtime tape instead of synthesising:\n" +
                    r.render(),
            )
            assertTrue(
                r.stdout.trim().split(Regex("\\s+")).any { it != SENTINEL_F && it != SENTINEL_D },
                "the broken-stub sentinel came back for the $label spelling — the rewrite never " +
                    "fired, so this test would have passed on identically-wrong answers:\n" +
                    r.render(),
            )
        }
        assertEquals(
            explicit.stdout.trim(), withCapture.stdout.trim(),
            "capturing the value changed the gradient. This is the claim of §0.4.501: a " +
                "captured runtime value must be indistinguishable from the same value passed " +
                "in as a trailing parameter.\n" +
                "  explicit: ${explicit.stdout.trim()}\n" +
                "  captured: ${withCapture.stdout.trim()}",
        )
        assertEquals(
            withLiteral.stdout.trim(), withCapture.stdout.trim(),
            "the capture and the inlined value disagree.\n" +
                "  inlined:  ${withLiteral.stdout.trim()}\n" +
                "  captured: ${withCapture.stdout.trim()}",
        )
        if (expected != null) {
            assertEquals(
                expected, withCapture.stdout.trim(),
                "all three spellings agree, but not with the analytic gradient",
            )
        }
    }

    /** Compile [user] expecting the §0.4.499 compile-time refusal; return its text. */
    private fun assertRefusal(stub: String, user: String): String {
        val result = compileAndRun(stub, user)
        assertTrue(
            result.exitCode != 0,
            "this capture is not supported, so the build must fail rather than degrade " +
                "silently; got:\n${result.render()}",
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

    private fun collector(into: MutableList<CompileMessage>): MessageCollector =
        object : MessageCollector {
            override fun clear() {}
            override fun hasErrors(): Boolean =
                into.any { it.severity == CompilerMessageSeverity.ERROR }
            override fun report(
                severity: CompilerMessageSeverity,
                message: String,
                location: CompilerMessageSourceLocation?,
            ) {
                into += CompileMessage(severity, message)
            }
        }

    /** Compile [stub] + [user] WITH the plugin, then run `MainKt`. */
    private fun compileAndRun(
        stub: String,
        user: String,
        pluginOptions: Array<String> = emptyArray(),
    ): RunResult = compile(stub, user, pluginOptions, withPlugin = true)

    /** Compile [printed] + [driver] with NO plugin — plain user Kotlin over `:core`. */
    private fun compilePlainAndRun(printed: String, driver: String): RunResult =
        compile(printed, driver, emptyArray(), withPlugin = false)

    private fun compile(
        first: String,
        second: String,
        pluginOptions: Array<String>,
        withPlugin: Boolean,
    ): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-captured-runtime").toFile()
        try {
            File(tempDir, if (withPlugin) "Stub.kt" else "Printed.kt").writeText(first)
            File(tempDir, "Main.kt").writeText(second)
            val outDir = File(tempDir, "out").apply { mkdirs() }
            val collected = mutableListOf<CompileMessage>()
            val args = K2JVMCompilerArguments().apply {
                freeArgs = listOf(tempDir.absolutePath)
                if (withPlugin) {
                    pluginClasspaths = pluginClasspath()
                    if (pluginOptions.isNotEmpty()) this.pluginOptions = pluginOptions
                }
                destination = outDir.absolutePath
                classpath = System.getProperty("java.class.path")
                noStdlib = true
                noReflect = true
            }
            val exitCode = K2JVMCompiler().exec(collector(collected), Services.EMPTY, args).code
            if (exitCode != 0) return RunResult(exitCode, collected, "")

            val originalOut = System.out
            val baos = ByteArrayOutputStream()
            val loader = URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader)
            return try {
                System.setOut(PrintStream(baos, true, Charsets.UTF_8))
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
         * as nothing but this means the IR rewrite never fired. */
        private const val SENTINEL_F = "-1.0"
        private const val SENTINEL_D = "-1.0"

        /**
         * A value the constant evaluator CANNOT fold: the result of a call. Slice 1
         * folds a `val` whose initializer FIR can evaluate, so an initializer that is
         * merely non-literal (`2.0f + 1.0f`) would still fold and this whole test file
         * would be testing §0.4.500 again.
         */
        private val RUNTIME_SCALE = """
            fun runtimeScale(base: Float): Float = base + 1.0f
        """.trimIndent()

        /**
         * Doubling, not adding: `0.2 + 0.1` is `0.30000000000000004`, so the inlined
         * control would have had to spell the sum's rounding error rather than the
         * number. `0.15 * 2.0` IS the double nearest 0.3 — scaling by a power of two
         * is exact — so all three spellings can carry the same f64 value.
         */
        private val RUNTIME_SCALE_D = """
            fun runtimeScale(base: Double): Double = base * 2.0
        """.trimIndent()

        /** The recognised reverse-mode FQNs as source stubs, each returning a sentinel
         * so a test that passes without the rewrite firing is impossible. */
        private val SCALAR_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
            fun grad2(f: (Float, Float) -> Float): (Float, Float) -> Pair<Float, Float> =
                { _, _ -> Pair(-1.0f, -1.0f) }
            fun grad3(f: (Float, Float, Float) -> Float):
                    (Float, Float, Float) -> Triple<Float, Float, Float> =
                { _, _, _ -> Triple(-1.0f, -1.0f, -1.0f) }
            fun valueAndGrad(f: (Float) -> Float): (Float) -> Pair<Float, Float> =
                { _ -> Pair(-1.0f, -1.0f) }
            fun valueAndGrad2(f: (Float, Float) -> Float):
                    (Float, Float) -> Triple<Float, Float, Float> =
                { _, _ -> Triple(-1.0f, -1.0f, -1.0f) }
        """.trimIndent()

        private val DOUBLE_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Double) -> Double): (Double) -> Double = { _ -> -1.0 }
            fun grad2(f: (Double, Double) -> Double): (Double, Double) -> Pair<Double, Double> =
                { _, _ -> Pair(-1.0, -1.0) }
        """.trimIndent()

        /**
         * Int / Long captures get their OWN stub files: `grad2(f: (Float, Int) -> Float)`
         * and `grad2(f: (Float, Float) -> Float)` erase to the same JVM signature, so
         * they cannot live in one package with the rest.
         */
        private val INT_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
            fun grad2(f: (Float, Int) -> Float): (Float, Int) -> Pair<Float, Int> =
                { _, _ -> Pair(-1.0f, -1) }
        """.trimIndent()

        private val LONG_STUB = """
            package io.tlaloc.autograd
            fun grad(f: (Double) -> Double): (Double) -> Double = { _ -> -1.0 }
            fun grad2(f: (Double, Long) -> Double): (Double, Long) -> Pair<Double, Long> =
                { _, _ -> Pair(-1.0, -1L) }
        """.trimIndent()

        private val JVP_STUB = """
            package io.tlaloc.autograd
            fun jvp(f: (Float) -> Float): (Float, Float) -> Float = { _, _ -> -1.0f }
        """.trimIndent()
    }
}
