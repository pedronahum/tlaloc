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
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Invokes K2JVMCompiler in-process with the Tlaloc plugin on the plugin classpath and
 * asserts that the correct diagnostic fires for each test source.
 *
 * The plugin jar + `:ir` + `:core` jvmJar paths are injected via system properties wired
 * in [build.gradle.kts].
 */
class TlalocPluginDiagnosticTest {

    // --------- Intrinsic recognition (session 1 legacy, trimmed) ---------

    @Test
    fun `no warning for unrelated function calls`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                fun main() {
                    println("hello")
                    val x = listOf(1, 2, 3).sum()
                    println(x)
                }
            """.trimIndent(),
        )

        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertFalse(
            result.messages.any { it.message.contains("Tlaloc") },
            "no Tlaloc diagnostic expected, got:\n${result.renderMessages()}",
        )
    }

    @Test
    fun `no warning for same-named function in a different package`() {
        val result = compile(
            stub = """
                package some.other.pkg
                fun grad(f: (Float) -> Float): (Float) -> Float = f
            """.trimIndent(),
            user = """
                import some.other.pkg.grad
                fun main() {
                    val g = grad { x: Float -> x * x }
                    println(g(2.0f))
                }
            """.trimIndent(),
        )

        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertFalse(
            result.messages.any { it.message.contains("Tlaloc") },
            "no Tlaloc diagnostic expected for shadowed fqn, got:\n${result.renderMessages()}",
        )
    }

    // --------- Lambda lowering: positive cases ---------

    @Test
    fun `lowers simple binary op lambda`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x }
                    println(g(2.0f))
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f32) -> f32")
        assertContains(dxir, "mul(%0, %0)")
    }

    @Test
    fun `lowers valueAndGrad lambda`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.valueAndGrad
                fun main() {
                    val vg = valueAndGrad { x: Float -> x + x }
                    println(vg(3.0f))
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        assertContains(lowered.single(), "fn valueAndGrad_body(%0: f32) -> f32")
        assertContains(lowered.single(), "add(%0, %0)")
    }

    @Test
    fun `lowers two-param grad2 lambda`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad2
                fun main() {
                    val g = grad2 { a: Float, b: Float -> a * b + a }
                    println(g(1.0f, 2.0f))
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad2_body(%0: f32, %1: f32) -> f32")
        assertContains(dxir, "mul(%0, %1)")
        // The add's lhs should be the mul result, rhs should be %0 (= param a)
        assertContains(dxir, "add(%2, %0)")
    }

    @Test
    fun `lowers lambda with val bindings`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float ->
                        val y = x * x
                        val z = y + x
                        z
                    }
                    println(g(3.0f))
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "mul(%0, %0)")
        assertContains(dxir, "add(%1, %0)")
        assertContains(dxir, "return %2")
    }

    @Test
    fun `lowers lambda with numeric literal`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> 2.0f * x + x }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "const 2.0 : f32")
    }

    @Test
    fun `lowers unary minus`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> -x }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        assertContains(lowered.single(), "neg(%0)")
    }

    // --------- Lambda lowering: :core DScalar surface ---------

    @Test
    fun `lowers DScalar binary op lambda`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DScalar
                import io.tlaloc.core.times
                import io.tlaloc.core.unaryMinus
                fun main() {
                    val g = grad { x: DScalar -> x * x }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f32) -> f32")
        assertContains(dxir, "mul(%0, %0)")
    }

    @Test
    fun `lowers FloatScalar binary op lambda`() {
        val result = compile(
            stub = AUTOGRAD_STUB_FLOAT_SCALAR,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.FloatScalar
                import io.tlaloc.core.plus
                fun main() {
                    val g = grad { x: FloatScalar -> x + x }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f32) -> f32")
        assertContains(dxir, "add(%0, %0)")
    }

    @Test
    fun `lowers DoubleScalar lambda at f64`() {
        val result = compile(
            stub = AUTOGRAD_STUB_DOUBLE_SCALAR,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DoubleScalar
                import io.tlaloc.core.times
                fun main() {
                    val g = grad { x: DoubleScalar -> x * x }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f64) -> f64")
        assertContains(dxir, "mul(%0, %0)")
    }

    @Test
    fun `lowers DScalar unary minus`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DScalar
                import io.tlaloc.core.times
                import io.tlaloc.core.unaryMinus
                fun main() {
                    val g = grad { x: DScalar -> -x }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        assertContains(lowered.single(), "neg(%0)")
    }

    @Test
    fun `lowers two-param DScalar grad2 lambda`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad2
                import io.tlaloc.core.DScalar
                import io.tlaloc.core.plus
                import io.tlaloc.core.times
                fun main() {
                    val g = grad2 { a: DScalar, b: DScalar -> a * b + a }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad2_body(%0: f32, %1: f32) -> f32")
        assertContains(dxir, "mul(%0, %1)")
        assertContains(dxir, "add(%2, %0)")
    }

    // --------- Lambda lowering: DTensor<ScalarShape, F32> unary surface ---------

    @Test
    fun `lowers DTensor scalar relu`() {
        val result = compile(
            stub = AUTOGRAD_STUB_DTENSOR_SCALAR,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.ScalarShape
                import io.tlaloc.core.ops.relu
                fun main() {
                    val g = grad { x: DTensor<ScalarShape, F32> -> x.relu() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "fn grad_body(%0: f32) -> f32")
        assertContains(dxir, "relu(%0)")
    }

    @Test
    fun `lowers DTensor scalar sigmoid`() {
        val result = compile(
            stub = AUTOGRAD_STUB_DTENSOR_SCALAR,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.ScalarShape
                import io.tlaloc.core.ops.sigmoid
                fun main() {
                    val g = grad { x: DTensor<ScalarShape, F32> -> x.sigmoid() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        assertContains(lowered.single(), "sigmoid(%0)")
    }

    @Test
    fun `lowers DTensor scalar chain of unary ops`() {
        val result = compile(
            stub = AUTOGRAD_STUB_DTENSOR_SCALAR,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.ScalarShape
                import io.tlaloc.core.ops.exp
                import io.tlaloc.core.ops.log
                import io.tlaloc.core.ops.tanh
                fun main() {
                    val g = grad { x: DTensor<ScalarShape, F32> -> x.exp().log().tanh() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        assertContains(dxir, "exp(%0)")
        assertContains(dxir, "log(%1)")
        assertContains(dxir, "tanh(%2)")
    }

    // --------- IR-phase extension (session 4 scaffolding) ---------

    @Test
    fun `ir generation extension observes successful lowering handoff`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x }
                    println(g(2.0f))
                }
            """.trimIndent(),
        )
        val irMessages = result.messages
            .filter { it.message.contains("Tlaloc IR extension saw handoff") }
            .map { it.message }
        assertEquals(
            1,
            irMessages.size,
            "expected exactly one IR-phase handoff message, got ${irMessages.size}:\n${result.renderMessages()}",
        )
        val msg = irMessages.single()
        assertContains(msg, "grad_body")
        assertContains(msg, "mul(%0, %0)")
    }

    // --------- IR-phase rewrite: actually execute the replaced call ---------

    /**
     * Stage A end-to-end: the IR transform replaces `grad { x -> x * x }` with a
     * synthesised lambda that computes the *gradient* of the body w.r.t. its parameter.
     * `d(x²)/dx = 2x`, so `g(3.0f) == 6.0f`. The "broken" stub returns `{ _ -> -1.0f }`,
     * so this test only passes when the IR transform actually replaced the call. A
     * diagnostic-only check could not distinguish.
     */
    @Test
    fun `ir transform replaces grad call with reverse-mode gradient lambda`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x }
                    println(g(3.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals(
            "6.0",
            result.stdout.trim(),
            "IR transform should have produced the gradient (2x at x=3 = 6.0); got ${result.stdout.trim()}",
        )
    }

    @Test
    fun `ir transform produces gradient for val binding and unary minus`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float ->
                        val y = x * x
                        -y + x
                    }
                    println(g(3.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // d(-x² + x)/dx = -2x + 1.  At x=3: -5.0.
        assertEquals("-5.0", result.stdout.trim(), "unexpected stdout")
    }

    /**
     * Stage A definition-of-done from §11.8.1 / §0.4.2: cubed primal differentiates
     * correctly via the IR-rewrite path. `d(x³)/dx = 3x²`; at x=3 → 27.0.
     */
    @Test
    fun `ir transform produces gradient of cubed primal`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x * x }
                    println(g(3.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("27.0", result.stdout.trim(), "expected d(x³)/dx at x=3 = 27.0")
    }

    // --------- Per-VjpRule equivalence: IR transform vs. analytic gradient ---------
    //
    // For each registered VjpRule, run a representative primal under the IR-transform path
    // and check the captured stdout matches the analytic gradient. The runtime tape lives
    // in `:autograd` and operates on Tracer/Tensor types, not the (Float)->Float surface
    // these tests use, so true tape-vs-IR comparison would need a separate harness; the
    // analytic-value comparisons here pin both implementations to the same math.

    @Test
    fun `ir transform gradient of add a-plus-a equals 2`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() { println(grad { x: Float -> x + x }(7.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        assertEquals("2.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of sub a-minus-a equals 0`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() { println(grad { x: Float -> x - x }(5.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        assertEquals("0.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of div by literal`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() { println(grad { x: Float -> x / 2.0f }(10.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        // d(x/2)/dx = 1/2 = 0.5
        assertEquals("0.5", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of literal divided by x`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() { println(grad { x: Float -> 1.0f / x }(2.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        // d(1/x)/dx = -1/x².  At x=2: -0.25
        assertEquals("-0.25", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of negated cube`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() { println(grad { x: Float -> -(x * x * x) }(2.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        // d(-x³)/dx = -3x².  At x=2: -12
        assertEquals("-12.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of polynomial x4 plus 2x`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float ->
                        val xx = x * x
                        val xxxx = xx * xx
                        xxxx + 2.0f * x
                    }
                    println(g(2.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        // d(x⁴ + 2x)/dx = 4x³ + 2.  At x=2: 34
        assertEquals("34.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of identity equals 1`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() { println(grad { x: Float -> x }(7.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        assertEquals("1.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of constant equals 0`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                fun main() { println(grad { x: Float -> 4.0f }(7.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, result.messages.toString())
        assertEquals("0.0", result.stdout.trim())
    }

    // --------- IR-phase multi-param + Pair/Triple boxing (§0.4.4 follow-up 1) ---------
    //
    // With `DxirReverseTransform` relaxed to N active params and `DxirToIrSynthesis` now
    // able to construct `kotlin.Pair.<init>` / `kotlin.Triple.<init>` calls, the remaining
    // three intrinsics — `grad2`, `valueAndGrad`, `valueAndGrad2` — route through the same
    // IR-rewrite path that §0.4.3 shipped for `grad`. The broken stub's grad2/valueAndGrad/
    // valueAndGrad2 bodies all return `-1.0f`-shaped nonsense, so these tests only pass
    // when the IR transform replaced the call and produced the real gradient math.

    @Test
    fun `ir transform produces pair of gradients for grad2`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad2
                fun main() {
                    val g = grad2 { a: Float, b: Float -> a * b + a }
                    println(g(2.0f, 3.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // ∂(a*b + a)/∂a = b + 1 → 4.0 at b=3.  ∂(a*b + a)/∂b = a → 2.0 at a=2.
        assertEquals("(4.0, 2.0)", result.stdout.trim())
    }

    @Test
    fun `ir transform produces value and gradient for valueAndGrad`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.valueAndGrad
                fun main() {
                    val vg = valueAndGrad { x: Float -> x * x }
                    println(vg(3.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // x² at x=3 = 9; d(x²)/dx at x=3 = 6.
        assertEquals("(9.0, 6.0)", result.stdout.trim())
    }

    @Test
    fun `ir transform produces triple for valueAndGrad2`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.valueAndGrad2
                fun main() {
                    val vg = valueAndGrad2 { a: Float, b: Float -> a * b + a }
                    println(vg(2.0f, 3.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // f(2,3) = 2*3 + 2 = 8.  ∂a = b + 1 = 4.  ∂b = a = 2.
        assertEquals("(8.0, 4.0, 2.0)", result.stdout.trim())
    }

    @Test
    fun `ir transform grad2 handles unused param with zero gradient`() {
        // f(a, b) = a * a (b unused).  ∂a = 2a, ∂b = 0.
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad2
                fun main() {
                    val g = grad2 { a: Float, b: Float -> a * a }
                    println(g(5.0f, 99.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // ∂/∂a (a²) at a=5 = 10.  ∂/∂b (a²) = 0 regardless of b's value.
        assertEquals("(10.0, 0.0)", result.stdout.trim())
    }

    // --------- IR-phase RELU via ReluRule (§0.4.7) ---------
    //
    // With `OpKind.STEP` + `ReluRule = upstream * STEP(operand)` registered in
    // `VjpRegistry`, the IR-rewrite path handles `grad { x -> x.relu() }` end-to-end
    // without the runtime tape. `Float.relu()` is the scalar surface added to :core
    // so that the FIR lowering maps the call to `OpKind.RELU` via the `io.tlaloc.core`
    // FQN table. `DxirToIrSynthesis` lowers `STEP` as `if (x > 0) 1 else 0` via
    // `irIfThenElse`, so the broken-stub compileAndRun exercises the full path.

    @Test
    fun `ir transform gradient of relu at positive input equals 1`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.relu
                fun main() { println(grad { x: Float -> x.relu() }(2.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // d(relu(x))/dx at x > 0 is 1.
        assertEquals("1.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of relu at negative input equals 0`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.relu
                fun main() { println(grad { x: Float -> x.relu() }(-3.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // d(relu(x))/dx at x < 0 is 0.
        assertEquals("0.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of relu at zero input equals 0`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.relu
                fun main() { println(grad { x: Float -> x.relu() }(0.0f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // At x == 0 the `compare GT 0` predicate is false, so STEP yields 0 —
        // matching XLA's semantics (not the mathematical H(0) = 1/2 convention).
        assertEquals("0.0", result.stdout.trim())
    }

    @Test
    fun `ir transform valueAndGrad of relu synthesises both forward and gradient`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.valueAndGrad
                import io.tlaloc.core.relu
                fun main() { println(valueAndGrad { x: Float -> x.relu() }(2.5f)) }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // valueAndGrad prepends the cloned primal RELU to the returns list, so the
        // synthesis path must lower both RELU (forward) and STEP (adjoint). At x=2.5
        // the forward value is 2.5 and the gradient is 1.0.
        assertEquals("(2.5, 1.0)", result.stdout.trim())
    }

    @Test
    fun `ir transform valueAndGrad handles val binding`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN,
            user = """
                import io.tlaloc.autograd.valueAndGrad
                fun main() {
                    val vg = valueAndGrad { x: Float ->
                        val y = x * x
                        -y + x
                    }
                    println(vg(3.0f))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        // f(3) = -(3²) + 3 = -6.  d/dx (-x² + x) = -2x + 1 → -5 at x=3.
        assertEquals("(-6.0, -5.0)", result.stdout.trim())
    }

    // --------- Lambda lowering: DTensor<Rank1<_>, F32> reduction surface (§0.4.10) ---------

    @Test
    fun `lowers DTensor Rank1 sum lambda`() {
        val result = compile(
            stub = AUTOGRAD_STUB_BROKEN_RANK1,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Rank1
                import io.tlaloc.core.Sym
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad { x: DTensor<Rank1<Sym>, F32> -> x.sum() }
                    println(g)
                }
            """.trimIndent(),
        )
        val lowered = result.loweredMessages()
        assertEquals(1, lowered.size, "expected 1 LAMBDA_LOWERED, got ${lowered.size}:\n${result.renderMessages()}")
        val dxir = lowered.single()
        // Rank-1 param types print as `f32[-1]` — the sentinel dim is a §0.4.10 decision
        // (only rank / dtype are dereferenced by any rule; no dim value is read).
        assertContains(dxir, "fn grad_body(%0: f32[-1]) -> f32")
        assertContains(dxir, "sum(%0)")
    }

    // --------- IR-phase rewrite for rank-1 SUM gradient (§0.4.10) ---------

    /**
     * End-to-end: `grad { x: DTensor<Rank1<Sym>, F32> -> x.sum() }` at `x = [1,2,3,4]`
     * produces the unit-broadcast gradient `[1,1,1,1]` via the IR-rewrite path. The broken
     * stub returns `[-1,-1,-1,-1]` — so the expected `1.0,1.0,1.0,1.0` only appears when
     * the synthesis path actually fires (DxirToIrSynthesis lowers SumRule's BROADCAST into
     * `io.tlaloc.core.ops.broadcastLike`).
     */
    @Test
    fun `ir transform produces rank-1 sum gradient via broadcastLike`() {
        val result = compileAndRun(
            stub = AUTOGRAD_STUB_BROKEN_RANK1,
            user = """
                import io.tlaloc.autograd.grad
                import io.tlaloc.core.DTensor
                import io.tlaloc.core.F32
                import io.tlaloc.core.Rank1
                import io.tlaloc.core.Sym
                import io.tlaloc.core.Tensors
                import io.tlaloc.core.hostF32
                import io.tlaloc.core.ops.sum
                fun main() {
                    val g = grad { x: DTensor<Rank1<Sym>, F32> -> x.sum() }
                    val input = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                    val out = g(input)
                    println(out.hostF32().joinToString(","))
                }
            """.trimIndent(),
        )
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals(
            "1.0,1.0,1.0,1.0",
            result.stdout.trim(),
            "IR transform should have produced the rank-1 unit-broadcast gradient; got ${result.stdout.trim()}",
        )
    }

    // --------- Stage B.4a: FIR-side `if/else` → OpKind.IF → gradient (§0.4.24) ---------
    //
    // These tests exercise the end-to-end B.4a pipeline: user-written Kotlin `if` → FIR
    // FirWhenExpression → FirLambdaToDxirLowering emits OpKind.IF with a STEP-based Bool
    // predicate → PhiCalculus.apply (no engine-backed rewrites fire on pure-IF primals)
    // → DxirReverseTransform's §0.4.23 IfRule reverses the branches → DxirToIrSynthesis
    // lowers the synthesised IF back to Kotlin `if/else`. The broken stub's `-1.0f`
    // sentinel confirms the rewrite fired — the analytic gradient values can only appear
    // when the IR transform replaced the call.

    @Test
    fun `ir transform gradient of if x-sq else neg-x picks active branch`() {
        // f(x) = if (x > 0) x*x else -x.  d/dx = if (x > 0) 2x else -1.
        // At x=2 (then): 4.0. At x=-3 (else): -1.0. At x=0 (STEP(0)=0 → else): -1.0.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float -> if (x > 0f) x * x else -x }
                println(g(2.0f))
                println(g(-3.0f))
                println(g(0.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("4.0\n-1.0\n-1.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of if-abs is sign`() {
        // f(x) = if (x > 0) x else -x  (i.e., abs(x)).  d/dx = sign(x).
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float -> if (x > 0f) x else -x }
                println(g(2.0f))
                println(g(-3.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("1.0\n-1.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of if with multi-op branches`() {
        // f(x) = if (x > 0) x*x + x else -(x*x*x).  d/dx = if (x > 0) 2x+1 else -3x².
        // At x=2: 5.0. At x=-1: -3.0.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float -> if (x > 0f) x * x + x else -(x * x * x) }
                println(g(2.0f))
                println(g(-1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("5.0\n-3.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of max via grad2 picks active operand`() {
        // f(a, b) = if (a > b) a else b  (i.e., max(a, b)).
        // ∂a = if (a > b) 1 else 0; ∂b = if (a > b) 0 else 1.
        val src = """
            import io.tlaloc.autograd.grad2
            fun main() {
                val g = grad2 { a: Float, b: Float -> if (a > b) a else b }
                println(g(5.0f, 2.0f))
                println(g(1.0f, 7.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("(1.0, 0.0)\n(0.0, 1.0)", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of if with less-than predicate`() {
        // f(x) = if (x < 0) -x else x  (abs via `<`).  d/dx = sign(x). Verifies the
        // `FirOperation.LT` lowering (operand flip → STEP(SUB(rhs, lhs))).
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float -> if (x < 0f) -x else x }
                println(g(4.0f))
                println(g(-2.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("1.0\n-1.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of if composed with outer arithmetic`() {
        // f(x) = 2f * (if (x > 0) x*x else -x) + 1f.
        // d/dx = 2f * (if (x > 0) 2x else -1).
        // At x=3 (then): 2 * 6 = 12.  At x=-2 (else): 2 * -1 = -2.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float -> 2.0f * (if (x > 0f) x * x else -x) + 1.0f }
                println(g(3.0f))
                println(g(-2.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("12.0\n-2.0", result.stdout.trim())
    }

    @Test
    fun `lambda with when-subject falls back to runtime tape`() {
        // `when (x) { else -> ... }` has a non-null subjectVariable and is outside the
        // B.4a surface — the lowering throws LoweringException → TLALOC_LAMBDA_UNSUPPORTED.
        // The runtime-tape stub returns -1.0f, so the output pins the fallback path fired.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float -> when (x) { 1.0f -> 2.0f * x; else -> x } }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val unsupported = result.messages.filter {
            it.message.contains("Tlaloc could not lower lambda")
        }
        assertEquals(
            1,
            unsupported.size,
            "expected TLALOC_LAMBDA_UNSUPPORTED fallback, got:\n" +
                result.messages.joinToString("\n") { "[${it.severity}] ${it.message}" },
        )
        assertEquals("-1.0", result.stdout.trim(), "expected broken-stub sentinel")
    }

    // --------- Stage B.4b: `for (i in 0 until N)` → OpKind.WHILE → gradient (§0.4.25) ---------
    //
    // End-to-end: user-written `var d = x; for (i in 0 until N) d = <expr>` inside
    // `grad { ... }` → FirLambdaToDxirLowering emits OpKind.WHILE in the canonical
    // C5-pattern shape → PhiCalculus.apply unrolls (C5 for concrete N + pure-mul back-edge)
    // → DxirReverseTransform produces the straight-line gradient → synthesis emits Kotlin
    // bytecode. The broken stub's `-1.0f` sentinel pins that the rewrite fired.

    @Test
    fun `ir transform gradient of iterate5 loop produces 32`() {
        // f(x) = x · 2^5 = 32x.  d/dx = 32.0f. Load-bearing C5 e2e test — pure-MUL
        // carried back-edge, 5 concrete iterations.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    for (i in 0 until 5) d = d * 2.0f
                    d
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("32.0", result.stdout.trim())
    }

    @Test
    fun `lowering accepts while-loop with trailing if-break`() {
        // §0.4.50 Gap 3 — the FIR surface now accepts `if (cond) break` at the tail of
        // a while body; the lowering hoists the break condition into the WHILE cond via
        // LAND+NOT. PhiCalculus/SCT closure of the LAND-composed WHILE is follow-up
        // work (documented in §0.4.50); this probe only verifies the surface acceptance
        // by asserting no TLALOC_LAMBDA_UNSUPPORTED diagnostic is emitted.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    var k = 0
                    while (k < 10) {
                        d = d * 2.0f
                        k = k + 1
                        if (k > 2) break
                    }
                    d
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val unsupported = result.messages.filter {
            it.message.contains("Tlaloc could not lower lambda")
        }
        assertEquals(
            0,
            unsupported.size,
            "expected no TLALOC_LAMBDA_UNSUPPORTED (FIR surface should accept if-break), got:\n" +
                result.messages.joinToString("\n") { "[${it.severity}] ${it.message}" },
        )
    }

    @Test
    fun `ir transform gradient of symbolic-T raw while-loop matches 2 power T`() {
        // §0.4.53 — raw `while (k < T)` where T is a grad2 lambda parameter (Float).
        // C6's `TripCount.Symbolic` path should fire, producing a closed form
        // `2^T · x`. d(2^T · x)/dx = 2^T. Tested at three concrete runtime T values
        // to confirm the SAME compiled lambda handles all of them (O(1) in T).
        val src = """
            import io.tlaloc.autograd.grad2
            import kotlin.math.pow
            fun main() {
                val g = grad2 { x: Float, T: Float ->
                    var d = x
                    var k = 0.0f
                    while (k < T) {
                        d = d * 2.0f
                        k = k + 1.0f
                    }
                    d
                }
                for (tval in listOf(3.0f, 5.0f, 10.0f)) {
                    val (dx, _) = g(1.0f, tval)
                    val expected = 2.0f.pow(tval)
                    println("${'$'}tval:${'$'}dx:${'$'}expected")
                }
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val lines = result.stdout.trim().lines()
        assertEquals(3, lines.size, "expected 3 lines, got:\n${result.stdout}")
        for (line in lines) {
            val (tval, dx, expected) = line.split(":").map { it.toFloat() }
            assertTrue(
                kotlin.math.abs(dx + 1.0f) > 1e-3f,
                "T=$tval dx=$dx matches broken-stub sentinel; IR transform did not fire",
            )
            assertTrue(
                kotlin.math.abs(dx - expected) / kotlin.math.abs(expected) < 1e-4f,
                "T=$tval: dx=$dx expected=$expected — symbolic-T C6 closure wrong",
            )
        }
    }

    @Test
    fun `ir transform gradient of raw while-loop iterate5 produces 32`() {
        // §0.4.50 — raw `while (cond) { ... }` form of the iterate5 kernel. User writes
        // the counter `k` explicitly (no synthetic for-desugaring); lowerRawWhileLoop
        // emits OpKind.WHILE with carried [d, k]. Cond region yields STEP(SUB(5, k)) →
        // C5 detectSimpleLoop matches the counter pattern and unrolls to d * 2^5 = 32x.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    var k = 0
                    while (k < 5) {
                        d = d * 2.0f
                        k = k + 1
                    }
                    d
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("32.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of additive loop is 1`() {
        // f(x) = x + 3 (via `d = d + 1` over 3 iterations).  d/dx = 1.0f.  C5 or C6
        // closes this (pure-ADD back-edge, concrete N).
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    for (i in 0 until 3) d = d + 1.0f
                    d
                }
                println(g(7.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("1.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of empty-range loop is 1`() {
        // `for (i in 0 until 0)` runs zero iterations → d = x unchanged → d/dx = 1.0f.
        // C5 handles the zero-trip case natively.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    for (i in 0 until 0) d = d * 2.0f
                    d
                }
                println(g(5.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("1.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of var reassignment outside loop`() {
        // `var d = x; d = d * 3.0f; d` (no loop). Exercises the FirVariableAssignment
        // arm in lowerStatement — rebinds env[d] to MUL(x, 3). d/dx = 3.0.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    d = d * 3.0f
                    d
                }
                println(g(2.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("3.0", result.stdout.trim())
    }

    @Test
    fun `lambda with while loop falls back to runtime tape`() {
        // Raw `while (cond) { ... }` is out of B.4b scope (only desugared-for-loops
        // are recognised). The FirWhileLoop statement hits `lowerStatement`'s else
        // branch → LoweringException → TLALOC_LAMBDA_UNSUPPORTED → broken-stub sentinel.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    while (d < 100.0f) d = d * 2.0f
                    d
                }
                println(g(1.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("-1.0", result.stdout.trim(), "expected runtime-tape fallback sentinel")
    }

    @Test
    fun `lambda using loop index in body compiles end-to-end via CAST`() {
        // §0.4.40 — body references the loop index `i` via `i.toFloat()`. The loop
        // param is now bound in env to the counter block-arg (i32); `.toFloat()`
        // lowers to OpKind.CAST via CAST_OP_MAP, CastRule emits no gradient
        // contribution (Int operand is non-differentiable), and DxirToIrSynthesis.irCast
        // emits the Kotlin `Int.toFloat()` member call.
        //
        // f(x) = x + 0 + 1 + 2 = x + 3.  f'(x) = 1.  Pins both the FIR `i` binding
        // and the CAST pipeline end-to-end (pre-§0.4.40 this test pinned the fallback
        // to runtime tape via the -1.0 sentinel).
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var d = x
                    for (i in 0 until 3) d = d + i.toFloat()
                    d
                }
                println(g(0.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("1.0", result.stdout.trim())
    }

    @Test
    fun `weighted-sum with loop-index CAST gradient matches analytic`() {
        // §0.4.40 — loop-index binding + CAST in a multiplicative weight. Pins the
        // MUL-of-CAST path in the gradient (MulRule's adjoint for the MUL(x, cast)
        // uses the cast node as the read-primal-operand).
        //
        // f(x) = Σ_{i=0..4} x · i  =  x · 10.  f'(x) = 10.
        val src = """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var sum = 0.0f
                    for (i in 0 until 5) sum = sum + x * i.toFloat()
                    sum
                }
                println(g(2.0f))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("10.0", result.stdout.trim())
    }

    // --------- Stage D.1d-S3: rank-1 GATHER via `arr[i]` (§0.4.42) ---------

    @Test
    fun `ir transform gradient of sum of three gathers is one-hot triple`() {
        // §0.4.42 — scalar-indexed rank-1 read via the new `operator fun get`
        // on DTensor. `grad { arr -> arr[0] + arr[1] + arr[2] }` at a 4-element
        // input should yield gradient `[1, 1, 1, 0]` — three one-hots summed
        // elementwise via the rank-1 plus synthesis arm.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun main() {
                val g = grad { arr: DTensor<Rank1<Sym>, F32> -> arr[0] + arr[1] + arr[2] }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(10.0f, 20.0f, 30.0f, 40.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("1.0,1.0,1.0,0.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of scaled gather scales the one-hot`() {
        // `grad { arr -> 5.0f * arr[2] }` at a 4-element input should yield
        // gradient `[0, 0, 5, 0]` — MulRule composed with GatherRule. Pins
        // the scalar-weighted single-gather path.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun main() {
                val g = grad { arr: DTensor<Rank1<Sym>, F32> -> 5.0f * arr[2] }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("0.0,0.0,5.0,0.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of duplicate-index gathers accumulates`() {
        // `grad { arr -> arr[1] + arr[1] }` should yield `[0, 2, 0, 0]` —
        // gradAccum's outer ADD sums the two one-hot-at-1 contributions.
        // Same-index-read is the canonical test case for accumulator correctness.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun main() {
                val g = grad { arr: DTensor<Rank1<Sym>, F32> -> arr[1] + arr[1] }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("0.0,2.0,0.0,0.0", result.stdout.trim())
    }

    @Test
    fun `ir transform gradient of squared-gather matches chain rule`() {
        // `grad { arr -> arr[2] * arr[2] }` at input `[1, 2, 3, 4]` should yield
        // `[0, 0, 6, 0]` — MulRule's `(a, b) -> (b·up, a·up)` doubles each
        // GatherRule contribution for a squared-self read (arr[2] fed to both
        // MUL operands). Gradient = 2·arr[2] · one-hot[2] = 2·3 = 6 at slot 2.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun main() {
                val g = grad { arr: DTensor<Rank1<Sym>, F32> -> arr[2] * arr[2] }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("0.0,0.0,6.0,0.0", result.stdout.trim())
    }

    // --------- Stage D.1d-S4: loop-driven rank-1 gather (§0.4.43) ---------

    @Test
    fun `ir transform gradient of per-segment brachistochrone kernel is rank-1`() {
        // §0.4.43 — the paper-faithful Brachistochrone kernel: two loop-carried vars
        // (velocity + accumulated time), per-segment height `y[i]` via GATHER,
        // energy-conservation sqrt inside the body. Combines every piece from
        // S1 (loop-index + CAST), S2 (GATHER substrate + GatherRule), S3 (FIR `arr[i]`
        // + irGather/irScatter + rank-1 ADD synthesis), and §0.4.39's multi-var
        // loop body.
        //
        // Primal: T(y) = Σ 2·dx / (v_{k-1} + v_k) where v_k² = v_{k-1}² + 2·g·y[k],
        // with v_0 = 0, g = 0.5 (so 2g = 1), dx = 1.
        //
        // Finite-difference cross-check against Kotlin-side reference.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            import io.tlaloc.core.sqrt
            fun prim(y: DTensor<Rank1<Sym>, F32>): Float {
                var v = 0.0f
                var t = 0.0f
                for (i in 0 until 4) {
                    val v_new = (v * v + y[i]).sqrt()
                    t = t + 2.0f / (v + v_new)
                    v = v_new
                }
                return t
            }
            fun main() {
                val g = grad { y: DTensor<Rank1<Sym>, F32> ->
                    var v = 0.0f
                    var t = 0.0f
                    for (i in 0 until 4) {
                        val v_new = (v * v + y[i]).sqrt()
                        t = t + 2.0f / (v + v_new)
                        v = v_new
                    }
                    t
                }
                // Finite-difference verification: perturb slot k by ±ε and compare to
                // the plugin's rank-1 gradient at slot k. Tolerance wider than earlier
                // tests because the 4-deep sqrt chain accumulates f32 rounding.
                val base = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
                val input = Tensors.f32Vector<Sym>(base)
                val analytic = g(input).hostF32()
                val eps = 1.0e-3f
                print(analytic.joinToString(","))
                print(";")
                for (k in 0 until 4) {
                    val plus = base.copyOf().also { it[k] += eps }
                    val minus = base.copyOf().also { it[k] -= eps }
                    val fd = (prim(Tensors.f32Vector<Sym>(plus)) - prim(Tensors.f32Vector<Sym>(minus))) / (2.0f * eps)
                    print(fd)
                    if (k < 3) print(",")
                }
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        val parts = result.stdout.trim().split(";")
        assertEquals(2, parts.size, "expected 'analytic;fd' layout, got:\n${result.stdout}")
        val analytic = parts[0].split(",").map { it.toFloat() }
        val fd = parts[1].split(",").map { it.toFloat() }
        assertEquals(4, analytic.size)
        assertEquals(4, fd.size)
        // Sentinel-reject.
        for ((k, a) in analytic.withIndex()) {
            assertTrue(
                kotlin.math.abs(a + 1.0f) > 1e-3f,
                "slot $k analytic=$a matches broken-stub sentinel; IR transform did not fire",
            )
        }
        // Per-slot relative-error tolerance 5e-3. Gradient is negative (T decreases
        // as any y[k] grows — bead gets faster).
        for (k in 0 until 4) {
            val relErr = kotlin.math.abs(analytic[k] - fd[k]) / (kotlin.math.abs(fd[k]) + 1.0e-7f)
            assertTrue(
                relErr < 5e-3f,
                "slot $k: analytic=${analytic[k]} vs fd=${fd[k]} → relErr=$relErr exceeds 5e-3",
            )
            assertTrue(
                analytic[k] < 0f,
                "slot $k: expected negative dT/dy[$k], got ${analytic[k]}",
            )
        }
    }

    @Test
    fun `ir transform gradient of sum-over-loop equals ones`() {
        // §0.4.43 — the S4 headline shape: accumulate `arr[i]` over a for-loop with
        // the loop index `i` bound (§0.4.40) and GATHER emission (§0.4.42). C5 unrolls
        // the 4-iter loop into 4 chained `ADD(prev, GATHER(arr, const_i))` ops; each
        // gather's adjoint is a one-hot, gradAccum's rank-1 ADD sums them into
        // `[1, 1, 1, 1]`.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            import io.tlaloc.core.Tensors
            import io.tlaloc.core.hostF32
            import io.tlaloc.core.ops.get
            fun main() {
                val g = grad { arr: DTensor<Rank1<Sym>, F32> ->
                    var total = 0.0f
                    for (i in 0 until 4) total = total + arr[i]
                    total
                }
                val input = Tensors.f32Vector<Sym>(floatArrayOf(10.0f, 20.0f, 30.0f, 40.0f))
                println(g(input).hostF32().joinToString(","))
            }
        """.trimIndent()
        val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT, user = src)
        assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
        assertEquals("1.0,1.0,1.0,1.0", result.stdout.trim())
    }

    // --------- Stage B.3 coarsening cache integration (§0.4.26) ---------
    //
    // These smoke-test the cache wiring in `TlalocIrGenerationExtension`. The cache
    // contract itself is covered by `:ir`'s [DxirCanonicalTest] + [CoarseningCacheTest];
    // here we just verify the plugin plumbs through without breaking the pipeline.

    @Test
    fun `cache in-memory mode preserves iterate5 gradient`() {
        // With `tlaloc.cache.dir=:memory:` the in-memory cache sits between the FIR
        // handoff and PhiCalculus.apply. Correctness is unchanged: iterate5 produces
        // 32.0 whether the result came from a fresh PhiCalculus run or a cache hit.
        val prev = System.getProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY)
        System.setProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY, ":memory:")
        try {
            val src = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float ->
                        var d = x
                        for (i in 0 until 5) d = d * 2.0f
                        d
                    }
                    println(g(1.0f))
                }
            """.trimIndent()
            val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
            assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
            assertEquals("32.0", result.stdout.trim())
        } finally {
            if (prev == null) System.clearProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY, prev)
        }
    }

    @Test
    fun `cache disk mode preserves if-else gradient`() {
        // Disk cache under a test tempdir. Verifies the full serialise → deserialise
        // round-trip through a real filesystem doesn't perturb the gradient.
        val tempDir = java.nio.file.Files.createTempDirectory("tlaloc-cache-plugin-test")
        val prev = System.getProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY)
        System.setProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY, tempDir.toString())
        try {
            val src = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> if (x > 0f) x * x else -x }
                    println(g(2.0f))
                    println(g(-3.0f))
                }
            """.trimIndent()
            val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
            assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
            assertEquals("4.0\n-1.0", result.stdout.trim())
        } finally {
            if (prev == null) System.clearProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.CACHE_DIR_PROPERTY, prev)
            tempDir.toFile().deleteRecursively()
        }
    }

    // --------- Stage C.3b.3a: SOI-based coarsening opt-in (§0.4.33) ---------

    @Test
    fun `soi coarsening preserves grad correctness for straight-line primal`() {
        // With `tlaloc.soi.enabled=true`, `grad { x -> x * x }` should route through
        // PhiCalculus.coarsenFunction (wrap in COARSENED) → DxirReverseTransform
        // splices gradient_body → synthesis emits straight-line Kotlin. Expected
        // gradient at x=3 is 6.0 regardless of whether SOI is enabled.
        val prev = System.getProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
        System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, "true")
        try {
            val src = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x }
                    println(g(3.0f))
                }
            """.trimIndent()
            val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
            assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
            assertEquals("6.0", result.stdout.trim())
        } finally {
            if (prev == null) System.clearProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, prev)
        }
    }

    @Test
    fun `soi coarsening falls back for valueAndGrad`() {
        // valueAndGrad isn't in C.3b.3a scope (includeForward not supported alongside
        // COARSENED). The plugin detects this and routes through the existing
        // PhiCalculus.apply path. Output pins the equivalent gradient correctness.
        val prev = System.getProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
        System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, "true")
        try {
            val src = """
                import io.tlaloc.autograd.valueAndGrad
                fun main() {
                    val vg = valueAndGrad { x: Float -> x * x }
                    println(vg(3.0f))
                }
            """.trimIndent()
            val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
            assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
            assertEquals("(9.0, 6.0)", result.stdout.trim())
        } finally {
            if (prev == null) System.clearProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, prev)
        }
    }

    @Test
    fun `soi coarsening preserves correctness for more complex polynomial`() {
        // `grad { x -> (x+1)*(x+1) + 2*x }`. d/dx = 2(x+1) + 2. At x=3: 2*4 + 2 = 10.
        val prev = System.getProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
        System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, "true")
        try {
            val src = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float ->
                        val p = x + 1.0f
                        val sq = p * p
                        sq + 2.0f * x
                    }
                    println(g(3.0f))
                }
            """.trimIndent()
            val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
            assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
            assertEquals("10.0", result.stdout.trim())
        } finally {
            if (prev == null) System.clearProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, prev)
        }
    }

    // --------- Stage C.4: empirical L tuning (§0.4.36) ---------

    @Test
    fun `soi size limit property triggers multi-SOI branch coarsening on if primal`() {
        // With SOI_ENABLED + SOI_SIZE_LIMIT=5, an if-expression primal routes through
        // coarsenMultiSoi's branch-coarsening path. Gradient correctness must match the
        // uncoarsened pipeline at both branch activations.
        val prevEnabled = System.getProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
        val prevLimit = System.getProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY)
        System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, "true")
        System.setProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY, "5")
        try {
            val src = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> if (x > 0f) x * x + x else -x - 1.0f }
                    println(g(2.0f))
                    println(g(-3.0f))
                }
            """.trimIndent()
            val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
            assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
            // At x=2 (then): d/dx(x*x+x) = 2x+1 = 5.  At x=-3 (else): d/dx(-x-1) = -1.
            assertEquals("5.0\n-1.0", result.stdout.trim())
        } finally {
            if (prevEnabled == null) System.clearProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, prevEnabled)
            if (prevLimit == null) System.clearProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY, prevLimit)
        }
    }

    @Test
    fun `soi size limit property invalid value falls back to default 50`() {
        // Invalid property values (non-numeric, zero, negative) should fall back to 50
        // rather than erroring — default-safe behaviour.
        val prev = System.getProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
        val prevLimit = System.getProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY)
        System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, "true")
        System.setProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY, "not-a-number")
        try {
            val src = """
                import io.tlaloc.autograd.grad
                fun main() {
                    val g = grad { x: Float -> x * x }
                    println(g(4.0f))
                }
            """.trimIndent()
            val result = compileAndRun(stub = AUTOGRAD_STUB_BROKEN, user = src)
            assertEquals(0, result.exitCode, "compile failed:\n${result.messages}")
            assertEquals("8.0", result.stdout.trim())
        } finally {
            if (prev == null) System.clearProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.SOI_ENABLED_PROPERTY, prev)
            if (prevLimit == null) System.clearProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY)
            else System.setProperty(TlalocIrGenerationExtension.SOI_SIZE_LIMIT_PROPERTY, prevLimit)
        }
    }

    // --------- Lambda lowering: unsupported ---------

    @Test
    fun `unsupported op emits LAMBDA_UNSUPPORTED`() {
        val result = compile(
            stub = AUTOGRAD_STUB,
            user = """
                import io.tlaloc.autograd.grad
                fun cube(x: Float): Float = x * x * x
                fun main() {
                    val g = grad { x: Float -> cube(x) }
                    println(g(1.0f))
                }
            """.trimIndent(),
        )
        val unsupported = result.messages
            .filter { it.message.contains("Tlaloc could not lower lambda") }
            .map { it.message }
        assertEquals(
            1,
            unsupported.size,
            "expected 1 LAMBDA_UNSUPPORTED, got ${unsupported.size}:\n${result.renderMessages()}",
        )
        assertContains(unsupported.single(), "cube")
        assertTrue(
            result.loweredMessages().isEmpty(),
            "expected no LAMBDA_LOWERED on unsupported op, got:\n${result.renderMessages()}",
        )
    }

    // --------- Helpers ---------

    private fun assertContains(actual: String, expected: String) {
        assertTrue(
            actual.contains(expected),
            "expected substring:\n  $expected\nnot found in:\n$actual",
        )
    }

    private fun CompileResult.loweredMessages(): List<String> =
        messages.filter { it.message.startsWith("Tlaloc lowered lambda to dxir:") }
            .map { it.message.substringAfter("Tlaloc lowered lambda to dxir:\n").trimEnd() }

    private fun CompileResult.renderMessages(): String =
        messages.joinToString("\n") { "[${it.severity}] ${it.message}" }

    private fun pluginClasspath(): Array<String> = arrayOf(
        System.getProperty("tlaloc.plugin.jar") ?: error("tlaloc.plugin.jar not set"),
        System.getProperty("tlaloc.ir.jar") ?: error("tlaloc.ir.jar not set"),
        System.getProperty("tlaloc.core.jar") ?: error("tlaloc.core.jar not set"),
    )

    private fun compile(stub: String, user: String): CompileResult {
        val tempDir = Files.createTempDirectory("tlaloc-plugin-test").toFile()
        try {
            File(tempDir, "stub.kt").writeText(stub)
            File(tempDir, "user.kt").writeText(user)
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
            return CompileResult(exitCode, collected)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    private data class CompileMessage(
        val severity: CompilerMessageSeverity,
        val message: String,
    )

    private data class CompileResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
    )

    private data class RunResult(
        val exitCode: Int,
        val messages: List<CompileMessage>,
        val stdout: String,
    )

    /**
     * Compiles [stub] + [user], then — if compilation succeeded — runs the produced
     * `MainKt.main()` in a fresh [URLClassLoader] with stdout captured. The classloader
     * bridges to the host JVM's parent so `kotlin.jvm.internal.*` resolves; the compiled
     * output dir is its first URL so user classes take precedence.
     */
    private fun compileAndRun(stub: String, user: String): RunResult {
        val tempDir = Files.createTempDirectory("tlaloc-plugin-run").toFile()
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
            val capturedOut = PrintStream(baos, /* autoFlush = */ true, Charsets.UTF_8)
            val urls = arrayOf(outDir.toURI().toURL())
            val loader = URLClassLoader(urls, javaClass.classLoader)
            return try {
                System.setOut(capturedOut)
                val mainCls = loader.loadClass("MainKt")
                val mainMethod = mainCls.getMethod("main")
                mainMethod.invoke(null)
                RunResult(0, collected, baos.toString(Charsets.UTF_8))
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
            import io.tlaloc.core.DScalar
            fun grad(f: (Float) -> Float): (Float) -> Float = f
            @JvmName("gradDScalar")
            fun grad(f: (DScalar) -> DScalar): (DScalar) -> DScalar = f
            fun grad2(f: (Float, Float) -> Float): (Float, Float) -> Pair<Float, Float> = { a, b -> f(a, b) to f(a, b) }
            @JvmName("grad2DScalar")
            fun grad2(f: (DScalar, DScalar) -> DScalar): (DScalar, DScalar) -> Pair<DScalar, DScalar> = { a, b -> f(a, b) to f(a, b) }
            fun valueAndGrad(f: (Float) -> Float): (Float) -> Pair<Float, Float> = { x -> f(x) to f(x) }
            fun valueAndGrad2(f: (Float, Float) -> Float): (Float, Float) -> Triple<Float, Float, Float> = { a, b -> val v = f(a, b); Triple(v, a, b) }
        """.trimIndent()

        private val AUTOGRAD_STUB_FLOAT_SCALAR = """
            package io.tlaloc.autograd
            import io.tlaloc.core.FloatScalar
            fun grad(f: (FloatScalar) -> FloatScalar): (FloatScalar) -> FloatScalar = f
        """.trimIndent()

        private val AUTOGRAD_STUB_DOUBLE_SCALAR = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DoubleScalar
            fun grad(f: (DoubleScalar) -> DoubleScalar): (DoubleScalar) -> DoubleScalar = f
        """.trimIndent()

        /**
         * A "broken" grad stub used by the IR-transform-execution tests. The runtime-tape
         * path would produce nonsense here — the only way the compiled class returns the
         * correct forward value is if the IR transform actually replaced the `grad { ... }`
         * call site with a synthesised forward lambda. If the transform ever regresses,
         * these tests will fail at runtime with a distinctive wrong output, not silently
         * pass like an identity-stubbed check would.
         */
        private val AUTOGRAD_STUB_BROKEN = """
            package io.tlaloc.autograd
            fun grad(f: (Float) -> Float): (Float) -> Float = { _ -> -1.0f }
            fun grad2(f: (Float, Float) -> Float): (Float, Float) -> Pair<Float, Float> =
                { _, _ -> -1.0f to -1.0f }
            fun valueAndGrad(f: (Float) -> Float): (Float) -> Pair<Float, Float> =
                { _ -> -1.0f to -1.0f }
            fun valueAndGrad2(f: (Float, Float) -> Float): (Float, Float) -> Triple<Float, Float, Float> =
                { _, _ -> Triple(-1.0f, -1.0f, -1.0f) }
        """.trimIndent()


        private val AUTOGRAD_STUB_DTENSOR_SCALAR = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.ScalarShape
            fun grad(f: (DTensor<ScalarShape, F32>) -> DTensor<ScalarShape, F32>):
                    (DTensor<ScalarShape, F32>) -> DTensor<ScalarShape, F32> = f
        """.trimIndent()

        /**
         * Rank-1 DTensor grad stub with a sentinel body: always returns `[-1, -1, -1, -1]`
         * regardless of input. The compileAndRun test only passes when the IR-rewrite path
         * actually replaces the `grad { ... }` call with the synthesised gradient lambda
         * — otherwise the broken stub's sentinel output surfaces and the assertion fails
         * loudly. Mirrors `AUTOGRAD_STUB_BROKEN`'s role for the scalar surface (§0.4.3+).
         */
        private val AUTOGRAD_STUB_BROKEN_RANK1 = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.ScalarShape
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> DTensor<ScalarShape, F32>):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(floatArrayOf(-1.0f, -1.0f, -1.0f, -1.0f)), intArrayOf(4), F32) }
        """.trimIndent()

        /**
         * §0.4.42 — rank-1 param + Float-returning primal. Exercises the new GATHER
         * path: `grad { arr -> arr[0] + arr[1] }` has return type `Float` (primitive),
         * distinct from §0.4.11's `DTensor<ScalarShape, F32>`-returning SUM case.
         * Broken-stub body returns the `[-1, -1, -1, -1]` sentinel so the synthesis
         * path's rank-1 gradient output is pinned against a distinctive wrong value.
         */
        private val AUTOGRAD_STUB_BROKEN_RANK1_TO_FLOAT = """
            package io.tlaloc.autograd
            import io.tlaloc.core.DTensor
            import io.tlaloc.core.F32
            import io.tlaloc.core.HostF32Storage
            import io.tlaloc.core.Rank1
            import io.tlaloc.core.Sym
            fun grad(f: (DTensor<Rank1<Sym>, F32>) -> Float):
                    (DTensor<Rank1<Sym>, F32>) -> DTensor<Rank1<Sym>, F32> =
                { _ -> DTensor(HostF32Storage(floatArrayOf(-1.0f, -1.0f, -1.0f, -1.0f)), intArrayOf(4), F32) }
        """.trimIndent()
    }
}
