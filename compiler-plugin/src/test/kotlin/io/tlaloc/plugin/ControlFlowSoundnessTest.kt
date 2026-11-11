package io.tlaloc.plugin

import kotlin.math.abs
import kotlin.math.cos
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Control flow that once gave a wrong gradient with no diagnostic. Each value is compared with
 * the derivative written out here.
 */
class ControlFlowSoundnessTest {

    private fun assertNear(want: Double, got: Double, what: String) =
        assertTrue(abs(want - got) < 1e-4 * maxOf(1.0, abs(want)), "$what: got $got, want $want")

    @Test
    fun `an if whose branch divides keeps its own value in the gradient`() {
        val r = F64TestHarness.compileAndRun(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            fun main() {
                val g = grad { x: Float ->
                    val y = if (x > 0f) 1f / x else x
                    val s = y.sin()
                    val t = x * 2f
                    val u = t * 3f
                    val v = u + x
                    s + v
                }
                println("pos " + g(2f))
                println("neg " + g(-3f))
            }
            """.trimIndent(),
        )
        assertEquals(0, r.exitCode, r.describe())
        // d/dx [sin(1/x) + 7x] = -cos(1/x)/x^2 + 7; d/dx [sin(x) + 7x] = cos(x) + 7.
        assertNear(-cos(0.5) / 4 + 7, r.value("pos"), "x = 2")
        assertNear(cos(-3.0) + 7, r.value("neg"), "x = -3")
    }

    @Test
    fun `a return inside a branch that does not end the lambda is refused`() {
        val r = F64TestHarness.compileAndRun(
            """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    val scale = if (x > 1f) { return@grad x * x } else { 2f }
                    x * scale
                }
                println(g(3f))
            }
            """.trimIndent(),
        )
        assertTrue(r.exitCode != 0, r.describe())
        assertTrue(r.describe().contains("a `return` before the end of the lambda"), r.describe())
    }

    @Test
    fun `a return inside a loop is refused`() {
        val r = F64TestHarness.compileAndRun(
            """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    var s = x
                    for (i in 0 until 3) {
                        if (s > 10f) { return@grad s }
                        s = s * x
                    }
                    s
                }
                println(g(3f))
            }
            """.trimIndent(),
        )
        assertTrue(r.exitCode != 0, r.describe())
    }

    @Test
    fun `a return at the end of the lambda or of its last branch is accepted`() {
        val r = F64TestHarness.compileAndRun(
            """
            import io.tlaloc.autograd.grad
            fun main() {
                val g = grad { x: Float ->
                    if (x > 1f) { return@grad x * x } else { return@grad 2f * x }
                }
                println("a " + g(3f))
                println("b " + g(0.5f))
            }
            """.trimIndent(),
        )
        assertEquals(0, r.exitCode, r.describe())
        assertNear(6.0, r.value("a"), "x = 3")
        assertNear(2.0, r.value("b"), "x = 0.5")
    }

    @Test
    fun `a loop with a negative symbolic count runs no iterations`() {
        val r = F64TestHarness.compileAndRun(
            """
            import io.tlaloc.autograd.grad2
            fun main() {
                val g = grad2 { x: Float, n: Int ->
                    var s = x
                    for (i in 0 until n) { s = s * 2f }
                    s
                }
                println("three " + g(1f, 3).first)
                println("zero " + g(1f, 0).first)
                println("negative " + g(1f, -2).first)
            }
            """.trimIndent(),
        )
        assertEquals(0, r.exitCode, r.describe())
        assertNear(8.0, r.value("three"), "n = 3")
        assertNear(1.0, r.value("zero"), "n = 0")
        assertNear(1.0, r.value("negative"), "n = -2")
    }
}

/**
 * Random scalar programs with branches read after the branch, each differentiated by the plugin
 * and by central differences of the same source. A program the plugin refuses is counted, and
 * must not be most of them; a program it accepts must agree.
 */
class ControlFlowFuzzTest {

    private class Gen(seed: Long) {
        private val r = java.util.Random(seed)
        private var vars = mutableListOf("x")
        private var next = 0

        private fun any() = vars[r.nextInt(vars.size)]

        private fun lit() = "%.3f".format(0.25 + 1.5 * r.nextDouble())

        private fun expr(): String {
            val a = any()
            val b = any()
            return when (r.nextInt(8)) {
                0 -> "$a + $b"
                1 -> "$a - $b"
                2 -> "$a * $b * 0.5"
                3 -> "$a / (1.5 + $b * $b)"
                4 -> "$a.sin()"
                5 -> "$a.cos() * ${lit()}"
                6 -> "($a * 0.3).exp()"
                else -> "($a * $a + 1.0).sqrt()"
            }
        }

        private fun fresh(): String {
            val n = "v${next++}"
            return n
        }

        fun body(): String {
            vars = mutableListOf("x")
            next = 0
            val out = StringBuilder()
            repeat(3 + r.nextInt(6)) {
                val n = fresh()
                if (r.nextInt(3) == 0) {
                    out.appendLine("val $n = if (${any()} > ${"%.3f".format(r.nextDouble() - 0.5)}) (${expr()}) else (${expr()})")
                } else {
                    out.appendLine("val $n = ${expr()}")
                }
                vars.add(n)
            }
            out.appendLine("${vars.last()} + ${any()} * 0.5 + x")
            return out.toString()
        }
    }

    @Test
    fun `gradients agree with central differences`() {
        val gen = Gen(20261111L)
        val count = 60
        val src = StringBuilder(
            """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            import kotlin.math.abs

            """.trimIndent() + "\n",
        )
        val calls = StringBuilder()
        for (k in 0 until count) {
            val body = gen.body()
            src.appendLine("fun f$k(x: Double): Double {\n${body.trim().lines().dropLast(1).joinToString("\n")}\nreturn ${body.trim().lines().last()}\n}")
            src.appendLine("fun g$k(): (Double) -> Double = grad { x: Double ->\n$body}")
            calls.appendLine("    check($k, ::f$k, ::g$k)")
        }
        src.appendLine(
            """
            var refused = 0
            var wrong = 0
            fun check(k: Int, f: (Double) -> Double, g: () -> (Double) -> Double) {
                val points = doubleArrayOf(-1.7, -0.6, 0.35, 0.9, 2.3)
                val d = try { g() } catch (e: Throwable) { refused++; return }
                for (x in points) {
                    val h = 1e-6
                    val fd = (f(x + h) - f(x - h)) / (2 * h)
                    if (abs(f(x + h * 100) - f(x - h * 100) - 200 * h * fd) > 1e-3) continue
                    val got = try { d(x) } catch (e: Throwable) { refused++; return }
                    if (abs(got - fd) > 1e-5 * maxOf(1.0, abs(fd))) {
                        wrong++
                        println("wrong program ${'$'}k at ${'$'}x: got ${'$'}got, central difference ${'$'}fd")
                    }
                }
            }
            fun main() {
            ${calls}
                println("refused " + refused)
                println("wrong " + wrong)
            }
            """.trimIndent(),
        )
        val r = F64TestHarness.compileAndRun(
            src.toString(),
            pluginOptions = arrayOf("plugin:io.tlaloc.plugin:strictLowering=false"),
        )
        assertEquals(0, r.exitCode, r.describe())
        assertEquals(0.0, r.value("wrong"), r.stdout)
        assertTrue(r.value("refused") < count / 2.0, r.stdout)
    }
}
