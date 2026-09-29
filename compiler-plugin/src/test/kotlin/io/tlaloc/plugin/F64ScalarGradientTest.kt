package io.tlaloc.plugin

import io.tlaloc.plugin.F64TestHarness.assertClose
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.test.Test

/**
 * Scalar `Double` lambdas under every scalar transformation, checked against analytic
 * derivatives computed here in Double.
 *
 * Tolerance 1e-13 relative: each body is a handful of double operations, each rounding by
 * at most one ulp (1.1e-16), and `kotlin.math` is accurate to about one ulp, so body and
 * reference agree to about 1e-15. An F32 value anywhere in the chain rounds by up to 6e-8,
 * which this bound rejects by five orders of magnitude.
 */
class F64ScalarGradientTest {

    private val tol = 1e-13

    @Test
    fun `grad, grad2, valueAndGrad, jvp and vjp of Double lambdas are double precision`() {
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.grad2
            import io.tlaloc.autograd.valueAndGrad
            import io.tlaloc.autograd.valueAndGrad2
            import io.tlaloc.autograd.jvp
            import io.tlaloc.autograd.valueAndJvp
            import io.tlaloc.autograd.vjp
            import io.tlaloc.core.exp
            import io.tlaloc.core.log
            import io.tlaloc.core.sin
            import io.tlaloc.core.cos
            import io.tlaloc.core.tanh
            import io.tlaloc.core.sqrt
            fun main() {
                val cube = grad { x: Double -> x * x * x }
                println("cube " + cube(1.1))
                val mixed = grad { x: Double -> x.exp() * x.sin() + x.log() / x.sqrt() - x.cos() * x.tanh() }
                println("mixed " + mixed(0.7))
                val g2 = grad2 { x: Double, y: Double -> x * x * y + y.exp() / x }
                val (gx, gy) = g2(1.3, -0.4)
                println("g2 " + gx + "," + gy)
                val (v, d) = valueAndGrad { x: Double -> x.sin() * x.sin() }(0.3)
                println("vg " + v + "," + d)
                val vg2 = valueAndGrad2 { x: Double, y: Double -> x / y }(2.0, 3.0)
                println("vg2 " + vg2.first + "," + vg2.second + "," + vg2.third)
                val j = jvp { x: Double -> x.exp() * x }
                println("jvp " + j(0.9, 0.25))
                val (jv, jd) = valueAndJvp { x: Double -> x.tanh() }(0.4, 2.0)
                println("vjvp " + jv + "," + jd)
                val p = vjp { x: Double -> x.cos() * x }
                println("vjp " + p(1.7, -0.5))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        assertClose(doubleArrayOf(3 * 1.1 * 1.1), r.values("cube"), tol, "cube")
        val x = 0.7
        val mixed = exp(x) * sin(x) + exp(x) * cos(x) +
            (1 / x * sqrt(x) - ln(x) * 0.5 / sqrt(x)) / x -
            (-sin(x) * tanh(x) + cos(x) * (1 - tanh(x) * tanh(x)))
        assertClose(doubleArrayOf(mixed), r.values("mixed"), tol, "mixed")
        assertClose(
            doubleArrayOf(2 * 1.3 * -0.4 - exp(-0.4) / (1.3 * 1.3), 1.3 * 1.3 + exp(-0.4) / 1.3),
            r.values("g2"), tol, "grad2",
        )
        assertClose(doubleArrayOf(sin(0.3) * sin(0.3), 2 * sin(0.3) * cos(0.3)), r.values("vg"), tol, "valueAndGrad")
        assertClose(doubleArrayOf(2.0 / 3.0, 1.0 / 3.0, -2.0 / 9.0), r.values("vg2"), tol, "valueAndGrad2")
        assertClose(doubleArrayOf((exp(0.9) * 0.9 + exp(0.9)) * 0.25), r.values("jvp"), tol, "jvp")
        assertClose(doubleArrayOf(tanh(0.4), (1 - tanh(0.4) * tanh(0.4)) * 2.0), r.values("vjvp"), tol, "valueAndJvp")
        assertClose(doubleArrayOf((-sin(1.7) * 1.7 + cos(1.7)) * -0.5), r.values("vjp"), tol, "vjp")
    }

    @Test
    fun `inputs, literals and captured Doubles never pass through a Float`() {
        // Each value below differs from its F32 rounding by more than the tolerance allows:
        //  - the input 1 + 1e-10 rounds to exactly 1f, which would give d(x³)/dx = 3;
        //  - the literal 0.1 and the captured 0.3 round to 0.1f and 0.3f, 1.5e-8 and 4e-9
        //    relative away;
        //  - the captured 1e-9 offset is below F32's resolution at 1 (6e-8), so it vanishes.
        // `jvp` takes no captured runtime value (any dtype), so it reads a `const val`.
        val src = """
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.jvp
            fun scaleOf(v: Double) = v
            const val C = 0.3
            fun main() {
                val c = scaleOf(0.3)
                val off = scaleOf(1e-9)
                val g = grad { x: Double -> x * x * x * 0.1 * c }
                println("g " + g(1.0 + 1e-10))
                val h = grad { x: Double -> (x + off) * (x + off) }
                println("h " + h(1.0))
                val j = jvp { x: Double -> x * x * C }
                println("j " + j(1.0 + 1e-10, 1.0))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        val x = 1.0 + 1e-10
        assertClose(doubleArrayOf(3 * x * x * 0.1 * 0.3), r.values("g"), tol, "input and constants")
        assertClose(doubleArrayOf(2 * (1.0 + 1e-9)), r.values("h"), 1e-15, "captured offset")
        assertClose(doubleArrayOf(2 * x * 0.3), r.values("j"), tol, "jvp")
    }

    @Test
    fun `customVjp and customJvp run on Doubles`() {
        val src = """
            import io.tlaloc.autograd.customJvp
            import io.tlaloc.autograd.customVjp
            import io.tlaloc.autograd.grad
            import io.tlaloc.autograd.jvp
            import io.tlaloc.core.exp
            fun main() {
                val g = grad { x: Double ->
                    val sq = customVjp({ t: Double -> t * t }, { u: Double, t: Double -> u * 2.0 * t * 0.1 })
                    sq(x) * 3.0
                }
                println("vjp " + g(0.7))
                val j = jvp { x: Double ->
                    val e = customJvp({ t: Double -> t.exp() }, { t: Double, dt: Double -> t.exp() * dt * 0.1 })
                    e(x)
                }
                println("jvp " + j(0.7, 2.0))
            }
        """.trimIndent()
        val r = F64TestHarness.run(src)
        // The user rules are scaled by 0.1 on purpose: only the user rule, run in Double,
        // produces these numbers.
        assertClose(doubleArrayOf(3.0 * 2.0 * 0.7 * 0.1), r.values("vjp"), tol, "customVjp")
        assertClose(doubleArrayOf(exp(0.7) * 2.0 * 0.1), r.values("jvp"), tol, "customJvp")
    }
}
