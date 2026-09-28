package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.times
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The RK4 integrator: its order of accuracy, and the derivatives of a captured
 * integration with respect to the initial state and to a parameter of `f`.
 */
class OdeTest {

    /** RK4 in Double on `y' = M·y` for a 2×2 `M`, the reference for finite differences. */
    private fun rk4Linear(m: DoubleArray, y0: DoubleArray, dt: Double, steps: Int): DoubleArray {
        fun f(y: DoubleArray) = doubleArrayOf(m[0] * y[0] + m[1] * y[1], m[2] * y[0] + m[3] * y[1])
        fun axpy(y: DoubleArray, a: Double, k: DoubleArray) = DoubleArray(2) { y[it] + a * k[it] }
        var y = y0
        repeat(steps) {
            val k1 = f(y)
            val k2 = f(axpy(y, dt / 2, k1))
            val k3 = f(axpy(y, dt / 2, k2))
            val k4 = f(axpy(y, dt, k3))
            y = DoubleArray(2) { y[it] + dt / 6 * (k1[it] + 2 * k2[it] + 2 * k3[it] + k4[it]) }
        }
        return y
    }

    @Test
    fun rk4IsFourthOrder() {
        // y' = −y, y(0) = 1, to t = 2. One RK4 step multiplies y by
        // R(−dt) = 1 − dt + dt²/2 − dt³/6 + dt⁴/24 exactly (a wrong coefficient changes
        // R), and the error |R(−dt)ⁿ − e⁻²| falls by 19.75 from n = 4 to 8 (the
        // asymptotic ratio 16 needs smaller steps than F32 can resolve here).
        fun rk4Of(steps: Int) = rk4(Tensors.f32Scalar(1f), 0f, 2f / steps, steps) { _, y -> y * -1f }.hostF32().single()
        fun exactRk4(steps: Int): Double {
            val z = -2.0 / steps
            return Math.pow(1 + z + z * z / 2 + z * z * z / 6 + z * z * z * z / 24, steps.toDouble())
        }
        for (steps in listOf(4, 8)) {
            assertTrue(abs(rk4Of(steps) - exactRk4(steps)) < 2e-7, "$steps steps: ${rk4Of(steps)} vs ${exactRk4(steps)}")
        }
        val ratio = abs(rk4Of(4) - exp(-2.0)) / abs(rk4Of(8) - exp(-2.0))
        assertTrue(abs(ratio - 19.75) < 0.5, "error ratio for dt → dt/2 is $ratio, RK4's is 19.75")
        // Zero steps return the initial state; t is passed through (y' = cos t).
        assertEquals(3f, rk4(Tensors.f32Scalar(3f), 0f, 0.1f, 0) { _, y -> y }.hostF32().single())
        val s = rk4(Tensors.f32Scalar(0f), 0f, 0.1f, 20) { t, y -> Tensors.f32Scalar(cos(t)) }
        assertTrue(abs(s.hostF32().single() - sin(2.0)) < 1e-5, "∫₀² cos = ${s.hostF32().single()}")
    }

    /** `Σ W ⊙ y(T)` for `y' = M·y`, captured with `M` and `y0` as leaves. */
    private fun captured(steps: Int, dt: Float, w: FloatArray) = captureN(
        listOf(
            Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(0f, 1f, -4f, -0.3f)),
            Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(1f, 0f)),
            Tensors.f32Matrix<Sym, Sym>(2, 1, w),
        ),
        name = "oscillator",
    ) { leaves ->
        @Suppress("UNCHECKED_CAST")
        val m = leaves[0] as Tracer<Rank2<Sym, Sym>>
        @Suppress("UNCHECKED_CAST")
        val y0 = leaves[1] as Tracer<Rank2<Sym, Sym>>
        @Suppress("UNCHECKED_CAST")
        val weights = leaves[2] as Tracer<Rank2<Sym, Sym>>
        val yT = rk4(y0, 0f, dt, steps) { _, y -> m matmul y }
        (yT * weights).sum() as Tracer<Shape>
    }

    @Test
    fun gradientAndTangentOfACapturedIntegrationMatchFiniteDifferences() {
        // A damped oscillator (ω = 2, damping 0.3) over 40 steps of 0.05. The
        // reference is central differences of the Double integrator (h = 1e-5,
        // error ~1e-10); the captured graph runs in F32 through 160 matrix
        // products, so the bound is 1e-4 of the largest entry.
        val m = doubleArrayOf(0.0, 1.0, -4.0, -0.3)
        val y0 = doubleArrayOf(1.0, 0.0)
        val w = floatArrayOf(0.7f, -1.3f)
        val fn = captured(40, 0.05f, w)
        fun loss(mm: DoubleArray, yy: DoubleArray): Double {
            val y = rk4Linear(mm, yy, 0.05, 40)
            return w[0] * y[0] + w[1] * y[1]
        }
        val inputs = listOf(FloatArray(4) { m[it].toFloat() }, floatArrayOf(1f, 0f), w)
        assertTrue(abs(DxirInterpreter.evalFunction(fn, inputs).single().single() - loss(m, y0)) < 1e-5)
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), inputs)
        val h = 1e-5
        fun fd(x: DoubleArray, l: (DoubleArray) -> Double) = DoubleArray(x.size) { i ->
            (l(x.copyOf().also { it[i] += h }) - l(x.copyOf().also { it[i] -= h })) / (2 * h)
        }
        val dM = fd(m) { loss(it, y0) }
        val dY = fd(y0) { loss(m, it) }
        val scale = maxOf(dM.maxOf { abs(it) }, dY.maxOf { abs(it) })
        for (i in 0 until 4) assertTrue(abs(g[0][i] - dM[i]) <= 1e-4 * scale, "dM[$i] = ${g[0][i]}, want ${dM[i]}")
        for (i in 0 until 2) assertTrue(abs(g[1][i] - dY[i]) <= 1e-4 * scale, "dy0[$i] = ${g[1][i]}, want ${dY[i]}")

        // Forward mode: the tangent along (δM, δy0) equals ⟨∇, δ⟩.
        val vm = floatArrayOf(0.2f, -0.1f, 0.5f, 0.3f)
        val vy = floatArrayOf(-0.4f, 0.6f)
        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), inputs + listOf(vm, vy, FloatArray(2)),
        )
        val want = (0 until 4).sumOf { dM[it] * vm[it] } + (0 until 2).sumOf { dY[it] * vy[it] }
        assertTrue(abs(jvp[1].single() - want) <= 1e-4 * scale, "tangent ${jvp[1].single()}, want $want")
    }

    @Test
    fun gradientDescentThroughTheIntegratorRecoversADampingCoefficient() {
        // Observe y(T) of the oscillator with damping 0.3, then fit the damping
        // entry of M starting from 1.0, by gradient descent on the squared miss
        // (step 0.5: the loss's curvature at the optimum is about 2.3).
        // One capture serves every step: M is an input of the captured function.
        val target = rk4Linear(doubleArrayOf(0.0, 1.0, -4.0, -0.3), doubleArrayOf(1.0, 0.0), 0.05, 40)
        val fn = captureN(
            listOf(
                Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(0f, 1f, -4f, -1f)),
                Tensors.f32Matrix<Sym, Sym>(2, 1, floatArrayOf(target[0].toFloat(), target[1].toFloat())),
            ),
            name = "fit",
        ) { leaves ->
            @Suppress("UNCHECKED_CAST")
            val m = leaves[0] as Tracer<Rank2<Sym, Sym>>
            @Suppress("UNCHECKED_CAST")
            val obs = leaves[1] as Tracer<Rank2<Sym, Sym>>
            // The known initial state, a constant on the same tape.
            val y0 = obs.constant<Rank2<Sym, Sym>>(floatArrayOf(1f, 0f), intArrayOf(2, 1))
            val miss = rk4(y0, 0f, 0.05f, 40) { _, y -> m matmul y } - obs
            (miss * miss).sum() as Tracer<Shape>
        }
        val grad = DxirReverseTransform.apply(fn)
        var damping = 1.0f
        val obs = floatArrayOf(target[0].toFloat(), target[1].toFloat())
        repeat(100) {
            val g = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(0f, 1f, -4f, -damping), obs))
            // M[1][1] = −damping, so d loss / d damping = −d loss / d M[1][1].
            damping -= 0.5f * -g[0][3]
        }
        assertTrue(abs(damping - 0.3f) < 1e-3f, "fitted damping $damping, want 0.3")
    }
}
