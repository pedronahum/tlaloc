package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * EXPM1 and LOG1P: values near 0 where `exp(x) − 1` and `log(1 + x)` cancel, and their
 * gradients through both transforms (`Σ f(x) ⊙ w`, so a dropped upstream factor cannot
 * pass, and the JVP against the VJP through a body that chains both).
 */
class DxirExpm1Log1pTest {

    private val scalar = DxirType(F32, emptyList())
    private val r1 = DxirType(F32, listOf(4))

    private fun unary(kind: OpKind, dtype: io.tlaloc.core.DType = F32) = DxirBuilder.function("u") {
        val t = DxirType(dtype, listOf(4))
        val x = param("x", t)
        listOf(op(kind, listOf(x), t))
    }

    @Test
    fun valuesNearZeroKeepTheirPrecision() {
        val x = floatArrayOf(1e-10f, -3e-9f, 0.5f, -0.25f)
        val e = DxirInterpreter.evalFunction(unary(OpKind.EXPM1), listOf(x))[0]
        val l = DxirInterpreter.evalFunction(unary(OpKind.LOG1P), listOf(x))[0]
        for (i in x.indices) {
            val xd = x[i].toDouble()
            assertTrue(abs(e[i] - expm1(xd)) <= 1e-7 * abs(expm1(xd)), "expm1(${x[i]}) = ${e[i]}")
            assertTrue(abs(l[i] - ln1p(xd)) <= 1e-7 * abs(ln1p(xd)), "log1p(${x[i]}) = ${l[i]}")
        }
        // The composed forms lose the small arguments entirely in f32.
        assertEquals(0f, kotlin.math.exp(1e-10f) - 1f)

        val xd = doubleArrayOf(1e-20, -1e-18, 2.0, -0.5)
        val e64 = DxirInterpreterF64.evalFunction(unary(OpKind.EXPM1, F64), listOf(xd))[0]
        val l64 = DxirInterpreterF64.evalFunction(unary(OpKind.LOG1P, F64), listOf(xd))[0]
        for (i in xd.indices) {
            assertEquals(expm1(xd[i]), e64[i])
            assertEquals(ln1p(xd[i]), l64[i])
        }
    }

    @Test
    fun expm1GradientMatchesAnalytic() {
        // loss = Σ expm1(x) ⊙ w  ⇒  dx = w ⊙ exp(x), dw = expm1(x).
        val fn = DxirBuilder.function("expm1_loss") {
            val x = param("x", r1)
            val w = param("w", r1)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(op(OpKind.EXPM1, listOf(x), r1), w), r1)), scalar))
        }
        val x = floatArrayOf(0.3f, -0.7f, 1e-9f, 2.0f)
        val w = floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f)
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        for (i in 0 until 4) {
            val xd = x[i].toDouble()
            assertTrue(abs(out[0][i] - w[i] * exp(xd)) < 1e-5, "dx[$i] = ${out[0][i]}")
            assertTrue(abs(out[1][i] - expm1(xd)) <= 1e-7 * abs(expm1(xd)), "dw[$i] = ${out[1][i]}")
        }
    }

    @Test
    fun log1pGradientMatchesAnalytic() {
        // loss = Σ log1p(x) ⊙ w  ⇒  dx = w / (1 + x), dw = log1p(x).
        val fn = DxirBuilder.function("log1p_loss") {
            val x = param("x", r1)
            val w = param("w", r1)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(op(OpKind.LOG1P, listOf(x), r1), w), r1)), scalar))
        }
        val x = floatArrayOf(0.3f, -0.7f, 1e-9f, 5.0f)
        val w = floatArrayOf(1.5f, -0.8f, 0.25f, 2.0f)
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        for (i in 0 until 4) {
            val xd = x[i].toDouble()
            assertTrue(abs(out[0][i] - w[i] / (1 + xd)) < 1e-5, "dx[$i] = ${out[0][i]}")
            assertTrue(abs(out[1][i] - ln1p(xd)) <= 1e-7 * abs(ln1p(xd)), "dw[$i] = ${out[1][i]}")
        }
    }

    @Test
    fun jvpMatchesVjpThroughBothOps() {
        // f(x, w) = Σ log1p(expm1(x) ⊙ w ⊙ w).
        val fn = DxirBuilder.function("chain") {
            val x = param("x", r1)
            val w = param("w", r1)
            val e = op(OpKind.EXPM1, listOf(x), r1)
            val p = op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(e, w), r1), w), r1)
            listOf(op(OpKind.SUM, listOf(op(OpKind.LOG1P, listOf(p), r1)), scalar))
        }
        val x = floatArrayOf(0.3f, -0.2f, 0.9f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f)
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f)
        val vw = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f)
        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in 0 until 4) dot += grads[0][i].toDouble() * vx[i] + grads[1][i].toDouble() * vw[i]
        val tangent = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(x, w, vx, vw))[1].single()
        assertTrue(abs(dot - tangent) < 1e-5, "⟨grad, v⟩ = $dot, tangent = $tangent")
    }
}
