package io.tlaloc.ir.passes

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.tanh
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [DxirInterpreterF64] on F64 gradient graphs of the shape the plugin lowers `grad {}`
 * bodies to. Gradients are compared with fourth-order central differences (`h = 1e-3`,
 * good to about 1e-12 on these O(1) functions) of the same function in plain Double;
 * tolerance 1e-9 of the largest entry, which an F32 value anywhere in the graph (1e-8 or
 * worse) fails.
 */
class DxirInterpreterF64Test {

    private fun t(dtype: DType, vararg dims: Int) = DxirType(dtype, dims.toList())

    private fun fd4(x: DoubleArray, f: (DoubleArray) -> Double): DoubleArray = DoubleArray(x.size) { i ->
        val h = 1e-3
        fun at(d: Double) = f(x.copyOf().also { it[i] += d })
        (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h)
    }

    private fun assertClose(want: DoubleArray, got: DoubleArray, tol: Double, what: String) {
        assertEquals(want.size, got.size, what)
        val scale = max(1e-300, want.maxOf { abs(it) })
        for (i in want.indices) {
            assertTrue(abs(want[i] - got[i]) <= tol * scale, "$what[$i] = ${got[i]}, want ${want[i]}")
        }
    }

    private val m = doubleArrayOf(0.31, -0.72, 1.13, 0.05, -0.44, 0.87, -1.29, 0.66, 0.92, 0.18, -0.35, -0.81)
    private val v = doubleArrayOf(0.25, -0.6, 0.9, 0.12)

    /** `loss = Σᵢ (Σⱼ bᵢⱼ)(maxⱼ bᵢⱼ) + 0.7·mean(b)`, `b = (0.3m² + eᵐ − tanh(m + v)·v) / (m² + 1.5)`. */
    private fun elementwise(dtype: DType): DxirFunction = DxirBuilder.function("elementwise") {
        fun c(x: Double) = const(if (dtype == F64) x else x.toFloat(), t(dtype))
        val mm = param("m", t(dtype, 3, 4))
        val vv = param("v", t(dtype, 4))
        val m2 = op(OpKind.MUL, listOf(mm, mm), t(dtype, 3, 4))
        fun splat(x: Double) = op(OpKind.BROADCAST, listOf(c(x), m2), t(dtype, 3, 4), attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
        val a = op(
            OpKind.SUB,
            listOf(
                op(OpKind.ADD, listOf(op(OpKind.MUL, listOf(m2, splat(0.3)), t(dtype, 3, 4)), op(OpKind.EXP, listOf(mm), t(dtype, 3, 4))), t(dtype, 3, 4)),
                op(OpKind.MUL, listOf(op(OpKind.TANH, listOf(op(OpKind.ADD, listOf(mm, vv), t(dtype, 3, 4))), t(dtype, 3, 4)), vv), t(dtype, 3, 4)),
            ),
            t(dtype, 3, 4),
        )
        val b = op(OpKind.DIV, listOf(a, op(OpKind.ADD, listOf(m2, splat(1.5)), t(dtype, 3, 4))), t(dtype, 3, 4))
        val s = op(OpKind.SUM, listOf(b), t(dtype, 3), attrs = mapOf("reduction_dims" to listOf(1)))
        val mx = op(OpKind.MAX, listOf(b), t(dtype, 3), attrs = mapOf("reduction_dims" to listOf(1)))
        val left = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(s, mx), t(dtype, 3))), t(dtype))
        val right = op(OpKind.MUL, listOf(c(0.7), op(OpKind.MEAN, listOf(b), t(dtype))), t(dtype))
        listOf(op(OpKind.ADD, listOf(left, right), t(dtype)))
    }

    private fun elementwiseLoss(mm: DoubleArray, vv: DoubleArray): Double {
        val bb = DoubleArray(12) { k ->
            val j = k % 4
            (mm[k] * mm[k] * 0.3 + exp(mm[k]) - tanh(mm[k] + vv[j]) * vv[j]) / (mm[k] * mm[k] + 1.5)
        }
        var loss = 0.0
        for (i in 0 until 3) {
            var s = 0.0
            var mx = Double.NEGATIVE_INFINITY
            for (j in 0 until 4) { s += bb[i * 4 + j]; mx = max(mx, bb[i * 4 + j]) }
            loss += s * mx
        }
        return loss + 0.7 * bb.sum() / 12
    }

    @Test
    fun f64GradientOfElementwiseBroadcastAndReductionsMatchesFiniteDifferences() {
        val g = DxirReverseTransform.apply(elementwise(F64), includeForward = true)
        val out = DxirInterpreterF64.evalFunction(g, listOf(m, v))
        assertClose(doubleArrayOf(elementwiseLoss(m, v)), out[0], 1e-14, "value")
        assertClose(fd4(m) { elementwiseLoss(it, v) }, out[1], 1e-9, "d/dm")
        assertClose(fd4(v) { elementwiseLoss(m, it) }, out[2], 1e-9, "d/dv")
    }

    private val a = m
    private val b = doubleArrayOf(0.7, -0.2, 0.15, 1.1, -0.9, 0.4, 0.33, -0.55)

    /** `Σ softmax(ab)·log softmax(ab) + mean((aᵀa)²)`. */
    private fun matmulSoftmax(dtype: DType): DxirFunction = DxirBuilder.function("matmul_softmax") {
        val aa = param("a", t(dtype, 3, 4))
        val bb = param("b", t(dtype, 4, 2))
        val c = op(OpKind.MATMUL, listOf(aa, bb), t(dtype, 3, 2))
        val s = op(OpKind.SOFTMAX, listOf(c), t(dtype, 3, 2), attrs = mapOf("axis" to 1))
        val e = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(s, op(OpKind.LOG, listOf(s), t(dtype, 3, 2))), t(dtype, 3, 2))), t(dtype))
        val tt = op(OpKind.MATMUL, listOf(op(OpKind.TRANSPOSE, listOf(aa), t(dtype, 4, 3), attrs = mapOf("permutation" to listOf(1, 0))), aa), t(dtype, 4, 4))
        listOf(op(OpKind.ADD, listOf(e, op(OpKind.MEAN, listOf(op(OpKind.MUL, listOf(tt, tt), t(dtype, 4, 4))), t(dtype))), t(dtype)))
    }

    private fun matmulSoftmaxLoss(aa: DoubleArray, bb: DoubleArray): Double {
        val c = DoubleArray(6) { k -> (0 until 4).sumOf { aa[(k / 2) * 4 + it] * bb[it * 2 + k % 2] } }
        var loss = 0.0
        for (i in 0 until 3) {
            val z = exp(c[2 * i]) + exp(c[2 * i + 1])
            for (j in 0 until 2) {
                val p = exp(c[2 * i + j]) / z
                loss += p * ln(p)
            }
        }
        var tt = 0.0
        for (p in 0 until 4) for (q in 0 until 4) {
            val x = (0 until 3).sumOf { aa[it * 4 + p] * aa[it * 4 + q] }
            tt += x * x
        }
        return loss + tt / 16
    }

    @Test
    fun f64GradientOfMatmulTransposeAndSoftmaxMatchesFiniteDifferences() {
        val g = DxirReverseTransform.apply(matmulSoftmax(F64))
        val out = DxirInterpreterF64.evalFunction(g, listOf(a, b))
        assertClose(fd4(a) { matmulSoftmaxLoss(it, b) }, out[0], 1e-9, "d/da")
        assertClose(fd4(b) { matmulSoftmaxLoss(a, it) }, out[1], 1e-9, "d/db")
    }

    @Test
    fun f64ConstantsAndInputsKeepTheirLowBits() {
        // d/dx Σ 0.1·x³ at x = 1 + 1e-10: F32 rounds x to 1 and 0.1 to 0.1f, 1.5e-8 away.
        val fn = DxirBuilder.function("cube") {
            val x = param("x", t(F64, 2))
            val c = op(OpKind.BROADCAST, listOf(const(0.1, t(F64)), x), t(F64, 2), attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(c, op(OpKind.MUL, listOf(x, op(OpKind.MUL, listOf(x, x), t(F64, 2))), t(F64, 2))), t(F64, 2))), t(F64)))
        }
        val x = doubleArrayOf(1.0 + 1e-10, 2.0)
        val g = DxirInterpreterF64.evalFunction(DxirReverseTransform.apply(fn), listOf(x)).single()
        assertClose(DoubleArray(2) { 0.3 * x[it] * x[it] }, g, 1e-15, "gradient")
    }

    @Test
    fun f32NodesKeepF32WidthAndMatchTheF32InterpreterOnElementwiseOps() {
        // An elementwise F32 graph: each op rounds once to F32 in both interpreters, so the
        // bits agree. (Reductions are not in this graph: DxirInterpreterF64 accumulates
        // them in Double.)
        val fn = DxirBuilder.function("ew") {
            val x = param("x", t(F32, 5))
            val y = param("y", t(F32, 5))
            listOf(op(OpKind.DIV, listOf(op(OpKind.EXP, listOf(op(OpKind.MUL, listOf(x, y), t(F32, 5))), t(F32, 5)), op(OpKind.ADD, listOf(x, y), t(F32, 5))), t(F32, 5)))
        }
        val x = floatArrayOf(0.1f, -0.7f, 1.3f, 2.9f, -3.3f)
        val y = floatArrayOf(1.7f, 0.2f, -0.45f, 0.61f, 1.9f)
        val f32 = DxirInterpreter.evalFunction(fn, listOf(x, y)).single()
        val f64 = DxirInterpreterF64.evalFunction(
            fn,
            listOf(DoubleArray(5) { x[it].toDouble() }, DoubleArray(5) { y[it].toDouble() }),
        ).single()
        for (i in 0 until 5) assertEquals(f32[i].toDouble(), f64[i], "element $i")
    }

    @Test
    fun anF64ConstantHoldingAFloatIsRefusedAtConstruction() {
        val e = assertFailsWith<IllegalArgumentException> {
            DxirBuilder.function("bad") { listOf(const(0.1f, t(F64))) }
        }
        assertTrue("must be built from a Double" in (e.message ?: ""), e.message)
    }
}
