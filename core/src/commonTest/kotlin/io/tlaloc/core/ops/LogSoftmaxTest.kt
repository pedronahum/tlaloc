package io.tlaloc.core.ops

import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.SoftmaxKernels
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.expm1
import io.tlaloc.core.hostF64
import io.tlaloc.core.log1p
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The host `logSoftmax` and `crossEntropyLoss` against exact values: a confident row's
 * maximum keeps its log probability `−log1p(Σ_{j≠i} exp(x_j − M))`, which `x − M −
 * log Σ exp(x − M)` rounds to 0 (arXiv 2610.31642).
 */
class LogSoftmaxTest {

    private fun f64(rows: Int, cols: Int, vararg v: Double): DTensor<Rank2<Sym, Sym>, F64> =
        DTensor(HostF64Storage(v.copyOf()), intArrayOf(rows, cols), F64)

    @Test
    fun confidentF32RowKeepsItsLogProbability() {
        val x = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(9f, -9f, -9f))
        val ls = x.logSoftmax().hostF32()
        val want0 = -ln1p(2 * exp(-18.0))
        assertTrue(abs(ls[0] - want0) / abs(want0) < 1e-6, "log p_max ${ls[0]} against $want0")
        assertTrue(abs(ls[1] - (-18.0 + want0)) < 1e-5, "log p_1 ${ls[1]}")
        val labels = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(1f, 0f, 0f))
        val loss = crossEntropyLoss(x, labels).hostF32().single()
        assertTrue(abs(loss + want0) / abs(want0) < 1e-6, "cross entropy $loss against ${-want0}")
    }

    @Test
    fun confidentF64RowPastItsCliff() {
        val ls = f64(1, 3, 40.0, 0.0, 0.0).logSoftmax().hostF64()
        val want0 = -ln1p(2 * exp(-40.0))
        assertTrue(ls[0] != 0.0 && abs(ls[0] - want0) / abs(want0) < 1e-14, "log p_max ${ls[0]} against $want0")
    }

    @Test
    fun axisZeroTiesAndInfinities() {
        // Columns of a [2, 2]: (1, 1) ties, (0, −inf).
        val ls = SoftmaxKernels.logSoftmax(doubleArrayOf(1.0, 0.0, 1.0, Double.NEGATIVE_INFINITY), intArrayOf(2, 2), 0)
        assertTrue(abs(ls[0] + ln(2.0)) < 1e-15 && abs(ls[2] + ln(2.0)) < 1e-15, ls.toList().toString())
        assertEquals(0.0, ls[1])
        assertEquals(Double.NEGATIVE_INFINITY, ls[3])
        val nan = SoftmaxKernels.logSoftmax(doubleArrayOf(Double.NaN, 0.0), intArrayOf(2), 0)
        assertTrue(nan.all { it.isNaN() })
    }

    @Test
    fun ordinaryRowMatchesTheUsualForm() {
        val x = doubleArrayOf(0.3, -1.2, 2.5, 0.0)
        val ls = SoftmaxKernels.logSoftmax(x, intArrayOf(4), 0)
        val m = x.max()
        val lse = m + ln(x.sumOf { exp(it - m) })
        for (k in x.indices) assertTrue(abs(ls[k] - (x[k] - lse)) < 1e-15, "k=$k")
    }

    @Test
    fun softCrossEntropyKeepsTheAxisAndSkipsZeroTeacherProbabilities() {
        // Rows along axis 1 of a [2, 3]; the second teacher row puts exactly 0 on class 1
        // (a logit 1000 below), where the student's log probability is −1000.
        val student = f64(2, 3, 0.5, -1.0, 2.0, 0.0, -1000.0, 1.0)
        val teacher = f64(2, 3, 1.0, 0.0, -1.0, 0.0, -1000.0, 0.0)
        val out = student.softCrossEntropy(teacher, 1)
        assertEquals(listOf(2, 1), out.dims.toList())
        val v = out.hostF64()
        for (r in 0 until 2) {
            val lq = SoftmaxKernels.logSoftmax(DoubleArray(3) { student.hostF64()[3 * r + it] }, intArrayOf(3), 0)
            val lp = SoftmaxKernels.logSoftmax(DoubleArray(3) { teacher.hostF64()[3 * r + it] }, intArrayOf(3), 0)
            val want = -(0 until 3).sumOf { if (exp(lp[it]) == 0.0) 0.0 else exp(lp[it]) * lq[it] }
            assertTrue(abs(v[r] - want) < 1e-14, "row $r: ${v[r]} against $want")
        }
        assertTrue(v.all { it.isFinite() })
        val total = softCrossEntropyLoss(student, teacher).hostF64().single()
        assertTrue(abs(total - (v[0] + v[1])) < 1e-14)

        val s32 = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(0.5f, -1f, 2f))
        val t32 = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(1f, 0f, -1f))
        assertTrue(abs(s32.softCrossEntropy(t32).hostF32().single() - v[0]) < 1e-6)
    }

    @Test
    fun expm1AndLog1pKeepSmallArguments() {
        val x = Tensors.f32Matrix<Sym, Sym>(1, 3, floatArrayOf(1e-10f, -2e-9f, 0.5f))
        val e = x.expm1().hostF32()
        val l = x.log1p().hostF32()
        assertEquals(1e-10f, e[0])
        assertEquals(-2e-9f, l[1])
        assertTrue(abs(e[2] - kotlin.math.expm1(0.5)) < 1e-7 && abs(l[2] - ln1p(0.5)) < 1e-7)
        assertEquals(kotlin.math.expm1(1e-20), f64(1, 1, 1e-20).expm1().hostF64().single())
        assertEquals(1e-20f.expm1(), 1e-20f)
        assertEquals(ln1p(-1e-18), (-1e-18).log1p())
    }
}
