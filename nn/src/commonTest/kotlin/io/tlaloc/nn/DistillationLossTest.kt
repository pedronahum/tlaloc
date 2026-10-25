package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertTrue

class DistillationLossTest {

    private fun tensor(v: FloatArray, vararg dims: Int) = DTensor<Shape, F32>(HostF32Storage(v), dims, F32)

    private fun logSoftmax(x: List<Double>): List<Double> {
        val m = x.max()
        val i = x.indexOf(m)
        val rest = x.indices.filter { it != i }.sumOf { exp(x[it] - m) }
        return x.map { it - m - ln1p(rest) }
    }

    /** `q − p` of one row at temperature [t], from the logit difference in Double. */
    private fun qMinusP(zs: List<Double>, zt: List<Double>, t: Double): List<Double> {
        val s = zs.map { it / t }
        val te = zt.map { it / t }
        val lp = logSoftmax(te)
        val lq = logSoftmax(s)
        val p = lp.map { exp(it) }
        val delta = s.indices.map { s[it] - te[it] }
        val mean = p.indices.sumOf { p[it] * delta[it] }
        val d = delta.map { it - mean }
        val a = p.indices.sumOf { p[it] * expm1(d[it]) }
        return d.indices.map {
            val r = d[it] - ln1p(a)
            if (r <= 1.0) p[it] * expm1(r) else exp(lq[it]) - p[it]
        }
    }

    @Test
    fun valueAndStudentGradientAtATemperature() {
        // Two rows of four classes at T = 2: the loss is T²/rows · Σ cross entropies, the
        // student's gradient T/rows · (q − p).
        val zs = floatArrayOf(1f, -0.5f, 2f, 0f, 0.3f, 0.1f, -1f, 4f)
        val zt = floatArrayOf(0.5f, 0f, 3f, -1f, 0f, 0f, 0f, 2f)
        val temperature = 2f
        val fn = captureN(listOf(tensor(zs, 2, 4), tensor(zt, 2, 4))) { l -> distillationLoss(l[0], l[1], temperature) }
        val loss = DxirInterpreter.evalFunction(fn, listOf(zs, zt)).single()[0]
        val grad = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(zs, zt))[0]
        var want = 0.0
        for (r in 0 until 2) {
            val s = (0 until 4).map { zs[4 * r + it].toDouble() }
            val t = (0 until 4).map { zt[4 * r + it].toDouble() }
            val lq = logSoftmax(s.map { it / temperature })
            val p = logSoftmax(t.map { it / temperature }).map { exp(it) }
            want -= p.indices.sumOf { p[it] * lq[it] }
            val g = qMinusP(s, t, temperature.toDouble())
            for (k in 0 until 4) {
                val w = temperature / 2 * g[k]
                assertTrue(abs(grad[4 * r + k] - w) < 1e-6, "row $r d_$k ${grad[4 * r + k]} against $w")
            }
        }
        want *= temperature * temperature / 2
        assertTrue(abs(loss - want) < 1e-5, "loss $loss against $want")
    }

    @Test
    fun studentGradientNearTheTeacher() {
        // A student a millionth from its teacher still gets the right gradient: 1e-7 per
        // entry against probabilities of about 0.1.
        val n = 8
        val zt = FloatArray(n) { (-3.0 + 6.0 * ((it * 3) % n) / n).toFloat() }
        val zs = FloatArray(n) { zt[it] + (1e-6 * (it - 3.5)).toFloat() }
        val fn = captureN(listOf(tensor(zs, 1, n), tensor(zt, 1, n))) { l -> distillationLoss(l[0], l[1]) }
        val grad = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(zs, zt))[0]
        val want = qMinusP(zs.map { it.toDouble() }, zt.map { it.toDouble() }, 1.0)
        val scale = want.maxOf { abs(it) }
        for (k in 0 until n) {
            assertTrue(abs(grad[k] - want[k]) / scale < 1e-5, "d_$k ${grad[k]} against ${want[k]}")
        }
    }
}
