package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * SOFT_CROSS_ENTROPY, its gradients, EXPM1 and LOG1P on the device (arXiv 2610.31642,
 * section 3): a student a millionth from its teacher, where `q − p` computed as a
 * difference has no correct digit left in f32, and a row whose teacher probability
 * underflows, against closed forms in Double.
 */
class PjrtSoftCrossEntropyTest {

    private fun t(dtype: DType, vararg dims: Int) = DxirType(dtype, dims.toList())

    private fun loss(rows: Int, n: Int): DxirFunction = DxirBuilder.function("sce") {
        val s = param("student", t(F32, rows, n))
        val te = param("teacher", t(F32, rows, n))
        val r = op(OpKind.SOFT_CROSS_ENTROPY, listOf(s, te), t(F32, rows, 1), attrs = mapOf("axis" to 1))
        listOf(op(OpKind.SUM, listOf(r), t(F32)))
    }

    private fun logSoftmax(x: List<Double>): List<Double> {
        val m = x.max()
        val i = x.indexOf(m)
        val rest = x.indices.filter { it != i }.sumOf { exp(x[it] - m) }
        return x.map { it - m - ln1p(rest) }
    }

    /** `(loss, q − p, −p ⊙ (log q − Σ p log q))` for one row, `q − p` from the logit difference. */
    private fun exact(zs: List<Double>, zt: List<Double>): Triple<Double, List<Double>, List<Double>> {
        val lq = logSoftmax(zs)
        val lp = logSoftmax(zt)
        val p = lp.map { exp(it) }
        val delta = zs.indices.map { zs[it] - zt[it] }
        val mean = p.indices.sumOf { p[it] * delta[it] }
        val d = delta.map { it - mean }
        val a = p.indices.sumOf { if (d[it] <= 1.0) p[it] * expm1(d[it]) else exp(lp[it] + d[it]) - p[it] }
        val gs = d.indices.map {
            val s = d[it] - ln1p(a)
            if (s <= 1.0) p[it] * expm1(s) else exp(lq[it]) - p[it]
        }
        val lqMean = p.indices.sumOf { p[it] * lq[it] }
        val gt = p.indices.map { -p[it] * (lq[it] - lqMean) }
        return Triple(-p.indices.sumOf { p[it] * lq[it] }, gs, gt)
    }

    @Test
    fun gradientsOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val n = 8
        // Row 0: the student a millionth from the teacher. Row 1: a teacher logit 200 below
        // the others where the student puts its mass.
        val zt0 = FloatArray(n) { (-3.0 + 6.0 * ((it * 5) % n) / n).toFloat() }
        val zs0 = FloatArray(n) { zt0[it] + (1e-6 * (if (it % 2 == 0) 1 else -1) * (1 + it)).toFloat() }
        val zt1 = floatArrayOf(0f, -200f, 5f, 1f, 0f, 0f, 2f, -1f)
        val zs1 = floatArrayOf(3f, 90f, -2f, 0f, 1f, 0f, 0f, 0f)
        val zs = zs0 + zs1
        val zt = zt0 + zt1
        val rows = (0 until 2).map { r ->
            exact((0 until n).map { zs[r * n + it].toDouble() }, (0 until n).map { zt[r * n + it].toDouble() })
        }
        TestBackend.session().use { session ->
            val value = (session.runOnHost(loss(2, n), listOf(zs, zt))[0] as FloatArray)[0]
            val want = rows.sumOf { it.first }
            assertTrue(abs(value - want) <= 1e-5 * abs(want), "loss $value against $want")

            val g = session.runOnHost(DxirReverseTransform.apply(loss(2, n)), listOf(zs, zt))
            val gs = g[0] as FloatArray
            val gt = g[1] as FloatArray
            for (r in 0 until 2) {
                val (_, ws, wt) = rows[r]
                val scaleS = ws.maxOf { abs(it) }
                val scaleT = wt.maxOf { abs(it) }
                for (k in 0 until n) {
                    val es = abs(gs[r * n + k] - ws[k]) / scaleS
                    val et = abs(gt[r * n + k] - wt[k]) / scaleT
                    assertTrue(es < 1e-5, "row $r gs_$k device ${gs[r * n + k]}, exact ${ws[k]}")
                    assertTrue(et < 1e-5, "row $r gt_$k device ${gt[r * n + k]}, exact ${wt[k]}")
                }
            }
            println("[pjrt] soft cross entropy, student 1e-6 from the teacher: gs ${gs.take(4)}")

            val x = floatArrayOf(1e-10f, -3e-9f, 0.5f, -0.25f)
            for ((kind, f) in listOf(OpKind.EXPM1 to { v: Double -> expm1(v) }, OpKind.LOG1P to { v: Double -> ln1p(v) })) {
                val fn = DxirBuilder.function(kind.name.lowercase()) {
                    val p = param("x", t(F32, 4))
                    listOf(op(kind, listOf(p), t(F32, 4)))
                }
                val y = session.runOnHost(fn, listOf(x))[0] as FloatArray
                for (i in x.indices) {
                    val w = f(x[i].toDouble())
                    assertTrue(abs(y[i] - w) <= 1e-6 * abs(w), "$kind(${x[i]}) device ${y[i]}, exact $w")
                }
            }
        }
    }
}
