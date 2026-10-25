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
import kotlin.math.expm1
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * SOFT_CROSS_ENTROPY and its gradients (Barron, Gnanapragasam and Tschannen, "A Fix for
 * Softmax and Cross-Entropy", arXiv 2610.31642, section 3). The student's gradient
 * `q − p` cancels when computed as a difference once the student is close to the
 * teacher; the rule builds it from `student − teacher` in log space. The references are
 * closed forms in Double: `q_k − p_k = p_k·expm1(s_k)` with `s = log q − log p` taken from
 * the logit difference, or `q − p` itself where the two are far apart.
 */
class SoftCrossEntropyTest {

    private fun lossFn(n: Int, rows: Int = 1, dtype: DType = F32): DxirFunction {
        val t = DxirType(dtype, listOf(rows, n))
        return DxirBuilder.function("sce") {
            val s = param("student", t)
            val te = param("teacher", t)
            val r = op(OpKind.SOFT_CROSS_ENTROPY, listOf(s, te), DxirType(dtype, listOf(rows, 1)), attrs = mapOf("axis" to 1))
            listOf(op(OpKind.SUM, listOf(r), DxirType(dtype, emptyList())))
        }
    }

    private fun logSoftmax(x: DoubleArray): DoubleArray {
        val m = x.max()
        val i = x.indices.first { x[it] == m }
        val rest = x.indices.filter { it != i }.sumOf { exp(x[it] - m) }
        return DoubleArray(x.size) { x[it] - m - ln1p(rest) }
    }

    /** `q − p` for one row, from the logit difference in Double. */
    private fun exactStudentGrad(zs: DoubleArray, zt: DoubleArray): DoubleArray {
        val logP = logSoftmax(zt)
        val logQ = logSoftmax(zs)
        val p = logP.map { exp(it) }
        val delta = DoubleArray(zs.size) { zs[it] - zt[it] }
        val mean = p.indices.sumOf { p[it] * delta[it] }
        val d = DoubleArray(zs.size) { delta[it] - mean }
        val a = p.indices.sumOf { if (d[it] <= 1.0) p[it] * expm1(d[it]) else exp(logP[it] + d[it]) - p[it] }
        return DoubleArray(zs.size) {
            val s = d[it] - ln1p(a)
            if (s <= 1.0) p[it] * expm1(s) else exp(logQ[it]) - p[it]
        }
    }

    /** `−p ⊙ (log q − Σ p log q)` for one row. */
    private fun exactTeacherGrad(zs: DoubleArray, zt: DoubleArray): DoubleArray {
        val logQ = logSoftmax(zs)
        val p = logSoftmax(zt).map { exp(it) }
        val mean = p.indices.sumOf { p[it] * logQ[it] }
        return DoubleArray(zs.size) { -p[it] * (logQ[it] - mean) }
    }

    /** Largest absolute error over the largest absolute value. */
    private fun scaledErr(got: FloatArray, want: DoubleArray): Double =
        got.indices.maxOf { abs(got[it] - want[it]) } / want.maxOf { abs(it) }

    private fun grads(zs: FloatArray, zt: FloatArray): List<FloatArray> =
        DxirInterpreter.evalFunction(DxirReverseTransform.apply(lossFn(zs.size)), listOf(zs, zt))

    @Test
    fun forwardValueIsTheSoftTargetCrossEntropy() {
        val zs = floatArrayOf(0.5f, -1.0f, 2.0f, 0.0f, 3.0f, -2.0f)
        val zt = floatArrayOf(1.0f, 0.0f, -1.0f, 2.0f, 2.5f, -0.5f)
        val got = DxirInterpreter.evalFunction(lossFn(3, rows = 2), listOf(zs, zt))[0][0]
        var want = 0.0
        for (r in 0 until 2) {
            val lq = logSoftmax(DoubleArray(3) { zs[3 * r + it].toDouble() })
            val lp = logSoftmax(DoubleArray(3) { zt[3 * r + it].toDouble() })
            for (k in 0 until 3) want -= exp(lp[k]) * lq[k]
        }
        assertTrue(abs(got - want) < 1e-6, "loss $got against $want")
    }

    @Test
    fun studentGradientDoesNotCancelNearTheTeacher() {
        // The student a millionth from the teacher: q − p is about 1e-7 while q and p are
        // about 0.1, whose f32 spacing is 7e-9.
        val n = 8
        val zt = FloatArray(n) { (-3.0 + 6.0 * ((it * 5) % n) / n).toFloat() }
        val zs = FloatArray(n) { zt[it] + (1e-6 * (if (it % 2 == 0) 1 else -1) * (1 + it)).toFloat() }
        val want = exactStudentGrad(DoubleArray(n) { zs[it].toDouble() }, DoubleArray(n) { zt[it].toDouble() })
        val gs = grads(zs, zt)[0]
        assertTrue(scaledErr(gs, want) < 1e-5, "gs ${gs.toList()} against ${want.toList()}")
        assertTrue(abs(gs.sum()) <= 1e-12f, "sums to zero: ${gs.sum()}")

        // The same loss written as Σ −p ⊙ logSoftmax(student) differentiates through
        // LOG_SOFTMAX to softmax(student) − p, a difference of two nearly equal terms.
        val t = DxirType(F32, listOf(1, n))
        val composed = DxirBuilder.function("composed") {
            val s = param("student", t)
            val p = param("p", t)
            val ls = op(OpKind.LOG_SOFTMAX, listOf(s), t, attrs = mapOf("axis" to 1))
            listOf(op(OpKind.NEG, listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(p, ls), t)), DxirType(F32, emptyList()))), DxirType(F32, emptyList())))
        }
        val lp = logSoftmax(DoubleArray(n) { zt[it].toDouble() })
        val pF = FloatArray(n) { exp(lp[it]).toFloat() }
        val old = DxirInterpreter.evalFunction(DxirReverseTransform.apply(composed), listOf(zs, pF))[0]
        assertTrue(scaledErr(old, want) > 1e-3, "the composed form cancels: ${scaledErr(old, want)}")
    }

    @Test
    fun studentGradientWhenBothAreConfident() {
        // Both put 1 − 2e-13 on class 0 (past the f32 cliff); the student moves class 2 by
        // 1e-3. Every entry of q − p is about 1e-16.
        val zt = floatArrayOf(30f, 0f, 0f)
        val zs = floatArrayOf(30f, 0f, 0.001f)
        val want = exactStudentGrad(DoubleArray(3) { zs[it].toDouble() }, DoubleArray(3) { zt[it].toDouble() })
        val gs = grads(zs, zt)[0]
        for (k in 0..2) {
            assertTrue(abs(gs[k] - want[k]) / abs(want[k]) < 1e-3, "gs_$k ${gs[k]} against ${want[k]}")
        }
    }

    @Test
    fun studentGradientFarFromTheTeacher() {
        // A teacher probability that underflows (a logit 200 below) where the student puts
        // its mass, and student logits 90 apart: the guarded branches, against q − p.
        val zt = floatArrayOf(0f, -200f, 5f, 1f)
        val zs = floatArrayOf(3f, 90f, -2f, 0f)
        val zsD = DoubleArray(4) { zs[it].toDouble() }
        val ztD = DoubleArray(4) { zt[it].toDouble() }
        val lq = logSoftmax(zsD)
        val lp = logSoftmax(ztD)
        val want = DoubleArray(4) { exp(lq[it]) - exp(lp[it]) }
        val (gs, gt) = grads(zs, zt)
        for (k in 0..3) {
            assertTrue(gs[k].isFinite() && gt[k].isFinite(), "finite: ${gs.toList()} ${gt.toList()}")
            assertTrue(abs(gs[k] - want[k]) <= 1e-6 * maxOf(1.0, abs(want[k])), "gs_$k ${gs[k]} against ${want[k]}")
        }
        val wantT = exactTeacherGrad(zsD, ztD)
        assertTrue(scaledErr(gt, wantT) < 1e-5, "gt ${gt.toList()} against ${wantT.toList()}")
    }

    @Test
    fun teacherGradientAndSeveralRows() {
        val zs = floatArrayOf(0.5f, -1.0f, 2.0f, 0.0f, 3.0f, -2.0f)
        val zt = floatArrayOf(1.0f, 0.0f, -1.0f, 12.0f, 2.5f, -0.5f)
        val (gs, gt) = DxirInterpreter.evalFunction(DxirReverseTransform.apply(lossFn(3, rows = 2)), listOf(zs, zt))
        for (r in 0 until 2) {
            val s = DoubleArray(3) { zs[3 * r + it].toDouble() }
            val t = DoubleArray(3) { zt[3 * r + it].toDouble() }
            val ws = exactStudentGrad(s, t)
            val wt = exactTeacherGrad(s, t)
            for (k in 0 until 3) {
                assertTrue(abs(gs[3 * r + k] - ws[k]) < 1e-6, "row $r gs_$k ${gs[3 * r + k]} against ${ws[k]}")
                assertTrue(abs(gt[3 * r + k] - wt[k]) < 1e-6, "row $r gt_$k ${gt[3 * r + k]} against ${wt[k]}")
            }
        }
    }

    @Test
    fun forwardModeTangentIsTheGradientDotTheDirection() {
        val n = 5
        val zt = floatArrayOf(0.2f, -1.0f, 1.5f, 0.0f, 0.7f)
        val zs = FloatArray(n) { zt[it] + 1e-5f * (it - 2) }
        val vs = floatArrayOf(1f, -2f, 0.5f, 0f, 3f)
        val vt = floatArrayOf(-1f, 0.5f, 2f, 1f, 0f)
        val t = DxirType(F32, listOf(1, n))
        val fn = DxirBuilder.function("sce") {
            val s = param("student", t)
            val te = param("teacher", t)
            listOf(op(OpKind.SOFT_CROSS_ENTROPY, listOf(s, te), DxirType(F32, listOf(1, 1)), attrs = mapOf("axis" to 1)))
        }
        val tangent = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(zs, zt, vs, vt))[1][0]
        val zsD = DoubleArray(n) { zs[it].toDouble() }
        val ztD = DoubleArray(n) { zt[it].toDouble() }
        val ws = exactStudentGrad(zsD, ztD)
        val wt = exactTeacherGrad(zsD, ztD)
        val want = (0 until n).sumOf { ws[it] * vs[it] + wt[it] * vt[it] }
        assertTrue(abs(tangent - want) < 1e-6 * maxOf(1.0, abs(want)), "tangent $tangent against $want")
    }

    @Test
    fun f64StudentGradientNearTheTeacher() {
        // 1e-12 apart in f64: q − p is about 1e-13 against a spacing of 1.4e-17 at 0.1.
        // To first order q − p = ε·p ⊙ (v − Σ p v), off by a relative ε.
        val n = 6
        val zt = DoubleArray(n) { -2.0 + 0.8 * ((it * 7) % n) }
        val v = DoubleArray(n) { (it - 2.5) / 3.0 }
        val eps = 1e-12
        val zs = DoubleArray(n) { zt[it] + eps * v[it] }
        val dv = DoubleArray(n) { zs[it] - zt[it] }
        val p = logSoftmax(zt).map { exp(it) }
        val mean = p.indices.sumOf { p[it] * dv[it] }
        val want = DoubleArray(n) { p[it] * (dv[it] - mean) }
        val gs = DxirInterpreterF64.evalFunction(DxirReverseTransform.apply(lossFn(n, dtype = F64)), listOf(zs, zt))[0]
        val scale = want.maxOf { abs(it) }
        for (k in 0 until n) assertTrue(abs(gs[k] - want[k]) / scale < 1e-9, "gs_$k ${gs[k]} against ${want[k]}")
    }
}
