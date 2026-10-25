package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Softmax collapse (Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and
 * Cross-Entropy", arXiv 2610.31642): with one logit ahead of the others by more than
 * the precision's cliff (16.6 in f32, 36.7 in f64), `p − 1` at that logit rounds to 0
 * while the other entries of the gradient stay positive. LOG_SOFTMAX's value, and
 * the LOG_SOFTMAX and SOFTMAX rules in both modes, keep that entry and sum to zero.
 * The references are the exact values, computed in Double from closed forms; the
 * usual implementations share the defect, so none of them is the reference.
 */
class SoftmaxCollapseTest {

    private val scalar32 = DxirType(F32, emptyList())

    /** Exact log-softmax of a row: `x − M − log1p(Σ_{j≠i} exp(x_j − M))`. */
    private fun exactLogSoftmax(x: DoubleArray): DoubleArray {
        val i = x.indices.maxBy { x[it] }
        val rest = x.indices.filter { it != i }.sumOf { exp(x[it] - x[i]) }
        return DoubleArray(x.size) { x[it] - x[i] - ln1p(rest) }
    }

    private fun relErr(got: Double, want: Double): Double = abs(got - want) / abs(want)

    /** `Σ labels ⊙ LOG_SOFTMAX(x)` over a row of [n] classes (the un-negated cross entropy). */
    private fun ceFn(n: Int, dtype: io.tlaloc.core.DType = F32): DxirFunction {
        val t = DxirType(dtype, listOf(1, n))
        return DxirBuilder.function("ce") {
            val x = param("x", t)
            val labels = param("labels", t)
            val ls = op(OpKind.LOG_SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(labels, ls), t)), DxirType(dtype, emptyList())))
        }
    }

    @Test
    fun logSoftmaxKeepsTheConfidentLogProbability() {
        // The paper's example: [9, −9, −9] in f32. LOG(SOFTMAX) gives 0 at the target.
        val t = DxirType(F32, listOf(1, 3))
        val fn = DxirBuilder.function("ls") {
            val x = param("x", t)
            listOf(op(OpKind.LOG_SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1)))
        }
        val got = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(9f, -9f, -9f)))[0]
        val want = exactLogSoftmax(doubleArrayOf(9.0, -9.0, -9.0))
        assertTrue(relErr(got[0].toDouble(), want[0]) < 1e-6, "log p_max ${got[0]} against ${want[0]}")
        assertTrue(abs(got[0] + 3.046e-8f) < 1e-11f, "the paper's −3.046e-8, got ${got[0]}")
        for (k in 1..2) assertTrue(relErr(got[k].toDouble(), want[k]) < 1e-6, "log p_$k ${got[k]}")

        val old = DxirBuilder.function("old") {
            val x = param("x", t)
            val y = op(OpKind.SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1))
            listOf(op(OpKind.LOG, listOf(y), t))
        }
        assertEquals(0f, DxirInterpreter.evalFunction(old, listOf(floatArrayOf(9f, -9f, -9f)))[0][0], "LOG(SOFTMAX) collapses")
    }

    @Test
    fun logSoftmaxHandlesTiesAndMinusInfinity() {
        val t = DxirType(F32, listOf(1, 3))
        val fn = DxirBuilder.function("ls") {
            val x = param("x", t)
            listOf(op(OpKind.LOG_SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1)))
        }
        val ties = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(2f, 2f, 2f)))[0]
        for (v in ties) assertTrue(abs(v + ln(3.0).toFloat()) < 1e-7f, "ties: $v against −ln 3")
        val inf = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(0f, Float.NEGATIVE_INFINITY, 0f)))[0]
        assertTrue(abs(inf[0] + ln(2.0).toFloat()) < 1e-7f && inf[1] == Float.NEGATIVE_INFINITY, inf.toList().toString())
        // A logit 200 below the maximum: the softmax underflows, its log does not.
        val far = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(0f, -200f, 0f)))[0]
        assertTrue(abs(far[1] - (-200f - ln(2.0).toFloat())) < 1e-4f, "far: ${far[1]}")
    }

    @Test
    fun crossEntropyGradientAtTheTargetDoesNotCollapse() {
        // d/dx Σ labels ⊙ logSoftmax(x) = labels − softmax(x). The paper's vector: the
        // target's entry is 3.046e-8, the others −1.523e-8; the usual form gives 0.
        val grad = DxirReverseTransform.apply(ceFn(3))
        val x = floatArrayOf(9f, -9f, -9f)
        val labels = floatArrayOf(1f, 0f, 0f)
        val dx = DxirInterpreter.evalFunction(grad, listOf(x, labels))[0]
        val e = exp(-18.0)
        val want = doubleArrayOf(2 * e / (1 + 2 * e), -e / (1 + 2 * e), -e / (1 + 2 * e))
        for (k in 0..2) assertTrue(relErr(dx[k].toDouble(), want[k]) < 1e-5, "dx_$k ${dx[k]} against ${want[k]}")
        assertTrue(abs(dx.sum()) < 1e-13f, "sums to zero: ${dx.toList()}")
    }

    @Test
    fun crossEntropyGradientPastTheF32CliffForManyClasses() {
        // 50 classes, the target 25 ahead of logits spread over [−4, 4].
        val n = 50
        val xd = DoubleArray(n) { if (it == 7) 25.0 else -4.0 + 8.0 * it / n }
        val x = FloatArray(n) { xd[it].toFloat() }
        val labels = FloatArray(n) { if (it == 7) 1f else 0f }
        val dx = DxirInterpreter.evalFunction(DxirReverseTransform.apply(ceFn(n)), listOf(x, labels))[0]
        val p = exactLogSoftmax(DoubleArray(n) { x[it].toDouble() }).map { exp(it) }
        for (k in 0 until n) {
            val want = (if (k == 7) 1.0 else 0.0) - p[k]
            val wantK = if (k == 7) p.indices.filter { it != 7 }.sumOf { p[it] } else want
            assertTrue(relErr(dx[k].toDouble(), wantK) < 1e-5, "dx_$k ${dx[k]} against $wantK")
        }
    }

    @Test
    fun crossEntropyGradientPastTheF64Cliff() {
        // Margin 40 in f64 (cliff 36.7): the target's entry is 2·exp(−40)/(1 + 2·exp(−40)).
        val grad = DxirReverseTransform.apply(ceFn(3, F64))
        val dx = DxirInterpreterF64.evalFunction(grad, listOf(doubleArrayOf(40.0, 0.0, 0.0), doubleArrayOf(1.0, 0.0, 0.0)))[0]
        val e = exp(-40.0)
        assertTrue(relErr(dx[0], 2 * e / (1 + 2 * e)) < 1e-12, "dx_0 ${dx[0]}")
        assertTrue(relErr(dx[1], -e / (1 + 2 * e)) < 1e-12, "dx_1 ${dx[1]}")
    }

    @Test
    fun softmaxGradientAtTheDominantEntryDoesNotCollapse() {
        // loss = Σ w ⊙ softmax(x), w one-hot at the dominant entry: dx_0 = p_0(1 − p_0),
        // which the usual y ⊙ (w − Σ w⊙y) rounds to 0 in f32.
        val t = DxirType(F32, listOf(1, 3))
        val fn = DxirBuilder.function("sm") {
            val x = param("x", t)
            val w = param("w", t)
            val y = op(OpKind.SOFTMAX, listOf(x), t, attrs = mapOf("axis" to 1))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(w, y), t)), scalar32))
        }
        val dx = DxirInterpreter.evalFunction(
            DxirReverseTransform.apply(fn), listOf(floatArrayOf(9f, -9f, -9f), floatArrayOf(1f, 0f, 0f)),
        )[0]
        val e = exp(-18.0)
        val p0 = 1 / (1 + 2 * e)
        val p1 = e / (1 + 2 * e)
        assertTrue(relErr(dx[0].toDouble(), p0 * (1 - p0)) < 1e-5, "dx_0 ${dx[0]} against ${p0 * (1 - p0)}")
        assertTrue(relErr(dx[1].toDouble(), -p1 * p0) < 1e-5, "dx_1 ${dx[1]}")
    }

    @Test
    fun forwardModeTangentsKeepTheDominantEntry() {
        val t = DxirType(F32, listOf(1, 3))
        val x = floatArrayOf(9f, -9f, -9f)
        val v = floatArrayOf(1f, 0f, 0f)
        val e = exp(-18.0)
        val p = doubleArrayOf(1 / (1 + 2 * e), e / (1 + 2 * e), e / (1 + 2 * e))

        val ls = DxirBuilder.function("ls") {
            val xp = param("x", t)
            listOf(op(OpKind.LOG_SOFTMAX, listOf(xp), t, attrs = mapOf("axis" to 1)))
        }
        // d logSoftmax · v = v − Σ p⊙v: 1 − p_0 at the dominant entry, which rounds to 0
        // as written, and −p_0 elsewhere.
        val dls = DxirInterpreter.evalFunction(DxirForwardTransform.apply(ls), listOf(x, v))[1]
        assertTrue(relErr(dls[0].toDouble(), p[1] + p[2]) < 1e-5, "d ls_0 ${dls[0]} against ${p[1] + p[2]}")
        assertTrue(relErr(dls[1].toDouble(), -p[0]) < 1e-6, "d ls_1 ${dls[1]}")

        val sm = DxirBuilder.function("sm") {
            val xp = param("x", t)
            listOf(op(OpKind.SOFTMAX, listOf(xp), t, attrs = mapOf("axis" to 1)))
        }
        // d softmax · v = p ⊙ (v − Σ p⊙v): p_0·(1 − p_0) at the dominant entry, −p_j·p_0 elsewhere.
        val dsm = DxirInterpreter.evalFunction(DxirForwardTransform.apply(sm), listOf(x, v))[1]
        assertTrue(relErr(dsm[0].toDouble(), p[0] * (p[1] + p[2])) < 1e-5, "d sm_0 ${dsm[0]}")
        assertTrue(relErr(dsm[1].toDouble(), -p[1] * p[0]) < 1e-5, "d sm_1 ${dsm[1]}")
    }

    @Test
    fun ordinaryRowsAreUnchanged() {
        // No probability above 1/2: the rules are the usual formulas.
        val n = 4
        val x = floatArrayOf(0.3f, -0.2f, 0.1f, 0.4f)
        val labels = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val dx = DxirInterpreter.evalFunction(DxirReverseTransform.apply(ceFn(n)), listOf(x, labels))[0]
        val p = exactLogSoftmax(DoubleArray(n) { x[it].toDouble() }).map { exp(it) }
        for (k in 0 until n) assertTrue(abs(dx[k] - (labels[k] - p[k])) < 1e-6, "dx_$k ${dx[k]}")
    }
}
