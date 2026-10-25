package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln1p
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * LOG_SOFTMAX and the cross-entropy gradient on the device, against exact values for
 * rows past each precision's cliff (arXiv 2610.31642): the maximum's log probability
 * and the target's gradient entry stay nonzero, and a row with a tie and a logit far
 * below the maximum is finite where the softmax underflows.
 */
class PjrtLogSoftmaxTest {

    private fun t(dtype: DType, vararg dims: Int) = DxirType(dtype, dims.toList())

    /** `[value = logSoftmax(x), d/dx Σ labels ⊙ logSoftmax(x)]` over rows of [n] classes. */
    private fun ce(dtype: DType, rows: Int, n: Int): DxirFunction {
        val fn = DxirBuilder.function("ce_${dtype.name}") {
            val x = param("x", t(dtype, rows, n))
            val labels = param("labels", t(dtype, rows, n))
            val ls = op(OpKind.LOG_SOFTMAX, listOf(x), t(dtype, rows, n), attrs = mapOf("axis" to 1))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(labels, ls), t(dtype, rows, n))), t(dtype)))
        }
        return DxirReverseTransform.apply(fn)
    }

    private fun values(dtype: DType, rows: Int, n: Int): DxirFunction = DxirBuilder.function("ls_${dtype.name}") {
        val x = param("x", t(dtype, rows, n))
        listOf(op(OpKind.LOG_SOFTMAX, listOf(x), t(dtype, rows, n), attrs = mapOf("axis" to 1)))
    }

    /** Exact `labels − softmax(x)` and `logSoftmax(x)` per row, in Double. */
    private fun exact(x: DoubleArray, labels: DoubleArray, n: Int): Pair<DoubleArray, DoubleArray> {
        val grad = DoubleArray(x.size)
        val ls = DoubleArray(x.size)
        for (r in 0 until x.size / n) {
            val row = (0 until n).map { x[r * n + it] }
            val i = row.indices.maxBy { row[it] }
            val rest = row.indices.filter { it != i }.sumOf { exp(row[it] - row[i]) }
            for (k in 0 until n) ls[r * n + k] = row[k] - row[i] - ln1p(rest)
            val p = (0 until n).map { exp(ls[r * n + it]) }
            val mass = (0 until n).sumOf { labels[r * n + it] }
            for (k in 0 until n) grad[r * n + k] = labels[r * n + k] - p[k] * mass
            // The entry at the target of a one-hot row, written from the others.
            if (mass == 1.0 && labels[r * n + i] == 1.0) {
                grad[r * n + i] = (0 until n).filter { it != i }.sumOf { p[it] }
            }
        }
        return grad to ls
    }

    private fun assertRel(name: String, got: Double, want: Double, tol: Double) {
        // Below f32's smallest normal an entry is compared absolutely (the device flushes).
        val err = if (abs(want) < 1e-30) abs(got) / 1e-30 else abs(got - want) / abs(want)
        assertTrue(err <= tol, "$name: device $got, exact $want (relative error $err)")
    }

    @Test
    fun confidentRowsOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // Row 0: the paper's [9, −9, −9]; row 1: a margin of 25 over 3 classes; row 2: a
        // tie at the maximum and a logit 200 below it.
        val n = 3
        val x = doubleArrayOf(9.0, -9.0, -9.0, 25.0, 0.0, -1.0, 2.0, 2.0, -198.0)
        val labels = doubleArrayOf(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        val (wantGrad, wantLs) = exact(x, labels, n)
        TestBackend.session().use { session ->
            val g32 = session.runOnHost(ce(F32, 3, n), listOf(FloatArray(x.size) { x[it].toFloat() }, FloatArray(x.size) { labels[it].toFloat() }))
            val v32 = session.runOnHost(values(F32, 3, n), listOf(FloatArray(x.size) { x[it].toFloat() }))
            val dx = g32[0] as FloatArray
            val ls = v32[0] as FloatArray
            for (k in x.indices) {
                assertRel("f32 d_x[$k]", dx[k].toDouble(), wantGrad[k], 1e-4)
                if (wantLs[k].isFinite()) assertRel("f32 logSoftmax[$k]", ls[k].toDouble(), wantLs[k], 1e-4)
            }
            println("[pjrt] f32 [9, −9, −9]: logSoftmax ${ls.take(3)}, gradient ${dx.take(3)}")

            // f64 past its cliff of 36.7: a margin of 40.
            val x64 = doubleArrayOf(40.0, 0.0, 0.0)
            val l64 = doubleArrayOf(1.0, 0.0, 0.0)
            val (wg, wl) = exact(x64, l64, n)
            val g64 = session.runOnHost(ce(F64, 1, n), listOf(x64, l64))[0] as DoubleArray
            val v64 = session.runOnHost(values(F64, 1, n), listOf(x64))[0] as DoubleArray
            for (k in 0 until n) {
                assertRel("f64 d_x[$k]", g64[k], wg[k], TestBackend.f64Tolerance(1e-12))
                assertRel("f64 logSoftmax[$k]", v64[k], wl[k], TestBackend.f64Tolerance(1e-12))
            }
        }
    }
}
