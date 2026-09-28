package io.tlaloc.autograd

import io.tlaloc.core.LinalgKernels
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The capture API records the linear-algebra ops with their attrs, and the
 * captured function's gradient matches central finite differences in Double.
 * Tolerance as in `DxirLinalgGradTest`: 1e-4 of the largest entry against an F32
 * error below 1e-5 on these well-conditioned matrices.
 */
class TracedLinalgTest {

    private val n = 3
    private val a = doubleArrayOf(4.0, 1.2, -0.5, 0.8, 3.5, 0.7, -0.3, 0.9, 5.0)
    private val b = doubleArrayOf(1.0, -2.0, 0.5, 0.3, 1.7, -0.9)

    private fun f(x: DoubleArray) = FloatArray(x.size) { x[it].toFloat() }

    private fun loss(aa: DoubleArray, bb: DoubleArray): Double {
        val l = LinalgKernels.cholesky(aa, n)
        val y = LinalgKernels.triangularSolve(l, bb, n, 2, lower = true, transposeA = false, unitDiagonal = false)
        val x = LinalgKernels.triangularSolve(l, y, n, 2, lower = true, transposeA = true, unitDiagonal = false)
        val t = LinalgKernels.triangle(aa, n, n, 1.0, 1.0, 0.0)
        return x.sumOf { it * it } + t.sumOf { it * it * it }
    }

    @Test
    fun capturedLinalgGradientMatchesFiniteDifferences() {
        val fn = capture2(
            f = { x: Tracer<Rank2<Sym, Sym>>, y: Tracer<Rank2<Sym, Sym>> ->
                val l = x.cholesky()
                val z = l.triangularSolve(l.triangularSolve(y, true), true, true, false)
                val t = x.tril()
                (z * z).sum() + (t * t * t).sum()
            },
            a = Tensors.f32Matrix<Sym, Sym>(n, n, f(a)),
            b = Tensors.f32Matrix<Sym, Sym>(n, 2, f(b)),
            name = "linalg",
        )
        val kinds = fn.body.filterIsInstance<DxirOp>().map { it.op }
        assertTrue(OpKind.CHOLESKY in kinds && OpKind.TRIANGLE in kinds, "captured $kinds")
        val solves = fn.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.TRIANGULAR_SOLVE }
        assertEquals(listOf(false, true), solves.map { it.attrs["transpose_a"] })

        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(a), f(b)))
        val h = 1e-5
        for ((k, x0) in listOf(a, b).withIndex()) {
            val want = DoubleArray(x0.size) { i ->
                val p = x0.copyOf().also { it[i] += h }
                val m = x0.copyOf().also { it[i] -= h }
                if (k == 0) (loss(p, b) - loss(m, b)) / (2 * h) else (loss(a, p) - loss(a, m)) / (2 * h)
            }
            val scale = want.maxOf { abs(it) }
            for (i in want.indices) {
                assertTrue(abs(want[i] - g[k][i]) <= 1e-4 * scale, "d/d${"ab"[k]}[$i] = ${g[k][i]}, want ${want[i]}")
            }
        }
    }

    @Test
    fun capturedLogDetHasGradientTheSymmetrizedInverse() {
        val fn = capture(
            f = { x: Tracer<Rank2<Sym, Sym>> -> x.logDetSpd() },
            input = Tensors.f32Matrix<Sym, Sym>(n, n, f(a)),
            name = "logdet",
        )
        val l = LinalgKernels.cholesky(a, n)
        val want = 2 * (0 until n).sumOf { kotlin.math.ln(l[it * n + it]) }
        val got = DxirInterpreter.evalFunction(fn, listOf(f(a))).single().single()
        assertTrue(abs(got - want) < 1e-5, "log det = $got, want $want")
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(a))).single()
        val eye = DoubleArray(n * n) { if (it / n == it % n) 1.0 else 0.0 }
        val inv = LinalgKernels.triangularSolve(
            l, LinalgKernels.triangularSolve(l, eye, n, n, true, false, false), n, n, true, true, false,
        )
        for (i in inv.indices) assertTrue(abs(inv[i] - g[i]) < 1e-5, "dA[$i] = ${g[i]}, want ${inv[i]}")
    }

    @Test
    fun capturedSolveAndDetGradientsMatchFiniteDifferences() {
        val g0 = doubleArrayOf(0.0, 2.0, 1.0, 1.0, 0.5, 3.0, 4.0, 1.0, 0.2)
        val fn = capture2(
            f = { x: Tracer<Rank2<Sym, Sym>>, y: Tracer<Rank2<Sym, Sym>> ->
                val s = x.solve(y, true)
                (s * s).sum() + x.det()
            },
            a = Tensors.f32Matrix<Sym, Sym>(n, n, f(g0)),
            b = Tensors.f32Matrix<Sym, Sym>(n, 2, f(b)),
            name = "solve_det",
        )
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(g0), f(b)))
        fun lossD(aa: DoubleArray, bb: DoubleArray) =
            LinalgKernels.solve(aa, bb, n, 2, true).sumOf { it * it } + LinalgKernels.det(aa, n)
        val h = 1e-5
        for ((k, x0) in listOf(g0, b).withIndex()) {
            val want = DoubleArray(x0.size) { i ->
                val p = x0.copyOf().also { it[i] += h }
                val m = x0.copyOf().also { it[i] -= h }
                if (k == 0) (lossD(p, b) - lossD(m, b)) / (2 * h) else (lossD(g0, p) - lossD(g0, m)) / (2 * h)
            }
            val scale = want.maxOf { abs(it) }
            for (i in want.indices) {
                assertTrue(abs(want[i] - g[k][i]) <= 1e-4 * scale, "d/d${"ab"[k]}[$i] = ${g[k][i]}, want ${want[i]}")
            }
        }
    }

    @Test
    fun capturedQrGradientMatchesFiniteDifferences() {
        val a0 = doubleArrayOf(1.2, -0.7, 0.4, 2.1, -0.9, 0.5, 0.6, -1.3)
        val fn = capture(
            f = { x: Tracer<Rank2<Sym, Sym>> ->
                val q = x.qrQ()
                val r = x.qrR()
                (q * q * q).sum() + (r * r * r).sum()
            },
            input = Tensors.f32Matrix<Sym, Sym>(4, 2, f(a0)),
            name = "qr",
        )
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(a0))).single()
        fun lossD(x: DoubleArray): Double {
            val (q, r) = LinalgKernels.qr(x, 4, 2)
            return q.sumOf { it * it * it } + r.sumOf { it * it * it }
        }
        val h = 1e-5
        val want = DoubleArray(a0.size) { i ->
            (lossD(a0.copyOf().also { it[i] += h }) - lossD(a0.copyOf().also { it[i] -= h })) / (2 * h)
        }
        val scale = want.maxOf { abs(it) }
        for (i in want.indices) assertTrue(abs(want[i] - g[i]) <= 1e-4 * scale, "dA[$i] = ${g[i]}, want ${want[i]}")
    }

    @Test
    fun capturedEighGradientMatchesFiniteDifferences() {
        val fn = capture(
            f = { x: Tracer<Rank2<Sym, Sym>> ->
                val w = x.eighValues()
                val v = x.eighVectors()
                (w * w * w).sum() + (v * v * v).sum()
            },
            input = Tensors.f32Matrix<Sym, Sym>(n, n, f(a)),
            name = "eigh",
        )
        val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(f(a))).single()
        fun lossD(x: DoubleArray): Double {
            val (w, v) = LinalgKernels.eigh(x, n)
            return w.sumOf { it * it * it } + v.sumOf { it * it * it }
        }
        val h = 1e-5
        val want = DoubleArray(a.size) { i ->
            (lossD(a.copyOf().also { it[i] += h }) - lossD(a.copyOf().also { it[i] -= h })) / (2 * h)
        }
        val scale = want.maxOf { abs(it) }
        for (i in want.indices) assertTrue(abs(want[i] - g[i]) <= 1e-4 * scale, "dA[$i] = ${g[i]}, want ${want[i]}")
    }
}
