package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.LinalgKernels
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * CHOLESKY, TRIANGULAR_SOLVE and TRIANGLE through XLA: `stablehlo.cholesky` and
 * `stablehlo.triangular_solve` compile and run on the device, forwards and
 * gradient graphs.
 *
 * - F32: the device agrees with the interpreter.
 * - F64: the gradient and tangent graphs, emitted as f64, agree with central
 *   finite differences of the Double kernels. This is where the derivative rules
 *   are checked in F64: the interpreter and the `grad {}` host path are F32 only.
 */
class PjrtLinalgTest {

    private fun t(dtype: DType, vararg dims: Int) = DxirType(dtype, dims.toList())

    private val n = 4
    private val spd = doubleArrayOf(
        6.2, 1.1, -0.7, 0.4,
        1.3, 5.1, 0.9, -1.2,
        -0.5, 0.9, 4.8, 0.6,
        0.4, -1.0, 0.6, 5.5,
    )
    private val tri = doubleArrayOf(
        2.0, 9.0, -7.0, 5.0,
        0.6, 1.7, 8.0, -3.0,
        -0.4, 0.3, 2.4, 4.0,
        0.9, -0.8, 0.5, 1.9,
    )
    private val rhs = doubleArrayOf(0.7, -1.2, 0.4, 2.1, -0.3, 0.8, 1.5, -0.6)
    private val w44 = DoubleArray(16) { ((it * 7) % 13 - 6) / 6.0 }
    private val w42 = DoubleArray(8) { ((it * 5) % 9 - 4) / 4.0 }

    private fun choleskyLoss(dtype: DType, k: Int) = DxirBuilder.function("chol_loss") {
        val a = param("a", t(dtype, k, k))
        val w = param("w", t(dtype, k, k))
        val l = op(OpKind.CHOLESKY, listOf(a), t(dtype, k, k))
        listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(l, w), t(dtype, k, k))), t(dtype)))
    }

    private fun solveLoss(dtype: DType, lower: Boolean, transposeA: Boolean, unit: Boolean) =
        DxirBuilder.function("trsm_loss") {
            val a = param("a", t(dtype, n, n))
            val b = param("b", t(dtype, n, 2))
            val w = param("w", t(dtype, n, 2))
            val x = op(
                OpKind.TRIANGULAR_SOLVE, listOf(a, b), t(dtype, n, 2),
                attrs = mapOf("lower" to lower, "transpose_a" to transposeA, "unit_diagonal" to unit),
            )
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, w), t(dtype, n, 2))), t(dtype)))
        }

    private fun triFor(lower: Boolean) = if (lower) tri else DoubleArray(n * n) { tri[(it % n) * n + it / n] }

    private fun fdGrad(x: DoubleArray, h: Double, loss: (DoubleArray) -> Double) = DoubleArray(x.size) { i ->
        (loss(x.copyOf().also { it[i] += h }) - loss(x.copyOf().also { it[i] -= h })) / (2 * h)
    }

    private fun dot(a: DoubleArray, b: DoubleArray) = a.indices.sumOf { a[it] * b[it] }

    private fun maxRelDiff(want: DoubleArray, got: DoubleArray): Double {
        val scale = max(1e-300, want.maxOf { abs(it) })
        return want.indices.maxOf { abs(want[it] - got[it]) } / scale
    }

    private fun f32(a: DoubleArray) = FloatArray(a.size) { a[it].toFloat() }

    @Test
    fun f32ForwardAndGradientGraphsMatchTheInterpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val cases = buildList<Pair<DxirFunction, List<FloatArray>>> {
            add(choleskyLoss(F32, n) to listOf(f32(spd), f32(w44)))
            for (lower in listOf(true, false)) for (tr in listOf(false, true)) for (u in listOf(false, true)) {
                add(solveLoss(F32, lower, tr, u) to listOf(f32(triFor(lower)), f32(rhs), f32(w42)))
            }
        }
        TestBackend.session().use { session ->
            for ((fn, inputs) in cases) {
                for (g in listOf(fn, DxirReverseTransform.apply(fn))) {
                    val want = DxirInterpreter.evalFunction(g, inputs)
                    val got = session.runOn(g, inputs)
                    for (r in want.indices) {
                        val w = DoubleArray(want[r].size) { want[r][it].toDouble() }
                        val d = maxRelDiff(w, DoubleArray(got[r].size) { got[r][it].toDouble() })
                        // TF32 dots on the GB10 (the gradient bodies multiply matrices).
                        assertTrue(
                            d <= TestBackend.defaultDotRelTolerance,
                            "${g.name} result $r: device differs from the interpreter by $d of the largest entry",
                        )
                    }
                }
            }
        }
    }

    @Test
    fun f64GradientAndTangentGraphsMatchFiniteDifferences() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // Central differences in Double with h = 1e-6 on O(1) entries: truncation
        // O(h²) ≈ 1e-12, rounding O(ε/h) ≈ 1e-10, so the reference is good to about
        // 1e-9 of the largest entry, and the f64 device result to about 1e-13.
        // Tolerance 1e-7.
        val h = 1e-6
        TestBackend.session().use { session ->
            val chol = choleskyLoss(F64, n)
            val g = session.runOnF64(DxirReverseTransform.apply(chol), listOf(spd, w44))
            val want = fdGrad(spd, h) { dot(LinalgKernels.cholesky(it, n), w44) }
            assertTrue(maxRelDiff(want, g[0]) <= 1e-7, "cholesky dA: ${maxRelDiff(want, g[0])}")

            val v = DoubleArray(16) { ((it * 3) % 7 - 3) / 3.0 }
            val jvp = session.runOnF64(DxirForwardTransform.apply(chol), listOf(spd, w44, v, DoubleArray(16)))
            val wantT = dot(want, v)
            assertTrue(abs(jvp[1][0] - wantT) <= 1e-7 * max(1.0, abs(wantT)), "cholesky tangent ${jvp[1][0]} vs $wantT")

            for (lower in listOf(true, false)) for (tr in listOf(false, true)) for (u in listOf(false, true)) {
                val a = triFor(lower)
                val gs = session.runOnF64(DxirReverseTransform.apply(solveLoss(F64, lower, tr, u)), listOf(a, rhs, w42))
                fun loss(aa: DoubleArray, bb: DoubleArray) =
                    dot(LinalgKernels.triangularSolve(aa, bb, n, 2, lower, tr, u), w42)
                val tag = "lower=$lower transpose=$tr unit=$u"
                val dA = maxRelDiff(fdGrad(a, h) { loss(it, rhs) }, gs[0])
                val dB = maxRelDiff(fdGrad(rhs, h) { loss(a, it) }, gs[1])
                assertTrue(dA <= 1e-7 && dB <= 1e-7, "triangular solve $tag: dA $dA, dB $dB")
            }
        }
    }

    @Test
    fun f64CholeskyGradientOfAnIllConditionedMatrix() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // Hilbert(5), condition number 4.8e5. The gradient has entries up to ~1e5,
        // and a perturbation of h moves them by up to cond·h relative, so the step
        // is 1e-7 and the reference good to about cond²·ε/h ≈ 1e-3 of the largest
        // entry at worst; the measured agreement is far better. Tolerance 1e-4.
        val k = 5
        val hil = DoubleArray(k * k) { 1.0 / ((it / k) + (it % k) + 1) }
        val w = DoubleArray(k * k) { ((it * 7) % 11 - 5) / 5.0 }
        TestBackend.session().use { session ->
            val l = session.runOnF64(DxirBuilder.function("chol") {
                listOf(op(OpKind.CHOLESKY, listOf(param("a", t(F64, k, k))), t(F64, k, k)))
            }, listOf(hil)).single()
            val lRef = LinalgKernels.cholesky(hil, k)
            assertTrue(maxRelDiff(lRef, l) <= 1e-12, "Hilbert(5) factor: ${maxRelDiff(lRef, l)}")
            val g = session.runOnF64(DxirReverseTransform.apply(choleskyLoss(F64, k)), listOf(hil, w))
            val want = fdGrad(hil, 1e-7) { dot(LinalgKernels.cholesky(it, k), w) }
            val d = maxRelDiff(want, g[0])
            println("[pjrt-linalg] Hilbert(5) cholesky gradient: max |device − fd| = $d of the largest entry")
            assertTrue(d <= 1e-4, "Hilbert(5) cholesky dA: $d")
        }
    }

    @Test
    fun f64SolveSpdGradientMatchesFiniteDifferences() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // The plugin's lowering of solveSpd: cholesky, then L⁻¹·B, then L⁻ᵀ·Y.
        // Tolerance as in f64GradientAndTangentGraphsMatchFiniteDifferences.
        val fn = DxirBuilder.function("solve_spd_loss") {
            val a = param("a", t(F64, n, n))
            val b = param("b", t(F64, n, 2))
            val w = param("w", t(F64, n, 2))
            val l = op(OpKind.CHOLESKY, listOf(a), t(F64, n, n))
            fun solve(rhs: io.tlaloc.ir.DxirNode, tr: Boolean) = op(
                OpKind.TRIANGULAR_SOLVE, listOf(l, rhs), t(F64, n, 2),
                attrs = mapOf("lower" to true, "transpose_a" to tr, "unit_diagonal" to false),
            )
            val x = solve(solve(b, false), true)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, w), t(F64, n, 2))), t(F64)))
        }
        fun lossD(a: DoubleArray, b: DoubleArray): Double {
            val l = LinalgKernels.cholesky(a, n)
            val y = LinalgKernels.triangularSolve(l, b, n, 2, true, false, false)
            return dot(LinalgKernels.triangularSolve(l, y, n, 2, true, true, false), w42)
        }
        TestBackend.session().use { session ->
            val g = session.runOnF64(DxirReverseTransform.apply(fn), listOf(spd, rhs, w42))
            val dA = maxRelDiff(fdGrad(spd, 1e-6) { lossD(it, rhs) }, g[0])
            val dB = maxRelDiff(fdGrad(rhs, 1e-6) { lossD(spd, it) }, g[1])
            assertTrue(dA <= 1e-7 && dB <= 1e-7, "solveSpd: dA $dA, dB $dB")
        }
    }
}
