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

    @Test
    fun f64LogDetGradientAndHessianVectorProductMatchFiniteDifferences() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // The plugin's lowering of logDetSpd, in f64.
        val fn = DxirBuilder.function("log_det") {
            val a = param("a", t(F64, n, n))
            val l = op(OpKind.CHOLESKY, listOf(a), t(F64, n, n))
            val diag = op(
                OpKind.SUM,
                listOf(op(OpKind.TRIANGLE, listOf(l), t(F64, n, n), attrs = mapOf("lower" to 0.0, "diagonal" to 1.0, "upper" to 0.0))),
                t(F64, n),
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            val half = op(OpKind.SUM, listOf(op(OpKind.LOG, listOf(diag), t(F64, n))), t(F64))
            listOf(op(OpKind.ADD, listOf(half, half), t(F64)))
        }
        fun logDet(a: DoubleArray): Double {
            val l = LinalgKernels.cholesky(a, n)
            return 2 * (0 until n).sumOf { kotlin.math.ln(l[it * n + it]) }
        }
        val grad = DxirReverseTransform.apply(fn)
        val v = DoubleArray(n * n) { ((it * 3) % 7 - 3) / 3.0 }
        TestBackend.session().use { session ->
            val value = session.runOnF64(fn, listOf(spd)).single().single()
            assertTrue(abs(value - logDet(spd)) <= 1e-13 * abs(logDet(spd)), "log det $value")
            val g = session.runOnF64(grad, listOf(spd)).single()
            val gWant = fdGrad(spd, 1e-6, ::logDet)
            assertTrue(maxRelDiff(gWant, g) <= 1e-7, "d log det: ${maxRelDiff(gWant, g)}")
            // Hessian-vector product: forward over reverse, against central
            // differences of the device gradient along v (h = 1e-5 on an f64
            // gradient exact to ~1e-15: error ~1e-10).
            val hv = session.runOnF64(DxirForwardTransform.apply(grad), listOf(spd, v))[1]
            val h = 1e-5
            val gp = session.runOnF64(grad, listOf(DoubleArray(n * n) { spd[it] + h * v[it] })).single()
            val gm = session.runOnF64(grad, listOf(DoubleArray(n * n) { spd[it] - h * v[it] })).single()
            val hvWant = DoubleArray(n * n) { (gp[it] - gm[it]) / (2 * h) }
            assertTrue(maxRelDiff(hvWant, hv) <= 1e-7, "Hessian·v: ${maxRelDiff(hvWant, hv)}")
        }
    }

    @Test
    fun invSpdLoweringRunsOnTheDeviceInF32AndF64() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        // solveSpd against TRIANGLE(BROADCAST(1, A), 0, 1, 0), the plugin's lowering.
        fun loss(dtype: DType) = DxirBuilder.function("inv_loss") {
            val a = param("a", t(dtype, n, n))
            val w = param("w", t(dtype, n, n))
            val one = const(if (dtype == F64) 1.0 else 1.0f, t(dtype))
            val ones = op(OpKind.BROADCAST, listOf(one, a), t(dtype, n, n), attrs = mapOf("broadcast_dimensions" to emptyList<Int>()))
            val eye = op(OpKind.TRIANGLE, listOf(ones), t(dtype, n, n), attrs = mapOf("lower" to 0.0, "diagonal" to 1.0, "upper" to 0.0))
            val l = op(OpKind.CHOLESKY, listOf(a), t(dtype, n, n))
            fun solve(rhs: io.tlaloc.ir.DxirNode, tr: Boolean) = op(
                OpKind.TRIANGULAR_SOLVE, listOf(l, rhs), t(dtype, n, n),
                attrs = mapOf("lower" to true, "transpose_a" to tr, "unit_diagonal" to false),
            )
            val inv = solve(solve(eye, false), true)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(inv, w), t(dtype, n, n))), t(dtype)))
        }
        fun lossD(a: DoubleArray): Double {
            val l = LinalgKernels.cholesky(a, n)
            val eye = DoubleArray(n * n) { if (it / n == it % n) 1.0 else 0.0 }
            val y = LinalgKernels.triangularSolve(l, eye, n, n, true, false, false)
            return dot(LinalgKernels.triangularSolve(l, y, n, n, true, true, false), w44)
        }
        TestBackend.session().use { session ->
            val f32 = loss(F32)
            for (g in listOf(f32, DxirReverseTransform.apply(f32))) {
                val inputs = listOf(f32(spd), f32(w44))
                val want = DxirInterpreter.evalFunction(g, inputs)
                val got = session.runOn(g, inputs)
                for (r in want.indices) {
                    val d = maxRelDiff(
                        DoubleArray(want[r].size) { want[r][it].toDouble() },
                        DoubleArray(got[r].size) { got[r][it].toDouble() },
                    )
                    assertTrue(d <= TestBackend.defaultDotRelTolerance, "${g.name} result $r: $d")
                }
            }
            val g64 = session.runOnF64(DxirReverseTransform.apply(loss(F64)), listOf(spd, w44))
            val d = maxRelDiff(fdGrad(spd, 1e-6, ::lossD), g64[0])
            assertTrue(d <= 1e-7, "invSpd f64 dA: $d")
        }
    }

    // Nonsymmetric; its first column's largest entry is in row 2, so LU pivots.
    private val gen = doubleArrayOf(
        0.5, 2.0, -1.0, 0.3,
        1.2, -0.4, 0.8, 2.2,
        -3.0, 0.7, 1.5, -0.2,
        0.9, 1.1, -0.6, 1.4,
    )

    private fun generalSolveLoss(dtype: DType, transposeA: Boolean) = DxirBuilder.function("solve_loss") {
        val a = param("a", t(dtype, n, n))
        val b = param("b", t(dtype, n, 2))
        val w = param("w", t(dtype, n, 2))
        val x = op(OpKind.SOLVE, listOf(a, b), t(dtype, n, 2), attrs = mapOf("transpose_a" to transposeA))
        listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(x, w), t(dtype, n, 2))), t(dtype)))
    }

    private fun detFn(dtype: DType, k: Int) = DxirBuilder.function("det") {
        listOf(op(OpKind.DET, listOf(param("a", t(dtype, k, k))), t(dtype)))
    }

    @Test
    fun luLoweringMatchesTheKernelOnPivotingSingularAndOneByOneInputs() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session().use { session ->
            // The while loop against LinalgKernels, in f64.
            for (tr in listOf(false, true)) {
                val fn = DxirBuilder.function("solve") {
                    listOf(
                        op(
                            OpKind.SOLVE, listOf(param("a", t(F64, n, n)), param("b", t(F64, n, 2))), t(F64, n, 2),
                            attrs = mapOf("transpose_a" to tr),
                        ),
                    )
                }
                val got = session.runOnF64(fn, listOf(gen, rhs)).single()
                val want = LinalgKernels.solve(gen, rhs, n, 2, tr)
                assertTrue(maxRelDiff(want, got) <= 1e-13, "solve transpose=$tr: ${maxRelDiff(want, got)}")
            }
            val det = session.runOnF64(detFn(F64, n), listOf(gen)).single().single()
            assertTrue(abs(det - LinalgKernels.det(gen, n)) <= 1e-13 * abs(det), "det $det")
            // A permutation matrix (det −1), a singular matrix (det exactly 0), 1×1.
            assertTrue(session.runOnF64(detFn(F64, 2), listOf(doubleArrayOf(0.0, 1.0, 1.0, 0.0))).single().single() == -1.0)
            val singular = doubleArrayOf(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0)
            assertTrue(session.runOnF64(detFn(F64, 3), listOf(singular)).single().single() == 0.0, "singular det")
            assertTrue(session.runOnF64(detFn(F64, 1), listOf(doubleArrayOf(-2.5))).single().single() == -2.5)
        }
    }

    @Test
    fun solveAndDetGraphsMatchTheInterpreterInF32AndFiniteDifferencesInF64() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        TestBackend.session().use { session ->
            val f32Cases = listOf(
                generalSolveLoss(F32, false) to listOf(f32(gen), f32(rhs), f32(w42)),
                generalSolveLoss(F32, true) to listOf(f32(gen), f32(rhs), f32(w42)),
                detFn(F32, n) to listOf(f32(gen)),
            )
            for ((fn, inputs) in f32Cases) {
                for (g in listOf(fn, DxirReverseTransform.apply(fn))) {
                    val want = DxirInterpreter.evalFunction(g, inputs)
                    val got = session.runOn(g, inputs)
                    for (r in want.indices) {
                        val d = maxRelDiff(
                            DoubleArray(want[r].size) { want[r][it].toDouble() },
                            DoubleArray(got[r].size) { got[r][it].toDouble() },
                        )
                        assertTrue(d <= TestBackend.defaultDotRelTolerance, "${g.name} result $r: $d")
                    }
                }
            }
            val h = 1e-6
            for (tr in listOf(false, true)) {
                val g = session.runOnF64(DxirReverseTransform.apply(generalSolveLoss(F64, tr)), listOf(gen, rhs, w42))
                fun loss(aa: DoubleArray, bb: DoubleArray) = dot(LinalgKernels.solve(aa, bb, n, 2, tr), w42)
                val dA = maxRelDiff(fdGrad(gen, h) { loss(it, rhs) }, g[0])
                val dB = maxRelDiff(fdGrad(rhs, h) { loss(gen, it) }, g[1])
                assertTrue(dA <= 1e-7 && dB <= 1e-7, "solve transpose=$tr: dA $dA, dB $dB")
            }
            val detGrad = DxirReverseTransform.apply(detFn(F64, n))
            val gd = session.runOnF64(detGrad, listOf(gen)).single()
            val dd = maxRelDiff(fdGrad(gen, h) { LinalgKernels.det(it, n) }, gd)
            assertTrue(dd <= 1e-7, "d det: $dd")
            // Hessian-vector product of det, forward over reverse.
            val v = DoubleArray(n * n) { ((it * 3) % 7 - 3) / 3.0 }
            val hv = session.runOnF64(DxirForwardTransform.apply(detGrad), listOf(gen, v))[1]
            val gp = session.runOnF64(detGrad, listOf(DoubleArray(n * n) { gen[it] + 1e-5 * v[it] })).single()
            val gm = session.runOnF64(detGrad, listOf(DoubleArray(n * n) { gen[it] - 1e-5 * v[it] })).single()
            val dh = maxRelDiff(DoubleArray(n * n) { (gp[it] - gm[it]) / 2e-5 }, hv)
            assertTrue(dh <= 1e-7, "Hessian·v of det: $dh")
        }
    }

    private val tall = doubleArrayOf(
        1.2, -0.7, 0.3,
        0.4, 2.1, -1.1,
        -0.9, 0.5, 1.7,
        0.6, -1.3, 0.2,
        1.5, 0.8, -0.4,
    )

    private fun qrLoss(dtype: DType) = DxirBuilder.function("qr_loss") {
        val a = param("a", t(dtype, 5, 3))
        val wq = param("wq", t(dtype, 5, 3))
        val wr = param("wr", t(dtype, 3, 3))
        val q = op(OpKind.QR_Q, listOf(a), t(dtype, 5, 3))
        val r = op(OpKind.QR_R, listOf(a), t(dtype, 3, 3))
        val lq = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(q, wq), t(dtype, 5, 3))), t(dtype))
        val lr = op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(r, wr), t(dtype, 3, 3))), t(dtype))
        listOf(op(OpKind.ADD, listOf(lq, lr), t(dtype)))
    }

    @Test
    fun qrLoweringMatchesTheKernelAndItsGradientsMatchFiniteDifferences() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val wq = DoubleArray(15) { ((it * 7) % 11 - 5) / 5.0 }
        val wr = DoubleArray(9) { ((it * 5) % 7 - 3) / 3.0 }
        TestBackend.session().use { session ->
            // Forward in f64 against LinalgKernels.qr: a tall matrix, a square one, 1×1,
            // and one whose first column is already zero below the diagonal (τ = 0).
            val cases = listOf(
                Triple(5, 3, tall),
                Triple(4, 4, gen),
                Triple(1, 1, doubleArrayOf(-2.0)),
                Triple(3, 2, doubleArrayOf(3.0, 1.0, 0.0, 2.0, 0.0, -1.0)),
            )
            for ((m, k, a) in cases) {
                val (qWant, rWant) = LinalgKernels.qr(a, m, k)
                for ((kind, want) in listOf(OpKind.QR_Q to qWant, OpKind.QR_R to rWant)) {
                    val outT = if (kind == OpKind.QR_Q) t(F64, m, k) else t(F64, k, k)
                    val fn = DxirBuilder.function("qr") { listOf(op(kind, listOf(param("a", t(F64, m, k))), outT)) }
                    val got = session.runOnF64(fn, listOf(a)).single()
                    assertTrue(maxRelDiff(want, got) <= 1e-13, "$kind of $m×$k: ${maxRelDiff(want, got)}")
                }
            }
            // F32 forward and gradient graphs against the interpreter.
            val f32Loss = qrLoss(F32)
            for (g in listOf(f32Loss, DxirReverseTransform.apply(f32Loss))) {
                val inputs = listOf(f32(tall), f32(wq), f32(wr))
                val want = DxirInterpreter.evalFunction(g, inputs)
                val got = session.runOn(g, inputs)
                for (r in want.indices) {
                    val d = maxRelDiff(
                        DoubleArray(want[r].size) { want[r][it].toDouble() },
                        DoubleArray(got[r].size) { got[r][it].toDouble() },
                    )
                    assertTrue(d <= TestBackend.defaultDotRelTolerance, "${g.name} result $r: $d")
                }
            }
            // F64 gradient against finite differences.
            fun lossD(a: DoubleArray): Double {
                val (q, r) = LinalgKernels.qr(a, 5, 3)
                return dot(q, wq) + dot(r, wr)
            }
            val g = session.runOnF64(DxirReverseTransform.apply(qrLoss(F64)), listOf(tall, wq, wr))
            val d = maxRelDiff(fdGrad(tall, 1e-6, ::lossD), g[0])
            assertTrue(d <= 1e-7, "qr dA: $d")
        }
    }
}
