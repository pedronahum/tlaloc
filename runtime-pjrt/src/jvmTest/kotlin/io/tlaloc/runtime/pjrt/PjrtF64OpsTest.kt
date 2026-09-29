package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreterF64
import io.tlaloc.ir.passes.DxirReverseTransform
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The ops the F32 smoke tests certify on the device (conv, transposed conv, pooling, flip,
 * tan/atan, lgamma/digamma) and the linear-algebra kinds, at F64: each loss and its
 * reverse-transform gradient graph run through [PjrtSession.runOnHost] and are compared
 * with [DxirInterpreterF64] on the same graph. The graphs and attributes are the smoke
 * tests' (PjrtConvSmokeTest, PjrtConvTransposeSmokeTest, PjrtPoolingSmokeTest,
 * PjrtFlipSmokeTest, PjrtTanAtanSmokeTest, PjrtLgammaDigammaSmokeTest) at F64.
 *
 * Tolerance 1e-11 of the largest entry: both sides compute in f64 and differ in summation
 * order and in the last bits of the transcendental and special functions (XLA's lgamma and
 * digamma are accurate to a few ulp in f64). An f32 step anywhere would differ by 1e-8 or
 * more. The LU, QR and Jacobi lowerings are `stablehlo.while` loops, so the linear-algebra
 * graphs also check that those loops run at f64.
 */
class PjrtF64OpsTest {

    private fun t(vararg dims: Int) = DxirType(F64, dims.toList())

    private val scalar = t()

    private fun maxRelDiff(want: DoubleArray, got: DoubleArray): Double {
        val scale = max(1e-300, want.maxOf { abs(it) })
        return want.indices.maxOf { abs(want[it] - got[it]) } / scale
    }

    /** `Σ y²` of `y = body(params)`. */
    private fun squareLoss(name: String, params: List<DxirType>, yType: DxirType, body: DxirBuilder.(List<DxirNode>) -> DxirNode) =
        DxirBuilder.function(name) {
            val ps = params.mapIndexed { i, p -> param("p$i", p) }
            val y = body(ps)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), yType)), scalar))
        }

    private fun random(n: Int, seed: Int, lo: Double = -0.5, hi: Double = 0.5): DoubleArray {
        val r = kotlin.random.Random(seed)
        return DoubleArray(n) { lo + (hi - lo) * r.nextDouble() }
    }

    private val cases: List<Pair<DxirFunction, List<DoubleArray>>> by lazy {
        val spd = doubleArrayOf(6.2, 1.1, -0.7, 0.4, 1.3, 5.1, 0.9, -1.2, -0.5, 0.9, 4.8, 0.6, 0.4, -1.0, 0.6, 5.5)
        val gen = doubleArrayOf(0.5, 2.0, -1.0, 0.3, 1.2, -0.4, 0.8, 2.2, -3.0, 0.7, 1.5, -0.2, 0.9, 1.1, -0.6, 1.4)
        val tall = doubleArrayOf(1.2, -0.7, 0.3, 0.4, 2.1, -1.1, -0.9, 0.5, 1.7, 0.6, -1.3, 0.2, 1.5, 0.8, -0.4)
        val rhs = doubleArrayOf(0.7, -1.2, 0.4, 2.1, -0.3, 0.8, 1.5, -0.6)
        listOf(
            squareLoss("conv", listOf(t(2, 2, 5, 4), t(3, 2, 3, 2)), t(2, 3, 2, 4)) { ps ->
                op(
                    OpKind.CONV2D, ps, t(2, 3, 2, 4),
                    attrs = mapOf(
                        "window_strides" to listOf(2, 1),
                        "padding" to listOf(listOf(1, 0), listOf(1, 1)),
                        "rhs_dilation" to listOf(1, 2),
                    ),
                )
            } to listOf(random(80, 1), random(36, 2)),
            squareLoss("conv_transpose", listOf(t(1, 2, 3, 3), t(2, 3, 2, 2)), t(1, 3, 6, 6)) { ps ->
                op(
                    OpKind.CONV_TRANSPOSE2D, ps, t(1, 3, 6, 6),
                    attrs = mapOf(
                        "window_strides" to listOf(1, 1),
                        "lhs_dilation" to listOf(2, 2),
                        "padding" to listOf(listOf(1, 1), listOf(1, 1)),
                    ),
                )
            } to listOf(random(18, 3), random(24, 4)),
            squareLoss("avg_pool", listOf(t(2, 3, 6, 6)), t(2, 3, 3, 3)) { ps ->
                op(
                    OpKind.AVGPOOL2D, ps, t(2, 3, 3, 3),
                    attrs = mapOf(
                        "window" to listOf(3, 3),
                        "window_strides" to listOf(2, 2),
                        "padding" to listOf(listOf(1, 0), listOf(1, 0)),
                    ),
                )
            } to listOf(random(216, 5)),
            // Distinct values keep the max-pool argmax unambiguous on both sides.
            squareLoss("max_pool", listOf(t(2, 3, 6, 6)), t(2, 3, 3, 3)) { ps ->
                op(OpKind.MAXPOOL2D, ps, t(2, 3, 3, 3), attrs = mapOf("window" to listOf(2, 2)))
            } to listOf((0 until 216).shuffled(kotlin.random.Random(9)).map { it * 0.07 - 7.56 }.toDoubleArray()),
            squareLoss("flip", listOf(t(2, 3)), t(2, 3)) { ps ->
                val w = const(doubleArrayOf(1.5, -0.8, 0.25, 2.0, -1.1, 0.6), t(2, 3))
                op(OpKind.MUL, listOf(op(OpKind.REVERSE, ps, t(2, 3), attrs = mapOf("dimensions" to listOf(0, 1))), w), t(2, 3))
            } to listOf(doubleArrayOf(0.3, -0.7, 1.1, 0.1, -1.3, 0.9)),
            squareLoss("tan_atan", listOf(t(2, 3)), t(2, 3)) { ps ->
                op(OpKind.ADD, listOf(op(OpKind.TAN, ps, t(2, 3)), op(OpKind.ATAN, ps, t(2, 3))), t(2, 3))
            } to listOf(doubleArrayOf(0.3, -0.7, 1.1, 0.1, -1.3, 0.9)),
            // On the positive axis, away from the poles.
            squareLoss("lgamma_digamma", listOf(t(2, 3)), t(2, 3)) { ps ->
                op(OpKind.ADD, listOf(op(OpKind.LGAMMA, ps, t(2, 3)), op(OpKind.DIGAMMA, ps, t(2, 3))), t(2, 3))
            } to listOf(doubleArrayOf(0.5, 1.3, 2.7, 4.1, 0.9, 3.3)),
            squareLoss("cholesky", listOf(t(4, 4)), t(4, 4)) { ps -> op(OpKind.CHOLESKY, ps, t(4, 4)) } to listOf(spd),
            squareLoss("solve", listOf(t(4, 4), t(4, 2)), t(4, 2)) { ps ->
                op(OpKind.SOLVE, ps, t(4, 2), attrs = mapOf("transpose_a" to false))
            } to listOf(gen, rhs),
            squareLoss("det", listOf(t(4, 4)), scalar) { ps -> op(OpKind.DET, ps, scalar) } to listOf(gen),
            // Σ Q² is constant (Q has orthonormal columns), so the loss is Σ (Q ⊙ Q)² instead.
            squareLoss("qr_q", listOf(t(5, 3)), t(5, 3)) { ps ->
                val q = op(OpKind.QR_Q, ps, t(5, 3))
                op(OpKind.MUL, listOf(q, q), t(5, 3))
            } to listOf(tall),
            squareLoss("qr_r", listOf(t(5, 3)), t(3, 3)) { ps -> op(OpKind.QR_R, ps, t(3, 3)) } to listOf(tall),
            squareLoss("eigh_w", listOf(t(4, 4)), t(4)) { ps -> op(OpKind.EIGH_W, ps, t(4)) } to listOf(gen),
        )
    }

    @Test
    fun f64LossesAndGradientsOfConvPoolingSpecialAndLinalgOpsMatchTheF64Interpreter() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val failures = mutableListOf<String>()
        TestBackend.session().use { session ->
            for ((fn, inputs) in cases) {
                for (g in listOf(fn, DxirReverseTransform.apply(fn))) {
                    val want = DxirInterpreterF64.evalFunction(g, inputs)
                    val got = session.runOnHost(g, inputs)
                    for (r in want.indices) {
                        val d = maxRelDiff(want[r], got[r] as DoubleArray)
                        println("[pjrt-f64-ops] ${g.name} result $r: $d")
                        if (!(d <= 1e-11)) failures += "${g.name} result $r: device differs from the F64 interpreter by $d"
                    }
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.joinToString("\n"))
    }
}
