package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Forward values and gradients against JAX (`jnp.linalg.cholesky` with
 * `symmetrize_input=True`, `jax.scipy.linalg.solve_triangular`), float64 references
 * from `harness/python/linalg_jax_goldens.py` (jax 0.10.0, CPU). Each loss is
 * `Σ y³` of the op's result. Tlaloc runs in the F32 interpreter here, so the bound
 * is 1e-4 of the largest entry, the F32 tolerance `DxirLinalgGradTest` justifies.
 *
 * Pinning JAX's triangular-solve gradient also pins the convention: JAX returns a
 * gradient only on the triangle the solve reads, as Tlaloc does.
 */
class LinalgJaxParityTest {

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

    // harness/python/linalg_jax_goldens.py
    val choleskyValue = doubleArrayOf(2.4899799195977463, 0.0, 0.0, 0.0, 0.48193159734149943, 2.2062959764011425, 0.0, 0.0, -0.24096579867074966, 0.46055880223085477, 2.128337631476418, 0.0, 0.1606438657804998, -0.5336633740135301, 0.4155792899907212, 2.239797045572299)
    val choleskyGrad = doubleArrayOf(3.9543073099555373, -0.7174215644614106, 0.5188007607661466, -0.4692643391297796, -0.7174215644614106, 3.7203570669819923, -0.6732313537063224, 1.1178082056831493, 0.5188007607661466, -0.6732313537063224, 3.2968329581538844, -0.5342952465162896, -0.4692643391297796, 1.1178082056831493, -0.5342952465162896, 3.359695568358448)
    val solveLowerNValue = doubleArrayOf(0.35, -0.6, 0.11176470588235296, 1.4470588235294117, -0.08063725490196079, 0.0524509803921569, 0.6919633642930856, 0.5639060887512899)
    val solveLowerNGradA = doubleArrayOf(-0.4269375518023272, 0.0, 0.0, 0.0, 2.2281698100296983, -5.760168582056423, 0.0, 0.0, -0.008416124930425966, 0.16308399267349044, -0.006739199417645123, 0.0, 0.036646408460739, -0.8110496431386093, 0.03462836626244022, -0.8062702117367292)
    val solveLowerNGradB = doubleArrayOf(-0.307588337827235, -0.8909891167364324, 0.40417868823024955, 3.949387251517143, -0.14937641558780757, -0.10116311731026435, 0.7560209960902191, 0.5020895951538594)
    val solveLowerTValue = doubleArrayOf(-0.2605263157894737, -0.6829463364293085, 0.6578947368421053, 1.0162538699690402, -0.2894736842105263, 0.39912280701754393, 0.7894736842105263, -0.3157894736842105)
    val solveLowerTGradA = doubleArrayOf(0.5043297683379125, 0.0, 0.0, 0.0, -0.7779760206160661, -2.0800885996423073, 0.0, 0.0, -0.2497641257749887, -0.4181619143874793, -0.03851138834331137, 0.0, 0.14055669255558442, -0.07707734989012956, 0.013249526447272507, -0.8297241890472692)
    val solveLowerTGradB = doubleArrayOf(0.10181094182825487, 0.6996235476633212, 0.7278764054098095, 1.575612738872447, 0.030727706941502352, 0.11877610110345216, 1.2342702723128907, 0.45821574879920757)
    val solveUpperNValue = doubleArrayOf(-0.2605263157894737, -0.6829463364293085, 0.6578947368421053, 1.0162538699690402, -0.2894736842105263, 0.39912280701754393, 0.7894736842105263, -0.3157894736842105)
    val solveUpperNGradA = doubleArrayOf(0.5043297683379125, -0.7779760206160661, -0.2497641257749887, 0.14055669255558442, 0.0, -2.0800885996423073, -0.4181619143874793, -0.07707734989012956, 0.0, 0.0, -0.03851138834331137, 0.013249526447272507, 0.0, 0.0, 0.0, -0.8297241890472692)
    val solveUpperNGradB = doubleArrayOf(0.10181094182825487, 0.6996235476633212, 0.7278764054098095, 1.575612738872447, 0.030727706941502352, 0.11877610110345216, 1.2342702723128907, 0.45821574879920757)
    val solveUpperTValue = doubleArrayOf(0.35, -0.6, 0.11176470588235296, 1.4470588235294117, -0.08063725490196079, 0.0524509803921569, 0.6919633642930856, 0.5639060887512899)
    val solveUpperTGradA = doubleArrayOf(-0.4269375518023272, 2.2281698100296983, -0.008416124930425966, 0.036646408460739, 0.0, -5.760168582056423, 0.16308399267349044, -0.8110496431386093, 0.0, 0.0, -0.006739199417645123, 0.03462836626244022, 0.0, 0.0, 0.0, -0.8062702117367292)
    val solveUpperTGradB = doubleArrayOf(-0.307588337827235, -0.8909891167364324, 0.40417868823024955, 3.949387251517143, -0.14937641558780757, -0.10116311731026435, 0.7560209960902191, 0.5020895951538594)

    private val scalar = DxirType(F32, emptyList())

    private fun f(a: DoubleArray) = FloatArray(a.size) { a[it].toFloat() }

    private fun assertClose(want: DoubleArray, got: FloatArray, what: String) {
        assertEquals(want.size, got.size, "$what size")
        val scale = max(1e-12, want.maxOf { abs(it) })
        for (i in want.indices) {
            assertTrue(abs(want[i] - got[i]) <= 1e-4 * scale, "$what[$i] = ${got[i]}, JAX ${want[i]}")
        }
    }

    /** `(y, Σ y³)` for `y = body(params)`; returns the function and y's node. */
    private fun cubeLoss(
        name: String,
        params: List<DxirType>,
        yType: DxirType,
        body: DxirBuilder.(List<io.tlaloc.ir.DxirNode>) -> io.tlaloc.ir.DxirNode,
    ): Pair<DxirFunction, DxirFunction> {
        val value = DxirBuilder.function("${name}_value") {
            val ps = params.mapIndexed { i, t -> param("p$i", t) }
            listOf(body(ps))
        }
        val loss = DxirBuilder.function(name) {
            val ps = params.mapIndexed { i, t -> param("p$i", t) }
            val y = body(ps)
            val y3 = op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(y, y), yType), y), yType)
            listOf(op(OpKind.SUM, listOf(y3), scalar))
        }
        return value to loss
    }

    @Test
    fun choleskyMatchesJax() {
        val t = DxirType(F32, listOf(n, n))
        val (value, loss) = cubeLoss("chol", listOf(t), t) { ps -> op(OpKind.CHOLESKY, listOf(ps[0]), t) }
        assertClose(choleskyValue, DxirInterpreter.evalFunction(value, listOf(f(spd))).single(), "L")
        assertClose(choleskyGrad, DxirInterpreter.evalFunction(DxirReverseTransform.apply(loss), listOf(f(spd))).single(), "dA")
    }

    @Test
    fun triangularSolveMatchesJaxForEveryTriangleAndTranspose() {
        val at = DxirType(F32, listOf(n, n))
        val bt = DxirType(F32, listOf(n, 2))
        val cases = listOf(
            Triple(true, false, listOf(solveLowerNValue, solveLowerNGradA, solveLowerNGradB)),
            Triple(true, true, listOf(solveLowerTValue, solveLowerTGradA, solveLowerTGradB)),
            Triple(false, false, listOf(solveUpperNValue, solveUpperNGradA, solveUpperNGradB)),
            Triple(false, true, listOf(solveUpperTValue, solveUpperTGradA, solveUpperTGradB)),
        )
        for ((lower, transposeA, want) in cases) {
            val a = if (lower) tri else DoubleArray(n * n) { tri[(it % n) * n + it / n] }
            val (value, loss) = cubeLoss("trsm", listOf(at, bt), bt) { ps ->
                op(
                    OpKind.TRIANGULAR_SOLVE, ps, bt,
                    attrs = mapOf("lower" to lower, "transpose_a" to transposeA, "unit_diagonal" to false),
                )
            }
            val tag = "lower=$lower transpose=$transposeA"
            assertClose(want[0], DxirInterpreter.evalFunction(value, listOf(f(a), f(rhs))).single(), "X $tag")
            val g = DxirInterpreter.evalFunction(DxirReverseTransform.apply(loss), listOf(f(a), f(rhs)))
            assertClose(want[1], g[0], "dA $tag")
            assertClose(want[2], g[1], "dB $tag")
        }
    }
}
