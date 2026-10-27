package io.tlaloc.ir

import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirInterpreterF64
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RaggedDotTest {

    private val m = 7
    private val k = 3
    private val n = 4
    // Group 1 is empty and the last row is in no group.
    private val sizes = intArrayOf(2, 0, 3, 1)
    private val g = sizes.size
    private val r = Random(5)
    private fun values(size: Int) = DoubleArray(size) { r.nextDouble() - 0.5 }

    /** The group of each row, by hand. */
    private val groupOf = intArrayOf(0, 0, 2, 2, 2, 3, -1)

    private fun rows(): DxirFunction = DxirBuilder.function("rd") {
        val lhs = param("lhs", DxirType(F64, listOf(m, k)))
        val rhs = param("rhs", DxirType(F64, listOf(g, k, n)))
        val s = param("sizes", DxirType(I32, listOf(g)))
        listOf(op(OpKind.RAGGED_DOT, listOf(lhs, rhs, s), DxirType(F64, listOf(m, n))))
    }

    private fun contraction(): DxirFunction = DxirBuilder.function("rdc") {
        val lhs = param("lhs", DxirType(F64, listOf(m, k)))
        val rhs = param("rhs", DxirType(F64, listOf(m, n)))
        val s = param("sizes", DxirType(I32, listOf(g)))
        listOf(op(OpKind.RAGGED_DOT, listOf(lhs, rhs, s), DxirType(F64, listOf(g, k, n)), attrs = mapOf("ragged" to RaggedDotAttrs.CONTRACTION)))
    }

    private val sizesD = DoubleArray(g) { sizes[it].toDouble() }

    @Test
    fun rowsMultipliesEachGroupByItsMatrix() {
        val lhs = values(m * k)
        val rhs = values(g * k * n)
        val y = DxirInterpreterF64.evalFunction(rows(), listOf(lhs, rhs, sizesD))[0]
        for (i in 0 until m) for (j in 0 until n) {
            val e = groupOf[i]
            val want = if (e < 0) 0.0 else (0 until k).sumOf { lhs[i * k + it] * rhs[(e * k + it) * n + j] }
            assertEquals(want, y[i * n + j], 1e-15, "[$i, $j]")
        }
    }

    @Test
    fun contractionSumsEachGroupsOuterProducts() {
        val lhs = values(m * k)
        val rhs = values(m * n)
        val y = DxirInterpreterF64.evalFunction(contraction(), listOf(lhs, rhs, sizesD))[0]
        for (e in 0 until g) for (l in 0 until k) for (j in 0 until n) {
            val want = (0 until m).filter { groupOf[it] == e }.sumOf { lhs[it * k + l] * rhs[it * n + j] }
            assertEquals(want, y[(e * k + l) * n + j], 1e-15, "[$e, $l, $j]")
        }
    }

    /** Σ f(lhs, rhs) ⊙ w: its gradient against central differences, for both operands. */
    private fun checkGradient(f: DxirFunction) {
        val outType = f.returns[0].type
        val loss = DxirBuilder.function("loss") {
            val lhs = param("lhs", f.params[0].type)
            val rhs = param("rhs", f.params[1].type)
            val s = param("sizes", f.params[2].type)
            val w = param("w", outType)
            val y = op(OpKind.RAGGED_DOT, listOf(lhs, rhs, s), outType, attrs = (f.returns[0] as DxirOp).attrs)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, w), outType)), DxirType(F64, emptyList())))
        }
        val grad = DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 2)
        val lhs = values(f.params[0].type.elementCount.toInt())
        val rhs = values(f.params[1].type.elementCount.toInt())
        val w = values(outType.elementCount.toInt())
        val got = DxirInterpreterF64.evalFunction(grad, listOf(lhs, rhs, sizesD, w))
        assertEquals(2, got.size)
        fun value(a: DoubleArray, b: DoubleArray) = DxirInterpreterF64.evalFunction(loss, listOf(a, b, sizesD, w))[0][0]
        val h = 1e-6
        for ((which, x) in listOf(lhs, rhs).withIndex()) for (i in x.indices) {
            val plus = x.copyOf().also { it[i] += h }
            val minus = x.copyOf().also { it[i] -= h }
            val fd = if (which == 0) (value(plus, rhs) - value(minus, rhs)) / (2 * h) else (value(lhs, plus) - value(lhs, minus)) / (2 * h)
            assertEquals(fd, got[which][i], 1e-8, "${f.name} d${if (which == 0) "lhs" else "rhs"}[$i]")
        }
        // The tangent: the directional derivative along random directions.
        val jvp = DxirForwardTransform.apply(f)
        val dl = values(lhs.size)
        val dr = values(rhs.size)
        val tangent = DxirInterpreterF64.evalFunction(jvp, listOf(lhs, rhs, sizesD, dl, dr, DoubleArray(g))).last()
        val ya = DxirInterpreterF64.evalFunction(f, listOf(DoubleArray(lhs.size) { lhs[it] + h * dl[it] }, DoubleArray(rhs.size) { rhs[it] + h * dr[it] }, sizesD))[0]
        val yb = DxirInterpreterF64.evalFunction(f, listOf(DoubleArray(lhs.size) { lhs[it] - h * dl[it] }, DoubleArray(rhs.size) { rhs[it] - h * dr[it] }, sizesD))[0]
        for (i in tangent.indices) assertTrue(abs((ya[i] - yb[i]) / (2 * h) - tangent[i]) < 1e-8, "${f.name} tangent[$i]")
    }

    @Test
    fun theGradientsOfBothFormsMatchFiniteDifferences() {
        checkGradient(rows())
        checkGradient(contraction())
    }
}
