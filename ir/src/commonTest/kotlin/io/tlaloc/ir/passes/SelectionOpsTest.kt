package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * TOP_K, TAKE_ALONG and TAKE_ALONG_GRAD: values, the gradient of a gather with repeated
 * indices (which must accumulate), the gradient through positions chosen by TOP_K inside
 * the same function, and forward mode against reverse.
 */
class SelectionOpsTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun topKOrdersByValueThenPosition() {
        val t = DxirType(F32, listOf(2, 5))
        val fn = DxirBuilder.function("topk") {
            val x = param("x", t)
            listOf(op(OpKind.TOP_K, listOf(x), DxirType(I32, listOf(2, 3)), attrs = mapOf("k" to 3)))
        }
        val x = floatArrayOf(0.5f, 2f, -1f, 2f, 1f, 9f, 8f, 7f, 6f, 5f)
        val got = DxirInterpreter.evalFunction(fn, listOf(x))[0]
        assertContentEquals(floatArrayOf(1f, 3f, 4f, 0f, 1f, 2f), got)
    }

    @Test
    fun takeAlongGathersAndItsGradientAccumulates() {
        // x [2, 3, 2], idx [2, 4] with a repeat in each batch.
        val xT = DxirType(F32, listOf(2, 3, 2))
        val iT = DxirType(I32, listOf(2, 4))
        val oT = DxirType(F32, listOf(2, 4, 2))
        val fn = DxirBuilder.function("take") {
            val x = param("x", xT)
            val idx = param("idx", iT)
            val w = param("w", oT)
            val g = op(OpKind.TAKE_ALONG, listOf(x, idx), oT)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(g, w), oT)), scalar))
        }
        val x = FloatArray(12) { it.toFloat() }
        val idx = floatArrayOf(2f, 0f, 2f, 1f, 0f, 0f, 1f, 2f)
        val w = FloatArray(16) { 1f + it }
        val take = DxirBuilder.function("t") {
            val xp = param("x", xT)
            val ip = param("idx", iT)
            listOf(op(OpKind.TAKE_ALONG, listOf(xp, ip), oT))
        }
        val gathered = DxirInterpreter.evalFunction(take, listOf(x, idx))[0]
        assertContentEquals(floatArrayOf(4f, 5f, 0f, 1f, 4f, 5f, 2f, 3f, 6f, 7f, 6f, 7f, 8f, 9f, 10f, 11f), gathered)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, idx, w))
        val dx = grads[0]
        val want = FloatArray(12)
        for (b in 0 until 2) for (j in 0 until 4) {
            val i = idx[b * 4 + j].toInt()
            for (e in 0 until 2) want[(b * 3 + i) * 2 + e] += w[(b * 4 + j) * 2 + e]
        }
        assertContentEquals(want, dx)
        assertTrue(grads[1].all { it == 0f }, "the indices get no gradient")

        // F64 through the same rules.
        val fn64 = DxirBuilder.function("take64") {
            val xp = param("x", DxirType(F64, listOf(2, 3, 2)))
            val ip = param("idx", iT)
            val wp = param("w", DxirType(F64, listOf(2, 4, 2)))
            val g = op(OpKind.TAKE_ALONG, listOf(xp, ip), DxirType(F64, listOf(2, 4, 2)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(g, wp), DxirType(F64, listOf(2, 4, 2)))), DxirType(F64, emptyList())))
        }
        val dx64 = DxirInterpreterF64.evalFunction(
            DxirReverseTransform.apply(fn64),
            listOf(DoubleArray(12) { it.toDouble() }, DoubleArray(8) { idx[it].toDouble() }, DoubleArray(16) { 1.0 + it }),
        )[0]
        assertContentEquals(DoubleArray(12) { want[it].toDouble() }, dx64)
    }

    @Test
    fun gradientThroughPositionsChosenInTheSameFunction() {
        // loss = Σ take(x, topK(s, 2))², with the scores s a function of x: the
        // positions are constants of the reverse pass, the gathered values are not.
        val xT = DxirType(F32, listOf(1, 4, 1))
        val fn = DxirBuilder.function("routed") {
            val x = param("x", xT)
            val s = op(OpKind.RESHAPE, listOf(x), DxirType(F32, listOf(1, 4)))
            val idx = op(OpKind.TOP_K, listOf(s), DxirType(I32, listOf(1, 2)), attrs = mapOf("k" to 2))
            val g = op(OpKind.TAKE_ALONG, listOf(x, idx), DxirType(F32, listOf(1, 2, 1)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(g, g), DxirType(F32, listOf(1, 2, 1)))), scalar))
        }
        val x = floatArrayOf(0.5f, 3f, -2f, 1.5f)
        val dx = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x))[0]
        assertContentEquals(floatArrayOf(0f, 6f, 0f, 3f), dx)

        val v = floatArrayOf(0.1f, -0.2f, 0.3f, 0.4f)
        val tangent = DxirInterpreter.evalFunction(DxirForwardTransform.apply(fn), listOf(x, v))[1].single()
        val dot = (0 until 4).sumOf { dx[it].toDouble() * v[it] }
        assertTrue(abs(tangent - dot) < 1e-6, "tangent $tangent against ⟨grad, v⟩ = $dot")
    }

    @Test
    fun takeAlongOfIntegerPositions() {
        // Positions of positions: idx [1, 3, 2] picked per query by a [1, 4] routing.
        val fn = DxirBuilder.function("ii") {
            val table = param("table", DxirType(I32, listOf(1, 3, 2)))
            val route = param("route", DxirType(I32, listOf(1, 4)))
            listOf(op(OpKind.TAKE_ALONG, listOf(table, route), DxirType(I32, listOf(1, 4, 2))))
        }
        val got = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(7f, 1f, 4f, 2f, 0f, 5f), floatArrayOf(2f, 2f, 0f, 1f)))[0]
        assertContentEquals(floatArrayOf(0f, 5f, 0f, 5f, 7f, 1f, 4f, 2f), got)
    }
}
