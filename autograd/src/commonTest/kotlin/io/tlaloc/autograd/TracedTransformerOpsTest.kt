package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.ops.sum
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The transformer trace spellings: each op's traced forward equals what the
 * interpreter computes for the captured graph, and the reverse transform's
 * gradient equals central finite differences of the captured primal.
 */
class TracedTransformerOpsTest {

    private fun tensor(dims: IntArray, seed: Int): DTensor<Shape, F32> {
        val n = dims.fold(1) { a, d -> a * d }
        // A deterministic spread of distinct values (distinct so max has no ties).
        val v = FloatArray(n) { i -> ((i * 7 + seed * 13) % 23 - 11) / 7f + i * 1e-3f }
        return DTensor(HostF32Storage(v), dims, F32)
    }

    /** Captures `sum(f(x) ⊙ w)` for a fixed weight w, so every output element matters. */
    private fun captureWeighted(x: DTensor<Shape, F32>, f: (Tracer<Shape>) -> Tracer<Shape>): Pair<DxirFunction, FloatArray> {
        var traced = FloatArray(0)
        val fn = captureN(listOf(x)) { leaves ->
            val y = f(leaves[0])
            traced = y.entry.value.copyOf()
            val w = FloatArray(y.size) { i -> 0.3f + 0.1f * (i % 5) - 0.05f * (i % 3) }
            (y * y.constant<Shape>(w, y.dims)).sum()
        }
        return fn to traced
    }

    private fun checkGradient(x: DTensor<Shape, F32>, f: (Tracer<Shape>) -> Tracer<Shape>) {
        val (primal, _) = captureWeighted(x, f)
        val grad = DxirReverseTransform.apply(primal)
        val xv = (x.storage as HostF32Storage).data
        val analytic = DxirInterpreter.evalFunction(grad, listOf(xv)).single()
        val h = 1e-2f
        for (i in xv.indices) {
            val plus = xv.copyOf().also { it[i] += h }
            val minus = xv.copyOf().also { it[i] -= h }
            val fp = DxirInterpreter.evalFunction(primal, listOf(plus)).single()[0]
            val fm = DxirInterpreter.evalFunction(primal, listOf(minus)).single()[0]
            val fd = (fp - fm) / (2 * h)
            assertTrue(
                abs(fd - analytic[i]) <= 2e-2f * max(1f, abs(fd)),
                "gradient element $i: analytic ${analytic[i]} vs finite difference $fd",
            )
        }
    }

    private fun checkForward(x: DTensor<Shape, F32>, f: (Tracer<Shape>) -> Tracer<Shape>) {
        var traced = FloatArray(0)
        val fn = captureN(listOf(x)) { leaves -> f(leaves[0]).also { traced = it.entry.value.copyOf() } }
        val interpreted = DxirInterpreter.evalFunction(fn, listOf((x.storage as HostF32Storage).data)).single()
        assertEquals(traced.size, interpreted.size)
        for (i in traced.indices) {
            assertTrue(abs(traced[i] - interpreted[i]) <= 1e-6f * max(1f, abs(traced[i])), "element $i")
        }
    }

    @Test
    fun transposeMovesAxesAndItsAdjointIsTheInversePermutation() {
        val x = tensor(intArrayOf(2, 3, 4), 1)
        var out: Tracer<Shape>? = null
        captureN(listOf(x)) { leaves -> leaves[0].transpose<Shape>(2, 0, 1).also { out = it } }
        assertContentEquals(intArrayOf(4, 2, 3), out!!.dims)
        // out[k, i, j] = x[i, j, k]
        val xv = (x.storage as HostF32Storage).data
        assertEquals(xv[1 * 12 + 2 * 4 + 3], out!!.entry.value[3 * 6 + 1 * 3 + 2])
        checkForward(x) { it.transpose(2, 0, 1) }
        checkGradient(x) { it.transpose(2, 0, 1) }
        checkGradient(tensor(intArrayOf(2, 2, 3, 2), 2)) { it.transpose(0, 2, 1, 3) }
    }

    @Test
    fun softmaxRowsSumToOneOnAnyAxis() {
        val x = tensor(intArrayOf(2, 3, 4), 3)
        for (axis in listOf(-1, 1, 0)) {
            var out: Tracer<Shape>? = null
            captureN(listOf(x)) { leaves -> leaves[0].softmax(axis).also { out = it } }
            val a = if (axis < 0) axis + 3 else axis
            val sums = out!!.toDTensor().sum(a)
            for (s in (sums.storage as HostF32Storage).data) assertTrue(abs(s - 1f) < 1e-5f, "axis $axis sum $s")
            checkForward(x) { it.softmax(axis) }
            checkGradient(x) { it.softmax(axis) }
        }
    }

    @Test
    fun maxOverAxesDropsThemAndRoutesTheGradientToTheArgmax() {
        val x = tensor(intArrayOf(3, 5), 4)
        checkForward(x) { it.max(intArrayOf(1)) }
        checkGradient(x) { it.max(intArrayOf(1)) }
        checkGradient(tensor(intArrayOf(2, 3, 4), 5)) { it.max(intArrayOf(0, 2)) }
    }

    @Test
    fun broadcastToStretchesSizeOneAxesAndPrependsMissingOnes() {
        // [3, 1] -> [2, 3, 4]: one prepended axis and one stretched axis. The
        // gradient must sum over both.
        val x = tensor(intArrayOf(3, 1), 6)
        var out: Tracer<Shape>? = null
        val fn = captureN(listOf(x)) { leaves -> leaves[0].broadcastTo<Shape>(intArrayOf(2, 3, 4)).also { out = it } }
        assertContentEquals(intArrayOf(2, 3, 4), out!!.dims)
        val broadcasts = fn.body.filterIsInstance<io.tlaloc.ir.DxirOp>().filter { it.op == OpKind.BROADCAST }
        assertEquals(listOf(0, 1, 2), broadcasts.single().attrs["broadcast_dimensions"])
        checkForward(x) { it.broadcastTo(intArrayOf(2, 3, 4)) }
        checkGradient(x) { it.broadcastTo(intArrayOf(2, 3, 4)) }
        checkGradient(tensor(intArrayOf(4), 7)) { it.broadcastTo(intArrayOf(3, 4)) }
        assertFailsWith<IllegalArgumentException> {
            captureN(listOf(x)) { leaves -> leaves[0].broadcastTo<Shape>(intArrayOf(2, 4)) }
        }
    }

    @Test
    fun splatIsOneScalarConstantBroadcast() {
        val x = tensor(intArrayOf(8, 8), 8)
        val fn = captureN(listOf(x)) { leaves -> leaves[0] * leaves[0].splat(0.5f) }
        val consts = fn.body.filterIsInstance<io.tlaloc.ir.DxirConst>()
        assertEquals(1, consts.size)
        assertTrue(consts.single().type.isScalar, "the constant is a scalar, not an 8x8 literal")
        checkGradient(x) { it * it.splat(0.5f) }
    }

    @Test
    fun logSoftmaxMatchesTheLogOfSoftmaxAndStaysFiniteForLargeLogits() {
        val x = tensor(intArrayOf(3, 6), 9)
        checkGradient(x) { it.logSoftmax() }
        // Logits 200 apart: log(softmax) underflows to -inf, the stable form does not.
        val wide = DTensor<Shape, F32>(HostF32Storage(floatArrayOf(0f, 200f, -200f, 1f)), intArrayOf(1, 4), F32)
        var out: Tracer<Shape>? = null
        captureN(listOf(wide)) { leaves -> leaves[0].logSoftmax().also { out = it } }
        val v = out!!.entry.value
        assertEquals(-200f, v[0], 1e-3f)
        assertEquals(0f, v[1], 1e-3f)
        assertEquals(-400f, v[2], 1e-3f)
        val lse = 200.0 + kotlin.math.ln(1.0 + exp(-200.0) + exp(-199.0))
        assertEquals((1.0 - lse).toFloat(), v[3], 1e-3f)
    }
}
