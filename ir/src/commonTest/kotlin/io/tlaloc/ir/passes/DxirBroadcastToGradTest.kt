package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.pretty
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.371 — Phase A2b (DiffKT parity): rank-increasing `broadcastTo` (NumPy
 * right-alignment — the operand maps to the TRAILING output axes, new leading
 * axes are replicated) differentiates through both transforms at IR level.
 *
 * The primal BROADCAST carries a right-aligned `broadcast_dimensions` suffix;
 * BroadcastRule's adjoint sums the upstream over the COMPLEMENT (the new
 * leading axes), a pure compile-time-position reduction — no operand extent is
 * ever read, so the same rule is sentinel-safe inside `grad {}`.
 */
class DxirBroadcastToGradTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun broadcastToRankIncreaseGradient() {
        // loss = Σ broadcast(x:[2,2] → [3,2,2]) ⊙ w:[3,2,2]
        //   dx[i,j] = Σ_k w[k,i,j]   (x[i,j] appears in all 3 leading slices)
        //   dw[k,i,j] = x[i,j]       (= the broadcast primal)
        val fn = DxirBuilder.function("bcastto_loss") {
            val x = param("x", DxirType(F32, listOf(2, 2)))
            val w = param("w", DxirType(F32, listOf(3, 2, 2)))
            val bx = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(3, 2, 2)),
                attrs = mapOf("broadcast_dimensions" to listOf(1, 2)),
            )
            val p = op(OpKind.MUL, listOf(bx, w), DxirType(F32, listOf(3, 2, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(1f, 3f, 5f, 2f)
        val w = FloatArray(12) { (it * 0.5f) - 2f }
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        // dx[i,j] = Σ_k w[k,i,j]  (w row-major on [3,2,2], strides [4,2,1]).
        for (i in 0 until 2) for (j in 0 until 2) {
            val want = (0 until 3).sumOf { k -> w[k * 4 + i * 2 + j].toDouble() }.toFloat()
            val got = out[0][i * 2 + j]
            assertTrue(abs(got - want) < 1e-5f, "dx[$i,$j] = $got, want $want")
        }
        // dw[k,i,j] = x[i,j].
        for (k in 0 until 3) for (i in 0 until 2) for (j in 0 until 2) {
            val got = out[1][k * 4 + i * 2 + j]
            val want = x[i * 2 + j]
            assertTrue(abs(got - want) < 1e-6f, "dw[$k,$i,$j] = $got, want $want")
        }
    }

    @Test
    fun broadcastToJvpVjpCrossIdentity() {
        // f(x, w) = Σ broadcast(x:[4] → [2,3,4]) ⊙ w:[2,3,4] — a rank-1 → rank-3
        // right-aligned broadcast; ⟨∇f, v⟩ must equal the forward tangent.
        val fn = DxirBuilder.function("bcastto_chain") {
            val x = param("x", DxirType(F32, listOf(4)))
            val w = param("w", DxirType(F32, listOf(2, 3, 4)))
            val bx = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(2, 3, 4)),
                attrs = mapOf("broadcast_dimensions" to listOf(2)),
            )
            val p = op(OpKind.MUL, listOf(bx, w), DxirType(F32, listOf(2, 3, 4)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val x = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f)
        val w = FloatArray(24) { (it * 0.13f) - 1.1f }
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f)
        val vw = FloatArray(24) { (it * 0.017f) - 0.2f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in 0 until 4) dot += grads[0][i].toDouble() * vx[i]
        for (i in 0 until 24) dot += grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through broadcastTo: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }

    /** `Σ broadcast(x) ⊙ w` for a BROADCAST of x to [outDims] along [bdims]; returns (fn, dx per element by brute force). */
    private fun stretchCase(inDims: List<Int>, outDims: List<Int>, bdims: List<Int>): Pair<FloatArray, FloatArray> {
        val fn = DxirBuilder.function("bcast_stretch") {
            val x = param("x", DxirType(F32, inDims))
            val w = param("w", DxirType(F32, outDims))
            val bx = op(OpKind.BROADCAST, listOf(x), DxirType(F32, outDims), attrs = mapOf("broadcast_dimensions" to bdims))
            val p = op(OpKind.MUL, listOf(bx, w), DxirType(F32, outDims))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val nIn = inDims.fold(1) { a, d -> a * d }
        val nOut = outDims.fold(1) { a, d -> a * d }
        val x = FloatArray(nIn) { 0.5f + it }
        val w = FloatArray(nOut) { (it * 0.37f) - 1.3f }
        val got = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))[0]
        // dx[j] = Σ over output positions whose source is j of w.
        val want = FloatArray(nIn)
        val outStrides = IntArray(outDims.size).also { s -> var a = 1; for (k in outDims.indices.reversed()) { s[k] = a; a *= outDims[k] } }
        val inStrides = IntArray(inDims.size).also { s -> var a = 1; for (k in inDims.indices.reversed()) { s[k] = a; a *= inDims[k] } }
        for (o in 0 until nOut) {
            var src = 0
            for ((j, od) in bdims.withIndex()) {
                val coord = (o / outStrides[od]) % outDims[od]
                if (inDims[j] != 1) src += coord * inStrides[j]
            }
            want[src] += w[o]
        }
        return got to want
    }

    @Test
    fun rankIncreasingBroadcastThatAlsoStretchesASizeOneAxis() {
        // [3,1] -> [2,3,4]: axis 0 is new, and input axis 1 (size 1) stretches to 4.
        // The adjoint must sum over both, back to [3,1].
        for ((inDims, outDims, bdims) in listOf(
            Triple(listOf(3, 1), listOf(2, 3, 4), listOf(1, 2)),
            Triple(listOf(1, 4), listOf(3, 5, 4), listOf(0, 2)),
            Triple(listOf(1, 1), listOf(2, 3, 4), listOf(1, 2)),
            Triple(listOf(1), listOf(3, 2), listOf(1)),
        )) {
            val (got, want) = stretchCase(inDims, outDims, bdims)
            assertTrue(got.size == want.size, "$inDims -> $outDims: gradient has ${got.size} elements, want ${want.size}")
            for (i in want.indices) {
                assertTrue(abs(got[i] - want[i]) < 1e-4f, "$inDims -> $outDims dx[$i] = ${got[i]}, want ${want[i]}")
            }
        }
    }

    @Test
    fun underSentinelDimsTheAdjointDefersTheStretchToSumTo() {
        // Under `grad {}` extents are -1, so whether an input axis was stretched
        // is only known at run time: the adjoint sums the inserted axes and then
        // SUM_TOs against the input, which reads its runtime shape.
        val sym2 = DxirType(F32, listOf(-1, -1))
        val sym3 = DxirType(F32, listOf(-1, -1, -1))
        val fn = DxirBuilder.function("bcast_sym") {
            val x = param("x", sym2)
            val w = param("w", sym3)
            val bx = op(OpKind.BROADCAST, listOf(x), sym3, attrs = mapOf("broadcast_dimensions" to listOf(1, 2)))
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(bx, w), sym3)), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val dx = grad.returns[0] as io.tlaloc.ir.DxirOp
        assertTrue(dx.op == OpKind.SUM_TO, "dx must be a SUM_TO, got ${dx.op}:\n${grad.pretty()}")
        assertTrue(dx.operands[1].id == grad.params[0].id, "SUM_TO's template must be x:\n${grad.pretty()}")
        val partial = dx.operands[0] as io.tlaloc.ir.DxirOp
        assertTrue(partial.op == OpKind.SUM && partial.attrs["reduction_dims"] == listOf(0), grad.pretty())
    }
}
