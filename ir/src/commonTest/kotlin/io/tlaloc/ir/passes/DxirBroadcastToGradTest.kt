package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
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
}
