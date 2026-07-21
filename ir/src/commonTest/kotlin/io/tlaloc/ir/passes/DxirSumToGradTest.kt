package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.373 — Phase A2b (DiffKT parity): the runtime-extent SUM_TO adjoint that
 * closes the in-place size-1 broadcast deferral. BroadcastRule's adjoint for an
 * in-place stretch (`[1,C]→[N,C]`, `[N,1]→[N,C]`) must sum the upstream over
 * exactly the axes that were size-1 in the operand and KEEP them size-1 — but
 * which axes those are is unknowable under the -1 sentinel dims of `grad {}`,
 * so it emits `SUM_TO(upstream, template=operand)` reading the extent from the
 * operand's ACTUAL runtime shape (the reverse mirror of BROADCAST's stretch).
 */
class DxirSumToGradTest {

    private val scalar = DxirType(F32, emptyList())

    /** SUM_TO is numpy unbroadcast: pin the shape/value contract directly. */
    @Test
    fun sumToInterpreterPins() {
        // U=[2,3] → T=[1,3]: sum over axis 0, keep it size-1.
        run {
            val fn = DxirBuilder.function("sumto_10") {
                val v = param("v", DxirType(F32, listOf(2, 3)))
                val t = param("t", DxirType(F32, listOf(1, 3)))
                listOf(op(OpKind.SUM_TO, listOf(v, t), DxirType(F32, listOf(1, 3))))
            }
            val v = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(3)))[0]
            assertEquals(3, out.size)
            // col sums: [1+4, 2+5, 3+6] = [5, 7, 9]
            assertTrue(abs(out[0] - 5f) < 1e-6f && abs(out[1] - 7f) < 1e-6f && abs(out[2] - 9f) < 1e-6f)
        }
        // U=[2,3] → T=[2,1]: sum over axis 1, keep it size-1.
        run {
            val fn = DxirBuilder.function("sumto_01") {
                val v = param("v", DxirType(F32, listOf(2, 3)))
                val t = param("t", DxirType(F32, listOf(2, 1)))
                listOf(op(OpKind.SUM_TO, listOf(v, t), DxirType(F32, listOf(2, 1))))
            }
            val v = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(2)))[0]
            assertEquals(2, out.size)
            // row sums: [1+2+3, 4+5+6] = [6, 15]
            assertTrue(abs(out[0] - 6f) < 1e-6f && abs(out[1] - 15f) < 1e-6f)
        }
        // U=[3,2] → T=[2] (rank-reducing leading sum, numpy right-alignment).
        run {
            val fn = DxirBuilder.function("sumto_lead") {
                val v = param("v", DxirType(F32, listOf(3, 2)))
                val t = param("t", DxirType(F32, listOf(2)))
                listOf(op(OpKind.SUM_TO, listOf(v, t), DxirType(F32, listOf(2))))
            }
            val v = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(2)))[0]
            assertEquals(2, out.size)
            // col sums over the leading axis: [1+3+5, 2+4+6] = [9, 12]
            assertTrue(abs(out[0] - 9f) < 1e-6f && abs(out[1] - 12f) < 1e-6f)
        }
        // Identity U==T: reduces nothing, passes through.
        run {
            val fn = DxirBuilder.function("sumto_id") {
                val v = param("v", DxirType(F32, listOf(2, 2)))
                val t = param("t", DxirType(F32, listOf(2, 2)))
                listOf(op(OpKind.SUM_TO, listOf(v, t), DxirType(F32, listOf(2, 2))))
            }
            val v = floatArrayOf(7f, 8f, 9f, 10f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(4)))[0]
            assertTrue(v.indices.all { abs(out[it] - v[it]) < 1e-6f })
        }
    }

    /**
     * In-place size-1 stretch gradient: loss = Σ broadcast(x:[1,2] → [3,2]) ⊙ w.
     *   dx[0,j] = Σ_k w[k,j]   (x[0,j] feeds all 3 rows) — SHAPE [1,2].
     *   dw[k,j] = broadcast(x)[k,j] = x[0,j].
     * BroadcastRule takes the empty-reduceDims branch (equal rank, identity axis
     * map) and emits SUM_TO(upstream, x) — summing the stretched axis 0, keepdim.
     */
    @Test
    fun inPlaceStretchGradientRow() {
        val fn = DxirBuilder.function("stretch_row_loss") {
            val x = param("x", DxirType(F32, listOf(1, 2)))
            val w = param("w", DxirType(F32, listOf(3, 2)))
            val bx = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(3, 2)),
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
            )
            val p = op(OpKind.MUL, listOf(bx, w), DxirType(F32, listOf(3, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(2f, -1f)
        val w = FloatArray(6) { (it * 0.5f) - 1f } // [-1, -0.5, 0, 0.5, 1, 1.5]
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        // dx SHAPE must be [1,2] (size-1 axis kept).
        assertEquals(2, out[0].size, "dx must keep the stretched axis as size-1 → 2 elements")
        for (j in 0 until 2) {
            val want = (0 until 3).sumOf { k -> w[k * 2 + j].toDouble() }.toFloat()
            assertTrue(abs(out[0][j] - want) < 1e-5f, "dx[0,$j] = ${out[0][j]}, want $want")
        }
        for (k in 0 until 3) for (j in 0 until 2) {
            val want = x[j]
            assertTrue(abs(out[1][k * 2 + j] - want) < 1e-6f, "dw[$k,$j] = ${out[1][k * 2 + j]}, want $want")
        }
    }

    /** The column mirror: broadcast(x:[3,1] → [3,2]) — stretched axis is axis 1. */
    @Test
    fun inPlaceStretchGradientCol() {
        val fn = DxirBuilder.function("stretch_col_loss") {
            val x = param("x", DxirType(F32, listOf(3, 1)))
            val w = param("w", DxirType(F32, listOf(3, 2)))
            val bx = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(3, 2)),
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
            )
            val p = op(OpKind.MUL, listOf(bx, w), DxirType(F32, listOf(3, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val x = floatArrayOf(1f, 2f, 3f)
        val w = FloatArray(6) { (it * 0.3f) - 0.5f }
        val out = DxirInterpreter.evalFunction(grad, listOf(x, w))
        assertEquals(3, out[0].size, "dx must keep the stretched axis as size-1 → 3 elements")
        for (k in 0 until 3) {
            val want = (0 until 2).sumOf { j -> w[k * 2 + j].toDouble() }.toFloat()
            assertTrue(abs(out[0][k] - want) < 1e-5f, "dx[$k,0] = ${out[0][k]}, want $want")
        }
    }

    /** JVP⇄VJP cross-identity through the in-place stretch — each mode certifies the other. */
    @Test
    fun inPlaceStretchJvpVjpCrossIdentity() {
        val fn = DxirBuilder.function("stretch_chain") {
            val x = param("x", DxirType(F32, listOf(1, 3)))
            val w = param("w", DxirType(F32, listOf(4, 3)))
            val bx = op(
                OpKind.BROADCAST, listOf(x), DxirType(F32, listOf(4, 3)),
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1)),
            )
            val p = op(OpKind.MUL, listOf(bx, w), DxirType(F32, listOf(4, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val x = floatArrayOf(0.3f, -1.2f, 2.1f)
        val w = FloatArray(12) { (it * 0.13f) - 1.1f }
        val vx = floatArrayOf(0.11f, -0.23f, 0.37f)
        val vw = FloatArray(12) { (it * 0.017f) - 0.2f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x, w))
        var dot = 0.0
        for (i in 0 until 3) dot += grads[0][i].toDouble() * vx[i]
        for (i in 0 until 12) dot += grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(x, w, vx, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through in-place stretch: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
