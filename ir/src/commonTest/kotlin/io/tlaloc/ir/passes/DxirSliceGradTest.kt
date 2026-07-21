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
 * §0.4.374 — Phase A2b (DiffKT parity): the runtime-extent PAD_TO adjoint that
 * makes single-axis `slice` differentiable in `grad {}`. SliceRule zero-pads the
 * upstream back into the sliced operand's window, but the trailing pad per axis
 * (`high[i] = operand.dim[i] − start[i] − upstream.dim[i]`) reads the operand's
 * extent — a -1 sentinel under `grad {}`. PAD_TO reads that extent from the
 * operand's ACTUAL runtime shape instead of baking it as an attr (the reverse
 * mirror of SLICE, dual of SUM_TO's runtime-extent unbroadcast).
 */
class DxirSliceGradTest {

    private val scalar = DxirType(F32, emptyList())

    /** PAD_TO is a zero-pad-to-template: pin the shape/value contract directly. */
    @Test
    fun padToInterpreterPins() {
        // value=[2,2] into template=[2,4] with low=[0,1]: cols 1..2 filled, 0 and 3 zero.
        run {
            val fn = DxirBuilder.function("padto_row") {
                val v = param("v", DxirType(F32, listOf(2, 2)))
                val t = param("t", DxirType(F32, listOf(2, 4)))
                listOf(op(OpKind.PAD_TO, listOf(v, t), DxirType(F32, listOf(2, 4)), attrs = mapOf("low" to listOf(0, 1))))
            }
            val v = floatArrayOf(1f, 2f, 3f, 4f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(8)))[0]
            assertEquals(8, out.size)
            assertEquals(listOf(0f, 1f, 2f, 0f, 0f, 3f, 4f, 0f), out.toList())
        }
        // value=[2,2] into template=[4,2] with low=[1,0]: rows 1..2 filled.
        run {
            val fn = DxirBuilder.function("padto_col") {
                val v = param("v", DxirType(F32, listOf(2, 2)))
                val t = param("t", DxirType(F32, listOf(4, 2)))
                listOf(op(OpKind.PAD_TO, listOf(v, t), DxirType(F32, listOf(4, 2)), attrs = mapOf("low" to listOf(1, 0))))
            }
            val v = floatArrayOf(5f, 6f, 7f, 8f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(8)))[0]
            assertEquals(listOf(0f, 0f, 5f, 6f, 7f, 8f, 0f, 0f), out.toList())
        }
    }

    /**
     * slice gradient: loss = Σ slice(a[4,2] → rows 1..3, axis 0) ⊙ w[2,2].
     *   d slice_out = w, so da = PAD_TO(w, a, low=[1,0]) — zeros outside rows 1..2.
     *   dw = slice(a) recomputed = a[rows 1..2].
     * SliceRule takes the runtime-extent PAD_TO path (a's row extent is a -1
     * sentinel in a real grad {}; here it is concrete so the interpreter runs).
     */
    @Test
    fun sliceGradientPadsBackViaTemplate() {
        val fn = DxirBuilder.function("slice_loss") {
            val a = param("a", DxirType(F32, listOf(4, 2)))
            val w = param("w", DxirType(F32, listOf(2, 2)))
            val sl = op(
                OpKind.SLICE, listOf(a), DxirType(F32, listOf(2, 2)),
                attrs = mapOf(
                    "start_indices" to listOf(1, 0),
                    "limit_indices" to listOf(3, 2),
                    "strides" to listOf(1, 1),
                ),
            )
            val p = op(OpKind.MUL, listOf(sl, w), DxirType(F32, listOf(2, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val a = FloatArray(8) { it.toFloat() } // rows: [0,1],[2,3],[4,5],[6,7]
        val w = floatArrayOf(10f, 20f, 30f, 40f)
        val out = DxirInterpreter.evalFunction(grad, listOf(a, w))
        // da = w zero-padded into rows 1..2 of a [4,2].
        assertEquals(8, out[0].size, "da must keep a's full shape [4,2]")
        assertEquals(listOf(0f, 0f, 10f, 20f, 30f, 40f, 0f, 0f), out[0].toList(), "da")
        // dw = a[rows 1..2] = [2,3,4,5].
        assertEquals(listOf(2f, 3f, 4f, 5f), out[1].toList(), "dw = slice(a)")
    }

    /** JVP⇄VJP cross-identity through the slice — each mode certifies the other. */
    @Test
    fun sliceJvpVjpCrossIdentity() {
        val fn = DxirBuilder.function("slice_chain") {
            val a = param("a", DxirType(F32, listOf(4, 3)))
            val w = param("w", DxirType(F32, listOf(2, 3)))
            val sl = op(
                OpKind.SLICE, listOf(a), DxirType(F32, listOf(2, 3)),
                attrs = mapOf(
                    "start_indices" to listOf(1, 0),
                    "limit_indices" to listOf(3, 3),
                    "strides" to listOf(1, 1),
                ),
            )
            val p = op(OpKind.MUL, listOf(sl, w), DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val a = FloatArray(12) { (it * 0.21f) - 1.0f }
        val w = FloatArray(6) { (it * 0.13f) - 0.4f }
        val va = FloatArray(12) { (it * 0.03f) - 0.1f }
        val vw = FloatArray(6) { (it * 0.017f) - 0.05f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(a, w))
        var dot = 0.0
        for (i in 0 until 12) dot += grads[0][i].toDouble() * va[i]
        for (i in 0 until 6) dot += grads[1][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(a, w, va, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through slice: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
