package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.360 — shape-plumbing activation pins (DiffKT-gap item 2):
 * CONCAT/SLICE go from dormant enum entries to fully differentiable;
 * WHERE/COMPARE/PAD are new. Forward semantics and analytic gradients
 * both pinned through [DxirReverseTransform] + [DxirInterpreter].
 */
class DxirShapePlumbingTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun concatForwardAndGradientSliceBack() {
        // concat([2,2] ⊕ [2,3], dim=1) → [2,5]; loss = sum. Gradients are
        // ones at each operand's own shape (upstream sliced back).
        val aT = DxirType(F32, listOf(2, 2))
        val bT = DxirType(F32, listOf(2, 3))
        val fn = DxirBuilder.function("concat_loss") {
            val a = param("a", aT)
            val b = param("b", bT)
            val c = op(
                OpKind.CONCAT, listOf(a, b), DxirType(F32, listOf(2, 5)),
                attrs = mapOf("dimension" to 1),
            )
            listOf(op(OpKind.SUM, listOf(c), scalar))
        }
        // Forward pin.
        val fwd = DxirInterpreter.evalFunction(
            fn, listOf(floatArrayOf(1f, 2f, 3f, 4f), floatArrayOf(5f, 6f, 7f, 8f, 9f, 10f)),
        )
        assertEquals(55f, fwd.single().single())
        // Gradient pin.
        val grad = DxirReverseTransform.apply(fn)
        val out = DxirInterpreter.evalFunction(
            grad, listOf(floatArrayOf(1f, 2f, 3f, 4f), floatArrayOf(5f, 6f, 7f, 8f, 9f, 10f)),
        )
        assertEquals(List(4) { 1f }, out[0].toList())
        assertEquals(List(6) { 1f }, out[1].toList())
    }

    @Test
    fun sliceForwardAndGradientPadsBack() {
        // slice(x[2,4] → rows 0..2, cols 1..3); loss = sum. dx = zeros with
        // ones in the sliced window.
        val xT = DxirType(F32, listOf(2, 4))
        val fn = DxirBuilder.function("slice_loss") {
            val x = param("x", xT)
            val sl = op(
                OpKind.SLICE, listOf(x), DxirType(F32, listOf(2, 2)),
                attrs = mapOf(
                    "start_indices" to listOf(0, 1),
                    "limit_indices" to listOf(2, 3),
                    "strides" to listOf(1, 1),
                ),
            )
            listOf(op(OpKind.SUM, listOf(sl), scalar))
        }
        val x = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        assertEquals(2f + 3f + 6f + 7f, DxirInterpreter.evalFunction(fn, listOf(x)).single().single())
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x))
        assertEquals(listOf(0f, 1f, 1f, 0f, 0f, 1f, 1f, 0f), out.single().toList())
    }

    @Test
    fun whereRoutesGradientsByComparisonMask() {
        // y = where(a > b, a, b) (elementwise max via plumbing); loss = sum.
        // da = mask, db = 1 - mask.
        val t = DxirType(F32, listOf(4))
        val bT = DxirType(Bool, listOf(4))
        val fn = DxirBuilder.function("where_loss") {
            val a = param("a", t)
            val b = param("b", t)
            val m = op(OpKind.COMPARE, listOf(a, b), bT, attrs = mapOf("direction" to "GT"))
            val y = op(OpKind.WHERE, listOf(m, a, b), t)
            listOf(op(OpKind.SUM, listOf(y), scalar))
        }
        val a = floatArrayOf(1f, 5f, 2f, 7f)
        val b = floatArrayOf(3f, 4f, 2f, 6f)
        // Forward: elementwise max, ties (2 vs 2, GT false) take b.
        assertEquals(listOf(3f, 5f, 2f, 7f).sum(), DxirInterpreter.evalFunction(fn, listOf(a, b)).single().single())
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(a, b))
        assertEquals(listOf(0f, 1f, 0f, 1f), out[0].toList(), "da = (a > b) mask")
        assertEquals(listOf(1f, 0f, 1f, 0f), out[1].toList(), "db = complement")
    }

    @Test
    fun padForwardAndGradientSlicesBack() {
        val xT = DxirType(F32, listOf(2, 2))
        val fn = DxirBuilder.function("pad_loss") {
            val x = param("x", xT)
            val p = op(
                OpKind.PAD, listOf(x), DxirType(F32, listOf(3, 4)),
                attrs = mapOf("low" to listOf(1, 1), "high" to listOf(0, 1)),
            )
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val x = floatArrayOf(1f, 2f, 3f, 4f)
        assertEquals(10f, DxirInterpreter.evalFunction(fn, listOf(x)).single().single())
        val out = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(x))
        assertEquals(List(4) { 1f }, out.single().toList())
    }
}
