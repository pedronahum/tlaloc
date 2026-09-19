package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.pretty
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.386 — the avgpool adjoint is SENTINEL-SAFE, the pooling twin of
 * `ConvAdjointSentinelSafetyTest`.
 *
 * The pre-fusion spelling had two extent dependencies, and the second was worse
 * than conv's: besides solving the transposed conv's `padding` from the primal's
 * extents, it folded channels into the batch dim with a `RESHAPE` whose target was
 * the literal `n * c` — which under `grad {}`'s -1 sentinels is **1**, so the
 * reshape quietly claimed a shape the data did not have. Both are gone:
 * [OpKind.AVGPOOL2D_GRAD] inverts the window at execution time and needs no
 * channel fold at all (the adjoint is per-channel, so the depthwise-via-batch-fold
 * trick — which only existed to dodge grouped-conv support — is unnecessary).
 */
class AvgPoolAdjointSentinelSafetyTest {

    private val sym = DxirType(F32, listOf(-1, -1, -1, -1))
    private val scalar = DxirType(F32, emptyList())

    private val primalAttrs = mapOf(
        "window" to listOf(2, 2),
        "window_strides" to listOf(2, 2),
        "padding" to listOf(listOf(0, 0), listOf(0, 0)),
    )

    /** An avgpool loss with every extent symbolic, as `grad {}` sees it. */
    private fun sentinelAvgPoolLossFn(): DxirFunction =
        DxirBuilder.function("avgpool_loss_symbolic") {
            val x = param("x", sym)
            val y = op(OpKind.AVGPOOL2D, listOf(x), sym, attrs = primalAttrs)
            val y2 = op(OpKind.MUL, listOf(y, y), sym)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }

    private fun adjoints(fn: DxirFunction): List<DxirOp> =
        fn.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.AVGPOOL2D_GRAD }

    @Test
    fun adjointCarriesOnlyLiteralPrimalAttrsUnderSentinelDims() {
        val grads = DxirReverseTransform.apply(sentinelAvgPoolLossFn())
        val adj = adjoints(grads)
        assertEquals(1, adj.size, "expected exactly one fused adjoint:\n${grads.pretty()}")
        val node = adj.single()
        assertEquals(
            primalAttrs, node.attrs,
            "the adjoint must carry only the primal's literal attrs; got ${node.attrs}",
        )
        assertEquals(2, node.operands.size, "AVGPOOL2D_GRAD takes (upstream, xTemplate)")
        // The result type IS the template's, and both are the symbolic param type —
        // no extent was invented anywhere.
        assertContentEquals(sym.dims.toIntArray(), node.type.dims.toIntArray())
        assertContentEquals(
            node.operands[1].type.dims.toIntArray(), node.type.dims.toIntArray(),
            "result type must equal the template's",
        )
    }

    @Test
    fun theTemplateIsThePrimalInputAndNoChannelFoldRemains() {
        val grads = DxirReverseTransform.apply(sentinelAvgPoolLossFn())
        val x = grads.params[0]
        assertEquals(
            x.id, adjoints(grads).single().operands[1].id,
            "the template must be the tensor being differentiated:\n${grads.pretty()}",
        )
        val kinds = grads.body.filterIsInstance<DxirOp>().map { it.op }
        // The channel-folding RESHAPE (whose `n * c` target was 1 under sentinels)
        // and the lhs-dilated CONV_TRANSPOSE2D it fed are both gone.
        assertTrue(
            OpKind.RESHAPE !in kinds,
            "the fused adjoint needs no channel fold:\n${grads.pretty()}",
        )
        assertTrue(
            OpKind.CONV_TRANSPOSE2D !in kinds,
            "the fused adjoint replaces the splat-kernel transposed conv:\n${grads.pretty()}",
        )
    }
}
