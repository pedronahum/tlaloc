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
 * §0.4.385 — the conv adjoint is SENTINEL-SAFE: under `grad {}`'s -1 dims the
 * gradient graph must contain no extent-derived number at all.
 *
 * This is the regression pin for the bug the fused adjoint ops exist to fix.
 * Conv2dRule used to solve the adjoint `padding` from the primal's extents at
 * TRANSFORM time; with symbolic dims that solve is arithmetic on -1s, and a
 * stride-1 padding-1 conv came out `padding=[[-3,1],[-3,1]]` where `[[1,1],[1,1]]`
 * is correct. Nothing downstream objected — the interpreter, the emitter and the
 * host twins all honour whatever attrs they are handed — so the gradient was
 * silently wrong rather than loudly absent.
 *
 * The rule now emits [OpKind.CONV2D_DATA_ADJOINT] / [OpKind.CONV2D_KERNEL_ADJOINT],
 * which carry the target tensor as a shape-only template operand and solve at
 * execution time. What this test asserts is the invariant that makes them safe:
 * every attr on an adjoint node is a literal copied off the primal, and no attr —
 * and no type — anywhere in the body encodes a solved extent.
 */
class ConvAdjointSentinelSafetyTest {

    private val sym = DxirType(F32, listOf(-1, -1, -1, -1))
    private val scalar = DxirType(F32, emptyList())

    private val primalAttrs = mapOf(
        "window_strides" to listOf(2, 1),
        "padding" to listOf(listOf(1, 0), listOf(1, 1)),
        "rhs_dilation" to listOf(1, 2),
    )

    /** The general-attr conv loss, with every extent symbolic (as `grad {}` sees it). */
    private fun sentinelConvLossFn(): DxirFunction =
        DxirBuilder.function("conv_loss_symbolic") {
            val x = param("x", sym)
            val w = param("w", sym)
            val y = op(OpKind.CONV2D, listOf(x, w), sym, attrs = primalAttrs)
            val y2 = op(OpKind.MUL, listOf(y, y), sym)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }

    @Test
    fun adjointsCarryOnlyLiteralPrimalAttrsUnderSentinelDims() {
        val grads = DxirReverseTransform.apply(sentinelConvLossFn())
        val adjoints = grads.body.filterIsInstance<DxirOp>().filter {
            it.op == OpKind.CONV2D_DATA_ADJOINT || it.op == OpKind.CONV2D_KERNEL_ADJOINT
        }
        assertEquals(
            2, adjoints.size,
            "expected exactly one data and one kernel adjoint; body was:\n${grads.pretty()}",
        )
        for (adj in adjoints) {
            // The attrs are the primal's own literals — byte-identical, so nothing
            // in them can have been computed from a -1 extent.
            assertEquals(
                primalAttrs, adj.attrs,
                "${adj.op} must carry only the primal's literal attrs; got ${adj.attrs}",
            )
            // Three operands: the two conv operands plus the shape template.
            assertEquals(3, adj.operands.size, "${adj.op} needs a shape template operand")
            // The result type IS the template's type, and both are the symbolic
            // param type — i.e. no solved extent leaked into a type either.
            assertContentEquals(
                sym.dims.toIntArray(), adj.type.dims.toIntArray(),
                "${adj.op} result type must be the template's",
            )
            assertContentEquals(
                adj.operands[2].type.dims.toIntArray(), adj.type.dims.toIntArray(),
                "${adj.op} result type must equal its template's",
            )
        }
    }

    @Test
    fun adjointTemplatesAreTheTensorsTheyDifferentiate() {
        val grads = DxirReverseTransform.apply(sentinelConvLossFn())
        val x = grads.params[0]
        val w = grads.params[1]
        val data = grads.body.filterIsInstance<DxirOp>()
            .single { it.op == OpKind.CONV2D_DATA_ADJOINT }
        val kernel = grads.body.filterIsInstance<DxirOp>()
            .single { it.op == OpKind.CONV2D_KERNEL_ADJOINT }
        // dX: (upstream, kernel, xTemplate) — the template is the primal INPUT.
        assertEquals(x.id, data.operands[2].id, "dX's template must be x:\n${grads.pretty()}")
        assertEquals(w.id, data.operands[1].id, "dX convolves against the kernel")
        // dW: (x, upstream, wTemplate) — the template is the primal KERNEL.
        assertEquals(w.id, kernel.operands[2].id, "dW's template must be w:\n${grads.pretty()}")
        assertEquals(x.id, kernel.operands[0].id, "dW convolves the primal input")
    }

    @Test
    fun noSolvedPaddingOrRank4TransposesRemainInTheBody() {
        val grads = DxirReverseTransform.apply(sentinelConvLossFn())
        val kinds = grads.body.filterIsInstance<DxirOp>().map { it.op }
        // The pre-fusion spelling emitted a lhs-dilated CONV_TRANSPOSE2D for dX and
        // TRANSPOSE+CONV2D+TRANSPOSE for dW; none of those may reappear, because
        // their padding would have to be solved from extents.
        assertTrue(
            OpKind.CONV_TRANSPOSE2D !in kinds,
            "the adjoint must not fall back to a baked-padding transposed conv:\n${grads.pretty()}",
        )
        assertTrue(
            kinds.none { it == OpKind.TRANSPOSE },
            "the kernel adjoint fuses its batch↔feature transposes:\n${grads.pretty()}",
        )
        // And no node in the body may carry a padding attr with a value that is not
        // one of the primal's literals.
        for (n in grads.body.filterIsInstance<DxirOp>()) {
            val pad = n.attrs["padding"] ?: continue
            assertEquals(
                primalAttrs.getValue("padding"), pad,
                "${n.op} carries a padding attr that is not the primal's literal: $pad",
            )
        }
    }
}
