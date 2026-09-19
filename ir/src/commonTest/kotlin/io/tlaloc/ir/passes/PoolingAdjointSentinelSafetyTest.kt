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
 * §0.4.386 (avgpool) / §0.4.389 (maxpool) — the pooling adjoints are
 * SENTINEL-SAFE: under `grad {}`'s -1 dims the gradient body must contain no
 * extent-derived number and no extent-shaped node at all.
 *
 * These are the regression pins for the defects the fused ops exist to fix, and
 * each pooling kind had its own:
 *
 * - avgpool's adjoint was `RESHAPE([N,C,·,·] → [N·C,1,·,·]) → CONV_TRANSPOSE2D(splat,
 *   lhs_dilation = stride, padding solved from extents) → RESHAPE back`. Two extent
 *   dependencies, and the reshape was the worse: its target baked `n * c`, which
 *   under sentinels is **1** — a node claiming a shape the data did not have.
 * - maxpool's adjoint was nearest-upsample-and-mask through RANK-6 intermediates
 *   (`[N,C,Ho,1,Wo,1] → broadcast → [N,C,Ho,kh,Wo,kw] → [N,C,H,W]`) plus a
 *   `h % kh == 0` guard. Every one of those types bakes `n`/`c`/`Ho`/`Wo`, and the
 *   guard evaluated `-1 % 2 == -1 ≠ 0`, so the rule rejected every symbolic maxpool
 *   outright — which is why the plan had maxpool parked last behind "rank-6
 *   intermediates" and the `tensorIrType` single-representative blocker.
 *
 * Both now invert the window per input element at EXECUTION time, so the body is
 * rank-4 throughout and each adjoint's result type is simply the tensor it
 * differentiates w.r.t.
 */
class PoolingAdjointSentinelSafetyTest {

    private val sym = DxirType(F32, listOf(-1, -1, -1, -1))
    private val scalar = DxirType(F32, emptyList())

    private fun lossFn(kind: OpKind, attrs: Map<String, Any>): DxirFunction =
        DxirBuilder.function("pool_loss_symbolic") {
            val x = param("x", sym)
            val y = op(kind, listOf(x), sym, attrs = attrs)
            val y2 = op(OpKind.MUL, listOf(y, y), sym)
            listOf(op(OpKind.SUM, listOf(y2), scalar))
        }

    private fun avgPoolFn() = lossFn(
        OpKind.AVGPOOL2D,
        mapOf(
            "window" to listOf(2, 2),
            "window_strides" to listOf(2, 2),
            "padding" to listOf(listOf(0, 0), listOf(0, 0)),
        ),
    )

    // No `window_strides` attr: the rule defaults it to the window, which is the
    // non-overlapping pool its v1 scope (and the emitter's expansion) requires.
    private fun maxPoolFn() = lossFn(OpKind.MAXPOOL2D, mapOf("window" to listOf(2, 2)))

    private fun opsOf(fn: DxirFunction): List<DxirOp> = fn.body.filterIsInstance<DxirOp>()

    // ---- avgpool -----------------------------------------------------------

    @Test
    fun avgPoolAdjointCarriesOnlyLiteralPrimalAttrsUnderSentinelDims() {
        val primalAttrs = mapOf(
            "window" to listOf(2, 2),
            "window_strides" to listOf(2, 2),
            "padding" to listOf(listOf(0, 0), listOf(0, 0)),
        )
        val grads = DxirReverseTransform.apply(avgPoolFn())
        val adj = opsOf(grads).filter { it.op == OpKind.AVGPOOL2D_GRAD }
        assertEquals(1, adj.size, "expected exactly one fused adjoint:\n${grads.pretty()}")
        val node = adj.single()
        assertEquals(
            primalAttrs, node.attrs,
            "the adjoint must carry only the primal's literal attrs; got ${node.attrs}",
        )
        assertEquals(2, node.operands.size, "AVGPOOL2D_GRAD takes (upstream, xTemplate)")
        assertContentEquals(sym.dims.toIntArray(), node.type.dims.toIntArray())
        assertContentEquals(
            node.operands[1].type.dims.toIntArray(), node.type.dims.toIntArray(),
            "result type must equal the template's",
        )
    }

    @Test
    fun avgPoolBodyHasNoChannelFoldOrTransposedConv() {
        val grads = DxirReverseTransform.apply(avgPoolFn())
        assertEquals(
            grads.params[0].id,
            opsOf(grads).single { it.op == OpKind.AVGPOOL2D_GRAD }.operands[1].id,
            "the template must be the tensor being differentiated:\n${grads.pretty()}",
        )
        val kinds = opsOf(grads).map { it.op }
        assertTrue(
            OpKind.RESHAPE !in kinds,
            "the fused adjoint needs no channel fold:\n${grads.pretty()}",
        )
        assertTrue(
            OpKind.CONV_TRANSPOSE2D !in kinds,
            "the fused adjoint replaces the splat-kernel transposed conv:\n${grads.pretty()}",
        )
    }

    // ---- maxpool -----------------------------------------------------------

    @Test
    fun maxPoolAdjointCarriesOnlyLiteralPrimalAttrsUnderSentinelDims() {
        val grads = DxirReverseTransform.apply(maxPoolFn())
        val adj = opsOf(grads).filter { it.op == OpKind.MAXPOOL2D_GRAD }
        assertEquals(1, adj.size, "expected exactly one fused adjoint:\n${grads.pretty()}")
        val node = adj.single()
        assertEquals(
            mapOf(
                "window" to listOf(2, 2),
                "window_strides" to listOf(2, 2),
                "padding" to listOf(listOf(0, 0), listOf(0, 0)),
            ),
            node.attrs,
            "the adjoint must carry only the primal's literal attrs; got ${node.attrs}",
        )
        assertEquals(3, node.operands.size, "MAXPOOL2D_GRAD takes (upstream, x, y)")
        // The result is x's shape, and x is a VALUE operand here (its elements are
        // compared against the window max) — but its type is still the symbolic
        // param type, so nothing was invented.
        assertContentEquals(sym.dims.toIntArray(), node.type.dims.toIntArray())
        assertContentEquals(
            node.operands[1].type.dims.toIntArray(), node.type.dims.toIntArray(),
            "result type must equal x's",
        )
        assertEquals(
            grads.params[0].id, node.operands[1].id,
            "operand 1 must be the primal input:\n${grads.pretty()}",
        )
    }

    @Test
    fun maxPoolBodyHasNoRank6UpsampleChain() {
        val grads = DxirReverseTransform.apply(maxPoolFn())
        val kinds = opsOf(grads).map { it.op }
        // The nearest-upsample spelling's signature nodes are gone. (BROADCAST is
        // NOT banned: SumRule's scalar seed splat is a legitimate rank-4
        // `broadcast(const, template)`, and `rank ≤ 4` below is what actually
        // excludes the upsample chain — its whole point was a rank-6 stretch.)
        for (banned in listOf(OpKind.RESHAPE, OpKind.COMPARE, OpKind.WHERE)) {
            assertTrue(banned !in kinds, "$banned must not appear in the body:\n${grads.pretty()}")
        }
        // This is the pin for the "rank-6 intermediates" blocker: nothing in the
        // body exceeds rank 4, so the synthesis needs no rank-6 shape witness and
        // the body never mixes ranks (which is what the single-representative
        // `tensorIrType` could not handle).
        assertTrue(
            opsOf(grads).all { it.type.rank <= 4 },
            "every body node must be rank ≤ 4:\n${grads.pretty()}",
        )
        // The rule still materialises the pooled value for the mask — as a rank-4
        // MAXPOOL2D whose own attrs are the primal's literals.
        val recomputed = opsOf(grads).filter { it.op == OpKind.MAXPOOL2D }
        assertEquals(1, recomputed.size, "expected the recomputed pooled value:\n${grads.pretty()}")
        assertEquals(
            grads.params[0].id, recomputed.single().operands[0].id,
            "the recomputed maxpool must read the primal input",
        )
    }
}
