package io.tlaloc.plugin

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Synthesis types a templated `BROADCAST` with its template's IrType and lowers it to a
 * call that takes the template's run-time shape, so it must refuse one whose result
 * shape disagrees with the template ([BroadcastTemplate.agrees]; the GP regression
 * that motivated the typing rule is in `LinalgGradientTest`).
 */
class BroadcastTemplateTest {
    private fun t(vararg dims: Int) = DxirType(F32, dims.toList())

    private fun broadcast(operand: DxirType, template: DxirType, result: DxirType) =
        DxirOp(3, OpKind.BROADCAST, listOf(DxirParam(1, "v", operand), DxirParam(2, "tpl", template)), type = result)

    @Test
    fun `a splat or stretch shaped like its template agrees`() {
        assertTrue(BroadcastTemplate.agrees(broadcast(t(), t(-1, -1), t(-1, -1))))
        assertTrue(BroadcastTemplate.agrees(broadcast(t(-1, 1), t(-1, -1), t(-1, -1))))
        assertTrue(BroadcastTemplate.agrees(broadcast(t(), t(3, 4), t(3, 4))))
        // An unknown extent on either side cannot disagree.
        assertTrue(BroadcastTemplate.agrees(broadcast(t(), t(3, -1), t(-1, 4))))
    }

    @Test
    fun `a result of another rank or another known extent is refused`() {
        assertFalse(BroadcastTemplate.agrees(broadcast(t(), t(-1), t(-1, -1))))
        assertFalse(BroadcastTemplate.agrees(broadcast(t(), t(-1, -1), t(-1))))
        assertFalse(BroadcastTemplate.agrees(broadcast(t(), t(3, 4), t(3, 5))))
        assertFalse(BroadcastTemplate.agrees(broadcast(t(4, 1), t(4, 1), t(4, 3))))
    }

    @Test
    fun `a BROADCAST without a template does not qualify`() {
        val op = DxirOp(2, OpKind.BROADCAST, listOf(DxirParam(1, "v", t())), type = t(-1, -1))
        assertFalse(BroadcastTemplate.agrees(op))
    }
}
