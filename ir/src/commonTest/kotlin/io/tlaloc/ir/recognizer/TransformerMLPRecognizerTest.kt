package io.tlaloc.ir.recognizer

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.314 — TransformerMLP recognizer tests.
 *
 * 1 positive match (canonical Llama MLP) + 1 commutativity check (down
 * MATMUL with reversed operand order) + 4 adversarial cases pinning the
 * three new diagnostics this recognizer adds beyond the SwiGLU base.
 */
class TransformerMLPRecognizerTest {

    private val xType = DxirType(F32, listOf(8, 64))
    private val wProjType = DxirType(F32, listOf(64, 256))
    private val wDownType = DxirType(F32, listOf(256, 64))
    private val projType = DxirType(F32, listOf(8, 256))
    private val outType = DxirType(F32, listOf(8, 64))

    @Test
    fun transformerMLPPositiveMatch() {
        val fn = DxirBuilder.function("transformer_mlp") {
            val x = param("x", xType)
            val wGate = param("w_gate", wProjType)
            val wUp = param("w_up", wProjType)
            val wDown = param("w_down", wDownType)
            val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val siluUp = op(OpKind.MUL, listOf(gateAct, upProj), projType)
            val out = op(OpKind.MATMUL, listOf(siluUp, wDown), outType)
            listOf(out)
        }
        val matches = recognizeTransformerMLP(fn)
        assertEquals(1, matches.size, "expected 1 TransformerMLP match")
        val m = matches.single()
        assertEquals(5, m.ops.size, "MATMUL+MATMUL+SILU+MUL+MATMUL = 5 ops")
        assertEquals("x", (m.xInput as io.tlaloc.ir.DxirParam).name)
        assertEquals("w_gate", (m.wGate as io.tlaloc.ir.DxirParam).name)
        assertEquals("w_up", (m.wUp as io.tlaloc.ir.DxirParam).name)
        assertEquals("w_down", (m.wDown as io.tlaloc.ir.DxirParam).name)
        assertEquals(outType, m.outputType)
    }

    @Test
    fun transformerMLPMatchesDownMatmulWithReversedOperandOrder() {
        // out = MATMUL(W_down, silu_up) — non-Llama, but recognizer must
        // still match. Coarsener will preserve the operand order.
        val fn = DxirBuilder.function("transformer_mlp_reversed_down") {
            val x = param("x", xType)
            val wGate = param("w_gate", wProjType)
            val wUp = param("w_up", wProjType)
            // For reversed-down matmul: MATMUL(W_down_T, silu_up) — W_down's
            // contracting dim becomes the right operand. Use a shape where
            // both orderings are valid.
            val wDownRev = param("w_down_rev", DxirType(F32, listOf(64, 256)))
            val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val siluUp = op(OpKind.MUL, listOf(gateAct, upProj), projType)
            val out = op(OpKind.MATMUL, listOf(wDownRev, siluUp), DxirType(F32, listOf(64, 256)))
            listOf(out)
        }
        val matches = recognizeTransformerMLP(fn)
        assertEquals(1, matches.size)
    }

    @Test
    fun transformerMLPAdversarial1GatingMulHasMultipleConsumers() {
        // silu_up feeds both the down-proj and a residual ADD — declines
        // because absorbing silu_up would lose the residual branch.
        val fn = DxirBuilder.function("multi_consumer_silu_up") {
            val x = param("x", xType)
            val wGate = param("w_gate", wProjType)
            val wUp = param("w_up", wProjType)
            val wDown = param("w_down", wDownType)
            val skip = param("skip", projType)
            val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val siluUp = op(OpKind.MUL, listOf(gateAct, upProj), projType)
            val downOut = op(OpKind.MATMUL, listOf(siluUp, wDown), outType)
            val skipBranch = op(OpKind.ADD, listOf(siluUp, skip), projType)
            // Return both so neither dead-code-eliminates.
            listOf(downOut, skipBranch)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeTransformerMLP(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue(
            "2 consumers" in diag.single().reason,
            "diagnostic should report multi-consumer silu_up; got '${diag.single().reason}'",
        )
    }

    @Test
    fun transformerMLPAdversarial2GatingMulConsumerIsNotMatmul() {
        // silu_up feeds an ADD, not a MATMUL — bare SwiGLU shape, no
        // down-projection envelope.
        val fn = DxirBuilder.function("siluup_add") {
            val x = param("x", xType)
            val wGate = param("w_gate", wProjType)
            val wUp = param("w_up", wProjType)
            val skip = param("skip", projType)
            val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val siluUp = op(OpKind.MUL, listOf(gateAct, upProj), projType)
            val out = op(OpKind.ADD, listOf(siluUp, skip), projType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeTransformerMLP(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue(
            "consumer is ADD" in diag.single().reason,
            "diagnostic should name the wrong consumer kind; got '${diag.single().reason}'",
        )
    }

    @Test
    fun transformerMLPAdversarial3SiluOperandIsNotMatmul() {
        // Inherited from the SwiGLU-half: SILU on a bare param.
        val fn = DxirBuilder.function("silu_on_param") {
            val x = param("x", xType)
            val act = op(OpKind.SILU, listOf(x), xType)
            listOf(act)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeTransformerMLP(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("expected MATMUL(x, W_gate)" in diag.single().reason)
    }

    @Test
    fun transformerMLPAdversarial4BareSwiGLUWithNoDownProj() {
        // Canonical SwiGLU returned directly — no down-proj. SwiGLU
        // recognizer matches; TransformerMLP doesn't (gating MUL has zero
        // consumers, since it's a function return).
        val fn = DxirBuilder.function("bare_swiglu") {
            val x = param("x", xType)
            val wGate = param("w_gate", wProjType)
            val wUp = param("w_up", wProjType)
            val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val out = op(OpKind.MUL, listOf(gateAct, upProj), projType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeTransformerMLP(fn, diag)
        assertEquals(0, matches.size, "TransformerMLP doesn't match bare SwiGLU (no down-proj)")
        assertEquals(1, diag.size)
        assertTrue(
            "0 consumers" in diag.single().reason,
            "diagnostic should report the gating MUL has no consumers; got '${diag.single().reason}'",
        )
        // SwiGLU recognizer still matches the same function — exactly the
        // overlap-resolution scenario resolveLargestMatch handles.
        val swiGLUMatches = recognizeSwiGLU(fn)
        assertEquals(1, swiGLUMatches.size, "bare SwiGLU still matched by recognizeSwiGLU")
    }
}
