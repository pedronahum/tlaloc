package io.tlaloc.ir.recognizer

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.267 — SwiGLU recognizer tests.
 *
 * 1 positive match (canonical operand order) + 1 commutativity check
 * (gating MUL with reversed operand order) + 3 adversarial cases.
 * Mirrors `RmsNormRopeCrossEntropyTest`'s shape for adjacent recognizers.
 */
class SwiGLURecognizerTest {

    private val xType = DxirType(F32, listOf(8, 64))
    private val wType = DxirType(F32, listOf(64, 256))
    private val projType = DxirType(F32, listOf(8, 256))

    @Test
    fun swiGLUPositiveMatch() {
        val fn = DxirBuilder.function("swiglu") {
            val x = param("x", xType)
            val wGate = param("w_gate", wType)
            val wUp = param("w_up", wType)
            val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val out = op(OpKind.MUL, listOf(gateAct, upProj), projType)
            listOf(out)
        }
        val matches = recognizeSwiGLU(fn)
        assertEquals(1, matches.size, "expected 1 SwiGLU match")
        val m = matches.single()
        assertEquals(4, m.ops.size, "MATMUL+MATMUL+SILU+MUL = 4 ops")
        assertEquals("x", (m.xInput as io.tlaloc.ir.DxirParam).name)
        assertEquals("w_gate", (m.wGate as io.tlaloc.ir.DxirParam).name)
        assertEquals("w_up", (m.wUp as io.tlaloc.ir.DxirParam).name)
        assertEquals(projType, m.outputType)
    }

    @Test
    fun swiGLUMatchesGatingMulWithReversedOperandOrder() {
        // out = MUL(up_proj, SILU(gate_proj)) — same activation, gating MUL
        // operands flipped. Recognizer must still match (MUL is commutative
        // for the recognition's purposes).
        val fn = DxirBuilder.function("swiglu_reversed") {
            val x = param("x", xType)
            val wGate = param("w_gate", wType)
            val wUp = param("w_up", wType)
            val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val out = op(OpKind.MUL, listOf(upProj, gateAct), projType)
            listOf(out)
        }
        val matches = recognizeSwiGLU(fn)
        assertEquals(1, matches.size)
    }

    @Test
    fun swiGLUAdversarial1SiluOperandNotMatmul() {
        // SILU on a bare param, no preceding MATMUL — likely a generic
        // activation, not a gate.
        val fn = DxirBuilder.function("silu_on_param") {
            val x = param("x", xType)
            val act = op(OpKind.SILU, listOf(x), xType)
            listOf(act)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeSwiGLU(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("expected MATMUL" in diag.single().reason)
    }

    @Test
    fun swiGLUAdversarial2SiluHasNoMulConsumer() {
        // SILU(MATMUL(x, W)) returned directly — gate computed but never
        // gated. (No MUL consumer.)
        val fn = DxirBuilder.function("silu_matmul_only") {
            val x = param("x", xType)
            val w = param("w", wType)
            val proj = op(OpKind.MATMUL, listOf(x, w), projType)
            val act = op(OpKind.SILU, listOf(proj), projType)
            listOf(act)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeSwiGLU(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("no MUL consumer" in diag.single().reason)
    }

    @Test
    fun swiGLUAdversarial3GateAndUpDontShareInput() {
        // SILU(MATMUL(x1, W_gate)) gated against MATMUL(x2, W_up) — two
        // independent inputs, not a co-rooted gate.
        val fn = DxirBuilder.function("two_inputs") {
            val x1 = param("x1", xType)
            val x2 = param("x2", xType)
            val wGate = param("w_gate", wType)
            val wUp = param("w_up", wType)
            val gateProj = op(OpKind.MATMUL, listOf(x1, wGate), projType)
            val upProj = op(OpKind.MATMUL, listOf(x2, wUp), projType)
            val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
            val out = op(OpKind.MUL, listOf(gateAct, upProj), projType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeSwiGLU(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("share 0 operand" in diag.single().reason)
    }
}
