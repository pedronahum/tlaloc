package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeSwiGLU
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Layer 4 §0.4.268 — SwiGLU analytical-backward coarsener tests.
 *
 * Structural-only verification (op count, body signatures,
 * readsPrimalIndices). Numerical correctness of the analytical VJP is
 * out of v1 scope. Mirrors `RmsNormCoarsenerTest` /
 * `CrossEntropyCoarsenerTest` shape.
 */
class SwiGLUCoarsenerTest {

    private val xType = DxirType(F32, listOf(8, 64))
    private val wType = DxirType(F32, listOf(64, 256))
    private val projType = DxirType(F32, listOf(8, 256))

    private fun buildSwiGLUFn(): DxirFunction = DxirBuilder.function("swiglu") {
        val x = param("x", xType)
        val wGate = param("w_gate", wType)
        val wUp = param("w_up", wType)
        val gateProj = op(OpKind.MATMUL, listOf(x, wGate), projType)
        val upProj = op(OpKind.MATMUL, listOf(x, wUp), projType)
        val gateAct = op(OpKind.SILU, listOf(gateProj), projType)
        val out = op(OpKind.MUL, listOf(gateAct, upProj), projType)
        listOf(out)
    }

    @Test
    fun coarsensSwiGLUIntoOneCoarsenedOp() {
        val fn = buildSwiGLUFn()
        val matches = recognizeSwiGLU(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(3, co.operands.size, "COARSENED takes (x, w_gate, w_up)")
        assertEquals(projType, co.types.single())
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsenedOpHasSwiGLUPrimalAndGradientBodies() {
        val fn = buildSwiGLUFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeSwiGLU(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as DxirFunction
        assertEquals("swiglu_primal", primal.name)
        assertEquals(3, primal.params.size, "primal takes (x, w_gate, w_up)")
        assertEquals(xType, primal.params[0].type)
        assertEquals(wType, primal.params[1].type)
        assertEquals(wType, primal.params[2].type)
        assertEquals(1, primal.returns.size)
        assertEquals(projType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as DxirFunction
        assertEquals("swiglu_grad", grad.name)
        // K=1 (single-result COARSENED) + N=3 = 4 params.
        assertEquals(4, grad.params.size)
        assertEquals(projType, grad.params[0].type, "param[0] is upstream dy")
        assertEquals(xType, grad.params[1].type, "param[1] is x")
        assertEquals(wType, grad.params[2].type, "param[2] is w_gate")
        assertEquals(wType, grad.params[3].type, "param[3] is w_up")
        assertEquals(3, grad.returns.size, "returns (d_x, d_w_gate, d_w_up)")
        assertEquals(xType, grad.returns[0].type)
        assertEquals(wType, grad.returns[1].type)
        assertEquals(wType, grad.returns[2].type)
    }

    @Test
    fun gradientBodyContainsSigmoidForSiluDerivative() {
        // SILU'(z) = sig + silu − silu · sig. Pinning a SIGMOID op in the
        // gradient body catches regressions where the derivative collapses
        // to "dy * 1" (which would happen if you forgot to multiply by
        // SILU'(gate)).
        val fn = buildSwiGLUFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeSwiGLU(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction
        val sigmoidCount = grad.body.count { it is DxirOp && it.op == OpKind.SIGMOID }
        assertEquals(1, sigmoidCount, "exactly one SIGMOID (the SILU' building block)")
    }

    @Test
    fun gradientBodyHasSixMatmuls() {
        // 2 recompute matmuls (gate, up) + 2 d_x matmuls (d_x_gate, d_x_up)
        // + 2 d_W matmuls (d_W_gate, d_W_up) = 6. Pinning the count catches
        // regressions where one of the matmul-VJP branches is dropped.
        val fn = buildSwiGLUFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeSwiGLU(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction
        val matmulCount = grad.body.count { it is DxirOp && it.op == OpKind.MATMUL }
        assertEquals(6, matmulCount, "2 recompute + 2 d_x + 2 d_W matmuls")
    }

    @Test
    fun readsPrimalIndicesIncludesAllThreeOperands() {
        val fn = buildSwiGLUFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeSwiGLU(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        // x recomputes both gate and up + provides x^T for d_W matmuls.
        // w_gate provides W_gate^T for d_x_gate. w_up provides W_up^T for d_x_up.
        assertEquals(setOf(0, 1, 2), reads, "all three primal operands dereferenced")
    }
}
