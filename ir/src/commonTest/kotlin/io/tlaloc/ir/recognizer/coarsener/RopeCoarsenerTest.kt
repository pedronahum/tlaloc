package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeRope
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.265 — RoPE analytical-backward coarsener tests.
 *
 * Structural-only verification (op count, body signatures,
 * readsPrimalIndices). Numerical correctness of the analytical VJP is
 * out of v1 scope — the gradient body's role here is to give downstream
 * lowering a single op to substitute. Mirrors `RmsNormCoarsenerTest`.
 */
class RopeCoarsenerTest {

    private val xType = DxirType(F32, listOf(8, 64))
    private val thetaType = DxirType(F32, listOf(8, 64))

    /**
     * Canonical SUB form: `out = (xReal · cos) − (xImag · sin)`. Mirrors
     * `RmsNormRopeCrossEntropyTest.ropePositiveMatch`.
     */
    private fun buildRopeFnSub(): DxirFunction = DxirBuilder.function("rope_sub") {
        val theta = param("theta", thetaType)
        val xReal = param("x_real", xType)
        val xImag = param("x_imag", xType)
        val cosT = op(OpKind.COS, listOf(theta), xType)
        val sinT = op(OpKind.SIN, listOf(theta), xType)
        val a = op(OpKind.MUL, listOf(xReal, cosT), xType)
        val b = op(OpKind.MUL, listOf(xImag, sinT), xType)
        val out = op(OpKind.SUB, listOf(a, b), xType)
        listOf(out)
    }

    /** Same shape as [buildRopeFnSub] but the recombination is ADD. */
    private fun buildRopeFnAdd(): DxirFunction = DxirBuilder.function("rope_add") {
        val theta = param("theta", thetaType)
        val xReal = param("x_real", xType)
        val xImag = param("x_imag", xType)
        val cosT = op(OpKind.COS, listOf(theta), xType)
        val sinT = op(OpKind.SIN, listOf(theta), xType)
        val a = op(OpKind.MUL, listOf(xReal, cosT), xType)
        val b = op(OpKind.MUL, listOf(xImag, sinT), xType)
        val out = op(OpKind.ADD, listOf(a, b), xType)
        listOf(out)
    }

    @Test
    fun coarsensCanonicalRopeSubFormIntoOneCoarsenedOp() {
        val fn = buildRopeFnSub()
        val matches = recognizeRope(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(3, co.operands.size, "COARSENED takes (x_real, x_imag, theta)")
        assertEquals(xType, co.types.single())
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsensRopeAddFormIntoOneCoarsenedOp() {
        val fn = buildRopeFnAdd()
        val matches = recognizeRope(fn)
        assertEquals(1, matches.size)

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size)
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(3, co.operands.size)
        assertEquals(xType, co.types.single())
    }

    @Test
    fun coarsenedOpHasRopePrimalAndGradientBodies() {
        val fn = buildRopeFnSub()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRope(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as DxirFunction
        assertEquals("rope_primal", primal.name)
        assertEquals(3, primal.params.size, "primal takes (x_real, x_imag, theta)")
        assertEquals(xType, primal.params[0].type)
        assertEquals(xType, primal.params[1].type)
        assertEquals(thetaType, primal.params[2].type)
        assertEquals(1, primal.returns.size)
        assertEquals(xType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as DxirFunction
        assertEquals("rope_grad", grad.name)
        // K=1 (single-result COARSENED) + N=3 (x_real, x_imag, theta) = 4 params.
        assertEquals(4, grad.params.size)
        assertEquals(xType, grad.params[0].type, "param[0] is upstream dy")
        assertEquals(xType, grad.params[1].type, "param[1] is x_real")
        assertEquals(xType, grad.params[2].type, "param[2] is x_imag")
        assertEquals(thetaType, grad.params[3].type, "param[3] is theta")
        assertEquals(3, grad.returns.size, "returns (d_x_real, d_x_imag, d_theta)")
        assertEquals(xType, grad.returns[0].type)
        assertEquals(xType, grad.returns[1].type)
        assertEquals(thetaType, grad.returns[2].type)
    }

    @Test
    fun gradientBodyForSubFormNegatesImagBranch() {
        // SUB-form (cosMul-first) ⇒ d_x_imag = NEG(dy · sin). ADD-form
        // ⇒ no NEG (both branches are bare MULs). Checking the NEG-count
        // pins the sign-handling logic without requiring numerical eval.
        val coSub = coarsenRecognizedPatterns(buildRopeFnSub(), recognizeRope(buildRopeFnSub()))
            .body.filterIsInstance<DxirOp>().single()
        val gradSub = coSub.attrs["gradient_body"] as DxirFunction
        val negCountSub = gradSub.body.count { it is DxirOp && it.op == OpKind.NEG }
        assertEquals(1, negCountSub, "SUB-form gradient must wrap d_x_imag in a NEG")

        val coAdd = coarsenRecognizedPatterns(buildRopeFnAdd(), recognizeRope(buildRopeFnAdd()))
            .body.filterIsInstance<DxirOp>().single()
        val gradAdd = coAdd.attrs["gradient_body"] as DxirFunction
        val negCountAdd = gradAdd.body.count { it is DxirOp && it.op == OpKind.NEG }
        assertEquals(0, negCountAdd, "ADD-form gradient has no NEG ops")

        // Last return (d_theta) is a const-zero in both forms.
        for (grad in listOf(gradSub, gradAdd)) {
            val dThetaRet = grad.returns[2]
            assertTrue(
                dThetaRet is io.tlaloc.ir.DxirConst,
                "d_theta is a const (theta treated as fixed positional embedding)",
            )
            assertEquals(0.0f, (dThetaRet as io.tlaloc.ir.DxirConst).value)
        }
    }

    @Test
    fun readsPrimalIndicesIncludesThetaOnly() {
        val fn = buildRopeFnSub()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRope(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        // Gradient body recomputes cos/sin from theta (param idx 2 in primal).
        // x_real (idx 0) and x_imag (idx 1) flow into the gradient only via
        // the upstream dy, so they don't appear in reads.
        assertEquals(setOf(2), reads, "only theta is dereferenced; x_real/x_imag flow via dy")
    }
}
