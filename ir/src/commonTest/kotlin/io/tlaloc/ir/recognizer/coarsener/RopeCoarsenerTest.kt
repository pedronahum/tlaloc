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
        // §0.4.292 — pre-fix this test counted total NEG ops in the gradient
        // body (1 in SUB, 0 in ADD). Post-fix the d_theta arm contributes its
        // own NEG ops (αSin's base `-sin·xR` term, plus a sign flip per recombine
        // form), so the total NEG count is no longer a useful invariant. Pin
        // the actual semantic instead: d_x_imag is wrapped in NEG iff SUB-form
        // (cosMul-first), and d_theta is now an ADD-of-two-MULs structure
        // (no longer a const-zero shortcut).
        val coSub = coarsenRecognizedPatterns(buildRopeFnSub(), recognizeRope(buildRopeFnSub()))
            .body.filterIsInstance<DxirOp>().single()
        val gradSub = coSub.attrs["gradient_body"] as DxirFunction
        val dXImagSub = gradSub.returns[1]
        assertTrue(
            dXImagSub is DxirOp && dXImagSub.op == OpKind.NEG,
            "SUB-form (cosMul-first) d_x_imag must be wrapped in a NEG; got $dXImagSub",
        )

        val coAdd = coarsenRecognizedPatterns(buildRopeFnAdd(), recognizeRope(buildRopeFnAdd()))
            .body.filterIsInstance<DxirOp>().single()
        val gradAdd = coAdd.attrs["gradient_body"] as DxirFunction
        val dXImagAdd = gradAdd.returns[1]
        assertTrue(
            !(dXImagAdd is DxirOp && dXImagAdd.op == OpKind.NEG),
            "ADD-form d_x_imag must not be a NEG (it is dy·sin directly); got $dXImagAdd",
        )

        // d_theta is now the analytical ADD-of-two-MULs structure: ADD(signedT1, signedT2)
        // where signedT1 covers the `-sin(θ)·xR` chain-rule contribution and signedT2
        // covers the `cos(θ)·xI` contribution. Both forms must produce a non-const,
        // non-zero gradient body return.
        for (grad in listOf(gradSub, gradAdd)) {
            val dThetaRet = grad.returns[2]
            assertTrue(
                dThetaRet is DxirOp && dThetaRet.op == OpKind.ADD,
                "d_theta should combine the cos- and sin-chain-rule arms via ADD; got $dThetaRet",
            )
        }
    }

    @Test
    fun readsPrimalIndicesIncludesAllThree() {
        // §0.4.292 — pre-fix d_theta was a const(0), so x_real / x_imag flowed
        // into the gradient only via dy (their values were never dereferenced
        // inside the gradient body). Post-fix d_theta = dy · (-sin(θ)·xR +
        // cos(θ)·xI) explicitly multiplies x_real and x_imag, so all three
        // primal operand subgraphs now need to live in the gradient function
        // for ref-integrity. computeGradientReads picks up the dependency.
        val fn = buildRopeFnSub()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRope(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        assertEquals(
            setOf(0, 1, 2), reads,
            "all three operands (x_real, x_imag, theta) are dereferenced by d_theta + d_x_*'s recompute",
        )
    }
}
