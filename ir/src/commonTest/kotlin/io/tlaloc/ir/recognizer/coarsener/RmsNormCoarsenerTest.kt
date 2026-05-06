package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeRmsNorm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.264 — RMS-norm analytical-backward coarsener tests.
 *
 * Structural-only verification (op count, body signatures, readsPrimalIndices).
 * Numerical correctness of the analytical VJP is out of v1 scope; the gradient
 * body's role here is to give downstream lowering a single op to substitute.
 */
class RmsNormCoarsenerTest {

    private val xType = DxirType(F32, listOf(8, 64))                // [..., D]
    private val mType = DxirType(F32, listOf(8, 1))                 // mean keepdims
    private val epsType = DxirType(F32, listOf(8, 1))               // matches mean's shape

    private fun buildRmsNormFn(): DxirFunction = DxirBuilder.function("rms_norm") {
        val x = param("x", xType)
        val sq = op(OpKind.MUL, listOf(x, x), xType)
        val mean = op(OpKind.MEAN, listOf(sq), mType)
        val r = op(OpKind.RSQRT, listOf(mean), mType)
        val out = op(OpKind.MUL, listOf(x, r), xType)
        listOf(out)
    }

    private fun buildRmsNormFnWithEps(): DxirFunction = DxirBuilder.function("rms_norm_eps") {
        val x = param("x", xType)
        val eps = const(1e-6f, epsType)
        val sq = op(OpKind.MUL, listOf(x, x), xType)
        val mean = op(OpKind.MEAN, listOf(sq), mType)
        val stabilised = op(OpKind.ADD, listOf(mean, eps), mType)
        val r = op(OpKind.RSQRT, listOf(stabilised), mType)
        val out = op(OpKind.MUL, listOf(x, r), xType)
        listOf(out)
    }

    @Test
    fun coarsensCanonicalRmsNormGraphIntoOneCoarsenedOp() {
        val fn = buildRmsNormFn()
        val matches = recognizeRmsNorm(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        // Body: only the COARSENED remains; sqMul / mean / rsqrt / finalMul absorbed.
        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(1, co.operands.size, "no-eps: COARSENED takes (x)")
        assertEquals(xType, co.types.single())
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsensRmsNormWithEpsSurfacesEpsAsOuterOperand() {
        val fn = buildRmsNormFnWithEps()
        val matches = recognizeRmsNorm(fn)
        assertEquals(1, matches.size)

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        // Body: the COARSENED + the eps DxirConst (which lives outside the
        // absorbed set as an outer operand).
        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one COARSENED op; got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(2, co.operands.size, "with-eps: COARSENED takes (x, eps)")
        assertEquals(xType, co.types.single())
    }

    @Test
    fun coarsenedOpHasRmsNormPrimalAndGradientBodies() {
        val fn = buildRmsNormFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRmsNorm(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as DxirFunction
        assertEquals("rms_norm_primal", primal.name)
        assertEquals(1, primal.params.size, "no-eps: primal takes (x)")
        assertEquals(xType, primal.params[0].type)
        assertEquals(1, primal.returns.size)
        assertEquals(xType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as DxirFunction
        assertEquals("rms_norm_grad", grad.name)
        // K=1 (single-result COARSENED) + N=1 (x operand) = 2 params.
        assertEquals(2, grad.params.size)
        assertEquals(xType, grad.params[0].type, "param[0] is upstream dy")
        assertEquals(xType, grad.params[1].type, "param[1] is x")
        assertEquals(1, grad.returns.size, "returns (dx)")
        assertEquals(xType, grad.returns[0].type)
    }

    @Test
    fun gradientBodyForEpsVariantEmitsAnalyticalDeps() {
        // §0.4.292 — the prior "deps = const(0)" shortcut produced disagreement
        // with PyTorch's torch.autograd.grad. Coarsener now emits the analytical
        //   d_eps = -0.5 · r³ · SUM(dy · x, last-axis, keep-dims)
        // which structurally is a MUL of (-0.5 broadcast) and (r³ · sum_dyx).
        val fn = buildRmsNormFnWithEps()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRmsNorm(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction

        // With eps: signature is (dy, x, eps) → (dx, d_eps).
        assertEquals(3, grad.params.size, "(dy, x, eps)")
        assertEquals(xType, grad.params[0].type)
        assertEquals(xType, grad.params[1].type)
        assertEquals(epsType, grad.params[2].type)
        assertEquals(2, grad.returns.size, "(dx, d_eps)")
        assertEquals(xType, grad.returns[0].type)
        assertEquals(epsType, grad.returns[1].type)

        // Second return is now a MUL (the outermost -0.5 · (r³ · sum_dyx)), not a const.
        val depsRet = grad.returns[1]
        assertTrue(
            depsRet is DxirOp && depsRet.op == OpKind.MUL,
            "d_eps should be the MUL closing the analytical chain, not a const-zero shortcut; got $depsRet",
        )
        assertEquals(epsType, depsRet.type)

        // Pin the analytical structure: the gradient body should contain a SUM
        // (over the dy · x reduction) in addition to the existing MEAN(s).
        val sums = grad.body.filterIsInstance<DxirOp>().count { it.op == OpKind.SUM }
        assertTrue(sums >= 1, "expected at least one SUM in the gradient body for d_eps's reduction")
    }

    @Test
    fun readsPrimalIndicesIncludesX() {
        val fn = buildRmsNormFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRmsNorm(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        // The gradient body recomputes r = rsqrt(mean(x²)), so it dereferences
        // the x param. param idx 0 (after stripping the upstream-dy param at idx 0).
        assertTrue(0 in reads, "gradient body recomputes r and references x (param idx 0)")
    }
}
