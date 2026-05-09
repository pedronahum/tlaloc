package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeLayerNorm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * §0.4.319 — LayerNorm analytical-backward coarsener tests. Mirrors
 * [RmsNormCoarsenerTest]'s shape: structural-only verification (op count,
 * body signatures, readsPrimalIndices). Numerical correctness of the
 * analytical VJP is out of v1 scope.
 */
class LayerNormCoarsenerTest {

    private val xType = DxirType(F32, listOf(8, 64))    // [..., D]
    private val mType = DxirType(F32, listOf(8, 1))     // mean keepdims
    private val epsType = DxirType(F32, listOf(8, 1))   // matches sqrt's shape

    private fun buildLayerNormFn(): DxirFunction = DxirBuilder.function("layer_norm") {
        val x = param("x", xType)
        val mean1 = op(OpKind.MEAN, listOf(x), mType)
        val sub = op(OpKind.SUB, listOf(x, mean1), xType)
        val sq = op(OpKind.MUL, listOf(sub, sub), xType)
        val mean2 = op(OpKind.MEAN, listOf(sq), mType)
        val s = op(OpKind.SQRT, listOf(mean2), mType)
        val out = op(OpKind.DIV, listOf(sub, s), xType)
        listOf(out)
    }

    private fun buildLayerNormFnWithEps(): DxirFunction = DxirBuilder.function("layer_norm_eps") {
        val x = param("x", xType)
        val eps = const(1e-5f, epsType)
        val mean1 = op(OpKind.MEAN, listOf(x), mType)
        val sub = op(OpKind.SUB, listOf(x, mean1), xType)
        val sq = op(OpKind.MUL, listOf(sub, sub), xType)
        val mean2 = op(OpKind.MEAN, listOf(sq), mType)
        val stabilised = op(OpKind.ADD, listOf(mean2, eps), mType)
        val s = op(OpKind.SQRT, listOf(stabilised), mType)
        val out = op(OpKind.DIV, listOf(sub, s), xType)
        listOf(out)
    }

    @Test
    fun coarsensCanonicalLayerNormGraphIntoOneCoarsenedOp() {
        val fn = buildLayerNormFn()
        val matches = recognizeLayerNorm(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        // All 6 forward ops absorbed; only the COARSENED remains.
        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(1, co.operands.size, "no-eps: COARSENED takes (x)")
        assertEquals(xType, co.types.single())
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsensLayerNormWithEpsSurfacesEpsAsOuterOperand() {
        val fn = buildLayerNormFnWithEps()
        val matches = recognizeLayerNorm(fn)
        assertEquals(1, matches.size)

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one COARSENED op; got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(2, co.operands.size, "with-eps: COARSENED takes (x, eps)")
        assertEquals(xType, co.types.single())
    }

    @Test
    fun coarsenedOpHasLayerNormPrimalAndGradientBodies() {
        val fn = buildLayerNormFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeLayerNorm(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as DxirFunction
        assertEquals("layer_norm_primal", primal.name)
        assertEquals(1, primal.params.size, "no-eps: primal takes (x)")
        assertEquals(xType, primal.params[0].type)
        assertEquals(1, primal.returns.size)
        assertEquals(xType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as DxirFunction
        assertEquals("layer_norm_grad", grad.name)
        // K=1 (single-result COARSENED) + N=1 (x operand) = 2 params.
        assertEquals(2, grad.params.size)
        assertEquals(xType, grad.params[0].type, "param[0] is upstream dy")
        assertEquals(xType, grad.params[1].type, "param[1] is x")
        assertEquals(1, grad.returns.size, "returns (dx)")
        assertEquals(xType, grad.returns[0].type)
    }

    @Test
    fun gradientBodyForEpsVariantEmitsAnalyticalDeps() {
        // Mirrors RmsNorm §0.4.292: no const-zero shortcut. d_eps must be
        // emitted as a real expression; structurally it's the closing
        // MUL of (-0.5 broadcast) and (sum(dy·c) / s³).
        val fn = buildLayerNormFnWithEps()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeLayerNorm(fn))
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

        // Second return is now a MUL (the outermost -0.5 · …), not a const.
        val depsRet = grad.returns[1]
        assertTrue(
            depsRet is DxirOp && depsRet.op == OpKind.MUL,
            "d_eps should be the closing MUL of the analytical chain, not a const-zero shortcut; got $depsRet",
        )
        assertEquals(epsType, depsRet.type)

        // Pin the analytical structure: at least one SUM (over dy · c) +
        // recognise the inner DIV by s³ pattern is present.
        val gradOps = grad.body.filterIsInstance<DxirOp>()
        val sums = gradOps.count { it.op == OpKind.SUM }
        assertTrue(sums >= 1, "expected at least one SUM in the gradient body for d_eps's reduction")
    }

    @Test
    fun readsPrimalIndicesIncludesX() {
        val fn = buildLayerNormFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeLayerNorm(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        // Gradient body recomputes mean(x), centred = x − mean, etc., so
        // x param (idx 0 after stripping the upstream-dy at idx 0) is read.
        assertTrue(0 in reads, "gradient body recomputes mean(x) and references x (param idx 0)")
    }
}
