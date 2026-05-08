package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.ir.recognizer.recognizeSwiGLU
import io.tlaloc.ir.recognizer.recognizeTransformerMLP
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.314 — TransformerMLP analytical-backward coarsener tests +
 * the v2 compound-vs-primitive overlap-resolution test.
 *
 * Structural verification only (op count, body signatures,
 * readsPrimalIndices, anchor placement). Numerical correctness lands
 * later in Phase-4 FD-validation territory; the SwiGLU coarsener
 * already validates the inner math so this test focuses on the
 * down-proj envelope.
 */
class TransformerMLPCoarsenerTest {

    private val xType = DxirType(F32, listOf(8, 64))
    private val wProjType = DxirType(F32, listOf(64, 256))
    private val wDownType = DxirType(F32, listOf(256, 64))
    private val projType = DxirType(F32, listOf(8, 256))
    private val outType = DxirType(F32, listOf(8, 64))

    private fun buildTransformerMLPFn(): DxirFunction = DxirBuilder.function("transformer_mlp") {
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

    @Test
    fun coarsensTransformerMLPIntoOneCoarsenedOp() {
        val fn = buildTransformerMLPFn()
        val matches = recognizeTransformerMLP(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(4, co.operands.size, "COARSENED takes (x, w_gate, w_up, w_down)")
        assertEquals(outType, co.types.single())
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsenedOpHasTransformerMLPPrimalAndGradientBodies() {
        val fn = buildTransformerMLPFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeTransformerMLP(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as DxirFunction
        assertEquals("transformer_mlp_primal", primal.name)
        assertEquals(4, primal.params.size, "primal takes (x, w_gate, w_up, w_down)")
        assertEquals(xType, primal.params[0].type)
        assertEquals(wProjType, primal.params[1].type)
        assertEquals(wProjType, primal.params[2].type)
        assertEquals(wDownType, primal.params[3].type)
        assertEquals(1, primal.returns.size)
        assertEquals(outType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as DxirFunction
        assertEquals("transformer_mlp_grad", grad.name)
        // K=1 (single-result COARSENED) + N=4 = 5 params.
        assertEquals(5, grad.params.size)
        assertEquals(outType, grad.params[0].type, "param[0] is upstream dy")
        assertEquals(xType, grad.params[1].type)
        assertEquals(wProjType, grad.params[2].type)
        assertEquals(wProjType, grad.params[3].type)
        assertEquals(wDownType, grad.params[4].type)
        assertEquals(4, grad.returns.size, "returns (d_x, d_w_gate, d_w_up, d_w_down)")
        assertEquals(xType, grad.returns[0].type)
        assertEquals(wProjType, grad.returns[1].type)
        assertEquals(wProjType, grad.returns[2].type)
        assertEquals(wDownType, grad.returns[3].type)
    }

    @Test
    fun gradientBodyHasEightMatmuls() {
        // SwiGLU gradient had 6 matmuls (2 recompute + 2 d_x + 2 d_W).
        // TransformerMLP adds one recompute (silu_up itself isn't a matmul,
        // but we don't need to recompute it explicitly — the SwiGLU body
        // does that), and 2 down-proj VJPs (d_silu_up + d_w_down). So:
        // 2 recompute (gate, up) + 2 down-proj VJPs (d_silu_up, d_w_down)
        // + 2 d_x VJPs (d_x_gate, d_x_up) + 2 d_W VJPs (d_W_gate, d_W_up)
        // = 8.
        val fn = buildTransformerMLPFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeTransformerMLP(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction
        val matmulCount = grad.body.count { it is DxirOp && it.op == OpKind.MATMUL }
        assertEquals(8, matmulCount, "2 recompute + 2 down-proj + 2 d_x + 2 d_W matmuls")
    }

    @Test
    fun gradientBodyContainsSigmoidForSiluDerivative() {
        // SILU'(z) = sig + silu − silu · sig — pin the SIGMOID op as in the
        // SwiGLU coarsener test. Catches regressions where the SILU
        // derivative collapses to "dy * 1".
        val fn = buildTransformerMLPFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeTransformerMLP(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction
        val sigmoidCount = grad.body.count { it is DxirOp && it.op == OpKind.SIGMOID }
        assertEquals(1, sigmoidCount, "exactly one SIGMOID (the SILU' building block)")
    }

    @Test
    fun readsPrimalIndicesIncludesAllFourOperands() {
        val fn = buildTransformerMLPFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeTransformerMLP(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        // x, W_gate, W_up needed for the SwiGLU half (recompute matmuls +
        // matmul-VJP transposes). W_down needed for the down-proj VJP
        // (W_down^T in d_silu_up + silu_up itself depends on W_down via
        // the recompute). All four primal operands are dereferenced.
        assertEquals(setOf(0, 1, 2, 3), reads, "all four primal operands dereferenced")
    }

    @Test
    fun resolveLargestMatchPicksTransformerMLPOverBareSwiGLU() {
        // The v2 compound-vs-primitive overlap test: with both recognizers
        // active over the same SILU anchor, recognizeAll's
        // resolveLargestMatch must pick TransformerMLP (5 ops) and discard
        // the bare SwiGLU (4 ops) that the SwiGLU recognizer also produced.
        // Pre-§0.4.314 the resolver had no production overlaps to mediate;
        // this is the test that lights it up on real Llama-shaped code.
        val fn = buildTransformerMLPFn()

        val rawSwiGLU = recognizeSwiGLU(fn)
        val rawCompound = recognizeTransformerMLP(fn)
        assertEquals(1, rawSwiGLU.size, "SwiGLU recognizer matched the inner shape")
        assertEquals(1, rawCompound.size, "TransformerMLP recognizer matched the full shape")

        val resolved = recognizeAll(fn)
        assertEquals(1, resolved.size, "after resolveLargestMatch only one match survives")
        val winner = resolved.single()
        assertTrue(
            winner is io.tlaloc.ir.recognizer.RecognitionMatch.TransformerMLP,
            "the larger match should win; got ${winner::class.simpleName}",
        )
        assertEquals(5, winner.ops.size, "winning match owns all 5 ops")
    }

    @Test
    fun coarsenRecognizedPatternsViaRecognizeAllProducesSingleCoarsened() {
        // End-to-end: feed the function through recognizeAll +
        // coarsenRecognizedPatterns and confirm exactly one COARSENED op
        // results, attributed to the TransformerMLP coarsener (not the
        // SwiGLU coarsener). The primal body name is the tell.
        val fn = buildTransformerMLPFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeAll(fn))
        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "single COARSENED, not two (would mean both coarseners ran)")
        val primal = ops.single().attrs["primal_body"] as DxirFunction
        assertEquals(
            "transformer_mlp_primal",
            primal.name,
            "primal body should come from coarsenTransformerMLP, not coarsenSwiGLU",
        )
    }
}
