package io.tlaloc.benchmarks

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.272 — Recognize + coarsen the LlamaDecoderPrimal.
 *
 * Phase 2 step 2 of the dual-track Llama-decoder benchmark plan
 * (`memory/llama_benchmark_dual_track.md`). The structural assertion
 * the rest of the plan rests on: every L4 recognizer + coarsener
 * landed in Phase 1 actually fires on the LlamaDecoderPrimal that
 * §0.4.271 ships, and the coarsener leaves the unrecognized residue
 * (regular MATMULs + residual ADDs) untouched.
 *
 * # Why the count is 6, not 5
 *
 * The original dual-track plan said "all 5 patterns" because there are
 * 5 *distinct* recognizers (RmsNorm, RoPE, FlashAttention, SwiGLU,
 * CrossEntropy). But the layer has **two** RmsNorms — one pre-attention
 * and one pre-MLP — so recognizeAll fires the RmsNorm recognizer twice.
 * Total matches: 6.
 *
 * # Direct DXIR-level — not via grad{} plugin
 *
 * The plan's "via the grad{} plugin path" framing is skipped here.
 * Direct calls to recognizeAll + coarsenRecognizedPatterns suffice for
 * the structural assertion and match the existing OOPSLA benchmark
 * convention. The grad{} plugin path is exercised at the
 * compiler-plugin module's own test surface, not here.
 */
class LlamaDecoderRecognizeCoarsenTest {

    private fun buildAndCoarsen(): Pair<DxirFunction, DxirFunction> {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val matches = recognizeAll(raw)
        val coarsened = coarsenRecognizedPatterns(raw, matches)
        return raw to coarsened
    }

    @Test
    fun recognizeAllFindsSixMatchesWithExpectedPatternMultiset() {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val matches = recognizeAll(raw)
        assertEquals(6, matches.size, "expected 6 matches; got ${matches.map { it.patternName }}")

        val patternCounts = matches.map { it.patternName }.groupingBy { it }.eachCount()
        assertEquals(2, patternCounts["RmsNorm"], "pre-attn + pre-MLP norms")
        assertEquals(1, patternCounts["Rope"])
        assertEquals(1, patternCounts["FlashAttention"])
        assertEquals(1, patternCounts["SwiGLU"])
        assertEquals(1, patternCounts["CrossEntropy"])
    }

    @Test
    fun coarsenedFunctionHasSixCoarsenedOps() {
        val (_, coarsened) = buildAndCoarsen()
        val coarsenedOps = coarsened.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
        assertEquals(6, coarsenedOps.size, "6 COARSENED ops post-coarsen")
    }

    @Test
    fun eachCoarsenedOpCarriesExpectedPrimalBodyName() {
        // Pin that each match's coarsener produced the expected primal body.
        // Catches accidental swap (e.g. a future pattern rename without
        // updating the coarsener's `DxirBuilder.function("…")` name).
        val (_, coarsened) = buildAndCoarsen()
        val primalNames = coarsened.body
            .filterIsInstance<DxirOp>()
            .filter { it.op == OpKind.COARSENED }
            .map { (it.attrs["primal_body"] as DxirFunction).name }

        val nameCounts = primalNames.groupingBy { it }.eachCount()
        assertEquals(2, nameCounts["rms_norm_primal"])
        assertEquals(1, nameCounts["rope_primal"])
        assertEquals(1, nameCounts["flash_attention_primal"])
        assertEquals(1, nameCounts["swiglu_primal"])
        assertEquals(1, nameCounts["cross_entropy_primal"])
    }

    @Test
    fun coarsenedFunctionPreservesSixUnrecognizedMatmuls() {
        // Body matmuls after coarsening:
        //   Q, K, V (3)        — projections; no recognizer matches plain QKV
        //   output projection  — post-attention
        //   down projection    — post-SwiGLU
        //   LM head            — pre-CrossEntropy (CE absorbs SOFTMAX onward, not the matmul)
        // = 6 matmuls. The attention's inner Q·K and P·V matmuls are
        // absorbed by FlashAttention; SwiGLU's gate and up matmuls are
        // absorbed by SwiGLU.
        val (_, coarsened) = buildAndCoarsen()
        val matmuls = coarsened.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.MATMUL }
        assertEquals(6, matmuls.size, "Q,K,V,O,down,lm_head matmuls remain after coarsening")
    }

    @Test
    fun coarsenedFunctionPreservesTwoResidualAdds() {
        // 2 residual ADDs (post-attn, post-MLP). The RmsNorm coarsener
        // absorbs the eps ADD inside each norm, so the only ADDs left
        // in the outer body are the residuals.
        val (_, coarsened) = buildAndCoarsen()
        val adds = coarsened.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.ADD }
        assertEquals(2, adds.size, "2 residual ADDs remain (RmsNorm eps ADDs absorbed)")
    }

    @Test
    fun decomposeCoarsenedRemovesAllSixCoarsenedOpsFromLlamaDecoder() {
        // IR-level pin for the §0.4.275 decomposeCoarsened pass on the full
        // LlamaDecoder primal: every un-annotated COARSENED must inline
        // back to its primal_body, so the post-decompose body holds zero
        // COARSENED ops while preserving the 6 unrecognized matmuls and
        // 2 residual ADDs from the outer body. The §0.4.276 emit test pins
        // this indirectly (`no stablehlo.custom_call`); this test pins it
        // directly at the IR level so a regression in DecomposeCoarsened
        // surfaces here without needing the full emit pipeline.
        val (_, coarsened) = buildAndCoarsen()
        val coarsenedCount = coarsened.body.filterIsInstance<DxirOp>()
            .count { it.op == OpKind.COARSENED }
        assertEquals(6, coarsenedCount, "prerequisite: 6 COARSENED ops")

        val decomposed = decomposeCoarsened(coarsened)
        val ops = decomposed.body.filterIsInstance<DxirOp>()
        assertEquals(
            0, ops.count { it.op == OpKind.COARSENED },
            "all 6 COARSENED ops must be inlined; got ${ops.filter { it.op == OpKind.COARSENED }.size}",
        )
        // 10 matmuls total post-decompose:
        //   6 outer preserved (Q,K,V,O,down,lm_head — never absorbed)
        // + 2 inlined from FlashAttention's primal_body (Q·K, P·V)
        // + 2 inlined from SwiGLU's primal_body (gate, up)
        // The 6-vs-10 jump is the structural signal that FlashAttention
        // and SwiGLU re-expanded; if either coarsener stops absorbing
        // those matmuls, this drops back to 6 here.
        assertEquals(
            10, ops.count { it.op == OpKind.MATMUL },
            "expected 10 matmuls (6 outer + 2 from FlashAttention + 2 from SwiGLU)",
        )
        // ADDs in the decomposed body = 2 residuals + ADDs from inlined
        // primal_bodies (RmsNorm eps ADD, RoPE recombine ADD, etc.). The
        // exact count depends on each pattern's primal_body shape; pin
        // only that we have AT LEAST the 2 residuals.
        val addCount = ops.count { it.op == OpKind.ADD }
        assertEquals(true, addCount >= 2, "expected ≥ 2 ADDs (residuals); got $addCount")
    }
}
