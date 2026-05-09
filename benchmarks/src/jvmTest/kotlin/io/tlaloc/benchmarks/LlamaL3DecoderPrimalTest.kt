package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.RecognitionMatch
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.recognizeAll
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.323 — LlamaL3DecoderPrimal builds + recognizes correctly.
 *
 * Proves the GQA recognizer + LayerNorm-companion + RmsNorm + RoPE +
 * SwiGLU + CrossEntropy all fire on a single Llama-3-shape model, and
 * pins the recognition coverage fingerprint analogous to §0.4.315 on
 * the parent [LlamaDecoderPrimal] but with GroupedQueryAttention now
 * present.
 */
class LlamaL3DecoderPrimalTest {

    @Test
    fun tinyL3ConfigBuildsValidPrimal() {
        val fn = LlamaL3DecoderPrimal.build(LlamaL3DecoderConfig.tiny)
        assertEquals("llama_l3_decoder_layer_loss", fn.name)
        assertTrue(fn.body.isNotEmpty())
        assertEquals(1, fn.returns.size)
        assertEquals(DxirType(F32, listOf()), fn.returns.single().type, "scalar loss")
    }

    @Test
    fun tinyL3ConfigParamCountAndShapes() {
        // Same 13 params as parent LlamaDecoderPrimal; the only shape
        // delta is k_w / v_w at [D, kvDim] instead of [D, D].
        val cfg = LlamaL3DecoderConfig.tiny
        val fn = LlamaL3DecoderPrimal.build(cfg)
        assertEquals(13, fn.params.size)
        val byName = fn.params.associateBy { it.name }
        assertEquals(DxirType(F32, listOf(cfg.dModel, cfg.dModel)), byName["q_w"]!!.type, "Q at full dim")
        assertEquals(DxirType(F32, listOf(cfg.dModel, cfg.kvDim)), byName["k_w"]!!.type, "K at kvDim (GQA)")
        assertEquals(DxirType(F32, listOf(cfg.dModel, cfg.kvDim)), byName["v_w"]!!.type, "V at kvDim (GQA)")
    }

    @Test
    fun llama3_8bConfigBuildsWithoutThrowing() {
        val fn = LlamaL3DecoderPrimal.build(LlamaL3DecoderConfig.llama3_8b)
        assertEquals("llama_l3_decoder_layer_loss", fn.name)
        // Pin the canonical Llama-3-8B numbers from the docblock.
        assertEquals(14336, LlamaL3DecoderConfig.llama3_8b.dFf)
        assertEquals(2048, LlamaL3DecoderConfig.llama3_8b.tokens)
        assertEquals(4, LlamaL3DecoderConfig.llama3_8b.groupRatio, "Llama-3 ratio is 32:8 = 4")
        assertEquals(1024, LlamaL3DecoderConfig.llama3_8b.kvDim, "kvDim = 8 KV-heads × 128 headDim")
    }

    @Test
    fun configValidationRejectsNonDivisibleHeadCounts() {
        assertFailsWith<IllegalArgumentException> {
            LlamaL3DecoderConfig(
                batch = 1, seq = 16, dModel = 64,
                nHeads = 4, nKvHeads = 3,   // 4 % 3 != 0
                headDim = 16, ffnMult = 4.0, vocab = 256,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            LlamaL3DecoderConfig(
                batch = 1, seq = 16, dModel = 64,
                nHeads = 4, nKvHeads = 4,   // group_ratio = 1, not GQA
                headDim = 16, ffnMult = 4.0, vocab = 256,
            )
        }
    }

    @Test
    fun recognizesGqaPlusOtherPatternsOnTinyConfig() {
        val fn = LlamaL3DecoderPrimal.build(LlamaL3DecoderConfig.tiny)
        val matches = recognizeAll(fn)

        val byPattern = matches.groupingBy { it.patternName }.eachCount()
        assertEquals(
            1, byPattern["GroupedQueryAttention"] ?: 0,
            "GQA recognizer should fire exactly once on the K/V expansion chain; got $byPattern",
        )
        // The other patterns from the parent model are still expected.
        assertTrue((byPattern["RmsNorm"] ?: 0) >= 2, "two RmsNorm sites (pre-attn + pre-MLP)")
        assertTrue((byPattern["Rope"] ?: 0) >= 1, "RoPE on Q")
        assertTrue(
            (byPattern["TransformerMLP"] ?: 0) >= 1 || (byPattern["SwiGLU"] ?: 0) >= 1,
            "MLP block recognized as TransformerMLP or SwiGLU",
        )
        assertTrue((byPattern["CrossEntropy"] ?: 0) >= 1, "LM-head cross-entropy")

        // FlashAttention should NOT separately fire — the GQA match
        // absorbs the same softmax+matmuls and is larger, so the
        // §0.4.282 resolver should pick GQA only.
        assertEquals(
            0, byPattern["FlashAttention"] ?: 0,
            "FlashAttention should be eliminated by resolveLargestMatch in favour of GroupedQueryAttention",
        )
    }

    @Test
    fun recognitionCoverageOnTinyConfig() {
        // Llama-3-shape analogue of §0.4.315's
        // `reportsStructuralRecognitionCoverageOnForward`. Pins the
        // recognized op count + per-pattern absorbed count on the tiny
        // L3 config so a recognizer regression surfaces here.
        val fn = LlamaL3DecoderPrimal.build(LlamaL3DecoderConfig.tiny)
        val matches = recognizeAll(fn)

        data class PatternCoverage(val pattern: String, val opIds: Set<Int>)
        val absorbedByPattern = matches.map { m ->
            PatternCoverage(m.patternName, m.ops.map { it.id }.toSet())
        }
        val absorbedOpIds = absorbedByPattern.flatMap { it.opIds }.toSet()
        assertEquals(
            absorbedByPattern.sumOf { it.opIds.size }, absorbedOpIds.size,
            "no overlapping matches after resolveLargestMatch",
        )

        val allOps = fn.body.filterIsInstance<DxirOp>()
        val residueOps = allOps.filter { it.id !in absorbedOpIds }
        val residueByKind = residueOps.groupingBy { it.op }.eachCount()

        println(
            buildString {
                appendLine("[l3-recognition-report]")
                appendLine("  total forward ops:          ${allOps.size}")
                val pct = "%.1f%%".format((absorbedOpIds.size * 100.0) / allOps.size)
                appendLine("  absorbed by recognizers:    ${absorbedOpIds.size} ($pct)")
                for (cov in absorbedByPattern.sortedByDescending { it.opIds.size }) {
                    appendLine("    - ${cov.pattern.padEnd(24)} ${cov.opIds.size} ops")
                }
                appendLine("  unrecognized residue:       ${residueOps.size}")
                for ((kind, count) in residueByKind.entries.sortedByDescending { it.value }) {
                    appendLine("    - ${kind.toString().padEnd(24)} $count ops")
                }
            },
        )

        // Pin the §0.4.323 baseline:
        //   total forward ops = 41
        //   absorbed = 34 (82.9%)
        //   residue = 7 (5 MATMUL Q/K/V/O/lm_head + 2 ADD residuals)
        // The K^T TRANSPOSE that the parent model leaves in residue is
        // absorbed by GQA's outer chain here.
        assertEquals(41, allOps.size, "total forward ops on tiny L3 config")
        assertEquals(34, absorbedOpIds.size, "absorbed ops; drop = recognizer regression")
        assertEquals(7, residueOps.size, "residue = unrecognized primitives")

        val absorbedByName = absorbedByPattern.associate { it.pattern to it.opIds.size }
        assertEquals(
            10, absorbedByName["GroupedQueryAttention"],
            "GQA absorbs 3 attention + 4 K-chain (TRANSPOSE + RESHAPE + BROADCAST + RESHAPE) + 3 V-chain",
        )
        assertEquals(5, absorbedByName["TransformerMLP"], "SwiGLU + down-proj (parent shape)")
        assertEquals(5, absorbedByName["Rope"], "RoPE on Q (parent shape)")
        assertEquals(4, absorbedByName["CrossEntropy"], "LM-head loss (parent shape)")
        val rmsNormTotal = absorbedByPattern.filter { it.pattern == "RmsNorm" }.sumOf { it.opIds.size }
        assertEquals(10, rmsNormTotal, "RmsNorm × 2 (parent shape)")

        // Residue: 5 MATMUL + 2 ADD. Note no TRANSPOSE in residue
        // because GQA absorbed it.
        assertEquals(5, residueByKind[OpKind.MATMUL], "Q/K/V/O/lm_head projections")
        assertEquals(2, residueByKind[OpKind.ADD], "post-attn + post-MLP residuals")
        assertTrue(residueByKind[OpKind.TRANSPOSE] == null, "K^T TRANSPOSE absorbed into GQA")
    }

    @Test
    fun coarsensGqaOnLlamaL3WithKtTransposeInChain() {
        // §0.4.324 — the v3 coarsener handles TRANSPOSE in K's outer
        // chain, so the LlamaL3 model coarsens end-to-end. K's chain is
        // [TRANSPOSE (K^T), RESHAPE (flatten)]; V's chain is [RESHAPE]
        // only. The asymmetric per-side chains are processed
        // independently — both succeed.
        val fn = LlamaL3DecoderPrimal.build(LlamaL3DecoderConfig.tiny)
        val matches = recognizeAll(fn)
        assertEquals(
            1, matches.count { it is RecognitionMatch.GroupedQueryAttention },
            "recognizer prerequisite",
        )

        val coarsened = coarsenRecognizedPatterns(fn, matches)
        val coarsenedOps = coarsened.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
        val primalNames = coarsenedOps.mapNotNull {
            (it.attrs["primal_body"] as? io.tlaloc.ir.DxirFunction)?.name
        }
        assertTrue(
            primalNames.any { it == "gqa_primal" },
            "GQA coarsener should fire post-§0.4.324; got primals=$primalNames",
        )
        assertTrue(
            primalNames.any { it == "rms_norm_primal" },
            "RmsNorm should also coarsen; got primals=$primalNames",
        )
        // The full Llama-3 model now produces COARSENED for: GQA,
        // RmsNorm × 2, RoPE, TransformerMLP, CrossEntropy. Pin the
        // count as a sentinel.
        assertEquals(
            6, coarsenedOps.size,
            "expected 6 COARSENED ops on tiny L3 (GQA + RmsNorm×2 + Rope + TransformerMLP + CrossEntropy); got ${primalNames}",
        )
    }
}
