package io.tlaloc.benchmarks

import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.stablehlo.toStablehlo
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.310 — diagnostic that pins **why the comparison matrix shows
 * Tlaloc-PJRT ≈ JAX rather than Tlaloc beating JAX**:
 *
 *   The §0.4.286 / §0.4.292 / §0.4.308 production GPU pipeline is
 *
 *     build → recognize → coarsen → decomposeCoarsened → toStablehlo
 *                                   ^^^^^^^^^^^^^^^^^^
 *
 *   `decomposeCoarsened` inlines every COARSENED op's primal_body back
 *   to primitives **before** emit. The MLIR fed to PJRT-XLA contains
 *   zero `stablehlo.custom_call` ops — exactly the same kind of MLIR
 *   JAX produces from equivalent Python. **Tlaloc's coarsening info is
 *   thrown away at runtime.**
 *
 *   The §0.4.270 [io.tlaloc.ir.recognizer.kernel.lowerKernelChoice]
 *   machinery exists and would attach `kernel_descriptor` attrs that
 *   the StableHLO emitter could turn into `stablehlo.custom_call
 *   @<kernel_name>`. It's just not wired into the LlamaDecoder
 *   pipeline.
 *
 *   This test asserts the situation as of HEAD so any future work that
 *   wires `lowerKernelChoice` in will fail this test loudly (forcing
 *   a follow-up update of the assertions).
 *
 * # Why this matters for the matrix
 *
 * Tlaloc's pitch was supposed to be: "we recognize FlashAttention
 * structurally and emit a fused kernel; XLA only pattern-matches at
 * MLIR level". Today, neither side has the fused kernel — Tlaloc
 * decomposes, XLA pattern-matches. Both arrive at similar SASS, hence
 * the ~similar wall-clock.
 *
 * To USE coarsening on the GPU path, two things need to happen:
 *   1. Replace `decomposeCoarsened` with `lowerKernelChoice(target=GPU)`
 *      in the LlamaDecoder pipeline. Cheap — just plumbing.
 *   2. Register custom-call handlers with PJRT-XLA so the plugin
 *      knows what to do when it sees `stablehlo.custom_call
 *      @flash_attention(...)`. **Heavy** — needs hand-written
 *      CUDA/CUTLASS kernels that beat XLA's auto-fusion.
 *
 * Without (2), step (1) alone makes XLA error on unknown custom_calls.
 * The "Tlaloc beats JAX via coarsening" story requires both.
 */
class LlamaDecoderCoarseningEmitDiagnosticTest {

    @Test
    fun forwardMlirHasZeroCustomCallsToday() {
        // The current LlamaDecoder forward pipeline emits pure decomposed
        // primitives. If this assertion ever fails, someone has wired
        // lowerKernelChoice in — update the docs and re-validate.
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val decomposed = decomposeCoarsened(coarsened)
        val mlir = decomposed.toStablehlo("")

        val customCalls = mlir.split("\n").count { "stablehlo.custom_call" in it }
        assertEquals(
            0, customCalls,
            "production GPU dispatch path emits 0 custom_calls today; " +
                "if this fails, lowerKernelChoice has been wired in — " +
                "update the diagnostic + the matrix narrative.",
        )

        // Sanity: the pipeline DID coarsen first (proves the recognizer +
        // coarsener machinery does match patterns; we just throw the info
        // away at decompose time).
        val coarsenedOps = coarsened.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertTrue(
            coarsenedOps >= 5,
            "expected ≥5 COARSENED ops post-coarsen (RmsNorm×2 + RoPE + FlashAttention + TransformerMLP + CrossEntropy); " +
                "got $coarsenedOps",
        )

        val decomposedOps = decomposed.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(
            0, decomposedOps,
            "decomposeCoarsened should leave 0 COARSENED ops (matching the 0 custom_calls in emit)",
        )

        println(
            "[coarsening-emit-diag] forward: coarsened=$coarsenedOps → decomposed=$decomposedOps → emitted custom_calls=$customCalls",
        )
    }

    @Test
    fun backwardMlirHasZeroCustomCallsToday() {
        // Same observation for the gradient path. Tlaloc's AD inlines each
        // coarsener's gradient_body (the analytical VJP), then decompose
        // strips any leftover COARSENED ops.
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val grad = DxirReverseTransform.apply(coarsened)
        val gradDecomposed = decomposeCoarsened(grad)
        val mlir = gradDecomposed.toStablehlo("")

        val customCalls = mlir.split("\n").count { "stablehlo.custom_call" in it }
        assertEquals(
            0, customCalls,
            "production GPU backward path emits 0 custom_calls today",
        )
        val coarsenedOps = gradDecomposed.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(0, coarsenedOps)

        println("[coarsening-emit-diag] backward: emitted custom_calls=$customCalls")
    }

    /**
     * §0.4.315 — structural-recognition coverage report. Answers the
     * question: "what fraction of LlamaDecoder is structurally
     * recognized as a known pattern?" Useful as a record across
     * commits — a regression in coverage (recognizer stops matching
     * a pattern it used to) shows up as a drop in the absorbed-ops
     * count here.
     *
     * The recognition fraction is computed against the **forward** body
     * because that's the canonical IR shape; the gradient body is
     * derived from it and would double-count the same patterns. We
     * count each [DxirOp] in the raw `LlamaDecoderPrimal.build` output;
     * params and constants are excluded (they aren't computational
     * ops in the recognized-vs-residue sense).
     *
     * # What's expected on the medium config (HEAD as of §0.4.315)
     *
     * 35 forward ops; 27 absorbed (77.1%) across 6 matches:
     *   - TransformerMLP: 5 ops  (SwiGLU + down-proj)
     *   - FlashAttention: 3 ops  (Q·K^T, softmax, P·V)
     *   - RmsNorm × 2:    10 ops (square, reduce, eps-add, rsqrt, mul) × 2
     *   - RoPE:           5 ops  (cos, sin, mul-cos, mul-sin, sub)
     *   - CrossEntropy:   4 ops  (softmax + log + mul + reduce chain)
     *
     * Unrecognized residue (8 ops, 22.9%):
     *   - MATMUL × 5: Q / K / V / O / lm_head projections
     *   - ADD × 2:    post-attention + post-MLP residuals
     *   - TRANSPOSE × 1: K^T for the attention scores
     */
    @Test
    fun reportsStructuralRecognitionCoverageOnForward() {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.medium)
        val matches = recognizeAll(raw)

        // 1. Bucket recognized ops by pattern. Use absorbedOpIds-equivalent
        //    semantics: every op in `match.ops` gets attributed to that
        //    pattern. Same op can't be in two matches because
        //    resolveLargestMatch already de-duped overlaps.
        data class PatternCoverage(val pattern: String, val opIds: Set<Int>)
        val absorbedByPattern = matches.map { m ->
            PatternCoverage(m.patternName, m.ops.map { it.id }.toSet())
        }
        val absorbedOpIds = absorbedByPattern.flatMap { it.opIds }.toSet()
        // Sanity: union size == sum of sizes (no overlaps).
        assertEquals(
            absorbedByPattern.sumOf { it.opIds.size }, absorbedOpIds.size,
            "resolveLargestMatch should have produced non-overlapping matches",
        )

        // 2. Bucket the forward body's DxirOps by recognized vs residue.
        //    Residue is grouped by OpKind for the report.
        val allOps = raw.body.filterIsInstance<DxirOp>()
        val residueOps = allOps.filter { it.id !in absorbedOpIds }
        val residueByKind = residueOps.groupingBy { it.op }.eachCount()

        // 3. Print the report. Single block so a `-i` grep can extract it.
        println(
            buildString {
                appendLine("[coarsening-recognition-report]")
                appendLine("  total forward ops:          ${allOps.size}")
                appendLine("  absorbed by recognizers:    ${absorbedOpIds.size} (${pct(absorbedOpIds.size, allOps.size)})")
                for (cov in absorbedByPattern.sortedByDescending { it.opIds.size }) {
                    appendLine("    - ${cov.pattern.padEnd(18)} ${cov.opIds.size} ops")
                }
                appendLine("  unrecognized residue:       ${residueOps.size} (${pct(residueOps.size, allOps.size)})")
                for ((kind, count) in residueByKind.entries.sortedByDescending { it.value }) {
                    appendLine("    - ${kind.toString().padEnd(18)} $count ops")
                }
            },
        )

        // 4. Pin coverage so a regression in any recognizer shows up as
        //    a numeric drop here. §0.4.315 baseline:
        //    35 forward ops, 27 absorbed, 8 residue.
        assertEquals(35, allOps.size, "raw forward body op count (params/consts excluded)")
        assertEquals(27, absorbedOpIds.size, "recognized ops; drop = recognizer regression")
        assertEquals(8, residueOps.size, "residue = unrecognized primitives")

        // Pin the per-pattern absorbed-op counts so a coarsener that
        // accidentally narrows its match scope surfaces here.
        val absorbedByName = absorbedByPattern.associate { it.pattern to it.opIds.size }
        assertEquals(5, absorbedByName["TransformerMLP"], "SwiGLU + down-proj = 5 ops")
        assertEquals(3, absorbedByName["FlashAttention"], "Q·K, softmax, P·V")
        assertEquals(5, absorbedByName["Rope"], "cos, sin, 2 muls, sub")
        assertEquals(4, absorbedByName["CrossEntropy"], "softmax + log + mul + reduce chain")
        // RmsNorm fires twice — sum of both occurrences.
        val rmsNormTotal = absorbedByPattern.filter { it.pattern == "RmsNorm" }.sumOf { it.opIds.size }
        assertEquals(10, rmsNormTotal, "RmsNorm × 2 occurrences × 5 ops each")

        // Pin the residue shape — Q/K/V/O/lm_head matmuls + 2 residual
        // ADDs + 1 attention transpose. If a coarsener starts absorbing
        // these (e.g. a future "AttentionWithKt" recognizer absorbs the
        // K^T transpose), the residue counts shift — update the asserts.
        assertEquals(5, residueByKind[OpKind.MATMUL], "Q/K/V/O/lm_head projections")
        assertEquals(2, residueByKind[OpKind.ADD], "post-attn + post-MLP residuals")
        assertEquals(1, residueByKind[OpKind.TRANSPOSE], "K^T transpose for attention")
    }

    private fun pct(num: Int, denom: Int): String {
        if (denom == 0) return "n/a"
        val p = (num * 100.0) / denom
        return "%.1f%%".format(p)
    }
}
