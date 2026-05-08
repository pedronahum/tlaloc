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
}
