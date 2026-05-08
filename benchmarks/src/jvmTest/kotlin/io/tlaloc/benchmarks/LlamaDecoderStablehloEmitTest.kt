package io.tlaloc.benchmarks

import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.recognizeAll
import io.tlaloc.stablehlo.toStablehlo
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.276 — Phase 3a step 3: LlamaDecoderPrimal end-to-end emit through
 * the CPU baseline pipeline.
 *
 * With §0.4.274 (keep-dims reduce in :stablehlo) + §0.4.275
 * (decomposeCoarsened pass) both landed, the full LlamaDecoderPrimal —
 * raw or coarsened-then-decomposed — must lower cleanly to StableHLO MLIR.
 * That's the dispatch surface the runtime-iree JNI work (§0.4.277+) will
 * consume.
 *
 * # The CPU baseline pipeline pinned here
 *
 * ```
 * LlamaDecoderPrimal.build(config)
 *   → recognizeAll                       (6 patterns matched)
 *   → coarsenRecognizedPatterns          (6 COARSENED ops produced)
 *   → decomposeCoarsened                 (no kernel_descriptor → inline back)
 *   → toStablehlo                        (full StableHLO MLIR)
 * ```
 *
 * GPU/TPU runs will diverge after `coarsenRecognizedPatterns` by calling
 * `lowerKernelChoice` instead of `decomposeCoarsened`, attaching
 * kernel_descriptors that emit as `stablehlo.custom_call`. The raw vs
 * decomposed MLIR sizes give an empirical sense of how much "fusion
 * surface" the coarsening unlocks for non-CPU targets.
 */
class LlamaDecoderStablehloEmitTest {

    private fun cpuBaselineEmit(): String {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
        val decomposed = decomposeCoarsened(coarsened)
        return decomposed.toStablehlo("")
    }

    @Test
    fun rawFormEmitsValidStablehlo() {
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val mlir = raw.toStablehlo("")
        assertTrue(mlir.isNotEmpty(), "raw emit must produce non-empty MLIR")
        assertTrue(
            "func.func @llama_decoder_layer_loss" in mlir,
            "expected func.func declaration; first 200 chars: ${mlir.take(200)}",
        )
    }

    @Test
    fun coarsenedThenDecomposedEmitsValidStablehlo() {
        val mlir = cpuBaselineEmit()
        assertTrue(mlir.isNotEmpty(), "CPU baseline emit must produce non-empty MLIR")
        assertTrue(
            "func.func @llama_decoder_layer_loss" in mlir,
            "expected func.func declaration in CPU baseline emit",
        )
        // Decomposition path: no custom_call should remain.
        assertTrue(
            "stablehlo.custom_call" !in mlir,
            "CPU baseline decomposes COARSENED ops; no custom_call should appear",
        )
    }

    @Test
    fun cpuBaselineEmitContainsExpectedStablehloOps() {
        val mlir = cpuBaselineEmit()
        // Matmuls (Q/K/V/O/lm_head + attention's Q·K and P·V from inlined
        // FlashAttention + TransformerMLP's gate, up, and down matmuls
        // inlined → 5+2+3 = 10). §0.4.314 reshuffled the breakdown — same total.
        assertTrue(
            "stablehlo.dot_general" in mlir || "stablehlo.dot" in mlir,
            "expected matmul lowering",
        )
        // Residual ADDs + RmsNorm eps ADDs (inlined) + RoPE recombine if ADD form.
        assertTrue("stablehlo.add" in mlir, "expected stablehlo.add")
        // RmsNorm internals: square via mul, mean via reduce + divide, rsqrt.
        assertTrue("stablehlo.reduce" in mlir, "expected stablehlo.reduce (RmsNorm mean / CrossEntropy sum)")
        assertTrue("stablehlo.rsqrt" in mlir, "expected stablehlo.rsqrt (RmsNorm)")
        // Keep-dims path emits broadcast_in_dim after the reduce.
        assertTrue(
            "stablehlo.broadcast_in_dim" in mlir,
            "expected broadcast_in_dim from keep-dims MEAN re-inflation",
        )
        // SOFTMAX from attention + CrossEntropy lowers via custom emitSoftmax.
        // It should produce something — stablehlo lowers softmax via exp + reduce + divide.
        assertTrue(
            "stablehlo.exponential" in mlir || "stablehlo.exp" in mlir,
            "expected stablehlo.exponential (from SOFTMAX lowering)",
        )
        // SILU lowers via SIGMOID (logistic) + multiply.
        assertTrue("stablehlo.logistic" in mlir, "expected stablehlo.logistic (from SILU lowering)")
        // RoPE: COS + SIN.
        assertTrue("stablehlo.cosine" in mlir, "expected stablehlo.cosine (from RoPE)")
        assertTrue("stablehlo.sine" in mlir, "expected stablehlo.sine (from RoPE)")
        // CrossEntropy: LOG.
        assertTrue("stablehlo.log" in mlir, "expected stablehlo.log (from CrossEntropy)")
    }

    @Test
    fun emitFunctionSignatureCarriesThirteenParams() {
        // The :stablehlo emitter names params %<id> rather than %<name>,
        // so we check the function signature contains the right param
        // count by counting commas in the (...) param list.
        val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        val mlir = raw.toStablehlo("")
        val funcLine = mlir.lineSequence().first { "func.func @llama_decoder_layer_loss" in it }
        val paramSection = funcLine.substringAfter("(").substringBefore(") ->")
        val paramCount = paramSection.split(",").size
        assertEquals(13, paramCount, "expected 13 params in MLIR signature; got: $funcLine")
    }

    @Test
    fun dumpsCpuBaselineMlirToBuildDir() {
        // Side-effect test: writes the CPU baseline MLIR to
        // benchmarks/build/llama-decoder-cpu-baseline.mlir for manual
        // inspection (and for piping through iree-compile during Phase 3
        // bring-up). Keeps a reproducible artifact path tied to the
        // smoke gate so the file always reflects HEAD's emit output.
        val mlir = cpuBaselineEmit()
        val target = java.io.File("build/llama-decoder-cpu-baseline.mlir")
        target.parentFile?.mkdirs()
        target.writeText(mlir)
        assertTrue(target.exists() && target.length() > 0, "MLIR dump file must exist + be non-empty")
        // Also note that wrapping in a `module { }` may be needed for
        // iree-compile; the function-level emit doesn't include it.
        // If iree-compile fails on this file, prepend `module { ... }`.
        println("[llama-decoder] CPU baseline MLIR written to ${target.absolutePath} (${mlir.length} chars)")
    }

    @Test
    fun emitIsDeterministicAcrossCalls() {
        // Pin determinism so byte-for-byte snapshot comparisons work for
        // Phase 4's regression-detection layer.
        val mlir1 = cpuBaselineEmit()
        val mlir2 = cpuBaselineEmit()
        assertEquals(mlir1, mlir2, "CPU baseline emit must be deterministic")
    }
}
