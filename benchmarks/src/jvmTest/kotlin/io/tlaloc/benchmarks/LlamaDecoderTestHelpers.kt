package io.tlaloc.benchmarks

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeAll

/**
 * Shared helpers for LlamaDecoder integration tests (§0.4.288 self-consistency,
 * §0.4.289 IREE-vs-PyTorch agreement). Centralizes the CPU baseline pipeline
 * and the deterministic input-synthesis recipe so the two tests can't drift
 * out of sync.
 */
internal fun llamaCpuBaselinePipeline(): DxirFunction {
    val raw = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
    val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
    return decomposeCoarsened(coarsened)
}

/**
 * §0.4.325 — kernel-lowering pipeline for [target]. Mirrors
 * [llamaCpuBaselinePipeline] but inserts [lowerKernelChoice] between
 * `coarsen` and `decomposeCoarsened`, so COARSENED ops whose pattern has
 * a registered [io.tlaloc.ir.recognizer.kernel.KernelTemplate] entry for
 * [target] are stamped with a `kernel_descriptor` attr instead of being
 * inlined. The StableHLO emitter then materializes those into
 * `stablehlo.custom_call @<kernelName>(...)` ops.
 *
 * Patterns without a kernel-template entry (today: everything except
 * FlashAttention) still decompose — `lowerKernelChoice` leaves them as
 * un-annotated COARSENED, then `decomposeCoarsened` inlines them. The
 * resulting MLIR is a hybrid: one `custom_call` per recognized
 * FlashAttention region, primitives elsewhere.
 *
 * @param config decoder config (tiny / medium / large).
 * @param target device target whose kernel templates drive the per-COARSENED
 *   decision. [KernelTarget.CPU_GENERIC] forces every COARSENED to decompose,
 *   producing the same MLIR as [llamaCpuBaselinePipeline] (the default
 *   production GPU path today).
 */
internal fun llamaKernelLoweredForwardPipeline(
    config: LlamaDecoderConfig,
    target: KernelTarget,
): DxirFunction {
    val raw = LlamaDecoderPrimal.build(config)
    val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
    val lowered = lowerKernelChoice(coarsened, target)
    return decomposeCoarsened(lowered)
}

/**
 * §0.4.325 — backward-mode parallel of [llamaKernelLoweredForwardPipeline].
 * Splices the gradient via [DxirReverseTransform] then runs the same
 * lowerKernelChoice → decomposeCoarsened tail.
 */
internal fun llamaKernelLoweredBackwardPipeline(
    config: LlamaDecoderConfig,
    target: KernelTarget,
): DxirFunction {
    val raw = LlamaDecoderPrimal.build(config)
    val coarsened = coarsenRecognizedPatterns(raw, recognizeAll(raw))
    val grad = DxirReverseTransform.apply(coarsened)
    val lowered = lowerKernelChoice(grad, target)
    return decomposeCoarsened(lowered)
}

/**
 * Deterministic input synthesis keyed by param name (LlamaDecoderPrimal picks
 * stable names for its `param("x_in", …)` etc. declarations):
 *
 *   - `labels` → one-hot per row at a random target token (so the cross-entropy
 *     `SUM(labels * log(probs))` term has a sane distribution),
 *   - `eps_attn` / `eps_mlp` → standard RmsNorm epsilon 1e-5 broadcast,
 *   - everything else (weights, activations, RoPE theta) → small Gaussian
 *     ~N(0, 0.05²) so RmsNorm + softmax stay well-conditioned.
 *
 * Uses [java.util.Random.nextGaussian] (deterministic by seed) so two calls
 * with the same seed produce byte-identical FloatArrays.
 */
internal fun llamaSynthesizeInputs(seed: Long, fn: DxirFunction): List<FloatArray> {
    val rng = java.util.Random(seed)
    return fn.params.map { p ->
        val n = p.type.elementCount.toInt()
        when (p.name) {
            "labels" -> {
                val arr = FloatArray(n)
                val (rows, cols) = p.type.dims
                for (i in 0 until rows) {
                    val target = rng.nextInt(cols)
                    arr[i * cols + target] = 1.0f
                }
                arr
            }
            "eps_attn", "eps_mlp" -> FloatArray(n) { 1e-5f }
            else -> FloatArray(n) { (rng.nextGaussian() * 0.05).toFloat() }
        }
    }
}
