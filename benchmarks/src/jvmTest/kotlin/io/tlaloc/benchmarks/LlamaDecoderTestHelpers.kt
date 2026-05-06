package io.tlaloc.benchmarks

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened
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
