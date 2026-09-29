package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.maestro.serving.HfServingExport
import java.nio.file.Path

/**
 * Writes a Hugging Face checkpoint directory (Llama or Qwen3, as
 * `HfServingExport` reads them; a merged LoRA fine-tune is one) as a serving
 * artifact that [ServingModel] loads: pages of 16 tokens, a KV pool that
 * holds [maxBatch] sequences of [maxContext] tokens, prefill entries, and
 * full-history pools (no windowed layers).
 *
 * For every other option (context ladders, numBlocks, weight quantization,
 * Triton) use `io.tlaloc.maestro.serving.HfServingExport` directly.
 */
object ServingExport {

    private const val BLOCK_SIZE = 16

    /** [export] with the weights staged in the family's default width. */
    @JvmStatic
    fun export(checkpoint: Path, artifact: Path, maxBatch: Int, maxContext: Int): Path =
        export(checkpoint, artifact, maxBatch, maxContext, null)

    /**
     * Exports [checkpoint] into [artifact]. [bf16Weights] stages the weights
     * as bf16 (true), f32 (false), or the family's default (null).
     * Returns [artifact].
     */
    @JvmStatic
    fun export(checkpoint: Path, artifact: Path, maxBatch: Int, maxContext: Int, bf16Weights: Boolean?): Path {
        require(maxBatch >= 1 && maxContext >= 1) { "ServingExport: maxBatch=$maxBatch and maxContext=$maxContext must be positive" }
        val pagesPerSeq = (maxContext + BLOCK_SIZE - 1) / BLOCK_SIZE
        HfCheckpoint.open(checkpoint).use { ckpt ->
            val config = when (bf16Weights) {
                null -> ckpt.config
                true -> ckpt.config.copy(weightDType = BF16)
                false -> ckpt.config.copy(weightDType = F32)
            }
            HfServingExport.export(
                ckpt = ckpt,
                dir = artifact,
                config = config,
                policy = DecodeBucketPolicy(maxBatch = maxBatch, maxContext = maxContext, blockSize = BLOCK_SIZE),
                // Page 0 is scratch; every sequence of a full batch gets its pages.
                numBlocks = 1 + maxBatch * pagesPerSeq,
                modelName = HfServingExport.modelNameFor(checkpoint),
                windowedKv = false,
            )
        }
        return artifact
    }
}
