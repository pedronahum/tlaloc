package io.tlaloc.maestro.serving

import io.tlaloc.core.F32
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfModelFamily
import io.tlaloc.ir.inference.WeightQuant
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The manifest's `modelHash` and every entry's `cacheKey` cover the staged weights'
 * values: a fine-tune exported under its base model's name and hash is another model.
 */
class ModelHashCoversWeightsTest {

    private val config = HfDecoderConfig(
        architecture = "LlamaForCausalLM", modelType = "llama",
        hiddenSize = 16, intermediateSize = 24, numLayers = 1,
        numHeads = 4, numKvHeads = 2, headDim = 4, vocabSize = 29,
        rmsNormEps = 1e-6, ropeTheta = 1e4, maxPositionEmbeddings = 64,
        tieWordEmbeddings = false, attentionBias = false, torchDtype = "float32",
        ropeScalingType = null, family = HfModelFamily.Llama,
        weightDType = F32, weightQuant = WeightQuant.NONE,
    )

    private fun export(salt: Int): ServingManifest {
        val dir = Files.createTempDirectory("tlaloc-model-hash")
        try {
            val policy = DecodeBucketPolicy(maxBatch = 1, maxContext = 16, blockSize = 4, minContext = 16)
            val model = config.toDecodeModelShape(numBlocks = 9, blockSize = 4)
            return ServingArtifactWriter.export(
                dir = dir, modelName = "base", modelHash = "base-hash", model = model,
                ladder = ServingArtifactWriter.ladderOf(policy),
                specs = policy.allBuckets.map { HfDecoderGraph.spec(config, model, it) },
                stageWeight = { slot ->
                    val n = slot.type.dims.fold(1) { a, b -> a * b }
                    FloatArray(n) { ((it * 31 + salt) % 13) / 13f }
                },
                build = { HfDecoderGraph.build(it, config) },
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `other weights under one name get another hash and other cache keys`() {
        val base = export(salt = 0)
        val again = export(salt = 0)
        val tuned = export(salt = 1)
        assertTrue(base.modelHash.startsWith("base-hash:weights-"), base.modelHash)
        assertEquals(base.modelHash, again.modelHash, "same weights, same hash")
        assertEquals(base.entries.map { it.cacheKey }, again.entries.map { it.cacheKey })
        assertNotEquals(base.modelHash, tuned.modelHash, "other weights, other hash")
        for ((b, t) in base.entries.zip(tuned.entries)) {
            assertNotEquals(b.cacheKey, t.cacheKey, b.entryId)
            assertTrue(tuned.modelHash in t.cacheKey, t.cacheKey)
        }
    }
}
