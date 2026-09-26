package io.tlaloc.maestro.serving

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I8
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfModelFamily
import io.tlaloc.ir.inference.WeightQuant
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * An artifact with int8 weights: the weight table lists each quantized
 * Linear as `i8` (one byte per element) followed by its `f32` scales, the
 * entry bodies take them as `i8` and `f32` parameters, and the manifest reads
 * back as written.
 */
class Int8WeightArtifactExportTest {

    private val config = HfDecoderConfig(
        architecture = "Qwen3ForCausalLM", modelType = "qwen3",
        hiddenSize = 16, intermediateSize = 24, numLayers = 2,
        numHeads = 4, numKvHeads = 2, headDim = 8, vocabSize = 29,
        rmsNormEps = 1e-6, ropeTheta = 1e6, maxPositionEmbeddings = 64,
        tieWordEmbeddings = true, attentionBias = false, torchDtype = "bfloat16",
        ropeScalingType = null, family = HfModelFamily.Qwen3,
        weightDType = BF16, weightQuant = WeightQuant.INT8,
    )

    @Test
    fun quantizedLinearsAreInt8FilesFollowedByTheirScales() {
        val dir = Files.createTempDirectory("tlaloc-int8-artifact")
        try {
            val policy = DecodeBucketPolicy(maxBatch = 1, maxContext = 16, blockSize = 4, minContext = 16)
            val model = config.toDecodeModelShape(numBlocks = 9, blockSize = 4)
            val specs = policy.allBuckets.map { HfDecoderGraph.spec(config, model, it) } +
                HfDecoderGraph.spec(config, model, DecodeBucket(1, 16), DecodeGraphKind.PREFILL)
            val m = ServingArtifactWriter.export(
                dir = dir, modelName = "tiny-int8", modelHash = "tiny-int8-hash", model = model,
                ladder = ServingArtifactWriter.ladderOf(policy), specs = specs,
                writeWeight = { slot, out ->
                    val n = slot.type.dims.fold(1L) { a, b -> a * b }
                    val bytes = n * slot.type.dtype.sizeBytes
                    out.write(ByteArray(bytes.toInt()) { (it % 7).toByte() })
                    bytes
                },
                build = { HfDecoderGraph.build(it, config) },
            )
            val byName = m.weights.table.associateBy { it.name }
            assertEquals(I8.name, byName.getValue("qProj0").dtype)
            assertEquals(16L * 32, byName.getValue("qProj0").byteLength)
            assertEquals(F32.name, byName.getValue("qProj0Scale").dtype)
            assertEquals(4L * 32, byName.getValue("qProj0Scale").byteLength)
            assertEquals(I8.name, byName.getValue("downProj1").dtype)
            assertEquals(4L * 16, byName.getValue("downProj1Scale").byteLength)
            // The embedding table and the norms keep the weight dtype.
            assertEquals(config.weightDType.name, byName.getValue("embedTokens").dtype)
            assertEquals(config.weightDType.name, byName.getValue("inputNorm0").dtype)
            val names = m.weights.table.map { it.name }
            assertEquals(names.indexOf("qProj0") + 1, names.indexOf("qProj0Scale"))
            for (e in m.entries) {
                val body = Files.readString(dir.resolve(e.bodyPath))
                assertTrue("tensor<16x32xi8>" in body, "${e.entryId}: q_proj as an int8 parameter")
                assertTrue(
                    Regex("""stablehlo\.convert %\w+ : \(tensor<16x32xi8>\) -> tensor<16x32xbf16>""").containsMatchIn(body),
                    "${e.entryId}: the codes widened to bf16",
                )
            }
            assertEquals(m, ServingManifest.fromJson(Files.readString(dir.resolve(ServingManifest.FILE_NAME))))
        } finally {
            dir.toFile().deleteRecursively()
        }
    }
}
