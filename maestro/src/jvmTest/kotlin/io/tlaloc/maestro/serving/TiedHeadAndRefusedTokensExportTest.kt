package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfModelFamily
import java.nio.file.Files
import java.nio.file.Path
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Two things an artifact of a tied, multimodal-style checkpoint says about
 * itself: its head reads the embedding table (no second copy of the table in
 * the weight table, one `dot_general` contracting the hidden axis of both),
 * and the token ids a server must refuse are listed in the manifest.
 */
class TiedHeadAndRefusedTokensExportTest {

    private val config = HfDecoderConfig(
        architecture = "Qwen3ForCausalLM", modelType = "qwen3",
        hiddenSize = 16, intermediateSize = 24, numLayers = 2,
        numHeads = 4, numKvHeads = 2, headDim = 8, vocabSize = 29,
        rmsNormEps = 1e-6, ropeTheta = 1e6, maxPositionEmbeddings = 64,
        tieWordEmbeddings = true, attentionBias = false, torchDtype = "float32",
        ropeScalingType = null, family = HfModelFamily.Qwen3,
        refusedTokenIds = mapOf(27 to "image_token_id", 28 to "video_token_id"),
    )

    private fun export(dir: Path, c: HfDecoderConfig, refused: Map<Int, String> = c.refusedTokenIds): ServingManifest {
        val policy = DecodeBucketPolicy(maxBatch = 1, maxContext = 16, blockSize = 4, minContext = 16)
        val model = c.toDecodeModelShape(numBlocks = 9, blockSize = 4)
        val specs = policy.allBuckets.map { HfDecoderGraph.spec(c, model, it) } +
            HfDecoderGraph.spec(c, model, DecodeBucket(1, 16), DecodeGraphKind.PREFILL)
        val rng = Random(3)
        return ServingArtifactWriter.export(
            dir = dir, modelName = "tiny-tied", modelHash = "tiny-tied-hash", model = model,
            ladder = ServingArtifactWriter.ladderOf(policy), specs = specs,
            stageWeight = { slot -> FloatArray(slot.type.dims.fold(1) { a, b -> a * b }) { rng.nextFloat() - 0.5f } },
            build = { HfDecoderGraph.build(it, c) },
            refusedTokenIds = refused,
        )
    }

    private fun <T> withArtifact(c: HfDecoderConfig, refused: Map<Int, String> = c.refusedTokenIds, body: (Path, ServingManifest) -> T): T {
        val dir = Files.createTempDirectory("tlaloc-tied-artifact")
        try {
            return body(dir, export(dir, c, refused))
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun theTiedHeadIsNotASecondCopyOfTheTable() {
        val direct = withArtifact(config) { dir, m ->
            val names = m.weights.table.map { it.name }
            assertFalse("lmHead" in names, "no head slot: $names")
            for (e in m.entries) {
                val body = Files.readString(dir.resolve(e.bodyPath))
                // The head: [rows, 16] x [29, 16] contracting the hidden axis of both.
                assertTrue(
                    Regex("stablehlo\\.dot_general %\\w+, %\\w+, contracting_dims = \\[1\\] x \\[1\\] : " +
                        "\\(tensor<1x16xf32>, tensor<29x16xf32>\\) -> tensor<1x29xf32>").containsMatchIn(body),
                    "${e.entryId}: no head dot_general against the table",
                )
                // No transpose of the table (a prefill body transposes its attention rows).
                assertFalse(
                    Regex("stablehlo\\.transpose .*\\(tensor<29x16xf32>\\)").containsMatchIn(body),
                    "${e.entryId}: a transpose of the table in the body",
                )
            }
            m.weights.table.sumOf { it.byteLength }
        }
        // Control: with the copy the table is there twice.
        val copy = withArtifact(config.copy(tiedHeadCopy = true)) { _, m ->
            assertTrue("lmHead" in m.weights.table.map { it.name })
            m.weights.table.sumOf { it.byteLength }
        }
        assertEquals(29L * 16 * 4, copy - direct, "the copy costs vocab x hidden f32 values")
    }

    @Test
    fun theRefusedTokensAreInTheManifestAndSurviveTheWire() = withArtifact(config) { dir, m ->
        assertEquals(
            listOf(ServingRefusedToken(27, "image_token_id"), ServingRefusedToken(28, "video_token_id")),
            m.model.refusedTokens,
        )
        val text = Files.readString(dir.resolve(ServingManifest.FILE_NAME))
        assertTrue(
            "\"refusedTokens\":[{\"id\":27,\"configKey\":\"image_token_id\"},{\"id\":28,\"configKey\":\"video_token_id\"}]" in text,
            text,
        )
        assertEquals(m, ServingManifest.fromJson(text))
    }

    @Test
    fun withoutRefusedTokensTheManifestHasNoSuchKey() = withArtifact(config, refused = emptyMap()) { dir, m ->
        assertEquals(emptyList(), m.model.refusedTokens)
        assertFalse("refusedTokens" in Files.readString(dir.resolve(ServingManifest.FILE_NAME)))
    }

    @Test
    fun aRefusedIdOutsideTheVocabularyIsRefusedByName() {
        val e = assertFailsWith<IllegalArgumentException> {
            withArtifact(config, refused = mapOf(29 to "image_token_id")) { _, _ -> }
        }
        assertTrue("refused token 29 (image_token_id) is outside the vocabulary [0, 29)" in e.message!!, e.message)
    }
}
