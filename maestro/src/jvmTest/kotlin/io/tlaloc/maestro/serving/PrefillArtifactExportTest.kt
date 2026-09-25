package io.tlaloc.maestro.serving

import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderGraph
import java.nio.file.Files
import java.nio.file.Path
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * A Llama serving artifact with prefill entries, exported the way
 * `HfServingExport` exports a checkpoint (same specs, same builder, same
 * writer), over a small random-weight config so that no checkpoint is needed.
 * Checks the manifest the Triton backend's sequence mode reads: version, the
 * two kinds of entries, their shapes, and StableHLO bodies that the backend's
 * signature check will accept.
 */
class PrefillArtifactExportTest {

    private val config = HfDecoderConfig(
        architecture = "LlamaForCausalLM",
        modelType = "llama",
        hiddenSize = 16,
        intermediateSize = 40,
        numLayers = 2,
        numHeads = 4,
        numKvHeads = 2,
        headDim = 4,
        vocabSize = 23,
        rmsNormEps = 1e-5,
        ropeTheta = 10000.0,
        maxPositionEmbeddings = 64,
        tieWordEmbeddings = false,
        attentionBias = false,
        torchDtype = "float32",
        ropeScalingType = null,
    )

    private val policy = DecodeBucketPolicy(maxBatch = 2, maxContext = 16, blockSize = 4, minContext = 8)
    private val model = config.toDecodeModelShape(numBlocks = 9, blockSize = 4)

    /** Exports with prefill entries up to batch [prefillMaxBatch] (0: none), as HfServingExport does. */
    private fun export(dir: Path, prefillMaxBatch: Int): ServingManifest {
        val rng = Random(7)
        return ServingArtifactWriter.export(
            dir = dir, modelName = "tiny-llama", modelHash = "tiny-llama-hash", model = model,
            ladder = ServingArtifactWriter.ladderOf(policy),
            specs = HfServingExport.specs(config, model, policy, prefillMaxBatch),
            stageWeight = { slot ->
                FloatArray(slot.type.dims.fold(1) { a, b -> a * b }) { rng.nextFloat() - 0.5f }
            },
            build = { HfDecoderGraph.build(it, config) },
        )
    }

    @Test
    fun prefillEntriesAreWrittenBesideTheDecodeLadder() {
        val dir = Files.createTempDirectory("tlaloc-prefill-artifact")
        try {
            val m = export(dir, prefillMaxBatch = 1)
            assertEquals(ServingManifest.SCHEMA_VERSION, m.schemaVersion)
            assertEquals(
                listOf("decode_b1_c8", "decode_b1_c16", "decode_b2_c8", "decode_b2_c16", "prefill_b1_c8", "prefill_b1_c16"),
                m.entries.map { it.entryId },
            )
            val p = m.entryFor(DecodeGraphKind.PREFILL, 1, 16)
            assertEquals(16, p.tokensPerSeq)
            assertEquals(listOf(1, 16), p.inputs.first { it.name == "tokenIds" }.type.dims)
            assertEquals(listOf(16), p.inputs.first { it.name == "slotMapping" }.type.dims)
            assertEquals(listOf(1, 1, config.vocabSize), p.outputs[0].type.dims)
            // The prefill signature is the decode signature: same slots, same order.
            val d = m.entryFor(DecodeGraphKind.DECODE, 1, 16)
            assertEquals(d.inputs.map { it.name to it.role }, p.inputs.map { it.name to it.role })
            assertEquals(d.outputs.map { it.name to it.role }, p.outputs.map { it.name to it.role })
            // The emitted body takes the chunk and returns one row of logits.
            val body = Files.readString(dir.resolve(p.bodyPath))
            assertTrue("func.func public @main(" in body || "func.func @main(" in body)
            assertTrue("tensor<1x16xi32>" in body, "tokenIds [1, 16] in the prefill body")
            assertTrue("tensor<1x1x23xf32>" in body, "last-position logits in the prefill body")
            // It round-trips and a sequence-mode Triton config is written for it.
            val back = ServingManifest.fromJson(Files.readString(dir.resolve(ServingManifest.FILE_NAME)))
            assertEquals(m, back)
            val cfg = TritonModelRepository.config(back, "tiny")
            assertTrue("max_batch_size: 2\n" in cfg, cfg)
            assertTrue("a request of several tokens runs as a prefill chunk" in cfg, cfg)
            assertTrue("{ name: \"LOGITS\" data_type: TYPE_FP32 dims: [ 23 ] }" in cfg, cfg)
            assertTrue("max_candidate_sequences: 8\n" in cfg, cfg)
            assertTrue("share a prefill call" !in cfg, cfg)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun withoutPrefillTheConfigSaysPromptsRunAsDecodeSteps() {
        val dir = Files.createTempDirectory("tlaloc-noprefill-artifact")
        try {
            val m = export(dir, prefillMaxBatch = 0)
            assertTrue(m.entries.none { it.kind == DecodeGraphKind.PREFILL })
            val cfg = TritonModelRepository.config(m, "tiny")
            assertTrue("a request of several tokens runs as decode steps" in cfg, cfg)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun batchedPrefillEntriesTakeSeveralPromptsInOneCall() {
        val dir = Files.createTempDirectory("tlaloc-batched-prefill-artifact")
        try {
            val m = export(dir, prefillMaxBatch = 2)
            assertEquals(
                listOf(
                    "decode_b1_c8", "decode_b1_c16", "decode_b2_c8", "decode_b2_c16",
                    "prefill_b1_c8", "prefill_b1_c16", "prefill_b2_c8", "prefill_b2_c16",
                ),
                m.entries.map { it.entryId },
            )
            val p = m.entryFor(DecodeGraphKind.PREFILL, 2, 16)
            assertEquals(listOf(2, 16), p.inputs.first { it.name == "tokenIds" }.type.dims)
            assertEquals(listOf(2, 4), p.inputs.first { it.name == "blockTables" }.type.dims)
            assertEquals(listOf(32), p.inputs.first { it.name == "slotMapping" }.type.dims)
            assertEquals(listOf(2, 1, config.vocabSize), p.outputs[0].type.dims)
            val body = Files.readString(dir.resolve(p.bodyPath))
            assertTrue("tensor<2x16xi32>" in body, "tokenIds [2, 16] in the batched prefill body")
            assertTrue("tensor<2x1x23xf32>" in body, "each row's last-position logits")
            val cfg = TritonModelRepository.config(m, "tiny")
            assertTrue("The prompts of up to 2 sequences in one batch share a prefill call." in cfg, cfg)
            assertTrue("max_batch_size: 2\n" in cfg, cfg)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun theDefaultPrefillLadderIsTheBatchLadderAndALargerOneIsRefusedByName() {
        val specs = HfServingExport.specs(config, model, policy)
        assertEquals(
            listOf(1 to 8, 1 to 16, 2 to 8, 2 to 16),
            specs.filter { it.kind == DecodeGraphKind.PREFILL }.map { it.bucket.batch to it.bucket.maxContext },
        )
        assertEquals(4, HfServingExport.specs(config, model, policy, prefillMaxBatch = 0).size)
        val e = assertFailsWith<IllegalArgumentException> {
            HfServingExport.specs(config, model, policy, prefillMaxBatch = 3)
        }
        assertTrue("prefillMaxBatch 3" in e.message!! && "largest decode batch 2" in e.message!!, e.message!!)
    }
}
