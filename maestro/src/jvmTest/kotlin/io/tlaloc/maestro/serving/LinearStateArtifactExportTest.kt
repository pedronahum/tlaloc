package io.tlaloc.maestro.serving

import io.tlaloc.core.F32
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
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
 * A qwen3_5 artifact (Gated DeltaNet layers): tlaloc-serving-v4 with the
 * linear-attention state pools in the manifest, the state roles in every
 * entry, and a Triton model that admits no more sequences than there are
 * state slots.
 */
class LinearStateArtifactExportTest {

    private val config = HfDecoderConfig.parse(
        """
        {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": false,
         "text_config": {"hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 4,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 31,
           "max_position_embeddings": 64,
           "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
           "linear_conv_kernel_dim": 4, "linear_key_head_dim": 4, "linear_num_key_heads": 2,
           "linear_num_value_heads": 4, "linear_value_head_dim": 4,
           "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}}}
        """.trimIndent(),
    ).copy(weightDType = F32)

    private fun export(dir: Path, slots: Int = 3): ServingManifest {
        val policy = DecodeBucketPolicy(maxBatch = 2, maxContext = 32, blockSize = 4, minContext = 32)
        val model = config.toDecodeModelShape(numBlocks = 17, blockSize = 4, stateSlots = slots)
        val specs = policy.allBuckets.map { HfDecoderGraph.spec(config, model, it) } +
            HfDecoderGraph.spec(config, model, DecodeBucket(2, 32), DecodeGraphKind.PREFILL)
        val rng = Random(11)
        return ServingArtifactWriter.export(
            dir = dir, modelName = "tiny-qwen35", modelHash = "tiny-qwen35-hash", model = model,
            ladder = ServingArtifactWriter.ladderOf(policy), specs = specs,
            stageWeight = { slot -> FloatArray(slot.type.dims.fold(1) { a, b -> a * b }) { rng.nextFloat() - 0.5f } },
            build = { HfDecoderGraph.build(it, config) },
        )
    }

    @Test
    fun theArtifactIsV4AndPublishesTheStatePools() {
        val dir = Files.createTempDirectory("qwen35-artifact")
        val m = export(dir)
        assertEquals(ServingManifest.SCHEMA_VERSION_4, m.schemaVersion)
        val ls = m.model.linearState!!
        assertEquals(listOf(0, 1, 2), ls.layers)
        assertEquals(3, ls.numSlots)
        assertEquals(listOf(3, 3, 32), ls.convStateDims)
        assertEquals(listOf(3, 4, 4, 4), ls.recurrentStateDims)
        val back = ServingManifest.fromJson(Files.readString(dir.resolve(ServingManifest.FILE_NAME)))
        assertEquals(m, back)
        for (e in m.entries) {
            assertEquals(1, e.inputs.count { it.role == DecodeSlotRole.STATE_SLOTS }, e.entryId)
            assertEquals(6, e.inputs.count { it.role == DecodeSlotRole.STATE_POOL_IN }, e.entryId)
            assertEquals(2, e.inputs.count { it.role == DecodeSlotRole.KV_POOL_IN }, e.entryId)
            for ((i, o) in e.donationPairs) assertEquals(e.inputs[i].type, e.outputs[o].type, e.entryId)
        }
    }

    @Test
    fun aManifestThatDisagreesAboutItsStatePoolsIsRefusedByName() {
        val m = export(Files.createTempDirectory("qwen35-artifact"))
        val ex = assertFailsWith<IllegalArgumentException> {
            m.copy(model = m.model.copy(linearState = null))
        }
        assertTrue("state" in ex.message!!, ex.message!!)
        val ex2 = assertFailsWith<IllegalArgumentException> {
            m.copy(model = m.model.copy(linearState = m.model.linearState!!.copy(layers = listOf(0, 1))))
        }
        assertTrue("STATE_POOL_IN" in ex2.message!!, ex2.message!!)
    }

    @Test
    fun theTritonModelAdmitsNoMoreSequencesThanStateSlots() {
        val m = export(Files.createTempDirectory("qwen35-artifact"), slots = 3)
        val cfg = TritonModelRepository.config(m, "qwen35")
        assertTrue("max_candidate_sequences: 3\n" in cfg, cfg)
    }
}
