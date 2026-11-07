package io.tlaloc.maestro.serving

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.WeightQuant
import java.nio.file.Files
import java.nio.file.Path
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A Ripple artifact with a DFlash2 draft, at the checkpoint's quantization: NVFP4 MLPs, bf16
 * attention and draft. The draft keeps only KV pools, so a v5 manifest without linear state
 * and its Triton model. Neither program is run here (the numbers are `HfRippleDFlash2Test`'s).
 */
class RippleServingArtifactTest {
    private val rng = Random(20261105)
    private fun linear(o: Int, i: Int) = FloatArray(o * i) { (0.25 * rng.nextGaussian()).toFloat() }
    private fun gain(n: Int) = FloatArray(n) { 1f }

    private class Tensors {
        val all = ArrayList<SafetensorsTensor>()
        fun put(name: String, data: FloatArray, vararg dims: Int) {
            all += SafetensorsTensor(name, F32, dims, HostF32Storage(data))
        }
    }

    private fun Tensors.attention(p: String) {
        put(p + "self_attn.q_proj.weight", linear(HEADS * HD, HIDDEN), HEADS * HD, HIDDEN)
        put(p + "self_attn.k_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
        put(p + "self_attn.v_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
        put(p + "self_attn.o_proj.weight", linear(HIDDEN, HEADS * HD), HIDDEN, HEADS * HD)
        put(p + "self_attn.q_norm.weight", gain(HD), HD)
        put(p + "self_attn.k_norm.weight", gain(HD), HD)
        put(p + "mlp.gate_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
        put(p + "mlp.up_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
        put(p + "mlp.down_proj.weight", linear(HIDDEN, INTER), HIDDEN, INTER)
    }

    private fun writeTarget(dir: Path) {
        val t = Tensors()
        t.put("model.embed_tokens.weight", linear(VOCAB, HIDDEN), VOCAB, HIDDEN)
        t.put("lm_head.weight", linear(VOCAB, HIDDEN), VOCAB, HIDDEN)
        t.put("model.norm.weight", gain(HIDDEN), HIDDEN)
        for (l in 0 until LAYERS) {
            val p = "model.layers.$l."
            t.attention(p)
            t.put(p + "self_attn.g_proj.weight", linear(HEADS * HD, HIDDEN), HEADS * HD, HIDDEN)
            t.put(p + "self_attn.value_res_lambda", floatArrayOf(0f), 1)
            for (n in listOf("input", "post_attention", "pre_feedforward", "post_feedforward")) {
                t.put(p + n + "_layernorm.weight", gain(HIDDEN), HIDDEN)
            }
        }
        SafetensorsFileWriter.write(dir.resolve("model.safetensors"), t.all)
        Files.writeString(
            dir.resolve("config.json"),
            """
            {"architectures": ["RicursiveForCausalLM"], "model_type": "ricursive", "dtype": "float32",
             "vocab_size": $VOCAB, "hidden_size": $HIDDEN, "intermediate_size": $INTER, "num_hidden_layers": $LAYERS,
             "num_attention_heads": $HEADS, "num_key_value_heads": $KV, "head_dim": $HD,
             "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "sliding_window": 8,
             "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
             "hybrid_layer_pattern": [1, 1, 1, 1, 0], "is_hybrid_swa": true, "hidden_act": "silu",
             "use_gated_attention": true, "gate_full_precision": true, "use_valres": true, "ffn_clamp_threshold": 20.0,
             "embed_scale": 2.5, "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": [2],
             "pad_token_id": 0, "use_cache": true}
            """.trimIndent(),
        )
    }

    private fun writeDraft(dir: Path) {
        val t = Tensors()
        val groups = HIDDEN / GROUP
        t.put("fc.weight", linear(HIDDEN, 2 * HIDDEN), HIDDEN, 2 * HIDDEN)
        t.put("hidden_norm.weight", gain(HIDDEN), HIDDEN)
        t.put("norm.weight", gain(HIDDEN), HIDDEN)
        t.put("candidate_selector.hidden_projection.weight", linear(RANK, HIDDEN), RANK, HIDDEN)
        t.put("candidate_selector.predecessor_codebook", linear(VOCAB, RANK), VOCAB, RANK)
        t.put("candidate_selector.successor_codebook", linear(VOCAB, RANK), VOCAB, RANK)
        for (l in 0 until 3) {
            val p = "layers.$l."
            t.attention(p)
            t.put(p + "input_layernorm.weight", gain(HIDDEN), HIDDEN)
            t.put(p + "post_attention_layernorm.weight", gain(HIDDEN), HIDDEN)
            for (c in listOf("attention_conv", "mlp_conv")) {
                t.put("$p$c.base_kernel", linear(2 * TAPS, HIDDEN), 2, TAPS, HIDDEN)
                t.put("$p$c.kernel_projection.weight", linear(2 * TAPS * groups, HIDDEN), 2 * TAPS * groups, HIDDEN)
            }
        }
        SafetensorsFileWriter.write(dir.resolve("model.safetensors"), t.all)
        Files.writeString(
            dir.resolve("config.json"),
            """
            {"architectures": ["DFlash2DraftModel"], "attention_bias": false, "is_causal": false, "model_type": "qwen3",
             "dflash_config": {"block_size": 4, "conv_group_size": $GROUP, "conv_kernel_size": $TAPS, "mask_token_id": 60,
                               "selector_rank": $RANK, "selector_top_k": 4, "target_layer_ids": [1, 4]},
             "draft_vocab_size": $VOCAB, "head_dim": $HD, "hidden_act": "silu", "hidden_size": $HIDDEN,
             "intermediate_size": $INTER, "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
             "no_rope_layers": [2], "num_attention_heads": $HEADS, "num_hidden_layers": 3,
             "num_key_value_heads": $KV, "num_target_layers": 5, "rope_theta": 10000.0, "sliding_window": 8,
             "tie_word_embeddings": false, "vocab_size": $VOCAB}
            """.trimIndent(),
        )
    }

    @Test
    fun anNvfp4TargetWithABf16DraftExports() {
        val dir = Files.createTempDirectory("ripple-artifact")
        try {
            val target = dir.resolve("target").also { Files.createDirectories(it) }
            val draft = dir.resolve("draft").also { Files.createDirectories(it) }
            writeTarget(target)
            writeDraft(draft)
            val art = dir.resolve("artifact")
            val manifest = HfCheckpoint.open(target, draftDir = draft).use { ckpt ->
                HfServingExport.export(
                    ckpt = ckpt, dir = art,
                    config = ckpt.config.copy(
                        weightDType = BF16, weightQuant = WeightQuant.NVFP4, attentionQuant = WeightQuant.NONE, mtpDraftTokens = 3,
                    ),
                    policy = DecodeBucketPolicy(maxBatch = 1, maxContext = 32, blockSize = 4, minContext = 32),
                    numBlocks = 16,
                    prefillChunk = 8,
                    cudaKernels = true,
                    modelName = "tiny-ripple",
                )
            }
            assertEquals(ServingManifest.SCHEMA_VERSION_5, manifest.schemaVersion)
            assertEquals(null, manifest.model.linearState)
            // The draft policy the backend reads instead of its MTP defaults, and it reads back.
            assertEquals(ServingDraftPolicy(0.02, 16), manifest.model.draftPolicy)
            assertEquals(manifest.model, ServingModelShape.fromJson(io.tlaloc.core.io.parseJson(manifest.model.toJson()) as io.tlaloc.core.io.JsonObject))
            // The draft's sliding layers keep a ring after the target's (window class 1): its window
            // of 8, a prefill call of 8 and the 4 query rows, in pages of 4, for the two sequences the 16
            // full-history pages hold; its full layer does not.
            assertEquals(listOf(ServingWindowedKv(8, listOf(0, 1), 11, 5, trailingRows = 4)), manifest.model.draftWindowedKv)
            for (e in manifest.entries) {
                val draftPools = e.inputs.filter { it.name.matches(Regex("draft(Key|Value)Cache[01]")) }
                assertEquals(listOf(1, 1, 1, 1), draftPools.map { it.windowClass }, e.entryId)
                assertTrue(draftPools.all { it.role == io.tlaloc.ir.inference.DecodeSlotRole.WINDOW_KV_POOL_IN && it.type.dims[0] == 11 }, e.entryId)
                assertEquals(io.tlaloc.ir.inference.DecodeSlotRole.KV_POOL_IN, e.inputs.single { it.name == "draftKeyCache2" }.role)
                assertEquals(1, e.inputs.single { it.name == "draftWindowBlockTables0" }.windowClass)
                assertEquals(e.inputs, e.inputs.map { ServingSlot.fromJson(io.tlaloc.core.io.parseJson(it.toJson()) as io.tlaloc.core.io.JsonObject) })
            }
            val weights = manifest.weights.table.map { it.name }
            assertEquals(weights.size, weights.toSet().size)
            assertTrue(weights.any { "Draft" in it }, weights.toString())

            val model = TritonModelRepository.write(
                art, dir.resolve("repository"), "tiny-ripple", TritonModelRepository.KvMode.SEQUENCE,
                TritonModelRepository.SequenceOptions(decoupled = true),
            )
            val config = Files.readString(model.resolve("config.pbtxt"))
            // 15 pages of 4 tokens hold one sequence of the largest context, 32 tokens.
            assertTrue("max_candidate_sequences: 1\n" in config, config)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    private companion object {
        const val VOCAB = 64
        const val HIDDEN = 64
        const val INTER = 128
        const val LAYERS = 5
        const val HEADS = 6
        const val KV = 3
        const val HD = 8
        const val GROUP = 4
        const val TAPS = 4
        const val RANK = 5
    }
}
