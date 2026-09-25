package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.recognizer.quant.KvQuantConfig

// The layer between a HuggingFace decoder-only checkpoint and the tensor roles
// the decode graph needs. Pure: no file is opened here (that is HfCheckpoint,
// jvmMain).
//
//   1. [HfModelFamily]    - what a family's config.json may say, and how each
//                           of its layers is built. Llama, Qwen3 and Muse
//                           Glimmer (text only) today.
//   2. [HfDecoderConfig]  - config.json, read through :core's strict parseJson,
//                           into the numbers and per-layer specs a graph is
//                           built from. A key the family does not know is
//                           refused by name.
//   3. [HfDecoderNames]   - the HF parameter-name <-> [DecoderWeightRole]
//                           bijection, plus the dims every role must have.
//
// HuggingFace stores every nn.Linear weight as [out_features, in_features],
// because F.linear(x, W) computes x @ W.T. [HfDecoderNames.expectedDims]
// states that layout, and HfCheckpointTest checks it against a real
// checkpoint with rectangular GQA projections and MLP. The embedding table is
// a lookup [vocab, hidden], not a Linear; lm_head is a Linear of the same
// dims, which is why a tied checkpoint can reuse the table for it.

/** Which non-layer tensor, or which part of which layer, a checkpoint entry is. */
sealed interface DecoderWeightRole {
    /** `model.embed_tokens.weight`, `[vocabSize, hiddenSize]`. */
    data object EmbedTokens : DecoderWeightRole

    /** `model.norm.weight`, the final RMSNorm gain, `[hiddenSize]`. */
    data object FinalNorm : DecoderWeightRole

    /**
     * `lm_head.weight`, `[vocabSize, hiddenSize]`. Under tied embeddings this
     * role reads [EmbedTokens]'s tensor; see [HfDecoderConfig.tieWordEmbeddings].
     */
    data object LmHead : DecoderWeightRole

    /** One tensor inside `model.layers.$layer`. */
    data class Layer(val layer: Int, val part: DecoderLayerPart) : DecoderWeightRole {
        init {
            require(layer >= 0) { "DecoderWeightRole.Layer: layer index must be >= 0, got $layer" }
        }
    }
}

/**
 * The tensors a decoder layer can carry, named by role. Which of them a layer
 * has is [DecoderLayerSpec.parts]; the HF spelling of each is
 * [HfModelFamily.leaf].
 */
enum class DecoderLayerPart {
    Q_PROJ, K_PROJ, V_PROJ, O_PROJ,
    GATE_PROJ, UP_PROJ, DOWN_PROJ,
    INPUT_LAYERNORM, POST_ATTENTION_LAYERNORM,

    /** Per-head RMSNorm gain on the queries, `[headDim]`, applied before RoPE (Qwen3). */
    Q_NORM,

    /** Per-head RMSNorm gain on the keys, `[headDim]`, applied before RoPE (Qwen3). */
    K_NORM,

    /**
     * The attention output gate, `[numHeads * headDim, hiddenSize]`: the
     * attention output is multiplied by `sigmoid(x @ gate^T)` before o_proj,
     * where `x` is the layer's normalized input (Muse Glimmer).
     */
    ATTN_GATE_PROJ,

    /**
     * RMSNorm gain on the attention output, before its residual add
     * (Gemma-style families; Muse Glimmer spells it `post_attention_layernorm`).
     */
    ATTENTION_OUTPUT_NORM,

    /** RMSNorm gain on the MLP output, before its residual add (`post_feedforward_layernorm`). */
    FEEDFORWARD_OUTPUT_NORM,
    ;

    /** True for the RMSNorm gains, which are rank-1 and not transposed. */
    val isNorm: Boolean
        get() = this == INPUT_LAYERNORM || this == POST_ATTENTION_LAYERNORM ||
            this == Q_NORM || this == K_NORM ||
            this == ATTENTION_OUTPUT_NORM || this == FEEDFORWARD_OUTPUT_NORM
}

/** How a layer's attention sees the context. */
enum class AttentionKind {
    /** Every earlier position (causal). */
    FULL,

    /** Only the last [DecoderLayerSpec.slidingWindow] positions. */
    SLIDING,
}

/**
 * What one decoder layer computes, beyond what every layer shares (RMSNorm,
 * q/k/v/o projections, paged attention, a gated SiLU MLP, two residual adds).
 *
 * The fields are the per-layer differences between the families this repo
 * reads. The decode graph implements every one of them.
 */
data class DecoderLayerSpec(
    val attention: AttentionKind = AttentionKind.FULL,
    /**
     * The window of a [AttentionKind.SLIDING] layer, in positions: a query
     * at position `p` sees the positions `p - window + 1 .. p`, as
     * transformers' sliding-window mask does.
     */
    val slidingWindow: Int? = null,
    /** False for a layer that applies no rotary embedding (NoPE). */
    val rope: Boolean = true,
    /** Per-head RMSNorm on q and k before RoPE, as Qwen3 and Muse Glimmer do. */
    val qkNorm: Boolean = false,
    /** An RMSNorm on the attention output before its residual add, as Gemma 2 and 3 do. */
    val postAttentionOutputNorm: Boolean = false,
    /** An RMSNorm on the MLP output before its residual add, as Gemma 2 and 3 do. */
    val postFeedforwardNorm: Boolean = false,
    /**
     * Whether the q/k norms carry a learned gain ([DecoderLayerPart.Q_NORM],
     * [DecoderLayerPart.K_NORM]). Muse Glimmer's have none.
     */
    val qkNormGain: Boolean = true,
    /** The attention output is gated by `sigmoid` of a projection of the layer input (Muse Glimmer). */
    val attentionOutputGate: Boolean = false,
) {
    init {
        require((attention == AttentionKind.SLIDING) == (slidingWindow != null)) {
            "DecoderLayerSpec: a sliding layer needs a window and a full one has none " +
                "(attention=$attention, slidingWindow=$slidingWindow)"
        }
        require(slidingWindow == null || slidingWindow >= 1) {
            "DecoderLayerSpec: slidingWindow must be >= 1, got $slidingWindow"
        }
    }

    /**
     * The layer's weight tensors in the order the decode graph takes them:
     * input norm, q/k/v, the q/k norm gains when [qkNorm] and [qkNormGain],
     * the attention gate when [attentionOutputGate], o, the attention output
     * norm when [postAttentionOutputNorm], the norm before the MLP,
     * gate/up/down, and the MLP output norm when [postFeedforwardNorm].
     * Llama's and Qwen3's orders are unchanged by the optional parts.
     */
    val parts: List<DecoderLayerPart>
        get() = buildList {
            add(DecoderLayerPart.INPUT_LAYERNORM)
            add(DecoderLayerPart.Q_PROJ)
            add(DecoderLayerPart.K_PROJ)
            add(DecoderLayerPart.V_PROJ)
            if (qkNorm && qkNormGain) {
                add(DecoderLayerPart.Q_NORM)
                add(DecoderLayerPart.K_NORM)
            }
            if (attentionOutputGate) add(DecoderLayerPart.ATTN_GATE_PROJ)
            add(DecoderLayerPart.O_PROJ)
            if (postAttentionOutputNorm) add(DecoderLayerPart.ATTENTION_OUTPUT_NORM)
            add(DecoderLayerPart.POST_ATTENTION_LAYERNORM)
            add(DecoderLayerPart.GATE_PROJ)
            add(DecoderLayerPart.UP_PROJ)
            add(DecoderLayerPart.DOWN_PROJ)
            if (postFeedforwardNorm) add(DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM)
        }

    /**
     * The settings of this layer the decode graph does not implement, by
     * name. Empty: every setting above is implemented. Kept so a setting added
     * here before its graph code is refused rather than dropped.
     */
    fun unsupported(): List<String> = emptyList()
}

/**
 * A family of HuggingFace decoder-only checkpoints: which `architectures`
 * name it, which `config.json` keys it may carry, how each layer is built,
 * and how its tensors are spelled.
 *
 * Adding a family means adding an object here: its architectures, the keys
 * its parser reads beyond the shared ones ([extraKeys]), [layers], which
 * turns its config into one [DecoderLayerSpec] per layer, and [refine] for
 * the model-wide numerics beyond Llama's (norm conventions, scale factors,
 * soft-capping). A multimodal checkpoint whose decoder config sits under a
 * key of its own names that key in [textConfigKey].
 */
sealed class HfModelFamily(
    /** Short name used in messages and in serving model hashes. */
    val id: String,
    /** `architectures[0]` values this family covers. */
    val architectures: Set<String>,
    /** `model_type` values this family covers. */
    val modelTypes: Set<String>,
) {
    /** Keys this family reads in addition to [SHARED_KEYS]. */
    open val extraKeys: Set<String> = emptySet()

    /**
     * The key of the decoder's own config inside `config.json`, for a
     * multimodal checkpoint (`"text_config"`), or null when the file is the
     * decoder config itself.
     */
    open val textConfigKey: String? = null

    /** Keys the OUTER config may carry when [textConfigKey] is set; any other is refused. */
    open val outerKeys: Set<String> = emptySet()

    /** The prefix of the decoder's tensors in the checkpoint: `model.` or `model.language_model.`. */
    open val modelPrefix: String = "model."

    /**
     * The dtype the decode graph stages this family's weights in. F32 for
     * the families small enough to widen; BF16 where the f32 copy would not
     * fit the device.
     */
    open val defaultWeightDType: DType = F32

    /**
     * The config after the family's model-wide settings are read from
     * [root] (the decoder config) and [outer] (the file, which is [root] for
     * a flat config). The default reads nothing.
     */
    open fun refine(root: JsonObject, outer: JsonObject, config: HfDecoderConfig): HfDecoderConfig = config

    /** The layer every layer is when the config states no per-layer differences. */
    abstract val defaultLayer: DecoderLayerSpec

    /** One spec per layer, from the config's own keys. */
    open fun layers(root: JsonObject, numLayers: Int): List<DecoderLayerSpec> =
        List(numLayers) { defaultLayer }

    /** The HF leaf spelling of a layer part, under `model.layers.N.`. */
    open fun leaf(part: DecoderLayerPart): String = when (part) {
        DecoderLayerPart.Q_PROJ -> "self_attn.q_proj.weight"
        DecoderLayerPart.K_PROJ -> "self_attn.k_proj.weight"
        DecoderLayerPart.V_PROJ -> "self_attn.v_proj.weight"
        DecoderLayerPart.O_PROJ -> "self_attn.o_proj.weight"
        DecoderLayerPart.GATE_PROJ -> "mlp.gate_proj.weight"
        DecoderLayerPart.UP_PROJ -> "mlp.up_proj.weight"
        DecoderLayerPart.DOWN_PROJ -> "mlp.down_proj.weight"
        DecoderLayerPart.INPUT_LAYERNORM -> "input_layernorm.weight"
        DecoderLayerPart.POST_ATTENTION_LAYERNORM -> "post_attention_layernorm.weight"
        DecoderLayerPart.Q_NORM -> "self_attn.q_norm.weight"
        DecoderLayerPart.K_NORM -> "self_attn.k_norm.weight"
        DecoderLayerPart.ATTN_GATE_PROJ -> "self_attn.gate_proj.weight"
        DecoderLayerPart.ATTENTION_OUTPUT_NORM -> "post_attention_output_norm.weight"
        DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM -> "post_feedforward_layernorm.weight"
    }

    /** Every key this family's config may carry: read, or known not to change the forward pass. */
    val knownKeys: Set<String> get() = SHARED_KEYS + INERT_KEYS + extraKeys

    final override fun toString(): String = id

    /** `LlamaForCausalLM`: every layer is full attention with RoPE and no q/k norm. */
    data object Llama : HfModelFamily("llama", setOf("LlamaForCausalLM"), setOf("llama")) {
        override val defaultLayer: DecoderLayerSpec = DecoderLayerSpec()
    }

    /**
     * `Qwen3ForCausalLM`: Llama's layer plus a per-head RMSNorm on q and k
     * before RoPE. Sliding-window layers are read from `use_sliding_window`,
     * `sliding_window`, `max_window_layers` and `layer_types` the way
     * transformers' `Qwen3Config` reads them, and refused by the graph.
     */
    data object Qwen3 : HfModelFamily("qwen3", setOf("Qwen3ForCausalLM"), setOf("qwen3")) {
        override val extraKeys: Set<String> =
            setOf("use_sliding_window", "sliding_window", "max_window_layers", "layer_types")
        override val defaultLayer: DecoderLayerSpec = DecoderLayerSpec(qkNorm = true)

        override fun layers(root: JsonObject, numLayers: Int): List<DecoderLayerSpec> {
            val useSliding = (root["use_sliding_window"] as? JsonBool)?.value ?: false
            val window = if (useSliding) root.optIntKey("sliding_window") else null
            val maxWindowLayers = root.optIntKey("max_window_layers") ?: numLayers
            val types = layerTypes(root, numLayers) ?: List(numLayers) { l ->
                if (window != null && l >= maxWindowLayers) SLIDING_ATTENTION else FULL_ATTENTION
            }
            return types.mapIndexed { l, t ->
                when (t) {
                    FULL_ATTENTION -> defaultLayer
                    SLIDING_ATTENTION -> defaultLayer.copy(
                        attention = AttentionKind.SLIDING,
                        slidingWindow = window ?: throw JsonException(
                            "HfDecoderConfig: layer_types[$l] is '$SLIDING_ATTENTION' but the " +
                                "config states no sliding_window (or use_sliding_window is false)",
                        ),
                    )
                    else -> throw JsonException(
                        "HfDecoderConfig: layer_types[$l] = '$t' is not one of " +
                            "'$FULL_ATTENTION' or '$SLIDING_ATTENTION'",
                    )
                }
            }
        }
    }

    /**
     * `MuseGlimmerForConditionalGeneration`, text only: the decoder under
     * `text_config`, tensors under `model.language_model.`, and the vision
     * encoder's tensors left unread. Every layer has
     *
     * - (1 + w) RMSNorms: before attention, on the attention output (eps
     *   `post_norm_eps`), before the MLP, and on the MLP output (eps
     *   `post_norm_eps`);
     * - gainless per-head RMSNorms on q and k, q then scaled by
     *   `qk_scale_factor`;
     * - RoPE on the layers whose `layer_rope_theta` is non-zero (with the
     *   global theta, as transformers applies it), none on the others;
     * - sliding-window or full attention from `layer_types`;
     * - the attention output multiplied by `sigmoid(gate_proj(x))`.
     *
     * The embeddings go through a gainless RMSNorm, the final norm is a
     * plain `w` RMSNorm, and the logits are `cap * tanh(z * output_multiplier / cap)`
     * with `cap = final_logit_softcapping`. Weights stay bf16 on the device.
     * The image and video placeholder tokens are refused by name
     * ([HfDecoderConfig.refusedTokenIds]).
     */
    data object MuseGlimmer : HfModelFamily(
        "muse_glimmer", setOf("MuseGlimmerForConditionalGeneration"), setOf("muse_glimmer"),
    ) {
        override val extraKeys: Set<String> = setOf(
            "sliding_window", "layer_types", "layer_rope_theta", "hidden_activation",
            "final_logit_softcapping", "qk_scale_factor", "output_multiplier", "post_norm_eps",
        )
        override val textConfigKey: String = "text_config"
        override val outerKeys: Set<String> = setOf(
            "architectures", "model_type", "dtype", "torch_dtype", "transformers_version",
            "text_config", "vision_config", "image_token_id", "video_token_id",
            "out_hidden_size", "projector_hidden_act", "projector_hidden_size",
        )
        override val modelPrefix: String = "model.language_model."
        override val defaultWeightDType: DType = BF16
        override val defaultLayer: DecoderLayerSpec = DecoderLayerSpec(
            qkNorm = true, qkNormGain = false, attentionOutputGate = true,
            postAttentionOutputNorm = true, postFeedforwardNorm = true,
        )

        override fun leaf(part: DecoderLayerPart): String = when (part) {
            // The norm in front of the MLP; Llama's name for it is Muse's
            // name for the attention output norm.
            DecoderLayerPart.POST_ATTENTION_LAYERNORM -> "pre_feedforward_layernorm.weight"
            DecoderLayerPart.ATTENTION_OUTPUT_NORM -> "post_attention_layernorm.weight"
            else -> super.leaf(part)
        }

        override fun layers(root: JsonObject, numLayers: Int): List<DecoderLayerSpec> {
            // transformers' defaults: every 4th layer counted back from the
            // last is full attention without RoPE, the others sliding with RoPE.
            fun lastOfFour(l: Int) = (numLayers - 1 - l) % 4 == 0
            val types = layerTypes(root, numLayers)
                ?: List(numLayers) { if (lastOfFour(it)) FULL_ATTENTION else SLIDING_ATTENTION }
            val thetas = layerRopeThetas(root, numLayers)
            val window = root.optIntKey("sliding_window") ?: 2048
            return types.mapIndexed { l, t ->
                val rope = thetas?.let { it[l] != 0.0 } ?: !lastOfFour(l)
                when (t) {
                    FULL_ATTENTION -> defaultLayer.copy(rope = rope)
                    SLIDING_ATTENTION -> defaultLayer.copy(
                        attention = AttentionKind.SLIDING, slidingWindow = window, rope = rope,
                    )
                    else -> throw JsonException(
                        "HfDecoderConfig: layer_types[$l] = '$t' is not one of " +
                            "'$FULL_ATTENTION' or '$SLIDING_ATTENTION'",
                    )
                }
            }
        }

        override fun refine(root: JsonObject, outer: JsonObject, config: HfDecoderConfig): HfDecoderConfig {
            // transformers builds one rotary table from the global theta and
            // uses layer_rope_theta only as on/off. A layer theta that is
            // neither 0 nor the global one would mean something else to
            // whoever wrote it, so it is refused rather than guessed at.
            layerRopeThetas(root, config.numLayers)?.forEachIndexed { l, th ->
                if (th != 0.0 && th != config.ropeTheta) {
                    throw JsonException(
                        "HfDecoderConfig: layer_rope_theta[$l] = $th is neither 0 (no RoPE) nor " +
                            "the global rope_theta ${config.ropeTheta}; transformers would apply " +
                            "the global theta to it. Refused by name",
                    )
                }
            }
            val softcap = (root["final_logit_softcapping"] as? JsonNumber)?.value ?: 20.0
            return config.copy(
                finalLogitSoftcap = softcap,
                logitMultiplier = (root["output_multiplier"] as? JsonNumber)?.value ?: 0.19611613513818404,
                queryScale = (root["qk_scale_factor"] as? JsonNumber)?.value ?: 3.87,
                postNormEps = (root["post_norm_eps"] as? JsonNumber)?.value ?: 1e-8,
                embeddingNorm = true,
                layerNormGainPlusOne = true,
                refusedTokenIds = buildMap {
                    outer.optIntKey("image_token_id")?.let { put(it, "image_token_id") }
                    outer.optIntKey("video_token_id")?.let { put(it, "video_token_id") }
                },
            )
        }
    }

    companion object {
        /** Every family this repo reads. */
        val ALL: List<HfModelFamily> get() = listOf(Llama, Qwen3, MuseGlimmer)

        /** The family whose [architectures] contain [architecture], or null. */
        fun forArchitecture(architecture: String): HfModelFamily? =
            ALL.firstOrNull { architecture in it.architectures }

        /** The family whose [modelTypes] contain [modelType], or null. */
        fun forModelType(modelType: String): HfModelFamily? =
            ALL.firstOrNull { modelType in it.modelTypes }

        /** Keys every family reads. */
        val SHARED_KEYS: Set<String> = setOf(
            "architectures", "model_type", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "rms_norm_eps", "rope_theta", "rope_scaling", "rope_parameters",
            "max_position_embeddings", "tie_word_embeddings", "attention_bias", "mlp_bias",
            "hidden_act", "torch_dtype", "dtype",
        )

        /**
         * Keys that do not change an inference forward pass: token ids the
         * tokenizer and sampler use, training-only settings, bookkeeping.
         * `pretraining_tp` splits a matmul into slices whose sum is the same
         * product.
         */
        val INERT_KEYS: Set<String> = setOf(
            "bos_token_id", "eos_token_id", "pad_token_id", "initializer_range", "use_cache",
            "transformers_version", "attention_dropout", "pretraining_tp", "_name_or_path",
        )

        private const val FULL_ATTENTION = "full_attention"
        private const val SLIDING_ATTENTION = "sliding_attention"

        private fun layerTypes(root: JsonObject, numLayers: Int): List<String>? {
            val arr = root["layer_types"] as? JsonArray ?: return null
            if (arr.elements.size != numLayers) {
                throw JsonException(
                    "HfDecoderConfig: layer_types has ${arr.elements.size} entries for " +
                        "$numLayers layers",
                )
            }
            return arr.elements.mapIndexed { i, e ->
                (e as? JsonString)?.value
                    ?: throw JsonException("HfDecoderConfig: layer_types[$i] is not a string")
            }
        }

        private fun layerRopeThetas(root: JsonObject, numLayers: Int): List<Double>? {
            val arr = root["layer_rope_theta"] as? JsonArray ?: return null
            if (arr.elements.size != numLayers) {
                throw JsonException(
                    "HfDecoderConfig: layer_rope_theta has ${arr.elements.size} entries for " +
                        "$numLayers layers",
                )
            }
            return arr.elements.mapIndexed { i, e ->
                (e as? JsonNumber)?.value
                    ?: throw JsonException("HfDecoderConfig: layer_rope_theta[$i] is not a number")
            }
        }

        private fun JsonObject.optIntKey(key: String): Int? =
            (this[key] as? JsonNumber)?.asInt("HfDecoderConfig: $key")
    }
}

/**
 * `config.json`, as far as a decode graph cares.
 *
 * Read through :core's strict [parseJson], which refuses trailing commas, NaN
 * literals and duplicate keys: a checkpoint is untrusted input, and a config
 * that parses leniently into wrong numbers gives a model that runs and is
 * wrong. Every key must be one the [family] knows ([HfModelFamily.knownKeys]);
 * any other key is refused by name at [parse], because a key nobody reads may
 * change the forward pass.
 */
data class HfDecoderConfig(
    /** `architectures[0]`, verbatim, e.g. `"LlamaForCausalLM"`. */
    val architecture: String,
    /** `model_type`, verbatim, e.g. `"llama"`. */
    val modelType: String,
    val hiddenSize: Int,
    val intermediateSize: Int,
    val numLayers: Int,
    val numHeads: Int,
    /**
     * `num_key_value_heads`. Defaults to [numHeads] when absent, which is HF's
     * own rule and means plain MHA.
     */
    val numKvHeads: Int,
    /**
     * `head_dim` when the config states it, otherwise `hiddenSize / numHeads`.
     * Newer configs state it and it is not always the quotient (Qwen3-0.6B:
     * hidden 1024, 16 heads, head_dim 128).
     */
    val headDim: Int,
    val vocabSize: Int,
    val rmsNormEps: Double,
    val ropeTheta: Double,
    val maxPositionEmbeddings: Int,
    /** True when the head reuses the embedding table (`lm_head = embed_tokens`). */
    val tieWordEmbeddings: Boolean,
    /** `attention_bias`: bias vectors on q/k/v/o. Refused by the graph. */
    val attentionBias: Boolean,
    /** `torch_dtype` (or `dtype`) verbatim (`"bfloat16"`, `"float32"`, ...), or null. */
    val torchDtype: String?,
    /**
     * `rope_scaling.rope_type` (or the legacy `type` key, or
     * `rope_parameters.rope_type`) when the config scales RoPE, else null.
     * Kept as a string and refused by name; see [toDecodeModelShape].
     */
    val ropeScalingType: String?,
    /** The family this config was read as. */
    val family: HfModelFamily = HfModelFamily.Llama,
    /** `hidden_act`. The graph implements `"silu"` only. */
    val hiddenAct: String = "silu",
    /** `mlp_bias`: bias vectors on gate/up/down. Refused by the graph. */
    val mlpBias: Boolean = false,
    /**
     * One spec per layer, or null when every layer is [HfModelFamily.defaultLayer].
     * May be longer than [numLayers]: a copy with fewer layers keeps the
     * first [numLayers] entries, which is how a reduced model is made.
     */
    val layers: List<DecoderLayerSpec>? = null,
    /**
     * `final_logit_softcapping`: the logits become `cap * tanh(z / cap)`,
     * where `z` is the head's output times [logitMultiplier].
     */
    val finalLogitSoftcap: Double? = null,
    /** Gemma-style `attn_logit_softcapping`. Refused by the graph. */
    val attnLogitSoftcap: Double? = null,
    /**
     * The dtype the weights are staged in on the device. F32 widens a bf16
     * checkpoint exactly; BF16 keeps it as stored, and every projection then
     * rounds its f32 input to bf16 and accumulates in f32 (see [HfDecoderGraph]).
     */
    val weightDType: DType = family.defaultWeightDType,
    /** The embedding rows go through a gainless RMSNorm (eps [rmsNormEps]) before the first layer. */
    val embeddingNorm: Boolean = false,
    /** The layer RMSNorms multiply by `1 + w` rather than `w` (the final norm keeps `w`). */
    val layerNormGainPlusOne: Boolean = false,
    /** The eps of the attention and MLP output norms, when it is not [rmsNormEps]. */
    val postNormEps: Double? = null,
    /** A factor the normalized queries are multiplied by, before RoPE (`qk_scale_factor`). */
    val queryScale: Double = 1.0,
    /** A factor the head's output is multiplied by before soft-capping (`output_multiplier`). */
    val logitMultiplier: Double = 1.0,
    /**
     * Token ids a text-only graph must not be fed, with the config key that
     * names each: a multimodal checkpoint's image and video placeholders,
     * whose rows transformers replaces with vision features. See
     * [checkTextOnlyTokens].
     */
    val refusedTokenIds: Map<Int, String> = emptyMap(),
) {
    init {
        require(hiddenSize >= 1 && intermediateSize >= 1) {
            "HfDecoderConfig: hidden_size/intermediate_size must be >= 1, got $hiddenSize/$intermediateSize"
        }
        require(numLayers >= 1) { "HfDecoderConfig: num_hidden_layers must be >= 1, got $numLayers" }
        require(numHeads >= 1) { "HfDecoderConfig: num_attention_heads must be >= 1, got $numHeads" }
        require(headDim >= 1) { "HfDecoderConfig: head_dim must be >= 1, got $headDim" }
        require(vocabSize >= 1) { "HfDecoderConfig: vocab_size must be >= 1, got $vocabSize" }
        require(numKvHeads in 1..numHeads && numHeads % numKvHeads == 0) {
            "HfDecoderConfig: num_key_value_heads $numKvHeads must divide num_attention_heads " +
                "$numHeads (GQA groups every $numKvHeads-th query head onto one KV head)"
        }
        require(layers == null || layers.size >= numLayers) {
            "HfDecoderConfig: ${layers?.size} layer specs for $numLayers layers"
        }
        require(weightDType == F32 || weightDType == BF16) {
            "HfDecoderConfig: weights are staged as F32 or BF16, not $weightDType"
        }
        require(finalLogitSoftcap == null || finalLogitSoftcap > 0.0) {
            "HfDecoderConfig: final_logit_softcapping must be > 0, got $finalLogitSoftcap"
        }
    }

    /**
     * Refuse by name any id in [ids] that [refusedTokenIds] lists: a text-only
     * graph would embed the placeholder's own row where transformers puts
     * vision features, and serve a different model.
     */
    fun checkTextOnlyTokens(ids: IntArray) {
        for ((i, id) in ids.withIndex()) {
            val key = refusedTokenIds[id] ?: continue
            throw JsonException(
                "HfDecoderConfig ($family): token $i is $id, the $key placeholder; this graph " +
                    "is text only (the vision encoder is not read), so it is refused by name",
            )
        }
    }

    /** The eps of the attention and MLP output norms. */
    val outputNormEps: Double get() = postNormEps ?: rmsNormEps

    /** The spec of layer [l]. */
    fun layer(l: Int): DecoderLayerSpec {
        require(l in 0 until numLayers) { "HfDecoderConfig.layer: $l is outside 0..${numLayers - 1}" }
        return layers?.get(l) ?: family.defaultLayer
    }

    /** `numHeads * headDim`: the width q_proj projects up to and o_proj projects down from. */
    val qProjOut: Int get() = numHeads * headDim

    /** `numKvHeads * headDim`: narrower than [qProjOut] under GQA. */
    val kvProjOut: Int get() = numKvHeads * headDim

    /** The GQA fan-out: how many query heads share one KV head. */
    val gqaGroup: Int get() = numHeads / numKvHeads

    /** True when q/k/v all project to the same width, i.e. plain MHA. */
    val isMultiHead: Boolean get() = numKvHeads == numHeads

    /** [torchDtype] mapped to a Tlaloc [DType], or null when unstated/unmapped. */
    val storageDType: DType?
        get() = when (torchDtype) {
            "bfloat16" -> BF16
            "float32", "float" -> F32
            else -> null
        }

    /**
     * Everything this config asks for that the decode graph does not
     * implement, one phrase per item, naming the config key. Empty when the
     * graph can be built.
     */
    fun unsupportedFeatures(): List<String> = buildList {
        if (ropeScalingType != null) {
            add(
                "rope_scaling '$ropeScalingType' (the graph implements plain theta=$ropeTheta " +
                    "RoPE; llama3/linear/dynamic/yarn scaling is a per-family formula and is not " +
                    "supported)",
            )
        }
        if (attentionBias) {
            add(
                "attention_bias=true (q/k/v/o carry bias vectors the name mapping has no roles " +
                    "for; Llama and Qwen3 have none, Qwen2 does)",
            )
        }
        if (mlpBias) add("mlp_bias=true (gate/up/down carry bias vectors)")
        if (hiddenAct != "silu") add("hidden_act '$hiddenAct' (the MLP is SwiGLU with SiLU)")
        if (attnLogitSoftcap != null) add("attn_logit_softcapping $attnLogitSoftcap")
        for (l in 0 until numLayers) {
            for (u in layer(l).unsupported()) add("layer $l: $u")
        }
    }

    /**
     * The shape record a decode graph is compiled against. The pool geometry
     * ([numBlocks], [blockSize]) is a serving decision, not a checkpoint fact.
     *
     * Refuses by name everything in [unsupportedFeatures]: each changes the
     * arithmetic of the graph and none is implemented, so dropping it would
     * give a model that serves and disagrees with its own reference.
     */
    fun toDecodeModelShape(
        numBlocks: Int,
        blockSize: Int,
        dtype: DType = F32,
        kvQuant: KvQuantConfig? = null,
    ): DecodeModelShape {
        val unsupported = unsupportedFeatures()
        if (unsupported.isNotEmpty()) {
            throw JsonException(
                "HfDecoderConfig ($family): refused BY NAME, the decode graph does not " +
                    "implement: " + unsupported.distinct().take(8).joinToString("; ") +
                    if (unsupported.size > 8) "; and ${unsupported.size - 8} more" else "",
            )
        }
        return DecodeModelShape(
            vocabSize = vocabSize,
            hiddenSize = hiddenSize,
            numHeads = numHeads,
            numKvHeads = numKvHeads,
            headDim = headDim,
            numLayers = numLayers,
            numBlocks = numBlocks,
            blockSize = blockSize,
            dtype = dtype,
            kvDtype = if (kvQuant != null) io.tlaloc.core.I32 else dtype,
            kvQuant = kvQuant,
        )
    }

    companion object {
        /** Architectures some family claims. */
        val SUPPORTED_ARCHITECTURES: Set<String>
            get() = HfModelFamily.ALL.flatMap { it.architectures }.toSet()

        /**
         * Parse a `config.json`.
         *
         * The family comes from `architectures[0]`. [strictArchitecture] false
         * lets a caller read a config whose `architectures` names no family
         * (tiny-random test models, Llama-shaped derivatives); the family is
         * then taken from `model_type`, or Llama. The default refuses by
         * name, because a config that says `MixtralForCausalLM` has experts
         * no family has roles for. The key check applies either way.
         */
        fun parse(json: String, strictArchitecture: Boolean = true): HfDecoderConfig {
            val outer = parseJson(json) as? JsonObject
                ?: throw JsonException("HfDecoderConfig: config.json is not a JSON object")

            val arch = (outer["architectures"] as? JsonArray)
                ?.elements?.firstOrNull()
                ?.let { (it as? JsonString)?.value }
                ?: "<unstated>"
            val modelType = (outer["model_type"] as? JsonString)?.value ?: "<unstated>"
            val family = HfModelFamily.forArchitecture(arch)
                ?: if (strictArchitecture) {
                    throw JsonException(
                        "HfDecoderConfig: architectures[0] = '$arch' is not one of " +
                            "$SUPPORTED_ARCHITECTURES. Pass strictArchitecture=false only when " +
                            "you have checked that the parameter names and the arithmetic " +
                            "really are one of these families'",
                    )
                } else {
                    HfModelFamily.forModelType(modelType) ?: HfModelFamily.Llama
                }

            // A multimodal file: its own keys are checked against the
            // family's outer set, and the decoder config is the nested object.
            val nestedKey = family.textConfigKey
            val root = if (nestedKey == null) {
                outer
            } else {
                val unknownOuter = outer.fields.keys.filter { it !in family.outerKeys }.sorted()
                if (unknownOuter.isNotEmpty()) {
                    throw JsonException(
                        "HfDecoderConfig: config.json has key(s) " +
                            "${unknownOuter.joinToString { "'$it'" }} that the $family family " +
                            "neither reads nor knows to be inert. Refused by name",
                    )
                }
                outer[nestedKey] as? JsonObject
                    ?: throw JsonException("HfDecoderConfig: config.json has no '$nestedKey' object")
            }
            val unknown = root.fields.keys.filter { it !in family.knownKeys }.sorted()
            if (unknown.isNotEmpty()) {
                throw JsonException(
                    "HfDecoderConfig: config.json has key(s) ${unknown.joinToString { "'$it'" }} " +
                        "that the $family family neither reads nor knows to be inert. Refused " +
                        "by name: an unread key can change the forward pass (a window, a " +
                        "scaling, a cap) and the model would serve and be wrong",
                )
            }

            val hidden = root.reqInt("hidden_size")
            val heads = root.reqInt("num_attention_heads")
            val numLayers = root.reqInt("num_hidden_layers")
            val ropeParams = root["rope_parameters"] as? JsonObject
            return HfDecoderConfig(
                architecture = arch,
                modelType = modelType,
                hiddenSize = hidden,
                intermediateSize = root.reqInt("intermediate_size"),
                numLayers = numLayers,
                numHeads = heads,
                numKvHeads = root.optInt("num_key_value_heads") ?: heads,
                headDim = root.optInt("head_dim") ?: run {
                    if (hidden % heads != 0) {
                        throw JsonException(
                            "HfDecoderConfig: config states no head_dim and hidden_size $hidden " +
                                "is not divisible by num_attention_heads $heads, so there is no " +
                                "quotient to fall back to",
                        )
                    }
                    hidden / heads
                },
                vocabSize = root.reqInt("vocab_size"),
                rmsNormEps = root.optDouble("rms_norm_eps") ?: 1e-6,
                ropeTheta = root.optDouble("rope_theta")
                    ?: (ropeParams?.get("rope_theta") as? JsonNumber)?.value
                    ?: 10000.0,
                maxPositionEmbeddings = root.optInt("max_position_embeddings") ?: 2048,
                tieWordEmbeddings = (root["tie_word_embeddings"] as? JsonBool)?.value ?: false,
                attentionBias = (root["attention_bias"] as? JsonBool)?.value ?: false,
                torchDtype = (root["torch_dtype"] as? JsonString)?.value
                    ?: (root["dtype"] as? JsonString)?.value
                    ?: (outer["torch_dtype"] as? JsonString)?.value
                    ?: (outer["dtype"] as? JsonString)?.value,
                ropeScalingType = ropeScalingTypeOf(root["rope_scaling"])
                    ?: ropeScalingTypeOf(ropeParams),
                family = family,
                hiddenAct = (root["hidden_act"] as? JsonString)?.value
                    ?: (root["hidden_activation"] as? JsonString)?.value
                    ?: "silu",
                mlpBias = (root["mlp_bias"] as? JsonBool)?.value ?: false,
                layers = family.layers(root, numLayers),
            ).let { family.refine(root, outer, it) }
        }

        /**
         * `rope_scaling` is null, absent, or an object. HF spelled the kind
         * `type` before transformers 4.43 and `rope_type` after; both are read
         * so an old checkpoint is refused as loudly as a new one.
         * `{"rope_type": "default"}` means no scaling.
         */
        private fun ropeScalingTypeOf(v: JsonValue?): String? {
            if (v == null || v == JsonNull) return null
            val o = v as? JsonObject ?: return null
            val kind = (o["rope_type"] as? JsonString)?.value
                ?: (o["type"] as? JsonString)?.value
                ?: "<unnamed>"
            return if (kind == "default") null else kind
        }

        private fun JsonObject.reqInt(key: String): Int =
            (this[key] as? JsonNumber)?.asInt("HfDecoderConfig: $key")
                ?: throw JsonException(
                    "HfDecoderConfig: config.json has no numeric '$key'; it is required to know " +
                        "the model's shape and there is no defensible default for it",
                )

        private fun JsonObject.optInt(key: String): Int? =
            (this[key] as? JsonNumber)?.asInt("HfDecoderConfig: $key")

        private fun JsonObject.optDouble(key: String): Double? = (this[key] as? JsonNumber)?.value
    }
}

/**
 * The HF parameter-name <-> [DecoderWeightRole] mapping, and the dims each
 * role must have. Pure string and integer arithmetic; no file, no tensor.
 */
object HfDecoderNames {

    /** The embedding table's name under the `model.` prefix; see [hfName] for other prefixes. */
    const val EMBED_TOKENS: String = "model.embed_tokens.weight"
    /** The final norm's name under the `model.` prefix; see [hfName] for other prefixes. */
    const val FINAL_NORM: String = "model.norm.weight"
    const val LM_HEAD: String = "lm_head.weight"

    /** The HF leaf spelling of each layer part, under `model.layers.N.`. */
    fun leaf(part: DecoderLayerPart, family: HfModelFamily = HfModelFamily.Llama): String =
        family.leaf(part)

    /** The full HF tensor name for a role. */
    fun hfName(role: DecoderWeightRole, family: HfModelFamily = HfModelFamily.Llama): String =
        when (role) {
            DecoderWeightRole.EmbedTokens -> "${family.modelPrefix}embed_tokens.weight"
            DecoderWeightRole.FinalNorm -> "${family.modelPrefix}norm.weight"
            DecoderWeightRole.LmHead -> LM_HEAD
            is DecoderWeightRole.Layer -> "${family.modelPrefix}layers.${role.layer}.${family.leaf(role.part)}"
        }

    /**
     * The inverse of [hfName], or null when the name is not one this mapping
     * covers. Null is a fact, not a failure: a checkpoint may carry
     * `model.rotary_emb.inv_freq` (a derived buffer some vintages persist).
     */
    fun role(name: String, family: HfModelFamily = HfModelFamily.Llama): DecoderWeightRole? {
        when (name) {
            hfName(DecoderWeightRole.EmbedTokens, family) -> return DecoderWeightRole.EmbedTokens
            hfName(DecoderWeightRole.FinalNorm, family) -> return DecoderWeightRole.FinalNorm
            LM_HEAD -> return DecoderWeightRole.LmHead
        }
        val layerPrefix = "${family.modelPrefix}layers."
        if (!name.startsWith(layerPrefix)) return null
        val rest = name.substring(layerPrefix.length)
        val dot = rest.indexOf('.')
        if (dot <= 0) return null
        val idx = rest.substring(0, dot).toIntOrNull() ?: return null
        if (idx < 0) return null
        val leaf = rest.substring(dot + 1)
        val part = DecoderLayerPart.entries.firstOrNull { family.leaf(it) == leaf } ?: return null
        return DecoderWeightRole.Layer(idx, part)
    }

    /**
     * Every role a decode graph of this config needs, in the graph's order:
     * embeddings, each layer's [DecoderLayerSpec.parts], the final norm and
     * the head. [DecoderWeightRole.LmHead] is listed under tied embeddings
     * too: the graph needs the role, and where its bytes come from is
     * HfCheckpoint's question.
     */
    fun roles(config: HfDecoderConfig): List<DecoderWeightRole> = buildList {
        add(DecoderWeightRole.EmbedTokens)
        for (l in 0 until config.numLayers) {
            for (p in config.layer(l).parts) add(DecoderWeightRole.Layer(l, p))
        }
        add(DecoderWeightRole.FinalNorm)
        add(DecoderWeightRole.LmHead)
    }

    /**
     * The dims a role's tensor must have in the file, given the config, in
     * HF's `[out_features, in_features]` Linear layout. On a GQA model K/V
     * (`[kvProjOut, hiddenSize]`) and gate/up (`[intermediate, hiddenSize]`)
     * tell that layout from its transpose; on a square model nothing does,
     * which is why the certification runs against real rectangular
     * checkpoints.
     */
    fun expectedDims(role: DecoderWeightRole, config: HfDecoderConfig): IntArray = when (role) {
        DecoderWeightRole.EmbedTokens -> intArrayOf(config.vocabSize, config.hiddenSize)
        DecoderWeightRole.FinalNorm -> intArrayOf(config.hiddenSize)
        DecoderWeightRole.LmHead -> intArrayOf(config.vocabSize, config.hiddenSize)
        is DecoderWeightRole.Layer -> when (role.part) {
            DecoderLayerPart.Q_PROJ -> intArrayOf(config.qProjOut, config.hiddenSize)
            DecoderLayerPart.K_PROJ -> intArrayOf(config.kvProjOut, config.hiddenSize)
            DecoderLayerPart.V_PROJ -> intArrayOf(config.kvProjOut, config.hiddenSize)
            DecoderLayerPart.O_PROJ -> intArrayOf(config.hiddenSize, config.qProjOut)
            DecoderLayerPart.GATE_PROJ -> intArrayOf(config.intermediateSize, config.hiddenSize)
            DecoderLayerPart.UP_PROJ -> intArrayOf(config.intermediateSize, config.hiddenSize)
            DecoderLayerPart.DOWN_PROJ -> intArrayOf(config.hiddenSize, config.intermediateSize)
            DecoderLayerPart.INPUT_LAYERNORM -> intArrayOf(config.hiddenSize)
            DecoderLayerPart.POST_ATTENTION_LAYERNORM -> intArrayOf(config.hiddenSize)
            DecoderLayerPart.Q_NORM -> intArrayOf(config.headDim)
            DecoderLayerPart.K_NORM -> intArrayOf(config.headDim)
            DecoderLayerPart.ATTN_GATE_PROJ -> intArrayOf(config.qProjOut, config.hiddenSize)
            DecoderLayerPart.ATTENTION_OUTPUT_NORM -> intArrayOf(config.hiddenSize)
            DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM -> intArrayOf(config.hiddenSize)
        }
    }

    /**
     * True when this role's file tensor is a Linear weight and therefore
     * stored as `[out, in]`, so a matmul against it needs `x @ W^T` or a
     * transpose at ingestion. False for the embedding table and the norm gains.
     */
    fun isTransposedLinear(role: DecoderWeightRole): Boolean = when (role) {
        DecoderWeightRole.EmbedTokens, DecoderWeightRole.FinalNorm -> false
        DecoderWeightRole.LmHead -> true
        is DecoderWeightRole.Layer -> !role.part.isNorm
    }
}
