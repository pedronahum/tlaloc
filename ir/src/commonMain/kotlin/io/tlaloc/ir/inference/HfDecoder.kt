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

    /**
     * A quantized copy of [LmHead] that only the MTP head's drafts read
     * ([HfDecoderConfig.mtpDraftHeadQuant]): drafts are proposals the target
     * verifies with [LmHead], so their head may round.
     */
    data object DraftHead : DecoderWeightRole

    /** One tensor inside `model.layers.$layer`. */
    data class Layer(val layer: Int, val part: DecoderLayerPart) : DecoderWeightRole {
        init {
            require(layer >= 0) { "DecoderWeightRole.Layer: layer index must be >= 0, got $layer" }
        }
    }

    /** A tensor of the multi-token-prediction head outside its layer (`mtp.fc.weight`, `mtp.norm.weight`, ...). */
    data class Mtp(val part: MtpPart) : DecoderWeightRole

    /** One tensor inside `mtp.layers.0`, the MTP head's decoder layer. */
    data class MtpLayer(val part: DecoderLayerPart) : DecoderWeightRole
}

/** The decoder-layer part of a [DecoderWeightRole.Layer] or [DecoderWeightRole.MtpLayer], else null. */
val DecoderWeightRole.layerPart: DecoderLayerPart?
    get() = when (this) {
        is DecoderWeightRole.Layer -> part
        is DecoderWeightRole.MtpLayer -> part
        else -> null
    }

/** The same layer's role for [part] (a role of [DecoderWeightRole.Layer] or [DecoderWeightRole.MtpLayer]). */
fun DecoderWeightRole.withPart(part: DecoderLayerPart): DecoderWeightRole = when (this) {
    is DecoderWeightRole.Layer -> DecoderWeightRole.Layer(layer, part)
    is DecoderWeightRole.MtpLayer -> DecoderWeightRole.MtpLayer(part)
    else -> throw IllegalArgumentException("DecoderWeightRole.withPart: $this is not a layer's tensor")
}

/**
 * The tensors of a Qwen3.5 multi-token-prediction head outside its layer, as
 * vLLM's `Qwen3_5MultiTokenPredictor` reads them: at position p it predicts
 * the token at p + 2 from the token at p + 1 and the target's final-norm
 * hidden state at p, `fc([pre_fc_norm_embedding(e) | pre_fc_norm_hidden(h)])`
 * through one full-attention layer and `norm`.
 */
enum class MtpPart(val leaf: String) {
    /** `[hidden, 2 * hidden]`: the embedding and the hidden state, concatenated, to the layer's input. */
    FC("fc.weight"),
    PRE_FC_NORM_EMBEDDING("pre_fc_norm_embedding.weight"),
    PRE_FC_NORM_HIDDEN("pre_fc_norm_hidden.weight"),
    NORM("norm.weight"),
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

    // A Gated DeltaNet layer's tensors (Qwen3.5's `linear_attn.*`).

    /** The q/k/v projection, `[2 Hk Dk + Hv Dv, hidden]`, before the conv. */
    IN_PROJ_QKV,

    /** The output gate's projection, `[Hv Dv, hidden]`. */
    IN_PROJ_Z,

    /** The projection to beta's logits, `[Hv, hidden]`. */
    IN_PROJ_B,

    /** The projection to the decay's input, `[Hv, hidden]`. */
    IN_PROJ_A,

    /** The depthwise causal conv's kernel: `[C, 1, K]` in the file, staged `[K, C]`. */
    CONV1D,

    /** `dt_bias`, `[Hv]`, added before the softplus of the decay. */
    DT_BIAS,

    /** `A_log`, `[Hv]`: the decay rate is `exp(A_log)`. Staged f32. */
    A_LOG,

    /** The gated RMSNorm's gain over each value head, `[Dv]` (multiplies by `w`). Staged f32. */
    LINEAR_NORM,

    /** The output projection, `[hidden, Hv Dv]`. */
    OUT_PROJ,

    // A mixture-of-experts MLP's tensors (Qwen3.5-MoE's `mlp.*`).

    /** The router, `[numExperts, hidden]`. */
    ROUTER,

    /** Every expert's gate and up projections, `[numExperts, 2 * expertIntermediate, hidden]`, kept as stored. */
    EXPERTS_GATE_UP,

    /** Every expert's down projection, `[numExperts, hidden, expertIntermediate]`, kept as stored. */
    EXPERTS_DOWN,

    /** The shared expert's gate projection, `[sharedIntermediate, hidden]`. */
    SHARED_GATE_PROJ,

    /** The shared expert's up projection, `[sharedIntermediate, hidden]`. */
    SHARED_UP_PROJ,

    /** The shared expert's down projection, `[hidden, sharedIntermediate]`. */
    SHARED_DOWN_PROJ,

    /** The shared expert's output gate, `[1, hidden]`: its output is scaled by `sigmoid(x · g)`. */
    SHARED_EXPERT_GATE,

    /** The router's selection bias, `[numExperts]` (`e_score_correction_bias`): added to the scores to choose experts only. */
    ROUTER_BIAS,

    // Multi-head latent attention (DeepSeek-V3's `self_attn.*`, see [MlaConfig]).

    /** The query's down projection, `[qLoraRank, hidden]`. */
    Q_A_PROJ,

    /** RMSNorm gain on the query latent, `[qLoraRank]`. */
    Q_A_NORM,

    /** The query's up projection, `[heads * (nope + rope), qLoraRank]`. */
    Q_B_PROJ,

    /** The key/value latent and the shared rope key, `[kvLoraRank + rope, hidden]`. */
    KV_A_PROJ,

    /** RMSNorm gain on the key/value latent, `[kvLoraRank]`. */
    KV_A_NORM,

    /**
     * The latent's up projection to each head's key (nope) and value,
     * `[heads * (nope + v), kvLoraRank]`. Staged as stored: the graph uses
     * each head's blocks to map queries into the latent and outputs out of it.
     */
    KV_B_PROJ,

    // Hyper-connections ([HyperConnectionConfig]): the mixing before the attention and before the MLP.

    /** The attention's mixing weights, `[(2 + n) n, n * hidden]` for n streams. */
    ATTN_HC_FN,

    /** The attention's mixing biases, `[(2 + n) n]`. Staged f32. */
    ATTN_HC_BASE,

    /** The attention's mixing scales (pre, post, comb), `[3]`. Staged f32. */
    ATTN_HC_SCALE,

    /** The MLP's mixing weights, as [ATTN_HC_FN]. */
    FFN_HC_FN,

    /** The MLP's mixing biases, as [ATTN_HC_BASE]. */
    FFN_HC_BASE,

    /** The MLP's mixing scales, as [ATTN_HC_SCALE]. */
    FFN_HC_SCALE,
    ;

    /** True for the stacked expert weights, which are staged as stored (three-dimensional, not transposed). */
    val isExperts: Boolean get() = this == EXPERTS_GATE_UP || this == EXPERTS_DOWN

    /** True for the RMSNorm gains, which are rank-1 and not transposed. */
    val isNorm: Boolean
        get() = this == INPUT_LAYERNORM || this == POST_ATTENTION_LAYERNORM ||
            this == Q_NORM || this == K_NORM || this == Q_A_NORM || this == KV_A_NORM ||
            this == ATTENTION_OUTPUT_NORM || this == FEEDFORWARD_OUTPUT_NORM || this == LINEAR_NORM

    /** True for the rank-1 tensors: the norm gains, `dt_bias`, `A_log` and the router and mixing biases. Never transposed or quantized. */
    val isVector: Boolean get() = isNorm || this == DT_BIAS || this == A_LOG || alwaysF32

    /**
     * True for the tensors staged f32 whatever the weight dtype: `A_log`
     * and the gated norm's gain, which the checkpoints store in f32 and which
     * a bf16 copy would round, and `dt_bias`, which is added to them.
     */
    val alwaysF32: Boolean get() = this == A_LOG || this == LINEAR_NORM || this == DT_BIAS ||
        this == ROUTER_BIAS || this == ATTN_HC_BASE || this == ATTN_HC_SCALE || this == FFN_HC_BASE || this == FFN_HC_SCALE

    /**
     * True for the small projections that read the same input as a quantized
     * group (a Gated DeltaNet layer's b and a, a MoE layer's router and shared
     * expert gate): staged f32 when the projections are quantized, so their
     * matmuls take the f32 input. In bf16 against the same bf16 input, XLA's
     * dot merger joins them to the group and widens its codes into one bf16
     * weight on every call.
     */
    val f32BesideQuantized: Boolean get() =
        this == IN_PROJ_B || this == IN_PROJ_A || this == ROUTER || this == SHARED_EXPERT_GATE
}

/** A layer's MLP. */
enum class MlpKind {
    /** One SwiGLU MLP (gate, up, down). */
    DENSE,

    /** Routed SwiGLU experts plus a gated shared expert ([MoeConfig]). */
    MOE,
}

/** What mixes a layer's tokens. */
enum class TokenMixer {
    /** Paged attention over the KV pool. */
    ATTENTION,

    /** A Gated DeltaNet: a causal conv and a delta-rule recurrence over per-sequence state. */
    GATED_DELTA_NET,

    /** Multi-head latent attention ([MlaConfig]): paged attention over a pool of key/value latents. */
    MLA,
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
    /** What mixes the tokens. A [TokenMixer.GATED_DELTA_NET] layer has no attention settings. */
    val mixer: TokenMixer = TokenMixer.ATTENTION,
    /** The MLP: one SwiGLU, or a mixture of experts. */
    val mlp: MlpKind = MlpKind.DENSE,
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
    /**
     * `q_proj` also produces the output gate: per head `[query | gate]`,
     * `[numHeads * 2 * headDim, hidden]`, and the attention output is
     * multiplied by `sigmoid(gate)` before o_proj (Qwen3.5).
     */
    val queryGate: Boolean = false,
    /** The MoE shared expert's output is gated by `sigmoid(x · g)` ([DecoderLayerPart.SHARED_EXPERT_GATE]). */
    val sharedExpertGate: Boolean = true,
    /** The MoE router has a selection bias ([DecoderLayerPart.ROUTER_BIAS]). */
    val routerBias: Boolean = false,
    /** The layer is wrapped in hyper-connections ([HyperConnectionConfig]). */
    val hyperConnections: Boolean = false,
) {
    init {
        require((attention == AttentionKind.SLIDING) == (slidingWindow != null)) {
            "DecoderLayerSpec: a sliding layer needs a window and a full one has none " +
                "(attention=$attention, slidingWindow=$slidingWindow)"
        }
        require(slidingWindow == null || slidingWindow >= 1) {
            "DecoderLayerSpec: slidingWindow must be >= 1, got $slidingWindow"
        }
        require(!(queryGate && attentionOutputGate)) {
            "DecoderLayerSpec: an attention output gate comes from q_proj or from its own projection, not both"
        }
        require(
            mixer == TokenMixer.ATTENTION || (
                attention == AttentionKind.FULL && !qkNorm && !queryGate && !attentionOutputGate &&
                    !postAttentionOutputNorm
                ),
        ) {
            "DecoderLayerSpec: a $mixer layer has none of the attention settings, got $this"
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
            if (hyperConnections) {
                addAll(
                    listOf(
                        DecoderLayerPart.ATTN_HC_FN, DecoderLayerPart.ATTN_HC_BASE, DecoderLayerPart.ATTN_HC_SCALE,
                        DecoderLayerPart.FFN_HC_FN, DecoderLayerPart.FFN_HC_BASE, DecoderLayerPart.FFN_HC_SCALE,
                    ),
                )
            }
            add(DecoderLayerPart.INPUT_LAYERNORM)
            if (mixer == TokenMixer.MLA) {
                addAll(
                    listOf(
                        DecoderLayerPart.Q_A_PROJ, DecoderLayerPart.Q_A_NORM, DecoderLayerPart.Q_B_PROJ,
                        DecoderLayerPart.KV_A_PROJ, DecoderLayerPart.KV_A_NORM, DecoderLayerPart.KV_B_PROJ,
                        DecoderLayerPart.O_PROJ,
                    ),
                )
                add(DecoderLayerPart.POST_ATTENTION_LAYERNORM)
                addAll(mlpParts)
                return@buildList
            }
            if (mixer == TokenMixer.GATED_DELTA_NET) {
                addAll(
                    listOf(
                        DecoderLayerPart.IN_PROJ_QKV, DecoderLayerPart.IN_PROJ_Z, DecoderLayerPart.IN_PROJ_B,
                        DecoderLayerPart.IN_PROJ_A, DecoderLayerPart.CONV1D, DecoderLayerPart.DT_BIAS,
                        DecoderLayerPart.A_LOG, DecoderLayerPart.LINEAR_NORM, DecoderLayerPart.OUT_PROJ,
                    ),
                )
                add(DecoderLayerPart.POST_ATTENTION_LAYERNORM)
                addAll(mlpParts)
                if (postFeedforwardNorm) add(DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM)
                return@buildList
            }
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
            addAll(mlpParts)
            if (postFeedforwardNorm) add(DecoderLayerPart.FEEDFORWARD_OUTPUT_NORM)
        }

    private val mlpParts: List<DecoderLayerPart>
        get() = when (mlp) {
            MlpKind.DENSE -> listOf(DecoderLayerPart.GATE_PROJ, DecoderLayerPart.UP_PROJ, DecoderLayerPart.DOWN_PROJ)
            MlpKind.MOE -> listOfNotNull(
                DecoderLayerPart.ROUTER, DecoderLayerPart.ROUTER_BIAS.takeIf { routerBias },
                DecoderLayerPart.SHARED_GATE_PROJ, DecoderLayerPart.SHARED_UP_PROJ,
                DecoderLayerPart.SHARED_EXPERT_GATE.takeIf { sharedExpertGate }, DecoderLayerPart.SHARED_DOWN_PROJ,
                DecoderLayerPart.EXPERTS_GATE_UP, DecoderLayerPart.EXPERTS_DOWN,
            )
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
        DecoderLayerPart.IN_PROJ_QKV -> "linear_attn.in_proj_qkv.weight"
        DecoderLayerPart.IN_PROJ_Z -> "linear_attn.in_proj_z.weight"
        DecoderLayerPart.IN_PROJ_B -> "linear_attn.in_proj_b.weight"
        DecoderLayerPart.IN_PROJ_A -> "linear_attn.in_proj_a.weight"
        DecoderLayerPart.CONV1D -> "linear_attn.conv1d.weight"
        DecoderLayerPart.DT_BIAS -> "linear_attn.dt_bias"
        DecoderLayerPart.A_LOG -> "linear_attn.A_log"
        DecoderLayerPart.LINEAR_NORM -> "linear_attn.norm.weight"
        DecoderLayerPart.OUT_PROJ -> "linear_attn.out_proj.weight"
        DecoderLayerPart.ROUTER -> "mlp.gate.weight"
        DecoderLayerPart.EXPERTS_GATE_UP -> "mlp.experts.gate_up_proj"
        DecoderLayerPart.EXPERTS_DOWN -> "mlp.experts.down_proj"
        DecoderLayerPart.SHARED_GATE_PROJ -> "mlp.shared_expert.gate_proj.weight"
        DecoderLayerPart.SHARED_UP_PROJ -> "mlp.shared_expert.up_proj.weight"
        DecoderLayerPart.SHARED_DOWN_PROJ -> "mlp.shared_expert.down_proj.weight"
        DecoderLayerPart.SHARED_EXPERT_GATE -> "mlp.shared_expert_gate.weight"
        DecoderLayerPart.ROUTER_BIAS -> "mlp.gate.e_score_correction_bias"
        DecoderLayerPart.Q_A_PROJ -> "self_attn.q_a_proj.weight"
        DecoderLayerPart.Q_A_NORM -> "self_attn.q_a_layernorm.weight"
        DecoderLayerPart.Q_B_PROJ -> "self_attn.q_b_proj.weight"
        DecoderLayerPart.KV_A_PROJ -> "self_attn.kv_a_proj_with_mqa.weight"
        DecoderLayerPart.KV_A_NORM -> "self_attn.kv_a_layernorm.weight"
        DecoderLayerPart.KV_B_PROJ -> "self_attn.kv_b_proj.weight"
        DecoderLayerPart.ATTN_HC_FN -> "attn_hc.hc_fn"
        DecoderLayerPart.ATTN_HC_BASE -> "attn_hc.hc_base"
        DecoderLayerPart.ATTN_HC_SCALE -> "attn_hc.hc_scale"
        DecoderLayerPart.FFN_HC_FN -> "ffn_hc.hc_fn"
        DecoderLayerPart.FFN_HC_BASE -> "ffn_hc.hc_base"
        DecoderLayerPart.FFN_HC_SCALE -> "ffn_hc.hc_scale"
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

    /**
     * `Qwen3_5ForConditionalGeneration` (`qwen3_5`: Qwen3.5 and Qwen3.6/3.8
     * dense), text only: the decoder under `text_config`, tensors under
     * `model.language_model.`, and the vision encoder and the multi-token
     * prediction head (`mtp.*`) left unread. `layer_types` mixes
     *
     * - `linear_attention`: a Gated DeltaNet ([TokenMixer.GATED_DELTA_NET]);
     * - `full_attention`: attention whose `q_proj` carries the output gate
     *   ([DecoderLayerSpec.queryGate]), with per-head q/k RMSNorms and RoPE on
     *   the first `head_dim * partial_rotary_factor` channels.
     *
     * Every RMSNorm multiplies by `1 + w`, the final one and the q/k ones
     * included; the Gated DeltaNet's gated norm multiplies by `w`. The rotary
     * embedding is mRoPE, whose three position grids are equal for text, so
     * it is plain RoPE here; the image and video placeholder tokens are
     * refused by name. Weights stay bf16 on the device.
     */
    data object Qwen3_5 : HfModelFamily(
        "qwen3_5", setOf("Qwen3_5ForConditionalGeneration"), setOf("qwen3_5"),
    ) {
        override val extraKeys: Set<String> = setOf(
            "layer_types", "full_attention_interval", "attn_output_gate", "linear_conv_kernel_dim",
            "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
            "mamba_ssm_dtype", "mlp_only_layers", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings",
            "output_gate_type", "partial_rotary_factor",
        )
        override val textConfigKey: String = "text_config"
        override val outerKeys: Set<String> = setOf(
            "architectures", "model_type", "dtype", "torch_dtype", "transformers_version", "text_config",
            "vision_config", "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
            "tie_word_embeddings", "language_model_only", "quantization_config",
            // Repeated from text_config by some fine-tunes (Ornith 1.5); hidden_size must agree.
            "bos_token_id", "eos_token_id", "pad_token_id", "hidden_size",
        )
        override val modelPrefix: String = "model.language_model."
        override val defaultWeightDType: DType = BF16
        override val defaultLayer: DecoderLayerSpec = DecoderLayerSpec(qkNorm = true, queryGate = true)

        override fun layers(root: JsonObject, numLayers: Int): List<DecoderLayerSpec> =
            qwen35Layers(root, numLayers, MlpKind.DENSE)

        override fun refine(root: JsonObject, outer: JsonObject, config: HfDecoderConfig): HfDecoderConfig =
            qwen35Refine(root, outer, config)
    }

    /**
     * `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`: Qwen3.5-MoE,
     * Qwen3.6-35B-A3B): [Qwen3_5]'s layers with a mixture of experts for
     * every MLP. Each token goes to its `num_experts_per_tok` experts by
     * router probability, weighted by the renormalized probabilities
     * ([io.tlaloc.ir.OpKind.MOE_EXPERTS]), plus a shared expert scaled by
     * `sigmoid(x · shared_expert_gate)`.
     */
    data object Qwen3_5Moe : HfModelFamily(
        "qwen3_5_moe", setOf("Qwen3_5MoeForConditionalGeneration"), setOf("qwen3_5_moe"),
    ) {
        override val extraKeys: Set<String> = Qwen3_5.extraKeys + setOf(
            "num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
            "router_aux_loss_coef", "output_router_logits",
        )
        override val textConfigKey: String = "text_config"
        override val outerKeys: Set<String> get() = Qwen3_5.outerKeys
        override val modelPrefix: String = "model.language_model."
        override val defaultWeightDType: DType = BF16
        override val defaultLayer: DecoderLayerSpec = DecoderLayerSpec(qkNorm = true, queryGate = true, mlp = MlpKind.MOE)

        override fun intermediateSize(root: JsonObject): Int? = root.optIntKey("shared_expert_intermediate_size")

        override fun layers(root: JsonObject, numLayers: Int): List<DecoderLayerSpec> =
            qwen35Layers(root, numLayers, MlpKind.MOE)

        override fun refine(root: JsonObject, outer: JsonObject, config: HfDecoderConfig): HfDecoderConfig {
            fun req(key: String) = root.optIntKey(key) ?: throw JsonException("HfDecoderConfig: the $id family needs '$key'")
            return qwen35Refine(root, outer, config).copy(
                moe = MoeConfig(
                    numExperts = req("num_experts"),
                    topK = req("num_experts_per_tok"),
                    expertIntermediate = req("moe_intermediate_size"),
                    sharedIntermediate = req("shared_expert_intermediate_size"),
                ),
            )
        }
    }

    /**
     * `Xing4_0ForCausalLM` (Xing 4.0, its own modeling code): DeepSeek-V3's
     * multi-head latent attention with YaRN RoPE on interleaved pairs, four
     * residual streams joined by hyper-connections, and DeepSeek's MoE (a
     * sigmoid router with a selection bias, a shared expert) after
     * `first_k_dense_replace` dense layers. The KV pool holds the latent and
     * the rope key ([MlaConfig.poolDim]) as one KV head. Its MTP layer is not read.
     */
    data object Xing4_0 : HfModelFamily("xing4_0", setOf("Xing4_0ForCausalLM"), setOf("xing4_0")) {
        override val extraKeys: Set<String> = setOf(
            "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
            "moe_intermediate_size", "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
            "routed_scaling_factor", "scoring_func", "topk_method", "n_group", "topk_group", "norm_topk_prob",
            "first_k_dense_replace", "moe_layer_freq", "num_nextn_predict_layers", "hc_mult", "hc_sinkhorn_iters",
            "hc_eps", "mhc_h_res_clamp_min", "mhc_h_res_clamp_max", "rope_interleave",
            // Expert parallelism and the modeling code's location: inert for one device.
            "ep_size", "auto_map",
        )
        override val defaultWeightDType: DType = BF16
        override val defaultLayer: DecoderLayerSpec = DecoderLayerSpec(
            mixer = TokenMixer.MLA, mlp = MlpKind.MOE, sharedExpertGate = false, routerBias = true, hyperConnections = true,
        )

        override fun leaf(part: DecoderLayerPart): String = when (part) {
            DecoderLayerPart.SHARED_GATE_PROJ -> "mlp.shared_experts.gate_proj.weight"
            DecoderLayerPart.SHARED_UP_PROJ -> "mlp.shared_experts.up_proj.weight"
            DecoderLayerPart.SHARED_DOWN_PROJ -> "mlp.shared_experts.down_proj.weight"
            else -> super.leaf(part)
        }

        override fun layers(root: JsonObject, numLayers: Int): List<DecoderLayerSpec> {
            val dense = root.optIntKey("first_k_dense_replace") ?: 0
            return List(numLayers) { if (it < dense) defaultLayer.copy(mlp = MlpKind.DENSE) else defaultLayer }
        }

        override fun refine(root: JsonObject, outer: JsonObject, config: HfDecoderConfig): HfDecoderConfig {
            fun req(key: String) = root.optIntKey(key) ?: throw JsonException("HfDecoderConfig: the $id family needs '$key'")
            fun str(key: String) = (root[key] as? JsonString)?.value
            fun refuse(what: String): Nothing = throw JsonException("HfDecoderConfig: $what. Refused by name")
            if (str("scoring_func") != "sigmoid") refuse("scoring_func '${str("scoring_func")}'; the $id router is a sigmoid")
            if (str("topk_method") != null && str("topk_method") != "noaux_tc") refuse("topk_method '${str("topk_method")}'")
            if ((root.optIntKey("n_group") ?: 1) != 1 || (root.optIntKey("topk_group") ?: 1) != 1) {
                refuse("n_group/topk_group other than 1 (group-limited routing)")
            }
            if ((root["norm_topk_prob"] as? JsonBool)?.value == false) refuse("norm_topk_prob = false")
            if ((root.optIntKey("moe_layer_freq") ?: 1) != 1) refuse("moe_layer_freq other than 1")
            if ((root["rope_interleave"] as? JsonBool)?.value == false) refuse("rope_interleave = false")
            val nope = req("qk_nope_head_dim")
            val ropeDim = req("qk_rope_head_dim")
            val rope = (root["rope_scaling"] as? JsonObject) ?: (root["rope_parameters"] as? JsonObject)
            val (invFreq, mscale) = mlaRope(root, rope, ropeDim, config.ropeTheta, config.maxPositionEmbeddings)
            val mla = MlaConfig(
                qLoraRank = req("q_lora_rank"),
                kvLoraRank = req("kv_lora_rank"),
                nopeDim = nope,
                ropeDim = ropeDim,
                valueDim = req("v_head_dim"),
                invFreq = invFreq,
                scale = Math.pow((nope + ropeDim).toDouble(), -0.5) * mscale * mscale,
            )
            return config.copy(
                numKvHeads = 1,
                headDim = mla.poolDim,
                ropeScalingType = null,
                mla = mla,
                hyper = HyperConnectionConfig(
                    streams = req("hc_mult"),
                    sinkhornIters = req("hc_sinkhorn_iters"),
                    eps = (root["hc_eps"] as? JsonNumber)?.value ?: 1e-6,
                    clampMin = (root["mhc_h_res_clamp_min"] as? JsonNumber)?.value ?: -30.0,
                    clampMax = (root["mhc_h_res_clamp_max"] as? JsonNumber)?.value ?: 30.0,
                ),
                moe = MoeConfig(
                    numExperts = req("n_routed_experts"),
                    topK = req("num_experts_per_tok"),
                    expertIntermediate = req("moe_intermediate_size"),
                    sharedIntermediate = req("moe_intermediate_size") * (root.optIntKey("n_shared_experts") ?: 1),
                    routing = MoeRouting.SIGMOID_BIAS,
                    routedScale = (root["routed_scaling_factor"] as? JsonNumber)?.value ?: 1.0,
                ),
            )
        }

        /**
         * The rope frequencies over [dim] dims and the attention scale's
         * mscale: plain RoPE, or YaRN as transformers computes it
         * (`_compute_yarn_parameters`), whose cos/sin factor must be 1.
         */
        fun mlaRope(root: JsonObject, rope: JsonObject?, dim: Int, base: Double, maxPositions: Int): Pair<List<Double>, Double> {
            val kind = (rope?.get("rope_type") as? JsonString)?.value ?: (rope?.get("type") as? JsonString)?.value ?: "default"
            if (kind == "default") return List(dim / 2) { Math.pow(base, -2.0 * it / dim) } to 1.0
            if (kind != "yarn") throw JsonException("HfDecoderConfig: rope type '$kind'; the $id family reads YaRN or none. Refused by name")
            fun num(key: String) = (rope!![key] as? JsonNumber)?.value
            val factor = num("factor") ?: throw JsonException("HfDecoderConfig: YaRN needs a factor")
            val mscale = num("mscale")
            val mscaleAll = num("mscale_all_dim")
            fun getMscale(scale: Double, m: Double = 1.0) = if (scale <= 1) 1.0 else 0.1 * m * Math.log(scale) + 1.0
            val attentionFactor = if (mscale != null && mscale != 0.0 && mscaleAll != null && mscaleAll != 0.0) {
                getMscale(factor, mscale) / getMscale(factor, mscaleAll)
            } else {
                getMscale(factor)
            }
            if (kotlin.math.abs(attentionFactor - 1.0) > 1e-12) {
                throw JsonException("HfDecoderConfig: YaRN scales cos/sin by $attentionFactor; the $id graph takes 1. Refused by name")
            }
            val original = num("original_max_position_embeddings")?.toInt() ?: maxPositions
            val betaFast = num("beta_fast") ?: 32.0
            val betaSlow = num("beta_slow") ?: 1.0
            val truncate = (rope?.get("truncate") as? JsonBool)?.value ?: true
            fun correctionDim(rotations: Double) =
                dim * Math.log(original / (rotations * 2 * Math.PI)) / (2 * Math.log(base))
            var low = correctionDim(betaFast)
            var high = correctionDim(betaSlow)
            if (truncate) {
                low = Math.floor(low)
                high = Math.ceil(high)
            }
            low = maxOf(low, 0.0)
            high = minOf(high, dim - 1.0)
            if (low == high) high += 0.001
            // f32, as transformers computes them.
            val invFreq = List(dim / 2) { i ->
                val pos = Math.pow(base, (2.0 * i) / dim).toFloat()
                val extrapolation = 1f / pos
                val interpolation = 1f / (factor.toFloat() * pos)
                val ramp = ((i - low) / (high - low)).coerceIn(0.0, 1.0).toFloat()
                val extrapolationFactor = 1f - ramp
                (interpolation * (1f - extrapolationFactor) + extrapolation * extrapolationFactor).toDouble()
            }
            return invFreq to (if (mscaleAll != null && mscaleAll != 0.0) getMscale(factor, mscaleAll) else 1.0)
        }
    }

    /** The intermediate size when the config states no `intermediate_size`, or null to refuse. */
    open fun intermediateSize(root: JsonObject): Int? = null

    companion object {
        private val qwen35Linear = DecoderLayerSpec(mixer = TokenMixer.GATED_DELTA_NET)

        /** The layers of a Qwen3.5 config: `layer_types`, or three linear layers to one full. */
        private fun qwen35Layers(root: JsonObject, numLayers: Int, mlp: MlpKind): List<DecoderLayerSpec> {
            if ((root["attn_output_gate"] as? JsonBool)?.value == false) {
                throw JsonException(
                    "HfDecoderConfig: attn_output_gate = false; the qwen3_5 family reads q_proj as " +
                        "query and gate. Refused by name",
                )
            }
            val types = layerTypes(root, numLayers) ?: run {
                val every = root.optIntKey("full_attention_interval") ?: 4
                List(numLayers) { if ((it + 1) % every == 0) FULL_ATTENTION else LINEAR_ATTENTION }
            }
            return types.mapIndexed { l, t ->
                when (t) {
                    FULL_ATTENTION -> DecoderLayerSpec(qkNorm = true, queryGate = true, mlp = mlp)
                    LINEAR_ATTENTION -> qwen35Linear.copy(mlp = mlp)
                    else -> throw JsonException(
                        "HfDecoderConfig: layer_types[$l] = '$t' is not one of '$FULL_ATTENTION' or " +
                            "'$LINEAR_ATTENTION'",
                    )
                }
            }
        }

        /** `mtp_num_hidden_layers`: at most one layer, sharing the target's embeddings. */
        private fun mtpLayers(root: JsonObject): Int {
            val n = root.optIntKey("mtp_num_hidden_layers") ?: 0
            if (n > 1) throw JsonException("HfDecoderConfig: mtp_num_hidden_layers = $n; one MTP layer is read. Refused by name")
            if ((root["mtp_use_dedicated_embeddings"] as? JsonBool)?.value == true) {
                throw JsonException(
                    "HfDecoderConfig: mtp_use_dedicated_embeddings = true; the MTP head is read with the target's " +
                        "embeddings. Refused by name",
                )
            }
            return n
        }

        /** Qwen3.5's model-wide settings (see [Qwen3_5]). */
        private fun qwen35Refine(root: JsonObject, outer: JsonObject, config: HfDecoderConfig): HfDecoderConfig {
            // A quantized checkpoint: HfCheckpoint dequantizes its tensors by their
            // names (ModelOpt FP8 / NVFP4, block FP8), and only those formats.
            (outer["quantization_config"] as? JsonObject)?.let { q ->
                val method = (q["quant_method"] as? JsonString)?.value
                if (method != "modelopt" && method != "fp8") {
                    throw JsonException(
                        "HfDecoderConfig: quantization_config.quant_method = '$method'; the readable quantized " +
                            "checkpoints are ModelOpt ('modelopt': FP8 and NVFP4) and block FP8 ('fp8'). Refused by name",
                    )
                }
            }
            (root["mlp_only_layers"] as? JsonArray)?.let {
                if (it.elements.isNotEmpty()) {
                    throw JsonException("HfDecoderConfig: mlp_only_layers is not empty; refused by name")
                }
            }
            // The gated norm's activation: swish is silu, which the graph applies.
            val gateAct = (root["output_gate_type"] as? JsonString)?.value
            if (gateAct != null && gateAct != "swish" && gateAct != "silu") {
                throw JsonException(
                    "HfDecoderConfig: output_gate_type = '$gateAct'; the Gated DeltaNet's gated norm " +
                        "is implemented with silu (swish). Refused by name",
                )
            }
            val ssm = (root["mamba_ssm_dtype"] as? JsonString)?.value
            if (ssm != null && ssm != "float32") {
                throw JsonException(
                    "HfDecoderConfig: mamba_ssm_dtype = '$ssm'; the delta rule's state is f32. Refused by name",
                )
            }
            val rope = root["rope_parameters"] as? JsonObject
            val partial = (rope?.get("partial_rotary_factor") as? JsonNumber)?.value
                ?: (root["partial_rotary_factor"] as? JsonNumber)?.value ?: 1.0
            return config.copy(
                linearAttention = LinearAttentionConfig(
                    numKeyHeads = root.optIntKey("linear_num_key_heads") ?: 16,
                    numValueHeads = root.optIntKey("linear_num_value_heads") ?: 32,
                    keyHeadDim = root.optIntKey("linear_key_head_dim") ?: 128,
                    valueHeadDim = root.optIntKey("linear_value_head_dim") ?: 128,
                    convKernel = root.optIntKey("linear_conv_kernel_dim") ?: 4,
                ),
                partialRotaryFactor = partial,
                mtpLayers = mtpLayers(root),
                layerNormGainPlusOne = true,
                qkNormGainPlusOne = true,
                finalNormGainPlusOne = true,
                tieWordEmbeddings = (root["tie_word_embeddings"] as? JsonBool)?.value
                    ?: (outer["tie_word_embeddings"] as? JsonBool)?.value ?: false,
                refusedTokenIds = buildMap {
                    outer.optIntKey("image_token_id")?.let { put(it, "image_token_id") }
                    outer.optIntKey("video_token_id")?.let { put(it, "video_token_id") }
                },
            )
        }

        /** Every family this repo reads. */
        val ALL: List<HfModelFamily> get() = listOf(Llama, Qwen3, MuseGlimmer, Qwen3_5, Qwen3_5Moe, Xing4_0)

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
        private const val LINEAR_ATTENTION = "linear_attention"

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
    /**
     * Under [tieWordEmbeddings], stage the head as its own slot (the
     * embedding table transposed, a second copy on the device) instead of
     * contracting against the table itself. False by default; the copy is
     * kept as the control the direct head is compared with.
     */
    val tiedHeadCopy: Boolean = false,
    /**
     * Weight-only quantization of the layers' Linear weights, [WeightQuant.NONE]
     * by default. With [WeightQuant.INT8] they are staged as int8 codes plus
     * one f32 scale per output channel (see [WeightQuant] and
     * [HfDecoderGraph.weightSlots]); everything else keeps [weightDType].
     */
    val weightQuant: WeightQuant = WeightQuant.NONE,
    /** The Gated DeltaNet dims, for a family with [TokenMixer.GATED_DELTA_NET] layers; null otherwise. */
    val linearAttention: LinearAttentionConfig? = null,
    /**
     * `partial_rotary_factor`: the fraction of each head's leading channels
     * RoPE rotates ([rotaryDim]); the rest pass through.
     */
    val partialRotaryFactor: Double = 1.0,
    /** The per-head q/k RMSNorms multiply by `1 + w` (Qwen3.5). */
    val qkNormGainPlusOne: Boolean = false,
    /** The final RMSNorm multiplies by `1 + w` (Qwen3.5). */
    val finalNormGainPlusOne: Boolean = false,
    /** The experts of a family with [MlpKind.MOE] layers; null otherwise. */
    val moe: MoeConfig? = null,
    /** The checkpoint's MTP layers (`mtp_num_hidden_layers`): 0, or 1 for the Qwen3.5 family. */
    val mtpLayers: Int = 0,
    /**
     * Tokens the MTP head drafts per speculative step; 0 (the default) leaves
     * the head unread. A serving choice, like [weightQuant]: with it the
     * decode entries verify `1 + mtpDraftTokens` tokens per sequence and the
     * weights include the head's ([HfDecoderNames.mtpRoles]).
     */
    val mtpDraftTokens: Int = 0,
    /**
     * The format of a separate head for the MTP drafts ([DecoderWeightRole.DraftHead]),
     * or NONE (the default) for drafts through [DecoderWeightRole.LmHead]. The
     * target's tokens always use the full head, so outputs do not change; the
     * drafts may, and with them how many are accepted.
     */
    val mtpDraftHeadQuant: WeightQuant = WeightQuant.NONE,
    /**
     * The token ids the drafts are chosen from, `[0, mtpDraftVocab)`, through the
     * first rows of [DecoderWeightRole.DraftHead]; 0 (the default) for the whole
     * vocabulary. A byte-level BPE vocabulary lists its earliest (most frequent)
     * merges first, so a prefix keeps most drafts that get accepted while each
     * draft reads a fraction of the head. The target's tokens still use the full
     * head. Needs [mtpDraftHeadQuant].
     */
    val mtpDraftVocab: Int = 0,
    /**
     * The last token ids the drafts may also choose, after the [mtpDraftVocab] prefix
     * (a vocabulary's special tokens sit at its end: chat and reasoning markers); 0
     * (the default) for none. Needs an NVFP4 draft head.
     */
    val mtpDraftVocabTail: Int = 0,
    /**
     * The format of the LM head ([DecoderWeightRole.LmHead]), NONE (the
     * default, [weightDType]) or a quantized one. It changes the target's
     * logits, so the model's outputs: NVFP4 is how NVIDIA's NVFP4 checkpoints
     * store the head.
     */
    val headQuant: WeightQuant = WeightQuant.NONE,
    /** Multi-head latent attention's dims, for a family with [TokenMixer.MLA] layers; null otherwise. */
    val mla: MlaConfig? = null,
    /** The residual streams, for a family whose layers have [DecoderLayerSpec.hyperConnections]; null otherwise. */
    val hyper: HyperConnectionConfig? = null,
) {
    /** The rows of the drafts' head: [mtpDraftVocab] and [mtpDraftVocabTail], or the whole vocabulary. */
    val draftVocab: Int get() = if (mtpDraftVocab == 0) vocabSize else mtpDraftVocab + mtpDraftVocabTail

    init {
        require(hiddenSize >= 1 && intermediateSize >= 1) {
            "HfDecoderConfig: hidden_size/intermediate_size must be >= 1, got $hiddenSize/$intermediateSize"
        }
        require(numLayers >= 1) { "HfDecoderConfig: num_hidden_layers must be >= 1, got $numLayers" }
        require(mtpLayers in 0..1) { "HfDecoderConfig: mtp_num_hidden_layers $mtpLayers; one MTP layer is read" }
        require(mtpDraftTokens >= 0 && (mtpDraftTokens == 0 || mtpLayers == 1)) {
            "HfDecoderConfig: mtpDraftTokens $mtpDraftTokens needs the checkpoint's MTP layer (mtp_num_hidden_layers = $mtpLayers)"
        }
        require(mtpDraftHeadQuant == WeightQuant.NONE || mtpDraftTokens > 0) {
            "HfDecoderConfig: mtpDraftHeadQuant ${mtpDraftHeadQuant.tag} without MTP drafts (mtpDraftTokens = 0)"
        }
        require(mtpDraftVocab == 0 || (mtpDraftHeadQuant != WeightQuant.NONE && mtpDraftVocab % 16 == 0 && mtpDraftVocab in 16..vocabSize)) {
            "HfDecoderConfig: mtpDraftVocab $mtpDraftVocab needs a draft head of its own (mtpDraftHeadQuant) and a " +
                "multiple of 16 up to the vocabulary ($vocabSize)"
        }
        require(
            mtpDraftVocabTail == 0 ||
                (mtpDraftVocab > 0 && mtpDraftHeadQuant == WeightQuant.NVFP4 && mtpDraftVocabTail % 16 == 0 && mtpDraftVocab + mtpDraftVocabTail <= vocabSize),
        ) {
            "HfDecoderConfig: mtpDraftVocabTail $mtpDraftVocabTail needs a prefix (mtpDraftVocab), an NVFP4 draft head and " +
                "a multiple of 16, with the prefix within the vocabulary ($vocabSize)"
        }
        require(headQuant == WeightQuant.NONE || !tieWordEmbeddings || tiedHeadCopy) {
            "HfDecoderConfig: headQuant ${headQuant.tag} needs a head of its own; this one reads the embedding table"
        }
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
        require(partialRotaryFactor > 0.0 && partialRotaryFactor <= 1.0) {
            "HfDecoderConfig: partial_rotary_factor must be in (0, 1], got $partialRotaryFactor"
        }
    }

    /** How many leading channels of each head RoPE rotates: `head_dim * partial_rotary_factor`. */
    val rotaryDim: Int get() = (headDim * partialRotaryFactor).toInt()

    /** The Gated DeltaNet layers, ascending. */
    val linearLayers: List<Int>
        get() = (0 until numLayers).filter { layer(it).mixer == TokenMixer.GATED_DELTA_NET }

    /**
     * The [LinearStatePool] of this config's Gated DeltaNet layers with
     * [numSlots] sequence slots, or null when it has none.
     */
    fun linearStatePool(numSlots: Int): LinearStatePool? {
        val la = linearAttention ?: return null
        val layers = linearLayers
        if (layers.isEmpty()) return null
        return LinearStatePool(
            layers = layers,
            numSlots = numSlots,
            convChannels = la.convChannels,
            convKernel = la.convKernel,
            valueHeads = la.numValueHeads,
            keyDim = la.keyHeadDim,
            valueDim = la.valueHeadDim,
        )
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
    /**
     * The MTP head's decoder layer: a full-attention layer of the family
     * (gated query, q/k norms) with the model's MLP.
     */
    val mtpLayerSpec: DecoderLayerSpec
        get() {
            require(mtpLayers == 1) { "HfDecoderConfig: this checkpoint has no MTP layer" }
            return DecoderLayerSpec(qkNorm = true, queryGate = true, mlp = if (moe != null) MlpKind.MOE else MlpKind.DENSE)
        }

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
        if (moe == null && (0 until numLayers).any { layer(it).mlp == MlpKind.MOE }) {
            add("mixture-of-experts layers without the experts' config")
        }
        if (linearAttention == null && linearLayers.isNotEmpty()) {
            add("Gated DeltaNet layers $linearLayers without the linear-attention dims")
        }
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
        windowedKv: WindowedKvPool? = null,
        stateSlots: Int? = null,
        kvDtype: DType? = null,
        cudaKernels: Boolean = false,
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
            kvDtype = if (kvQuant != null) io.tlaloc.core.I32 else kvDtype ?: dtype,
            kvQuant = kvQuant,
            windowedKv = windowedKv,
            linearState = if (linearLayers.isEmpty()) {
                null
            } else {
                linearStatePool(
                    stateSlots ?: throw JsonException(
                        "HfDecoderConfig ($family): the model has Gated DeltaNet layers $linearLayers, so its " +
                            "decode shape needs a number of state slots (stateSlots)",
                    ),
                )
            },
            mtpDraftTokens = mtpDraftTokens,
            cudaKernels = cudaKernels,
        )
    }

    /**
     * The [WindowedKvPool] for this config's sliding-window layers, or null
     * when it has none: every sliding layer, the largest of their windows,
     * and a ring of [ringPages] pages (by default
     * [WindowedKvPool.defaultRingPages], capped at the pages of
     * [maxContext], beyond which a ring never wraps).
     *
     * With [prefillChunk] the default ring is long enough for a prefill call
     * of that many tokens past the window: `ceil((window - 1 + prefillChunk) /
     * blockSize)` pages when that is more than the default (still capped at
     * [maxContext]). A shorter ring would split each such call into calls of
     * at most `ringPages * blockSize - (window - 1)` tokens.
     *
     * [numBlocks] is the budget of each windowed pool. By default it holds as
     * many full rings as the full-history pool of [fullNumBlocks] pages holds
     * sequences of [maxContext] positions (rounded up), and never more pages
     * than that pool. A sequence never holds more windowed pages than full
     * ones, so a smaller budget can refuse a sequence the full pool would
     * take: the backend refuses it by name, as it does when the full pool
     * runs out.
     */
    fun windowedKvPool(
        blockSize: Int,
        maxContext: Int,
        fullNumBlocks: Int,
        ringPages: Int? = null,
        numBlocks: Int? = null,
        prefillChunk: Int? = null,
    ): WindowedKvPool? {
        require(prefillChunk == null || prefillChunk >= 1) {
            "HfDecoderConfig.windowedKvPool: prefillChunk must be >= 1, got $prefillChunk"
        }
        val sliding = (0 until numLayers).filter { layer(it).attention == AttentionKind.SLIDING }
        if (sliding.isEmpty()) return null
        val window = sliding.maxOf { layer(it).slidingWindow!! }
        val contextPages = (maxContext + blockSize - 1) / blockSize
        val forChunk = prefillChunk?.let { (window - 1 + it + blockSize - 1) / blockSize } ?: 0
        val ring = ringPages
            ?: minOf(maxOf(WindowedKvPool.defaultRingPages(window, blockSize), forChunk), contextPages)
        val sequences = (fullNumBlocks - 1 + contextPages - 1) / contextPages
        return WindowedKvPool(
            window = window,
            layers = sliding,
            numBlocks = numBlocks ?: minOf(fullNumBlocks, 1 + maxOf(1, sequences) * ring),
            ringPages = ring,
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
                (outer[nestedKey] as? JsonObject
                    ?: throw JsonException("HfDecoderConfig: config.json has no '$nestedKey' object")).also { nested ->
                    val outerHidden = (outer["hidden_size"] as? io.tlaloc.core.io.JsonNumber)?.value?.toInt()
                    val innerHidden = (nested["hidden_size"] as? io.tlaloc.core.io.JsonNumber)?.value?.toInt()
                    if (outerHidden != null && outerHidden != innerHidden) {
                        throw JsonException(
                            "HfDecoderConfig: config.json says hidden_size $outerHidden outside '$nestedKey' and " +
                                "$innerHidden inside it",
                        )
                    }
                }
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
                intermediateSize = root.optInt("intermediate_size") ?: family.intermediateSize(root) ?: root.reqInt("intermediate_size"),
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
            DecoderWeightRole.LmHead, DecoderWeightRole.DraftHead -> LM_HEAD
            is DecoderWeightRole.Layer -> "${family.modelPrefix}layers.${role.layer}.${family.leaf(role.part)}"
            is DecoderWeightRole.Mtp -> "$MTP_PREFIX${role.part.leaf}"
            is DecoderWeightRole.MtpLayer -> "${MTP_PREFIX}layers.0.${family.leaf(role.part)}"
        }

    /** The MTP head's tensors are at the checkpoint's top level, outside the language model. */
    const val MTP_PREFIX: String = "mtp."

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
        if (name.startsWith(MTP_PREFIX)) {
            val rest = name.removePrefix(MTP_PREFIX)
            MtpPart.entries.firstOrNull { it.leaf == rest }?.let { return DecoderWeightRole.Mtp(it) }
            val leaf = rest.removePrefix("layers.0.").takeIf { it != rest } ?: return null
            return DecoderLayerPart.entries.firstOrNull { family.leaf(it) == leaf }?.let { DecoderWeightRole.MtpLayer(it) }
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
        if (config.mtpDraftTokens > 0) addAll(mtpRoles(config))
        if (config.mtpDraftHeadQuant != WeightQuant.NONE) add(DecoderWeightRole.DraftHead)
    }

    /** The MTP head's roles: its four tensors, then its layer's parts. */
    fun mtpRoles(config: HfDecoderConfig): List<DecoderWeightRole> =
        MtpPart.entries.map { DecoderWeightRole.Mtp(it) } +
            config.mtpLayerSpec.parts.map { DecoderWeightRole.MtpLayer(it) }

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
        DecoderWeightRole.DraftHead -> intArrayOf(config.draftVocab, config.hiddenSize)
        is DecoderWeightRole.Mtp -> when (role.part) {
            MtpPart.FC -> intArrayOf(config.hiddenSize, 2 * config.hiddenSize)
            else -> intArrayOf(config.hiddenSize)
        }
        is DecoderWeightRole.MtpLayer -> layerDims(role.part, config.mtpLayerSpec, config)
        is DecoderWeightRole.Layer -> layerDims(role.part, config.layer(role.layer), config)
    }

    private fun layerDims(part: DecoderLayerPart, spec: DecoderLayerSpec, config: HfDecoderConfig): IntArray {
        if (spec.mixer == TokenMixer.MLA || spec.hyperConnections) mlaDims(part, spec, config)?.let { return it }
        return when (part) {
            DecoderLayerPart.Q_PROJ -> intArrayOf(
                if (spec.queryGate) 2 * config.qProjOut else config.qProjOut,
                config.hiddenSize,
            )
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
            DecoderLayerPart.ROUTER, DecoderLayerPart.EXPERTS_GATE_UP, DecoderLayerPart.EXPERTS_DOWN,
            DecoderLayerPart.SHARED_GATE_PROJ, DecoderLayerPart.SHARED_UP_PROJ, DecoderLayerPart.SHARED_DOWN_PROJ,
            DecoderLayerPart.SHARED_EXPERT_GATE, DecoderLayerPart.ROUTER_BIAS -> moeDims(part, config)
            else -> {
                val la = config.linearAttention ?: throw JsonException(
                    "HfDecoderNames: ${part} needs the config's linear-attention dims",
                )
                when (part) {
                    DecoderLayerPart.IN_PROJ_QKV -> intArrayOf(la.convChannels, config.hiddenSize)
                    DecoderLayerPart.IN_PROJ_Z -> intArrayOf(la.valueWidth, config.hiddenSize)
                    DecoderLayerPart.IN_PROJ_B, DecoderLayerPart.IN_PROJ_A ->
                        intArrayOf(la.numValueHeads, config.hiddenSize)
                    DecoderLayerPart.CONV1D -> intArrayOf(la.convChannels, 1, la.convKernel)
                    DecoderLayerPart.DT_BIAS, DecoderLayerPart.A_LOG -> intArrayOf(la.numValueHeads)
                    DecoderLayerPart.LINEAR_NORM -> intArrayOf(la.valueHeadDim)
                    DecoderLayerPart.OUT_PROJ -> intArrayOf(config.hiddenSize, la.valueWidth)
                    else -> moeDims(part, config)
                }
            }
        }
    }

    /**
     * True when this role's file tensor is a Linear weight and therefore
     * stored as `[out, in]`, so a matmul against it needs `x @ W^T` or a
     * transpose at ingestion. False for the embedding table and the norm gains.
     */
    fun isTransposedLinear(role: DecoderWeightRole): Boolean = when (role) {
        DecoderWeightRole.EmbedTokens, DecoderWeightRole.FinalNorm -> false
        DecoderWeightRole.LmHead, DecoderWeightRole.DraftHead -> true
        is DecoderWeightRole.Mtp -> role.part == MtpPart.FC
        is DecoderWeightRole.Layer, is DecoderWeightRole.MtpLayer -> {
            val part = role.layerPart!!
            !part.isVector && part != DecoderLayerPart.CONV1D && !part.isExperts && part != DecoderLayerPart.KV_B_PROJ
        }
    }

    /** The dims of the parts of an MLA or hyper-connected layer that differ from the others', else null. */
    private fun mlaDims(part: DecoderLayerPart, spec: DecoderLayerSpec, config: HfDecoderConfig): IntArray? {
        val h = config.hiddenSize
        config.hyper?.let { hc ->
            when (part) {
                DecoderLayerPart.ATTN_HC_FN, DecoderLayerPart.FFN_HC_FN -> return intArrayOf(hc.mixOutputs, hc.streams * h)
                DecoderLayerPart.ATTN_HC_BASE, DecoderLayerPart.FFN_HC_BASE -> return intArrayOf(hc.mixOutputs)
                DecoderLayerPart.ATTN_HC_SCALE, DecoderLayerPart.FFN_HC_SCALE -> return intArrayOf(3)
                else -> {}
            }
        }
        if (spec.mixer != TokenMixer.MLA) return null
        val m = config.mla ?: throw JsonException("HfDecoderNames: $part needs the config's MLA dims")
        val heads = config.numHeads
        return when (part) {
            DecoderLayerPart.Q_A_PROJ -> intArrayOf(m.qLoraRank, h)
            DecoderLayerPart.Q_A_NORM -> intArrayOf(m.qLoraRank)
            DecoderLayerPart.Q_B_PROJ -> intArrayOf(heads * (m.nopeDim + m.ropeDim), m.qLoraRank)
            DecoderLayerPart.KV_A_PROJ -> intArrayOf(m.kvLoraRank + m.ropeDim, h)
            DecoderLayerPart.KV_A_NORM -> intArrayOf(m.kvLoraRank)
            DecoderLayerPart.KV_B_PROJ -> intArrayOf(heads * (m.nopeDim + m.valueDim), m.kvLoraRank)
            DecoderLayerPart.O_PROJ -> intArrayOf(h, heads * m.valueDim)
            else -> null
        }
    }

    private fun moeDims(part: DecoderLayerPart, config: HfDecoderConfig): IntArray {
        val m = config.moe ?: throw JsonException("HfDecoderNames: $part needs the config's experts")
        val h = config.hiddenSize
        return when (part) {
            DecoderLayerPart.ROUTER -> intArrayOf(m.numExperts, h)
            DecoderLayerPart.EXPERTS_GATE_UP -> intArrayOf(m.numExperts, 2 * m.expertIntermediate, h)
            DecoderLayerPart.EXPERTS_DOWN -> intArrayOf(m.numExperts, h, m.expertIntermediate)
            DecoderLayerPart.SHARED_GATE_PROJ, DecoderLayerPart.SHARED_UP_PROJ -> intArrayOf(m.sharedIntermediate, h)
            DecoderLayerPart.SHARED_DOWN_PROJ -> intArrayOf(h, m.sharedIntermediate)
            DecoderLayerPart.SHARED_EXPERT_GATE -> intArrayOf(1, h)
            DecoderLayerPart.ROUTER_BIAS -> intArrayOf(m.numExperts)
            else -> error("unreachable: $part")
        }
    }

    /**
     * True when [role] is staged quantized under [config]'s
     * [HfDecoderConfig.weightQuant]: a layer's Linear weight, and only when
     * quantization is on; and the drafts' head under
     * [HfDecoderConfig.mtpDraftHeadQuant]. The embedding table, the norms and
     * the head are never quantized.
     */
    fun isQuantized(role: DecoderWeightRole, config: HfDecoderConfig): Boolean = quantOf(role, config) != WeightQuant.NONE

    /** The format [role] is staged in: see [isQuantized]. */
    fun quantOf(role: DecoderWeightRole, config: HfDecoderConfig): WeightQuant = when {
        role == DecoderWeightRole.DraftHead -> config.mtpDraftHeadQuant
        role == DecoderWeightRole.LmHead -> config.headQuant
        role.layerPart !in QUANTIZED_PARTS -> WeightQuant.NONE
        config.weightQuant != WeightQuant.NVFP4 -> config.weightQuant
        (role is DecoderWeightRole.Layer || role is DecoderWeightRole.MtpLayer) && role.layerPart in NVFP4_PARTS -> WeightQuant.NVFP4
        else -> WeightQuant.FP8
    }

    /**
     * The layer parts [WeightQuant.NVFP4] stores as NVFP4: the MLP projections and the routed experts,
     * the MTP layer's too (rounded from the checkpoint's bf16 when it stores them so).
     */
    val NVFP4_PARTS: Set<DecoderLayerPart> = setOf(
        DecoderLayerPart.GATE_PROJ, DecoderLayerPart.UP_PROJ, DecoderLayerPart.DOWN_PROJ,
        DecoderLayerPart.SHARED_GATE_PROJ, DecoderLayerPart.SHARED_UP_PROJ, DecoderLayerPart.SHARED_DOWN_PROJ,
        DecoderLayerPart.EXPERTS_GATE_UP, DecoderLayerPart.EXPERTS_DOWN,
        DecoderLayerPart.Q_A_PROJ, DecoderLayerPart.Q_B_PROJ, DecoderLayerPart.KV_A_PROJ,
    )

    /** The layer parts [WeightQuant] quantizes: the large projections. */
    val QUANTIZED_PARTS: Set<DecoderLayerPart> = setOf(
        DecoderLayerPart.Q_PROJ, DecoderLayerPart.K_PROJ, DecoderLayerPart.V_PROJ, DecoderLayerPart.O_PROJ,
        DecoderLayerPart.ATTN_GATE_PROJ, DecoderLayerPart.GATE_PROJ, DecoderLayerPart.UP_PROJ, DecoderLayerPart.DOWN_PROJ,
        DecoderLayerPart.IN_PROJ_QKV, DecoderLayerPart.IN_PROJ_Z, DecoderLayerPart.OUT_PROJ,
        DecoderLayerPart.SHARED_GATE_PROJ, DecoderLayerPart.SHARED_UP_PROJ, DecoderLayerPart.SHARED_DOWN_PROJ,
        DecoderLayerPart.EXPERTS_GATE_UP, DecoderLayerPart.EXPERTS_DOWN,
    )
}

/** The dims of a Gated DeltaNet layer (`linear_*` keys of a Qwen3.5 config). */
data class LinearAttentionConfig(
    val numKeyHeads: Int,
    val numValueHeads: Int,
    val keyHeadDim: Int,
    val valueHeadDim: Int,
    val convKernel: Int,
) {
    init {
        require(numKeyHeads >= 1 && numValueHeads % numKeyHeads == 0) {
            "LinearAttentionConfig: linear_num_key_heads $numKeyHeads must divide linear_num_value_heads $numValueHeads"
        }
        require(keyHeadDim >= 1 && valueHeadDim >= 1 && convKernel >= 2) {
            "LinearAttentionConfig: head dims >= 1 and conv kernel >= 2, got $keyHeadDim/$valueHeadDim/$convKernel"
        }
    }

    /** `Hk * Dk`: the width of q and of k. */
    val keyWidth: Int get() = numKeyHeads * keyHeadDim

    /** `Hv * Dv`: the width of v, of the gate and of the output. */
    val valueWidth: Int get() = numValueHeads * valueHeadDim

    /** The conv's channels: q, k and v, `2 Hk Dk + Hv Dv`. */
    val convChannels: Int get() = 2 * keyWidth + valueWidth
}

/** The experts of a mixture-of-experts MLP (`num_experts`, `num_experts_per_tok`, ... of a Qwen3.5-MoE config). */
data class MoeConfig(
    val numExperts: Int,
    val topK: Int,
    val expertIntermediate: Int,
    val sharedIntermediate: Int,
    /** How the router's logits choose and weigh experts. */
    val routing: MoeRouting = MoeRouting.SOFTMAX,
    /** The routed experts' weights are multiplied by this (`routed_scaling_factor`). */
    val routedScale: Double = 1.0,
) {
    init {
        require(numExperts >= 1 && topK in 1..numExperts && expertIntermediate >= 1 && sharedIntermediate >= 1) {
            "MoeConfig: $numExperts experts, top $topK, intermediate $expertIntermediate, shared $sharedIntermediate"
        }
    }
}

/** How a mixture of experts routes. */
enum class MoeRouting {
    /** `softmax(logits)`, the top k, renormalized (Qwen3.5-MoE). */
    SOFTMAX,

    /**
     * `s = sigmoid(logits)`; the top k of `s + bias` ([DecoderLayerPart.ROUTER_BIAS]); the weights
     * are their `s`, normalized to sum 1 and times [MoeConfig.routedScale] (DeepSeek-V3's noaux_tc).
     */
    SIGMOID_BIAS,
}

/**
 * Multi-head latent attention (DeepSeek-V3): queries through a low-rank
 * latent, keys and values through a shared [kvLoraRank] latent plus one
 * [ropeDim]-wide rope key. The KV pool holds the latent and the rope key
 * ([poolDim] values per token, one "KV head"); attention runs in the latent
 * (each head's nope query mapped through its key block of
 * [DecoderLayerPart.KV_B_PROJ], its output back through its value block).
 */
data class MlaConfig(
    val qLoraRank: Int,
    val kvLoraRank: Int,
    val nopeDim: Int,
    val ropeDim: Int,
    val valueDim: Int,
    /** The rope frequencies, one per pair of rope dims (YaRN-blended for a YaRN config). */
    val invFreq: List<Double>,
    /** The softmax scale: `(nope + rope)^-1/2`, times YaRN's mscale squared. */
    val scale: Double,
) {
    init {
        require(qLoraRank >= 1 && kvLoraRank >= 1 && nopeDim >= 1 && ropeDim % 2 == 0 && valueDim >= 1) {
            "MlaConfig: q_lora_rank $qLoraRank, kv_lora_rank $kvLoraRank, nope $nopeDim, rope $ropeDim, v $valueDim"
        }
        require(invFreq.size == ropeDim / 2) { "MlaConfig: ${invFreq.size} rope frequencies for $ropeDim rope dims" }
    }

    /** Values per token in the KV pool: the latent, then the rope key. */
    val poolDim: Int get() = kvLoraRank + ropeDim
}

/**
 * Hyper-connections ([streams] residual streams): before each sublayer, a
 * map of the gainless-RMS-normalized streams gives the weights the sublayer
 * reads them with, the weights its output is added to each with, and a
 * [streams] x [streams] mixing of them, made doubly stochastic by
 * [sinkhornIters] Sinkhorn iterations of its exponentiated, clamped logits.
 */
data class HyperConnectionConfig(
    val streams: Int,
    val sinkhornIters: Int,
    val eps: Double,
    val clampMin: Double,
    val clampMax: Double,
) {
    /** Outputs of the mixing map: pre and post weights per stream, and the mixing matrix. */
    val mixOutputs: Int get() = (2 + streams) * streams
}
