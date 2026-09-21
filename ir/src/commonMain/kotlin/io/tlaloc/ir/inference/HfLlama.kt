package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.recognizer.quant.KvQuantConfig

// §0.4.478 (Phase H3c-1) — the layer between a HuggingFace Llama checkpoint
// and the tensor roles this repo's decode graph needs. Two halves, both pure:
//
//   1. [HfLlamaConfig]  — config.json, read through :core's STRICT parseJson,
//                          into the numbers a decode graph is built from.
//   2. [HfLlamaNames]   — the HF parameter-name <-> [LlamaWeightRole] bijection,
//                          plus the EXPECTED DIMS of every role, derived from
//                          the config. That last part is the whole point: a
//                          name mapping that does not also state the layout is
//                          a rename, and a rename cannot catch a transpose.
//
// PLACEMENT (`:ir`, `io.tlaloc.ir.inference`, commonMain). The output type is
// [DecodeModelShape], which lives in this package; the roles are defined by
// what the decode graph consumes, which is also this package. REJECTED:
// `:core`'s `io` package — it holds FORMATS (safetensors, JSON) and knows
// nothing about a transformer; putting Llama's parameter names there would
// make the format reader depend on one model family. REJECTED: `:maestro` —
// that module is about the deployment MANIFEST, and §0.4.468 already rejected
// it for weights on the same ground. REJECTED: jvmMain — none of this touches
// a file; opening one is [HfLlamaCheckpoint] in jvmMain and nothing else.
//
// THE LAYOUT FACT, stated once. **HuggingFace stores every `nn.Linear` weight
// TRANSPOSED relative to the mathematical matrix: `[out_features,
// in_features]`**, because `torch.nn.functional.linear(x, W)` computes
// `x @ W.T`. So `q_proj.weight` is `[numHeads*headDim, hiddenSize]`, not
// `[hiddenSize, numHeads*headDim]`. This is ASSERTED, not assumed: see
// [expectedDims], and see `HfLlamaCheckpointTest` which checks it against a
// real downloaded checkpoint where the GQA projections and the MLP are both
// RECTANGULAR — a square model (hidden == numHeads*headDim) cannot tell the
// two conventions apart, and TinyLlama's k/v (`[256, 2048]`) and gate/up
// (`[5632, 2048]`) can.
//
// The embedding is NOT a Linear: `model.embed_tokens.weight` is a real
// `[vocabSize, hiddenSize]` lookup table, so it happens to have the same dims
// either way and proves nothing about the convention. `lm_head` IS a Linear,
// which is exactly why a tied checkpoint can reuse the embedding table for it.

/** Which non-layer tensor, or which part of which layer, a checkpoint entry is. */
sealed interface LlamaWeightRole {
    /** `model.embed_tokens.weight` — `[vocabSize, hiddenSize]`. */
    data object EmbedTokens : LlamaWeightRole

    /** `model.norm.weight` — the final RMSNorm gain, `[hiddenSize]`. */
    data object FinalNorm : LlamaWeightRole

    /**
     * `lm_head.weight` — `[vocabSize, hiddenSize]`. Under tied embeddings this
     * role RESOLVES to [EmbedTokens]'s tensor and the name is absent from the
     * file; see [HfLlamaConfig.tieWordEmbeddings].
     */
    data object LmHead : LlamaWeightRole

    /** One tensor inside `model.layers.$layer`. */
    data class Layer(val layer: Int, val part: LlamaLayerPart) : LlamaWeightRole {
        init {
            require(layer >= 0) { "LlamaWeightRole.Layer: layer index must be >= 0, got $layer" }
        }
    }
}

/**
 * The nine tensors a Llama decoder layer carries, named by ROLE. The HF
 * spelling of each is in [HfLlamaNames]; nothing downstream of this enum
 * should contain the string `self_attn`.
 */
enum class LlamaLayerPart {
    Q_PROJ, K_PROJ, V_PROJ, O_PROJ,
    GATE_PROJ, UP_PROJ, DOWN_PROJ,
    INPUT_LAYERNORM, POST_ATTENTION_LAYERNORM,
    ;

    /** True for the RMSNorm gains, which are rank-1 and not transposed. */
    val isNorm: Boolean get() = this == INPUT_LAYERNORM || this == POST_ATTENTION_LAYERNORM
}

/**
 * `config.json`, as far as a decode graph cares.
 *
 * Every field is read through :core's strict [parseJson] — the one that
 * refuses trailing commas, NaN literals and duplicate keys — because a
 * checkpoint is untrusted input and a config that parses "leniently" into
 * wrong numbers produces a model that runs and is wrong.
 *
 * REJECTED: carrying the whole JSON object and letting callers dig. A caller
 * that reads `config["num_key_value_heads"]` itself is a caller that will get
 * the MHA default wrong (see [numKvHeads]); the defaulting rules ARE the
 * knowledge this type exists to hold.
 */
data class HfLlamaConfig(
    /** `architectures[0]`, verbatim, e.g. `"LlamaForCausalLM"`. */
    val architecture: String,
    /** `model_type`, verbatim, e.g. `"llama"`. */
    val modelType: String,
    val hiddenSize: Int,
    val intermediateSize: Int,
    val numLayers: Int,
    val numHeads: Int,
    /**
     * `num_key_value_heads`. **Defaults to [numHeads] when absent** — that is
     * HF's own rule and it means plain MHA. Getting this default wrong turns
     * an MHA checkpoint into a GQA one with 1/N of its K/V, which loads
     * cleanly and computes nonsense.
     */
    val numKvHeads: Int,
    /**
     * `head_dim` when the config states it, otherwise `hiddenSize / numHeads`.
     * Newer Llama configs state it explicitly and it is NOT always the
     * quotient (Llama-3.2 and several derivatives decouple them), so the
     * explicit value wins whenever it is present.
     */
    val headDim: Int,
    val vocabSize: Int,
    val rmsNormEps: Double,
    val ropeTheta: Double,
    val maxPositionEmbeddings: Int,
    /** True when `lm_head.weight` is absent and the embedding table serves as it. */
    val tieWordEmbeddings: Boolean,
    /** `attention_bias` — bias vectors on q/k/v/o. Llama proper is false. */
    val attentionBias: Boolean,
    /** `torch_dtype` verbatim (`"bfloat16"`, `"float32"`, ...), or null. */
    val torchDtype: String?,
    /**
     * `rope_scaling.rope_type` (or the legacy `type` key) when the config
     * scales RoPE, else null. The VALUE is deliberately kept as a string and
     * not interpreted: v1 refuses a scaled rope by name rather than
     * implementing a family of scaling laws it cannot test.
     */
    val ropeScalingType: String?,
) {
    init {
        require(hiddenSize >= 1 && intermediateSize >= 1) {
            "HfLlamaConfig: hidden_size/intermediate_size must be >= 1, got $hiddenSize/$intermediateSize"
        }
        require(numLayers >= 1) { "HfLlamaConfig: num_hidden_layers must be >= 1, got $numLayers" }
        require(numHeads >= 1) { "HfLlamaConfig: num_attention_heads must be >= 1, got $numHeads" }
        require(headDim >= 1) { "HfLlamaConfig: head_dim must be >= 1, got $headDim" }
        require(vocabSize >= 1) { "HfLlamaConfig: vocab_size must be >= 1, got $vocabSize" }
        require(numKvHeads in 1..numHeads && numHeads % numKvHeads == 0) {
            "HfLlamaConfig: num_key_value_heads $numKvHeads must divide num_attention_heads " +
                "$numHeads (GQA groups every $numKvHeads-th query head onto one KV head)"
        }
    }

    /** `numHeads * headDim` — the width q_proj projects UP to and o_proj projects DOWN from. */
    val qProjOut: Int get() = numHeads * headDim

    /** `numKvHeads * headDim` — narrower than [qProjOut] under GQA. */
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
     * The shape record a decode graph is compiled against. The pool geometry
     * ([numBlocks], [blockSize]) is a SERVING decision, not a checkpoint fact,
     * so it is a parameter here and not a field above — the same checkpoint
     * serves at any pool size.
     *
     * Refuses a scaled rope and a biased attention BY NAME: both change the
     * arithmetic of the graph, neither is implemented, and a [DecodeModelShape]
     * carries no slot that could record them. Silently dropping either would
     * produce a model that serves and disagrees with its own reference.
     */
    fun toDecodeModelShape(
        numBlocks: Int,
        blockSize: Int,
        dtype: DType = F32,
        kvQuant: KvQuantConfig? = null,
    ): DecodeModelShape {
        if (ropeScalingType != null) {
            throw JsonException(
                "HfLlamaConfig: rope_scaling '$ropeScalingType' is refused BY NAME — this " +
                    "checkpoint's positions are not plain theta=$ropeTheta RoPE, and the decode " +
                    "graph implements only that. A NAMED DEFERRAL: llama3/linear/dynamic scaling " +
                    "is a per-family formula, and shipping one untested would make every long " +
                    "context silently wrong rather than loudly unsupported",
            )
        }
        if (attentionBias) {
            throw JsonException(
                "HfLlamaConfig: attention_bias=true is refused BY NAME — q/k/v/o carry bias " +
                    "vectors this name mapping has no roles for (Llama proper has none; Qwen2 " +
                    "does). Adding Q_BIAS/K_BIAS/V_BIAS to LlamaLayerPart is the fix, not " +
                    "ignoring them",
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
        /** Architectures this mapping claims to describe. */
        val SUPPORTED_ARCHITECTURES: Set<String> = setOf("LlamaForCausalLM")

        /**
         * Parse a `config.json`. [strictArchitecture] false lets a caller read
         * a Llama-SHAPED config whose `architectures` says something else (the
         * tiny-random test models, and the Mistral/Vicuna family that is
         * name-for-name identical); the default refuses by name, because a
         * config that says `MixtralForCausalLM` has experts this mapping has
         * no roles for and would load as a silently-truncated model.
         */
        fun parse(json: String, strictArchitecture: Boolean = true): HfLlamaConfig {
            val root = parseJson(json) as? JsonObject
                ?: throw JsonException("HfLlamaConfig: config.json is not a JSON object")

            val arch = (root["architectures"] as? JsonArray)
                ?.elements?.firstOrNull()
                ?.let { (it as? JsonString)?.value }
                ?: "<unstated>"
            val modelType = (root["model_type"] as? JsonString)?.value ?: "<unstated>"
            if (strictArchitecture && arch !in SUPPORTED_ARCHITECTURES) {
                throw JsonException(
                    "HfLlamaConfig: architectures[0] = '$arch' is not one of " +
                        "$SUPPORTED_ARCHITECTURES. Pass strictArchitecture=false only when you " +
                        "have checked that the parameter names and the arithmetic really are " +
                        "Llama's",
                )
            }

            val hidden = root.reqInt("hidden_size")
            val heads = root.reqInt("num_attention_heads")
            return HfLlamaConfig(
                architecture = arch,
                modelType = modelType,
                hiddenSize = hidden,
                intermediateSize = root.reqInt("intermediate_size"),
                numLayers = root.reqInt("num_hidden_layers"),
                numHeads = heads,
                numKvHeads = root.optInt("num_key_value_heads") ?: heads,
                headDim = root.optInt("head_dim") ?: run {
                    if (hidden % heads != 0) {
                        throw JsonException(
                            "HfLlamaConfig: config states no head_dim and hidden_size $hidden is " +
                                "not divisible by num_attention_heads $heads, so there is no " +
                                "quotient to fall back to",
                        )
                    }
                    hidden / heads
                },
                vocabSize = root.reqInt("vocab_size"),
                rmsNormEps = root.optDouble("rms_norm_eps") ?: 1e-6,
                ropeTheta = root.optDouble("rope_theta") ?: 10000.0,
                maxPositionEmbeddings = root.optInt("max_position_embeddings") ?: 2048,
                tieWordEmbeddings = (root["tie_word_embeddings"] as? JsonBool)?.value ?: false,
                attentionBias = (root["attention_bias"] as? JsonBool)?.value ?: false,
                torchDtype = (root["torch_dtype"] as? JsonString)?.value,
                ropeScalingType = ropeScalingTypeOf(root["rope_scaling"]),
            )
        }

        /**
         * `rope_scaling` is `null`, absent, or an object. HF spelled the kind
         * `type` before transformers 4.43 and `rope_type` after; both are read
         * so a 2023-vintage checkpoint is refused as loudly as a 2025 one
         * rather than slipping through as unscaled.
         */
        private fun ropeScalingTypeOf(v: io.tlaloc.core.io.JsonValue?): String? {
            val o = v as? JsonObject ?: return null
            val kind = (o["rope_type"] as? JsonString)?.value
                ?: (o["type"] as? JsonString)?.value
                ?: "<unnamed>"
            // `{"rope_type": "default"}` is HF's way of writing "no scaling".
            return if (kind == "default") null else kind
        }

        private fun JsonObject.reqInt(key: String): Int =
            (this[key] as? JsonNumber)?.asInt("HfLlamaConfig: $key")
                ?: throw JsonException(
                    "HfLlamaConfig: config.json has no numeric '$key'; it is required to know " +
                        "the model's shape and there is no defensible default for it",
                )

        private fun JsonObject.optInt(key: String): Int? =
            (this[key] as? JsonNumber)?.asInt("HfLlamaConfig: $key")

        private fun JsonObject.optDouble(key: String): Double? = (this[key] as? JsonNumber)?.value
    }
}

/**
 * The HF parameter-name <-> [LlamaWeightRole] mapping, and the dims each role
 * must have. Pure string and integer arithmetic; no file, no tensor.
 */
object HfLlamaNames {

    const val EMBED_TOKENS: String = "model.embed_tokens.weight"
    const val FINAL_NORM: String = "model.norm.weight"
    const val LM_HEAD: String = "lm_head.weight"

    /** The HF leaf spelling of each layer part, under `model.layers.N.`. */
    fun leaf(part: LlamaLayerPart): String = when (part) {
        LlamaLayerPart.Q_PROJ -> "self_attn.q_proj.weight"
        LlamaLayerPart.K_PROJ -> "self_attn.k_proj.weight"
        LlamaLayerPart.V_PROJ -> "self_attn.v_proj.weight"
        LlamaLayerPart.O_PROJ -> "self_attn.o_proj.weight"
        LlamaLayerPart.GATE_PROJ -> "mlp.gate_proj.weight"
        LlamaLayerPart.UP_PROJ -> "mlp.up_proj.weight"
        LlamaLayerPart.DOWN_PROJ -> "mlp.down_proj.weight"
        LlamaLayerPart.INPUT_LAYERNORM -> "input_layernorm.weight"
        LlamaLayerPart.POST_ATTENTION_LAYERNORM -> "post_attention_layernorm.weight"
    }

    /** The full HF tensor name for a role. */
    fun hfName(role: LlamaWeightRole): String = when (role) {
        LlamaWeightRole.EmbedTokens -> EMBED_TOKENS
        LlamaWeightRole.FinalNorm -> FINAL_NORM
        LlamaWeightRole.LmHead -> LM_HEAD
        is LlamaWeightRole.Layer -> "model.layers.${role.layer}.${leaf(role.part)}"
    }

    private val LEAF_TO_PART: Map<String, LlamaLayerPart> =
        LlamaLayerPart.entries.associateBy { leaf(it) }

    /**
     * The inverse of [hfName], or null when the name is not one this mapping
     * covers. Null is a fact, not a failure: a checkpoint legitimately carries
     * `model.rotary_emb.inv_freq` (a derived buffer some vintages persist) and
     * refusing to PARSE it is different from refusing to LOAD it.
     */
    fun role(name: String): LlamaWeightRole? {
        when (name) {
            EMBED_TOKENS -> return LlamaWeightRole.EmbedTokens
            FINAL_NORM -> return LlamaWeightRole.FinalNorm
            LM_HEAD -> return LlamaWeightRole.LmHead
        }
        if (!name.startsWith(LAYER_PREFIX)) return null
        val rest = name.substring(LAYER_PREFIX.length)
        val dot = rest.indexOf('.')
        if (dot <= 0) return null
        val idx = rest.substring(0, dot).toIntOrNull() ?: return null
        if (idx < 0) return null
        val part = LEAF_TO_PART[rest.substring(dot + 1)] ?: return null
        return LlamaWeightRole.Layer(idx, part)
    }

    private const val LAYER_PREFIX = "model.layers."

    /**
     * Every role a decode graph of this config needs, in a stable order:
     * embeddings, then each layer's nine tensors in [LlamaLayerPart] order,
     * then the final norm and the head. Under tied embeddings [LlamaWeightRole.LmHead]
     * is STILL in this list — it is a role the graph needs; where its bytes
     * come from is [HfLlamaCheckpoint]'s question, not this one's.
     */
    fun roles(config: HfLlamaConfig): List<LlamaWeightRole> = buildList {
        add(LlamaWeightRole.EmbedTokens)
        for (l in 0 until config.numLayers) {
            for (p in LlamaLayerPart.entries) add(LlamaWeightRole.Layer(l, p))
        }
        add(LlamaWeightRole.FinalNorm)
        add(LlamaWeightRole.LmHead)
    }

    /**
     * The dims a role's tensor MUST have in the file, given the config.
     *
     * **This encodes the HF transposed-Linear convention** (see this file's
     * header): a Linear's weight is `[out_features, in_features]`. The two
     * roles that discriminate the convention on any GQA model are K_PROJ/V_PROJ
     * (`[kvProjOut, hiddenSize]`, narrow-first) and GATE/UP (`[intermediate,
     * hiddenSize]`), and on a square model NOTHING discriminates it — which is
     * why the certification runs against a real rectangular checkpoint.
     */
    fun expectedDims(role: LlamaWeightRole, config: HfLlamaConfig): IntArray = when (role) {
        LlamaWeightRole.EmbedTokens -> intArrayOf(config.vocabSize, config.hiddenSize)
        LlamaWeightRole.FinalNorm -> intArrayOf(config.hiddenSize)
        LlamaWeightRole.LmHead -> intArrayOf(config.vocabSize, config.hiddenSize)
        is LlamaWeightRole.Layer -> when (role.part) {
            LlamaLayerPart.Q_PROJ -> intArrayOf(config.qProjOut, config.hiddenSize)
            LlamaLayerPart.K_PROJ -> intArrayOf(config.kvProjOut, config.hiddenSize)
            LlamaLayerPart.V_PROJ -> intArrayOf(config.kvProjOut, config.hiddenSize)
            LlamaLayerPart.O_PROJ -> intArrayOf(config.hiddenSize, config.qProjOut)
            LlamaLayerPart.GATE_PROJ -> intArrayOf(config.intermediateSize, config.hiddenSize)
            LlamaLayerPart.UP_PROJ -> intArrayOf(config.intermediateSize, config.hiddenSize)
            LlamaLayerPart.DOWN_PROJ -> intArrayOf(config.hiddenSize, config.intermediateSize)
            LlamaLayerPart.INPUT_LAYERNORM -> intArrayOf(config.hiddenSize)
            LlamaLayerPart.POST_ATTENTION_LAYERNORM -> intArrayOf(config.hiddenSize)
        }
    }

    /**
     * True when this role's file tensor is a Linear weight and therefore
     * stored TRANSPOSED — i.e. a matmul against it needs `x @ W^T`, or an
     * explicit transpose at ingestion. False for the embedding table and the
     * norm gains.
     */
    fun isTransposedLinear(role: LlamaWeightRole): Boolean = when (role) {
        LlamaWeightRole.EmbedTokens, LlamaWeightRole.FinalNorm -> false
        LlamaWeightRole.LmHead -> true
        is LlamaWeightRole.Layer -> !role.part.isNorm
    }
}
