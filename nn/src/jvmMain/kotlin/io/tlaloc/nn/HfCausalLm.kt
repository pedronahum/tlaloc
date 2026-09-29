package io.tlaloc.nn

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.floatArrayToBf16Bits
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.ir.inference.AttentionKind
import io.tlaloc.ir.inference.DecoderLayerPart
import io.tlaloc.ir.inference.DecoderWeightRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.ir.inference.HfDecoderNames
import io.tlaloc.ir.inference.HfModelFamily
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/**
 * A [CausalLM] read from, or written as, a Hugging Face checkpoint directory.
 *
 * Reads Llama (`LlamaForCausalLM`) and Qwen3 (`Qwen3ForCausalLM`)
 * checkpoints whose layers are all full-attention and that use none of the
 * options [load] names in its refusals. `nn.Linear` weights are `[out, in]`
 * in the file and `[in, out]` in [Dense], so they are transposed both ways.
 * Weights load as f32 whatever their stored width.
 */
class HfCausalLm(val model: CausalLM, val config: HfDecoderConfig) {

    /**
     * Writes [model]'s weights under their Hugging Face names into [outDir],
     * as [dtype] (BF16 or F32), next to copies of every other file in
     * [sourceDir] (config.json, tokenizer and generation files). The result
     * loads with [load], with transformers, and with the serving exporter.
     *
     * Files larger than [shardBytes] are split into
     * `model-0000i-of-0000n.safetensors` with a `model.safetensors.index.json`.
     * Under tied embeddings no `lm_head.weight` is written. A model with LoRA
     * adapters is refused: merge them first ([Lora.merge]).
     */
    fun save(
        sourceDir: Path,
        outDir: Path,
        dtype: DType = BF16,
        shardBytes: Long = 1_500_000_000L,
    ): Path {
        require(!Lora.hasAdapters(model)) {
            "HfCausalLm.save: the model has LoRA adapters, which a Hugging Face checkpoint has no tensors for; " +
                "save Lora.merge(model) for a checkpoint"
        }
        require(dtype == BF16 || dtype == F32) { "HfCausalLm.save: dtype must be BF16 or F32 (got ${dtype.name})" }
        require(sourceDir.toAbsolutePath().normalize() != outDir.toAbsolutePath().normalize()) {
            "HfCausalLm.save: outDir must differ from sourceDir ($sourceDir)"
        }
        Files.createDirectories(outDir)
        Files.list(sourceDir).use { files ->
            for (f in files) {
                val name = f.fileName.toString()
                if (!Files.isRegularFile(f) || name.endsWith(".safetensors") || name == INDEX) continue
                Files.copy(f, outDir.resolve(name), StandardCopyOption.REPLACE_EXISTING)
            }
        }

        val byKey = model.parameters.associate { it.key to it.tensor }
        val tensors = roleKeys(config).map { (role, key) ->
            val name = HfDecoderNames.hfName(role, config.family)
            val t = byKey.getValue(key)
            val values = if (isLinear(role)) transpose(t.hostF32(), t.dims[0], t.dims[1]) else t.hostF32()
            val dims = if (isLinear(role)) intArrayOf(t.dims[1], t.dims[0]) else t.dims.copyOf()
            if (dtype == BF16) SafetensorsTensor(name, BF16, dims, HostBf16Storage(floatArrayToBf16Bits(values)))
            else SafetensorsTensor(name, F32, dims, HostF32Storage(values))
        }

        val shards = ArrayList<MutableList<SafetensorsTensor>>()
        var current = 0L
        for (t in tensors) {
            val bytes = t.elementCount * dtype.sizeBytes
            if (shards.isEmpty() || (current + bytes > shardBytes && shards.last().isNotEmpty())) {
                shards.add(ArrayList())
                current = 0L
            }
            shards.last().add(t)
            current += bytes
        }
        val metadata = mapOf("format" to "pt")
        if (shards.size == 1) {
            SafetensorsFileWriter.write(outDir.resolve("model.safetensors"), shards[0], metadata)
        } else {
            val weightMap = LinkedHashMap<String, String>()
            for ((i, shard) in shards.withIndex()) {
                val file = "model-%05d-of-%05d.safetensors".format(i + 1, shards.size)
                SafetensorsFileWriter.write(outDir.resolve(file), shard, metadata)
                for (t in shard) weightMap[t.name] = file
            }
            val total = tensors.sumOf { it.elementCount * dtype.sizeBytes }
            val entries = weightMap.entries.joinToString(",\n    ") { "${jsonQuote(it.key)}: ${jsonQuote(it.value)}" }
            Files.writeString(
                outDir.resolve(INDEX),
                "{\n  \"metadata\": {\"total_size\": $total},\n  \"weight_map\": {\n    $entries\n  }\n}\n",
            )
        }
        return outDir
    }

    companion object {
        private const val INDEX = "model.safetensors.index.json"

        /** Reads the checkpoint in [dir]; refuses by name what [CausalLM] does not compute. */
        fun load(dir: Path): HfCausalLm = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config
            val refusals = buildList {
                if (c.family != HfModelFamily.Llama && c.family != HfModelFamily.Qwen3) add("family ${c.family.id}")
                for (l in 0 until c.numLayers) {
                    val spec = c.layer(l)
                    if (spec.attention != AttentionKind.FULL) add("layer $l is ${spec.attention} attention")
                    if (spec != c.family.defaultLayer.copy(attention = AttentionKind.FULL, slidingWindow = null)) {
                        add("layer $l spec $spec")
                    }
                }
                if (c.ropeScalingType != null) add("rope_scaling ${c.ropeScalingType}")
                if (c.attentionBias) add("attention_bias")
                if (c.mlpBias) add("mlp_bias")
                if (c.hiddenAct != "silu") add("hidden_act ${c.hiddenAct}")
                if (c.embeddingNorm || c.layerNormGainPlusOne || c.postNormEps != null) add("Gemma-style norms")
                if (c.queryScale != 1.0 || c.logitMultiplier != 1.0) add("query/logit scale factors")
                if (c.finalLogitSoftcap != null || c.attnLogitSoftcap != null) add("logit soft-capping")
            }.distinct()
            require(refusals.isEmpty()) {
                "HfCausalLm.load: $dir uses what CausalLM does not compute: ${refusals.joinToString("; ")}"
            }
            val qkNorm = c.family.defaultLayer.qkNorm
            val lm = CausalLmConfig(
                vocabSize = c.vocabSize, dModel = c.hiddenSize, numLayers = c.numLayers, numHeads = c.numHeads,
                ffHidden = c.intermediateSize, numKvHeads = c.numKvHeads, headDim = c.headDim,
                ropeTheta = c.ropeTheta.toFloat(), normEps = c.rmsNormEps.toFloat(),
                tiedEmbeddings = c.tieWordEmbeddings, qkNorm = qkNorm,
            )
            val values = HashMap<String, DTensor<*, F32>>()
            for ((role, key) in roleKeys(c)) {
                val t = ckpt.load(role)
                val raw = t.toF32Array()
                values[key] = if (isLinear(role)) {
                    DTensor<Shape, F32>(HostF32Storage(transpose(raw, t.dims[0], t.dims[1])), intArrayOf(t.dims[1], t.dims[0]), F32)
                } else {
                    DTensor<Shape, F32>(HostF32Storage(raw.copyOf()), t.dims.copyOf(), F32)
                }
            }
            fun dense(key: String) = Dense(values.getValue(key), null)
            fun norm(key: String, eps: Float) = RMSNorm(values.getValue(key), eps)
            val eps = lm.normEps
            val blocks = List(c.numLayers) { i ->
                val p = "blocks.$i"
                val attn = MultiHeadAttention(
                    dense("$p.attn.q.w"), dense("$p.attn.k.w"), dense("$p.attn.v.w"), dense("$p.attn.o.w"),
                    numHeads = lm.numHeads, numKvHeads = lm.numKvHeads, causal = true,
                    rope = RotaryEmbedding(lm.headDim, lm.ropeTheta),
                    qNorm = if (qkNorm) norm("$p.attn.qNorm.weight", eps) else null,
                    kNorm = if (qkNorm) norm("$p.attn.kNorm.weight", eps) else null,
                )
                TransformerBlock(
                    norm("$p.attnNorm.weight", eps), attn, norm("$p.mlpNorm.weight", eps),
                    SwiGLU(dense("$p.mlp.gate.w"), dense("$p.mlp.up.w"), dense("$p.mlp.down.w")),
                )
            }
            val model = CausalLM(
                Embedding(values.getValue("embed.table")),
                blocks,
                norm("norm.weight", eps),
                if (lm.tiedEmbeddings) null else dense("head.w"),
            )
            HfCausalLm(model, c)
        }

        /** Each Hugging Face weight role and the [CausalLM] parameter key holding it. */
        private fun roleKeys(c: HfDecoderConfig): List<Pair<DecoderWeightRole, String>> = buildList {
            add(DecoderWeightRole.EmbedTokens to "embed.table")
            for (i in 0 until c.numLayers) {
                for (part in c.layer(i).parts) {
                    val key = when (part) {
                        DecoderLayerPart.INPUT_LAYERNORM -> "attnNorm.weight"
                        DecoderLayerPart.Q_PROJ -> "attn.q.w"
                        DecoderLayerPart.K_PROJ -> "attn.k.w"
                        DecoderLayerPart.V_PROJ -> "attn.v.w"
                        DecoderLayerPart.O_PROJ -> "attn.o.w"
                        DecoderLayerPart.Q_NORM -> "attn.qNorm.weight"
                        DecoderLayerPart.K_NORM -> "attn.kNorm.weight"
                        DecoderLayerPart.POST_ATTENTION_LAYERNORM -> "mlpNorm.weight"
                        DecoderLayerPart.GATE_PROJ -> "mlp.gate.w"
                        DecoderLayerPart.UP_PROJ -> "mlp.up.w"
                        DecoderLayerPart.DOWN_PROJ -> "mlp.down.w"
                        else -> throw IllegalArgumentException("HfCausalLm: layer part $part has no CausalLM parameter")
                    }
                    add(DecoderWeightRole.Layer(i, part) to "blocks.$i.$key")
                }
            }
            add(DecoderWeightRole.FinalNorm to "norm.weight")
            if (!c.tieWordEmbeddings) add(DecoderWeightRole.LmHead to "head.w")
        }

        private fun isLinear(role: DecoderWeightRole): Boolean = when (role) {
            DecoderWeightRole.LmHead -> true
            is DecoderWeightRole.Layer -> role.part in setOf(
                DecoderLayerPart.Q_PROJ, DecoderLayerPart.K_PROJ, DecoderLayerPart.V_PROJ, DecoderLayerPart.O_PROJ,
                DecoderLayerPart.GATE_PROJ, DecoderLayerPart.UP_PROJ, DecoderLayerPart.DOWN_PROJ,
            )
            else -> false
        }

        /** `[rows, cols]` row-major to `[cols, rows]`. */
        private fun transpose(v: FloatArray, rows: Int, cols: Int): FloatArray {
            val out = FloatArray(v.size)
            for (r in 0 until rows) {
                val base = r * cols
                for (c in 0 until cols) out[c * rows + r] = v[base + c]
            }
            return out
        }
    }
}
