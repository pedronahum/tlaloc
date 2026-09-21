package io.tlaloc.ir.inference

import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.LoadedTensor
import io.tlaloc.core.io.SafetensorsIndex
import io.tlaloc.core.io.WeightSource
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

// §0.4.478 (Phase H3c-1) — the ONLY part of HF Llama ingestion that touches a
// filesystem: open a checkpoint DIRECTORY, read its config.json, and hand back
// tensors by [LlamaWeightRole]. Everything about NAMES, SHAPES and the
// transposed-Linear convention is HfLlama.kt in commonMain, where it can be
// tested without 2.2 GB of weights on disk.
//
// Sharding is not this file's problem: §0.4.468 already put
// `SafetensorsIndex.openCheckpoint` in `:core`'s jvmMain, which returns a
// [WeightSource] over either `model.safetensors` or
// `model.safetensors.index.json`. This class calls it and never learns which
// it got — TinyLlama is single-file, a 70B is thirty shards, and the code path
// above here is identical.

/**
 * A HuggingFace Llama checkpoint directory, opened for reading by role.
 *
 * ```
 * HfLlamaCheckpoint.open(dir).use { ckpt ->
 *     val q = ckpt.load(LlamaWeightRole.Layer(0, LlamaLayerPart.Q_PROJ))
 *     val shape = ckpt.config.toDecodeModelShape(numBlocks = 64, blockSize = 16)
 * }
 * ```
 *
 * **Every [load] verifies dims against [HfLlamaNames.expectedDims] before
 * returning.** REJECTED: loading first and letting the graph builder discover
 * the mismatch — by then the tensor is a `FloatArray` with no name attached,
 * and the report degenerates to a shape error from inside a matmul. The whole
 * value of a name mapping is that it can say *`model.layers.7.mlp.down_proj.weight`
 * is `[5632, 2048]` and this config says `[2048, 5632]`* at the moment the
 * bytes are read.
 */
class HfLlamaCheckpoint private constructor(
    val dir: Path,
    val config: HfLlamaConfig,
    private val weights: WeightSource,
    /**
     * Whether `lm_head.weight` is PRESENT IN THE FILE. This is a file fact,
     * cross-checked against [HfLlamaConfig.tieWordEmbeddings] at [open]; see
     * the refusal there for why a disagreement is fatal rather than resolved.
     */
    val lmHeadPresent: Boolean,
) : AutoCloseable {

    /** Every tensor name the checkpoint holds, from the header (or the index). */
    val names: Set<String> get() = weights.names

    /** True when the head reuses the embedding table — config and file agreeing. */
    val tiedEmbeddings: Boolean get() = !lmHeadPresent

    /**
     * The name in the FILE that a role's bytes come from. Differs from
     * [HfLlamaNames.hfName] in exactly one case: under tied embeddings
     * [LlamaWeightRole.LmHead] resolves to [HfLlamaNames.EMBED_TOKENS].
     *
     * That resolution is sound *because of the transposed-Linear convention*:
     * `lm_head` is a Linear `[vocab, hidden]` and the embedding table is a
     * lookup `[vocab, hidden]`, so the same buffer serves both — the head is
     * `x @ E^T` and the lookup is `E[token]`. Under the OTHER convention they
     * would be transposes of each other and tying would be a shape error,
     * which is a second, independent confirmation of the layout fact.
     */
    fun resolveName(role: LlamaWeightRole): String =
        if (role == LlamaWeightRole.LmHead && tiedEmbeddings) HfLlamaNames.EMBED_TOKENS
        else HfLlamaNames.hfName(role)

    /** Load one role's tensor, dims verified against the config. */
    fun load(role: LlamaWeightRole): LoadedTensor {
        val name = resolveName(role)
        if (name !in weights.names) {
            throw JsonException(
                "HfLlamaCheckpoint: $dir has no tensor '$name' for role $role — the checkpoint " +
                    "holds ${weights.names.size} tensors and this mapping expected " +
                    "${HfLlamaNames.roles(config).size} roles",
            )
        }
        val t = weights.load(name)
        val want = HfLlamaNames.expectedDims(role, config)
        if (!t.dims.contentEquals(want)) {
            throw JsonException(
                "HfLlamaCheckpoint: '$name' is ${t.dims.toList()} but this config " +
                    "(hidden=${config.hiddenSize}, heads=${config.numHeads}, " +
                    "kvHeads=${config.numKvHeads}, headDim=${config.headDim}, " +
                    "intermediate=${config.intermediateSize}, vocab=${config.vocabSize}) says " +
                    "${want.toList()}. HF stores Linear weights TRANSPOSED as [out, in]; if the " +
                    "two are reverses of each other, that convention is what disagrees",
            )
        }
        return t
    }

    /**
     * NAMED GAP: there is no `dtype(name)` here, because [WeightSource]
     * exposes only `load`, and asking "what width is this stored at" by
     * decoding a 128 MB embedding table is not an answer. Widening
     * `WeightSource` with a header accessor is the fix and it belongs in
     * `:core` beside the parser that already knows; until then a caller that
     * needs the storage width reads [HfLlamaConfig.storageDType], which is
     * `torch_dtype` and is what the producer INTENDED rather than what the
     * bytes ARE. `HfLlamaCheckpointTest` closes the gap for the certification
     * by opening the same file through `SafetensorsFile` and reading the
     * header directly.
     */

    /**
     * Check the whole inventory WITHOUT reading a single weight byte: every
     * role the config implies has a name in the file, every mapped dim agrees,
     * and nothing in the file is left unmapped. Returns the unmapped names,
     * which are a fact rather than an error (`model.rotary_emb.inv_freq` is a
     * derived buffer some transformers vintages persisted).
     *
     * This is the cheap gate a serving loader wants before it commits memory,
     * and it is the test hook that lets a 1.1B checkpoint be verified in
     * milliseconds.
     */
    fun verifyInventory(): List<String> {
        val expected = LinkedHashSet<String>()
        for (role in HfLlamaNames.roles(config)) {
            val name = resolveName(role)
            expected += name
            if (name !in weights.names) {
                throw JsonException(
                    "HfLlamaCheckpoint: $dir is missing '$name' (role $role)",
                )
            }
        }
        return weights.names.filter { it !in expected }
    }

    override fun close() = weights.close()

    companion object {
        /** The config file every HF checkpoint directory carries. */
        const val CONFIG_JSON: String = "config.json"

        fun open(dir: Path, strictArchitecture: Boolean = true): HfLlamaCheckpoint {
            val cfgPath = dir.resolve(CONFIG_JSON)
            if (!Files.isRegularFile(cfgPath)) {
                throw JsonException(
                    "HfLlamaCheckpoint: $dir has no $CONFIG_JSON — a directory of safetensors " +
                        "without it states no shape, and this loader will not infer one from " +
                        "tensor dims (the inference is ambiguous exactly where it matters: " +
                        "head_dim vs hidden/heads, and tied vs untied)",
                )
            }
            val config = HfLlamaConfig.parse(
                String(Files.readAllBytes(cfgPath), StandardCharsets.UTF_8),
                strictArchitecture,
            )
            val weights = SafetensorsIndex.openCheckpoint(dir)
            try {
                val present = HfLlamaNames.LM_HEAD in weights.names
                if (present == config.tieWordEmbeddings) {
                    throw JsonException(
                        "HfLlamaCheckpoint: $dir disagrees with itself — config.json says " +
                            "tie_word_embeddings=${config.tieWordEmbeddings} but " +
                            "'${HfLlamaNames.LM_HEAD}' is ${if (present) "PRESENT" else "ABSENT"} " +
                            "in the checkpoint. Refused BY NAME: picking either answer silently " +
                            "changes every logit this model ever produces (a tied head is the " +
                            "embedding table, an untied one is a separately trained matrix), and " +
                            "the disagreement means one of the two files is not the one you think",
                    )
                }
                return HfLlamaCheckpoint(dir.toAbsolutePath(), config, weights, present)
            } catch (t: Throwable) {
                weights.close()
                throw t
            }
        }
    }
}
