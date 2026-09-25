package io.tlaloc.ir.inference

import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.LoadedTensor
import io.tlaloc.core.io.SafetensorsEntry
import io.tlaloc.core.io.SafetensorsIndex
import io.tlaloc.core.io.WeightSource
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

// The only part of HF checkpoint ingestion that touches a filesystem: open a
// checkpoint directory, read its config.json, and hand back tensors by
// [DecoderWeightRole]. Names, shapes and the [out, in] Linear layout are
// HfDecoder.kt in commonMain. Sharding is SafetensorsIndex.openCheckpoint's
// job: it returns one WeightSource over `model.safetensors` or
// `model.safetensors.index.json`.

/**
 * A HuggingFace decoder-only checkpoint directory (any [HfModelFamily]),
 * opened for reading by role.
 *
 * ```
 * HfCheckpoint.open(dir).use { ckpt ->
 *     val q = ckpt.load(DecoderWeightRole.Layer(0, DecoderLayerPart.Q_PROJ))
 *     val shape = ckpt.config.toDecodeModelShape(numBlocks = 64, blockSize = 16)
 * }
 * ```
 *
 * **Every [load] verifies dims against [HfDecoderNames.expectedDims] before
 * returning.** REJECTED: loading first and letting the graph builder discover
 * the mismatch — by then the tensor is a `FloatArray` with no name attached,
 * and the report degenerates to a shape error from inside a matmul. The whole
 * value of a name mapping is that it can say *`model.layers.7.mlp.down_proj.weight`
 * is `[5632, 2048]` and this config says `[2048, 5632]`* at the moment the
 * bytes are read.
 */
class HfCheckpoint private constructor(
    val dir: Path,
    val config: HfDecoderConfig,
    private val weights: WeightSource,
    /**
     * Whether `lm_head.weight` is present in the file. A tied checkpoint may
     * carry it too (Qwen3-0.6B does, as a copy of the embedding table); see
     * [open] for what is refused.
     */
    val lmHeadPresent: Boolean,
) : AutoCloseable {

    /** Every tensor name the checkpoint holds, from the header (or the index). */
    val names: Set<String> get() = weights.names

    /** True when the head reuses the embedding table, as config.json says. */
    val tiedEmbeddings: Boolean get() = config.tieWordEmbeddings

    /**
     * The name in the file that a role's bytes come from. Under tied
     * embeddings [DecoderWeightRole.LmHead] resolves to
     * [HfDecoderNames.EMBED_TOKENS], which is what transformers does: it ties
     * `lm_head.weight` to the embedding table and ignores a stored copy.
     * That is sound because both are `[vocab, hidden]` (the head is `x @ E^T`,
     * the lookup is `E[token]`).
     */
    fun resolveName(role: DecoderWeightRole): String =
        if (role == DecoderWeightRole.LmHead && tiedEmbeddings) {
            HfDecoderNames.hfName(DecoderWeightRole.EmbedTokens, config.family)
        } else {
            HfDecoderNames.hfName(role, config.family)
        }

    /** Load one role's tensor, dims verified against the config. */
    fun load(role: DecoderWeightRole): LoadedTensor {
        val name = resolveName(role)
        if (name !in weights.names) {
            throw JsonException(
                "HfCheckpoint: $dir has no tensor '$name' for role $role — the checkpoint " +
                    "holds ${weights.names.size} tensors and this mapping expected " +
                    "${HfDecoderNames.roles(config).size} roles",
            )
        }
        val t = weights.load(name)
        val want = HfDecoderNames.expectedDims(role, config)
        if (!t.dims.contentEquals(want)) {
            throw JsonException(
                "HfCheckpoint: '$name' is ${t.dims.toList()} but this config " +
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
     * The header entry of one role's tensor, dims verified against the
     * config, without reading its bytes. Pair with [readBytes] to read a
     * tensor too large for one JVM array.
     */
    fun entry(role: DecoderWeightRole): SafetensorsEntry {
        val name = resolveName(role)
        if (name !in weights.names) {
            throw JsonException("HfCheckpoint: $dir has no tensor '$name' for role $role")
        }
        val e = weights.entry(name)
        val want = HfDecoderNames.expectedDims(role, config)
        if (e.dims != want.toList()) {
            throw JsonException(
                "HfCheckpoint: '$name' is ${e.dims} but this config says ${want.toList()}",
            )
        }
        return e
    }

    /** Raw bytes of one role's tensor, [length] of them from [byteOffset]; see [WeightSource.readBytes]. */
    fun readBytes(role: DecoderWeightRole, byteOffset: Long, into: ByteArray, offset: Int = 0, length: Int = into.size - offset) =
        weights.readBytes(resolveName(role), byteOffset, into, offset, length)

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
        for (role in HfDecoderNames.roles(config)) {
            val name = resolveName(role)
            expected += name
            if (name !in weights.names) {
                throw JsonException(
                    "HfCheckpoint: $dir is missing '$name' (role $role)",
                )
            }
        }
        // A tied checkpoint's stored copy of the head is accounted for by
        // verifyTiedHead, not left over.
        if (tiedEmbeddings && lmHeadPresent) expected += HfDecoderNames.LM_HEAD
        return weights.names.filter { it !in expected }
    }

    /**
     * For a tied checkpoint that also stores `lm_head.weight`: check that the
     * stored head is bit-for-bit the embedding table. transformers ignores
     * the stored copy of a tied head, and so does [resolveName]; if the two
     * differ, the file's head is not the one config.json describes and
     * serving either would disagree with somebody, so it is refused by name.
     * A no-op for an untied checkpoint or a tied one without the copy.
     */
    fun verifyTiedHead() {
        if (!tiedEmbeddings || !lmHeadPresent) return
        val head = weights.load(HfDecoderNames.LM_HEAD)
        val table = weights.load(HfDecoderNames.hfName(DecoderWeightRole.EmbedTokens, config.family))
        val a = head.toF32Array()
        val b = table.toF32Array()
        val same = head.dims.contentEquals(table.dims) && a.size == b.size &&
            a.indices.all { a[it].toRawBits() == b[it].toRawBits() }
        if (!same) {
            throw JsonException(
                "HfCheckpoint: $dir says tie_word_embeddings=true and also stores " +
                    "'${HfDecoderNames.LM_HEAD}', but the stored head is not the embedding " +
                    "table. Refused BY NAME: transformers would use the table and ignore the " +
                    "stored head, and a server reading either one would disagree with the other",
            )
        }
    }

    override fun close() = weights.close()

    companion object {
        /** The config file every HF checkpoint directory carries. */
        const val CONFIG_JSON: String = "config.json"

        fun open(dir: Path, strictArchitecture: Boolean = true): HfCheckpoint {
            val cfgPath = dir.resolve(CONFIG_JSON)
            if (!Files.isRegularFile(cfgPath)) {
                throw JsonException(
                    "HfCheckpoint: $dir has no $CONFIG_JSON — a directory of safetensors " +
                        "without it states no shape, and this loader will not infer one from " +
                        "tensor dims (the inference is ambiguous exactly where it matters: " +
                        "head_dim vs hidden/heads, and tied vs untied)",
                )
            }
            val config = HfDecoderConfig.parse(
                String(Files.readAllBytes(cfgPath), StandardCharsets.UTF_8),
                strictArchitecture,
            )
            val weights = SafetensorsIndex.openCheckpoint(dir)
            try {
                val present = HfDecoderNames.LM_HEAD in weights.names
                if (!present && !config.tieWordEmbeddings) {
                    throw JsonException(
                        "HfCheckpoint: $dir disagrees with itself — config.json says " +
                            "tie_word_embeddings=false but '${HfDecoderNames.LM_HEAD}' is ABSENT " +
                            "in the checkpoint. Refused BY NAME: an untied head is a separately " +
                            "trained matrix, and reading the embedding table in its place would " +
                            "change every logit this model produces",
                    )
                }
                return HfCheckpoint(dir.toAbsolutePath(), config, weights, present)
            } catch (t: Throwable) {
                weights.close()
                throw t
            }
        }
    }
}
