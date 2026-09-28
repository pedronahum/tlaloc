package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue

// transformers 5 does not run tokenizer.json as written. A tokenizer class
// with its own __init__ (LlamaTokenizer, Qwen2Tokenizer, GPT2Tokenizer,
// GemmaTokenizer) takes only the vocabulary, the merges and the
// post-processor from the file and builds the normalizer, pre-tokenizer,
// decoder and BPE flags itself. For TinyLlama that changes the ids: the
// file's Prepend("▁") normalizer becomes a Metaspace pre-tokenizer that
// prepends only at the start of the input, so "<s>hi" is [1, 2918] under
// AutoTokenizer and [1, 7251] under the file. The rules below reproduce
// those constructors, read from transformers 5.17.0.

/** The parts of tokenizer_config.json that change what AutoTokenizer does. */
internal class TokenizerConfig(private val o: JsonObject) {
    val tokenizerClass: String? = (o["tokenizer_class"] as? JsonString)?.value

    fun has(key: String): Boolean = o.fields.containsKey(key)

    fun bool(key: String): Boolean? = when (val v = o[key]) {
        is JsonBool -> v.value
        null, JsonNull -> null
        else -> unsupported("tokenizer_config.json '$key' is not a boolean")
    }

    /** A named token: a string, `{"content": ...}`, or null when set to null. */
    fun token(key: String): String? = tokenValue(o[key], key)

    /** Every token tokenizer_config.json names as special: `*_token` fields and the special-token lists. */
    fun namedSpecialTokens(): List<Pair<String, String>> {
        val out = ArrayList<Pair<String, String>>()
        for ((k, v) in o.fields) {
            when {
                k.endsWith("_token") && (v is JsonString || v is JsonObject) -> tokenValue(v, k)?.let { out.add(k to it) }
                k == "additional_special_tokens" || k == "extra_special_tokens" -> when (v) {
                    is JsonArray -> v.elements.forEachIndexed { i, e -> tokenValue(e, "$k[$i]")?.let { out.add(k to it) } }
                    is JsonObject -> v.fields.forEach { (name, e) -> tokenValue(e, "$k.$name")?.let { out.add(k to it) } }
                    JsonNull -> {}
                    else -> unsupported("tokenizer_config.json '$k' is neither a list nor an object")
                }
            }
        }
        return out
    }

    val chatTemplate: String? = when (val v = o["chat_template"]) {
        is JsonString -> v.value
        null, JsonNull -> null
        else -> unsupported("tokenizer_config.json 'chat_template' is not a single template string")
    }

    private fun tokenValue(v: JsonValue?, what: String): String? = when (v) {
        null, JsonNull -> null
        is JsonString -> v.value
        is JsonObject -> (v["content"] as? JsonString)?.value ?: unsupported("tokenizer_config.json '$what' has no content")
        else -> unsupported("tokenizer_config.json '$what' is not a token")
    }
}

internal const val QWEN2_PATTERN =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

private const val SPIECE = '▁'

/** Class defaults for the named tokens a config may leave out. */
internal fun classDefaultToken(tokenizerClass: String?, key: String): String? = when (tokenizerClass?.removeSuffix("Fast")) {
    "LlamaTokenizer" -> mapOf("unk_token" to "<unk>", "bos_token" to "<s>", "eos_token" to "</s>")[key]
    "Qwen2Tokenizer" -> mapOf("unk_token" to "<|endoftext|>", "eos_token" to "<|endoftext|>", "pad_token" to "<|endoftext|>")[key]
    "GPT2Tokenizer" -> mapOf("unk_token" to "<|endoftext|>", "bos_token" to "<|endoftext|>", "eos_token" to "<|endoftext|>")[key]
    "GemmaTokenizer" -> mapOf(
        "unk_token" to "<unk>", "bos_token" to "<bos>", "eos_token" to "<eos>",
        "pad_token" to "<pad>", "mask_token" to "<mask>",
    )[key]
    else -> null
}

/** Rewrites [spec] the way transformers' tokenizer class [config] names would build it. */
internal fun applyTransformersRules(spec: PipelineSpec, config: TokenizerConfig) {
    val m = spec.model
    when (val cls = config.tokenizerClass?.removeSuffix("Fast")) {
        null, "TokenizersBackend", "PreTrainedTokenizer" -> {}
        "Qwen2Tokenizer" -> {
            spec.normalizer = UnicodeNormalizer(UnicodeForm.NFC)
            spec.preTokenizer = SequencePreTokenizer(
                listOf(
                    SplitPreTokenizer(compileSplitRegex(QWEN2_PATTERN), SplitBehavior.ISOLATED),
                    ByteLevelPreTokenizer(addPrefixSpace = config.bool("add_prefix_space") ?: false, useRegex = false),
                ),
            )
            spec.decoder = ByteLevelDecoder
            m.unkToken = null; m.fuseUnk = false; m.byteFallback = false; m.ignoreMerges = false
        }
        "GPT2Tokenizer" -> {
            spec.normalizer = null
            spec.preTokenizer = ByteLevelPreTokenizer(addPrefixSpace = config.bool("add_prefix_space") ?: false, useRegex = true)
            spec.decoder = ByteLevelDecoder
            m.unkToken = null; m.fuseUnk = false; m.byteFallback = false; m.ignoreMerges = false
        }
        "LlamaTokenizer" -> {
            val addPrefixSpace = config.bool("add_prefix_space") ?: true
            val legacy = config.bool("legacy") ?: false
            val scheme = when {
                !addPrefixSpace -> PrependScheme.NEVER
                legacy -> PrependScheme.ALWAYS
                else -> PrependScheme.FIRST
            }
            spec.normalizer = null
            spec.preTokenizer = MetaspacePreTokenizer(SPIECE, scheme, split = false)
            val steps = mutableListOf(ReplaceDecoder(SPIECE.toString(), " "), ByteFallbackDecoder, FuseDecoder)
            if (addPrefixSpace) steps += StripDecoder(' ', 1, 0)
            spec.decoder = SequenceDecoder(steps)
            m.unkToken = null; m.fuseUnk = true; m.byteFallback = true; m.ignoreMerges = false
        }
        "GemmaTokenizer" -> {
            spec.normalizer = ReplaceNormalizer(" ", SPIECE.toString())
            spec.preTokenizer = SplitPreTokenizer(literalRegex(" "), SplitBehavior.MERGED_WITH_PREVIOUS)
            spec.decoder = SequenceDecoder(listOf(ReplaceDecoder(SPIECE.toString(), " "), ByteFallbackDecoder, FuseDecoder))
            m.unkToken = if (config.has("unk_token")) config.token("unk_token") else "<unk>"
            m.fuseUnk = true; m.byteFallback = true; m.ignoreMerges = false
        }
        else -> unsupported(
            "tokenizer_class '$cls' is not implemented: transformers may build its pipeline differently from " +
                "tokenizer.json. HfTokenizer.fromJson without a config gives the file's own pipeline.",
        )
    }

    // Named special tokens missing from added_tokens would be added by
    // transformers at load time, possibly with new ids. Refused rather than guessed.
    val added = spec.added.mapTo(HashSet()) { it.content }
    for ((key, token) in config.namedSpecialTokens()) {
        if (token !in added) unsupported("tokenizer_config.json '$key' names '$token', which is not in added_tokens")
    }

    // An explicit add_bos_token / add_eos_token makes transformers replace the
    // post-processor with a template of its own.
    if (config.has("add_bos_token") || config.has("add_eos_token")) {
        fun idOf(key: String): Int? {
            val t = (if (config.has(key)) config.token(key) else classDefaultToken(config.tokenizerClass, key)) ?: return null
            return spec.added.firstOrNull { it.content == t }?.id ?: m.vocab[t]
                ?: unsupported("tokenizer_config.json '$key' '$t' has no id")
        }
        val bos = if (config.bool("add_bos_token") == true) idOf("bos_token") else null
        val eos = if (config.bool("add_eos_token") == true) idOf("eos_token") else null
        spec.postProcessor = PostProcessor(listOfNotNull(bos).toIntArray(), listOfNotNull(eos).toIntArray())
    }
}
