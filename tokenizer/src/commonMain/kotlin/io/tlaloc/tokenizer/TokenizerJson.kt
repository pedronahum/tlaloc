package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue

/** A tokenizer.json or tokenizer_config.json this module does not implement, named. */
class UnsupportedTokenizerException(message: String) : IllegalArgumentException(message)

internal fun unsupported(message: String): Nothing = throw UnsupportedTokenizerException(message)

internal class ModelSpec(
    val vocab: HashMap<String, Int>,
    val merges: List<Pair<String, String>>,
    var unkToken: String?,
    var fuseUnk: Boolean,
    var byteFallback: Boolean,
    var ignoreMerges: Boolean,
)

internal class PipelineSpec(
    val model: ModelSpec,
    var normalizer: Normalizer?,
    var preTokenizer: PreTokenizer?,
    var postProcessor: PostProcessor,
    var decoder: Decoder?,
    val added: List<AddedToken>,
)

private fun JsonObject.bool(key: String, default: Boolean): Boolean = when (val v = this[key]) {
    null, JsonNull -> default
    is JsonBool -> v.value
    else -> unsupported("'$key' is not a boolean")
}

private fun JsonObject.optStr(key: String): String? = when (val v = this[key]) {
    null, JsonNull -> null
    is JsonString -> v.value
    else -> unsupported("'$key' is not a string")
}

private fun JsonObject.int(key: String, default: Int): Int = when (val v = this[key]) {
    null, JsonNull -> default
    is JsonNumber -> v.asInt(key)
    else -> unsupported("'$key' is not a number")
}

private fun JsonValue.objOrNull(): JsonObject? = when (this) {
    JsonNull -> null
    is JsonObject -> this
    else -> unsupported("expected an object or null")
}

private fun JsonObject.type(what: String): String = optStr("type") ?: unsupported("$what without a 'type'")

private fun JsonObject.checkKeys(what: String, allowed: Set<String>) {
    for (k in fields.keys) if (k !in allowed) unsupported("$what: field '$k' is not implemented")
}

internal fun parsePipeline(root: JsonObject): PipelineSpec {
    for (k in listOf("truncation", "padding")) {
        val v = root[k]
        if (v != null && v != JsonNull) unsupported("tokenizer.json sets '$k'; truncation and padding are not implemented")
    }
    val model = parseModel(root.obj("model"))
    val added = (root["added_tokens"] as? JsonArray)?.elements.orEmpty().map { parseAddedToken(it as JsonObject) }
    return PipelineSpec(
        model = model,
        normalizer = root["normalizer"]?.objOrNull()?.let(::parseNormalizer),
        preTokenizer = root["pre_tokenizer"]?.objOrNull()?.let(::parsePreTokenizer),
        postProcessor = root["post_processor"]?.objOrNull()?.let(::parsePostProcessor)
            ?: PostProcessor.IDENTITY,
        decoder = root["decoder"]?.objOrNull()?.let(::parseDecoder),
        added = added,
    )
}

private fun parseModel(m: JsonObject): ModelSpec {
    // gpt2's tokenizer.json predates the 'type' field; tokenizers reads it as BPE.
    val type = m.optStr("type") ?: "BPE"
    if (type != "BPE") unsupported("model type '$type' is not implemented; only BPE is")
    m.checkKeys(
        "BPE model",
        setOf(
            "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix",
            "fuse_unk", "byte_fallback", "ignore_merges", "vocab", "merges",
        ),
    )
    val dropout = m["dropout"]
    if (dropout != null && dropout != JsonNull) unsupported("BPE dropout is not implemented")
    for (k in listOf("continuing_subword_prefix", "end_of_word_suffix")) {
        val v = m.optStr(k)
        if (!v.isNullOrEmpty()) unsupported("BPE '$k' = '$v' is not implemented")
    }
    val vocabJson = m.obj("vocab")
    val vocab = HashMap<String, Int>(vocabJson.fields.size * 2)
    for ((token, id) in vocabJson.fields) {
        vocab[token] = (id as? JsonNumber)?.asInt("vocab['$token']") ?: unsupported("vocab['$token'] is not a number")
    }
    val merges = m.arr("merges").elements.mapIndexed { i, e ->
        when (e) {
            is JsonString -> {
                val parts = e.value.split(' ')
                if (parts.size != 2) unsupported("merge $i '${e.value}' is not two tokens separated by one space")
                parts[0] to parts[1]
            }
            is JsonArray -> {
                if (e.size != 2) unsupported("merge $i does not have two tokens")
                ((e[0] as? JsonString)?.value ?: unsupported("merge $i is not strings")) to
                    ((e[1] as? JsonString)?.value ?: unsupported("merge $i is not strings"))
            }
            else -> unsupported("merge $i is neither a string nor a pair")
        }
    }
    return ModelSpec(
        vocab = vocab,
        merges = merges,
        unkToken = m.optStr("unk_token"),
        fuseUnk = m.bool("fuse_unk", false),
        byteFallback = m.bool("byte_fallback", false),
        ignoreMerges = m.bool("ignore_merges", false),
    )
}

private fun parseAddedToken(o: JsonObject): AddedToken {
    val content = o.str("content")
    for (k in listOf("lstrip", "rstrip", "single_word")) {
        if (o.bool(k, false)) unsupported("added token '$content' sets $k, which is not implemented")
    }
    return AddedToken(
        id = o.int("id", -1).also { if (it < 0) unsupported("added token '$content' has no id") },
        content = content,
        special = o.bool("special", false),
        normalized = o.bool("normalized", true),
    )
}

/** A `pattern` object: `{"String": s}` or `{"Regex": r}`. */
private fun parsePattern(o: JsonObject, what: String): Pair<String, Boolean> {
    o.optStr("String")?.let { return it to false }
    o.optStr("Regex")?.let { return it to true }
    unsupported("$what: pattern is neither String nor Regex")
}

internal fun parseNormalizer(o: JsonObject): Normalizer = when (val t = o.type("normalizer")) {
    "NFC" -> UnicodeNormalizer(UnicodeForm.NFC)
    "NFD" -> UnicodeNormalizer(UnicodeForm.NFD)
    "NFKC" -> UnicodeNormalizer(UnicodeForm.NFKC)
    "NFKD" -> UnicodeNormalizer(UnicodeForm.NFKD)
    "Replace" -> {
        val (p, isRegex) = parsePattern(o.obj("pattern"), "Replace normalizer")
        if (isRegex) unsupported("Replace normalizer with a Regex pattern is not implemented")
        ReplaceNormalizer(p, o.str("content"))
    }
    "Prepend" -> PrependNormalizer(o.str("prepend"))
    "Sequence" -> SequenceNormalizer(o.arr("normalizers").elements.map { parseNormalizer(it as JsonObject) })
    else -> unsupported("normalizer '$t' is not implemented")
}

internal fun parsePreTokenizer(o: JsonObject): PreTokenizer = when (val t = o.type("pre_tokenizer")) {
    "Sequence" -> SequencePreTokenizer(o.arr("pretokenizers").elements.map { parsePreTokenizer(it as JsonObject) })
    "Split" -> {
        if (o.bool("invert", false)) unsupported("Split with invert=true is not implemented")
        val (p, isRegex) = parsePattern(o.obj("pattern"), "Split pre-tokenizer")
        val behavior = when (val b = o.str("behavior")) {
            "Isolated" -> SplitBehavior.ISOLATED
            "Removed" -> SplitBehavior.REMOVED
            "MergedWithPrevious" -> SplitBehavior.MERGED_WITH_PREVIOUS
            "MergedWithNext" -> SplitBehavior.MERGED_WITH_NEXT
            else -> unsupported("Split behavior '$b' is not implemented")
        }
        SplitPreTokenizer(if (isRegex) compileSplitRegex(p) else literalRegex(p), behavior)
    }
    "ByteLevel" -> ByteLevelPreTokenizer(o.bool("add_prefix_space", true), o.bool("use_regex", true))
    "Metaspace" -> MetaspacePreTokenizer(replacementChar(o), prependScheme(o), o.bool("split", true))
    else -> unsupported("pre_tokenizer '$t' is not implemented")
}

private fun replacementChar(o: JsonObject): Char {
    val r = o.str("replacement")
    if (r.length != 1) unsupported("Metaspace replacement '$r' is not one UTF-16 character")
    return r[0]
}

private fun prependScheme(o: JsonObject): PrependScheme = when (val s = o.optStr("prepend_scheme")) {
    "always" -> PrependScheme.ALWAYS
    "first" -> PrependScheme.FIRST
    "never" -> PrependScheme.NEVER
    null -> if (o.bool("add_prefix_space", true)) PrependScheme.ALWAYS else PrependScheme.NEVER
    else -> unsupported("Metaspace prepend_scheme '$s' is not implemented")
}

internal fun parsePostProcessor(o: JsonObject): PostProcessor =
    when (val t = o.type("post_processor")) {
        // ByteLevel's post-processing only trims offsets; ids pass through.
        "ByteLevel" -> PostProcessor.IDENTITY
        "TemplateProcessing" -> {
            val specials = o.obj("special_tokens")
            val prefix = IntList()
            val suffix = IntList()
            var seenA = false
            for (item in o.arr("single").elements) {
                val io = item as JsonObject
                val seq = io["Sequence"] as? JsonObject
                val special = io["SpecialToken"] as? JsonObject
                when {
                    seq != null -> {
                        if (seq.str("id") != "A") unsupported("TemplateProcessing single template names sequence '${seq.str("id")}'")
                        if (seenA) unsupported("TemplateProcessing single template has \$A twice")
                        seenA = true
                    }
                    special != null -> {
                        val name = special.str("id")
                        val ids = specials.obj(name).arr("ids").asIntList("special_tokens['$name'].ids")
                        for (id in ids) (if (seenA) suffix else prefix).add(id)
                    }
                    else -> unsupported("TemplateProcessing item $io is not implemented")
                }
            }
            if (!seenA) unsupported("TemplateProcessing single template has no \$A")
            PostProcessor(prefix.toIntArray(), suffix.toIntArray())
        }
        else -> unsupported("post_processor '$t' is not implemented")
    }

internal fun parseDecoder(o: JsonObject): Decoder = when (val t = o.type("decoder")) {
    // add_prefix_space, trim_offsets and use_regex do not change ByteLevel decoding.
    "ByteLevel" -> ByteLevelDecoder
    "Replace" -> {
        val (p, isRegex) = parsePattern(o.obj("pattern"), "Replace decoder")
        if (isRegex) unsupported("Replace decoder with a Regex pattern is not implemented")
        ReplaceDecoder(p, o.str("content"))
    }
    "ByteFallback" -> ByteFallbackDecoder
    "Fuse" -> FuseDecoder
    "Strip" -> {
        val c = o.str("content")
        if (c.length != 1) unsupported("Strip decoder content '$c' is not one character")
        StripDecoder(c[0], o.int("start", 0), o.int("stop", 0))
    }
    "Metaspace" -> MetaspaceDecoder(replacementChar(o), prependScheme(o))
    "Sequence" -> SequenceDecoder(o.arr("decoders").elements.map { parseDecoder(it as JsonObject) })
    else -> unsupported("decoder '$t' is not implemented")
}
