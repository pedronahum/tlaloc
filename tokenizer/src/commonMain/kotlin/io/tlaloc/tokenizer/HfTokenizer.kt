package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson

/**
 * A Hugging Face BPE tokenizer: encode, decode, streaming decode and chat
 * templates, giving the ids and strings transformers' `AutoTokenizer` gives.
 *
 * Built from a checkpoint's `tokenizer.json` and, when present, its
 * `tokenizer_config.json`. With the config, the tokenizer class it names
 * decides the pipeline the way transformers 5 decides it (see [fromJson]).
 * Checked id for id against transformers 5.17 / tokenizers 0.23 for Qwen3,
 * Muse Glimmer, GPT-2, TinyLlama and Gemma 4.
 *
 * Instances are immutable apart from an internal word cache and are safe to
 * share between threads. A [DecodeStream] is not.
 */
class HfTokenizer private constructor(
    private val model: Bpe,
    private val normalizer: Normalizer?,
    private val preTokenizer: PreTokenizer?,
    private val postProcessor: PostProcessor,
    private val decoder: Decoder?,
    added: List<AddedToken>,
    private val specialTokens: Map<String, String?>,
    private val templateSource: String?,
    private val splitSpecialByDefault: Boolean,
) {
    private val addedById = HashMap<Int, AddedToken>()
    private val addedByContent = HashMap<String, AddedToken>()
    private val rawMatcher: TokenMatcher
    private val normalizedMatcher: TokenMatcher
    private val modelTokens: Array<String?>

    init {
        for (t in added) {
            addedById[t.id] = t
            addedByContent[t.content] = t
        }
        rawMatcher = TokenMatcher(added.filter { !it.normalized }.map { it.content to it.id })
        normalizedMatcher = TokenMatcher(
            added.filter { it.normalized }.map { (normalizer?.normalize(it.content) ?: it.content) to it.id },
        )
        val maxId = model.vocab.values.maxOrNull() ?: -1
        modelTokens = arrayOfNulls(maxId + 1)
        for ((token, id) in model.vocab) modelTokens[id] = token
    }

    /** Number of distinct ids: the model's vocabulary plus added tokens outside it (`len(tokenizer)`). */
    val vocabSize: Int = HashSet<Int>(model.vocab.values).apply { added.forEach { add(it.id) } }.size

    /** The BOS token tokenizer_config.json names (or the tokenizer class's default), if any. */
    val bosToken: String? get() = specialTokens["bos_token"]

    /** The EOS token tokenizer_config.json names (or the tokenizer class's default), if any. */
    val eosToken: String? get() = specialTokens["eos_token"]

    val padToken: String? get() = specialTokens["pad_token"]

    val bosTokenId: Int? get() = bosToken?.let(::tokenToId)

    val eosTokenId: Int? get() = eosToken?.let(::tokenToId)

    /** The recognized chat template of this checkpoint, or null when it has none or it is not one [ChatTemplate] renders. */
    val chatTemplate: ChatTemplate? = templateSource?.let(ChatTemplate::recognize)

    fun tokenToId(token: String): Int? = addedByContent[token]?.id ?: model.vocab[token]

    fun idToToken(id: Int): String? = addedById[id]?.content ?: modelTokens.getOrNull(id)

    /** True for ids of added tokens marked special (the ones `skipSpecialTokens` removes). */
    fun isSpecial(id: Int): Boolean = addedById[id]?.special == true

    /**
     * The ids of [text], as transformers' `tokenizer(text).input_ids`.
     *
     * @param addSpecialTokens add what the post-processor adds (TinyLlama's
     *   and Muse Glimmer's BOS); transformers' `add_special_tokens`.
     * @param allowSpecial when false, special-token strings in [text] are
     *   tokenized as ordinary text, so untrusted input cannot inject a
     *   `<|im_end|>`; transformers' `split_special_tokens=True`. Added tokens
     *   that are not special (Qwen3's `<think>`) are still recognized.
     * @throws IllegalArgumentException if [text] contains an unpaired surrogate.
     */
    fun encode(text: String, addSpecialTokens: Boolean = true, allowSpecial: Boolean = !splitSpecialByDefault): IntArray {
        requireWellFormed(text)
        val keep: (Int) -> Boolean = if (allowSpecial) ALWAYS else { id -> addedById[id]?.special != true }
        val segments = ArrayList<Segment>()
        splitOnTokens(text, true, rawMatcher, keep, segments) { raw, atOrigin, out ->
            val normalized = normalizer?.normalize(raw) ?: raw
            splitOnTokens(normalized, atOrigin, normalizedMatcher, keep, out) { t, o, out2 -> out2.add(Segment(t, o, -1)) }
        }
        val ids = IntList(text.length / 3 + 8)
        for (seg in segments) {
            if (seg.id >= 0) { ids.add(seg.id); continue }
            val pieces = preTokenizer?.split(listOf(Piece(seg.text, seg.atOrigin))) ?: listOf(Piece(seg.text, seg.atOrigin))
            for (p in pieces) model.tokenize(p.text, ids)
        }
        val out = ids.toIntArray()
        return if (addSpecialTokens) postProcessor.apply(out) else out
    }

    /**
     * The text of [ids], as transformers' `tokenizer.decode(ids)`. Ids with no
     * token are dropped, as tokenizers drops them.
     */
    fun decode(ids: IntArray, skipSpecialTokens: Boolean = false): String {
        val tokens = ArrayList<String>(ids.size)
        for (id in ids) {
            val added = addedById[id]
            if (added != null) {
                if (!(skipSpecialTokens && added.special)) tokens.add(added.content)
                continue
            }
            val t = modelTokens.getOrNull(id) ?: continue
            // A model token whose text is also a special added token is skipped too.
            if (skipSpecialTokens && addedByContent[t]?.special == true) continue
            tokens.add(t)
        }
        val d = decoder ?: return tokens.joinToString(" ")
        return d.decodeChain(tokens).joinToString("")
    }

    fun decode(ids: List<Int>, skipSpecialTokens: Boolean = false): String = decode(ids.toIntArray(), skipSpecialTokens)

    /** A decoder for ids arriving one at a time; see [DecodeStream]. */
    fun decodeStream(skipSpecialTokens: Boolean = false): DecodeStream = DecodeStream(this, skipSpecialTokens)

    /**
     * [messages] rendered with [template] (by default the checkpoint's own),
     * as transformers' `apply_chat_template(messages, tokenize=False)`. The
     * result already holds the template's special tokens: encode it with
     * `addSpecialTokens = false`.
     *
     * @throws IllegalStateException if the checkpoint's template is not one [ChatTemplate] renders.
     */
    fun applyChatTemplate(
        messages: List<ChatMessage>,
        options: ChatOptions = ChatOptions(),
        template: ChatTemplate = chatTemplate ?: throw IllegalStateException(
            if (templateSource == null) {
                "this checkpoint has no chat template"
            } else {
                "this checkpoint's chat template is not one Tlaloc renders (known: ${ChatTemplate.entries.joinToString()}); " +
                    "pass template = ... to render it with one of those"
            },
        ),
    ): String = template.render(messages, options, bosToken, eosToken)

    companion object {
        private val ALWAYS: (Int) -> Boolean = { true }

        /**
         * A tokenizer from the text of `tokenizer.json`.
         *
         * Without [tokenizerConfigJson] the file's own pipeline runs, as the
         * tokenizers library runs it. With it, the tokenizer class the config
         * names decides the pipeline the way transformers 5 decides it:
         * `LlamaTokenizer`, `Qwen2Tokenizer`, `GPT2Tokenizer` and
         * `GemmaTokenizer` rebuild the normalizer, pre-tokenizer and decoder
         * from the class definition, and `TokenizersBackend` keeps the file's.
         * Any other class is refused.
         *
         * @param chatTemplate the Jinja source of the chat template
         *   (`chat_template.jinja`); by default the config's `chat_template`.
         * @throws UnsupportedTokenizerException naming the first component,
         *   flag or class that is not implemented.
         */
        fun fromJson(tokenizerJson: String, tokenizerConfigJson: String? = null, chatTemplate: String? = null): HfTokenizer {
            val root = parseJson(tokenizerJson) as? JsonObject ?: unsupported("tokenizer.json is not a JSON object")
            val spec = parsePipeline(root)
            val config = tokenizerConfigJson?.let {
                TokenizerConfig(parseJson(it) as? JsonObject ?: unsupported("tokenizer_config.json is not a JSON object"))
            }
            if (config != null) applyTransformersRules(spec, config)
            val specials = HashMap<String, String?>()
            for (key in listOf("bos_token", "eos_token", "unk_token", "pad_token")) {
                specials[key] = when {
                    config == null -> null
                    config.has(key) -> config.token(key)
                    else -> classDefaultToken(config.tokenizerClass, key)
                }
            }
            return HfTokenizer(
                model = buildModel(spec.model),
                normalizer = spec.normalizer,
                preTokenizer = spec.preTokenizer,
                postProcessor = spec.postProcessor,
                decoder = spec.decoder,
                added = spec.added,
                specialTokens = specials,
                templateSource = chatTemplate ?: config?.chatTemplate,
                splitSpecialByDefault = config?.bool("split_special_tokens") ?: false,
            )
        }

        private fun buildModel(m: ModelSpec): Bpe {
            val table = MergeTable(m.merges.size)
            for ((rank, pair) in m.merges.withIndex()) {
                val (a, b) = pair
                val ia = m.vocab[a] ?: unsupported("merge $rank: '$a' is not in the vocabulary")
                val ib = m.vocab[b] ?: unsupported("merge $rank: '$b' is not in the vocabulary")
                val merged = m.vocab[a + b] ?: unsupported("merge $rank: '${a + b}' is not in the vocabulary")
                table.put(ia, ib, rank, merged)
            }
            val unkId = m.unkToken?.let { m.vocab[it] ?: unsupported("unk_token '$it' is not in the vocabulary") } ?: -1
            return Bpe(m.vocab, table, unkId, m.fuseUnk, m.byteFallback, m.ignoreMerges)
        }
    }
}

/**
 * Decodes ids one at a time, as tokenizers' `DecodeStream` does: each [step]
 * returns the text the new id completes, or null while the text so far ends
 * in an incomplete character (U+FFFD) or adds nothing. Decoding a window of
 * recent ids keeps decoders that look at neighbouring tokens (a SentencePiece
 * leading-space strip) correct.
 */
class DecodeStream internal constructor(private val tokenizer: HfTokenizer, private val skipSpecialTokens: Boolean) {
    private var ids = IntList()
    private var prefix = ""
    private var prefixIndex = 0

    fun step(id: Int): String? {
        ids.add(id)
        val all = ids.toIntArray()
        val text = tokenizer.decode(all, skipSpecialTokens)
        // tokenizers compares UTF-8 lengths.
        if (utf8Length(text) > utf8Length(prefix) && !text.endsWith('�')) {
            check(text.startsWith(prefix)) { "decode stream: '$text' does not extend '$prefix'" }
            val fresh = text.substring(prefix.length)
            val newPrefixIndex = all.size - prefixIndex
            val kept = IntList(all.size - prefixIndex)
            for (i in prefixIndex until all.size) kept.add(all[i])
            ids = kept
            prefix = tokenizer.decode(ids.toIntArray(), skipSpecialTokens)
            prefixIndex = newPrefixIndex
            return fresh
        }
        return null
    }
}
