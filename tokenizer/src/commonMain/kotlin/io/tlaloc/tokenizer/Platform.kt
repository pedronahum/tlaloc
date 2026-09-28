package io.tlaloc.tokenizer

// The four things commonMain cannot do on its own: Unicode normalization, a
// regex engine whose character classes match the Oniguruma patterns Hugging
// Face tokenizers ship, a word cache safe to share between threads, and
// today's date for chat templates that print one.

internal enum class UnicodeForm { NFC, NFD, NFKC, NFKD }

internal expect fun normalizeUnicode(text: String, form: UnicodeForm): String

/** A compiled `Split` pattern: the `[start, end)` UTF-16 ranges of its non-empty matches, in order. */
internal fun interface SplitRegex {
    fun matches(text: String): IntArray
}

internal expect fun compileSplitRegex(pattern: String): SplitRegex

/** A bounded map from a pre-tokenized word to its ids. Implementations are safe for concurrent use. */
internal interface WordCache {
    fun get(word: String): IntArray?
    fun put(word: String, ids: IntArray)
}

internal expect fun newWordCache(capacity: Int): WordCache

/** Today's date as `YYYY-MM-DD` in the system time zone. */
internal expect fun todayIsoDate(): String
