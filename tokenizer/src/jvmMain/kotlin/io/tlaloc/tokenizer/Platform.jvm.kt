package io.tlaloc.tokenizer

import java.text.Normalizer
import java.time.LocalDate
import java.util.concurrent.ConcurrentHashMap
import java.util.regex.Pattern

internal actual fun normalizeUnicode(text: String, form: UnicodeForm): String = Normalizer.normalize(
    text,
    when (form) {
        UnicodeForm.NFC -> Normalizer.Form.NFC
        UnicodeForm.NFD -> Normalizer.Form.NFD
        UnicodeForm.NFKC -> Normalizer.Form.NFKC
        UnicodeForm.NFKD -> Normalizer.Form.NFKD
    },
)

// UNICODE_CHARACTER_CLASS makes \s, \w and \d follow the Unicode properties, as
// Oniguruma's do. Without it \s misses 19 whitespace code points (NBSP, U+2000
// to U+200A, U+3000, ...). With it, the Qwen2, Llama-4-style and GPT-2 split
// patterns give the same pieces as tokenizers 0.23 on all 1,112,064 code points.
// The flag also implies UNICODE_CASE, which `(?i:'s|...)` needs for the same agreement.
internal actual fun compileSplitRegex(pattern: String): SplitRegex {
    val p = Pattern.compile(pattern, Pattern.UNICODE_CHARACTER_CLASS)
    return SplitRegex { text ->
        val m = p.matcher(text)
        var out = IntArray(16)
        var n = 0
        while (m.find()) {
            if (m.end() == m.start()) continue
            if (n + 2 > out.size) out = out.copyOf(out.size * 2)
            out[n++] = m.start()
            out[n++] = m.end()
        }
        out.copyOf(n)
    }
}

// Cleared when full rather than evicting one entry at a time: a lookup never
// takes a lock, and a working set that fits is rebuilt within one batch.
internal actual fun newWordCache(capacity: Int): WordCache = object : WordCache {
    private val map = ConcurrentHashMap<String, IntArray>()

    override fun get(word: String): IntArray? = map[word]

    override fun put(word: String, ids: IntArray) {
        if (map.size >= capacity) map.clear()
        map[word] = ids
    }
}

internal actual fun todayIsoDate(): String = LocalDate.now().toString()
