package io.tlaloc.tokenizer

// The normalizers, pre-tokenizers, post-processors and decoders of
// tokenizer.json that the supported checkpoints use, each with the semantics
// of tokenizers 0.23. Anything else is refused by name when the file is read.

internal sealed interface Normalizer {
    fun normalize(text: String): String
}

internal class UnicodeNormalizer(private val form: UnicodeForm) : Normalizer {
    override fun normalize(text: String): String = normalizeUnicode(text, form)
}

internal class ReplaceNormalizer(private val pattern: String, private val content: String) : Normalizer {
    override fun normalize(text: String): String = text.replace(pattern, content)
}

internal class PrependNormalizer(private val prefix: String) : Normalizer {
    override fun normalize(text: String): String = if (text.isEmpty()) text else prefix + text
}

internal class SequenceNormalizer(private val steps: List<Normalizer>) : Normalizer {
    override fun normalize(text: String): String = steps.fold(text) { t, n -> n.normalize(t) }
}

/**
 * A piece of text on its way to the model. [atOrigin] is true when the piece
 * starts at offset 0 of the input, which Metaspace's `first` scheme reads.
 */
internal class Piece(val text: String, val atOrigin: Boolean)

internal sealed interface PreTokenizer {
    fun split(pieces: List<Piece>): List<Piece>
}

internal enum class SplitBehavior { ISOLATED, REMOVED, MERGED_WITH_PREVIOUS, MERGED_WITH_NEXT }

/** Splits each piece at the matches of [regex], keeping or merging the matched text per [behavior]. */
internal class SplitPreTokenizer(private val regex: SplitRegex, private val behavior: SplitBehavior) : PreTokenizer {
    override fun split(pieces: List<Piece>): List<Piece> {
        val out = ArrayList<Piece>()
        for (piece in pieces) splitOne(piece, out)
        return out
    }

    private fun splitOne(piece: Piece, out: MutableList<Piece>) {
        val text = piece.text
        val m = regex.matches(text)
        // Alternating [start, end, isMatch] ranges covering the text.
        val ranges = ArrayList<IntArray>()
        var last = 0
        var k = 0
        while (k < m.size) {
            if (m[k] > last) ranges.add(intArrayOf(last, m[k], 0))
            ranges.add(intArrayOf(m[k], m[k + 1], 1))
            last = m[k + 1]
            k += 2
        }
        if (last < text.length) ranges.add(intArrayOf(last, text.length, 0))
        val spans = ArrayList<IntArray>()
        when (behavior) {
            SplitBehavior.ISOLATED -> ranges.forEach { spans.add(intArrayOf(it[0], it[1])) }
            SplitBehavior.REMOVED -> ranges.filter { it[2] == 0 }.forEach { spans.add(intArrayOf(it[0], it[1])) }
            SplitBehavior.MERGED_WITH_PREVIOUS -> {
                var prevWasMatch = false
                for (r in ranges) {
                    if (r[2] == 1 && !prevWasMatch && spans.isNotEmpty()) {
                        spans.last()[1] = r[1]
                    } else {
                        spans.add(intArrayOf(r[0], r[1]))
                    }
                    prevWasMatch = r[2] == 1
                }
            }
            SplitBehavior.MERGED_WITH_NEXT -> {
                var prevWasMatch = false
                for (r in ranges) {
                    if (r[2] == 0 && prevWasMatch && spans.isNotEmpty()) {
                        spans.last()[1] = r[1]
                    } else {
                        spans.add(intArrayOf(r[0], r[1]))
                    }
                    prevWasMatch = r[2] == 1
                }
            }
        }
        for (s in spans) {
            if (s[1] > s[0]) out.add(Piece(text.substring(s[0], s[1]), piece.atOrigin && s[0] == 0))
        }
    }
}

/** GPT-2's pre-tokenizer: optional prefix space, optional GPT-2 split, then bytes to table characters. */
internal class ByteLevelPreTokenizer(private val addPrefixSpace: Boolean, useRegex: Boolean) : PreTokenizer {
    private val gpt2Split = if (useRegex) SplitPreTokenizer(compileSplitRegex(GPT2_PATTERN), SplitBehavior.ISOLATED) else null

    override fun split(pieces: List<Piece>): List<Piece> {
        val prefixed = if (addPrefixSpace) {
            pieces.map { if (it.text.startsWith(" ")) it else Piece(" " + it.text, it.atOrigin) }
        } else {
            pieces
        }
        val split = gpt2Split?.split(prefixed) ?: prefixed
        return split.map { Piece(ByteLevelTable.encode(it.text), it.atOrigin) }
    }

    companion object {
        const val GPT2_PATTERN = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    }
}

internal enum class PrependScheme { ALWAYS, FIRST, NEVER }

/** SentencePiece's space handling: spaces become [replacement], and a leading one is added per [scheme]. */
internal class MetaspacePreTokenizer(
    private val replacement: Char,
    private val scheme: PrependScheme,
    split: Boolean,
) : PreTokenizer {
    private val rep = replacement.toString()
    private val splitter = if (split) {
        SplitPreTokenizer(literalRegex(rep), SplitBehavior.MERGED_WITH_NEXT)
    } else {
        null
    }

    override fun split(pieces: List<Piece>): List<Piece> {
        val replaced = pieces.map { p ->
            val t = p.text.replace(" ", rep)
            val prepend = when (scheme) {
                PrependScheme.ALWAYS -> !t.startsWith(rep)
                PrependScheme.FIRST -> !t.startsWith(rep) && p.atOrigin
                PrependScheme.NEVER -> false
            }
            Piece(if (prepend) rep + t else t, p.atOrigin)
        }
        return splitter?.split(replaced) ?: replaced
    }
}

internal class SequencePreTokenizer(private val steps: List<PreTokenizer>) : PreTokenizer {
    override fun split(pieces: List<Piece>): List<Piece> = steps.fold(pieces) { p, s -> s.split(p) }
}

/** A `String` pattern: literal matches, found left to right without overlap. */
internal fun literalRegex(literal: String): SplitRegex {
    require(literal.isNotEmpty()) { "an empty Split pattern matches nothing useful" }
    return SplitRegex { text ->
        val out = IntList()
        var from = 0
        while (true) {
            val at = text.indexOf(literal, from)
            if (at < 0) break
            out.add(at)
            out.add(at + literal.length)
            from = at + literal.length
        }
        out.toIntArray()
    }
}

/** The single-sequence template of a TemplateProcessing post-processor. */
internal class PostProcessor(private val prefix: IntArray, private val suffix: IntArray) {
    fun apply(ids: IntArray): IntArray =
        if (prefix.isEmpty() && suffix.isEmpty()) ids else prefix + ids + suffix

    companion object {
        val IDENTITY = PostProcessor(IntArray(0), IntArray(0))
    }
}

internal sealed interface Decoder {
    fun decodeChain(tokens: List<String>): List<String>
}

internal object ByteLevelDecoder : Decoder {
    override fun decodeChain(tokens: List<String>): List<String> {
        val bytes = ByteBuilder()
        for (t in tokens) {
            // A token with any character outside the table contributes its own
            // UTF-8 bytes whole, as tokenizers does for added tokens.
            var mapped = true
            for (c in t) if (ByteLevelTable.byteOf(c) < 0) { mapped = false; break }
            if (mapped) {
                for (c in t) bytes.add(ByteLevelTable.byteOf(c))
            } else {
                for (b in t.encodeToByteArray()) bytes.add(b.toInt() and 0xFF)
            }
        }
        return listOf(decodeUtf8Lossy(bytes.array, bytes.size))
    }
}

internal class ReplaceDecoder(private val pattern: String, private val content: String) : Decoder {
    override fun decodeChain(tokens: List<String>): List<String> = tokens.map { it.replace(pattern, content) }
}

/** Runs of `<0xNN>` tokens become the text of their bytes, or one U+FFFD per byte when not valid UTF-8. */
internal object ByteFallbackDecoder : Decoder {
    override fun decodeChain(tokens: List<String>): List<String> {
        val out = ArrayList<String>(tokens.size)
        val pending = ByteBuilder()
        fun flush() {
            if (pending.size == 0) return
            val s = decodeUtf8Strict(pending.array, pending.size)
            if (s != null) out.add(s) else repeat(pending.size) { out.add("�") }
            pending.clear()
        }
        for (t in tokens) {
            val b = byteToken(t)
            if (b >= 0) {
                pending.add(b)
            } else {
                flush()
                out.add(t)
            }
        }
        flush()
        return out
    }

    private fun byteToken(t: String): Int {
        if (t.length != 6 || !t.startsWith("<0x") || t[5] != '>') return -1
        val hi = hexDigit(t[3])
        val lo = hexDigit(t[4])
        return if (hi < 0 || lo < 0) -1 else hi * 16 + lo
    }

    private fun hexDigit(c: Char): Int = when (c) {
        in '0'..'9' -> c - '0'
        in 'A'..'F' -> c - 'A' + 10
        in 'a'..'f' -> c - 'a' + 10
        else -> -1
    }
}

internal object FuseDecoder : Decoder {
    override fun decodeChain(tokens: List<String>): List<String> = listOf(tokens.joinToString(""))
}

/** Removes up to [start] leading and [stop] trailing [content] characters from each token. */
internal class StripDecoder(private val content: Char, private val start: Int, private val stop: Int) : Decoder {
    override fun decodeChain(tokens: List<String>): List<String> = tokens.map { t ->
        // tokenizers counts chars (code points); [content] is a BMP char, so a
        // surrogate never equals it and UTF-16 indexing gives the same cut.
        var s = 0
        while (s < start && s < t.length && t[s] == content) s++
        var e = t.length
        var k = 0
        while (k < stop && e > s && t[e - 1] == content) { e--; k++ }
        t.substring(s, e)
    }
}

/** SentencePiece's decoder: [replacement] becomes a space, except in the first token where it is dropped. */
internal class MetaspaceDecoder(private val replacement: Char, private val scheme: PrependScheme) : Decoder {
    override fun decodeChain(tokens: List<String>): List<String> = tokens.mapIndexed { i, t ->
        val sb = StringBuilder(t.length)
        for (c in t) {
            if (c == replacement) {
                if (!(i == 0 && scheme != PrependScheme.NEVER)) sb.append(' ')
            } else {
                sb.append(c)
            }
        }
        sb.toString()
    }
}

internal class SequenceDecoder(private val steps: List<Decoder>) : Decoder {
    override fun decodeChain(tokens: List<String>): List<String> = steps.fold(tokens) { t, d -> d.decodeChain(t) }
}

internal class ByteBuilder {
    var array = ByteArray(64)
        private set
    var size = 0
        private set

    fun add(b: Int) {
        if (size == array.size) array = array.copyOf(size * 2)
        array[size++] = b.toByte()
    }

    fun clear() { size = 0 }
}
