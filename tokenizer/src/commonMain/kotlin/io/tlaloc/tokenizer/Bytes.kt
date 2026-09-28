package io.tlaloc.tokenizer

/**
 * GPT-2's byte-to-character table: every byte gets a printable character, so a
 * byte-level vocabulary can be stored as text. Printable Latin-1 bytes map to
 * themselves; the other 68 map to U+0100 onwards in byte order.
 */
internal object ByteLevelTable {
    val byteToChar: CharArray = CharArray(256)

    /** Indexed by char code; -1 for a character that is not in the table. */
    val charToByte: IntArray = IntArray(0x144) { -1 }

    init {
        var extra = 0
        for (b in 0 until 256) {
            val printable = b in '!'.code..'~'.code || b in '¡'.code..'¬'.code || b in '®'.code..'ÿ'.code
            val c = if (printable) b else 256 + extra++
            byteToChar[b] = c.toChar()
            charToByte[c] = b
        }
    }

    fun byteOf(c: Char): Int = if (c.code < charToByte.size) charToByte[c.code] else -1

    /** [text]'s UTF-8 bytes, each written as its table character. */
    fun encode(text: String): String {
        val bytes = text.encodeToByteArray()
        val out = CharArray(bytes.size)
        for (i in bytes.indices) out[i] = byteToChar[bytes[i].toInt() and 0xFF]
        return out.concatToString()
    }
}

/** Refuses a string that has an unpaired surrogate: it has no UTF-8 form, and tokenizers cannot receive it. */
internal fun requireWellFormed(text: String) {
    var i = 0
    while (i < text.length) {
        val c = text[i]
        if (c.isHighSurrogate()) {
            require(i + 1 < text.length && text[i + 1].isLowSurrogate()) {
                "unpaired surrogate U+${c.code.toString(16).uppercase()} at index $i: the text has no UTF-8 form"
            }
            i += 2
            continue
        }
        require(!c.isLowSurrogate()) {
            "unpaired surrogate U+${c.code.toString(16).uppercase()} at index $i: the text has no UTF-8 form"
        }
        i++
    }
}

/** Number of UTF-8 bytes in a well-formed [text]. */
internal fun utf8Length(text: String): Int {
    var n = 0
    var i = 0
    while (i < text.length) {
        val c = text[i].code
        n += when {
            c < 0x80 -> 1
            c < 0x800 -> 2
            text[i].isHighSurrogate() && i + 1 < text.length && text[i + 1].isLowSurrogate() -> { i++; 4 }
            else -> 3
        }
        i++
    }
    return n
}

/**
 * [bytes] as UTF-8, each maximal invalid subsequence replaced by one U+FFFD:
 * the rule Rust's `String::from_utf8_lossy` follows, which the byte-level
 * decoder uses.
 */
internal fun decodeUtf8Lossy(bytes: ByteArray, length: Int = bytes.size): String {
    val sb = StringBuilder(length)
    var i = 0
    while (i < length) {
        val b0 = bytes[i].toInt() and 0xFF
        if (b0 < 0x80) {
            sb.append(b0.toChar()); i++; continue
        }
        val need: Int
        var lo = 0x80
        var hi = 0xBF
        when (b0) {
            in 0xC2..0xDF -> need = 1
            0xE0 -> { need = 2; lo = 0xA0 }
            in 0xE1..0xEC, 0xEE, 0xEF -> need = 2
            0xED -> { need = 2; hi = 0x9F }
            0xF0 -> { need = 3; lo = 0x90 }
            in 0xF1..0xF3 -> need = 3
            0xF4 -> { need = 3; hi = 0x8F }
            else -> { sb.append('�'); i++; continue }
        }
        var cp = b0 and (0x3F shr need)
        var j = i + 1
        var ok = true
        for (k in 0 until need) {
            if (j >= length) { ok = false; break }
            val b = bytes[j].toInt() and 0xFF
            val min = if (k == 0) lo else 0x80
            val max = if (k == 0) hi else 0xBF
            if (b < min || b > max) { ok = false; break }
            cp = (cp shl 6) or (b and 0x3F)
            j++
        }
        if (!ok) {
            sb.append('�'); i = j; continue
        }
        appendCodePoint(sb, cp)
        i = j
    }
    return sb.toString()
}

/** [bytes] as UTF-8, or null when they are not valid UTF-8. */
internal fun decodeUtf8Strict(bytes: ByteArray, length: Int): String? {
    val s = decodeUtf8Lossy(bytes, length)
    if (s.indexOf('�') < 0) return s
    // A U+FFFD may be real text (EF BF BD) rather than a replacement.
    val back = s.encodeToByteArray()
    if (back.size != length) return null
    for (k in 0 until length) if (back[k] != bytes[k]) return null
    return s
}

internal fun appendCodePoint(sb: StringBuilder, cp: Int) {
    if (cp < 0x10000) {
        sb.append(cp.toChar())
    } else {
        val v = cp - 0x10000
        sb.append((0xD800 + (v shr 10)).toChar())
        sb.append((0xDC00 + (v and 0x3FF)).toChar())
    }
}
