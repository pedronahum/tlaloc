package io.tlaloc.core.io

// §0.4.468 (Phase H2) — a minimal, STRICT JSON reader, existing for exactly
// one reason: the safetensors header is a JSON object and Tlaloc's `:core`
// has no dependencies at all (see core/build.gradle.kts: `commonMain {
// dependencies {} }`). The reader is deliberately small and deliberately
// strict.
//
// REJECTED alternatives, for the record:
//   - kotlinx.serialization-json: a first dependency on `:core`, the module
//     every other module depends on, to parse a header that is a flat map of
//     {dtype, shape, data_offsets}. The dependency is the cost; the parser is
//     ~150 lines.
//   - Hand-scanning the header with substring/indexOf: what "quick" loaders
//     do, and it is wrong the first time a tensor name contains a `{`, a `,`
//     or an escaped quote — all of which are legal in a JSON string and none
//     of which a scanner notices. A checkpoint is UNTRUSTED INPUT; the parser
//     that reads it must be a parser.
//   - A lenient parser (trailing commas, unquoted keys, NaN literals): every
//     leniency is a divergence from what the producer wrote. Refusing is the
//     honest response to a file we do not understand.
//
// SCOPE. This is not a general-purpose JSON library and is not advertised as
// one: numbers are carried as their raw text PLUS a Double, and integer reads
// go through [JsonNumber.asInt], which refuses anything that is not an exact
// integer literal — a safetensors `shape` entry of `1e3` or `4.0` is a
// malformed header, not a 1000 or a 4. Named deferral: streaming/incremental
// parsing (headers are bounded by [MAX_JSON_BYTES]'s caller, and we read them
// whole anyway).

/** A parsed JSON value. */
sealed interface JsonValue

data class JsonObject(val fields: Map<String, JsonValue>) : JsonValue {
    operator fun get(key: String): JsonValue? = fields[key]

    /** Field [key], refusing by name when absent or of the wrong shape. */
    fun obj(key: String): JsonObject = req(key) as? JsonObject
        ?: throw JsonException("field '$key' is not a JSON object")

    fun arr(key: String): JsonArray = req(key) as? JsonArray
        ?: throw JsonException("field '$key' is not a JSON array")

    fun str(key: String): String = (req(key) as? JsonString)?.value
        ?: throw JsonException("field '$key' is not a JSON string")

    private fun req(key: String): JsonValue =
        fields[key] ?: throw JsonException("missing required field '$key'")
}

data class JsonArray(val elements: List<JsonValue>) : JsonValue {
    val size: Int get() = elements.size
    operator fun get(i: Int): JsonValue = elements[i]

    /** The array as exact non-negative-or-negative Ints; refuses non-integers by name. */
    fun asIntList(what: String): List<Int> = elements.mapIndexed { i, e ->
        (e as? JsonNumber)?.asInt("$what[$i]")
            ?: throw JsonException("$what[$i] is not a number")
    }

    fun asLongList(what: String): List<Long> = elements.mapIndexed { i, e ->
        (e as? JsonNumber)?.asLong("$what[$i]")
            ?: throw JsonException("$what[$i] is not a number")
    }
}

data class JsonString(val value: String) : JsonValue

/**
 * A JSON number, carrying BOTH the raw literal text and its Double value.
 * The raw text is the authority for integer reads: `4.0` and `4` are the
 * same Double and NOT the same header, and a 64-bit offset beyond 2^53 is
 * exact in the text and rounded in the Double.
 */
data class JsonNumber(val raw: String, val value: Double) : JsonValue {
    fun asLong(what: String): Long =
        raw.toLongOrNull() ?: throw JsonException(
            "$what: expected an integer literal, got '$raw' " +
                "(a JSON number with a fraction or exponent is not an integer here)",
        )

    fun asInt(what: String): Int {
        val l = asLong(what)
        if (l < Int.MIN_VALUE || l > Int.MAX_VALUE) {
            throw JsonException("$what: integer $l does not fit in 32 bits")
        }
        return l.toInt()
    }
}

data class JsonBool(val value: Boolean) : JsonValue

data object JsonNull : JsonValue

/** Every refusal from this file and from [Safetensors] is this type. */
class JsonException(message: String) : IllegalArgumentException(message)

/** Parse [text] as a single JSON value; refuses trailing content by name. */
fun parseJson(text: String): JsonValue {
    val p = JsonParser(text)
    val v = p.parseValue()
    p.skipWs()
    if (!p.atEnd()) throw JsonException("trailing content after JSON value at offset ${p.pos}")
    return v
}

private class JsonParser(private val s: String) {
    var pos: Int = 0

    fun atEnd(): Boolean = pos >= s.length

    fun skipWs() {
        while (pos < s.length) {
            when (s[pos]) {
                ' ', '\t', '\n', '\r' -> pos++
                else -> return
            }
        }
    }

    private fun fail(msg: String): Nothing = throw JsonException("$msg at offset $pos")

    private fun expect(c: Char) {
        if (pos >= s.length || s[pos] != c) fail("expected '$c'")
        pos++
    }

    fun parseValue(): JsonValue {
        skipWs()
        if (atEnd()) fail("unexpected end of input")
        return when (val c = s[pos]) {
            '{' -> parseObject()
            '[' -> parseArray()
            '"' -> JsonString(parseString())
            't' -> { literal("true"); JsonBool(true) }
            'f' -> { literal("false"); JsonBool(false) }
            'n' -> { literal("null"); JsonNull }
            else -> if (c == '-' || c in '0'..'9') parseNumber() else fail("unexpected character '$c'")
        }
    }

    private fun literal(lit: String) {
        if (!s.startsWith(lit, pos)) fail("expected '$lit'")
        pos += lit.length
    }

    private fun parseObject(): JsonObject {
        expect('{')
        val out = LinkedHashMap<String, JsonValue>()
        skipWs()
        if (!atEnd() && s[pos] == '}') { pos++; return JsonObject(out) }
        while (true) {
            skipWs()
            val key = parseString()
            skipWs()
            expect(':')
            val v = parseValue()
            // Duplicate keys are refused, not last-wins: two entries for one
            // tensor name means two disagreeing offsets, and picking one
            // silently is how a loader reads the wrong bytes.
            if (out.containsKey(key)) fail("duplicate key '$key'")
            out[key] = v
            skipWs()
            if (atEnd()) fail("unexpected end of input inside object")
            when (s[pos]) {
                ',' -> pos++
                '}' -> { pos++; return JsonObject(out) }
                else -> fail("expected ',' or '}'")
            }
        }
    }

    private fun parseArray(): JsonArray {
        expect('[')
        val out = ArrayList<JsonValue>()
        skipWs()
        if (!atEnd() && s[pos] == ']') { pos++; return JsonArray(out) }
        while (true) {
            out.add(parseValue())
            skipWs()
            if (atEnd()) fail("unexpected end of input inside array")
            when (s[pos]) {
                ',' -> pos++
                ']' -> { pos++; return JsonArray(out) }
                else -> fail("expected ',' or ']'")
            }
        }
    }

    private fun parseString(): String {
        expect('"')
        val sb = StringBuilder()
        while (true) {
            if (atEnd()) fail("unterminated string")
            when (val c = s[pos]) {
                '"' -> { pos++; return sb.toString() }
                '\\' -> {
                    pos++
                    if (atEnd()) fail("unterminated escape")
                    when (val e = s[pos]) {
                        '"' -> { sb.append('"'); pos++ }
                        '\\' -> { sb.append('\\'); pos++ }
                        '/' -> { sb.append('/'); pos++ }
                        'b' -> { sb.append('\b'); pos++ }
                        'f' -> { sb.append('\u000C'); pos++ }
                        'n' -> { sb.append('\n'); pos++ }
                        'r' -> { sb.append('\r'); pos++ }
                        't' -> { sb.append('\t'); pos++ }
                        'u' -> {
                            pos++
                            if (pos + 4 > s.length) fail("truncated \\u escape")
                            val hex = s.substring(pos, pos + 4)
                            val code = hex.toIntOrNull(16) ?: fail("bad \\u escape '$hex'")
                            sb.append(code.toChar())
                            pos += 4
                        }
                        else -> fail("bad escape '\\$e'")
                    }
                }
                else -> {
                    // Raw control characters are illegal in JSON strings.
                    if (c < ' ') fail("raw control character U+%04X in string".format(c.code))
                    sb.append(c); pos++
                }
            }
        }
    }

    private fun parseNumber(): JsonNumber {
        val start = pos
        if (!atEnd() && s[pos] == '-') pos++
        if (atEnd() || s[pos] !in '0'..'9') fail("expected digit")
        if (s[pos] == '0') pos++ else while (pos < s.length && s[pos] in '0'..'9') pos++
        if (pos < s.length && s[pos] == '.') {
            pos++
            if (pos >= s.length || s[pos] !in '0'..'9') fail("expected digit after '.'")
            while (pos < s.length && s[pos] in '0'..'9') pos++
        }
        if (pos < s.length && (s[pos] == 'e' || s[pos] == 'E')) {
            pos++
            if (pos < s.length && (s[pos] == '+' || s[pos] == '-')) pos++
            if (pos >= s.length || s[pos] !in '0'..'9') fail("expected digit in exponent")
            while (pos < s.length && s[pos] in '0'..'9') pos++
        }
        val raw = s.substring(start, pos)
        return JsonNumber(raw, raw.toDouble())
    }
}

/** KMP-safe minimal `format` stand-in for the one control-char message above. */
private fun String.format(vararg args: Any): String {
    // Only the "%04X" case is used; keep it explicit rather than pulling a
    // platform formatter into commonMain.
    var out = this
    for (a in args) {
        val hex = (a as Int).toString(16).uppercase().padStart(4, '0')
        out = out.replaceFirst("%04X", hex)
    }
    return out
}

// ---------------------------------------------------------------------------
// §0.4.502 — the one WRITING primitive this file owes the safetensors writer.
//
// There is no JSON *serializer* here and there is not going to be one: the
// only JSON this repository emits from `:core` is a safetensors header, whose
// shape is fixed (an object of objects of a string, an int array and a
// two-element int array). What a header-builder cannot do by hand is quote a
// string correctly, because a tensor NAME is caller data — `"` and `\` are
// legal in it, and so are control characters — and a header that concatenates
// an unescaped name produces a file whose own reader cannot parse it. That is
// the one function below, and it is the exact inverse of the string case the
// parser above already implements.
// ---------------------------------------------------------------------------

/**
 * [s] as a quoted JSON string literal, escaped per RFC 8259 section 7. The escapes
 * are the two mandatory ones (`"` and `\`), the five short forms the parser
 * above accepts (`\b \f \n \r \t`), and `\u00XX` for every remaining control
 * character below 0x20. Everything else — including non-ASCII — is emitted
 * verbatim, because the header is written as UTF-8 and a reader that cannot
 * read UTF-8 cannot read the names HuggingFace already ships.
 *
 * DELIBERATELY NOT escaped: `/` (legal either way, and the safetensors files
 * produced by the reference implementation do not escape it) and the
 * surrogate-pair range (a lone surrogate cannot be produced by a valid Kotlin
 * String read from a valid source, and mangling one silently would be worse
 * than emitting it).
 */
fun jsonQuote(s: String): String {
    val sb = StringBuilder(s.length + 2)
    sb.append('"')
    for (c in s) {
        when (c) {
            '"' -> sb.append("\\\"")
            '\\' -> sb.append("\\\\")
            '\b' -> sb.append("\\b")
            '\u000C' -> sb.append("\\f")
            '\n' -> sb.append("\\n")
            '\r' -> sb.append("\\r")
            '\t' -> sb.append("\\t")
            else ->
                if (c.code < 0x20) {
                    sb.append("\\u").append(c.code.toString(16).padStart(4, '0'))
                } else {
                    sb.append(c)
                }
        }
    }
    sb.append('"')
    return sb.toString()
}
