package io.tlaloc.endpoint

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.jsonQuote
import java.math.BigDecimal

/**
 * [v] as Python's `json.dumps(v, ensure_ascii=False)` writes what `json.loads` read: keys in
 * their order, `", "` and `": "` separators, integers as written and other numbers as Python's
 * `repr` of the float. Chat templates write tool definitions and call arguments with `tojson`,
 * which transformers implements with `json.dumps`, so a prompt rendered here has the bytes
 * transformers renders.
 */
fun pythonJson(v: JsonValue): String = StringBuilder().also { write(it, v) }.toString()

private fun write(sb: StringBuilder, v: JsonValue) {
    when (v) {
        is JsonObject -> {
            sb.append('{')
            var first = true
            for ((k, x) in v.fields) {
                if (!first) sb.append(", ")
                first = false
                sb.append(jsonQuote(k)).append(": ")
                write(sb, x)
            }
            sb.append('}')
        }
        is JsonArray -> {
            sb.append('[')
            v.elements.forEachIndexed { i, x ->
                if (i > 0) sb.append(", ")
                write(sb, x)
            }
            sb.append(']')
        }
        is JsonString -> sb.append(jsonQuote(v.value))
        is JsonNumber -> sb.append(pythonNumber(v.raw))
        is JsonBool -> sb.append(if (v.value) "true" else "false")
        JsonNull -> sb.append("null")
    }
}

/** A JSON number as Python prints what `json.loads` made of it: an int as is, a float by `repr`. */
internal fun pythonNumber(raw: String): String {
    if (raw.none { it == '.' || it == 'e' || it == 'E' }) {
        return BigDecimal(raw).toBigIntegerExact().toString() // "-0" is the int 0
    }
    return pythonFloatRepr(raw.toDouble())
}

/** Python's `repr(float)`: the shortest digits that round-trip, positional for exponents -5 to 15. */
internal fun pythonFloatRepr(d: Double): String {
    if (d.isNaN()) return "NaN"
    if (d.isInfinite()) return if (d > 0) "Infinity" else "-Infinity"
    if (d == 0.0) return if (1.0 / d < 0) "-0.0" else "0.0"
    // Java's Double.toString gives the shortest round-tripping digits (JDK 19 and later).
    val bd = BigDecimal(d.toString()).stripTrailingZeros()
    val digits = bd.unscaledValue().abs().toString()
    val exp = digits.length - 1 - bd.scale() // decimal exponent of the first digit
    val sign = if (d < 0) "-" else ""
    return if (exp < -4 || exp >= 16) {
        val mantissa = if (digits.length == 1) digits else digits[0] + "." + digits.substring(1)
        sign + mantissa + "e" + (if (exp < 0) "-" else "+") + kotlin.math.abs(exp).toString().padStart(2, '0')
    } else {
        val plain = bd.abs().toPlainString()
        sign + if ('.' in plain) plain else "$plain.0"
    }
}
