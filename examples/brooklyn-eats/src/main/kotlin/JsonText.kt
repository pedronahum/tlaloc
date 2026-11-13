package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.jsonQuote

// JSON text from Kotlin values (maps keep their order; null map values are left out so tool
// results stay compact), and back from Tlaloc's parsed JsonValue.

/** Already-written JSON text, placed as is. */
class RawJson(val text: String)

fun toJson(v: Any?): String = StringBuilder().also { writeJson(it, v) }.toString()

private fun writeJson(sb: StringBuilder, v: Any?) {
    when (v) {
        null -> sb.append("null")
        is RawJson -> sb.append(v.text)
        is JsonValue -> writeJson(sb, v.toKotlin())
        is String -> sb.append(jsonQuote(v))
        is Boolean -> sb.append(v)
        is Int, is Long -> sb.append(v)
        is Double -> sb.append(if (!v.isFinite()) "null" else if (v == Math.rint(v) && Math.abs(v) < 1e15) v.toLong().toString() else v.toString())
        is Float -> writeJson(sb, v.toDouble())
        is Map<*, *> -> {
            sb.append('{')
            var first = true
            for ((k, x) in v) {
                if (x == null) continue
                if (!first) sb.append(',')
                first = false
                sb.append(jsonQuote(k.toString())).append(':')
                writeJson(sb, x)
            }
            sb.append('}')
        }
        is Iterable<*> -> {
            sb.append('[')
            var first = true
            for (x in v) {
                if (!first) sb.append(',')
                first = false
                writeJson(sb, x)
            }
            sb.append(']')
        }
        else -> throw IllegalArgumentException("no JSON form for ${v::class}")
    }
}

/** Maps, lists, strings, Long/Double, booleans and null. */
fun JsonValue.toKotlin(): Any? = when (this) {
    is JsonObject -> fields.mapValues { it.value.toKotlin() }
    is JsonArray -> elements.map { it.toKotlin() }
    is JsonString -> value
    is JsonNumber -> raw.toLongOrNull() ?: value
    is JsonBool -> value
    JsonNull -> null
}

/** The same value with object keys sorted: a key for recordings. */
fun canonicalJson(v: JsonValue): String = when (v) {
    is JsonObject -> v.fields.entries.sortedBy { it.key }
        .joinToString(",", "{", "}") { jsonQuote(it.key) + ":" + canonicalJson(it.value) }
    is JsonArray -> v.elements.joinToString(",", "[", "]") { canonicalJson(it) }
    is JsonString -> jsonQuote(v.value)
    is JsonNumber -> v.raw
    is JsonBool -> v.value.toString()
    JsonNull -> "null"
}

/** Lenient accessors for tool arguments (already checked against the schema). */
fun JsonObject.optString(k: String): String? = (this[k] as? JsonString)?.value?.takeIf { it.isNotBlank() }
fun JsonObject.optNumber(k: String): Double? = (this[k] as? JsonNumber)?.value
fun JsonObject.optInt(k: String): Int? = (this[k] as? JsonNumber)?.value?.toInt()
fun JsonObject.optObject(k: String): JsonObject? = this[k] as? JsonObject
