package io.tlaloc.endpoint

import io.tlaloc.core.io.jsonQuote

/** JSON text already written, placed as is. */
internal class RawJson(val text: String)

/** A JSON document from Kotlin values: maps, lists, strings, numbers, booleans, null and [RawJson]. */
internal fun json(v: Any?): String = StringBuilder().also { writeJson(it, v) }.toString()

internal fun jsonObject(vararg fields: Pair<String, Any?>): LinkedHashMap<String, Any?> = linkedMapOf(*fields)

private fun writeJson(sb: StringBuilder, v: Any?) {
    when (v) {
        null -> sb.append("null")
        is RawJson -> sb.append(v.text)
        is String -> sb.append(jsonQuote(v))
        is Boolean -> sb.append(v)
        is Int, is Long -> sb.append(v)
        is Double -> sb.append(if (v.isFinite()) v.toString() else "null")
        is Float -> sb.append(if (v.isFinite()) v.toString() else "null")
        is Map<*, *> -> {
            sb.append('{')
            var first = true
            for ((k, x) in v) {
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
        is IntArray -> writeJson(sb, v.toList())
        else -> throw IllegalArgumentException("no JSON form for ${v::class}")
    }
}
