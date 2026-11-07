package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson

/**
 * Splits TwIL-8B's generated text into reasoning, answer and tool calls. Its chat template ends
 * the prompt inside the reasoning block (`<think>\n`) unless thinking is off ([reasoning] false,
 * the prompt then ends `<think></think>`), so the text starts as reasoning and `</think>` ends it:
 *
 * ```
 *   reasoning\n</think>\n\nanswer\n<tool_call>\n<function=name>\n<parameter=p>\nvalue\n</parameter>\n</function>\n</tool_call>
 * ```
 *
 * A parameter's value is text; it becomes the JSON type the tool's schema in [tools] gives the
 * parameter (`integer`, `number`, `boolean`, `object`, `array`), and stays a string when the
 * schema says `string`, names no type, or the text does not parse as that type; `null` is null.
 * Text after the first tool call that is not another call is dropped. A block that does not
 * parse as a call, or that never closes, is answer text.
 */
class TwilOutputParser(reasoning: Boolean = true, tools: List<String> = emptyList()) : ChatOutputParser {
    private enum class State { THINK, AFTER_THINK, CONTENT, CALL, AFTER_CALL }

    private var state = if (reasoning) State.THINK else State.CONTENT
    private val buffer = StringBuilder()
    private var called = false
    private val types: Map<String, Map<String, String?>> = tools.mapNotNull { schemaOf(it) }.toMap()

    override fun feed(text: String): List<ChatOutputEvent> {
        buffer.append(text)
        val out = ArrayList<ChatOutputEvent>()
        while (step(out)) { /* until it needs more text */ }
        return out
    }

    override fun finish(): List<ChatOutputEvent> {
        val rest = buffer.toString()
        buffer.clear()
        val out = ArrayList<ChatOutputEvent>()
        when (state) {
            State.THINK -> if (rest.isNotEmpty()) out += ChatOutputEvent.Reasoning(rest)
            State.AFTER_THINK, State.CONTENT -> if (rest.isNotEmpty()) out += ChatOutputEvent.Content(rest)
            State.CALL -> if (!called) out += ChatOutputEvent.Content(TOOL_START + rest)
            State.AFTER_CALL -> {}
        }
        return out
    }

    private fun step(out: MutableList<ChatOutputEvent>): Boolean {
        val s = buffer.toString()
        if (s.isEmpty()) return false
        when (state) {
            State.THINK -> {
                val end = s.indexOf(THINK_END)
                if (end >= 0) {
                    val text = s.substring(0, end).removeSuffix("\n")
                    if (text.isNotEmpty()) out += ChatOutputEvent.Reasoning(text)
                    buffer.delete(0, end + THINK_END.length)
                    state = State.AFTER_THINK
                    return true
                }
                val hold = maxOf(partialSuffix(s, "\n" + THINK_END), partialSuffix(s, THINK_END))
                if (s.length > hold) out += ChatOutputEvent.Reasoning(s.substring(0, s.length - hold))
                buffer.delete(0, s.length - hold)
                return false
            }
            State.AFTER_THINK -> {
                // The newlines between the reasoning and the answer belong to neither.
                val n = s.indexOfFirst { it != '\n' }
                if (n < 0) {
                    buffer.clear()
                    return false
                }
                buffer.delete(0, n)
                state = State.CONTENT
                return true
            }
            State.CONTENT -> {
                val start = s.indexOf(TOOL_START)
                if (start >= 0) {
                    val text = s.substring(0, start).removeSuffix("\n")
                    if (text.isNotEmpty()) out += ChatOutputEvent.Content(text)
                    buffer.delete(0, start + TOOL_START.length)
                    state = State.CALL
                    return true
                }
                val hold = maxOf(partialSuffix(s, "\n" + TOOL_START), partialSuffix(s, TOOL_START))
                if (s.length > hold) out += ChatOutputEvent.Content(s.substring(0, s.length - hold))
                buffer.delete(0, s.length - hold)
                return false
            }
            State.CALL -> {
                val end = s.indexOf(TOOL_END)
                if (end < 0) return false
                val body = s.substring(0, end)
                val call = parseCall(body)
                if (call != null) {
                    out += ChatOutputEvent.Call(call)
                    called = true
                } else if (!called) {
                    out += ChatOutputEvent.Content(TOOL_START + body + TOOL_END)
                }
                buffer.delete(0, end + TOOL_END.length)
                state = State.AFTER_CALL
                return true
            }
            State.AFTER_CALL -> {
                val start = s.indexOf(TOOL_START)
                if (start >= 0) {
                    buffer.delete(0, start + TOOL_START.length)
                    state = State.CALL
                    return true
                }
                val hold = partialSuffix(s, TOOL_START)
                buffer.delete(0, s.length - hold)
                return false
            }
        }
    }

    private fun parseCall(body: String): ChatToolCall? {
        val f = FUNCTION.find(body) ?: return null
        val name = f.groupValues[1].trim()
        if (name.isEmpty()) return null
        val schema = types[name]
        val args = PARAMETER.findAll(f.groupValues[2]).joinToString(", ", "{", "}") { m ->
            val key = m.groupValues[1].trim()
            jsonQuote(key) + ": " + valueJson(m.groupValues[2], schema?.get(key))
        }
        return ChatToolCall(name, args)
    }

    /** A parameter's text as JSON, as its schema [type] reads it. */
    private fun valueJson(text: String, type: String?): String {
        if (text.lowercase() == "null") return "null"
        val parsed = when (type) {
            "integer", "int", "number", "float", "boolean", "bool", "object", "array" -> try {
                parseJson(text.trim())
            } catch (_: JsonException) {
                null
            }
            else -> null
        }
        val fits = when (type) {
            "integer", "int" -> parsed is JsonNumber && parsed.raw.none { it == '.' || it == 'e' || it == 'E' }
            "number", "float" -> parsed is JsonNumber
            "boolean", "bool" -> parsed is JsonBool
            "object" -> parsed is JsonObject
            "array" -> parsed is JsonArray
            else -> false
        }
        return if (fits) jsonText(parsed!!) else jsonQuote(text)
    }

    private companion object {
        const val THINK_END = "</think>"
        const val TOOL_START = "<tool_call>"
        const val TOOL_END = "</tool_call>"
        val FUNCTION = Regex("<function=([^>\\n]*)>\\n?(.*?)</function>", RegexOption.DOT_MATCHES_ALL)
        val PARAMETER = Regex("<parameter=([^>\\n]*)>\\n?(.*?)\\n?</parameter>", RegexOption.DOT_MATCHES_ALL)

        /** A tool's name and each parameter's schema type, from its definition. */
        fun schemaOf(tool: String): Pair<String, Map<String, String?>>? {
            val o = try {
                parseJson(tool) as? JsonObject
            } catch (_: JsonException) {
                null
            } ?: return null
            val f = o["function"] as? JsonObject ?: o
            val name = (f["name"] as? JsonString)?.value ?: return null
            val props = ((f["parameters"] as? JsonObject)?.get("properties") as? JsonObject)?.fields.orEmpty()
            return name to props.mapValues { (_, p) ->
                when (val t = (p as? JsonObject)?.get("type")) {
                    is JsonString -> t.value
                    is JsonArray -> t.elements.mapNotNull { (it as? JsonString)?.value }.firstOrNull { it != "null" }
                    else -> null
                }
            }
        }

        /** [v] with `", "` and `": "` separators and numbers as written. */
        fun jsonText(v: JsonValue): String = when (v) {
            is JsonObject -> v.fields.entries.joinToString(", ", "{", "}") { jsonQuote(it.key) + ": " + jsonText(it.value) }
            is JsonArray -> v.elements.joinToString(", ", "[", "]") { jsonText(it) }
            is JsonString -> jsonQuote(v.value)
            is JsonNumber -> v.raw
            is JsonBool -> v.value.toString()
            JsonNull -> "null"
        }
    }
}
