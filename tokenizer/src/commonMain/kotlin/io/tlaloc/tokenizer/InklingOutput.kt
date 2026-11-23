package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson

/**
 * What Inkling 2 generates (decoded with its special tokens): parts
 * `<|message_model|>[name]<|content_KIND|>body<|end_message|>`, ending with
 * `<|content_model_end_sampling|>`. A `content_thinking` body is reasoning, a `content_text` body
 * the answer, a `content_invoke_tool_json` body a tool call `{"name": ..., "args": {...}}` (the
 * form `ChatTemplate.INKLING` renders). A part of another kind is answer text. Nothing after the
 * end token is read.
 */
class InklingOutputParser : ChatOutputParser {
    private enum class Mode { HEADER, REASONING, CONTENT, TOOL, DONE }

    private var mode = Mode.HEADER
    private val buf = StringBuilder()
    private val tool = StringBuilder()

    override fun feed(text: String): List<ChatOutputEvent> {
        buf.append(text)
        val out = ArrayList<ChatOutputEvent>()
        while (mode != Mode.DONE && buf.isNotEmpty()) {
            val open = buf.indexOf("<|")
            if (open < 0) {
                // A trailing '<' may be a marker's start.
                val keep = if (buf.endsWith("<")) 1 else 0
                emit(buf.substring(0, buf.length - keep), out)
                buf.delete(0, buf.length - keep)
                break
            }
            if (open > 0) {
                emit(buf.substring(0, open), out)
                buf.delete(0, open)
                continue
            }
            val close = buf.indexOf("|>")
            if (close < 0) {
                // A marker may still be arriving; a long run without a close is text.
                if (buf.length > MAX_MARKER) {
                    emit(buf.substring(0, 2), out)
                    buf.delete(0, 2)
                    continue
                }
                break
            }
            val marker = buf.substring(0, close + 2)
            buf.delete(0, close + 2)
            marker(marker, out)
        }
        return out
    }

    override fun finish(): List<ChatOutputEvent> {
        val out = ArrayList<ChatOutputEvent>()
        if (mode != Mode.DONE && buf.isNotEmpty()) emit(buf.toString(), out)
        buf.setLength(0)
        if (mode == Mode.TOOL) call(out)
        mode = Mode.DONE
        return out
    }

    private fun emit(text: String, out: MutableList<ChatOutputEvent>) {
        if (text.isEmpty()) return
        when (mode) {
            Mode.REASONING -> out += ChatOutputEvent.Reasoning(text)
            Mode.CONTENT -> out += ChatOutputEvent.Content(text)
            Mode.TOOL -> tool.append(text)
            Mode.HEADER, Mode.DONE -> Unit // a turn header's name; the tool call names itself
        }
    }

    private fun marker(m: String, out: MutableList<ChatOutputEvent>) {
        when (m) {
            "<|message_model|>" -> mode = Mode.HEADER
            "<|content_thinking|>" -> mode = Mode.REASONING
            "<|content_text|>" -> mode = Mode.CONTENT
            "<|content_invoke_tool_json|>" -> { mode = Mode.TOOL; tool.setLength(0) }
            "<|end_message|>" -> { if (mode == Mode.TOOL) call(out); mode = Mode.HEADER }
            "<|content_model_end_sampling|>", "<|endoftext|>" -> { if (mode == Mode.TOOL) call(out); mode = Mode.DONE }
            else -> if (m.startsWith("<|content_")) mode = Mode.CONTENT else emit(m, out)
        }
    }

    private fun call(out: MutableList<ChatOutputEvent>) {
        val body = tool.toString()
        tool.setLength(0)
        val o = runCatching { parseJson(body) as JsonObject }.getOrNull()
        val name = (o?.get("name") as? JsonString)?.value
        if (o == null || name == null) {
            // Not a call the parser can read: kept as answer text.
            out += ChatOutputEvent.Content(body)
            return
        }
        out += ChatOutputEvent.Call(ChatToolCall(name, compact(o["args"] ?: JsonObject(emptyMap()))))
    }

    private fun compact(v: JsonValue): String = when (v) {
        is JsonObject -> v.fields.entries.joinToString(",", "{", "}") { jsonQuote(it.key) + ":" + compact(it.value) }
        is JsonArray -> v.elements.joinToString(",", "[", "]") { compact(it) }
        is JsonString -> jsonQuote(v.value)
        is JsonNumber -> v.raw
        is JsonBool -> v.value.toString()
        JsonNull -> "null"
    }

    private companion object {
        const val MAX_MARKER = 64
    }
}
