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

/** What a piece of Ripple's generated text is. */
sealed interface RippleEvent {
    /** Text inside the leading `<think>` block. */
    data class Reasoning(val text: String) : RippleEvent

    /** Text of the answer. */
    data class Content(val text: String) : RippleEvent

    /** A finished `<tool_call>` block. */
    data class Call(val call: ChatToolCall) : RippleEvent
}

/**
 * Splits Ripple's generated text, as it arrives, into reasoning, answer and tool calls, as the
 * parsers Ricursive publishes with the checkpoint do. The markers are plain text, not special
 * tokens, so a marker may arrive in pieces. Ripple decides per turn whether to reason: only text
 * that starts with `<think>` has reasoning.
 *
 * ```
 *   <think>\nreasoning\n</think>\nanswer\n<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
 * ```
 *
 * The newlines around the markers belong to the markers. Text after the first tool call is
 * dropped. A block that does not parse as a call, or that never closes, is answer text.
 */
class RippleOutputParser {
    private enum class State { START, THINK, AFTER_THINK, CONTENT, CALL, AFTER_CALL }

    private var state = State.START
    private val buffer = StringBuilder()
    private var called = false

    /** The events [text] completes; the parser keeps what may be part of a marker. */
    fun feed(text: String): List<RippleEvent> {
        buffer.append(text)
        val out = ArrayList<RippleEvent>()
        while (step(out)) { /* until it needs more text */ }
        return out
    }

    /** The events of what is left at the end of the text. */
    fun finish(): List<RippleEvent> {
        val rest = buffer.toString()
        buffer.clear()
        val out = ArrayList<RippleEvent>()
        when (state) {
            State.START, State.AFTER_THINK, State.CONTENT -> if (rest.isNotEmpty()) out += RippleEvent.Content(rest)
            State.THINK -> if (rest.isNotEmpty()) out += RippleEvent.Reasoning(rest)
            State.CALL -> if (!called) out += RippleEvent.Content(TOOL_START + rest)
            State.AFTER_CALL -> {}
        }
        return out
    }

    private fun step(out: MutableList<RippleEvent>): Boolean {
        val s = buffer.toString()
        if (s.isEmpty()) return false
        when (state) {
            State.START -> {
                if (s.startsWith(THINK_START)) {
                    if (s.length == THINK_START.length) return false
                    buffer.delete(0, THINK_START.length + if (s[THINK_START.length] == '\n') 1 else 0)
                    state = State.THINK
                    return true
                }
                if (THINK_START.startsWith(s)) return false
                state = State.CONTENT
                return true
            }
            State.THINK -> {
                val end = s.indexOf(THINK_END)
                if (end >= 0) {
                    val text = s.substring(0, end).removeSuffix("\n")
                    if (text.isNotEmpty()) out += RippleEvent.Reasoning(text)
                    buffer.delete(0, end + THINK_END.length)
                    state = State.AFTER_THINK
                    return true
                }
                val hold = maxOf(partialSuffix(s, "\n" + THINK_END), partialSuffix(s, THINK_END))
                if (s.length > hold) out += RippleEvent.Reasoning(s.substring(0, s.length - hold))
                buffer.delete(0, s.length - hold)
                return false
            }
            State.AFTER_THINK -> {
                if (s[0] == '\n') buffer.deleteAt(0)
                state = State.CONTENT
                return true
            }
            State.CONTENT -> {
                val start = s.indexOf(TOOL_START)
                if (start >= 0) {
                    val text = s.substring(0, start).removeSuffix("\n")
                    if (text.isNotEmpty()) out += RippleEvent.Content(text)
                    buffer.delete(0, start + TOOL_START.length)
                    state = State.CALL
                    return true
                }
                val hold = maxOf(partialSuffix(s, "\n" + TOOL_START), partialSuffix(s, TOOL_START))
                if (s.length > hold) out += RippleEvent.Content(s.substring(0, s.length - hold))
                buffer.delete(0, s.length - hold)
                return false
            }
            State.CALL -> {
                val end = s.indexOf(TOOL_END)
                if (end < 0) return false
                val body = s.substring(0, end)
                val call = parseCall(body)
                if (call != null) {
                    out += RippleEvent.Call(call)
                    called = true
                } else if (!called) {
                    out += RippleEvent.Content(TOOL_START + body + TOOL_END)
                }
                buffer.delete(0, end + TOOL_END.length)
                state = State.AFTER_CALL
                return true
            }
            State.AFTER_CALL -> {
                // Only further calls count once there has been one.
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
        val o = try {
            parseJson(body.trim()) as? JsonObject
        } catch (_: JsonException) {
            null
        } ?: return null
        val name = (o["name"] as? JsonString)?.value ?: return null
        val args = o["arguments"] as? JsonObject ?: return null
        return ChatToolCall(name, jsonText(args))
    }

    private fun jsonText(v: JsonValue): String = when (v) {
        is JsonObject -> v.fields.entries.joinToString(", ", "{", "}") { jsonQuote(it.key) + ": " + jsonText(it.value) }
        is JsonArray -> v.elements.joinToString(", ", "[", "]") { jsonText(it) }
        is JsonString -> jsonQuote(v.value)
        is JsonNumber -> v.raw
        is JsonBool -> v.value.toString()
        JsonNull -> "null"
    }

    /** The length of the longest suffix of [text] that is a strict prefix of [marker]. */
    private fun partialSuffix(text: String, marker: String): Int {
        for (i in minOf(text.length, marker.length - 1) downTo 1) {
            if (marker.startsWith(text.substring(text.length - i))) return i
        }
        return 0
    }

    private companion object {
        const val THINK_START = "<think>"
        const val THINK_END = "</think>"
        const val TOOL_START = "<tool_call>"
        const val TOOL_END = "</tool_call>"
    }
}

/** A whole generated turn, split: its reasoning, its answer and its tool calls. */
data class RippleOutput(val reasoning: String, val content: String, val toolCalls: List<ChatToolCall>) {
    companion object {
        fun parse(text: String): RippleOutput {
            val parser = RippleOutputParser()
            val events = parser.feed(text) + parser.finish()
            return RippleOutput(
                events.filterIsInstance<RippleEvent.Reasoning>().joinToString("") { it.text },
                events.filterIsInstance<RippleEvent.Content>().joinToString("") { it.text },
                events.filterIsInstance<RippleEvent.Call>().map { it.call },
            )
        }
    }
}
