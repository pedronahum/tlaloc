package io.tlaloc.tokenizer

/** What a piece of a model's generated turn is. */
sealed interface ChatOutputEvent {
    /** Text of the reasoning block. */
    data class Reasoning(val text: String) : ChatOutputEvent

    /** Text of the answer. */
    data class Content(val text: String) : ChatOutputEvent

    /** A finished tool call. */
    data class Call(val call: ChatToolCall) : ChatOutputEvent
}

/**
 * Splits a model's generated text, as it arrives, into reasoning, answer and tool calls. Markers
 * may arrive in pieces; a parser keeps what may be the start of one until it knows.
 */
interface ChatOutputParser {
    /** The events [text] completes. */
    fun feed(text: String): List<ChatOutputEvent>

    /** The events of what is left at the end of the text. */
    fun finish(): List<ChatOutputEvent>
}

/** A whole generated turn, split: its reasoning, its answer and its tool calls. */
data class ChatOutput(val reasoning: String, val content: String, val toolCalls: List<ChatToolCall>) {
    companion object {
        /** [text] through [parser], which must be new. */
        fun parse(text: String, parser: ChatOutputParser): ChatOutput = of(parser.feed(text) + parser.finish())

        /** The turn [events] make up. */
        fun of(events: List<ChatOutputEvent>): ChatOutput = ChatOutput(
            events.filterIsInstance<ChatOutputEvent.Reasoning>().joinToString("") { it.text },
            events.filterIsInstance<ChatOutputEvent.Content>().joinToString("") { it.text },
            events.filterIsInstance<ChatOutputEvent.Call>().map { it.call },
        )
    }
}

/** The length of the longest suffix of [text] that is a strict prefix of [marker]. */
internal fun partialSuffix(text: String, marker: String): Int {
    for (i in minOf(text.length, marker.length - 1) downTo 1) {
        if (marker.startsWith(text.substring(text.length - i))) return i
    }
    return 0
}
