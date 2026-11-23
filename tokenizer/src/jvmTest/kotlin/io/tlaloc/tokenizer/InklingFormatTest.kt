package io.tlaloc.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals

/** [InklingOutputParser] on what Inkling 2 generates, whole and in pieces. */
class InklingFormatTest {
    private val turn = "<|message_model|><|content_thinking|>The user asks <b>weather</b>.<|end_message|>" +
        "<|message_model|><|content_text|>Let me check.<|end_message|>" +
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":2,\"opts\":{\"a\":[1,2.5,null]}}}<|end_message|>" +
        "<|content_model_end_sampling|>ignored"

    private val want = ChatOutput(
        "The user asks <b>weather</b>.",
        "Let me check.",
        listOf(ChatToolCall("get_weather", "{\"city\":\"Paris\",\"days\":2,\"opts\":{\"a\":[1,2.5,null]}}")),
    )

    @Test
    fun wholeTurn() = assertEquals(want, ChatOutput.parse(turn, InklingOutputParser()))

    @Test
    fun anyPieces() {
        for (size in listOf(1, 2, 3, 5, 7, 13)) {
            val p = InklingOutputParser()
            val events = turn.chunked(size).flatMap { p.feed(it) } + p.finish()
            assertEquals(want, ChatOutput.of(events), "pieces of $size")
        }
    }

    @Test
    fun answerWithoutReasoningAndAnUnfinishedTurn() {
        assertEquals(ChatOutput("", "Brooklyn, Queens and the Bronx.", emptyList()),
            ChatOutput.parse("<|message_model|><|content_text|>Brooklyn, Queens and the Bronx.<|end_message|><|content_model_end_sampling|>", InklingOutputParser()))
        // Cut off mid-reasoning (a token budget): what arrived is kept.
        assertEquals(ChatOutput("Still thinking", "", emptyList()),
            ChatOutput.parse("<|message_model|><|content_thinking|>Still thinking", InklingOutputParser()))
    }

    @Test
    fun templateHasTheParser() {
        assertEquals(InklingOutputParser::class, ChatTemplate.INKLING.outputParser(ChatOptions())!!::class)
    }
}
