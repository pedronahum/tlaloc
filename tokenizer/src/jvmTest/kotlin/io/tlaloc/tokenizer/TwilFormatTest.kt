package io.tlaloc.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals

/** TwIL-8B's output parser; its chat template is checked against transformers in [GoldenTest]. */
class TwilFormatTest {
    private val tool = """{"type": "function", "function": {"name": "add_numbers", "parameters": {"type": "object", "properties": """ +
        """{"a": {"type": "integer"}, "b": {"type": "number"}, "note": {"type": "string"}, "opts": {"type": "object"}, "flag": {"type": ["boolean", "null"]}}}}}"""

    private fun parse(text: String, reasoning: Boolean = true) = ChatOutput.parse(text, TwilOutputParser(reasoning, listOf(tool)))

    private fun render(messages: List<ChatMessage>, options: ChatOptions) = ChatTemplate.TWIL.render(messages, options, "<s>", "<|im_end|>")

    @Test
    fun theTextStartsInTheReasoningBlock() {
        assertEquals(ChatOutput("R\nS", "A", emptyList()), parse("R\nS\n</think>\n\nA"))
        assertEquals(ChatOutput("R", "", emptyList()), parse("R"))
        assertEquals(ChatOutput("", "A", emptyList()), parse("</think>A"))
        // With thinking off the prompt closed the block: everything is answer.
        assertEquals(ChatOutput("", "R</think>A", emptyList()), parse("R</think>A", reasoning = false))
    }

    @Test
    fun parametersTakeTheirSchemaTypes() {
        val text = "R\n</think>\n\nCalling.\n<tool_call>\n<function=add_numbers>\n<parameter=a>\n23\n</parameter>\n<parameter=b>\n1.5\n</parameter>\n" +
            "<parameter=note>\n42\n</parameter>\n<parameter=opts>\n{\"x\": [1, 2]}\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n" +
            "<parameter=extra>\ntwo\nlines\n</parameter>\n</function>\n</tool_call>"
        assertEquals(
            ChatOutput(
                "R", "Calling.",
                listOf(ChatToolCall("add_numbers", """{"a": 23, "b": 1.5, "note": "42", "opts": {"x": [1, 2]}, "flag": true, "extra": "two\nlines"}""")),
            ),
            parse(text),
        )
        // Text that is not of the schema's type stays a string; null is null.
        assertEquals(
            listOf(ChatToolCall("add_numbers", """{"a": "2.5", "b": null}""")),
            parse("</think><tool_call>\n<function=add_numbers>\n<parameter=a>\n2.5\n</parameter>\n<parameter=b>\nnull\n</parameter>\n</function>\n</tool_call>").toolCalls,
        )
    }

    @Test
    fun anUnparsedBlockIsAnswerText() {
        assertEquals(ChatOutput("", "<tool_call>no function</tool_call>", emptyList()), parse("</think><tool_call>no function</tool_call>"))
        assertEquals(ChatOutput("", "<tool_call>\n<function=x>", emptyList()), parse("</think><tool_call>\n<function=x>"))
    }

    /** A rendered turn with a call parses back to that call. */
    @Test
    fun aRenderedTurnParsesBack() {
        val call = ChatToolCall("add_numbers", """{"a": 23, "b": 1.5, "note": "hi"}""")
        val turn = ChatMessage("assistant", "A", reasoningContent = "R", toolCalls = listOf(call))
        val text = render(listOf(ChatMessage("user", "U"), turn), ChatOptions(addGenerationPrompt = false, tools = listOf(tool)))
            .substringAfterLast("<|im_start|>assistant\n<think>\n").removeSuffix("<|im_end|>\n")
        assertEquals(ChatOutput("R", "A", listOf(call)), parse(text))
    }

    @Test
    fun anyChunkingGivesTheSameEvents() {
        val text = "R\nS\n</think>\n\nAn <answer>\n<tool_call>\n<function=add_numbers>\n<parameter=a>\n23\n</parameter>\n</function>\n</tool_call>\n" +
            "<tool_call>\n<function=add_numbers>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>"
        val whole = parse(text)
        assertEquals(2, whole.toolCalls.size)
        for (size in 1..14) {
            val parser = TwilOutputParser(true, listOf(tool))
            val events = text.chunked(size).flatMap { parser.feed(it) } + parser.finish()
            assertEquals(whole, ChatOutput.of(events), "chunks of $size")
        }
    }
}
