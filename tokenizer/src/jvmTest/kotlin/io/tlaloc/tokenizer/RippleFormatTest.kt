package io.tlaloc.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** Ripple's chat template, written out from its Jinja source, and its output parser. */
class RippleFormatTest {
    private val call = ChatToolCall("add_numbers", """{"a": 23, "b": 19}""")
    private val tool = """{"type": "function", "function": {"name": "add_numbers"}}"""

    private fun parse(text: String) = ChatOutput.parse(text, RippleOutputParser())

    private fun render(messages: List<ChatMessage>, options: ChatOptions = ChatOptions()) =
        ChatTemplate.RIPPLE.render(messages, options, "<|endoftext|>", "<|im_end|>")

    @Test
    fun thePlainTurns() {
        assertEquals(
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n<|im_start|>user\nV<|im_end|>\n<|im_start|>assistant\n",
            render(listOf(ChatMessage("system", "S"), ChatMessage("user", "U"), ChatMessage("assistant", "A"), ChatMessage("user", "V"))),
        )
        assertEquals("<|im_start|>user\nU<|im_end|>\n", render(listOf(ChatMessage("user", "U")), ChatOptions(addGenerationPrompt = false)))
    }

    @Test
    fun reasoningIsKeptUnlessCleared() {
        val turn = listOf(ChatMessage("user", "U"), ChatMessage("assistant", "A", reasoningContent = "R"))
        val options = ChatOptions(addGenerationPrompt = false)
        assertEquals("<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\nR\n</think>\nA<|im_end|>\n", render(turn, options))
        assertEquals("<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n", render(turn, options.copy(clearThinking = true)))
        // Content that already opens with its reasoning is not given it twice.
        assertEquals(
            "<|im_start|>assistant\n <think>x</think>A<|im_end|>\n",
            render(listOf(ChatMessage("assistant", " <think>x</think>A", reasoningContent = "R")), options),
        )
    }

    @Test
    fun toolsCallsAndResponses() {
        val text = render(
            listOf(
                ChatMessage("system", "S"), ChatMessage("user", "U"), ChatMessage("assistant", "", toolCalls = listOf(call)),
                ChatMessage("tool", "42"), ChatMessage("tool", "43"),
            ),
            ChatOptions(tools = listOf(tool)),
        )
        assertEquals(
            "<|im_start|>system\nS\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
                "You are provided with function signatures within <tools> XML tags:\n<tools>\n$tool\n</tools>\n\n" +
                "For each function call, return exactly one valid JSON object with function name and arguments inside XML tool-call tags. " +
                "The arguments field must be a JSON object, not a string, and strings must use valid JSON escaping:\n<tool_call>\n" +
                "{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call><|im_end|>\n" +
                "<|im_start|>user\nU<|im_end|>\n" +
                "<|im_start|>assistant\n<tool_call>\n{\"name\": \"add_numbers\", \"arguments\": {\"a\": 23, \"b\": 19}}\n</tool_call><|im_end|>\n" +
                "<|im_start|>user\n<tool_response>\n42\n</tool_response>\n<tool_response>\n43\n</tool_response><|im_end|>\n" +
                "<|im_start|>assistant\n",
            text,
        )
    }

    @Test
    fun anUnknownRoleIsRefused() {
        assertFailsWith<IllegalArgumentException> { render(listOf(ChatMessage("narrator", "hi"))) }
    }

    @Test
    fun aTurnIsSplitIntoReasoningContentAndCalls() {
        assertEquals(ChatOutput("", "Just this.", emptyList()), parse("Just this."))
        assertEquals(ChatOutput("R\nS", "A", emptyList()), parse("<think>\nR\nS\n</think>\nA"))
        assertEquals(ChatOutput("R", "", emptyList()), parse("<think>\nR"))
        assertEquals(
            ChatOutput("R", "A\n", listOf(call)),
            parse("<think>\nR\n</think>\nA\n\n<tool_call>\n{\"name\": \"add_numbers\", \"arguments\": {\"a\": 23, \"b\": 19}}\n</tool_call> dropped"),
        )
        assertEquals(ChatOutput("", "", listOf(call, call)), parse(("<tool_call>\n{\"name\": \"add_numbers\", \"arguments\": {\"a\": 23, \"b\": 19}}\n</tool_call>\n").repeat(2)))
        // <think> counts only at the start.
        assertEquals(ChatOutput("", "A <think>x</think>", emptyList()), parse("A <think>x</think>"))
        // A block that is no call stays text, closed or not.
        assertEquals(ChatOutput("", "<tool_call>not json</tool_call>", emptyList()), parse("<tool_call>not json</tool_call>"))
        assertEquals(ChatOutput("", "<tool_call>{\"name\"", emptyList()), parse("<tool_call>{\"name\""))
    }

    /** What a rendered turn says comes back: the template and the parser agree on the markers. */
    @Test
    fun aRenderedTurnParsesBack() {
        val turn = ChatMessage("assistant", "A", reasoningContent = "R", toolCalls = listOf(call))
        val text = render(listOf(turn), ChatOptions(addGenerationPrompt = false)).removePrefix("<|im_start|>assistant\n").removeSuffix("<|im_end|>\n")
        assertEquals(ChatOutput("R", "A", listOf(call)), parse(text))
    }

    @Test
    fun anyChunkingGivesTheSameEvents() {
        val text = "<think>\nR\nS\n</think>\nAn <answer>\n\n<tool_call>\n{\"name\": \"add_numbers\", \"arguments\": {\"a\": 23, \"b\": 19}}\n</tool_call>"
        val whole = parse(text)
        for (size in 1..12) {
            val parser = RippleOutputParser()
            val events = text.chunked(size).flatMap { parser.feed(it) } + parser.finish()
            assertEquals(whole, ChatOutput.of(events), "chunks of $size")
        }
    }
}
