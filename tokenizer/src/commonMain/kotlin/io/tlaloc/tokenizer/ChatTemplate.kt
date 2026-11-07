package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson

/**
 * One chat message. [reasoningContent] is the model's reasoning for an
 * assistant turn, for templates that render it (Qwen3, Muse Glimmer).
 */
data class ChatMessage(
    val role: String,
    val content: String,
    val reasoningContent: String? = null,
    /** The tool calls of an assistant turn, for the templates that render them (Ripple, TwIL). */
    val toolCalls: List<ChatToolCall> = emptyList(),
)

/** A tool call in a chat: its function's name and its arguments, a JSON object as text. */
data class ChatToolCall(val name: String, val argumentsJson: String)

/**
 * Variables a chat template reads. Each template ignores the ones it does not
 * know, as a Jinja template does.
 *
 * @property addGenerationPrompt end with the header of an assistant turn.
 * @property enableThinking Qwen3: `false` writes an empty `<think>` block after
 *   the generation prompt; `null` and `true` leave it out. TwIL: the generation
 *   prompt opens the reasoning block unless it is `false`.
 * @property currentDate Muse Glimmer: the `Current date:` line, `YYYY-MM-DD`;
 *   null prints today's date, as transformers' `strftime_now` does.
 * @property reasoningStrength Muse Glimmer: the `Reasoning strength:` line; null is `high`.
 * @property knowledgeCutoff Muse Glimmer: the `Knowledge cutoff:` line; null is `2026-01-04`.
 * @property tools Ripple and TwIL: each tool's definition, a JSON object as text as Python's
 *   `json.dumps` writes it. Ripple writes it as given; TwIL writes the `function` object's keys.
 * @property clearThinking Ripple: earlier turns' reasoning is left out; by default it is kept.
 * @property reasoningEffort TwIL: `low` adds the low-effort note to the last user turn.
 * @property truncateHistoryThinking TwIL: the reasoning of turns before the last user turn is
 *   left out; null is the template's default, true.
 */
data class ChatOptions(
    val addGenerationPrompt: Boolean = true,
    val enableThinking: Boolean? = null,
    val currentDate: String? = null,
    val reasoningStrength: String? = null,
    val knowledgeCutoff: String? = null,
    val tools: List<String> = emptyList(),
    val clearThinking: Boolean = false,
    val reasoningEffort: String? = null,
    val truncateHistoryThinking: Boolean? = null,
)

/**
 * The chat templates this module renders, each written out in Kotlin from the
 * checkpoint's Jinja source and checked against transformers'
 * `apply_chat_template` output. [RIPPLE] and [TWIL] render tools and tool calls; a
 * message whose role the template does not handle is refused.
 */
enum class ChatTemplate {
    /** Qwen3 (`<|im_start|>` turns, `<think>` reasoning blocks). */
    QWEN3,

    /** Zephyr, as TinyLlama-1.1B-Chat uses it (`<|user|>` turns closed by the EOS token). */
    ZEPHYR,

    /** Muse Glimmer (`<|start|>role<|message|>` turns, a generated system block). */
    MUSE_GLIMMER,

    /** Ripple (`<|im_start|>` turns, tool definitions and calls, reasoning kept in earlier turns). */
    RIPPLE,

    /**
     * webAI's TwIL-8B, balanced and verbose (`<|im_start|>` turns, a system turn always, the
     * generation prompt inside `<think>`, tool calls as `<function=...>` blocks).
     */
    TWIL,
    ;

    /**
     * A parser for what a model given a prompt rendered with [options] generates, or null when
     * the template's output is plain text.
     */
    fun outputParser(options: ChatOptions): ChatOutputParser? = when (this) {
        RIPPLE -> RippleOutputParser()
        TWIL -> TwilOutputParser(reasoning = options.enableThinking != false, tools = options.tools)
        else -> null
    }

    /** Whether the template renders tool definitions and calls. */
    val rendersTools: Boolean get() = this == RIPPLE || this == TWIL

    internal fun render(messages: List<ChatMessage>, options: ChatOptions, bosToken: String?, eosToken: String?): String {
        require(messages.isNotEmpty()) { "a chat needs at least one message" }
        return when (this) {
            QWEN3 -> renderQwen3(messages, options)
            ZEPHYR -> renderZephyr(messages, options, eosToken ?: throw IllegalStateException("the Zephyr template needs an eos_token"))
            MUSE_GLIMMER -> renderMuse(messages, options, bosToken ?: throw IllegalStateException("the Muse Glimmer template needs a bos_token"))
            RIPPLE -> renderRipple(messages, options)
            TWIL -> renderTwil(messages, options)
        }
    }

    companion object {
        // FNV-1a 64 of the template source's UTF-8 bytes. A template that
        // differs by one byte is not recognized: rendering it with a
        // hand-written renderer for a different source would be a guess.
        private val KNOWN = mapOf(
            0x959A4551391A6549uL to QWEN3,
            0xCD8B49000DB02C2CuL to ZEPHYR,
            0x67BBCD170CA4388EuL to MUSE_GLIMMER,
            0x2780B582ED5F15B2uL to RIPPLE,
            0xA6C61EC67F38BF63uL to TWIL,
        )

        /** The template whose Jinja source is exactly [source], or null. */
        fun recognize(source: String): ChatTemplate? = KNOWN[fnv1a64(source)]

        internal fun fnv1a64(s: String): ULong {
            var h = 0xcbf29ce484222325uL
            for (b in s.encodeToByteArray()) {
                h = h xor (b.toUByte().toULong())
                h *= 0x100000001b3uL
            }
            return h
        }
    }
}

private fun refuseRole(template: String, role: String): Nothing =
    throw IllegalArgumentException("the $template chat template does not render role '$role'")

// Python's str.strip/lstrip/rstrip with an explicit character.
private fun String.pyStrip(c: Char) = trim { it == c }
private fun String.pyLstrip(c: Char) = trimStart { it == c }
private fun String.pyRstrip(c: Char) = trimEnd { it == c }

private fun renderQwen3(messages: List<ChatMessage>, options: ChatOptions): String {
    val sb = StringBuilder()
    if (messages[0].role == "system") {
        sb.append("<|im_start|>system\n").append(messages[0].content).append("<|im_end|>\n")
    }
    // The index of the last user message that is not a wrapped tool response.
    var lastQuery = messages.size - 1
    for (i in messages.indices.reversed()) {
        val m = messages[i]
        if (m.role == "user" && !(m.content.startsWith("<tool_response>") && m.content.endsWith("</tool_response>"))) {
            lastQuery = i
            break
        }
    }
    for ((i, m) in messages.withIndex()) {
        var content = m.content
        when (m.role) {
            "user" -> sb.append("<|im_start|>user\n").append(content).append("<|im_end|>\n")
            "system" -> if (i != 0) sb.append("<|im_start|>system\n").append(content).append("<|im_end|>\n")
            "assistant" -> {
                var reasoning = ""
                if (m.reasoningContent != null) {
                    reasoning = m.reasoningContent
                } else if ("</think>" in content) {
                    reasoning = content.substringBefore("</think>").pyRstrip('\n').substringAfterLast("<think>").pyLstrip('\n')
                    content = content.substringAfterLast("</think>").pyLstrip('\n')
                }
                if (i > lastQuery && (i == messages.size - 1 || reasoning.isNotEmpty())) {
                    sb.append("<|im_start|>assistant\n<think>\n").append(reasoning.pyStrip('\n'))
                        .append("\n</think>\n\n").append(content.pyLstrip('\n'))
                } else {
                    sb.append("<|im_start|>assistant\n").append(content)
                }
                sb.append("<|im_end|>\n")
            }
            "tool" -> {
                if (i == 0 || messages[i - 1].role != "tool") sb.append("<|im_start|>user")
                sb.append("\n<tool_response>\n").append(content).append("\n</tool_response>")
                if (i == messages.size - 1 || messages[i + 1].role != "tool") sb.append("<|im_end|>\n")
            }
            else -> refuseRole("Qwen3", m.role)
        }
    }
    if (options.addGenerationPrompt) {
        sb.append("<|im_start|>assistant\n")
        if (options.enableThinking == false) sb.append("<think>\n\n</think>\n\n")
    }
    return sb.toString()
}

private fun renderZephyr(messages: List<ChatMessage>, options: ChatOptions, eos: String): String {
    val sb = StringBuilder()
    for ((i, m) in messages.withIndex()) {
        when (m.role) {
            "user", "system", "assistant" -> sb.append("<|").append(m.role).append("|>\n").append(m.content).append(eos).append('\n')
            else -> refuseRole("Zephyr", m.role)
        }
        if (i == messages.size - 1 && options.addGenerationPrompt) sb.append("<|assistant|>\n")
    }
    return sb.toString()
}

private fun renderMuse(messages: List<ChatMessage>, options: ChatOptions, bos: String): String {
    val reasoning = "Reasoning strength: " + (options.reasoningStrength?.takeIf { it.isNotEmpty() } ?: "high") + "."
    val recipients = "# Valid recipients: \"self\", \"user\"."
    val sb = StringBuilder(bos)
    if (messages.none { it.role == "system" }) {
        sb.append("<|start|>system<|message|>You are a helpful AI assistant.")
        sb.append("\nKnowledge cutoff: ").append(options.knowledgeCutoff?.takeIf { it.isNotEmpty() } ?: "2026-01-04").append('.')
        sb.append("\nCurrent date: ").append(options.currentDate?.takeIf { it.isNotEmpty() } ?: todayIsoDate()).append('.')
        sb.append("\n\n").append(reasoning).append("\n\n").append(recipients).append("<|eot|>")
    }
    for (m in messages) {
        when (m.role) {
            "system" -> {
                val text = m.content
                    .replace("Reasoning effort", "Reasoning strength")
                    .replace("Reasoning Effort", "Reasoning Strength")
                    .replace("reasoning effort", "reasoning strength")
                    .replace("REASONING EFFORT", "REASONING STRENGTH")
                sb.append("<|start|>system<|message|>").append(text)
                if ("reasoning strength" !in text.lowercase()) sb.append("\n\n").append(reasoning)
                sb.append("\n\n").append(recipients).append("<|eot|>")
            }
            "user" -> sb.append("<|start|>user<|message|>").append(m.content).append("<|eot|>")
            "assistant" -> {
                if (!m.reasoningContent.isNullOrEmpty()) {
                    sb.append("<|start|>assistant to=self<|message|>").append(m.reasoningContent).append("<|eom|>")
                }
                sb.append("<|start|>assistant to=user<|message|>").append(m.content).append("<|eot|>")
            }
            else -> refuseRole("Muse Glimmer", m.role)
        }
    }
    if (options.addGenerationPrompt) sb.append("<|start|>assistant")
    return sb.toString()
}

private fun renderRipple(messages: List<ChatMessage>, options: ChatOptions): String {
    val sb = StringBuilder()
    val leadingSystem = messages[0].role == "system"
    if (options.tools.isNotEmpty()) {
        sb.append("<|im_start|>system\n")
        if (leadingSystem) sb.append(messages[0].content).append("\n\n")
        sb.append("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools> XML tags:\n<tools>")
        for (tool in options.tools) sb.append('\n').append(tool)
        sb.append(
            "\n</tools>\n\nFor each function call, return exactly one valid JSON object with function name and arguments inside XML tool-call tags. " +
                "The arguments field must be a JSON object, not a string, and strings must use valid JSON escaping:\n<tool_call>\n" +
                "{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call><|im_end|>\n",
        )
    } else if (leadingSystem) {
        sb.append("<|im_start|>system\n").append(messages[0].content).append("<|im_end|>\n")
    }
    for ((i, m) in messages.withIndex()) {
        var content = m.content
        if (m.role == "assistant" && !m.reasoningContent.isNullOrEmpty() && !options.clearThinking && !content.trimStart().startsWith("<think>")) {
            content = "<think>\n" + m.reasoningContent + "\n</think>\n" + content
        }
        when {
            m.role == "system" && i == 0 -> {}
            m.role == "user" || m.role == "system" || (m.role == "assistant" && m.toolCalls.isEmpty()) ->
                sb.append("<|im_start|>").append(m.role).append('\n').append(content).append("<|im_end|>\n")
            m.role == "assistant" -> {
                sb.append("<|im_start|>assistant")
                if (content.isNotEmpty()) sb.append('\n').append(content)
                for (call in m.toolCalls) {
                    sb.append("\n<tool_call>\n{\"name\": ").append(jsonQuote(call.name))
                        .append(", \"arguments\": ").append(call.argumentsJson).append("}\n</tool_call>")
                }
                sb.append("<|im_end|>\n")
            }
            m.role == "tool" -> {
                if (i == 0 || messages[i - 1].role != "tool") sb.append("<|im_start|>user")
                sb.append("\n<tool_response>\n").append(m.content).append("\n</tool_response>")
                if (i == messages.size - 1 || messages[i + 1].role != "tool") sb.append("<|im_end|>\n")
            }
            else -> refuseRole("Ripple", m.role)
        }
    }
    if (options.addGenerationPrompt) sb.append("<|im_start|>assistant\n")
    return sb.toString()
}

private fun renderTwil(messages: List<ChatMessage>, options: ChatOptions): String {
    val enableThinking = options.enableThinking ?: true
    val lowEffort = options.reasoningEffort == "low"
    val truncate = options.truncateHistoryThinking ?: true
    val sb = StringBuilder()
    val systemMessage = if (messages[0].role == "system") messages[0].content else ""
    val loop = if (messages[0].role == "system") messages.drop(1) else messages
    val lastUser = loop.indexOfLast { it.role == "user" }
    sb.append("<|im_start|>system\n").append(systemMessage)
    if (options.tools.isNotEmpty()) {
        if (systemMessage.isNotEmpty()) sb.append("\n\n")
        sb.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for (text in options.tools) {
            val tool = parseJson(text) as? JsonObject ?: throw IllegalArgumentException("a tool definition is a JSON object: $text")
            val f = tool["function"] as? JsonObject ?: tool
            if (pyTruthy(f["defer_loading"])) continue
            sb.append('\n').append(
                f.fields.entries.filter { it.key != "defer_loading" && it.key != "strict" }
                    .joinToString(", ", "{", "}") { "\"" + it.key + "\": " + pyJson(it.value) },
            )
        }
        sb.append("\n</tools>")
        sb.append(TWIL_TOOL_INSTRUCTIONS)
    }
    sb.append("<|im_end|>\n")
    for ((i, m) in loop.withIndex()) {
        when (m.role) {
            "assistant" -> {
                val content = if (m.reasoningContent != null && m.reasoningContent.pyTrim().isNotEmpty()) {
                    "<think>\n" + m.reasoningContent + "\n</think>\n" + m.content
                } else if ("<think>" !in m.content && "</think>" !in m.content) {
                    "<think></think>" + m.content
                } else {
                    m.content
                }
                val earlier = truncate && i < lastUser
                if (m.toolCalls.isNotEmpty()) {
                    sb.append("<|im_start|>assistant\n")
                    if (content.pyTrim().isNotEmpty()) {
                        if (!earlier) {
                            sb.append(content.pyTrim()).append('\n')
                        } else {
                            var c = content
                            if ("</think>" in c) c = c.substringAfterLast("</think>") else if ("<think>" in c) c = c.substringBefore("<think>")
                            sb.append("<think></think>").append(c.pyTrim()).append('\n')
                        }
                    } else {
                        sb.append("<think></think>")
                    }
                    for (call in m.toolCalls) {
                        sb.append("<tool_call>\n<function=").append(call.name).append(">\n")
                        val args = parseJson(call.argumentsJson) as? JsonObject
                            ?: throw IllegalArgumentException("a tool call's arguments are a JSON object: ${call.argumentsJson}")
                        for ((k, v) in args.fields) sb.append("<parameter=").append(k).append(">\n").append(pyString(v)).append("\n</parameter>\n")
                        sb.append("</function>\n</tool_call>\n")
                    }
                    sb.append("<|im_end|>\n")
                } else if (!earlier) {
                    sb.append("<|im_start|>assistant\n").append(content.pyTrim()).append("<|im_end|>\n")
                } else {
                    var c = content
                    if ("<think>" in c && "</think>" in c) c = "<think></think>" + c.substringAfterLast("</think>")
                    sb.append("<|im_start|>assistant\n").append(c.pyTrim()).append("<|im_end|>\n")
                }
            }
            "user", "system" -> {
                sb.append("<|im_start|>").append(m.role).append('\n').append(m.content)
                if (m.role == "user" && i == lastUser && lowEffort) sb.append("\n\n{reasoning effort: low}")
                sb.append("<|im_end|>\n")
            }
            "tool" -> {
                if (i > 0 && loop[i - 1].role != "tool") sb.append("<|im_start|>user\n")
                sb.append("<tool_response>\n").append(m.content).append("\n</tool_response>\n")
                if (i == loop.size - 1 || loop[i + 1].role != "tool") sb.append("<|im_end|>\n")
            }
            else -> sb.append("<|im_start|>").append(m.role).append('\n').append(m.content).append("<|im_end|>\n")
        }
    }
    if (options.addGenerationPrompt) sb.append(if (enableThinking) "<|im_start|>assistant\n<think>\n" else "<|im_start|>assistant\n<think></think>")
    return sb.toString()
}

private const val TWIL_TOOL_INSTRUCTIONS =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n" +
        "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n" +
        "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner " +
        "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be " +
        "specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, " +
        "but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge " +
        "and do not tell the user about function calls\n</IMPORTANT>"

// Python's str.strip(): Unicode whitespace from both ends.
private fun String.pyTrim() = trim { it.isWhitespace() }

// Jinja truthiness of a JSON value read from Python.
private fun pyTruthy(v: JsonValue?): Boolean = when (v) {
    null, JsonNull -> false
    is JsonBool -> v.value
    is JsonString -> v.value.isNotEmpty()
    is JsonNumber -> v.value != 0.0
    is JsonArray -> v.elements.isNotEmpty()
    is JsonObject -> v.fields.isNotEmpty()
}

// `tojson`: Python's json.dumps, numbers as the Python-written text holds them.
private fun pyJson(v: JsonValue): String = when (v) {
    is JsonObject -> v.fields.entries.joinToString(", ", "{", "}") { jsonQuote(it.key) + ": " + pyJson(it.value) }
    is JsonArray -> v.elements.joinToString(", ", "[", "]") { pyJson(it) }
    is JsonString -> jsonQuote(v.value)
    is JsonNumber -> v.raw
    is JsonBool -> v.value.toString()
    JsonNull -> "null"
}

// TwIL's argument rendering: mappings and lists by `tojson`, anything else by Python's str().
private fun pyString(v: JsonValue): String = when (v) {
    is JsonObject, is JsonArray -> pyJson(v)
    is JsonString -> v.value
    is JsonNumber -> v.raw
    is JsonBool -> if (v.value) "True" else "False"
    JsonNull -> "None"
}
