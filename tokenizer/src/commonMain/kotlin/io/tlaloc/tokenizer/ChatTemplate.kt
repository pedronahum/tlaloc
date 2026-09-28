package io.tlaloc.tokenizer

/**
 * One chat message. [reasoningContent] is the model's reasoning for an
 * assistant turn, for templates that render it (Qwen3, Muse Glimmer).
 */
data class ChatMessage(
    val role: String,
    val content: String,
    val reasoningContent: String? = null,
)

/**
 * Variables a chat template reads. Each template ignores the ones it does not
 * know, as a Jinja template does.
 *
 * @property addGenerationPrompt end with the header of an assistant turn.
 * @property enableThinking Qwen3: `false` writes an empty `<think>` block after
 *   the generation prompt; `null` and `true` leave it out.
 * @property currentDate Muse Glimmer: the `Current date:` line, `YYYY-MM-DD`;
 *   null prints today's date, as transformers' `strftime_now` does.
 * @property reasoningStrength Muse Glimmer: the `Reasoning strength:` line; null is `high`.
 * @property knowledgeCutoff Muse Glimmer: the `Knowledge cutoff:` line; null is `2026-01-04`.
 */
data class ChatOptions(
    val addGenerationPrompt: Boolean = true,
    val enableThinking: Boolean? = null,
    val currentDate: String? = null,
    val reasoningStrength: String? = null,
    val knowledgeCutoff: String? = null,
)

/**
 * The chat templates this module renders, each written out in Kotlin from the
 * checkpoint's Jinja source and checked against transformers'
 * `apply_chat_template` output. Tools and tool calls are not rendered; a
 * message whose role the template does not handle is refused.
 */
enum class ChatTemplate {
    /** Qwen3 (`<|im_start|>` turns, `<think>` reasoning blocks). */
    QWEN3,

    /** Zephyr, as TinyLlama-1.1B-Chat uses it (`<|user|>` turns closed by the EOS token). */
    ZEPHYR,

    /** Muse Glimmer (`<|start|>role<|message|>` turns, a generated system block). */
    MUSE_GLIMMER,
    ;

    internal fun render(messages: List<ChatMessage>, options: ChatOptions, bosToken: String?, eosToken: String?): String {
        require(messages.isNotEmpty()) { "a chat needs at least one message" }
        return when (this) {
            QWEN3 -> renderQwen3(messages, options)
            ZEPHYR -> renderZephyr(messages, options, eosToken ?: throw IllegalStateException("the Zephyr template needs an eos_token"))
            MUSE_GLIMMER -> renderMuse(messages, options, bosToken ?: throw IllegalStateException("the Muse Glimmer template needs a bos_token"))
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
