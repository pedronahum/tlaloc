package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.endpoint.pythonJson
import io.tlaloc.tokenizer.ChatMessage
import io.tlaloc.tokenizer.ChatOptions
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.ChatToolCall
import io.tlaloc.tokenizer.HfTokenizer

// A conversation as the trainer sees it: the token ids of the whole chat rendered by the same
// Kotlin chat template tlaloc-endpoint renders requests with (HfTokenizer.applyChatTemplate), and
// a mask over the tokens the model produced: each assistant turn from the end of its generation
// prompt to its end token (<|im_end|>) inclusive. For TwIL that is the reasoning (the prompt
// already holds "<think>\n"), "</think>", the answer and the tool calls; for Ripple the turn's
// whole text after "<|im_start|>assistant\n".
//
// Tool definitions and call arguments go through pythonJson, as the endpoint passes them.

/** One message of a recorded conversation (OpenAI roles; tool calls' arguments as JSON text). */
class ConvMessage(
    val role: String,
    val content: String,
    val reasoning: String? = null,
    val toolCalls: List<ChatToolCall> = emptyList(),
) {
    fun forTemplate(): ChatMessage = ChatMessage(
        if (role == "developer") "system" else role, content, reasoning,
        toolCalls.map { ChatToolCall(it.name, pythonJson(parseJson(it.argumentsJson) as JsonObject)) },
    )
}

/** Token ids of a conversation, the assistant mask, and where each assistant turn starts and ends. */
class TrainingSequence(
    val ids: IntArray,
    val mask: BooleanArray,
    /** Per assistant message: [start, end) in [ids]; start is the length of the prompt it was generated from. */
    val spans: List<IntRange>,
    /** Per assistant message: the ids of the prompt it was generated from (a fresh render, generation prompt included). */
    val prompts: List<IntArray>,
    /** Why the conversation cannot be trained on as it is; empty when it can. */
    val problems: List<String>,
) {
    val ok: Boolean get() = problems.isEmpty()
    val maskedTokens: Int get() = mask.count { it }
}

/** Template options as the endpoint builds them from a request's tools (and no chat_template_kwargs). */
fun chatOptions(toolsJson: List<String>, addGenerationPrompt: Boolean): ChatOptions =
    ChatOptions(addGenerationPrompt = addGenerationPrompt, tools = toolsJson.map { pythonJson(parseJson(it)) })

/**
 * Renders [messages] for training. The whole chat is rendered once and encoded once; for every
 * assistant message k the chat before it, rendered with the generation prompt, must be a prefix of
 * the ids (that is what the model was given), and the chat through it must be a prefix too (its
 * turn ends there). A conversation where either fails is reported in [TrainingSequence.problems].
 */
fun renderForTraining(
    tokenizer: HfTokenizer,
    template: ChatTemplate,
    messages: List<ConvMessage>,
    toolsJson: List<String>,
): TrainingSequence {
    val chat = messages.map { it.forTemplate() }
    val withPrompt = chatOptions(toolsJson, true)
    val closed = chatOptions(toolsJson, false)
    fun encode(text: String) = tokenizer.encode(text, addSpecialTokens = false)
    val ids = encode(tokenizer.applyChatTemplate(chat, closed, template))
    val mask = BooleanArray(ids.size)
    val spans = ArrayList<IntRange>()
    val prompts = ArrayList<IntArray>()
    val problems = ArrayList<String>()
    val endId = tokenizer.tokenToId(END_TEXT) ?: error("the tokenizer has no $END_TEXT")
    for ((k, m) in chat.withIndex()) {
        if (m.role != "assistant") continue
        val prompt = encode(tokenizer.applyChatTemplate(chat.subList(0, k), withPrompt, template))
        val through = encode(tokenizer.applyChatTemplate(chat.subList(0, k + 1), closed, template))
        prompts += prompt
        if (!startsWith(ids, prompt)) {
            problems += "message $k: the prompt it was generated from is not a prefix of the chat's ids" + firstDiff(ids, prompt, tokenizer)
            spans += IntRange.EMPTY
            continue
        }
        if (!startsWith(ids, through)) {
            problems += "message $k: the chat through it is not a prefix of the chat's ids" + firstDiff(ids, through, tokenizer)
            spans += IntRange.EMPTY
            continue
        }
        // The turn ends at its last end token; what follows it ("\n") is the template's.
        var end = through.size
        while (end > prompt.size && through[end - 1] != endId) end--
        if (end == prompt.size) {
            problems += "message $k: no $END_TEXT after the prompt"
            spans += IntRange.EMPTY
            continue
        }
        for (i in prompt.size until end) mask[i] = true
        spans += prompt.size until end
    }
    return TrainingSequence(ids, mask, spans, prompts, problems)
}

const val END_TEXT = "<|im_end|>"

fun startsWith(a: IntArray, prefix: IntArray): Boolean {
    if (prefix.size > a.size) return false
    for (i in prefix.indices) if (a[i] != prefix[i]) return false
    return true
}

private fun firstDiff(a: IntArray, b: IntArray, tokenizer: HfTokenizer): String {
    var i = 0
    while (i < a.size && i < b.size && a[i] == b[i]) i++
    val ctx = { x: IntArray -> tokenizer.decode(x.copyOfRange(maxOf(0, i - 3), minOf(x.size, i + 4))) }
    return " (at token $i: ${toJson(ctx(a))} vs ${toJson(ctx(b))})"
}
