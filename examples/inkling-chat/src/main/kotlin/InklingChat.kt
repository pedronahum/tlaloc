import io.tlaloc.runtime.pjrt.serving.InklingModel
import io.tlaloc.tokenizer.ChatMessage
import io.tlaloc.tokenizer.ChatOptions
import io.tlaloc.tokenizer.ChatOutput
import io.tlaloc.tokenizer.ChatOutputEvent
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.InklingOutputParser
import io.tlaloc.tokenizer.load
import java.nio.file.Path

/**
 * A chat with Inkling-2-Mini on the GPU, in this JVM.
 *
 *   inkling-chat <snapshot dir> [--effort low|medium|high|...] [--context N] [--max-tokens N] [--prompt TEXT]...
 *
 * Each turn renders the whole conversation with Inkling's chat template. When the state already
 * holds a prefix of it (the earlier turns as they were generated), only the rest is appended;
 * otherwise the state is reset and the conversation runs again. With --prompt (repeatable) the chat
 * is those turns, printed, and the program exits; without it, turns are read from standard input.
 */
fun main(args: Array<String>) {
    fun opt(name: String) = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }
    val dir = Path.of(args.firstOrNull { !it.startsWith("--") } ?: error("usage: inkling-chat <snapshot dir> [--effort E] [--context N] [--max-tokens N] [--prompt TEXT]..."))
    val effort = opt("--effort") ?: "low"
    val context = opt("--context")?.toInt() ?: 8192
    val maxTokens = opt("--max-tokens")?.toInt() ?: 1024
    val prompts = args.indices.filter { args[it] == "--prompt" }.mapNotNull { args.getOrNull(it + 1) }
    val scripted = prompts.isNotEmpty()
    var turn = 0

    val tok = HfTokenizer.load(dir)
    val end = tok.tokenToId("<|content_model_end_sampling|>") ?: error("no <|content_model_end_sampling|> in the tokenizer")
    val t0 = System.nanoTime()
    InklingModel.open(dir, contextLength = context) { System.err.println(it) }.use { m ->
        System.err.println("inkling-chat: weights on the device in %.0f s; context %d tokens".format((System.nanoTime() - t0) / 1e9, context))
        val messages = ArrayList<ChatMessage>()
        val inState = ArrayList<Int>() // the tokens the state holds
        while (true) {
            val user = if (scripted) prompts.getOrNull(turn++) ?: return else run {
                print("\n> ")
                System.out.flush()
                readlnOrNull()?.trim()?.takeIf { it.isNotEmpty() } ?: return
            }
            messages += ChatMessage("user", user)
            val text = tok.applyChatTemplate(messages, ChatOptions(reasoningEffort = effort), ChatTemplate.INKLING)
            val ids = tok.encode(text, addSpecialTokens = false)
            val reuse = inState.size <= ids.size && (0 until inState.size).all { inState[it] == ids[it] }
            if (!reuse) {
                m.reset()
                inState.clear()
            }
            val prompt = ids.copyOfRange(inState.size, ids.size)
            inState += prompt.toList()
            val parser = InklingOutputParser()
            val events = ArrayList<ChatOutputEvent>()
            var thinking = false
            val stream = tok.decodeStream()
            val t1 = System.nanoTime()
            var first = 0L
            val out = m.generate(prompt, maxTokens = minOf(maxTokens, context - m.position - 1), stop = setOf(end)) { id ->
                if (first == 0L) first = System.nanoTime()
                for (e in parser.feed(stream.step(id) ?: "")) {
                    events += e
                    when (e) {
                        is ChatOutputEvent.Reasoning -> { if (!thinking) { print("[thinking] "); thinking = true }; print(e.text) }
                        is ChatOutputEvent.Content -> { if (thinking) { print("\n[answer] "); thinking = false }; print(e.text) }
                        is ChatOutputEvent.Call -> print("\n[tool call] ${e.call.name} ${e.call.argumentsJson}")
                    }
                    System.out.flush()
                }
            }
            events += parser.finish()
            // The state holds every generated token but the last.
            inState += out.dropLast(1)
            val s = (System.nanoTime() - first) / 1e9
            System.err.println("\n[%d prompt tokens (%d reused), first token after %.2f s; %d tokens at %.1f tokens/s]".format(
                prompt.size, ids.size - prompt.size, (first - t1) / 1e9, out.size, (out.size - 1) / s,
            ))
            val turn = ChatOutput.of(events)
            messages += ChatMessage("assistant", turn.content, turn.reasoning.ifEmpty { null }, turn.toolCalls)
        }
    }
}
