package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.nio.file.Files
import java.nio.file.Paths
import java.time.LocalDate
import java.time.ZoneId

// The chat stage: a conversation with a served student (by default the SFT one) and the tools,
// live by default (DOHMH and 311 from the Socrata API, the restaurants' own sites, SearXNG;
// reviews and OSM tags from the snapshot). The server must be running (run.sh chat starts it).
//
//   brooklyn-eats chat [--config twil-sft-bf16] [--tool-mode live|snapshot|record|replay]
//                      [--endpoint URL] [--model NAME] [--max-calls 8] [--temperature 0]
//                      [--prompt QUESTION]...      one or more questions in turn, then exit
//
// Without --prompt, questions are read from standard input, one per line; an empty line ends.
// Earlier questions and answers stay in the conversation.

private fun chatOpts(args: List<String>, name: String): List<String> =
    args.withIndex().filter { it.value == name }.mapNotNull { args.getOrNull(it.index + 1) }

fun chatCmd(args: List<String>): Int {
    fun opt(name: String) = chatOpts(args, name).lastOrNull()
    val home = exampleHome()
    val data = opt("--data")?.let { Paths.get(it) } ?: dataHome()
    val db = data.resolve("brooklyn.db").toAbsolutePath().normalize()
    if (!Files.exists(db)) { System.err.println("chat: $db not found; run the ingest stage first"); return 1 }
    val configName = opt("--config") ?: "twil-sft-bf16"
    val config = servingConfig(home, configName)
    val template = ChatTemplate.valueOf(config["TEMPLATE"] ?: "TWIL")
    val endpoint = opt("--endpoint") ?: "http://127.0.0.1:${System.getenv("BROOKLYN_ENDPOINT_PORT") ?: "8120"}/v1"
    val model = opt("--model") ?: config["SERVED_NAME"] ?: "twil-8b"
    val mode = ToolMode.parse(opt("--tool-mode") ?: "live")
    val maxCalls = opt("--max-calls")?.toInt() ?: 8
    val temperature = opt("--temperature")?.toDouble() ?: 0.0
    val maxModelLen = 16_384
    val prompts = chatOpts(args, "--prompt")
    val tokenizer = HfTokenizer.load(modelDir(config["MODEL_KEY"] ?: "twil_8b"))
    val client = ChatClient(endpoint, model)
    val toolDefs = TOOL_DEFS.map { it.openAi() }
    val promptOptions = chatOptions(toolDefs.map { toJson(it) }, true)
    // Live tools answer for today; snapshot tools for the cutoff the database was built at.
    val date = if (mode.online) LocalDate.now(ZoneId.of(NYC_TZ))
    else LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
    val conv = arrayListOf(ConvMessage("system", systemPrompt(date)))
    val wire = arrayListOf(messageJson(conv[0]))
    println("chat: $model at $endpoint, tools $mode, date $date (an empty line ends)")
    ToolServer(db, mode).use { tools ->
        var asked = 0
        while (true) {
            val question = if (prompts.isNotEmpty()) prompts.getOrNull(asked) else {
                print("\nyou> ")
                System.out.flush()
                readlnOrNull()?.trim()?.takeIf { it.isNotEmpty() }
            } ?: return 0
            if (prompts.isNotEmpty()) println("\nyou> $question")
            asked++
            conv += ConvMessage("user", question)
            wire += messageJson(conv.last())
            var calls = 0
            var turn = 0
            while (true) {
                val rendered = tokenizer.encode(tokenizer.applyChatTemplate(conv.map { it.forTemplate() }, promptOptions, template), addSpecialTokens = false).size
                val maxTokens = minOf(4096, maxModelLen - rendered - 1)
                if (maxTokens < 64) { println("chat: the conversation fills the context ($rendered tokens); start a new one"); return 0 }
                val c = try {
                    client.complete(linkedMapOf("model" to model, "messages" to wire, "tools" to toolDefs, "max_tokens" to maxTokens, "temperature" to temperature))
                } catch (e: Exception) {
                    System.err.println("chat: ${e.message}")
                    return 1
                }
                val ids = c.toolCalls.indices.map { "call_${asked}_${turn}_$it" }
                val msg = ConvMessage("assistant", c.content, c.reasoning, c.toolCalls)
                conv += msg
                wire += messageJson(msg, ids)
                turn++
                if (c.toolCalls.isEmpty() || calls >= maxCalls) {
                    println("\nbrooklyn-eats> ${c.content.trim()}")
                    println("  [%d tool calls, %d completion tokens in the last turn, %.1f s]".format(calls, c.completionTokens, c.millis / 1000.0))
                    break
                }
                for ((i, call) in c.toolCalls.withIndex()) {
                    calls++
                    val result = try { tools.call(call.name, call.argumentsJson) } catch (e: Exception) { toJson(mapOf("error" to (e.message ?: e::class.simpleName))) }
                    val error = runCatching { (parseJson(result) as? JsonObject)?.optString("error") }.getOrNull()
                    println("  · ${call.name} ${call.argumentsJson}" + (error?.let { " -> error: $it" } ?: " -> ${result.length} chars"))
                    conv += ConvMessage("tool", result)
                    wire += messageJson(conv.last(), toolCallId = ids[i])
                }
            }
        }
    }
    return 0
}
