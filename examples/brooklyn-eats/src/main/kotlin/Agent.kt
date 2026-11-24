package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.ChatToolCall
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.util.concurrent.Executors

// The agent loop: a chat with tlaloc-endpoint's OpenAI API (tools in the request), each tool call
// run by the tool server, its result appended as a tool message, until the model answers without
// calling a tool or a budget ends the episode. Every episode is one JSONL line: the messages as
// sent, each turn's sampling settings, seed, token counts and time, each call's arguments, schema
// check and result, the stop reason, and the verifier's verdict for offline tasks.
//
//   agent run    [--split dev] [--n 10] [--endpoint URL] [--model NAME] [--config twil-bf16] [--out FILE]
//                [--concurrency 1] [--temperature 0] [--top-p P] [--top-k K] [--max-calls 8]
//                [--max-tokens 12000] [--turn-tokens 4096] [--tool-mode snapshot]
//   agent parity FILE [--config twil-bf16]   trainer ids and assistant masks of recorded episodes
//   agent show   FILE [--n N]

/** The system message of every episode; [date] is the snapshot's "today". */
fun systemPrompt(date: LocalDate): String =
    "You are Brooklyn Eats, an assistant for restaurants, cafes and bars in Brooklyn, New York. Today is ${longDate(date)}. " +
        "Use the tools to look up facts: health inspections and letter grades (NYC DOHMH), 311 complaints, places and " +
        "opening hours (OpenStreetMap), Google reviews (up to 2021) and archived menus. State only what the tool results " +
        "say; never guess names, grades, scores, dates, prices or counts. If a place is outside Brooklyn or not in the " +
        "records, say so. Answer briefly, and give the address when several places share a name."

/** What an episode may spend and how each turn samples. */
data class AgentSettings(
    val maxCalls: Int = 8,
    /** Completion tokens over all turns of an episode. */
    val maxTokens: Int = 12_000,
    /** Completion tokens of one turn. */
    val turnTokens: Int = 4_096,
    /** Prompt plus completion; the served context. */
    val maxModelLen: Int = 16_384,
    val temperature: Double = 0.0,
    val topP: Double? = null,
    val topK: Int? = null,
    /** Base seed; each episode's seed is derived from it and the task id, each turn adds its index. */
    val seed: Long = 0,
) {
    fun json(): Map<String, Any?> = linkedMapOf(
        "max_calls" to maxCalls, "max_tokens" to maxTokens, "turn_tokens" to turnTokens, "max_model_len" to maxModelLen,
        "temperature" to temperature, "top_p" to topP, "top_k" to topK, "seed" to seed,
    )
}

/** An episode's seed: 24 bits (the backend's range) of SHA-256 of the base seed and the task id. */
fun episodeSeed(base: Long, taskId: String): Long =
    sha256Hex("$base:$taskId".toByteArray()).take(6).toLong(16)

/** One chat completion as the endpoint returned it. */
class Completion(
    val content: String,
    val reasoning: String?,
    val toolCalls: List<ChatToolCall>,
    val finish: String,
    val promptTokens: Int,
    val completionTokens: Int,
    val cachedTokens: Int?,
    val millis: Long,
)

class EndpointError(message: String) : Exception(message)

/** A client of an OpenAI-compatible chat endpoint (non-streaming requests). */
class ChatClient(val baseUrl: String, val model: String, private val apiKey: String? = System.getenv("TLALOC_API_KEY")) {
    private val http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build()

    fun complete(body: Map<String, Any?>): Completion {
        val t0 = System.nanoTime()
        val req = HttpRequest.newBuilder(URI.create("${baseUrl.trimEnd('/')}/chat/completions"))
            .timeout(Duration.ofMinutes(30))
            .header("Content-Type", "application/json")
            .also { b -> apiKey?.let { b.header("Authorization", "Bearer $it") } }
            .POST(HttpRequest.BodyPublishers.ofString(toJson(body)))
            .build()
        val r = http.send(req, HttpResponse.BodyHandlers.ofString())
        val ms = (System.nanoTime() - t0) / 1_000_000
        if (r.statusCode() != 200) throw EndpointError("HTTP ${r.statusCode()}: ${r.body().take(500)}")
        val o = parseJson(r.body()) as JsonObject
        val choice = (o["choices"] as JsonArray).elements[0] as JsonObject
        val msg = choice["message"] as JsonObject
        val usage = o["usage"] as? JsonObject
        val calls = (msg["tool_calls"] as? JsonArray)?.elements.orEmpty().map { c ->
            val f = (c as JsonObject)["function"] as JsonObject
            ChatToolCall((f["name"] as JsonString).value, (f["arguments"] as? JsonString)?.value ?: toJson(f["arguments"]?.toKotlin()))
        }
        return Completion(
            (msg["content"] as? JsonString)?.value ?: "",
            (msg["reasoning_content"] as? JsonString)?.value,
            calls,
            (choice["finish_reason"] as? JsonString)?.value ?: "",
            usage?.optInt("prompt_tokens") ?: -1,
            usage?.optInt("completion_tokens") ?: -1,
            (usage?.get("prompt_tokens_details") as? JsonObject)?.optInt("cached_tokens"),
            ms,
        )
    }
}

/** One message in the OpenAI request format. */
fun messageJson(m: ConvMessage, callIds: List<String> = emptyList(), toolCallId: String? = null): Map<String, Any?> {
    val o = linkedMapOf<String, Any?>("role" to m.role, "content" to m.content)
    if (m.reasoning != null) o["reasoning_content"] = m.reasoning
    if (m.toolCalls.isNotEmpty()) o["tool_calls"] = m.toolCalls.mapIndexed { i, c ->
        linkedMapOf("id" to callIds[i], "type" to "function", "function" to linkedMapOf("name" to c.name, "arguments" to c.argumentsJson))
    }
    if (toolCallId != null) o["tool_call_id"] = toolCallId
    return o
}

/** A recorded message back as a [ConvMessage]. */
fun parseConvMessage(o: JsonObject): ConvMessage = ConvMessage(
    o.str("role"),
    (o["content"] as? JsonString)?.value ?: "",
    (o["reasoning_content"] as? JsonString)?.value,
    (o["tool_calls"] as? JsonArray)?.elements.orEmpty().map { c ->
        val f = (c as JsonObject)["function"] as JsonObject
        ChatToolCall(f.str("name"), (f["arguments"] as? JsonString)?.value ?: "{}")
    },
)

/**
 * Runs one episode of [task]. [tools] runs a call and returns its JSON result; [promptLength]
 * gives the trainer's token count of the chat so far with the generation prompt (also how the
 * turn's max_tokens is kept inside the context).
 */
fun runEpisode(
    task: Task,
    client: ChatClient,
    toolDefs: List<Map<String, Any?>>,
    system: String,
    settings: AgentSettings,
    tools: (String, String) -> String,
    promptLength: (List<ConvMessage>) -> Int,
): Map<String, Any?> {
    val t0 = System.nanoTime()
    val started = Instant.now().toString()
    val seed = episodeSeed(settings.seed, task.id)
    val conv = arrayListOf(ConvMessage("system", system), ConvMessage("user", task.question))
    val wire = arrayListOf(messageJson(conv[0]), messageJson(conv[1]))
    val turns = ArrayList<Map<String, Any?>>()
    val calls = ArrayList<Map<String, Any?>>()
    val callList = ArrayList<ToolCall>()
    var stop = ""
    var answer = ""
    var completionTokens = 0
    var promptTokens = 0
    while (stop.isEmpty()) {
        val turn = turns.size
        val rendered = promptLength(conv)
        val maxTokens = minOf(settings.turnTokens, settings.maxTokens - completionTokens, settings.maxModelLen - rendered - 1)
        if (maxTokens < 64) {
            stop = if (settings.maxTokens - completionTokens < 64) "token_budget" else "context"
            break
        }
        val turnSeed = (seed + turn) and 0xFFFFFF
        val body = linkedMapOf<String, Any?>(
            "model" to client.model, "messages" to wire, "tools" to toolDefs, "max_tokens" to maxTokens,
            "temperature" to settings.temperature, "seed" to turnSeed,
        )
        settings.topP?.let { body["top_p"] = it }
        settings.topK?.let { body["top_k"] = it }
        val c = try {
            client.complete(body)
        } catch (e: Exception) {
            // The endpoint refuses a prompt plus max_tokens past the context it can hold (a
            // speculative step writes positions beyond its tokens, which the rendered length above
            // does not count). That is the context stop, not a failure to retry: the same request
            // is refused again.
            stop = if (isContextRefusal(e.message)) "context" else "error: ${e.message?.take(300)}"
            break
        }
        completionTokens += c.completionTokens
        promptTokens += c.promptTokens
        val ids = c.toolCalls.indices.map { "call_${turn}_$it" }
        val msg = ConvMessage("assistant", c.content, c.reasoning, c.toolCalls)
        conv += msg
        wire += messageJson(msg, ids)
        turns += linkedMapOf(
            "seed" to turnSeed, "max_tokens" to maxTokens, "finish_reason" to c.finish, "prompt_tokens" to c.promptTokens,
            "rendered_prompt_tokens" to rendered, "completion_tokens" to c.completionTokens, "tool_calls" to c.toolCalls.size,
            "timing" to linkedMapOf("ms" to c.millis, "cached_tokens" to c.cachedTokens),
        )
        if (c.toolCalls.isEmpty()) {
            answer = c.content
            stop = if (c.finish == "length") "length" else "answer"
            break
        }
        for ((i, call) in c.toolCalls.withIndex()) {
            if (callList.size >= settings.maxCalls) { stop = "call_budget"; break }
            val args = runCatching { parseJson(call.argumentsJson) }.getOrNull()
            val def = TOOL_BY_NAME[call.name]
            val schemaErrors = when {
                def == null -> listOf("unknown tool")
                args !is JsonObject -> listOf("arguments are not a JSON object")
                else -> validate(def.schema, args)
            }
            val ct = System.nanoTime()
            val result = try {
                tools(call.name, call.argumentsJson)
            } catch (e: Exception) {
                toJson(mapOf("error" to (e.message ?: e::class.simpleName)))
            }
            val ms = (System.nanoTime() - ct) / 1_000_000
            val error = runCatching { (parseJson(result) as? JsonObject)?.optString("error") }.getOrNull()
            callList += ToolCall(call.name, call.argumentsJson)
            calls += linkedMapOf(
                "turn" to turn, "id" to ids[i], "name" to call.name, "arguments" to call.argumentsJson,
                "schema_errors" to schemaErrors.ifEmpty { null }, "error" to error, "result_chars" to result.length,
                "timing" to linkedMapOf("ms" to ms),
            )
            val tm = ConvMessage("tool", result)
            conv += tm
            wire += messageJson(tm, toolCallId = ids[i])
        }
        if (stop.isEmpty() && completionTokens >= settings.maxTokens) stop = "token_budget"
    }
    return linkedMapOf(
        "id" to task.id, "split" to task.split, "category" to task.category, "kind" to task.kind, "question" to task.question,
        "offline" to task.offline, "model" to client.model, "settings" to settings.json(), "seed" to seed,
        "messages" to wire, "turns" to turns, "calls" to calls, "stop" to stop, "answer" to answer,
        "totals" to linkedMapOf(
            "turns" to turns.size, "calls" to calls.size, "valid_calls" to calls.count { it["schema_errors"] == null },
            "calls_without_error" to calls.count { it["schema_errors"] == null && it["error"] == null },
            "prompt_tokens" to promptTokens, "completion_tokens" to completionTokens,
        ),
        "timing" to linkedMapOf("started" to started, "ms" to (System.nanoTime() - t0) / 1_000_000),
    )
}

/** An episode without its times and server-state fields: what two runs of the same model and seed should agree on. */
fun normalizedEpisode(v: JsonValue): JsonValue = when (v) {
    is JsonObject -> JsonObject(v.fields.filterKeys { it != "timing" }.mapValues { normalizedEpisode(it.value) })
    is JsonArray -> JsonArray(v.elements.map { normalizedEpisode(it) })
    else -> v
}

/** SHA-256 of a trajectories file's normalized lines (canonical JSON, timing removed). */
fun trajectoriesSha256(f: Path): String =
    sha256Hex(Files.readAllLines(f).filter { it.isNotBlank() }.joinToString("") { canonicalJson(normalizedEpisode(parseJson(it))) + "\n" }.toByteArray())

/** [n] offline tasks of [tasks]: categories in turn, each category's tasks ordered by SHA-256 of the seed and id. */
fun sampleTasks(tasks: List<Task>, n: Int, seed: Long): List<Task> {
    val byCat = tasks.filter { it.offline }.groupBy { it.category }.toSortedMap()
        .mapValues { (_, ts) -> ArrayDeque(ts.sortedBy { sha256Hex("$seed:${it.id}".toByteArray()) }) }
    val out = ArrayList<Task>()
    while (out.size < n && byCat.values.any { it.isNotEmpty() }) {
        for (q in byCat.values) if (out.size < n) q.removeFirstOrNull()?.let { out += it }
    }
    return out
}

/** A serving config (serving/<name>.env): KEY=value lines. */
fun servingConfig(home: Path, name: String): Map<String, String> {
    val f = home.resolve("serving").resolve("$name.env")
    require(Files.exists(f)) { "no serving config $f" }
    return Files.readAllLines(f).map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") && '=' in it }
        .associate { it.substringBefore('=') to it.substringAfter('=') }
}

const val AGENT_DIR = "agent"

private fun opt(args: List<String>, name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }

fun agent(args: List<String>): Int {
    val home = exampleHome()
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    val db = data.resolve("brooklyn.db").toAbsolutePath().normalize()
    val sub = args.firstOrNull()?.takeIf { !it.startsWith("--") } ?: "run"
    val configName = opt(args, "--config") ?: "twil-bf16"
    val config = servingConfig(home, configName)
    val template = ChatTemplate.valueOf(config["TEMPLATE"] ?: "TWIL")
    val tokenizerDir = opt(args, "--tokenizer")?.let { Paths.get(it) } ?: modelDir(config["MODEL_KEY"] ?: "twil_8b")
    return when (sub) {
        "run" -> agentRun(args, home, data, db, config, template, tokenizerDir)
        "parity" -> agentParity(Paths.get(args.getOrNull(1) ?: return usageAgent()), template, tokenizerDir)
        "show" -> {
            val f = Paths.get(args.getOrNull(1) ?: return usageAgent())
            val n = opt(args, "--n")?.toInt() ?: Int.MAX_VALUE
            for (line in Files.readAllLines(f).filter { it.isNotBlank() }.take(n)) printEpisode(parseJson(line) as JsonObject)
            0
        }
        else -> usageAgent()
    }
}

private fun usageAgent(): Int {
    System.err.println(
        """usage: agent run [--split dev] [--n 10] [--endpoint URL] [--model NAME] [--config twil-bf16] [--out FILE]
                 [--concurrency 1] [--temperature 0] [--top-p P] [--top-k K] [--max-calls 8] [--max-tokens 12000]
                 [--turn-tokens 4096] [--tool-mode snapshot]
       agent parity FILE [--config twil-bf16]
       agent show FILE [--n N]""",
    )
    return 2
}

fun agentRun(args: List<String>, home: Path, data: Path, db: Path, config: Map<String, String>, template: ChatTemplate, tokenizerDir: Path): Int {
    if (!Files.exists(db)) { System.err.println("agent: $db not found; run the ingest stage first"); return 1 }
    val split = Split.parse(opt(args, "--split") ?: "dev")
    val taskFile = taskFile(data, split)
    if (!Files.exists(taskFile)) { System.err.println("agent: $taskFile not found; run the tasks stage first"); return 1 }
    val n = opt(args, "--n")?.toInt() ?: 10
    val endpoint = opt(args, "--endpoint") ?: "http://127.0.0.1:${System.getenv("BROOKLYN_ENDPOINT_PORT") ?: "8120"}/v1"
    val model = opt(args, "--model") ?: config["SERVED_NAME"] ?: "twil-8b"
    val concurrency = opt(args, "--concurrency")?.toInt() ?: 1
    val mode = ToolMode.parse(opt(args, "--tool-mode") ?: "snapshot")
    val settings = AgentSettings(
        maxCalls = opt(args, "--max-calls")?.toInt() ?: 8,
        maxTokens = opt(args, "--max-tokens")?.toInt() ?: 12_000,
        turnTokens = opt(args, "--turn-tokens")?.toInt() ?: 4_096,
        maxModelLen = opt(args, "--max-model-len")?.toInt() ?: 16_384,
        temperature = opt(args, "--temperature")?.toDouble() ?: 0.0,
        topP = opt(args, "--top-p")?.toDouble(),
        topK = opt(args, "--top-k")?.toInt(),
        seed = opt(args, "--seed")?.toLong() ?: manifestSeed("agent"),
    )
    val tasks = sampleTasks(readTasks(taskFile), n, manifestSeed("agent_sample"))
    val out = opt(args, "--out")?.let { Paths.get(it) } ?: data.resolve(AGENT_DIR).resolve("${split.label}-$n.jsonl")
    Files.createDirectories(out.toAbsolutePath().parent)
    val tokenizer = HfTokenizer.load(tokenizerDir)
    val client = ChatClient(endpoint, model)
    val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
    val system = systemPrompt(cutoff)
    println("agent: ${tasks.size} ${split.label} tasks, $endpoint model $model, template $template, tools $mode, concurrency $concurrency")
    println("agent: settings ${toJson(settings.json())}")
    val t0 = System.nanoTime()
    val scored = runAgentTasks(tasks, client, settings, concurrency, db, mode, tokenizer, template, system, cutoff)
    val seconds = (System.nanoTime() - t0) / 1e9
    Files.writeString(out, scored.joinToString("") { toJson(it) + "\n" })
    val totals = scored.map { it["totals"] as Map<*, *> }
    val calls = totals.sumOf { it["calls"] as Int }
    val good = totals.sumOf { it["calls_without_error"] as Int }
    val tokens = totals.sumOf { it["completion_tokens"] as Int }
    val stops = scored.groupingBy { (it["stop"] as String).substringBefore(':') }.eachCount()
    val passed = scored.count { ((it["verdict"] as Map<*, *>)["pass"]) == true }
    println("agent: wrote ${scored.size} episodes to $out")
    println("agent: stops $stops; tool calls $calls, $good schema-valid without a tool error; episodes with a successful call " +
        "${scored.count { (it["totals"] as Map<*, *>)["calls_without_error"] as Int > 0 }}; verifier passed $passed of ${scored.size}")
    println("agent: %d completion tokens in %.1f s (%.1f tokens/s over the run); normalized sha256 %s".format(tokens, seconds, tokens / seconds, trajectoriesSha256(out)))
    return if (scored.any { (it["stop"] as String).startsWith("error") }) 1 else 0
}

/**
 * Runs [tasks] as agent episodes against [client], [concurrency] at a time, tools in [mode], and
 * scores each answer with the offline verifier. [done] holds episodes already recorded (by task id;
 * they are not run again); [onEpisode] sees each new episode as it ends, unscored. Returns the
 * scored episodes in the order of [tasks].
 */
fun runAgentTasks(
    tasks: List<Task>,
    client: ChatClient,
    settings: AgentSettings,
    concurrency: Int,
    db: Path,
    mode: ToolMode,
    tokenizer: HfTokenizer,
    template: ChatTemplate,
    system: String,
    cutoff: LocalDate,
    done: Map<String, Map<String, Any?>> = emptyMap(),
    onEpisode: (Map<String, Any?>) -> Unit = {},
): List<Map<String, Any?>> {
    val toolDefs = TOOL_DEFS.map { it.openAi() }
    val promptOptions = chatOptions(toolDefs.map { toJson(it) }, true)
    val pool = Executors.newFixedThreadPool(concurrency)
    val servers = ThreadLocal.withInitial { ToolServer(db, mode) }
    val opened = java.util.concurrent.ConcurrentLinkedQueue<ToolServer>()
    val lock = Any()
    val futures = tasks.map { t ->
        done[t.id]?.let { java.util.concurrent.CompletableFuture.completedFuture(it) } ?: pool.submit<Map<String, Any?>> {
            val s = servers.get().also { if (it !in opened) opened += it }
            val ep = runEpisode(t, client, toolDefs, system, settings, { name, a -> s.call(name, a) }) { conv ->
                tokenizer.encode(tokenizer.applyChatTemplate(conv.map { it.forTemplate() }, promptOptions, template), addSpecialTokens = false).size
            }
            val totals = ep["totals"] as Map<*, *>
            synchronized(lock) {
                onEpisode(ep)
                println("  ${t.id} ${t.category}/${t.kind}: ${ep["stop"]}, ${totals["calls"]} calls, " +
                    "${totals["completion_tokens"]} tokens, ${(ep["timing"] as Map<*, *>)["ms"]} ms")
            }
            ep
        }
    }
    val episodes = try {
        futures.map { it.get() }
    } catch (e: Exception) {
        // Queued episodes are dropped so the process can end; the ones recorded so far stay with onEpisode.
        pool.shutdownNow()
        throw e
    } finally {
        pool.shutdown()
        opened.forEach { it.close() }
    }
    return ToolServer(db, ToolMode.SNAPSHOT).use { s ->
        val v = Verifier(s, cutoff)
        episodes.mapIndexed { i, ep ->
            val t = tasks[i]
            val calls = (ep["calls"] as List<*>).map { c -> c as Map<*, *>; ToolCall(c["name"] as String, c["arguments"] as String) }
            val verdict = v.verify(t, ep["answer"] as String, calls)
            LinkedHashMap(ep).also { it["verdict"] = verdict.json() }
        }
    }
}

/**
 * For each recorded episode: the trainer's ids (one render of the whole chat) against a fresh render
 * of each request (what the endpoint renders for those messages), the assistant mask's spans, and
 * the endpoint's reported prompt length of each turn against the trainer's.
 */
fun agentParity(f: Path, template: ChatTemplate, tokenizerDir: Path): Int {
    val tokenizer = HfTokenizer.load(tokenizerDir)
    val toolsJson = TOOL_DEFS.map { toJson(it.openAi()) }
    var bad = 0
    var turns = 0
    var lengthsEqual = 0
    var masked = 0
    var total = 0
    for (line in Files.readAllLines(f).filter { it.isNotBlank() }) {
        val ep = parseJson(line) as JsonObject
        val msgs = (ep["messages"] as JsonArray).elements.map { parseConvMessage(it as JsonObject) }
        val seq = renderForTraining(tokenizer, template, msgs, toolsJson)
        val epTurns = (ep["turns"] as JsonArray).elements.map { it as JsonObject }
        turns += epTurns.size
        masked += seq.maskedTokens
        total += seq.ids.size
        epTurns.forEachIndexed { i, t ->
            if (i < seq.prompts.size && t.optInt("prompt_tokens") == seq.prompts[i].size) lengthsEqual++
        }
        if (!seq.ok) {
            bad++
            println("${ep.str("id")}: " + seq.problems.joinToString("; "))
        }
    }
    println("parity: $turns turns; trainer ids hold every request's fresh render as a prefix in all but $bad episodes; " +
        "endpoint prompt_tokens equal to the trainer's prompt length in $lengthsEqual of $turns turns; $masked of $total tokens masked")
    return if (bad == 0) 0 else 1
}

private fun printEpisode(ep: JsonObject) {
    println("== ${ep.str("id")} ${ep.str("category")}/${ep.str("kind")}: ${ep.str("stop")}")
    for (m in (ep["messages"] as JsonArray).elements.map { it as JsonObject }) {
        val role = m.str("role")
        if (role == "system") continue
        (m["reasoning_content"] as? JsonString)?.value?.let { println("[$role reasoning] ${shorten(it, 400)}") }
        val content = (m["content"] as? JsonString)?.value ?: ""
        if (content.isNotEmpty()) println("[$role] ${shorten(content, if (role == "tool") 300 else 2000)}")
        (m["tool_calls"] as? JsonArray)?.elements?.forEach { c ->
            val fn = (c as JsonObject)["function"] as JsonObject
            println("[$role call] ${fn.str("name")} ${(fn["arguments"] as? JsonString)?.value}")
        }
    }
    (ep["verdict"] as? JsonObject)?.let { println("verdict: ${toJson(it.toKotlin())}") }
    println()
}

/** The serving config the agent stage runs (run.sh's BROOKLYN_SERVING_CONFIG). */
const val AGENT_CONFIG = "twil-bf16"
/** The dev questions the agent stage runs. */
const val AGENT_STAGE_N = 10

fun servingDir(home: Path): Path = System.getenv("BROOKLYN_SERVING")?.let { Paths.get(it) } ?: home.resolve("build").resolve("serving")

/** Git tree ids of [paths] (relative to the checkout's top) at HEAD, and whether the working tree changes any of them. */
fun gitTrees(home: Path, paths: List<String>): Map<String, Any?>? = runCatching {
    fun git(vararg a: String): String {
        val p = ProcessBuilder(listOf("git", "-C", home.toString()) + a).redirectErrorStream(true).start()
        val out = p.inputStream.readAllBytes().toString(Charsets.UTF_8).trim()
        check(p.waitFor() == 0) { out }
        return out
    }
    val top = git("rev-parse", "--show-toplevel")
    val m = linkedMapOf<String, Any?>()
    for (path in paths) m[path] = git("rev-parse", "HEAD:$path")
    m["clean"] = git("-C", top, "status", "--porcelain", "--", *paths.toTypedArray()).isEmpty()
    m
}.getOrNull()

/** The Tlaloc sources the serving artifact and the Triton backend are built from (the checkout this example is in). */
val SERVING_SOURCES = listOf(
    "core/src", "ir/src", "stablehlo/src", "autograd/src", "nn/src", "tokenizer/src", "maestro/src",
    "triton/backend", "triton/kernels", "triton/build_backend.sh",
)

/** A serving config's identity: the file's SHA-256 and the settings that make the served model. */
fun servingConfigIdentity(home: Path, name: String): Map<String, Any?>? {
    val envFile = home.resolve("serving").resolve("$name.env")
    if (!Files.exists(envFile)) return null
    val c = servingConfig(home, name)
    return linkedMapOf(
        "file" to "serving/$name.env", "sha256" to fileSha256(envFile), "model" to c["MODEL_KEY"], "draft" to c["DRAFT_KEY"],
        "export" to c["EXPORT_PROPS"], "triton_model" to c["TRITON_PROPS"], "triton_image" to c["TRITON_IMAGE"],
        "deterministic" to c["DETERMINISTIC"], "endpoint_max_sequences" to c["ENDPOINT_MAX_SEQUENCES"],
    ).filterValues { it != null }
}

/** SHA-256 over the serving artifact's file list and each file's SHA-256 (cached by size and time in [MODEL_HASHES]). */
fun servingArtifactIdentity(home: Path, data: Path, name: String): Map<String, Any?>? {
    val dir = servingDir(home).resolve(name).resolve("artifact")
    if (!Files.isDirectory(dir)) return null
    val hashes = HashCache(data.resolve(MODEL_HASHES))
    val files = Files.walk(dir).use { s -> s.filter { Files.isRegularFile(it) }.toList() }.sortedBy { dir.relativize(it).toString() }
    val listing = files.joinToString("") { "${dir.relativize(it)}\t${hashes.sha256(it)}\n" }
    hashes.save()
    return linkedMapOf("sha256" to sha256Hex(listing.toByteArray()), "files" to files.size.toLong(), "bytes" to files.sumOf { Files.size(it) })
}

/**
 * Manifest entries of the agent stage. Inputs: the serving config (export settings, Triton image
 * digest) and the Tlaloc sources the artifact and the backend are built from; the agent's system
 * prompt and settings. Derived: the serving artifact's files and the dev trajectories (normalized:
 * times removed).
 */
fun agentEntries(home: Path, data: Path): List<Entry> {
    val out = ArrayList<Entry>()
    out += Entry("agent", "inputs", "serving_config", false) { servingConfigIdentity(home, AGENT_CONFIG) }
    out += Entry("agent", "inputs", "serving_sources", false) { gitTrees(home, SERVING_SOURCES) }
    out += Entry("agent", "inputs", "agent_settings", false) {
        val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
        linkedMapOf(
            "system_prompt_sha256" to sha256Hex(systemPrompt(cutoff).toByteArray()), "settings" to toJson(AgentSettings().json().filterKeys { it != "seed" }),
            "questions" to AGENT_STAGE_N.toLong(), "split" to "dev",
        )
    }
    out += Entry("agent", "derived", "serving_artifact", false) { servingArtifactIdentity(home, data, AGENT_CONFIG) }
    // A smoke run of the agent loop: what the model does is sampled by a server that is not bitwise
    // reproducible (greedy decoding in batches of any order), so only the questions are compared.
    out += Entry(
        "agent", "derived", "trajectories_dev", true,
        setOf("sha256", "answers", "calls", "calls_without_error", "completion_tokens", "verifier_passed"),
    ) {
        val f = data.resolve(AGENT_DIR).resolve("dev-$AGENT_STAGE_N.jsonl")
        if (!Files.exists(f)) null else {
            val eps = Files.readAllLines(f).filter { it.isNotBlank() }.map { parseJson(it) as JsonObject }
            fun total(k: String) = eps.sumOf { (it["totals"] as JsonObject).optInt(k) ?: 0 }.toLong()
            linkedMapOf(
                "sha256" to trajectoriesSha256(f), "episodes" to eps.size.toLong(),
                "tasks" to sha256Hex(eps.joinToString("") { it.str("id") + "\n" }.toByteArray()),
                "answers" to eps.count { it.str("stop") == "answer" }.toLong(), "calls" to total("calls"),
                "calls_without_error" to total("calls_without_error"), "completion_tokens" to total("completion_tokens"),
                "verifier_passed" to eps.count { ((it["verdict"] as? JsonObject)?.get("pass") as? io.tlaloc.core.io.JsonBool)?.value == true }.toLong(),
            )
        }
    }
    return out
}

/** The endpoint's HTTP 400 for a prompt plus max_tokens past the model's context (OpenAiEndpoint). */
private val CONTEXT_REFUSAL = Regex("""HTTP 400: .*\d+ prompt tokens and max_tokens \d+ pass the model's \d+""")

fun isContextRefusal(message: String?): Boolean = message != null && CONTEXT_REFUSAL.containsMatchIn(message)
