package io.tlaloc.endpoint

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.ChatMessage
import io.tlaloc.tokenizer.ChatOptions
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.ChatToolCall
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.RippleEvent
import io.tlaloc.tokenizer.RippleOutputParser
import java.io.IOException
import java.io.OutputStream
import java.net.InetSocketAddress
import java.nio.file.Path
import java.security.SecureRandom
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/** How [OpenAiEndpoint] serves. */
data class EndpointConfig(
    /** The model's name in Triton. */
    val model: String,
    /** The OpenAI model id clients send; the Triton name by default. */
    val servedName: String = model,
    val host: String = "0.0.0.0",
    val port: Int = 8100,
    /** Live sequences kept for prefix reuse: the model's `max_candidate_sequences`. */
    val maxSequences: Int,
    val maxModelLen: Int = 131_072,
    val defaultMaxTokens: Int = 1024,
    /** Where the live sequences' ids are kept, for the next endpoint to end. */
    val stateFile: Path,
    /** Ids that end a generation; the checkpoint's end tokens. */
    val endTokens: List<Int>,
    /** Serve `temperature > 0` greedily instead of refusing it. */
    val allowSampling: Boolean = false,
    /** The artifact samples (`-Psampling=true`): temperature, top_p, top_k and seed reach the backend. */
    val sampling: Boolean = false,
    val verbose: Boolean = false,
)

/** A chat message as a request gives it. */
internal data class RequestMessage(
    val role: String,
    val content: String,
    val reasoning: String?,
    val toolCalls: List<ChatToolCall>,
    val toolCallId: String?,
) {
    fun forTemplate() = ChatMessage(role, content, reasoning, toolCalls)

    /** Equal as a turn of the conversation, reasoning aside. */
    fun sameTurn(o: RequestMessage) = role == o.role && content == o.content && toolCalls == o.toolCalls
}

private class BadRequest(message: String, val code: Int = 400) : Exception(message)

/**
 * An OpenAI-compatible endpoint in front of a decoupled speculative Tlaloc model served by
 * Triton, so that clients and benchmark harnesses written for vLLM run against it unchanged:
 *
 * - `POST /v1/chat/completions`: messages rendered by the checkpoint's chat template in Kotlin
 *   ([ChatTemplate]); tools, assistant tool calls and tool results for templates that render them;
 *   `chat_template_kwargs`. Ripple's output is split into `reasoning_content`, `content` and
 *   `tool_calls` ([RippleOutputParser]).
 * - `POST /v1/completions`: a prompt string or ids.
 * - `GET /v1/models`, `/health`, `/stats`, `/metrics`.
 *
 * Both POST routes take `max_tokens` (or `max_completion_tokens`), `stream`,
 * `stream_options.include_usage`, `stop` (strings), `stop_token_ids`, `ignore_eos`, and with a
 * sampling artifact `temperature`, `top_p`, `top_k` and `seed`.
 *
 * The backend keeps a sequence's KV between requests: a request whose tokens start with what a
 * live sequence holds sends only the rest, and a chat that extends a live conversation sends
 * the template's text after the reply as it was generated. A new sequence the backend refuses
 * for want of KV pages ends idle sequences and starts again. A client that goes away, or a stop
 * string, cancels its generation in the backend.
 */
class OpenAiEndpoint(
    private val config: EndpointConfig,
    private val tokenizer: HfTokenizer,
    private val triton: SequenceBackend,
    /** The chat template requests are rendered with: the checkpoint's by default. */
    private val template: ChatTemplate? = tokenizer.chatTemplate,
) : AutoCloseable {
    private val sequences = Sequences(triton, config.maxSequences, config.stateFile)
    private val random = SecureRandom()
    private val stats = Stats()
    private val server: HttpServer

    init {
        // Each streamed chunk is a small write; without TCP_NODELAY one can wait for the client's
        // delayed ACK (40 ms on Linux).
        System.setProperty("sun.net.httpserver.nodelay", "true")
        server = HttpServer.create(InetSocketAddress(config.host, config.port), 256)
        server.executor = Executors.newVirtualThreadPerTaskExecutor()
        server.createContext("/") { ex ->
            try {
                handle(ex)
            } catch (e: IOException) {
                // the client went away
            } catch (e: Exception) {
                runCatching { error(ex, 500, "${e::class.simpleName}: ${e.message}") }
            } finally {
                ex.close()
            }
        }
    }

    val port: Int get() = server.address.port

    fun start() = server.start()

    /** Stops serving and ends the live sequences. */
    override fun close() {
        server.stop(0)
        sequences.endAll()
    }

    private class Stats {
        val requests = AtomicLong()
        val reusedRequests = AtomicLong()
        val promptTokens = AtomicLong()
        val reusedTokens = AtomicLong()
        val completionTokens = AtomicLong()
        val steps = AtomicLong()
        val cancelled = AtomicLong()
        val evictions = AtomicLong()
    }

    private fun handle(ex: HttpExchange) {
        val path = ex.requestURI.path
        when {
            ex.requestMethod == "GET" && path == "/v1/models" -> reply(
                ex, 200,
                jsonObject(
                    "object" to "list",
                    "data" to listOf(
                        jsonObject("id" to config.servedName, "object" to "model", "owned_by" to "tlaloc", "max_model_len" to config.maxModelLen),
                    ),
                ),
            )
            ex.requestMethod == "GET" && (path == "/health" || path == "/v1/health") -> reply(ex, 200, jsonObject("status" to "ok"))
            ex.requestMethod == "GET" && path == "/stats" -> reply(ex, 200, statsJson())
            ex.requestMethod == "GET" && path == "/metrics" -> metrics(ex)
            ex.requestMethod == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions") -> {
                val body = try {
                    parseJson(ex.requestBody.readAllBytes().toString(Charsets.UTF_8).ifBlank { "{}" }) as? JsonObject
                        ?: throw BadRequest("the body is not a JSON object")
                } catch (e: JsonException) {
                    return error(ex, 400, "the body is not JSON: ${e.message}")
                }
                try {
                    if (path == "/v1/chat/completions") chat(ex, body) else completion(ex, body)
                } catch (e: BadRequest) {
                    error(ex, e.code, e.message ?: "bad request")
                }
            }
            else -> error(ex, 404, "no route ${ex.requestMethod} $path")
        }
    }

    private fun statsJson(): Map<String, Any?> {
        val c = stats.completionTokens.get()
        val s = stats.steps.get()
        return jsonObject(
            "requests" to stats.requests.get(), "reused_requests" to stats.reusedRequests.get(),
            "prompt_tokens" to stats.promptTokens.get(), "reused_tokens" to stats.reusedTokens.get(),
            "completion_tokens" to c, "steps" to s, "tokens_per_step" to if (s > 0) c.toDouble() / s else null,
            "cancelled" to stats.cancelled.get(), "evictions" to stats.evictions.get(), "live_sequences" to sequences.size,
        )
    }

    private fun metrics(ex: HttpExchange) {
        val m = statsJson()
        val text = buildString {
            for ((k, v) in m) if (v is Number) append("tlaloc_endpoint_").append(k).append(' ').append(v).append('\n')
        }
        val bytes = text.toByteArray()
        ex.responseHeaders.add("Content-Type", "text/plain; version=0.0.4")
        ex.sendResponseHeaders(200, bytes.size.toLong())
        ex.responseBody.write(bytes)
    }

    private fun reply(ex: HttpExchange, code: Int, obj: Any?) {
        val bytes = json(obj).toByteArray()
        ex.responseHeaders.add("Content-Type", "application/json")
        ex.sendResponseHeaders(code, bytes.size.toLong())
        ex.responseBody.write(bytes)
    }

    private fun error(ex: HttpExchange, code: Int, message: String) =
        reply(ex, code, jsonObject("error" to jsonObject("message" to message, "type" to "invalid_request_error", "code" to code)))

    // ---- requests ----

    private fun JsonObject.opt(key: String): JsonValue? = fields[key]?.takeIf { it != JsonNull }

    private fun JsonObject.intOpt(key: String): Int? = when (val v = opt(key)) {
        null -> null
        is JsonNumber -> v.asInt(key)
        else -> throw BadRequest("$key must be an integer")
    }

    private fun JsonObject.doubleOpt(key: String): Double? = when (val v = opt(key)) {
        null -> null
        is JsonNumber -> v.value
        else -> throw BadRequest("$key must be a number")
    }

    private fun JsonObject.boolOpt(key: String): Boolean? = when (val v = opt(key)) {
        null -> null
        is JsonBool -> v.value
        else -> throw BadRequest("$key must be a boolean")
    }

    private fun JsonObject.stringOpt(key: String): String? = when (val v = opt(key)) {
        null -> null
        is JsonString -> v.value
        else -> throw BadRequest("$key must be a string")
    }

    private fun text(v: JsonValue?, what: String): String = when (v) {
        null -> ""
        is JsonString -> v.value
        is JsonArray -> v.elements.joinToString("") { part ->
            val p = part as? JsonObject ?: throw BadRequest("$what: each content part is an object")
            when (p.stringOpt("type")) {
                "text" -> p.stringOpt("text") ?: throw BadRequest("$what: a text part needs text")
                else -> throw BadRequest("$what: only text content parts are served, not '${p.stringOpt("type")}'")
            }
        }
        else -> throw BadRequest("$what: content is a string or a list of text parts")
    }

    /** A tool call's arguments as the template writes them: the JSON object, by `json.dumps`. */
    private fun arguments(v: JsonValue?, what: String): String {
        val parsed = when (v) {
            is JsonString -> try {
                parseJson(v.value)
            } catch (e: JsonException) {
                throw BadRequest("$what: arguments are not JSON: ${e.message}")
            }
            is JsonObject -> v
            null -> JsonObject(emptyMap())
            else -> throw BadRequest("$what: arguments are a JSON object or its text")
        }
        if (parsed !is JsonObject) throw BadRequest("$what: arguments must be a JSON object")
        return pythonJson(parsed)
    }

    private fun messages(body: JsonObject): List<RequestMessage> {
        val list = body.opt("messages") as? JsonArray ?: throw BadRequest("messages is a list of messages")
        if (list.elements.isEmpty()) throw BadRequest("messages is empty")
        return list.elements.mapIndexed { i, e ->
            val m = e as? JsonObject ?: throw BadRequest("messages[$i] is not an object")
            val role = m.stringOpt("role") ?: throw BadRequest("messages[$i] has no role")
            val calls = (m.opt("tool_calls") as? JsonArray)?.elements.orEmpty().mapIndexed { j, c ->
                val co = c as? JsonObject ?: throw BadRequest("messages[$i].tool_calls[$j] is not an object")
                val f = co.opt("function") as? JsonObject ?: throw BadRequest("messages[$i].tool_calls[$j] has no function")
                ChatToolCall(
                    f.stringOpt("name") ?: throw BadRequest("messages[$i].tool_calls[$j] has no name"),
                    arguments(f.opt("arguments"), "messages[$i].tool_calls[$j]"),
                )
            }
            RequestMessage(
                if (role == "developer") "system" else role,
                text(m.opt("content"), "messages[$i]"),
                m.stringOpt("reasoning_content") ?: m.stringOpt("reasoning"),
                calls,
                m.stringOpt("tool_call_id"),
            )
        }
    }

    private fun options(body: JsonObject, tools: List<String>): ChatOptions {
        val kw = body.opt("chat_template_kwargs")?.let { it as? JsonObject ?: throw BadRequest("chat_template_kwargs is an object") }
            ?: JsonObject(emptyMap())
        val known = setOf("enable_thinking", "clear_thinking", "current_date", "reasoning_strength", "knowledge_cutoff")
        kw.fields.keys.firstOrNull { it !in known }?.let {
            throw BadRequest("chat_template_kwargs.$it is not a variable the Kotlin templates read (${known.joinToString()})")
        }
        return ChatOptions(
            addGenerationPrompt = true,
            enableThinking = kw.boolOpt("enable_thinking"),
            currentDate = kw.stringOpt("current_date"),
            reasoningStrength = kw.stringOpt("reasoning_strength") ?: body.stringOpt("reasoning_effort"),
            knowledgeCutoff = kw.stringOpt("knowledge_cutoff"),
            tools = tools,
            clearThinking = kw.boolOpt("clear_thinking") ?: false,
        )
    }

    private fun tools(body: JsonObject): List<String> {
        val tools = (body.opt("tools") as? JsonArray)?.elements.orEmpty()
        when (val choice = body.opt("tool_choice")) {
            null -> {}
            is JsonString -> when (choice.value) {
                "auto" -> {}
                "none" -> return emptyList()
                else -> throw BadRequest("tool_choice '${choice.value}' needs constrained decoding, which this endpoint does not do; send auto or none")
            }
            else -> throw BadRequest("tool_choice naming a function needs constrained decoding, which this endpoint does not do; send auto or none")
        }
        if (tools.isNotEmpty() && template != ChatTemplate.RIPPLE) {
            throw BadRequest("this model's chat template (${template ?: "none"}) does not render tools")
        }
        return tools.mapIndexed { i, t ->
            val o = t as? JsonObject ?: throw BadRequest("tools[$i] is not an object")
            if (o.stringOpt("type") != "function" || o.opt("function") !is JsonObject) throw BadRequest("tools[$i] is not a function tool")
            pythonJson(o)
        }
    }

    private fun render(messages: List<RequestMessage>, options: ChatOptions): String = try {
        tokenizer.applyChatTemplate(
            messages.map { it.forTemplate() }, options,
            template ?: throw BadRequest("this checkpoint's chat template is not one Tlaloc renders: serve /v1/completions"),
        )
    } catch (e: IllegalArgumentException) {
        throw BadRequest("chat template: ${e.message}")
    } catch (e: IllegalStateException) {
        throw BadRequest("chat template: ${e.message}")
    }

    private fun encode(text: String) = tokenizer.encode(text, addSpecialTokens = false)

    private fun chat(ex: HttpExchange, body: JsonObject) {
        val msgs = messages(body)
        val tools = tools(body)
        val options = options(body, tools)
        val ids = encode(render(msgs, options))
        // A chat whose messages extend a live conversation's (its reply included) sends the
        // template's text for the new messages after the reply as it was generated: a template
        // need not render an earlier turn as it rendered the last, so the tokens would not match.
        val extend = { s: Live ->
            val c = s.chat
            val n = c?.messages?.size ?: 0
            if (c == null || c.options != options || msgs.size <= n || s.pending == null ||
                !(0 until n - 1).all { msgs[it] == c.messages[it] } || !msgs[n - 1].sameTurn(c.messages[n - 1])
            ) {
                null
            } else {
                val reply = c.reply.trim()
                val text = render(msgs, options)
                val head = if (n > 1) render(msgs.subList(0, n - 1), options) else ""
                val k = if (reply.isEmpty()) -1 else text.indexOf(reply, if (text.startsWith(head)) head.length else 0)
                var after = if (k < 0) null else text.substring(k + reply.length)
                if (after != null && c.endText != null) after = if (after.startsWith(c.endText)) after.removePrefix(c.endText) else null
                after?.let { intArrayOf(s.pending!!) + encode(it) }
            }
        }
        // A new chat's leading system messages end at a checkpoint: with the backend's prefix
        // cache, other chats with the same system prompt start after it.
        val nSystem = msgs.indexOfFirst { it.role != "system" }.let { if (it < 0) msgs.size else it }
        val checkpoint = if (nSystem in 1 until msgs.size) {
            runCatching { encode(render(msgs.subList(0, nSystem), options.copy(addGenerationPrompt = false))) }.getOrNull()
                ?.takeIf { it.isNotEmpty() && it.size < ids.size && ids.copyOfRange(0, it.size).contentEquals(it) }?.size
        } else {
            null
        }
        generate(ex, body, ids, Chat(msgs, options), extend, checkpoint)
    }

    private fun completion(ex: HttpExchange, body: JsonObject) {
        val ids = when (val p = body.opt("prompt")) {
            is JsonString -> encode(p.value)
            is JsonArray -> if (p.elements.all { it is JsonNumber }) {
                p.asIntList("prompt").toIntArray()
            } else {
                throw BadRequest("prompt is one string or one list of token ids")
            }
            else -> throw BadRequest("prompt is one string or one list of token ids")
        }
        if (body.boolOpt("echo") == true) throw BadRequest("echo is not served")
        generate(ex, body, ids, null, null, null)
    }

    private class Chat(val messages: List<RequestMessage>, val options: ChatOptions)

    // ---- generation ----

    private fun generate(
        ex: HttpExchange,
        body: JsonObject,
        prompt: IntArray,
        chat: Chat?,
        extend: ((Live) -> IntArray?)?,
        checkpoint: Int?,
    ) {
        if (prompt.isEmpty()) throw BadRequest("the prompt has no tokens")
        if ((body.intOpt("n") ?: 1) != 1) throw BadRequest("n must be 1")
        if (body.opt("logprobs").let { it != null && it != JsonBool(false) } || body.opt("top_logprobs") != null) {
            throw BadRequest("logprobs are not served")
        }
        val temperature = body.doubleOpt("temperature") ?: 0.0
        if (temperature > 0 && !(config.allowSampling || config.sampling)) {
            throw BadRequest("this server decodes greedily; send temperature 0")
        }
        val asked = body.intOpt("max_completion_tokens") ?: body.intOpt("max_tokens")
        if (asked != null && asked < 1) throw BadRequest("max_tokens must be at least 1")
        if (prompt.size >= config.maxModelLen) {
            throw BadRequest("${prompt.size} prompt tokens reach the model's ${config.maxModelLen}")
        }
        val maxTokens = asked ?: minOf(config.defaultMaxTokens, config.maxModelLen - prompt.size)
        if (prompt.size + maxTokens > config.maxModelLen) {
            throw BadRequest("${prompt.size} prompt tokens and max_tokens $maxTokens pass the model's ${config.maxModelLen}")
        }
        val stopIds = if (body.boolOpt("ignore_eos") == true) emptyList() else config.endTokens
        val endIds = (stopIds + (body.opt("stop_token_ids") as? JsonArray)?.asIntList("stop_token_ids").orEmpty()).distinct()
        val stopStrings = when (val s = body.opt("stop")) {
            null -> emptyList()
            is JsonString -> listOf(s.value)
            is JsonArray -> s.elements.map { (it as? JsonString)?.value ?: throw BadRequest("stop is a string or a list of strings") }
            else -> throw BadRequest("stop is a string or a list of strings")
        }.filter { it.isNotEmpty() }
        if (stopStrings.size > 16) throw BadRequest("at most 16 stop strings")
        val stream = body.boolOpt("stream") == true
        val usageInStream = (body.opt("stream_options") as? JsonObject)?.boolOpt("include_usage") == true
        val params = LinkedHashMap<String, TritonParameter>()
        params["max_tokens"] = TritonParameter.Int64(maxTokens.toLong())
        if (endIds.isNotEmpty()) params["end_tokens"] = TritonParameter.Text(endIds.joinToString(","))
        if (config.sampling) {
            params["temperature"] = TritonParameter.Real(temperature)
            body.doubleOpt("top_p")?.let { params["top_p"] = TritonParameter.Real(it) }
            body.intOpt("top_k")?.takeIf { it > 0 }?.let { params["top_k"] = TritonParameter.Int64(it.toLong()) }
            params["seed"] = TritonParameter.Int64((body.intOpt("seed")?.toLong() ?: random.nextInt(1 shl 24).toLong()) and 0xFFFFFF)
        }

        val rid = (if (chat != null) "chatcmpl-" else "cmpl-") + UUID.randomUUID().toString().replace("-", "")
        val created = System.currentTimeMillis() / 1000
        val out = Output(ex, stream, chat != null, rid, created)
        val parser = if (chat != null && template == ChatTemplate.RIPPLE) RippleOutputParser() else null
        val stops = StopStrings(stopStrings)
        val t0 = System.nanoTime()

        var attempt = 0
        while (true) {
            val acquired = sequences.acquire(prompt, extend)
            val s = acquired.live
            val reused = if (acquired.start) 0 else s.kv.size
            val full = if (acquired.start) acquired.send else s.kv + acquired.send
            val p = LinkedHashMap(params)
            if (acquired.start && checkpoint != null) p["checkpoints"] = TritonParameter.Text(checkpoint.toString())
            val generated = ArrayList<Int>()
            val decoder = tokenizer.decodeStream(skipSpecialTokens = true)
            var steps = 0
            var finish = "length"
            var stoppedByString = false
            var g: TokenStream? = null
            try {
                g = triton.generate(s.id, acquired.send, start = acquired.start, parameters = p)
                while (true) {
                    val step = g.next() ?: break
                    steps++
                    if (steps == 1) out.begin()
                    for (t in step) {
                        if (generated.size >= maxTokens) break
                        generated += t
                        if (t in endIds) {
                            finish = "stop"
                            break
                        }
                        val piece = decoder.step(t) ?: continue
                        val visible = stops.feed(piece)
                        if (visible.isNotEmpty()) out.text(visible, parser)
                        if (stops.matched) {
                            finish = "stop"
                            stoppedByString = true
                            break
                        }
                    }
                    if (finish == "stop" || generated.size >= maxTokens) break
                }
            } catch (e: TritonException) {
                g?.close()
                sequences.drop(s)
                val pages = "KV page pool exhausted" in (e.message ?: "")
                if (acquired.start && pages && steps == 0 && attempt < config.maxSequences && sequences.evictIdle(s)) {
                    // A new sequence the pages could not hold: an idle one ended for it.
                    stats.evictions.incrementAndGet()
                    attempt++
                    continue
                }
                return out.fail(if (pages) 503 else 500, "backend: ${e.message?.take(400)}")
            } catch (e: IOException) {
                // The client went away: the generation is cancelled with the stream.
                g?.close()
                stats.cancelled.incrementAndGet()
                sequences.drop(s)
                return
            }
            val complete = finish == "stop" && !stoppedByString || generated.size >= maxTokens
            g.close() // a generation stopped early (a stop string) is cancelled
            if (!stoppedByString) {
                val rest = stops.finish()
                if (rest.isNotEmpty()) out.text(rest, parser)
            }
            val calls = out.finishText(parser)
            if (calls > 0) finish = "tool_calls"
            val usage = jsonObject(
                "prompt_tokens" to full.size, "completion_tokens" to generated.size, "total_tokens" to full.size + generated.size,
                "prompt_tokens_details" to jsonObject("cached_tokens" to reused),
            )
            stats.requests.incrementAndGet()
            if (reused > 0) stats.reusedRequests.incrementAndGet()
            stats.promptTokens.addAndGet(full.size.toLong())
            stats.reusedTokens.addAndGet(reused.toLong())
            stats.completionTokens.addAndGet(generated.size.toLong())
            stats.steps.addAndGet(steps.toLong())
            // The KV holds the prompt and every generated token but the last, unless the
            // generation was cancelled part way (then what it holds is not known: ended).
            if (complete && generated.isNotEmpty()) {
                val chatState = chat?.let {
                    ChatState(it.messages + out.message(), it.options, out.raw.toString(), if (finish != "length") tokenizer.decode(intArrayOf(generated.last())) else null)
                }
                sequences.release(s, full + generated.subList(0, generated.size - 1).toIntArray(), generated.last(), chatState)
            } else {
                sequences.drop(s)
            }
            if (config.verbose) {
                System.err.println(
                    "${rid.take(16)} seq ${s.id} prompt ${full.size} reused $reused sent ${acquired.send.size} out ${generated.size} " +
                        "steps $steps ${(System.nanoTime() - t0) / 1_000_000} ms finish $finish",
                )
            }
            try {
                out.end(finish, usage, usageInStream)
            } catch (_: IOException) {
            }
            return
        }
    }

    /** Holds back text that may begin a stop string; [matched] once one appears. */
    private class StopStrings(private val stops: List<String>) {
        private val held = StringBuilder()
        var matched = false
            private set
        private val longest = stops.maxOfOrNull { it.length } ?: 0

        fun feed(text: String): String {
            if (stops.isEmpty()) return text
            held.append(text)
            val s = held.toString()
            val at = stops.mapNotNull { stop -> s.indexOf(stop).takeIf { it >= 0 } }.minOrNull()
            if (at != null) {
                matched = true
                held.clear()
                return s.substring(0, at)
            }
            val keep = minOf(s.length, longest - 1)
            held.delete(0, s.length - keep)
            return s.substring(0, s.length - keep)
        }

        fun finish(): String = held.toString().also { held.clear() }
    }

    /** What a request writes: the streamed chunks, or the whole response at the end. */
    private inner class Output(
        private val ex: HttpExchange,
        private val stream: Boolean,
        private val chat: Boolean,
        private val rid: String,
        private val created: Long,
    ) {
        private var body: OutputStream? = null
        val raw = StringBuilder() // the generated text
        private val reasoning = StringBuilder()
        private val content = StringBuilder()
        private val calls = ArrayList<Pair<String, ChatToolCall>>()

        fun begin() {
            if (!stream || body != null) return
            ex.responseHeaders.add("Content-Type", "text/event-stream")
            ex.responseHeaders.add("Cache-Control", "no-cache")
            ex.sendResponseHeaders(200, 0)
            body = ex.responseBody
            if (chat) chunk(jsonObject("role" to "assistant", "content" to ""), null)
        }

        /** Generated text; true when something was written to the client. */
        fun text(t: String, parser: RippleOutputParser?): Boolean {
            raw.append(t)
            if (parser == null) {
                content.append(t)
                if (stream) {
                    if (chat) chunk(jsonObject("content" to t), null) else chunk(null, null, t)
                }
                return stream
            }
            return events(parser.feed(t))
        }

        private fun events(events: List<RippleEvent>): Boolean {
            for (e in events) {
                when (e) {
                    is RippleEvent.Reasoning -> {
                        reasoning.append(e.text)
                        if (stream) chunk(jsonObject("reasoning_content" to e.text), null)
                    }
                    is RippleEvent.Content -> {
                        content.append(e.text)
                        if (stream) chunk(jsonObject("content" to e.text), null)
                    }
                    is RippleEvent.Call -> {
                        val id = "call_" + UUID.randomUUID().toString().replace("-", "").take(24)
                        calls += id to e.call
                        if (stream) {
                            chunk(jsonObject("tool_calls" to listOf(toolCall(calls.size - 1, id, e.call))), null)
                        }
                    }
                }
            }
            return stream && events.isNotEmpty()
        }

        private fun toolCall(index: Int?, id: String, call: ChatToolCall) = jsonObject(
            "index" to index, "id" to id, "type" to "function",
            "function" to jsonObject("name" to call.name, "arguments" to call.argumentsJson),
        ).let { if (index == null) it.filterKeys { k -> k != "index" } else it }

        /** Ends the parser's text; the number of tool calls. */
        fun finishText(parser: RippleOutputParser?): Int {
            if (parser != null) events(parser.finish())
            return calls.size
        }

        /** The assistant message as the client receives it, for a later turn to match. */
        fun message() = RequestMessage("assistant", content.toString(), reasoning.toString().ifEmpty { null }, calls.map { it.second }, null)

        private fun chunk(delta: Map<String, Any?>?, finish: String?, text: String? = null, usage: Map<String, Any?>? = null) {
            val choices = if (usage != null) {
                emptyList()
            } else if (chat) {
                listOf(jsonObject("index" to 0, "delta" to (delta ?: emptyMap<String, Any?>()), "finish_reason" to finish))
            } else {
                listOf(jsonObject("index" to 0, "text" to (text ?: ""), "finish_reason" to finish))
            }
            val obj = jsonObject(
                "id" to rid, "object" to if (chat) "chat.completion.chunk" else "text_completion",
                "created" to created, "model" to config.servedName, "choices" to choices,
            )
            if (usage != null) obj["usage"] = usage
            send("data: ${json(obj)}\n\n")
        }

        private fun send(s: String) {
            val b = body ?: return
            b.write(s.toByteArray(Charsets.UTF_8))
            b.flush()
        }

        fun end(finish: String, usage: Map<String, Any?>, usageInStream: Boolean) {
            if (stream) {
                begin()
                if (chat) chunk(emptyMap(), finish) else chunk(null, finish)
                if (usageInStream) chunk(null, null, usage = usage)
                send("data: [DONE]\n\n")
                return
            }
            val obj = if (chat) {
                val message = jsonObject("role" to "assistant", "content" to content.toString().takeIf { it.isNotEmpty() || calls.isEmpty() })
                if (reasoning.isNotEmpty()) message["reasoning_content"] = reasoning.toString()
                if (calls.isNotEmpty()) message["tool_calls"] = calls.map { (id, c) -> toolCall(null, id, c) }
                jsonObject(
                    "id" to rid, "object" to "chat.completion", "created" to created, "model" to config.servedName,
                    "choices" to listOf(jsonObject("index" to 0, "message" to message, "finish_reason" to finish)), "usage" to usage,
                )
            } else {
                jsonObject(
                    "id" to rid, "object" to "text_completion", "created" to created, "model" to config.servedName,
                    "choices" to listOf(jsonObject("index" to 0, "text" to content.toString(), "finish_reason" to finish)), "usage" to usage,
                )
            }
            reply(ex, 200, obj)
        }

        fun fail(code: Int, message: String) {
            try {
                if (body != null) {
                    send("data: ${json(jsonObject("error" to jsonObject("message" to message, "code" to code)))}\n\n")
                    send("data: [DONE]\n\n")
                } else {
                    error(ex, code, message)
                }
            } catch (_: IOException) {
            }
        }
    }
}
