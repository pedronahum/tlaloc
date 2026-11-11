package io.tlaloc.endpoint

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.Socket
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The endpoint over a scripted backend: a byte-level tokenizer built here, Ripple's chat template,
 * and a [FakeBackend] that answers each prompt with a scripted reply and keeps every sequence's
 * tokens, so a test can check that what the backend holds is the conversation as rendered.
 */
class OpenAiEndpointTest {
    private val tokenizer = byteTokenizer()
    private val imEnd = tokenizer.tokenToId("<|im_end|>")!!
    private val backend = FakeBackend()
    private val http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).build()
    private val endpoints = ArrayList<OpenAiEndpoint>()

    @AfterTest fun stop() = endpoints.forEach { it.close() }

    private fun endpoint(
        maxSequences: Int = 4,
        template: ChatTemplate = ChatTemplate.RIPPLE,
        customize: (EndpointConfig) -> EndpointConfig = { it },
    ): OpenAiEndpoint {
        val config = customize(EndpointConfig(
            model = "fake", host = "127.0.0.1", port = 0, maxSequences = maxSequences, maxModelLen = 4096,
            stateFile = Files.createTempFile("endpoint", ".json"), endTokens = listOf(imEnd),
        ))
        return OpenAiEndpoint(config, tokenizer, backend, template).also { it.start(); endpoints += it }
    }

    private fun post(e: OpenAiEndpoint, path: String, body: String): Pair<Int, String> {
        val r = http.send(
            HttpRequest.newBuilder(URI.create("http://127.0.0.1:${e.port}$path")).header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build(),
            HttpResponse.BodyHandlers.ofString(),
        )
        return r.statusCode() to r.body()
    }

    private fun ok(e: OpenAiEndpoint, path: String, body: String): JsonObject {
        val (code, text) = post(e, path, body)
        assertEquals(200, code, text)
        return parseJson(text) as JsonObject
    }

    private fun JsonObject.message(): JsonObject = (arr("choices").elements[0] as JsonObject).obj("message")

    private fun JsonObject.cached(): Int = obj("usage").obj("prompt_tokens_details").int("cached_tokens")

    private fun chat(messages: String, extra: String = "") = """{"model":"fake","messages":$messages$extra}"""

    @Test fun aCompletionReturnsTheScriptedText() {
        val e = endpoint()
        backend.reply = { " Paris." }
        val r = ok(e, "/v1/completions", """{"prompt":"The capital of France is","max_tokens":20}""")
        val choice = r.arr("choices").elements[0] as JsonObject
        assertEquals(" Paris.", choice.str("text"))
        assertEquals("stop", choice.str("finish_reason"))
        assertEquals(tokenizer.encode("The capital of France is", addSpecialTokens = false).size, r.obj("usage").int("prompt_tokens"))
    }

    @Test fun maxTokensEndsAGeneration() {
        val e = endpoint()
        backend.reply = { "abcdefghijklmnop" }
        val choice = ok(e, "/v1/completions", """{"prompt":"x","max_tokens":5}""").arr("choices").elements[0] as JsonObject
        assertEquals("abcde", choice.str("text"))
        assertEquals("length", choice.str("finish_reason"))
    }

    @Test fun rippleReasoningAndToolCallsAreSplit() {
        val e = endpoint()
        backend.reply = { "<think>\nThe user wants weather.\n</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n</tool_call>" }
        val tools = ""","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]"""
        val r = ok(e, "/v1/chat/completions", chat("""[{"role":"user","content":"Weather in Paris?"}]""", tools))
        val m = r.message()
        assertEquals(JsonNull, m["content"])
        assertEquals("The user wants weather.", m.str("reasoning_content"))
        val call = m.arr("tool_calls").elements.single() as JsonObject
        assertEquals("get_weather", call.obj("function").str("name"))
        assertEquals("""{"city": "Paris", "days": 2}""", call.obj("function").str("arguments"))
        assertEquals("tool_calls", (r.arr("choices").elements[0] as JsonObject).str("finish_reason"))
        // The prompt the backend received: the template with the tool rendered by json.dumps.
        assertTrue("""{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}""" in backend.lastPromptText(tokenizer))
    }

    @Test fun twilReasoningAndToolCallsAreSplit() {
        val e = endpoint(template = ChatTemplate.TWIL)
        backend.reply = {
            "The user wants weather.\n</think>\n\nChecking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n" +
                "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
        }
        val tools = ""","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object",""" +
            """"properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]"""
        val r = ok(e, "/v1/chat/completions", chat("""[{"role":"user","content":"Weather in Paris?"}]""", tools))
        val m = r.message()
        assertEquals("Checking.", m.str("content"))
        assertEquals("The user wants weather.", m.str("reasoning_content"))
        val call = m.arr("tool_calls").elements.single() as JsonObject
        assertEquals("get_weather", call.obj("function").str("name"))
        assertEquals("""{"city": "Paris", "days": 2}""", call.obj("function").str("arguments"))
        assertEquals("tool_calls", (r.arr("choices").elements[0] as JsonObject).str("finish_reason"))
        val prompt = backend.lastPromptText(tokenizer)
        assertTrue("""{"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}""" in prompt)
        assertTrue(prompt.endsWith("<|im_start|>assistant\n<think>\n"))
    }

    @Test fun twilThinkingOffAndLowEffortReachTheTemplate() {
        val e = endpoint(template = ChatTemplate.TWIL)
        backend.reply = { "Paris." }
        val off = ok(e, "/v1/chat/completions", chat("""[{"role":"user","content":"Capital?"}]""", ""","chat_template_kwargs":{"enable_thinking":false}"""))
        assertEquals("Paris.", off.message().str("content"))
        assertTrue(backend.lastPromptText(tokenizer).endsWith("<|im_start|>assistant\n<think></think>"))
        ok(e, "/v1/chat/completions", chat("""[{"role":"user","content":"Capital of Spain?"}]""", ""","reasoning_effort":"low""""))
        assertTrue("Capital of Spain?\n\n{reasoning effort: low}<|im_end|>" in backend.lastPromptText(tokenizer))
    }

    @Test fun aStreamedChatCarriesTheSameParts() {
        val e = endpoint()
        backend.reply = { "<think>\nshort thought\n</think>\nThe answer is 42." }
        backend.stepSize = 2
        val (code, text) = post(e, "/v1/chat/completions", chat("""[{"role":"user","content":"Q?"}]""", ""","stream":true,"stream_options":{"include_usage":true}"""))
        assertEquals(200, code)
        val events = text.lines().filter { it.startsWith("data: ") }.map { it.removePrefix("data: ") }
        assertEquals("[DONE]", events.last())
        val chunks = events.dropLast(1).map { parseJson(it) as JsonObject }
        val deltas = chunks.flatMap { it.arr("choices").elements }.map { (it as JsonObject).obj("delta") }
        assertEquals("short thought", deltas.joinToString("") { (it["reasoning_content"] as? JsonString)?.value.orEmpty() })
        assertEquals("The answer is 42.", deltas.joinToString("") { (it["content"] as? JsonString)?.value.orEmpty() })
        assertEquals("stop", (chunks.flatMap { it.arr("choices").elements }.last() as JsonObject).str("finish_reason"))
        assertTrue(chunks.last().containsKey("usage"))
    }

    @Test fun aLaterTurnSendsOnlyItsNewTokens() {
        val e = endpoint()
        backend.reply = { p -> if ("Second" in p) "Two." else "<think>\nfirst\n</think>\nOne." }
        val first = """[{"role":"user","content":"First question"}]"""
        val r1 = ok(e, "/v1/chat/completions", chat(first))
        assertEquals(0, r1.cached())
        val m = r1.message()
        val second = """[{"role":"user","content":"First question"},""" +
            """{"role":"assistant","content":${jsonQuote(m.str("content"))},"reasoning_content":${jsonQuote(m.str("reasoning_content"))}},""" +
            """{"role":"user","content":"Second question"}]"""
        val r2 = ok(e, "/v1/chat/completions", chat(second))
        assertTrue(r2.cached() > 0, "the second turn reuses the first's KV")
        assertEquals("Two.", r2.message().str("content"))
        // What the backend holds before it generates the second reply is the conversation as rendered.
        val rendered = tokenizer.applyChatTemplate(
            listOf(
                io.tlaloc.tokenizer.ChatMessage("user", "First question"),
                io.tlaloc.tokenizer.ChatMessage("assistant", "One.", "first"),
                io.tlaloc.tokenizer.ChatMessage("user", "Second question"),
            ),
            template = ChatTemplate.RIPPLE,
        )
        assertEquals(rendered, backend.lastPromptText(tokenizer))
        assertEquals(1, backend.starts.get())
    }

    @Test fun aStopStringCutsTheTextAndCancels() {
        val e = endpoint()
        backend.reply = { "1, 2, 3, 4, 5, 6, 7, 8, 9" }
        backend.stepSize = 1
        val choice = ok(e, "/v1/completions", """{"prompt":"Count:","max_tokens":40,"stop":[", 5"]}""").arr("choices").elements[0] as JsonObject
        assertEquals("1, 2, 3, 4", choice.str("text"))
        assertEquals("stop", choice.str("finish_reason"))
        assertEquals(1, backend.cancelled.get())
        assertTrue(backend.ended.isNotEmpty(), "a cancelled sequence is ended: what its KV holds is not known")
    }

    @Test fun aNewSequenceRefusedForPagesEndsIdleOnes() {
        val e = endpoint(maxSequences = 4)
        backend.reply = { "ok" }
        backend.capacity = 120 // tokens across sequences
        val long = "x".repeat(50)
        ok(e, "/v1/completions", """{"prompt":"a$long","max_tokens":4}""")
        ok(e, "/v1/completions", """{"prompt":"b$long","max_tokens":4}""")
        // Two idle sequences hold about 110 tokens: the third needs one of them ended.
        val r = ok(e, "/v1/completions", """{"prompt":"c$long","max_tokens":4}""")
        assertEquals("ok", (r.arr("choices").elements[0] as JsonObject).str("text"))
        assertTrue(backend.refusals.get() >= 1)
        val stats = parseJson(http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:${e.port}/stats")).build(), HttpResponse.BodyHandlers.ofString()).body()) as JsonObject
        assertTrue((stats["evictions"] as JsonNumber).value >= 1)
    }

    @Test fun aGenerationOutOfPagesPartWayGoesOnInANewSequence() {
        val e = endpoint()
        val answer = "The answer is forty two, as computed above."
        // The reply goes on from whatever part of it the prompt already ends with.
        backend.reply = { p -> answer.substring((answer.length downTo 0).first { p.endsWith(answer.substring(0, it)) }) }
        ok(e, "/v1/completions", """{"prompt":"An idle sequence","max_tokens":64}""")
        backend.failOnceAfter = 12
        backend.stepSize = 4
        val (code, text) = post(e, "/v1/completions", """{"prompt":"Question?","max_tokens":64,"stream":true}""")
        assertEquals(200, code)
        val chunks = text.lines().filter { it.startsWith("data: {") }.map { parseJson(it.removePrefix("data: ")) as JsonObject }
        assertEquals(answer, chunks.flatMap { it.arr("choices").elements }.joinToString("") { (it as JsonObject).str("text") })
        assertEquals(-1, backend.failOnceAfter)
        // The idle sequence ended for it, and the new sequence was sent the prompt and the 12 tokens (bytes) generated so far.
        val stats = parseJson(http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:${e.port}/stats")).build(), HttpResponse.BodyHandlers.ofString()).body()) as JsonObject
        assertEquals(1.0, (stats["evictions"] as JsonNumber).value)
        assertTrue(tokenizer.decode(backend.prompts.last()).startsWith("Question?The answer i") && tokenizer.decode(backend.prompts.last()).length < "Question?".length + answer.length)
    }

    @Test fun aClientThatGoesAwayCancelsItsGeneration() {
        val e = endpoint()
        backend.reply = { "y".repeat(3000) }
        backend.stepSize = 1
        backend.stepMillis = 5
        Socket("127.0.0.1", e.port).use { s ->
            val body = """{"prompt":"go","max_tokens":3000,"stream":true}"""
            s.getOutputStream().write("POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: ${body.length}\r\n\r\n$body".toByteArray())
            val reader = BufferedReader(InputStreamReader(s.getInputStream()))
            var data = 0
            while (data < 5) if (reader.readLine()?.startsWith("data: ") == true) data++
        }
        val deadline = System.currentTimeMillis() + 10_000
        while (backend.cancelled.get() == 0 && System.currentTimeMillis() < deadline) Thread.sleep(20)
        assertEquals(1, backend.cancelled.get())
        assertTrue(backend.generatedTotal.get() < 3000)
    }

    @Test fun requestsTheEndpointCannotServeAreRefusedByName() {
        val e = endpoint()
        val user = """[{"role":"user","content":"hi"}]"""
        for ((extra, word) in listOf(
            ""","n":2""" to "n must be 1",
            ""","temperature":0.7""" to "greedily",
            ""","tool_choice":"required"""" to "constrained decoding",
            ""","chat_template_kwargs":{"foo":1}""" to "chat_template_kwargs.foo",
            ""","logprobs":true""" to "logprobs",
        )) {
            val (code, text) = post(e, "/v1/chat/completions", chat(user, extra))
            assertEquals(400, code, text)
            assertTrue(word in text, "$extra: $text")
        }
        val (code, text) = post(e, "/v1/chat/completions", chat("""[{"role":"robot","content":"hi"}]"""))
        assertEquals(400, code)
        assertTrue("robot" in text, text)
    }

    private fun send(e: OpenAiEndpoint, path: String, body: String, headers: Map<String, String>): HttpResponse<String> {
        val b = HttpRequest.newBuilder(URI.create("http://127.0.0.1:${e.port}$path"))
        for ((k, v) in headers) b.header(k, v)
        return http.send(b.POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString())
    }

    private val short = """{"prompt":"hi","max_tokens":3}"""

    @Test fun anApiKeyIsRequiredWhenOneIsSet() {
        backend.reply = { "ok" }
        val e = endpoint { it.copy(apiKey = "sesame") }
        val json = mapOf("Content-Type" to "application/json")
        assertEquals(401, send(e, "/v1/completions", short, json).statusCode())
        assertEquals(401, send(e, "/v1/completions", short, json + ("Authorization" to "Bearer wrong")).statusCode())
        assertEquals(200, send(e, "/v1/completions", short, json + ("Authorization" to "Bearer sesame")).statusCode())
        val get = { path: String -> http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:${e.port}$path")).GET().build(), HttpResponse.BodyHandlers.ofString()).statusCode() }
        assertEquals(200, get("/health"))
        assertEquals(401, get("/stats"))
    }

    @Test fun aPostThatIsNotJsonIsRefused() {
        val e = endpoint()
        assertEquals(415, send(e, "/v1/completions", short, mapOf("Content-Type" to "text/plain")).statusCode())
        assertEquals(415, send(e, "/v1/completions", short, emptyMap()).statusCode())
    }

    @Test fun anOversizedOrDeeplyNestedBodyIsRefused() {
        val e = endpoint { it.copy(maxBodyBytes = 1024) }
        assertEquals(413, post(e, "/v1/completions", "x".repeat(2048)).first)
        assertEquals(400, post(e, "/v1/completions", "[".repeat(1000)).first)
    }

    @Test fun aBackendErrorDoesNotNameOtherSequences() {
        backend.capacity = 0
        val e = endpoint()
        val (code, text) = post(e, "/v1/completions", short)
        assertEquals(503, code, text)
        assertTrue("sequence" !in text, text)
    }

    @Test fun aClientThatStopsReadingFreesItsSequence() {
        backend.reply = { if ("stall" in it) "x".repeat(300_000) else "ok" }
        backend.stepSize = 1
        val e = endpoint(maxSequences = 1) { it.copy(writeTimeoutMillis = 300, maxModelLen = 1_000_000) }
        val body = """{"prompt":"stall","max_tokens":300000,"stream":true}"""
        Socket("127.0.0.1", e.port).use { s ->
            s.getOutputStream().write("POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: ${body.length}\r\n\r\n$body".toByteArray())
            s.getOutputStream().flush()
            // Reads nothing: the socket's buffers fill and the endpoint's writes block.
            val deadline = System.nanoTime() + 60_000_000_000L
            while (backend.cancelled.get() == 0 && System.nanoTime() < deadline) Thread.sleep(20)
            assertEquals(1, backend.cancelled.get(), "the stalled generation was not cancelled")
            val (code, text) = post(e, "/v1/completions", short)
            assertEquals(200, code, text)
        }
    }

    @Test fun aReplyWithTokensPastItsEndTokenIsNotContinued() {
        backend.reply = { "ok" }
        backend.stepSize = 16
        backend.overrunAfterEnd = 2
        val e = endpoint()
        val first = chat("""[{"role":"user","content":"one"}]""")
        ok(e, "/v1/chat/completions", first)
        assertEquals(1, backend.starts.get())
        backend.overrunAfterEnd = 0
        val next = chat("""[{"role":"user","content":"one"},{"role":"assistant","content":"ok"},{"role":"user","content":"two"}]""")
        ok(e, "/v1/chat/completions", next)
        assertEquals(2, backend.starts.get(), "the second turn must start a new sequence")
        assertEquals(1, backend.ended.size)
    }

    @Test fun aFullQueueRefusesARequest() {
        backend.reply = { "x".repeat(200) }
        backend.stepSize = 1
        backend.stepMillis = 5
        val e = endpoint(maxSequences = 1) { it.copy(maxWaiting = 0) }
        val first = Thread.ofVirtual().start { post(e, "/v1/completions", """{"prompt":"a","max_tokens":200}""") }
        val deadline = System.nanoTime() + 30_000_000_000L
        while (backend.generatedTotal.get() == 0 && System.nanoTime() < deadline) Thread.sleep(5)
        val (code, text) = post(e, "/v1/completions", """{"prompt":"b","max_tokens":3}""")
        assertEquals(429, code, text)
        first.join()
    }

    /** A scripted sequence-mode backend. */
    private class FakeBackend : SequenceBackend {
        var reply: (String) -> String = { "" }
        var stepSize = 3
        var stepMillis = 0L
        // Tokens a step reports after the end token, as a chained step's acceptance can.
        @Volatile var overrunAfterEnd = 0
        var capacity = Int.MAX_VALUE
        // The next generation stops for want of pages once it has emitted this many tokens (-1: none does).
        @Volatile var failOnceAfter = -1
        val sequences = ConcurrentHashMap<Long, MutableList<Int>>()
        val prompts = CopyOnWriteArrayList<IntArray>()
        val ended = CopyOnWriteArrayList<Long>()
        val cancelled = java.util.concurrent.atomic.AtomicInteger()
        val refusals = java.util.concurrent.atomic.AtomicInteger()
        val starts = java.util.concurrent.atomic.AtomicInteger()
        val generatedTotal = java.util.concurrent.atomic.AtomicInteger()
        lateinit var tokenizer: HfTokenizer

        fun lastPromptText(t: HfTokenizer) = t.decode(prompts.last())

        override fun generate(sequenceId: Long, tokens: IntArray, start: Boolean, end: Boolean, parameters: Map<String, TritonParameter>): TokenStream {
            synchronized(this) {
                if (start) {
                    val held = sequences.values.sumOf { it.size }
                    if (held + tokens.size > capacity) {
                        refusals.incrementAndGet()
                        throw TritonException("model 'fake': KV page pool exhausted: sequence $sequenceId needs more pages")
                    }
                    starts.incrementAndGet()
                    sequences[sequenceId] = ArrayList()
                }
            }
            val seq = sequences[sequenceId] ?: throw TritonException("sequence $sequenceId has no KV state")
            seq += tokens.toList()
            prompts += seq.toIntArray()
            val max = (parameters["max_tokens"] as TritonParameter.Int64).value.toInt()
            val ends = (parameters["end_tokens"] as? TritonParameter.Text)?.value?.split(',')?.map { it.toInt() }.orEmpty()
            val t = tokenizerOf()
            val text = reply(t.decode(seq.toIntArray()))
            val out = (t.encode(text, addSpecialTokens = false).toList() + t.tokenToId("<|im_end|>")!!).toMutableList()
            var keep = minOf(out.size, max)
            for (i in 0 until keep) if (out[i] in ends) { keep = i + 1; break }
            val all = out.subList(0, keep) + List(overrunAfterEnd) { t.tokenToId("x")!! }
            return object : TokenStream {
                var i = 0
                var done = false
                // The KV holds every generated token but the last, as the real backend's.
                fun finish() {
                    if (!done) {
                        done = true
                        seq += all.dropLast(1)
                    }
                }

                override fun next(timeoutMillis: Long): IntArray? {
                    if (failOnceAfter in 0..i && i < all.size) {
                        failOnceAfter = -1
                        throw TritonException("model 'fake': KV page pool exhausted: sequence $sequenceId needs another page")
                    }
                    if (i >= all.size) {
                        finish()
                        return null
                    }
                    if (stepMillis > 0) Thread.sleep(stepMillis)
                    val step = all.subList(i, minOf(all.size, i + stepSize)).toIntArray()
                    i += step.size
                    generatedTotal.addAndGet(step.size)
                    if (i >= all.size) finish()
                    return step
                }

                override fun close() {
                    if (!done) {
                        done = true
                        cancelled.incrementAndGet()
                    }
                }
            }
        }

        private fun tokenizerOf() = tokenizer

        override fun end(sequenceId: Long) {
            ended += sequenceId
            sequences.remove(sequenceId)
        }
    }

    init {
        backend.tokenizer = tokenizer
    }

    private fun JsonObject.int(key: String) = (this[key] as JsonNumber).asInt(key)

    private fun JsonObject.containsKey(key: String) = fields.containsKey(key)

    private companion object {
        /** A byte-level BPE with one token per byte and Ripple's turn markers. */
        fun byteTokenizer(): HfTokenizer {
            val vocab = bytesToUnicode().entries.sortedBy { it.key }.joinToString(",") { (b, c) -> "${jsonQuote(c.toString())}:$b" }
            val json = """
                {"version":"1.0","added_tokens":[
                  {"id":256,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                  {"id":257,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
                 "normalizer":null,
                 "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
                 "post_processor":null,
                 "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
                 "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                  "vocab":{$vocab},"merges":[]}}
            """.trimIndent()
            return HfTokenizer.fromJson(json)
        }

        /** GPT-2's map from bytes to the characters that stand for them in a byte-level vocabulary. */
        fun bytesToUnicode(): Map<Int, Char> {
            val bs = ArrayList<Int>()
            bs += ('!'.code..'~'.code)
            bs += ('¡'.code..'¬'.code)
            bs += ('®'.code..'ÿ'.code)
            val cs = ArrayList(bs)
            var n = 0
            for (b in 0..255) if (b !in bs) {
                bs += b
                cs += 256 + n
                n++
            }
            return bs.indices.associate { bs[it] to cs[it].toChar() }
        }
    }
}
