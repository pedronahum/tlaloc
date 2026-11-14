package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.endpoint.EndpointConfig
import io.tlaloc.endpoint.OpenAiEndpoint
import io.tlaloc.endpoint.SequenceBackend
import io.tlaloc.endpoint.TokenStream
import io.tlaloc.endpoint.TritonParameter
import io.tlaloc.endpoint.pythonJson
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.ChatToolCall
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A model stand-in behind the real OpenAiEndpoint: it replies with scripted token ids and records,
 * for every request, the tokens the model would condition on before generating (the sequence's
 * tokens so far plus those sent, as the backend's KV holds them; a sequence's last generated token
 * is run only when the next request sends it).
 */
class ScriptedBackend(private val replies: ArrayDeque<IntArray>) : SequenceBackend {
    private val history = HashMap<Long, MutableList<Int>>()
    val inputs = ArrayList<IntArray>()
    val starts = ArrayList<Boolean>()

    override fun generate(sequenceId: Long, tokens: IntArray, start: Boolean, end: Boolean, parameters: Map<String, TritonParameter>): TokenStream {
        val h = if (start) ArrayList<Int>().also { history[sequenceId] = it } else history.getValue(sequenceId)
        h += tokens.toList()
        inputs += h.toIntArray()
        starts += start
        val reply = replies.removeFirst()
        h += reply.dropLast(1)
        var i = 0
        return object : TokenStream {
            override fun next(timeoutMillis: Long): IntArray? = if (i < reply.size) intArrayOf(reply[i++]) else null
            override fun close() {}
        }
    }

    override fun end(sequenceId: Long) {
        history.remove(sequenceId)
    }
}

class AgentTest {
    private val date = LocalDate.parse("2026-11-13")
    private val toolDefs = TOOL_DEFS.map { it.openAi() }
    private val toolsJson = toolDefs.map { toJson(it) }

    private fun tokenizer(key: String): HfTokenizer? {
        val dir = runCatching { modelDir(key, hubCache(), readManifestJson(Path.of("manifest.json").toAbsolutePath())) }.getOrNull()
        if (dir == null || !Files.exists(dir.resolve("tokenizer.json"))) return null
        return HfTokenizer.load(dir)
    }

    private fun twil(): HfTokenizer = tokenizer("twil_8b").also { assumeTrue(it != null, "TwIL-8B's tokenizer is not in the Hugging Face cache (run.sh models)") }!!

    // A recorded conversation (synthetic): a question, two assistant turns with tool calls (one with
    // a nested object and a number, one with two calls and text before them), their tool responses,
    // and a final answer. Every assistant turn has reasoning.
    private val r1 = "The user wants Golden Lantern's grade. I should search for it first."
    private val r2 = "There is one match, id 50000001. Its record and its inspections give the grade."
    private val r3 = "The record says grade A from 2026-03-02, and the inspection list agrees."
    private val answer = "Golden Lantern Kitchen (12 Fake Street) has an A, from its inspection on March 2, 2026."
    private val search = ChatToolCall("search_restaurants", """{"query": "Golden Lantern", "near": {"lat": 40.68, "lon": -73.97}, "limit": 5}""")
    private val get = ChatToolCall("get_restaurant", """{"id": "50000001"}""")
    private val insp = ChatToolCall("get_inspections", """{"id": "50000001", "since": "2025-11-13"}""")
    private val toolResults = listOf(
        """{"results": [{"id": "50000001", "name": "Golden Lantern Kitchen", "address": "12 Fake Street", "grade": "A"}], "total": 1}""",
        """{"id": "50000001", "name": "Golden Lantern Kitchen", "grade": "A", "grade_date": "2026-03-02"}""",
        """{"inspections": [{"date": "2026-03-02", "score": 9, "grade": "A", "violations": []}]}""",
    )
    private val question = "What grade does Golden Lantern near Prospect Park have?"

    private fun conversation(): List<ConvMessage> = listOf(
        ConvMessage("system", systemPrompt(date)),
        ConvMessage("user", question),
        ConvMessage("assistant", "", r1, listOf(search)),
        ConvMessage("tool", toolResults[0]),
        ConvMessage("assistant", "I will read its record and its inspections.", r2, listOf(get, insp)),
        ConvMessage("tool", toolResults[1]),
        ConvMessage("tool", toolResults[2]),
        ConvMessage("assistant", answer, r3),
    )

    private fun task() = Task(
        "bk-test", "dev", "lookup", "grade", question, listOf("50000001"), emptyList(), null, emptyList(), "", true, 0, date.toString(),
    )

    private fun endpoint(tok: HfTokenizer, backend: SequenceBackend, dir: Path) = OpenAiEndpoint(
        EndpointConfig(
            model = "m", host = "127.0.0.1", port = 0, maxSequences = 4, maxModelLen = 16_384, stateFile = dir.resolve("state.json"),
            endTokens = listOf(tok.tokenToId(END_TEXT)!!), sampling = true,
        ),
        tok, backend,
    ).also { it.start() }

    /** The assistant turns' text as the model writes it after each generation prompt, written out by hand. */
    private fun expectedTwilTurns(): List<String> = listOf(
        "$r1\n</think>\n<tool_call>\n<function=search_restaurants>\n<parameter=query>\nGolden Lantern\n</parameter>\n" +
            "<parameter=near>\n{\"lat\": 40.68, \"lon\": -73.97}\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>\n<|im_end|>",
        "$r2\n</think>\nI will read its record and its inspections.\n<tool_call>\n<function=get_restaurant>\n<parameter=id>\n50000001\n</parameter>\n" +
            "</function>\n</tool_call>\n<tool_call>\n<function=get_inspections>\n<parameter=id>\n50000001\n</parameter>\n<parameter=since>\n" +
            "2025-11-13\n</parameter>\n</function>\n</tool_call>\n<|im_end|>",
        "$r3\n</think>\n$answer<|im_end|>",
    )

    private fun expectedRippleTurns(): List<String> = listOf(
        "<think>\n$r1\n</think>\n\n<tool_call>\n{\"name\": \"search_restaurants\", \"arguments\": {\"query\": \"Golden Lantern\", " +
            "\"near\": {\"lat\": 40.68, \"lon\": -73.97}, \"limit\": 5}}\n</tool_call><|im_end|>",
        "<think>\n$r2\n</think>\nI will read its record and its inspections.\n<tool_call>\n{\"name\": \"get_restaurant\", \"arguments\": " +
            "{\"id\": \"50000001\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_inspections\", \"arguments\": {\"id\": \"50000001\", " +
            "\"since\": \"2025-11-13\"}}\n</tool_call><|im_end|>",
        "<think>\n$r3\n</think>\n$answer<|im_end|>",
    )

    /**
     * The trainer's ids for the conversation against the endpoint: the agent loop runs the episode
     * through the real OpenAiEndpoint, whose model replies with the trainer's own tokens of each
     * assistant turn; what the model was given before each turn must be the trainer's ids up to that
     * turn, through the first request (rendered fresh) and the later ones (extending the live
     * sequence with the reply as generated). Then the mask: exactly the assistant turns.
     */
    private fun parity(tok: HfTokenizer, template: ChatTemplate, expectedTurns: List<String>) {
        val conv = conversation()
        val seq = renderForTraining(tok, template, conv, toolsJson)
        assertTrue(seq.ok, seq.problems.joinToString())
        val assistants = conv.indices.filter { conv[it].role == "assistant" }
        assertEquals(assistants.size, seq.spans.size)

        // The mask covers exactly the assistant turns' reasoning, tool calls and answers (and each turn's end token).
        val maskedText = seq.spans.map { tok.decode(seq.ids.copyOfRange(it.first, it.last + 1)) }
        assertEquals(expectedTurns, maskedText)
        assertEquals(seq.spans.sumOf { it.last - it.first + 1 }, seq.maskedTokens)
        val full = tok.decode(seq.ids)
        val unmasked = tok.decode(seq.ids.indices.filter { !seq.mask[it] }.map { seq.ids[it] }.toIntArray())
        var rest = full
        for (t in expectedTurns) rest = rest.replaceFirst(t, "")
        assertEquals(rest, unmasked)
        for (s in listOf(r1, r2, r3, answer, "I will read its record")) assertTrue(s !in unmasked, s)
        for (s in toolResults + listOf(question, systemPrompt(date))) assertTrue(s in unmasked && s !in maskedText.joinToString(""), s)

        // Each request rendered fresh by a new endpoint is the trainer's prefix.
        for ((k, a) in assistants.withIndex()) {
            val dir = Files.createTempDirectory("brooklyn-endpoint")
            val backend = ScriptedBackend(ArrayDeque(listOf(seq.ids.copyOfRange(seq.spans[k].first, seq.spans[k].last + 1))))
            val ep = endpoint(tok, backend, dir)
            try {
                val client = ChatClient("http://127.0.0.1:${ep.port}/v1", "m", apiKey = null)
                val c = client.complete(
                    linkedMapOf("model" to "m", "messages" to conv.subList(0, a).mapIndexed { i, m -> wireMessage(m, conv, i) }, "tools" to toolDefs, "max_tokens" to 2000),
                )
                assertEquals(listOf(true), backend.starts)
                assertEquals(seq.ids.copyOfRange(0, seq.spans[k].first).toList(), backend.inputs.single().toList(), "fresh request $k")
                assertEquals(seq.spans[k].first, c.promptTokens)
            } finally {
                ep.close()
            }
        }

        // The agent loop over one live conversation.
        val dir = Files.createTempDirectory("brooklyn-endpoint")
        val backend = ScriptedBackend(ArrayDeque(seq.spans.map { seq.ids.copyOfRange(it.first, it.last + 1) }))
        val ep = endpoint(tok, backend, dir)
        try {
            val client = ChatClient("http://127.0.0.1:${ep.port}/v1", "m", apiKey = null)
            val results = ArrayDeque(toolResults)
            val calls = ArrayList<Pair<String, String>>()
            val opts = chatOptions(toolsJson, true)
            val episode = runEpisode(task(), client, toolDefs, systemPrompt(date), AgentSettings(seed = 7), { n, a -> calls += n to a; results.removeFirst() }) { c ->
                tok.encode(tok.applyChatTemplate(c.map { it.forTemplate() }, opts, template), addSpecialTokens = false).size
            }
            assertEquals("answer", episode["stop"])
            assertEquals(answer, episode["answer"])
            assertEquals(listOf("search_restaurants", "get_restaurant", "get_inspections"), calls.map { it.first })
            // The recorded messages are the conversation.
            @Suppress("UNCHECKED_CAST")
            val recorded = (episode["messages"] as List<Map<String, Any?>>).map { parseConvMessage(parseJson(toJson(it)) as JsonObject) }
            assertEquals(conv.map { norm(it) }, recorded.map { norm(it) })
            // What the model was given before each turn is the trainer's ids up to that turn.
            assertEquals(listOf(true, false, false), backend.starts)
            for (k in seq.spans.indices) {
                assertEquals(seq.ids.copyOfRange(0, seq.spans[k].first).toList(), backend.inputs[k].toList(), "turn $k")
            }
            @Suppress("UNCHECKED_CAST")
            val turns = episode["turns"] as List<Map<String, Any?>>
            assertEquals(seq.spans.map { it.first }, turns.map { it["prompt_tokens"] })
            assertEquals(seq.spans.map { it.first }, turns.map { it["rendered_prompt_tokens"] })
            assertEquals(seq.spans.map { it.last - it.first + 1 }, turns.map { it["completion_tokens"] })

            // The recorded episode, re-read as the trainer reads it, renders to the same ids and mask.
            val again = renderForTraining(tok, template, recorded, toolsJson)
            assertEquals(seq.ids.toList(), again.ids.toList())
            assertEquals(seq.mask.toList(), again.mask.toList())
        } finally {
            ep.close()
        }
    }

    private fun wireMessage(m: ConvMessage, conv: List<ConvMessage>, i: Int): Map<String, Any?> =
        messageJson(m, m.toolCalls.indices.map { "call_${i}_$it" }, if (m.role == "tool") "call_x" else null)

    private fun norm(m: ConvMessage) = listOf(m.role, m.content, m.reasoning, m.toolCalls.map { it.name to pythonJson(parseJson(it.argumentsJson)) })

    @Test
    fun twilTrainerIdsEqualTheEndpointsAndTheMaskIsTheAssistantTurns() = parity(twil(), ChatTemplate.TWIL, expectedTwilTurns())

    @Test
    fun rippleTrainerIdsEqualTheEndpointsAndTheMaskIsTheAssistantTurns() {
        val tok = tokenizer("ripple_65b_nvfp4")
        assumeTrue(tok != null, "Ripple-65B's tokenizer is not in the Hugging Face cache (run.sh models)")
        parity(tok!!, ChatTemplate.RIPPLE, expectedRippleTurns())
    }

    @Test
    fun aTurnThatDoesNotExtendItsPromptIsReported() {
        // TwIL renders an assistant turn without reasoning as "<think></think>...", which the
        // generation prompt ("<think>\n") does not start: such an episode is not trained on as is.
        val tok = twil()
        val conv = listOf(ConvMessage("system", "s"), ConvMessage("user", "q"), ConvMessage("assistant", "a", null))
        val seq = renderForTraining(tok, ChatTemplate.TWIL, conv, toolsJson)
        assertTrue(!seq.ok && "not a prefix" in seq.problems.single(), seq.problems.joinToString())
    }

    @Test
    fun budgetsEndAnEpisode() {
        val tok = twil()
        val conv = conversation()
        val seq = renderForTraining(tok, ChatTemplate.TWIL, conv, toolsJson)
        val opts = chatOptions(toolsJson, true)
        val length = { c: List<ConvMessage> -> tok.encode(tok.applyChatTemplate(c.map { it.forTemplate() }, opts, ChatTemplate.TWIL), addSpecialTokens = false).size }
        fun run(settings: AgentSettings): Map<String, Any?> {
            val backend = ScriptedBackend(ArrayDeque(seq.spans.map { seq.ids.copyOfRange(it.first, it.last + 1) }))
            val ep = endpoint(tok, backend, Files.createTempDirectory("brooklyn-endpoint"))
            try {
                val results = ArrayDeque(toolResults)
                return runEpisode(task(), ChatClient("http://127.0.0.1:${ep.port}/v1", "m", apiKey = null), toolDefs, systemPrompt(date), settings, { _, _ -> results.removeFirst() }, length)
            } finally {
                ep.close()
            }
        }
        // One call allowed: the second turn's first call ends the episode before it runs.
        val calls = run(AgentSettings(maxCalls = 1))
        assertEquals("call_budget", calls["stop"])
        assertEquals(1, (calls["calls"] as List<*>).size)
        assertEquals("", calls["answer"])
        // A completion budget smaller than the first two turns: the third turn is not requested.
        val firstTwo = (0..1).sumOf { seq.spans[it].last - seq.spans[it].first + 1 }
        val tokens = run(AgentSettings(maxTokens = firstTwo + 10))
        assertEquals("token_budget", tokens["stop"])
        assertEquals(2, (tokens["turns"] as List<*>).size)
        // Each turn's max_tokens stays inside the context.
        @Suppress("UNCHECKED_CAST")
        val turns = run(AgentSettings(maxModelLen = 16_384, turnTokens = 4_096))["turns"] as List<Map<String, Any?>>
        for (t in turns) assertTrue((t["max_tokens"] as Int) + (t["rendered_prompt_tokens"] as Int) < 16_384)
    }

    @Test
    fun normalizedTrajectoriesIgnoreTimes() {
        val dir = Files.createTempDirectory("brooklyn-traj")
        val a = dir.resolve("a.jsonl")
        val b = dir.resolve("b.jsonl")
        Files.writeString(a, """{"id":"x","turns":[{"seed":1,"timing":{"ms":5}}],"timing":{"ms":10,"started":"t1"}}""" + "\n")
        Files.writeString(b, """{"id":"x","turns":[{"seed":1,"timing":{"ms":7}}],"timing":{"ms":99,"started":"t2"}}""" + "\n")
        assertEquals(trajectoriesSha256(a), trajectoriesSha256(b))
        Files.writeString(b, """{"id":"x","turns":[{"seed":2,"timing":{"ms":7}}],"timing":{"ms":99,"started":"t2"}}""" + "\n")
        assertTrue(trajectoriesSha256(a) != trajectoriesSha256(b))
    }

    @Test
    fun theSampleTakesCategoriesInTurnAndIsSeeded() {
        val tasks = (0 until 30).map { i ->
            Task("t$i", "dev", listOf("a", "b", "c")[i % 3], "k", "q$i", emptyList(), emptyList(), null, emptyList(), "", i != 4, 0, "2026-11-13")
        }
        val s = sampleTasks(tasks, 10, 5)
        assertEquals(10, s.size)
        assertEquals(listOf("a", "b", "c", "a", "b", "c", "a", "b", "c", "a"), s.map { it.category })
        assertTrue(s.all { it.offline })
        assertEquals(s.map { it.id }, sampleTasks(tasks, 10, 5).map { it.id })
        assertTrue(s.map { it.id } != sampleTasks(tasks, 10, 6).map { it.id })
        assertEquals(episodeSeed(1, "t1"), episodeSeed(1, "t1"))
        assertTrue(episodeSeed(1, "t1") in 0..0xFFFFFF)
    }
}
