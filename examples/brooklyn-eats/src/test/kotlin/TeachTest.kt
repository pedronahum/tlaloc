package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.ChatToolCall
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TeachTest {
    private val toolsJson = TOOL_DEFS.map { toJson(it.openAi()) }

    private fun twil(): HfTokenizer {
        val dir = runCatching { modelDir("twil_8b", hubCache(), readManifestJson(Path.of("manifest.json").toAbsolutePath())) }.getOrNull()
        assumeTrue(dir != null && Files.exists(dir.resolve("tokenizer.json")), "TwIL-8B's tokenizer is not in the Hugging Face cache (run.sh models)")
        return HfTokenizer.load(dir!!)
    }

    // A synthetic Ripple-style episode: answers start with a newline, reasoning ends with one, a call
    // has a nested object and a number, one turn has two calls.
    private fun episode(): JsonObject {
        val search = ChatToolCall("search_restaurants", """{"query": "Golden Lantern", "near": {"lat": 40.68, "lon": -73.97}, "limit": 5}""")
        val get = ChatToolCall("get_restaurant", """{"id": "50000001"}""")
        val insp = ChatToolCall("get_inspections", """{"id": "50000001", "since": "2025-11-13"}""")
        val msgs = listOf(
            messageJson(ConvMessage("system", "You are Brooklyn Eats.")),
            messageJson(ConvMessage("user", "What grade does Golden Lantern have?")),
            messageJson(ConvMessage("assistant", "\nLet me search.", "The user wants a grade. Search first.\n", listOf(search)), listOf("call_0_0")),
            messageJson(ConvMessage("tool", """{"results":[{"id":"50000001","name":"Golden Lantern Kitchen"}]}"""), toolCallId = "call_0_0"),
            messageJson(ConvMessage("assistant", "", "One match. Its record and inspections.", listOf(get, insp)), listOf("call_1_0", "call_1_1")),
            messageJson(ConvMessage("tool", """{"id":"50000001","grade":"A"}"""), toolCallId = "call_1_0"),
            messageJson(ConvMessage("tool", """{"inspections":[{"date":"2026-03-02","grade":"A"}]}"""), toolCallId = "call_1_1"),
            messageJson(ConvMessage("assistant", "\nGolden Lantern Kitchen has an **A** (March 2, 2026).", "Grade A from 2026-03-02.")),
        )
        return parseJson(toJson(linkedMapOf("id" to "t1", "messages" to msgs, "stop" to "answer"))) as JsonObject
    }

    @Test
    fun keptTrajectoryRendersForTwilAndParsesBackToItsCalls() {
        val tok = twil()
        val ep = episode()
        val seq = renderForTraining(tok, ChatTemplate.TWIL, twilMessages(ep), toolsJson)
        assertTrue(seq.ok, seq.problems.joinToString())
        assertEquals(3, seq.spans.size)
        val pb = parseBack(tok, ShardSequence(seq.ids, seq.mask), ep, toolsJson)
        assertEquals(emptyList(), pb.problems)
        assertEquals(3, pb.callsEqual)
        assertEquals(3, pb.textEqual)
        // The masked runs are exactly the assistant spans.
        assertEquals(seq.spans, maskRuns(seq.mask))
    }

    @Test
    fun parseBackReportsAChangedCall() {
        val tok = twil()
        val ep = episode()
        val seq = renderForTraining(tok, ChatTemplate.TWIL, twilMessages(ep), toolsJson)
        // The recorded episode says limit 6; the stored tokens say 5.
        val other = parseJson(toJson(ep.toKotlin()).replace("\\\"limit\\\": 5", "\\\"limit\\\": 6")) as JsonObject
        val pb = parseBack(tok, ShardSequence(seq.ids, seq.mask), other, toolsJson)
        assertEquals(2, pb.callsEqual)
        assertTrue(pb.problems.any { "limit" in it }, pb.problems.joinToString())
    }

    @Test
    fun shardsRoundTripAndAreByteIdenticalWhenWrittenAgain() {
        val dir = Files.createTempDirectory("shards")
        val a = Files.createTempDirectory("shards")
        try {
            val kept = (0 until SHARD_SEQUENCES + 3).map { i ->
                val ids = IntArray(5 + i % 7) { it * 31 + i }
                val mask = BooleanArray(ids.size) { it % 3 == 1 }
                KeptSequence("t$i", i % 2, "lookup", "grade", TrainingSequence(ids, mask, emptyList(), emptyList(), emptyList()), "sha$i", 1, 2)
            }
            writeShards(dir, kept)
            writeShards(a, kept)
            val files = shardFiles(dir).map { it.fileName.toString() }
            assertEquals(listOf("index.jsonl", "shard-00000.bin", "shard-00001.bin"), files)
            for (f in files) assertContentEquals(Files.readAllBytes(dir.resolve(f)), Files.readAllBytes(a.resolve(f)))
            assertEquals(shardsSha256(dir), shardsSha256(a))
            val back = readShard(dir.resolve("shard-00000.bin")) + readShard(dir.resolve("shard-00001.bin"))
            assertEquals(kept.size, back.size)
            for ((k, s) in kept.zip(back)) {
                assertContentEquals(k.seq.ids, s.ids)
                assertContentEquals(k.seq.mask, s.mask)
            }
            val index = Files.readAllLines(dir.resolve("index.jsonl")).map { parseJson(it) as JsonObject }
            assertEquals("shard-00001.bin", index[SHARD_SEQUENCES].str("shard"))
            assertEquals(0, index[SHARD_SEQUENCES].optInt("row"))
            // Writing fewer sequences removes the shard files of the earlier, longer write.
            writeShards(dir, kept.take(2))
            assertEquals(listOf("index.jsonl", "shard-00000.bin"), shardFiles(dir).map { it.fileName.toString() })
        } finally {
            dir.toFile().deleteRecursively()
            a.toFile().deleteRecursively()
        }
    }

    @Test
    fun runLogKeepsEverySession() {
        val f = Files.createTempFile("run", ".key")
        try {
            val first = RunLog("{\"k\":1}").begin(mapOf("commit" to "aaa"), 0)
            first.seconds = 12.5
            first.write(f)
            val again = RunLog.read(f)!!
            assertEquals("{\"k\":1}", again.keyText)
            assertEquals(12.5, again.priorSeconds)
            val second = again.begin(mapOf("commit" to "bbb"), 7)
            assertEquals(2, second.segments.size)
            assertEquals("bbb", (second.segments[1]["code"] as Map<*, *>)["commit"])
            assertEquals(7L, second.segments[1]["episodes_recorded"])
        } finally {
            Files.deleteIfExists(f)
        }
    }

    @Test
    fun keptToleranceIsThreeBinomialStandardErrors() {
        // 1000 episodes, 800 kept: 3 * sqrt(1000 * 0.8 * 0.2) = 37.9.
        assertEquals(38L, keptTolerance(1000, 800))
        assertEquals(1L, keptTolerance(8, 8))
        assertEquals(1L, keptTolerance(0, 0))
    }

    @Test
    fun numbersCompareByValueInVerify() {
        assertTrue(sameValue(parseJson("1") as JsonNumber, 1.0))
        assertTrue(sameValue(parseJson("0.25") as JsonNumber, 0.25))
        assertFalse(sameValue(parseJson("1") as JsonNumber, 2L))
    }
}
