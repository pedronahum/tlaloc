package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.ChatOutput
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.TwilOutputParser
import io.tlaloc.tokenizer.load
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.time.Instant
import java.time.LocalDate
import kotlin.math.ceil
import kotlin.math.sqrt

// Teacher trajectories: Ripple-65B (served as serving/ripple-nvfp4.env) runs the agent loop on the
// training split with seeded sampling, tools in snapshot mode; the verifier scores every answer; the
// passing trajectories are re-rendered in TwIL-8B's chat format with assistant masks and written as
// training shards.
//
//   teach run      [--concurrency 4] [--endpoint URL] [--fresh]   the sample passes without current results (needs the server)
//   teach pending                                                  sample passes without current results, one index per line
//   teach render                                                   shards from the recorded passes (no server)
//   teach check                                                    shards regenerate bit for bit; 10 sequences parse back to their tool calls
//   teach report                                                   yield per category, lengths, shard identities
//
// Files under <data>/teach/: sample-<s>.jsonl (every episode of pass s, scored) and sample-<s>.json
// (settings, provenance, metrics); shards/shard-NNNNN.bin, shards/index.jsonl and summary.json.
//
// Shard format (little-endian): the 8 bytes "BKEATS01", u32 count, then per sequence u32 length,
// length u32 token ids and length mask bytes (1: a token the student learns to produce). The
// sequences are in index.jsonl's order, a seeded shuffle of the kept trajectories.

const val TEACH_DIR = "teach"
const val TEACH_CONFIG = "ripple-nvfp4"
/** Questions of the training split the full scope samples (categories interleaved by the teach_sample seed). */
const val TEACH_N = 1000
/** Samples per question in the full scope. */
const val TEACH_SAMPLES = 1
const val QUICK_TEACH_N = 4
const val QUICK_TEACH_SAMPLES = 2
const val TEACH_TEMPERATURE = 0.7
const val TEACH_TOP_P = 0.95
/** Sequences per shard file. */
const val SHARD_SEQUENCES = 256
/** Longest kept sequence in TwIL's tokens: the served context of the agent and the evaluation. */
const val TEACH_MAX_TOKENS = 16_384
/** Sequences `teach check` decodes and parses back. */
const val PARSE_BACK_N = 10
val SHARD_MAGIC = "BKEATS01".toByteArray(Charsets.US_ASCII)

private fun opt(args: List<String>, name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }

fun isQuickData(data: Path): Boolean = data.resolve("brooklyn.db").let { Files.exists(it) && dbScope(it) == "quick" }

/** Questions and samples per question of [data]'s scope. */
fun teachSize(data: Path): Pair<Int, Int> = if (isQuickData(data)) QUICK_TEACH_N to QUICK_TEACH_SAMPLES else TEACH_N to TEACH_SAMPLES

/** Agent settings of sample pass [sample]: seeded sampling; the base seed is seeds.teach + [sample]. */
fun teachSettings(sample: Int, manifest: Path = manifestPath()): AgentSettings =
    AgentSettings(temperature = TEACH_TEMPERATURE, topP = TEACH_TOP_P, seed = manifestSeed("teach", manifest) + sample)

/** The questions: offline tasks of the training split, [n] of them, categories interleaved by seeds.teach_sample. */
fun teachTasks(data: Path, n: Int = teachSize(data).first): List<Task> =
    sampleTasks(readTasks(taskFile(data, Split.TRAIN)).filter { it.offline }, n, manifestSeed("teach_sample"))

fun teachDir(data: Path): Path = data.resolve(TEACH_DIR)
fun sampleFile(data: Path, sample: Int, ext: String = "jsonl"): Path = teachDir(data).resolve("sample-$sample.$ext")
fun shardDir(data: Path): Path = teachDir(data).resolve("shards")

/**
 * What a pass's episodes depend on, except which questions it runs (a longer list keeps the
 * episodes of a shorter one: the question order is a prefix-stable interleave).
 */
fun teachKey(home: Path, data: Path, sample: Int): Map<String, Any?> {
    val config = servingConfig(home, TEACH_CONFIG)
    return linkedMapOf(
        "serving_config_sha256" to fileSha256(home.resolve("serving").resolve("$TEACH_CONFIG.env")),
        "model_key" to config["MODEL_KEY"], "draft_key" to config["DRAFT_KEY"],
        "settings" to teachSettings(sample, home.resolve("manifest.json")).json(), "concurrency" to EVAL_CONCURRENCY.toLong(),
        "task_file_sha256" to fileSha256(taskFile(data, Split.TRAIN)),
    )
}

private fun sameJson(a: Any?, b: Any?): Boolean = canonicalJson(parseJson(toJson(a))) == canonicalJson(parseJson(toJson(b)))

/** Whether pass [sample] has results made with the current key for exactly the current questions. */
fun teachPassCurrent(home: Path, data: Path, sample: Int, ids: List<String>): Boolean {
    val res = sampleFile(data, sample, "json")
    if (!Files.exists(res) || !Files.exists(sampleFile(data, sample))) return false
    val o = parseJson(Files.readString(res)) as JsonObject
    return sameJson(o["key"], teachKey(home, data, sample)) && o.optString("task_ids_sha256") == taskIdsSha256(ids)
}

fun teachCmd(args: List<String>): Int {
    val home = exampleHome()
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    if (!Files.exists(data.resolve("brooklyn.db"))) { System.err.println("teach: ${data.resolve("brooklyn.db")} not found; run the ingest stage first"); return 1 }
    if (!Files.exists(taskFile(data, Split.TRAIN))) { System.err.println("teach: ${taskFile(data, Split.TRAIN)} not found; run the tasks stage first"); return 1 }
    return when (args.firstOrNull()) {
        "run" -> teachRun(home, data, args.drop(1))
        "pending" -> {
            val ids = teachTasks(data).map { it.id }
            for (s in 0 until teachSize(data).second) if (!teachPassCurrent(home, data, s, ids)) println(s)
            0
        }
        "render" -> teachRender(home, data, shardDir(data), teachDir(data).resolve("summary.json"), log = true)
        "check" -> teachCheck(home, data)
        "report" -> teachReport(data)
        else -> {
            System.err.println(
                """usage: teach run [--concurrency 4] [--endpoint URL] [--fresh]
       teach pending | render | check | report      (all take --data DIR)""",
            )
            2
        }
    }
}

/** Runs every sample pass that has no current results, resuming an interrupted one. */
fun teachRun(home: Path, data: Path, args: List<String>): Int {
    val db = data.resolve("brooklyn.db").toAbsolutePath().normalize()
    val (n, samples) = teachSize(data)
    val tasks = teachTasks(data, n)
    val ids = tasks.map { it.id }
    val config = servingConfig(home, TEACH_CONFIG)
    val template = ChatTemplate.valueOf(config.getValue("TEMPLATE"))
    val tokenizer = HfTokenizer.load(modelDir(config.getValue("MODEL_KEY")))
    val concurrency = opt(args, "--concurrency")?.toInt() ?: EVAL_CONCURRENCY
    val endpoint = opt(args, "--endpoint") ?: "http://127.0.0.1:${System.getenv("BROOKLYN_ENDPOINT_PORT") ?: "8120"}/v1"
    val client = ChatClient(endpoint, config.getValue("SERVED_NAME"))
    val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
    val fresh = "--fresh" in args
    Files.createDirectories(teachDir(data))
    var rc = 0
    for (s in 0 until samples) {
        if (!fresh && teachPassCurrent(home, data, s, ids)) { println("teach: pass $s is current (${tasks.size} questions)"); continue }
        val settings = teachSettings(s)
        val key = teachKey(home, data, s)
        val keyText = toJson(key)
        val out = sampleFile(data, s)
        val res = sampleFile(data, s, "json")
        val partial = sampleFile(data, s, "partial.jsonl")
        val partialKey = sampleFile(data, s, "partial.key")
        // Episodes already recorded with this key: a finished pass over other questions, then an interrupted one.
        val done = LinkedHashMap<String, Map<String, Any?>>()
        var prior = RunLog(keyText)
        val idSet = ids.toSet()
        @Suppress("UNCHECKED_CAST")
        fun keep(e: JsonObject) { if (e.str("id") in idSet && !e.str("stop").startsWith("error")) done[e.str("id")] = e.toKotlin() as Map<String, Any?> }
        if (!fresh && Files.exists(res) && Files.exists(out) && sameJson((parseJson(Files.readString(res)) as JsonObject)["key"], key)) {
            readEpisodes(out).forEach(::keep)
            prior = RunLog.fromResult(keyText, parseJson(Files.readString(res)) as JsonObject)
        }
        val resumed = RunLog.read(partialKey)
        if (!fresh && Files.exists(partial) && resumed?.keyText == keyText) {
            readEpisodes(partial).forEach(::keep)
            prior = resumed
        } else {
            Files.deleteIfExists(partial)
        }
        val log = prior.begin(codeCommit(home), done.size)
        log.write(partialKey)
        println("teach: pass $s: ${tasks.size} questions (${done.size} recorded), $endpoint model ${client.model}, concurrency $concurrency")
        println("teach: settings ${toJson(settings.json())}")
        val t0 = System.nanoTime()
        val writer = Files.newBufferedWriter(partial, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
        val scored = try {
            runAgentTasks(tasks, client, settings, concurrency, db, ToolMode.SNAPSHOT, tokenizer, template, systemPrompt(cutoff), cutoff, done) { ep ->
                writer.write(toJson(ep) + "\n")
                writer.flush()
                log.seconds = log.priorSeconds + (System.nanoTime() - t0) / 1e9
                log.write(partialKey)
            }
        } finally {
            writer.close()
        }
        log.seconds = log.priorSeconds + (System.nanoTime() - t0) / 1e9
        Files.writeString(out, scored.joinToString("") { toJson(it) + "\n" })
        val eps = readEpisodes(out)
        val metrics = evalMetrics(eps, log.seconds)
        val result = linkedMapOf(
            "config" to TEACH_CONFIG, "model" to client.model, "split" to "train", "sample" to s.toLong(), "scope" to dbScope(db),
            "key" to key, "tasks" to ids.size.toLong(), "task_ids_sha256" to taskIdsSha256(ids), "template" to template.name,
            "system_prompt_sha256" to sha256Hex(systemPrompt(cutoff).toByteArray()), "code" to codeCommit(home),
            "segments" to log.segments, "seconds" to log.seconds,
            "trajectories_sha256" to trajectoriesSha256(out), "metrics" to metrics,
        )
        Files.writeString(res, prettyJson(result) + "\n")
        Files.deleteIfExists(partial)
        Files.deleteIfExists(partialKey)
        println("teach: wrote $out and $res")
        println("teach: ${metricsLine(metrics)}")
        if (eps.any { it.str("stop").startsWith("error") }) rc = 1
    }
    return rc
}

/**
 * The sessions of a resumable run: each start or resume, with the code it ran, when it started and
 * how many episodes were already recorded. Kept in the run's .partial.key file (line 1 the key, line 2
 * the seconds so far, then one JSON object per session) and copied into its results file.
 */
class RunLog(val keyText: String, val priorSeconds: Double = 0.0, val segments: List<Map<String, Any?>> = emptyList()) {
    var seconds: Double = priorSeconds

    fun begin(code: Map<String, Any?>?, recorded: Int): RunLog =
        RunLog(keyText, priorSeconds, segments + linkedMapOf("started" to Instant.now().toString(), "code" to code, "episodes_recorded" to recorded.toLong()))

    fun write(f: Path) {
        Files.writeString(f, "$keyText\n$seconds\n" + segments.joinToString("") { toJson(it) + "\n" })
    }

    companion object {
        fun read(f: Path): RunLog? {
            if (!Files.exists(f)) return null
            val lines = Files.readAllLines(f)
            val key = lines.firstOrNull() ?: return null
            @Suppress("UNCHECKED_CAST")
            val segs = lines.drop(2).filter { it.isNotBlank() }.map { (parseJson(it) as JsonObject).toKotlin() as Map<String, Any?> }
            return RunLog(key, lines.getOrNull(1)?.toDoubleOrNull() ?: 0.0, segs)
        }

        /** The sessions and seconds a finished run's results file records (a run extended to more questions continues them). */
        @Suppress("UNCHECKED_CAST")
        fun fromResult(keyText: String, o: JsonObject): RunLog = RunLog(
            keyText, o.optNumber("seconds") ?: 0.0,
            (o["segments"] as? JsonArray)?.elements.orEmpty().map { (it as JsonObject).toKotlin() as Map<String, Any?> },
        )
    }
}

/** A kept trajectory: its episode, the TwIL rendering and where it came from. */
class KeptSequence(val id: String, val sample: Int, val category: String, val kind: String, val seq: TrainingSequence, val episodeSha256: String, val calls: Int, val turns: Int)

/** The episodes of every pass, in pass order, for the current questions; null when a pass is missing. */
fun teachEpisodes(data: Path): List<Pair<Int, List<JsonObject>>>? {
    val (_, samples) = teachSize(data)
    return (0 until samples).map { s ->
        val f = sampleFile(data, s)
        if (!Files.exists(f)) return null
        s to readEpisodes(f)
    }
}

/** SHA-256 over every pass's normalized episodes (timing removed), passes in order. */
fun teachTrajectoriesSha256(data: Path): String? {
    val (_, samples) = teachSize(data)
    val parts = (0 until samples).map { s -> sampleFile(data, s).takeIf { Files.exists(it) }?.let { trajectoriesSha256(it) } ?: return null }
    return sha256Hex(parts.joinToString("\n").toByteArray())
}

/** The assistant messages of [ep] as TwIL renders them: reasoning trimmed (TwIL's template trims the answer itself). */
fun twilMessages(ep: JsonObject): List<ConvMessage> = (ep["messages"] as JsonArray).elements.map { parseConvMessage(it as JsonObject) }.map { m ->
    if (m.role == "assistant") ConvMessage(m.role, m.content, m.reasoning?.trim(), m.toolCalls) else m
}

fun episodePassed(ep: JsonObject): Boolean = ((ep["verdict"] as? JsonObject)?.get("pass") as? JsonBool)?.value == true

/**
 * Writes the shards of the recorded passes into [dir] and the summary into [summaryFile]. Kept: an
 * episode the verifier passed that ended with an answer, whose TwIL rendering has every assistant
 * turn's prompt as a prefix of the chat (renderForTraining) and holds at most [TEACH_MAX_TOKENS]
 * tokens. Order: SHA-256 of seeds.teach_shuffle, the task id and the sample. Every file is a
 * function of the passes' episodes, the tokenizer and the seeds (no times).
 */
fun teachRender(home: Path, data: Path, dir: Path, summaryFile: Path, log: Boolean): Int {
    val passes = teachEpisodes(data) ?: run { System.err.println("teach render: a sample pass is missing; run `teach run` first"); return 1 }
    val tasks = teachTasks(data)
    val order = tasks.withIndex().associate { it.value.id to it.index }
    val tokenizer = HfTokenizer.load(modelDir(STUDENT_KEY))
    val toolsJson = TOOL_DEFS.map { toJson(it.openAi()) }
    val kept = ArrayList<KeptSequence>()
    val excluded = java.util.TreeMap<String, Int>()
    val perCat = java.util.TreeMap<String, IntArray>() // questions, episodes, passed, kept
    val keptQuestions = HashMap<String, MutableSet<String>>()
    for ((cat, ts) in tasks.groupBy { it.category }) perCat.getOrPut(cat) { IntArray(4) }[0] = ts.size
    for ((s, eps) in passes) {
        val byId = eps.associateBy { it.str("id") }
        for (t in tasks) {
            val ep = byId[t.id] ?: run { System.err.println("teach render: pass $s has no episode for ${t.id}; run `teach run`"); return 1 }
            val c = perCat.getValue(t.category)
            c[1]++
            if (!episodePassed(ep)) { excluded.merge("verifier failed", 1, Int::plus); continue }
            c[2]++
            if (ep.str("stop") != "answer") { excluded.merge("stop ${ep.str("stop")}", 1, Int::plus); continue }
            val seq = renderForTraining(tokenizer, ChatTemplate.TWIL, twilMessages(ep), toolsJson)
            if (!seq.ok) { excluded.merge("TwIL rendering: ${seq.problems.first().substringAfter(": ").substringBefore(" (")}", 1, Int::plus); continue }
            if (seq.ids.size > TEACH_MAX_TOKENS) { excluded.merge("longer than $TEACH_MAX_TOKENS tokens", 1, Int::plus); continue }
            c[3]++
            keptQuestions.getOrPut(t.category) { HashSet() } += t.id
            kept += KeptSequence(
                t.id, s, t.category, t.kind, seq, sha256Hex(canonicalJson(normalizedEpisode(ep)).toByteArray()),
                (ep["calls"] as JsonArray).elements.size, (ep["turns"] as JsonArray).elements.size,
            )
        }
    }
    val shuffleSeed = manifestSeed("teach_shuffle")
    kept.sortWith(compareBy({ sha256Hex("$shuffleSeed:${it.id}:${it.sample}".toByteArray()) }, { order[it.id] ?: 0 }, { it.sample }))
    writeShards(dir, kept)
    val lengths = kept.map { it.seq.ids.size }.sorted()
    fun pct(p: Double) = if (lengths.isEmpty()) null else lengths[((lengths.size - 1) * p).toInt()].toLong()
    val episodes = passes.sumOf { it.second.size }
    val summary = linkedMapOf(
        "scope" to dbScope(data.resolve("brooklyn.db")), "questions" to tasks.size.toLong(), "samples" to passes.size.toLong(),
        "episodes" to episodes.toLong(), "passed" to perCat.values.sumOf { it[2] }.toLong(), "kept" to kept.size.toLong(),
        "yield" to if (episodes == 0) null else kept.size.toDouble() / episodes,
        "questions_with_kept" to keptQuestions.values.sumOf { it.size }.toLong(),
        "excluded" to excluded.mapValues { it.value.toLong() },
        "per_category" to perCat.mapValues { (cat, c) ->
            linkedMapOf(
                "questions" to c[0].toLong(), "episodes" to c[1].toLong(), "passed" to c[2].toLong(), "kept" to c[3].toLong(),
                "yield" to if (c[1] == 0) null else c[3].toDouble() / c[1], "questions_with_kept" to (keptQuestions[cat]?.size ?: 0).toLong(),
            )
        },
        "tokens" to lengths.sumOf { it.toLong() }, "masked_tokens" to kept.sumOf { it.seq.maskedTokens.toLong() },
        "length_p50" to pct(0.5), "length_p90" to pct(0.9), "length_max" to lengths.lastOrNull()?.toLong(),
        "trajectories_sha256" to teachTrajectoriesSha256(data), "tokenizer" to STUDENT_KEY, "template" to "TWIL",
        "shards" to shardFiles(dir).map { linkedMapOf("file" to it.fileName.toString(), "sha256" to fileSha256(it)) },
        "shards_sha256" to shardsSha256(dir),
    )
    Files.writeString(summaryFile, prettyJson(summary) + "\n")
    if (log) {
        println("teach render: ${kept.size} of $episodes episodes kept (${passes.size} passes over ${tasks.size} questions); excluded $excluded")
        for ((cat, c) in perCat) println("  %-13s questions %4d  episodes %4d  passed %4d  kept %4d (%.1f%%)".format(cat, c[0], c[1], c[2], c[3], if (c[1] == 0) 0.0 else 100.0 * c[3] / c[1]))
        println("teach render: ${shardFiles(dir).size} shards in $dir, ${summary["tokens"]} tokens (${summary["masked_tokens"]} masked); length p50 ${summary["length_p50"]}, p90 ${summary["length_p90"]}, max ${summary["length_max"]}")
        println("teach render: shards sha256 ${summary["shards_sha256"]}")
    }
    return 0
}

const val STUDENT_KEY = "twil_8b"

/** Writes [kept] (in order) as shard files of [SHARD_SEQUENCES] sequences and index.jsonl into [dir], replacing earlier ones. */
fun writeShards(dir: Path, kept: List<KeptSequence>) {
    Files.createDirectories(dir)
    Files.list(dir).use { st -> st.filter { it.fileName.toString().startsWith("shard-") || it.fileName.toString() == "index.jsonl" }.toList() }.forEach { Files.delete(it) }
    val index = StringBuilder()
    for ((k, chunk) in kept.chunked(SHARD_SEQUENCES).withIndex()) {
        val name = "shard-%05d.bin".format(k)
        val bytes = chunk.sumOf { 4L + 5L * it.seq.ids.size } + SHARD_MAGIC.size + 4
        val buf = ByteBuffer.allocate(Math.toIntExact(bytes)).order(ByteOrder.LITTLE_ENDIAN)
        buf.put(SHARD_MAGIC).putInt(chunk.size)
        for ((row, ks) in chunk.withIndex()) {
            buf.putInt(ks.seq.ids.size)
            for (id in ks.seq.ids) buf.putInt(id)
            for (m in ks.seq.mask) buf.put(if (m) 1 else 0)
            index.append(
                toJson(
                    linkedMapOf(
                        "shard" to name, "row" to row, "id" to ks.id, "sample" to ks.sample, "category" to ks.category, "kind" to ks.kind,
                        "tokens" to ks.seq.ids.size, "masked_tokens" to ks.seq.maskedTokens, "assistant_turns" to ks.seq.spans.size,
                        "turns" to ks.turns, "calls" to ks.calls, "episode_sha256" to ks.episodeSha256,
                    ),
                ),
            ).append('\n')
        }
        Files.write(dir.resolve(name), buf.array())
    }
    Files.writeString(dir.resolve("index.jsonl"), index.toString())
}

fun shardFiles(dir: Path): List<Path> = if (!Files.isDirectory(dir)) emptyList() else
    Files.list(dir).use { st -> st.filter { it.fileName.toString().let { n -> n.startsWith("shard-") && n.endsWith(".bin") || n == "index.jsonl" } }.toList() }
        .sortedBy { it.fileName.toString() }

/** SHA-256 over the shard files' names and SHA-256 (index.jsonl included). */
fun shardsSha256(dir: Path): String? {
    val fs = shardFiles(dir)
    if (fs.isEmpty()) return null
    return sha256Hex(fs.joinToString("") { "${it.fileName}\t${fileSha256(it)}\n" }.toByteArray())
}

/** One sequence of a shard: token ids and mask. */
class ShardSequence(val ids: IntArray, val mask: BooleanArray)

fun readShard(f: Path): List<ShardSequence> {
    val buf = ByteBuffer.wrap(Files.readAllBytes(f)).order(ByteOrder.LITTLE_ENDIAN)
    val magic = ByteArray(SHARD_MAGIC.size).also { buf.get(it) }
    require(magic.contentEquals(SHARD_MAGIC)) { "$f is not a Brooklyn Eats shard" }
    val n = buf.getInt()
    return List(n) {
        val len = buf.getInt()
        val ids = IntArray(len) { buf.getInt() }
        val mask = BooleanArray(len) { buf.get().toInt() != 0 }
        ShardSequence(ids, mask)
    }.also { require(!buf.hasRemaining()) { "$f has bytes after its $n sequences" } }
}

/** The masked runs of [mask]: one per assistant turn. */
fun maskRuns(mask: BooleanArray): List<IntRange> {
    val out = ArrayList<IntRange>()
    var i = 0
    while (i < mask.size) {
        if (!mask[i]) { i++; continue }
        val s = i
        while (i < mask.size && mask[i]) i++
        out += s until i
    }
    return out
}

/** Result of decoding a stored sequence's assistant turns and parsing them with TwilOutputParser. */
class ParseBack(val turns: Int, val callsEqual: Int, val callsTotal: Int, val textEqual: Int, val problems: List<String>)

/**
 * Decodes each masked run of [s] (an assistant turn: reasoning, `</think>`, answer, calls,
 * `<|im_end|>`), parses it with TwilOutputParser and compares the tool calls (name and arguments as
 * canonical JSON) and the trimmed reasoning and answer with [ep]'s assistant messages.
 */
fun parseBack(tokenizer: HfTokenizer, s: ShardSequence, ep: JsonObject, toolsJson: List<String>): ParseBack {
    val assistant = twilMessages(ep).filter { it.role == "assistant" }
    val runs = maskRuns(s.mask)
    val problems = ArrayList<String>()
    if (runs.size != assistant.size) problems += "${runs.size} masked runs for ${assistant.size} assistant messages"
    var callsEqual = 0
    var callsTotal = 0
    var textEqual = 0
    for ((k, r) in runs.withIndex()) {
        val m = assistant.getOrNull(k) ?: break
        val text = tokenizer.decode(s.ids.copyOfRange(r.first, r.last + 1)).removeSuffix(END_TEXT)
        val out = ChatOutput.parse(text, TwilOutputParser(reasoning = true, tools = toolsJson))
        callsTotal += m.toolCalls.size
        if (out.toolCalls.size != m.toolCalls.size) problems += "turn $k: ${out.toolCalls.size} calls parsed, ${m.toolCalls.size} recorded"
        for ((i, c) in m.toolCalls.withIndex()) {
            val p = out.toolCalls.getOrNull(i) ?: continue
            if (p.name == c.name && canonicalJson(parseJson(p.argumentsJson)) == canonicalJson(parseJson(c.argumentsJson))) callsEqual++
            else problems += "turn $k call $i: parsed ${p.name} ${p.argumentsJson}, recorded ${c.name} ${c.argumentsJson}"
        }
        if (out.reasoning.trim() == (m.reasoning ?: "").trim() && out.content.trim() == m.content.trim()) textEqual++
        else problems += "turn $k: reasoning or answer text differs after parsing"
    }
    return ParseBack(runs.size, callsEqual, callsTotal, textEqual, problems)
}

/**
 * Regenerates the shards from the recorded passes into a scratch directory and compares every file
 * byte for byte with the shards on disk; then decodes the first [PARSE_BACK_N] sequences of the
 * index (a seeded shuffle) from the shard files and parses them back with TwilOutputParser.
 */
fun teachCheck(home: Path, data: Path): Int {
    val dir = shardDir(data)
    if (shardFiles(dir).isEmpty()) { System.err.println("teach check: no shards in $dir; run `teach render`"); return 1 }
    val tmp = teachDir(data).resolve("check-shards")
    tmp.toFile().deleteRecursively()
    val t0 = System.nanoTime()
    if (teachRender(home, data, tmp, teachDir(data).resolve("check-summary.json"), log = false) != 0) return 1
    val a = shardFiles(dir).map { it.fileName.toString() }
    val b = shardFiles(tmp).map { it.fileName.toString() }
    var bad = 0
    if (a != b) { bad++; println("teach check: files differ: $a vs $b") }
    for (name in a.intersect(b.toSet())) {
        if (!Files.readAllBytes(dir.resolve(name)).contentEquals(Files.readAllBytes(tmp.resolve(name)))) { bad++; println("teach check: $name differs") }
    }
    println("teach check: regenerated ${b.size} files in %.1f s; %s".format((System.nanoTime() - t0) / 1e9,
        if (bad == 0) "all ${a.size} byte-identical to $dir" else "$bad differ"))
    tmp.toFile().deleteRecursively()
    Files.deleteIfExists(teachDir(data).resolve("check-summary.json"))
    // Parse-back of stored sequences.
    val tokenizer = HfTokenizer.load(modelDir(STUDENT_KEY))
    val toolsJson = TOOL_DEFS.map { toJson(it.openAi()) }
    val episodes = HashMap<Pair<String, Int>, JsonObject>()
    for ((s, eps) in teachEpisodes(data)!!) for (e in eps) episodes[e.str("id") to s] = e
    val index = Files.readAllLines(dir.resolve("index.jsonl")).filter { it.isNotBlank() }.map { parseJson(it) as JsonObject }
    val shards = HashMap<String, List<ShardSequence>>()
    var sequencesOk = 0
    var calls = 0
    var callsEqual = 0
    var turns = 0
    var textEqual = 0
    // The first PARSE_BACK_N rows are listed one by one; every row of the index is parsed as well.
    var allOk = 0
    var allCalls = 0
    var allCallsEqual = 0
    for ((i, row) in index.withIndex()) {
        val seq = shards.getOrPut(row.str("shard")) { readShard(dir.resolve(row.str("shard"))) }[row.optInt("row")!!]
        val ep = episodes.getValue(row.str("id") to row.optInt("sample")!!)
        val pb = parseBack(tokenizer, seq, ep, toolsJson)
        allCalls += pb.callsTotal; allCallsEqual += pb.callsEqual
        if (pb.problems.isEmpty()) allOk++ else println("  ${row.str("id")} sample ${row.optInt("sample")}: ${pb.problems.joinToString("; ")}")
        if (i >= PARSE_BACK_N) continue
        calls += pb.callsTotal; callsEqual += pb.callsEqual; turns += pb.turns; textEqual += pb.textEqual
        if (pb.problems.isEmpty()) sequencesOk++
        println("  ${row.str("id")} ${row.str("category")}: ${pb.turns} turns, ${pb.callsEqual}/${pb.callsTotal} calls equal, ${pb.textEqual}/${pb.turns} turns' text equal")
    }
    val sampled = minOf(PARSE_BACK_N, index.size)
    println("teach check: parse-back of the first $sampled sequences: $sequencesOk without a difference; tool calls $callsEqual of $calls equal; " +
        "reasoning and answer text equal in $textEqual of $turns turns")
    println("teach check: parse-back of all ${index.size} sequences: $allOk without a difference; tool calls $allCallsEqual of $allCalls equal")
    return if (bad == 0 && allOk == index.size) 0 else 1
}

fun teachReport(data: Path): Int {
    val f = teachDir(data).resolve("summary.json")
    if (!Files.exists(f)) { System.err.println("teach report: $f not found; run `teach render`"); return 1 }
    println(Files.readString(f).trimEnd())
    for (s in 0 until teachSize(data).second) {
        val r = sampleFile(data, s, "json")
        if (Files.exists(r)) {
            val o = parseJson(Files.readString(r)) as JsonObject
            println("pass $s: ${o.optInt("tasks")} questions, %.0f s; ${metricsLine((o["metrics"] as JsonObject).toKotlin() as Map<String, Any?>)}".format(o.optNumber("seconds") ?: 0.0))
        }
    }
    return 0
}

/**
 * The spread a fresh teacher run's kept count is checked within: three binomial standard errors of
 * the recorded yield over the episodes (each sampled episode passes or not), at least 1.
 */
fun keptTolerance(episodes: Long, kept: Long): Long {
    if (episodes == 0L) return 1
    val p = kept.toDouble() / episodes
    return maxOf(1L, ceil(3 * sqrt(episodes * p * (1 - p))).toLong())
}

/**
 * Manifest entries of the teach stage. Inputs: the teacher's serving config and the sampling
 * settings. Derived: the recorded passes (normalized SHA-256 reported; a fresh run samples other
 * trajectories, so verify compares the kept count within [keptTolerance]) and the shards, which must
 * be byte-identical whenever the passes are the recorded ones.
 */
fun teachEntries(home: Path, data: Path): List<Entry> {
    val out = ArrayList<Entry>()
    out += Entry("teach", "inputs", "serving_config_ripple", false) { servingConfigIdentity(home, TEACH_CONFIG) }
    out += Entry("teach", "inputs", "teach_settings", false) {
        val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
        linkedMapOf(
            "system_prompt_sha256" to sha256Hex(systemPrompt(cutoff).toByteArray()),
            "settings" to toJson(AgentSettings(temperature = TEACH_TEMPERATURE, topP = TEACH_TOP_P).json().filterKeys { it != "seed" }),
            "seed" to "seeds.teach + sample index", "questions" to TEACH_N.toLong(), "samples" to TEACH_SAMPLES.toLong(),
            "quick_questions" to QUICK_TEACH_N.toLong(), "quick_samples" to QUICK_TEACH_SAMPLES.toLong(),
            "concurrency" to EVAL_CONCURRENCY.toLong(), "tools" to "snapshot", "student_template" to "TWIL",
            "shard_sequences" to SHARD_SEQUENCES.toLong(), "max_tokens" to TEACH_MAX_TOKENS.toLong(),
        )
    }
    val summaryFile = teachDir(data).resolve("summary.json")
    fun summary(): JsonObject? = if (Files.exists(summaryFile)) parseJson(Files.readString(summaryFile)) as JsonObject else null
    lateinit var traj: Entry
    traj = Entry(
        "teach", "derived", "trajectories", true,
        setOf("sha256", "passed", "kept", "yield", "kept_tolerance", "questions_with_kept", "produced_by", "seconds"),
        { want, now ->
            val w = want.optInt("kept")
            val tol = want.optInt("kept_tolerance") ?: 0
            val n = (now["kept"] as? Long)?.toInt()
            if (w != null && n != null && Math.abs(w - n) > tol) listOf("kept $w -> $n, more than kept_tolerance $tol") else emptyList()
        },
    ) {
        val sha = teachTrajectoriesSha256(data) ?: return@Entry null
        val ids = teachTasks(data).map { it.id }
        val (_, samples) = teachSize(data)
        if ((0 until samples).any { !teachPassCurrent(home, data, it, ids) }) traj.stale = "a sample pass was run with other settings, serving config or questions; rerun the teach stage"
        val s = summary()
        if (s == null || s.optString("trajectories_sha256") != sha) traj.stale = "summary.json was not rendered from these passes; run `teach render`"
        val episodes = s?.optInt("episodes")?.toLong() ?: 0
        val kept = s?.optInt("kept")?.toLong() ?: 0
        val res = (0 until samples).map { parseJson(Files.readString(sampleFile(data, it, "json"))) as JsonObject }
        linkedMapOf(
            "sha256" to sha, "tasks" to taskIdsSha256(ids), "episodes" to episodes, "passed" to s?.optInt("passed")?.toLong(),
            "kept" to kept, "yield" to s?.optNumber("yield"), "kept_tolerance" to keptTolerance(episodes, kept),
            "questions_with_kept" to s?.optInt("questions_with_kept")?.toLong(),
            "seconds" to res.sumOf { it.optNumber("seconds") ?: 0.0 },
            "produced_by" to res.mapNotNull { (it["code"] as? JsonObject)?.optString("commit") }.distinct().joinToString(","),
        )
    }
    out += traj
    lateinit var shards: Entry
    shards = Entry(
        "teach", "derived", "shards", true,
        // Byte identity is required only for the recorded passes: the check below compares the
        // hash when the passes are the ones the manifest records.
        setOf("sha256", "sequences", "tokens", "masked_tokens", "trajectories_sha256"),
        { want, now ->
            if (want.optString("trajectories_sha256") == now["trajectories_sha256"] && want.optString("sha256") != now["sha256"]) {
                listOf("sha256 ${want.optString("sha256")} -> ${now["sha256"]} from the recorded trajectories (they must regenerate bit for bit)")
            } else emptyList()
        },
    ) {
        val dir = shardDir(data)
        val sha = shardsSha256(dir) ?: return@Entry null
        val s = summary()
        if (s == null || s.optString("shards_sha256") != sha) shards.stale = "the shards are not the ones summary.json records; run `teach render`"
        else if (s.optString("trajectories_sha256") != teachTrajectoriesSha256(data)) shards.stale = "the shards were rendered from other passes; run `teach render`"
        linkedMapOf(
            "sha256" to sha, "files" to shardFiles(dir).size.toLong(), "sequences" to s?.optInt("kept")?.toLong(),
            "tokens" to (s?.get("tokens") as? io.tlaloc.core.io.JsonNumber)?.raw?.toLong(),
            "masked_tokens" to (s?.get("masked_tokens") as? io.tlaloc.core.io.JsonNumber)?.raw?.toLong(),
            "trajectories_sha256" to s?.optString("trajectories_sha256"),
        )
    }
    out += shards
    return out
}
