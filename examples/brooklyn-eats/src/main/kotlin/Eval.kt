package brooklyn

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.time.Instant
import java.time.LocalDate
import kotlin.math.sqrt

// Evaluation of a served model on a split with snapshot tools: the agent loop on every offline task
// (or a seeded sample of them), each answer scored by the verifier, and the metrics of the run.
//
//   eval run     --config C --split dev|test [--run 1] [--n N] [--concurrency 4] [--endpoint URL] [--fresh]
//   eval metrics FILE.jsonl...         metrics of recorded episode files
//   eval spread  A.jsonl B.jsonl       two runs of the same tasks: verdicts that differ, identical trajectories
//   eval rescore FILE.jsonl...         score recorded episodes again with the current verifier (and their .json's metrics)
//   baseline     [report]              the baseline runs' table and build/data/baseline/summary.json
//   baseline pending [--all]           the baseline runs without current results ("config split run" lines)
//
// A run writes <data>/baseline/<config>/<split>-r<run>.jsonl (episodes, scored) and .json (settings,
// provenance, metrics). Episodes are appended to <...>.partial.jsonl as they end, so a stopped run
// resumes where it stopped; --fresh starts over.

const val BASELINE_DIR = "baseline"

/** Tasks per split in the quick scope (the full scope runs every offline task). */
const val QUICK_EVAL_N = 4

/** Sequences in flight: the serving configs' -PmaxBatch. */
const val EVAL_CONCURRENCY = 4

/** The baseline runs: per serving config, (split, run). TwIL's dev split runs twice for the run-to-run spread. */
val BASELINE_RUNS: Map<String, List<Pair<Split, Int>>> = linkedMapOf(
    "twil-bf16" to listOf(Split.DEV to 1, Split.TEST to 1, Split.DEV to 2),
    "ripple-nvfp4" to listOf(Split.DEV to 1, Split.TEST to 1),
)

/** The agent settings every evaluation uses: greedy, the agent stage's budgets, the eval seed. */
fun evalSettings(): AgentSettings = AgentSettings(seed = manifestSeed("eval"))

fun evalFile(data: Path, config: String, split: Split, run: Int, ext: String = "jsonl"): Path =
    data.resolve(BASELINE_DIR).resolve(config).resolve("${split.label}-r$run.$ext")

/** The tasks a run evaluates: every offline task of the split (quick: [QUICK_EVAL_N]), categories interleaved by seed. */
fun evalTasks(data: Path, split: Split, n: Int? = null): List<Task> {
    val all = readTasks(taskFile(data, split)).filter { it.offline }
    val k = n ?: if (dbScope(data.resolve("brooklyn.db")) == "quick") QUICK_EVAL_N else all.size
    return sampleTasks(all, k, manifestSeed("eval_sample"))
}

fun taskIdsSha256(ids: List<String>): String = sha256Hex(ids.joinToString("") { "$it\n" }.toByteArray())

private fun opt(args: List<String>, name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }

fun evalCmd(args: List<String>): Int {
    val sub = args.firstOrNull() ?: return usageEval()
    return when (sub) {
        "run" -> evalRun(args.drop(1))
        "metrics" -> {
            for (f in args.drop(1)) {
                val eps = readEpisodes(Paths.get(f))
                println("$f: ${toJson(evalMetrics(eps, null))}")
            }
            0
        }
        "rescore" -> evalRescore(args.drop(1))
        "spread" -> {
            if (args.size < 3) return usageEval()
            println(toJson(spread(readEpisodes(Paths.get(args[1])), readEpisodes(Paths.get(args[2])))))
            0
        }
        else -> usageEval()
    }
}

private fun usageEval(): Int {
    System.err.println(
        """usage: eval run --config C --split dev|test [--run 1] [--n N] [--concurrency 4] [--endpoint URL] [--fresh]
       eval metrics FILE.jsonl...
       eval spread A.jsonl B.jsonl
       eval rescore FILE.jsonl...
(the evaluation of tuned models on the test split is a later stage; this runs any served config on a split)""",
    )
    return NOT_BUILT
}

fun readEpisodes(f: Path): List<JsonObject> = Files.readAllLines(f).filter { it.isNotBlank() }.map { parseJson(it) as JsonObject }

/** Settings and inputs a results file was produced with; a run whose key differs is not reused. */
fun evalKey(config: Map<String, String>, configSha: String, settings: AgentSettings, ids: List<String>, concurrency: Int): Map<String, Any?> = linkedMapOf(
    "serving_config_sha256" to configSha, "model_key" to config["MODEL_KEY"], "draft_key" to config["DRAFT_KEY"],
    "settings" to settings.json(), "concurrency" to concurrency.toLong(), "tasks" to ids.size.toLong(), "task_ids_sha256" to taskIdsSha256(ids),
)

/** Whether [res] and [out] exist and [res] was produced with [key]. */
fun resultKeyIs(res: Path, out: Path, key: Map<String, Any?>): Boolean {
    if (!Files.exists(res) || !Files.exists(out)) return false
    val old = parseJson(Files.readString(res)) as JsonObject
    return canonicalJson(parseJson(toJson(old["key"]))) == canonicalJson(parseJson(toJson(key)))
}

/** The key a run of [config] on [split] would have now. */
fun currentEvalKey(home: Path, data: Path, config: String, split: Split): Map<String, Any?> =
    evalKey(servingConfig(home, config), fileSha256(home.resolve("serving").resolve("$config.env")), evalSettings(), evalTasks(data, split).map { it.id }, EVAL_CONCURRENCY)

fun evalCurrent(data: Path, config: String, split: Split, run: Int): Boolean =
    resultKeyIs(evalFile(data, config, split, run, "json"), evalFile(data, config, split, run), currentEvalKey(exampleHome(), data, config, split))

fun evalRun(args: List<String>): Int {
    val home = exampleHome()
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    val db = data.resolve("brooklyn.db").toAbsolutePath().normalize()
    if (!Files.exists(db)) { System.err.println("eval: $db not found; run the ingest stage first"); return 1 }
    val configName = opt(args, "--config") ?: return usageEval()
    val split = Split.parse(opt(args, "--split") ?: return usageEval())
    if (!Files.exists(taskFile(data, split))) { System.err.println("eval: ${taskFile(data, split)} not found; run the tasks stage first"); return 1 }
    val run = opt(args, "--run")?.toInt() ?: 1
    val config = servingConfig(home, configName)
    val configSha = fileSha256(home.resolve("serving").resolve("$configName.env"))
    val template = ChatTemplate.valueOf(config["TEMPLATE"] ?: "TWIL")
    val tokenizerDir = opt(args, "--tokenizer")?.let { Paths.get(it) } ?: modelDir(config.getValue("MODEL_KEY"))
    val concurrency = opt(args, "--concurrency")?.toInt() ?: EVAL_CONCURRENCY
    val endpoint = opt(args, "--endpoint") ?: "http://127.0.0.1:${System.getenv("BROOKLYN_ENDPOINT_PORT") ?: "8120"}/v1"
    val model = opt(args, "--model") ?: config.getValue("SERVED_NAME")
    val settings = evalSettings()
    val tasks = evalTasks(data, split, opt(args, "--n")?.toInt())
    val key = evalKey(config, configSha, settings, tasks.map { it.id }, concurrency)
    val out = evalFile(data, configName, split, run)
    val res = evalFile(data, configName, split, run, "json")
    val partial = evalFile(data, configName, split, run, "partial.jsonl")
    val partialKey = evalFile(data, configName, split, run, "partial.key")
    Files.createDirectories(out.parent)
    val fresh = "--fresh" in args
    if (!fresh && resultKeyIs(res, out, key)) {
        println("eval: $res is current (same config, settings and tasks); --fresh reruns it")
        return 0
    }
    // Episodes of an interrupted run with the same key are kept.
    val keyText = toJson(key)
    val done = LinkedHashMap<String, Map<String, Any?>>()
    var priorSeconds = 0.0
    if (!fresh && Files.exists(partial) && Files.exists(partialKey) && Files.readAllLines(partialKey).firstOrNull() == keyText) {
        @Suppress("UNCHECKED_CAST")
        // An episode the endpoint failed (stop "error: ...") runs again.
        for (e in readEpisodes(partial)) if (!e.str("stop").startsWith("error")) done[e.str("id")] = e.toKotlin() as Map<String, Any?>
        priorSeconds = Files.readAllLines(partialKey).getOrNull(1)?.toDoubleOrNull() ?: 0.0
        println("eval: resuming, ${done.size} episodes already recorded in $partial")
    } else {
        Files.deleteIfExists(partial)
        Files.writeString(partialKey, keyText + "\n0\n")
    }
    val tokenizer = HfTokenizer.load(tokenizerDir)
    val client = ChatClient(endpoint, model)
    val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
    println("eval: $configName ${split.label} run $run: ${tasks.size} tasks, $endpoint model $model, template $template, concurrency $concurrency")
    println("eval: settings ${toJson(settings.json())}")
    val started = Instant.now().toString()
    val t0 = System.nanoTime()
    val writer = Files.newBufferedWriter(partial, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    val scored = try {
        runAgentTasks(tasks, client, settings, concurrency, db, ToolMode.SNAPSHOT, tokenizer, template, systemPrompt(cutoff), cutoff, done) { ep ->
            writer.write(toJson(ep) + "\n")
            writer.flush()
            val s = priorSeconds + (System.nanoTime() - t0) / 1e9
            Files.writeString(partialKey, "$keyText\n$s\n")
        }
    } finally {
        writer.close()
    }
    val seconds = priorSeconds + (System.nanoTime() - t0) / 1e9
    Files.writeString(out, scored.joinToString("") { toJson(it) + "\n" })
    val eps = readEpisodes(out)
    val metrics = evalMetrics(eps, seconds)
    val result = linkedMapOf(
        "config" to configName, "model" to model, "split" to split.label, "run" to run.toLong(), "scope" to dbScope(db),
        "key" to key, "template" to template.name, "task_file_sha256" to fileSha256(taskFile(data, split)),
        "system_prompt_sha256" to sha256Hex(systemPrompt(cutoff).toByteArray()), "code" to codeCommit(home),
        "started" to started, "resumed" to done.isNotEmpty(), "seconds" to seconds,
        "trajectories_sha256" to trajectoriesSha256(out), "metrics" to metrics,
    )
    Files.writeString(res, prettyJson(result) + "\n")
    Files.deleteIfExists(partial)
    Files.deleteIfExists(partialKey)
    println("eval: wrote $out and $res")
    println("eval: ${metricsLine(metrics)}")
    return if (eps.any { it.str("stop").startsWith("error") }) 1 else 0
}

/**
 * Scores recorded episodes again with the current verifier (the split files under the data
 * directory give the tasks). A results file next to an episode file gets its metrics recomputed;
 * its wall time is kept.
 */
fun evalRescore(args: List<String>): Int {
    val home = exampleHome()
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    val db = data.resolve("brooklyn.db")
    val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
    val tasks = Split.entries.filter { Files.exists(taskFile(data, it)) }.flatMap { readTasks(taskFile(data, it)) }.associateBy { it.id }
    ToolServer(db, ToolMode.SNAPSHOT).use { s ->
        val v = Verifier(s, cutoff)
        for (f in args.filter { it.endsWith(".jsonl") }.map { Paths.get(it) }) {
            var changed = 0
            val lines = readEpisodes(f).map { ep ->
                @Suppress("UNCHECKED_CAST") val m = LinkedHashMap(ep.toKotlin() as Map<String, Any?>)
                val t = tasks[ep.str("id")] ?: error("$f: task ${ep.str("id")} is not in the split files of $data")
                val calls = (ep["calls"] as JsonArray).elements.map { c -> c as JsonObject; ToolCall(c.str("name"), c.str("arguments")) }
                val verdict = v.verify(t, ep.str("answer"), calls).json()
                if (toJson(verdict) != toJson(ep["verdict"])) changed++
                m["verdict"] = verdict
                toJson(m)
            }
            Files.writeString(f, lines.joinToString("") { it + "\n" })
            println("rescore: $f: ${lines.size} episodes, $changed verdicts changed")
            val res = f.resolveSibling(f.fileName.toString().removeSuffix(".jsonl") + ".json")
            if (Files.exists(res)) {
                @Suppress("UNCHECKED_CAST") val r = LinkedHashMap((parseJson(Files.readString(res)) as JsonObject).toKotlin() as Map<String, Any?>)
                val metrics = evalMetrics(readEpisodes(f), (r["seconds"] as? Number)?.toDouble())
                r["metrics"] = metrics
                r["trajectories_sha256"] = trajectoriesSha256(f)
                r["rescored_by"] = codeCommit(home)
                Files.writeString(res, prettyJson(r) + "\n")
                println("rescore: $res: ${metricsLine(metrics)}")
            }
        }
    }
    return 0
}

private fun JsonObject.obj(k: String): JsonObject = this[k] as JsonObject
private fun JsonObject.int(k: String): Int = optInt(k) ?: 0

private fun percentile(xs: List<Double>, p: Double): Double? {
    if (xs.isEmpty()) return null
    val s = xs.sorted()
    return s[((s.size - 1) * p).let { Math.round(it).toInt() }]
}

private fun ratio(a: Number, b: Number): Double? = if (b.toDouble() == 0.0) null else a.toDouble() / b.toDouble()

/**
 * Metrics of scored episodes:
 *   task_success      verifier passes / scored tasks (with its standard error)
 *   fact_recall       required facts stated / required facts
 *   fact_precision    required facts stated / (those + invented items + extra restaurants named): of the
 *                     claims the verifier checks, the share that is right
 *   invented_rate     answers with at least one invented item / scored tasks
 *   schema_valid_rate tool calls valid against their schema / tool calls
 *   calls_per_task    tool calls / tasks
 *   tokens_per_s      completion tokens / wall time of the run (all sequences in flight)
 *   stream_tokens_per_s  completion tokens / time spent in chat requests, per sequence (prefill included)
 *   latency           episode and turn wall times (ms)
 */
fun evalMetrics(eps: List<JsonObject>, seconds: Double?): Map<String, Any?> {
    val verdicts = eps.map { it["verdict"] as? JsonObject }
    val scored = verdicts.filterNotNull().filter { (it["scored"] as? JsonBool)?.value == true }
    val passed = scored.count { (it["pass"] as? JsonBool)?.value == true }
    fun listSize(v: JsonObject, k: String) = (v[k] as? JsonArray)?.elements?.size ?: 0
    var required = 0; var found = 0
    for (v in scored) {
        val (f, r) = (v["facts"] as JsonString).value.split('/').map { it.toInt() }
        found += f; required += r
    }
    val invented = scored.sumOf { listSize(it, "invented") }
    val extra = scored.sumOf { listSize(it, "extra") }
    val withInvented = scored.count { listSize(it, "invented") > 0 }
    val totals = eps.map { it.obj("totals") }
    val calls = totals.sumOf { it.int("calls") }
    val valid = totals.sumOf { it.int("valid_calls") }
    val noError = totals.sumOf { it.int("calls_without_error") }
    val completion = totals.sumOf { it.int("completion_tokens").toLong() }
    val prompt = totals.sumOf { it.int("prompt_tokens").toLong() }
    val turns = eps.flatMap { (it["turns"] as JsonArray).elements.map { t -> t as JsonObject } }
    val turnMs = turns.map { (it.obj("timing").optNumber("ms") ?: 0.0) }
    val epMs = eps.map { it.obj("timing").optNumber("ms") ?: 0.0 }
    val stops = eps.groupingBy { it.str("stop").substringBefore(':') }.eachCount().toSortedMap()
    val p = ratio(passed, scored.size)
    val perCategory = eps.indices.groupBy { eps[it].str("category") }.toSortedMap().mapValues { (_, ix) ->
        val vs = ix.mapNotNull { verdicts[it] }.filter { (it["scored"] as? JsonBool)?.value == true }
        "${vs.count { (it["pass"] as? JsonBool)?.value == true }}/${vs.size}"
    }
    return linkedMapOf(
        "tasks" to eps.size.toLong(), "scored" to scored.size.toLong(), "passed" to passed.toLong(),
        "task_success" to p, "task_success_se" to p?.let { sqrt(it * (1 - it) / scored.size) },
        "facts_required" to required.toLong(), "facts_found" to found.toLong(), "fact_recall" to ratio(found, required),
        "invented_items" to invented.toLong(), "extra_items" to extra.toLong(),
        "fact_precision" to ratio(found, found + invented + extra),
        "answers_with_invented" to withInvented.toLong(), "invented_rate" to ratio(withInvented, scored.size),
        "calls" to calls.toLong(), "schema_valid_calls" to valid.toLong(), "schema_valid_rate" to ratio(valid, calls),
        "calls_without_error" to noError.toLong(), "calls_per_task" to ratio(calls, eps.size),
        "tasks_without_calls" to totals.count { it.int("calls") == 0 }.toLong(),
        "stops" to stops.mapValues { it.value.toLong() },
        "turns" to turns.size.toLong(), "prompt_tokens" to prompt, "completion_tokens" to completion,
        "completion_tokens_per_task" to ratio(completion, eps.size),
        "seconds" to seconds, "tokens_per_s" to seconds?.let { ratio(completion, it) },
        "stream_tokens_per_s" to ratio(completion, turnMs.sum() / 1000.0),
        "episode_ms_median" to percentile(epMs, 0.5), "episode_ms_p90" to percentile(epMs, 0.9), "episode_ms_max" to epMs.maxOrNull(),
        "turn_ms_median" to percentile(turnMs, 0.5), "turn_ms_p90" to percentile(turnMs, 0.9),
        "per_category" to perCategory,
    )
}

fun metricsLine(m: Map<String, Any?>): String {
    fun pct(k: String) = (m[k] as? Double)?.let { "%.1f%%".format(it * 100) } ?: "-"
    return "success ${m["passed"]}/${m["scored"]} (${pct("task_success")}), fact precision ${pct("fact_precision")}, recall ${pct("fact_recall")}, " +
        "invented in ${pct("invented_rate")} of answers, schema-valid calls ${pct("schema_valid_rate")}, " +
        "${"%.2f".format(m["calls_per_task"] as? Double ?: 0.0)} calls/task, stops ${m["stops"]}, " +
        "${"%.1f".format(m["tokens_per_s"] as? Double ?: 0.0)} tokens/s, episode median ${"%.1f".format(((m["episode_ms_median"] as? Double) ?: 0.0) / 1000)} s"
}

/** Two runs of the same tasks: pass counts, verdicts that differ, identical trajectories and answers. */
fun spread(a: List<JsonObject>, b: List<JsonObject>): Map<String, Any?> {
    val bb = b.associateBy { it.str("id") }
    val common = a.filter { it.str("id") in bb }
    fun pass(e: JsonObject) = ((e["verdict"] as? JsonObject)?.get("pass") as? JsonBool)?.value == true
    val flips = common.filter { pass(it) != pass(bb.getValue(it.str("id"))) }
    val same = common.count { canonicalJson(normalizedEpisode(it)) == canonicalJson(normalizedEpisode(bb.getValue(it.str("id")))) }
    val sameAnswer = common.count { it.str("answer") == bb.getValue(it.str("id")).str("answer") }
    return linkedMapOf(
        "tasks" to common.size.toLong(), "passed_a" to common.count { pass(it) }.toLong(), "passed_b" to common.count { pass(bb.getValue(it.str("id"))) }.toLong(),
        "verdicts_differ" to flips.size.toLong(), "pass_to_fail" to flips.count { pass(it) }.toLong(), "fail_to_pass" to flips.count { !pass(it) }.toLong(),
        "identical_trajectories" to same.toLong(), "identical_answers" to sameAnswer.toLong(),
        "differing_tasks" to flips.map { it.str("id") },
    )
}

fun baselineCmd(args: List<String>): Int {
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    return when (args.firstOrNull() ?: "report") {
        "pending" -> {
            for ((cfg, runs) in BASELINE_RUNS) for ((sp, r) in runs) {
                if ("--all" in args || !evalCurrent(data, cfg, sp, r)) println("$cfg ${sp.label} $r")
            }
            0
        }
        "report" -> baselineReport(data)
        else -> { System.err.println("usage: baseline [report | pending [--all]] [--data DIR]"); 2 }
    }
}

/** The baseline table: one row per run, the spread of TwIL's two dev runs, and Ripple against TwIL. */
fun baselineReport(data: Path): Int {
    val rows = ArrayList<Map<String, Any?>>()
    var missing = 0
    for ((cfg, runs) in BASELINE_RUNS) for ((sp, r) in runs) {
        val f = evalFile(data, cfg, sp, r, "json")
        if (!Files.exists(f)) { println("missing: $f"); missing++; continue }
        @Suppress("UNCHECKED_CAST")
        rows += (parseJson(Files.readString(f)) as JsonObject).toKotlin() as Map<String, Any?>
    }
    fun pct(x: Any?) = (x as? Double)?.let { "%.1f".format(it * 100) } ?: "-"
    println("| Model | Split | Run | Tasks | Success % (± s.e.) | Fact precision % | Fact recall % | Answers with invented items % | Schema-valid calls % | Calls / task | Completion tokens / task | Tokens/s (all streams) | Episode median s | Episode p90 s |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (row in rows) {
        @Suppress("UNCHECKED_CAST") val m = row["metrics"] as Map<String, Any?>
        fun num(k: String, d: Int = 1, div: Double = 1.0) = (m[k] as? Number)?.let { "%.${d}f".format(it.toDouble() / div) } ?: "-"
        println("| ${row["model"]} | ${row["split"]} | ${row["run"]} | ${m["scored"]} | ${m["passed"]} = ${pct(m["task_success"])} ± ${pct(m["task_success_se"])} | " +
            "${pct(m["fact_precision"])} | ${pct(m["fact_recall"])} | ${pct(m["invented_rate"])} | ${pct(m["schema_valid_rate"])} | ${num("calls_per_task", 2)} | " +
            "${num("completion_tokens_per_task", 0)} | ${num("tokens_per_s")} | ${num("episode_ms_median", 1, 1000.0)} | ${num("episode_ms_p90", 1, 1000.0)} |")
    }
    val summary = linkedMapOf<String, Any?>("runs" to rows.map { linkedMapOf("config" to it["config"], "split" to it["split"], "run" to it["run"], "metrics" to it["metrics"]) })
    val a = evalFile(data, "twil-bf16", Split.DEV, 1)
    val b = evalFile(data, "twil-bf16", Split.DEV, 2)
    if (Files.exists(a) && Files.exists(b)) {
        val s = spread(readEpisodes(a), readEpisodes(b))
        summary["twil_dev_spread"] = s
        println("TwIL dev, run 1 against run 2: passed ${s["passed_a"]} / ${s["passed_b"]} of ${s["tasks"]}; verdicts differ on ${s["verdicts_differ"]} " +
            "(${s["pass_to_fail"]} pass -> fail, ${s["fail_to_pass"]} fail -> pass); identical trajectories ${s["identical_trajectories"]}, identical answers ${s["identical_answers"]}")
    }
    for (sp in listOf("dev", "test")) {
        val t = rows.firstOrNull { it["config"] == "twil-bf16" && it["split"] == sp && it["run"] == 1L }
        val rp = rows.firstOrNull { it["config"] == "ripple-nvfp4" && it["split"] == sp && it["run"] == 1L }
        if (t != null && rp != null) {
            val ts = (t["metrics"] as Map<*, *>)["task_success"] as Double
            val rs = (rp["metrics"] as Map<*, *>)["task_success"] as Double
            summary["ripple_minus_twil_$sp"] = rs - ts
            println("$sp: Ripple ${pct(rs)}% against TwIL ${pct(ts)}% task success (${if (rs > ts) "Ripple higher" else "Ripple NOT higher"})")
        }
    }
    Files.createDirectories(data.resolve(BASELINE_DIR))
    Files.writeString(data.resolve(BASELINE_DIR).resolve("summary.json"), prettyJson(summary) + "\n")
    return if (missing == 0) 0 else 1
}

/**
 * Manifest entries of the baseline stage. Inputs: Ripple's serving config and the evaluation
 * settings. Derived: Ripple's serving artifact and, per run, the tasks, the verifier's pass count and
 * the trajectories' SHA-256. Serving is not bitwise reproducible (sequences share batches in any
 * order), so verify compares a run's pass count within the number of verdicts that differed between
 * TwIL's two dev runs (pass_tolerance), and reports the other values without failing on them.
 */
fun baselineEntries(home: Path, data: Path): List<Entry> {
    val out = ArrayList<Entry>()
    out += Entry("baseline", "inputs", "serving_config_ripple", false) { servingConfigIdentity(home, "ripple-nvfp4") }
    out += Entry("baseline", "inputs", "eval_settings", false) {
        val cutoff = LocalDate.parse(readManifestJson(home.resolve("manifest.json"))?.optString("cutoff") ?: DEFAULT_CUTOFF)
        linkedMapOf(
            "system_prompt_sha256" to sha256Hex(systemPrompt(cutoff).toByteArray()),
            "settings" to toJson(AgentSettings().json().filterKeys { it != "seed" }), "concurrency" to EVAL_CONCURRENCY.toLong(),
            "tools" to "snapshot", "quick_tasks_per_split" to QUICK_EVAL_N.toLong(),
            "runs" to BASELINE_RUNS.entries.joinToString("; ") { (c, rs) -> "$c: " + rs.joinToString(", ") { "${it.first.label} r${it.second}" } },
        )
    }
    out += Entry("baseline", "derived", "serving_artifact_ripple", false) { servingArtifactIdentity(home, data, "ripple-nvfp4") }
    val db = data.resolve("brooklyn.db")
    fun tolerance(): Long? {
        val a = evalFile(data, "twil-bf16", Split.DEV, 1)
        val b = evalFile(data, "twil-bf16", Split.DEV, 2)
        return if (Files.exists(a) && Files.exists(b)) spread(readEpisodes(a), readEpisodes(b))["verdicts_differ"] as Long else null
    }
    val reported = setOf(
        "passed", "pass_tolerance", "trajectories_sha256", "task_success", "fact_precision", "invented_rate", "schema_valid_rate",
        "calls", "completion_tokens", "produced_by",
    )
    for ((cfg, runs) in BASELINE_RUNS) for ((sp, r) in runs) {
        lateinit var e: Entry
        val check = { want: JsonObject, now: Map<String, Any?> ->
            val w = want.optInt("passed")
            val tol = want.optInt("pass_tolerance") ?: 0
            val n = (now["passed"] as? Long)?.toInt()
            if (w != null && n != null && Math.abs(w - n) > tol) listOf("passed $w -> $n, more than pass_tolerance $tol") else emptyList()
        }
        e = Entry("baseline", "derived", "${cfg.replace('-', '_')}_${sp.label}_r$r", true, reported, check) {
            val res = evalFile(data, cfg, sp, r, "json")
            if (!Files.exists(res) || !Files.exists(evalFile(data, cfg, sp, r))) null else {
                val o = parseJson(Files.readString(res)) as JsonObject
                val key = o["key"] as JsonObject
                if (Files.exists(db) && Files.exists(taskFile(data, sp))) {
                    val now = currentEvalKey(home, data, cfg, sp)
                    if (canonicalJson(parseJson(toJson(now))) != canonicalJson(parseJson(toJson(key)))) e.stale = "produced with other settings, serving config or tasks; rerun the baseline stage"
                }
                val m = o["metrics"] as JsonObject
                linkedMapOf<String, Any?>(
                    "tasks" to (key["task_ids_sha256"] as JsonString).value, "episodes" to m.optInt("tasks")?.toLong(),
                    "settings" to sha256Hex(canonicalJson(key).toByteArray()),
                    "passed" to m.optInt("passed")?.toLong(), "pass_tolerance" to tolerance(),
                    "task_success" to m.optNumber("task_success"), "fact_precision" to m.optNumber("fact_precision"),
                    "invented_rate" to m.optNumber("invented_rate"), "schema_valid_rate" to m.optNumber("schema_valid_rate"),
                    "calls" to m.optInt("calls")?.toLong(), "completion_tokens" to m.optInt("completion_tokens")?.toLong(),
                    "trajectories_sha256" to o.optString("trajectories_sha256"),
                    "produced_by" to ((o["code"] as? JsonObject)?.optString("commit")),
                ).filterValues { it != null }
            }
        }
        out += e
    }
    return out
}
