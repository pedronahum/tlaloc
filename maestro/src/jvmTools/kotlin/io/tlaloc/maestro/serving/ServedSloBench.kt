package io.tlaloc.maestro.serving

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson
import io.tlaloc.endpoint.TritonClient
import io.tlaloc.endpoint.TritonParameter
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.time.OffsetDateTime
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * `:maestro:servedSloBench` — goodput at a latency target for a decoupled model served by
 * Triton, over [TritonClient] (the endpoint's own gRPC client, so the benchmark speaks the
 * same protocol the serving frontend does).
 *
 *     ./gradlew :maestro:servedSloBench --args="localhost:8001 twil http=localhost:8000 \
 *         concurrency=1,2,4 rounds=2 promptTokens=128 maxNew=32 json=build/slo/twil.json"
 *
 * For each concurrency N in the ladder, N sequences run at once, closed loop: a sequence
 * sends its next request as soon as its last answer is back, which is what makes the
 * backend's own cohort batching form the batches. All N start on a barrier, so the
 * concurrency is real rather than accidental. The ladder runs `rounds` times and each
 * concurrency's figures are the median across rounds ([mergeRounds]).
 *
 * `mode=step` (the default) sends a request per token and asks NEXT_TOKENS, so a request's
 * own time is its tokens' time. `mode=stream` sends one request a turn with `max_tokens`
 * and times each streamed response; a response carrying k tokens shares its time evenly
 * among them, an approximation the tokens-per-step figure makes visible.
 *
 * The arithmetic — percentiles, the SLO judgement, the goodput table, the recommendation
 * across settings — is [SloBench] in commonMain, pinned by SloBenchTest with no server and
 * no device. This file only gathers samples and reports them.
 *
 * `compare=a.json,b.json` reads results back and recommends a setting; it needs no server.
 * To sweep a knob that needs a restart (`cohort_wait_microseconds` in the model's
 * config.pbtxt, `TLALOC_PREFILL_BUDGET` in the container's environment), start the server
 * per value with triton/run_server.sh and run this task once per value.
 */

private const val RESPONSE_TIMEOUT_MS = 120_000L
private const val DEFAULT_COHORT_WAIT_MICROS = 20_000L

private class Options(args: Array<String>) {
    val grpc: String = args[0]
    val model: String = args[1]
    private val kv = args.drop(2).map { a ->
        val i = a.indexOf('=')
        require(i > 0) { "option '$a' is not key=value" }
        a.substring(0, i) to a.substring(i + 1)
    }.toMap()

    fun str(key: String, default: String): String = kv[key]?.takeIf { it.isNotEmpty() } ?: default
    fun int(key: String, default: Int): Int = kv[key]?.takeIf { it.isNotEmpty() }?.toInt() ?: default
    fun long(key: String, default: Long): Long = kv[key]?.takeIf { it.isNotEmpty() }?.toLong() ?: default
    fun double(key: String, default: Double): Double = kv[key]?.takeIf { it.isNotEmpty() }?.toDouble() ?: default
    fun bool(key: String, default: Boolean): Boolean = kv[key]?.toBooleanStrict() ?: default
    fun list(key: String): List<String> = kv[key]?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() } ?: emptyList()
    fun ints(key: String, default: List<Int>): List<Int> = list(key).map { it.toInt() }.ifEmpty { default }
    fun doubles(key: String, default: List<Double>): List<Double> = list(key).map { it.toDouble() }.ifEmpty { default }
    val all: Map<String, String> get() = kv
}

private fun httpJson(http: HttpClient, url: String): JsonObject {
    val r = http.send(HttpRequest.newBuilder(URI.create(url)).GET().build(), HttpResponse.BodyHandlers.ofString())
    check(r.statusCode() == 200) { "$url: HTTP ${r.statusCode()}: ${r.body().take(300)}" }
    return parseJson(r.body()) as JsonObject
}

private fun JsonObject.strOrNull(key: String): String? = (this[key] as? JsonString)?.value
private fun JsonObject.numOrNull(key: String): Double? = (this[key] as? JsonNumber)?.value
private fun JsonObject.objOrNull(key: String): JsonObject? = this[key] as? JsonObject
private fun JsonObject.arrOrNull(key: String): JsonArray? = this[key] as? JsonArray

/** What the server is and what the model is configured with: the provenance a result needs. */
private class ServerInfo(
    val version: String?,
    val model: String?,
    val maxBatchSize: Int?,
    val vocabulary: Int?,
    val decoupled: Boolean,
    val cohortWaitMicroseconds: Long,
    val cohortWaitIsDefault: Boolean,
    val parameters: Map<String, String>,
)

private fun serverInfo(http: HttpClient, base: String, model: String): ServerInfo {
    val version = runCatching { httpJson(http, "$base/v2").strOrNull("version") }.getOrNull()
        ?: error("no Triton server at $base/v2")
    val cfg = runCatching { httpJson(http, "$base/v2/models/$model/config") }.getOrElse {
        error("no model '$model' at $base: ${it.message}")
    }
    val params = cfg.objOrNull("parameters")?.fields?.mapNotNull { (k, v) ->
        (v as? JsonObject)?.strOrNull("string_value")?.let { k to it }
    }?.toMap() ?: emptyMap()
    // The vocabulary is the width of the LOGITS row: an id at or above it is not a token
    // of this model, and a prompt full of them measures nothing.
    val vocab = cfg.arrOrNull("output")?.elements?.mapNotNull { it as? JsonObject }
        ?.firstOrNull { it.strOrNull("name") == "LOGITS" }
        ?.arrOrNull("dims")?.elements?.lastOrNull()?.let { (it as? JsonNumber)?.asInt("LOGITS dims") }
    val named = params["cohort_wait_microseconds"]?.toLongOrNull()
    return ServerInfo(
        version, cfg.strOrNull("name"), cfg.numOrNull("max_batch_size")?.toInt(), vocab,
        cfg.objOrNull("model_transaction_policy")?.let { (it["decoupled"] as? JsonBool)?.value } ?: false,
        named ?: DEFAULT_COHORT_WAIT_MICROS, named == null, params,
    )
}

/** (inference_count, execution_count): requests seen, and batches run. */
private fun modelStats(http: HttpClient, base: String, model: String): Pair<Long, Long> {
    val s = httpJson(http, "$base/v2/models/$model/stats").arrOrNull("model_stats")!!.elements.first() as JsonObject
    return (s.numOrNull("inference_count")!!.toLong()) to (s.numOrNull("execution_count")!!.toLong())
}

private fun git(vararg args: String): String = runCatching {
    val p = ProcessBuilder(listOf("git") + args).redirectErrorStream(true).start()
    p.inputStream.bufferedReader().readText().trim().also { p.waitFor(15, TimeUnit.SECONDS) }
}.getOrDefault("")

/** GPU utilization in percent, polled while a phase runs; unavailable when nvidia-smi is absent. */
private class GpuSampler(private val periodMs: Long, enabled: Boolean) {
    private val values = ArrayList<Int>()
    private val stop = AtomicBoolean(false)
    private var thread: Thread? = null

    private fun query(): List<Int> = runCatching {
        val p = ProcessBuilder("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
            .redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText()
        p.waitFor(20, TimeUnit.SECONDS)
        out.trim().split("\n").mapNotNull { it.trim().toIntOrNull() }
    }.getOrDefault(emptyList())

    private val available: Boolean = enabled && query().isNotEmpty()

    fun start() {
        if (!available) return
        thread = Thread {
            while (!stop.get()) {
                synchronized(values) { values.addAll(query()) }
                Thread.sleep(periodMs)
            }
        }.apply { isDaemon = true; start() }
    }

    fun snapshot(): Triple<Int, Double?, Int?> {
        stop.set(true)
        thread?.join(5000)
        val v = synchronized(values) { values.toList() }
        return Triple(v.size, median(v.map { it.toDouble() }), v.maxOrNull())
    }

    /** The median over a window, as the serving architecture's protocol takes it before timing. */
    fun idleWindow(seconds: Double): Double? {
        if (!available || seconds <= 0) return null
        val t0 = System.nanoTime()
        val vals = ArrayList<Int>()
        while ((System.nanoTime() - t0) / 1e9 < seconds) {
            vals.addAll(query())
            Thread.sleep(periodMs)
        }
        return median(vals.map { it.toDouble() })
    }
}

/** Peak memory in use (GiB): MemTotal minus MemAvailable, which on a unified-memory GPU includes the device. */
private class MemPeak {
    @Volatile var peak: Double = used()
    private val stop = AtomicBoolean(false)
    private val thread = Thread {
        while (!stop.get()) {
            peak = max(peak, used())
            Thread.sleep(200)
        }
    }.apply { isDaemon = true; start() }

    fun stop(): Double {
        stop.set(true)
        thread.join(2000)
        return peak
    }

    companion object {
        fun used(): Double = runCatching {
            val v = HashMap<String, Long>()
            Files.readAllLines(Path.of("/proc/meminfo")).forEach { line ->
                val parts = line.split(":", limit = 2)
                if (parts.size == 2) v[parts[0]] = parts[1].trim().split(Regex("\\s+")).first().toLong()
            }
            (v.getValue("MemTotal") - v.getValue("MemAvailable")) / 1048576.0
        }.getOrDefault(0.0)
    }
}

private fun promptIds(seed: Int, n: Int, maxId: Int, avoid: Set<Int>): IntArray {
    val rng = Random(seed)
    val out = ArrayList<Int>(n)
    while (out.size < n) {
        val t = rng.nextInt(1, max(2, maxId))
        if (t !in avoid) out.add(t)
    }
    return out.toIntArray()
}

private fun runPhase(o: Options, info: ServerInfo, http: HttpClient?, base: String, n: Int, round: Int, corridBase: Long, avoid: Set<Int>): SloPhase {
    val (host, port) = o.grpc.split(":").let { it[0] to (it.getOrNull(1)?.toInt() ?: 8001) }
    val promptTokens = o.int("promptTokens", 128)
    val maxNew = o.int("maxNew", 32)
    val mode = o.str("mode", "step")
    val seed = o.int("seed", 7)
    val tpot = ArrayList<Double>()
    val ttft = ArrayList<Double>()
    val errors = ArrayList<String>()
    // One slot per thread: a shared counter incremented from N threads would lose
    // updates, and the aggregate tokens/s depends on it.
    val counts = IntArray(n)
    val barrier = CyclicBarrier(n + 1)
    val gpu = GpuSampler(o.long("samplePeriodMs", 500), o.bool("gpu", true))
    val mem = MemPeak()
    val before = http?.let { modelStats(http, base, o.model) }
    val pool = Executors.newFixedThreadPool(n)
    gpu.start()

    fun one(i: Int) {
        val sid = corridBase + round * 1000L + i
        val prompt = promptIds(seed * 100003 + round * 1009 + i, promptTokens, o.int("maxId", info.vocabulary ?: 0), avoid)
        val client = TritonClient(host, port, o.model)
        var count = 0
        try {
            barrier.await(120, TimeUnit.SECONDS)
            val t0 = System.nanoTime()
            if (mode == "stream") {
                client.generate(sid, prompt, start = true, end = true,
                    parameters = mapOf("max_tokens" to TritonParameter.Int64(maxNew.toLong()))).use { g ->
                    var prev = 0L
                    var first = true
                    while (true) {
                        val toks = g.next(RESPONSE_TIMEOUT_MS) ?: break
                        val t = System.nanoTime()
                        if (first) {
                            // The first response's interval IS the time to first token: it
                            // contains the prefill. Counting it as a per-token time would
                            // put the whole prefill into the TPOT tail.
                            synchronized(ttft) { ttft.add((t - t0) / 1e6) }
                            first = false
                        } else {
                            // A step's reply may carry several tokens (a speculative model),
                            // so its interval is shared evenly among them.
                            val k = max(1, toks.size)
                            val each = (t - prev) / 1e6 / k
                            synchronized(tpot) { repeat(k) { tpot.add(each) } }
                        }
                        prev = t
                        count += toks.size
                    }
                }
                if (count == 0) synchronized(errors) { errors.add("sequence $sid: the stream ended without a token") }
            } else {
                // A request per token: the request's own time is its tokens' time.
                // max_tokens=1 is what makes the backend emit a token at all — without
                // it a decoupled request only appends the ids and answers with an empty
                // NEXT_TOKENS, which is a prefill and not a step.
                val oneToken = mapOf("max_tokens" to TritonParameter.Int64(1))
                var next: Int
                client.generate(sid, prompt, start = true, end = maxNew == 1, parameters = oneToken).use { g ->
                    val toks = g.next(RESPONSE_TIMEOUT_MS)?.takeIf { it.isNotEmpty() }
                        ?: error("the prompt request returned no token")
                    synchronized(ttft) { ttft.add((System.nanoTime() - t0) / 1e6) }
                    count += toks.size
                    next = toks.last()
                }
                for (k in 1 until maxNew) {
                    val t1 = System.nanoTime()
                    client.generate(sid, intArrayOf(next), start = false, end = k == maxNew - 1, parameters = oneToken).use { g ->
                        val toks = g.next(RESPONSE_TIMEOUT_MS)?.takeIf { it.isNotEmpty() }
                            ?: error("no token at step $k")
                        val dt = (System.nanoTime() - t1) / 1e6
                        val each = dt / toks.size
                        synchronized(tpot) { repeat(toks.size) { tpot.add(each) } }
                        count += toks.size
                        next = toks.last()
                    }
                }
            }
        } catch (e: Exception) {
            synchronized(errors) { errors.add("sequence $sid: ${e::class.simpleName}: ${e.message?.take(200)}") }
            runCatching { client.end(sid) }
        } finally {
            // Keep what the sequence did emit: a phase with a refusal reports short
            // figures, and dropping them would report a zero instead.
            counts[i] = count
            runCatching { client.close() }
        }
    }

    repeat(n) { i -> pool.submit { one(i) } }
    barrier.await(120, TimeUnit.SECONDS)
    val t0 = System.nanoTime()
    pool.shutdown()
    pool.awaitTermination(1, TimeUnit.HOURS)
    val wall = (System.nanoTime() - t0) / 1e9
    val (gpuSamples, gpuMedian, gpuPeak) = gpu.snapshot()
    val peakMem = mem.stop()
    val after = http?.let { modelStats(http, base, o.model) }
    // Triton's counters are Long; a phase that ran more than Int.MAX_VALUE requests is
    // not a benchmark anyone will read, so the difference is carried as an Int.
    return SloPhase(
        concurrency = n, round = round,
        tpotMs = synchronized(tpot) { tpot.toList() }, ttftMs = synchronized(ttft) { ttft.toList() },
        tokens = counts.sum(), wallSeconds = wall,
        requests = ((after?.first ?: 0L) - (before?.first ?: 0L)).toInt(),
        executions = ((after?.second ?: 0L) - (before?.second ?: 0L)).toInt(),
        peakMemoryGiB = peakMem, gpuMedianPercent = gpuMedian, gpuPeakPercent = gpuPeak?.toDouble(), gpuSamples = gpuSamples,
        errors = synchronized(errors) { errors.toList() },
    )
}

private fun f(v: Double?, unit: String = ""): String = if (v == null || !v.isFinite()) "n/a" else "%.1f%s".format(v, unit)

private fun printPhase(p: SloPhase) {
    // Utilization DURING a phase is the model working, not contention: a decode
    // step keeps the GPU near 100%. Contention is other work on the device, and
    // the only place it can be seen is the idle window before the ladder, which
    // main() reports and the result records. Labelling a busy phase CONTENDED
    // would mark every clean run unusable.
    val busy = if (p.gpuSamples > 0) "gpu median ${f(p.gpuMedianPercent)}% peak ${f(p.gpuPeakPercent)}%" else "gpu not sampled"
    val weak = if (p.gpuSamples in 1..2) " (${p.gpuSamples} samples: shorter than the sampling period)" else ""
    println(
        "N=${p.concurrency} round ${p.round}: TPOT p50/p95/p99 ${f(p.tpot.p50)}/${f(p.tpot.p95)}/${f(p.tpot.p99)} ms " +
            "(${p.tpotMs.size} samples) | TTFT p99 ${f(p.ttft.p99)} ms (${p.ttftMs.size}) | " +
            "${f(p.aggregateTokensPerSecond)} tok/s in all | ${p.requests} requests in ${p.executions} executions | " +
            "peak ${f(p.peakMemoryGiB)} GiB | $busy$weak",
    )
    p.errors.forEach { println("    FAILED $it") }
}

private fun jsonNum(v: Double?): String = if (v == null || !v.isFinite()) "null" else v.toString()
private fun jsonNum(v: Int?): String = v?.toString() ?: "null"
private fun jsonNum(v: Long?): String = v?.toString() ?: "null"

private fun percentilesJson(p: LatencyPercentiles): String =
    """{"p50":${jsonNum(p.p50)},"p95":${jsonNum(p.p95)},"p99":${jsonNum(p.p99)}}"""

private fun summaryJson(s: SloSummary): String = buildString {
    append("""{"concurrency":${s.concurrency},"rounds":${s.rounds},"tpotSamples":${s.tpotSamples},""")
    append(""""ttftSamples":${s.ttftSamples},"tpotMs":${percentilesJson(s.tpot)},"ttftMs":${percentilesJson(s.ttft)},""")
    append(""""tokens":${jsonNum(s.tokens)},"wallSeconds":${jsonNum(s.wallSeconds)},""")
    append(""""aggregateTokensPerSecond":${jsonNum(s.aggregateTokensPerSecond)},"requests":${jsonNum(s.requests)},""")
    append(""""executions":${jsonNum(s.executions)},"executionsPerRequest":${jsonNum(s.executionsPerRequest)}}""")
}

private fun phaseJson(p: SloPhase): String = buildString {
    append(summaryJson(summarize(p)).dropLast(1))
    append(""","round":${p.round},"peakMemoryGiB":${jsonNum(p.peakMemoryGiB)},""")
    append(""""gpu":{"samples":${p.gpuSamples},"median":${jsonNum(p.gpuMedianPercent)},"peak":${jsonNum(p.gpuPeakPercent)}},""")
    append(""""errors":[${p.errors.joinToString(",") { jsonQuote(it) }}]}""")
}

private fun targets(o: Options): List<SloTarget> {
    val tpot = o.doubles("sloTpotMs", listOf(50.0, 100.0, 200.0, 400.0))
    val ttft = o.doubles("sloTtftS", listOf(0.5, 2.0, 8.0))
    return tpot.flatMap { t -> ttft.map { s -> SloTarget(t, s) } }
}

private fun printGoodput(rows: List<GoodputRow>, label: String, rounds: Int, commit: String, dirty: Boolean) {
    println("\n== goodput${if (label.isEmpty()) "" else " — $label"} ($rounds rounds; commit ${commit.take(8)}${if (dirty) ", dirty tree" else ""})")
    println("%12s %11s %4s %14s   measured p99".format("p99 TPOT ms", "p99 TTFT s", "N", "goodput tok/s"))
    rows.forEach { r ->
        val extra = if (r.concurrency == 0) "  no concurrency met it"
        else "  TPOT ${f(r.measuredTpotP99Ms)} ms, TTFT ${f(r.measuredTtftP99Ms)} ms"
        println("%12.0f %11.1f %4d %14.2f%s".format(r.target.tpotP99Ms, r.target.ttftP99Seconds, r.concurrency, r.tokensPerSecond, extra))
    }
}

/** Reads a result file back as a run, for `compare=`. */
private fun readRun(file: Path): SloRun {
    val doc = parseJson(Files.readString(file)) as JsonObject
    val server = doc.objOrNull("server")
    val opts = doc.objOrNull("options")
    val by = doc.objOrNull("byConcurrency") ?: error("$file: no byConcurrency")
    val summaries = by.fields.mapKeys { it.key.toInt() }.mapValues { (_, v) ->
        val s = v as JsonObject
        fun lat(key: String) = s.objOrNull(key)?.let {
            LatencyPercentiles(it.numOrNull("p50"), it.numOrNull("p95"), it.numOrNull("p99"))
        } ?: LatencyPercentiles(null, null, null)
        SloSummary(
            concurrency = s.numOrNull("concurrency")!!.toInt(), rounds = s.numOrNull("rounds")!!.toInt(),
            tpotSamples = s.numOrNull("tpotSamples")!!.toInt(), ttftSamples = s.numOrNull("ttftSamples")!!.toInt(),
            tpot = lat("tpotMs"), ttft = lat("ttftMs"),
            tokens = s.numOrNull("tokens"), wallSeconds = s.numOrNull("wallSeconds"),
            aggregateTokensPerSecond = s.numOrNull("aggregateTokensPerSecond"),
            requests = s.numOrNull("requests"), executions = s.numOrNull("executions"),
            executionsPerRequest = s.numOrNull("executionsPerRequest"),
        )
    }
    val contendedAt = 5.0
    val phases = doc.arrOrNull("phases")?.elements?.mapNotNull { it as? JsonObject } ?: emptyList()
    val refused = phases.any { (it.arrOrNull("errors")?.size ?: 0) > 0 }
    // Only the idle window before the ladder says whether something ELSE held the
    // GPU. A phase's own utilization is the model working and is not contention.
    val idleBusy = doc.numOrNull("idleGpuUtilizationMedianPercent")?.let { it > contendedAt } ?: false
    val why = when {
        refused -> "a request failed or was refused"
        idleBusy -> "the GPU was busy before the ladder"
        else -> null
    }
    return SloRun(
        label = opts?.strOrNull("label") ?: file.fileName.toString(),
        cohortWaitMicroseconds = server?.numOrNull("cohortWaitMicroseconds")?.toLong(),
        prefillBudget = opts?.strOrNull("prefillBudget"),
        summaries = summaries, comparable = why == null, whyNot = why,
    )
}

private fun compare(files: List<String>, targets: List<SloTarget>): Int {
    val runs = files.map { readRun(Path.of(it)) }
    runs.forEach { r ->
        println("\n== ${r.label} — cohort_wait ${r.cohortWaitMicroseconds ?: "unknown"} us, " +
            "prefill budget ${r.prefillBudget ?: "(unset)"} — ${if (r.comparable) "comparable" else "NOT comparable: ${r.whyNot}"}")
        println("%3s %6s %12s %12s %13s %9s".format("N", "rounds", "p99 TPOT ms", "p99 TTFT ms", "tok/s in all", "exec/req"))
        r.summaries.toSortedMap().forEach { (n, s) ->
            println("%3d %6d %12s %12s %13s %9s".format(n, s.rounds, f(s.tpot.p99), f(s.ttft.p99), f(s.aggregateTokensPerSecond), f(s.executionsPerRequest)))
        }
    }
    println("\n== recommendation")
    var any = false
    targets.forEach { t ->
        val best = recommend(runs, t)
        if (best == null) {
            println("   p99 TPOT ${t.tpotP99Ms} ms, TTFT ${t.ttftP99Seconds} s: nothing comparable met it")
        } else {
            any = true
            val row = goodputTable(best.summaries, listOf(t)).first()
            println("   p99 TPOT ${t.tpotP99Ms} ms, TTFT ${t.ttftP99Seconds} s: ${best.label} — " +
                "cohort_wait ${best.cohortWaitMicroseconds ?: "unknown"} us, N=${row.concurrency}, " +
                "%.2f tok/s (measured p99 TPOT ${f(row.measuredTpotP99Ms)} ms, TTFT ${f(row.measuredTtftP99Ms)} ms)".format(row.tokensPerSecond))
        }
    }
    return if (any) 0 else 1
}

fun main(args: Array<String>) {
    require(args.size >= 2) {
        "usage: <grpc host:port> <model> [http=host:port] [concurrency=1,2,4,8] [rounds=3] " +
            "[promptTokens=128] [maxNew=64] [mode=step|stream] [sloTpotMs=50,100,200,400] " +
            "[sloTtftS=0.5,2,8] [maxId=] [avoidIds=] [seed=7] [label=] [json=FILE] " +
            "[idleWindowS=10] [contendedAt=5] [gpu=true] [compare=a.json,b.json]"
    }
    val o = Options(args)
    val tg = targets(o)
    val compareFiles = o.list("compare")
    if (compareFiles.isNotEmpty()) {
        kotlin.system.exitProcess(compare(compareFiles, tg))
    }

    val base = "http://${o.str("http", "localhost:8000")}"
    val http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).build()
    val info = serverInfo(http, base, o.model)
    require(info.decoupled) {
        "model '${o.model}' is not decoupled: this harness streams through TritonClient, which a " +
            "non-decoupled model does not support. Export it with -Pdecoupled=true, or measure it " +
            "with triton/multi_user_bench.py"
    }
    val maxId = o.int("maxId", info.vocabulary ?: 0)
    require(maxId > 1) { "the model's vocabulary is unknown and maxId was not given" }
    require(info.vocabulary == null || maxId <= info.vocabulary) {
        "maxId $maxId is above the model's vocabulary of ${info.vocabulary}: an id at or above it is not a token of this model"
    }
    val avoid = o.list("avoidIds").map { it.toInt() }.toSet()
    val ladder = o.ints("concurrency", listOf(1, 2, 4, 8))
    val rounds = o.int("rounds", 3)
    val contendedAt = o.double("contendedAt", 5.0)
    val commit = git("rev-parse", "HEAD")
    val dirty = git("status", "--porcelain").isNotEmpty()
    val corridBase = o.long("firstCorrid", (System.currentTimeMillis() / 1000 % 100000) * 1000 + 5000)

    println("server ${info.version} model ${o.model} mode ${o.str("mode", "step")} " +
        "prompt ${o.int("promptTokens", 128)} tokens, ${o.int("maxNew", 32)} new, ladder $ladder, $rounds round(s)")
    println("vocabulary ${info.vocabulary}, cohort_wait_microseconds ${info.cohortWaitMicroseconds}" +
        if (info.cohortWaitIsDefault) " (the backend default; this model does not set it)" else "")
    info.maxBatchSize?.let { mbs ->
        if (ladder.max() > mbs) println("note: the ladder goes to ${ladder.max()} but max_batch_size is $mbs, " +
            "so sequences beyond one batch wait for it — and a short KV pool refuses them by name")
    }
    val idle = GpuSampler(o.long("samplePeriodMs", 500), o.bool("gpu", true)).idleWindow(o.double("idleWindowS", 10.0))
    if (idle != null) {
        println("GPU utilization before the ladder: median ${f(idle)}% over ${o.double("idleWindowS", 10.0)} s — " +
            if (idle <= contendedAt) "idle" else "CONTENDED: something else holds the GPU; these numbers are not comparable with an idle run")
    }

    val phases = ArrayList<SloPhase>()
    for (r in 0 until rounds) for (n in ladder) {
        val p = runPhase(o, info, http, base, n, r, corridBase, avoid)
        printPhase(p)
        phases.add(p)
    }
    val summaries = mergeRounds(phases)
    val rows = goodputTable(summaries, tg)
    printGoodput(rows, o.str("label", ""), rounds, commit, dirty)
    val failed = phases.count { it.errors.isNotEmpty() }
    if (failed > 0) println("\nFAILED: $failed of ${phases.size} phases had a refused or failed request; " +
        "their figures are short, and the goodput above is not a clean result")

    o.str("json", "").takeIf { it.isNotEmpty() }?.let { out ->
        val path = Path.of(out)
        path.parent?.let { Files.createDirectories(it) }
        // Each piece is built separately. Nesting a joinToString lambda inside a raw
        // string makes the closing braces impossible to count by eye, and two objects
        // went unclosed exactly that way; the reader below is what caught it.
        val provenance = """{"date":${jsonQuote(OffsetDateTime.now().toString())},"host":${jsonQuote(System.getProperty("os.name"))},""" +
            """"machine":${jsonQuote(System.getProperty("os.arch"))},"commit":${jsonQuote(commit)},"dirty":$dirty,""" +
            """"branch":${jsonQuote(git("rev-parse", "--abbrev-ref", "HEAD"))}}"""
        val params = info.parameters.entries.joinToString(",") { (k, v) -> "${jsonQuote(k)}:${jsonQuote(v)}" }
        val server = """{"serverVersion":${jsonQuote(info.version ?: "")},"model":${jsonQuote(info.model ?: "")},""" +
            """"maxBatchSize":${jsonNum(info.maxBatchSize)},"vocabulary":${jsonNum(info.vocabulary)},"decoupled":${info.decoupled},""" +
            """"cohortWaitMicroseconds":${info.cohortWaitMicroseconds},"cohortWaitIsBackendDefault":${info.cohortWaitIsDefault},""" +
            """"parameters":{$params}}"""
        val options = o.all.entries.joinToString(",") { (k, v) -> "${jsonQuote(k)}:${jsonQuote(v)}" }
        val byConcurrency = summaries.entries.joinToString(",") { (k, v) -> "${jsonQuote(k.toString())}:${summaryJson(v)}" }
        val goodput = rows.joinToString(",") { r ->
            """{"tpotP99Ms":${r.target.tpotP99Ms},"ttftP99Seconds":${r.target.ttftP99Seconds},"concurrency":${r.concurrency},""" +
                """"goodputTokensPerSecond":${r.tokensPerSecond},"tpotP99MeasuredMs":${jsonNum(r.measuredTpotP99Ms)},""" +
                """"ttftP99MeasuredMs":${jsonNum(r.measuredTtftP99Ms)}}"""
        }
        val doc = """{"provenance":$provenance,"server":$server,"options":{$options},""" +
            """"idleGpuUtilizationMedianPercent":${jsonNum(idle)},"phases":[${phases.joinToString(",") { phaseJson(it) }}],""" +
            """"byConcurrency":{$byConcurrency},"goodput":[$goodput]}"""
        // The writer is checked by the reader: a result this program cannot parse
        // back is a result nobody can compare. The refusal names the region, because
        // "unexpected end of input" alone does not say which field broke it.
        try {
            parseJson(doc)
        } catch (e: io.tlaloc.core.io.JsonException) {
            val at = Regex("offset (\\d+)").find(e.message ?: "")?.groupValues?.get(1)?.toIntOrNull() ?: 0
            System.err.println("the result JSON is malformed near offset $at of ${doc.length}:")
            System.err.println(if (doc.length <= 20000) doc else doc.substring(max(0, at - 240), min(doc.length, at + 240)))
            throw e
        }
        Files.writeString(path, doc)
        println("\nwrote $out")
    }
    kotlin.system.exitProcess(if (failed > 0) 1 else 0)
}
