package io.tlaloc.maestro.serving

/**
 * Goodput at a latency target — the arithmetic behind `:maestro:servedSloBench`.
 *
 * A serving benchmark that reports the median step time at a fixed concurrency
 * answers "how fast is one request", and cannot answer "how many users can this
 * deployment hold while p99 stays under a target". The second is what a
 * deployment is chosen on, and it is what Google's XLA Serving Engine publishes
 * ("Throughput under latency target"). This file is that question's arithmetic,
 * kept apart from the client that gathers the samples so it can be pinned by
 * tests that need no server and no device.
 *
 * ## What a phase is
 *
 * One round of the concurrency ladder: N sequences decoding at once, each
 * sending its next request as soon as its last answer is back (closed loop, so
 * the backend's own cohort batching is what forms the batches), all N starting
 * together. A phase yields a time per emitted token (TPOT) and a time to the
 * first token (TTFT) per sequence, plus the tokens, the wall time, and the
 * request and execution counters that say how much of it was batching.
 *
 * ## Percentiles are nearest-rank, and the sample count is part of the result
 *
 * [percentile] takes the ceil(p/100 * N)-th smallest sample. So p99 of 100
 * values is the 99th of them sorted, and ONE tail sample in 100 is outside it;
 * p99 of three samples is their maximum. A phase with few samples therefore
 * reports a p99 that is really a worst case, which is why every figure carries
 * [SloSummary.tpotSamples] beside it rather than standing alone.
 *
 * The rank is computed in integer arithmetic: `(p * n + 99) / 100`. A floating
 * `ceil(p / 100.0 * n)` disagrees with the exact rank for some (p, n) pairs, and
 * a percentile that is off by one sample is a different number, not a rounding.
 */

/** p50, p95 and p99 of one sample; each null when nothing was measured. */
data class LatencyPercentiles(val p50: Double?, val p95: Double?, val p99: Double?) {
    companion object {
        fun of(samples: List<Double>): LatencyPercentiles =
            LatencyPercentiles(percentile(samples, 50), percentile(samples, 95), percentile(samples, 99))
    }
}

/**
 * The nearest-rank percentile of [samples] at [p] (0..100), or null when the
 * sample is empty — an empty sample gives no number rather than a zero that
 * would read as a measurement.
 */
fun percentile(samples: List<Double>, p: Int): Double? {
    if (samples.isEmpty()) return null
    require(p in 0..100) { "percentile: p must be 0..100, got $p" }
    val sorted = samples.sorted()
    val rank = ((p.toLong() * sorted.size + 99) / 100).toInt()
    return sorted[(rank - 1).coerceIn(0, sorted.size - 1)]
}

/** The median, averaging the two middle values of an even-sized sample; null when empty. */
fun median(values: List<Double>): Double? {
    if (values.isEmpty()) return null
    val s = values.sorted()
    val mid = s.size / 2
    return if (s.size % 2 == 1) s[mid] else (s[mid - 1] + s[mid]) / 2
}

/**
 * One phase of the ladder: [concurrency] sequences, round [round], with the raw
 * samples. [tpotMs] holds one time per EMITTED token — a speculative step that
 * returns k tokens contributes its time k times, evenly shared, which is an
 * approximation the tokens-per-step figure makes visible. [errors] names every
 * refused or failed request: a phase with one is short, not clean.
 */
data class SloPhase(
    val concurrency: Int,
    val round: Int,
    val tpotMs: List<Double>,
    val ttftMs: List<Double>,
    val tokens: Int,
    val wallSeconds: Double,
    val requests: Int,
    val executions: Int,
    val peakMemoryGiB: Double? = null,
    val gpuMedianPercent: Double? = null,
    val gpuPeakPercent: Double? = null,
    val gpuSamples: Int = 0,
    val errors: List<String> = emptyList(),
) {
    val tpot: LatencyPercentiles get() = LatencyPercentiles.of(tpotMs)
    val ttft: LatencyPercentiles get() = LatencyPercentiles.of(ttftMs)

    /** All sequences' tokens over the phase's wall time. */
    val aggregateTokensPerSecond: Double? get() = if (wallSeconds > 0) tokens / wallSeconds else null

    /** Executions per request: below 1 means the backend batched sequences together. */
    val executionsPerRequest: Double? get() = if (requests > 0) executions.toDouble() / requests else null
}

/** A phase's figures with the samples reduced to percentiles — one row of a table. */
data class SloSummary(
    val concurrency: Int,
    val rounds: Int,
    val tpotSamples: Int,
    val ttftSamples: Int,
    val tpot: LatencyPercentiles,
    val ttft: LatencyPercentiles,
    val tokens: Double?,
    val wallSeconds: Double?,
    val aggregateTokensPerSecond: Double?,
    val requests: Double?,
    val executions: Double?,
    val executionsPerRequest: Double?,
)

/** One phase as a summary of its own samples. */
fun summarize(phase: SloPhase): SloSummary = SloSummary(
    concurrency = phase.concurrency,
    rounds = 1,
    tpotSamples = phase.tpotMs.size,
    ttftSamples = phase.ttftMs.size,
    tpot = phase.tpot,
    ttft = phase.ttft,
    tokens = phase.tokens.toDouble(),
    wallSeconds = phase.wallSeconds,
    aggregateTokensPerSecond = phase.aggregateTokensPerSecond,
    requests = phase.requests.toDouble(),
    executions = phase.executions.toDouble(),
    executionsPerRequest = phase.executionsPerRequest,
)

/**
 * One summary per concurrency from the rounds that measured it: the MEDIAN of
 * each figure, and the sample counts summed. A p99 of medians is not a p99, so
 * the caller keeps every round's [SloPhase] for anyone who wants the real
 * distribution; this is the table's row, not a claim about the pooled tail.
 */
fun mergeRounds(phases: List<SloPhase>): Map<Int, SloSummary> {
    fun med(values: List<Double?>): Double? = median(values.filterNotNull())
    return phases.groupBy { it.concurrency }.toSortedMap().mapValues { (n, rs) ->
        val s = rs.map { summarize(it) }
        SloSummary(
            concurrency = n,
            rounds = rs.size,
            tpotSamples = rs.sumOf { it.tpotMs.size },
            ttftSamples = rs.sumOf { it.ttftMs.size },
            tpot = LatencyPercentiles(med(s.map { it.tpot.p50 }), med(s.map { it.tpot.p95 }), med(s.map { it.tpot.p99 })),
            ttft = LatencyPercentiles(med(s.map { it.ttft.p50 }), med(s.map { it.ttft.p95 }), med(s.map { it.ttft.p99 })),
            tokens = med(s.map { it.tokens }),
            wallSeconds = med(s.map { it.wallSeconds }),
            aggregateTokensPerSecond = med(s.map { it.aggregateTokensPerSecond }),
            requests = med(s.map { it.requests }),
            executions = med(s.map { it.executions }),
            executionsPerRequest = med(s.map { it.executionsPerRequest }),
        )
    }
}

/** A latency target pair: p99 time per output token, and p99 time to the first token. */
data class SloTarget(val tpotP99Ms: Double, val ttftP99Seconds: Double)

/**
 * Does [s] meet [t]? A dimension with no samples is not a failure but is not
 * evidence either: it is skipped, and a summary that measured NEITHER dimension
 * meets nothing. Both targets are compared in the units the summary carries —
 * milliseconds for TPOT, and [t]'s seconds converted for TTFT.
 */
fun meets(s: SloSummary, t: SloTarget): Boolean {
    val checks = ArrayList<Boolean>()
    s.tpot.p99?.let { checks.add(it <= t.tpotP99Ms) }
    s.ttft.p99?.let { checks.add(it <= t.ttftP99Seconds * 1000) }
    return checks.isNotEmpty() && checks.all { it }
}

/**
 * One row of the goodput table: the highest concurrency that meets [target],
 * and the aggregate throughput there. Concurrency 0 and throughput 0 mean no
 * concurrency met it — a result, not a gap in the table.
 */
data class GoodputRow(
    val target: SloTarget,
    val concurrency: Int,
    val tokensPerSecond: Double,
    val measuredTpotP99Ms: Double?,
    val measuredTtftP99Ms: Double?,
)

/** The goodput table over the ladder's summaries, one row per target. */
fun goodputTable(summaries: Map<Int, SloSummary>, targets: List<SloTarget>): List<GoodputRow> =
    targets.map { t ->
        val best = summaries.keys.sorted().lastOrNull { n -> summaries.getValue(n).let { meets(it, t) } }
        val s = best?.let { summaries.getValue(it) }
        GoodputRow(
            target = t,
            concurrency = best ?: 0,
            tokensPerSecond = s?.aggregateTokensPerSecond ?: 0.0,
            measuredTpotP99Ms = s?.tpot?.p99,
            measuredTtftP99Ms = s?.ttft?.p99,
        )
    }

/**
 * One run of the benchmark at one setting of the swept knobs, for comparing
 * settings. [comparable] is false when a request failed or the GPU was busy:
 * such a run is reported but never recommended, because its numbers are not
 * comparable with an idle run's.
 */
data class SloRun(
    val label: String,
    val cohortWaitMicroseconds: Long?,
    val prefillBudget: String?,
    val summaries: Map<Int, SloSummary>,
    val comparable: Boolean = true,
    val whyNot: String? = null,
)

/**
 * The run with the highest goodput at [target] among the comparable ones; ties
 * go to the SMALLER cohort wait, because a lone sequence should not be made to
 * wait for company that never comes. Null when nothing comparable met it.
 */
fun recommend(runs: List<SloRun>, target: SloTarget): SloRun? =
    runs.filter { it.comparable }
        .mapNotNull { r ->
            val row = goodputTable(r.summaries, listOf(target)).first()
            if (row.concurrency == 0) null else r to row
        }
        .minWithOrNull(
            compareBy<Pair<SloRun, GoodputRow>> { -it.second.tokensPerSecond }
                .thenBy { it.first.cohortWaitMicroseconds ?: Long.MAX_VALUE },
        )?.first
