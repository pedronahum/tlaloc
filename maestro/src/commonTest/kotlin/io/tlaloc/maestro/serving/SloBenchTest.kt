package io.tlaloc.maestro.serving

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The pins for [SloBench]'s arithmetic. These need no server and no device: the
 * numbers a deployment is chosen on must be right before any GPU is asked for
 * them, and a percentile that is off by one sample is a different number.
 *
 * The expectations here were first worked out against a prototype of the same
 * harness and are reproduced exactly: nearest-rank p99 of 1..100 is 99, one tail
 * sample in 100 is OUTSIDE p99, and a tie between two settings goes to the
 * smaller cohort wait.
 */
class SloBenchTest {

    private fun phase(
        n: Int,
        tpot: List<Double>,
        ttft: List<Double>,
        tokens: Int,
        wall: Double = 1.0,
        requests: Int = tokens,
        executions: Int = tokens,
        round: Int = 0,
        errors: List<String> = emptyList(),
    ) = SloPhase(n, round, tpot, ttft, tokens, wall, requests, executions, errors = errors)

    private fun summary(n: Int, tpotP99: Double, ttftP99: Double, tokensPerSecond: Double) =
        summarize(phase(n, List(100) { tpotP99 }, List(4) { ttftP99 }, (tokensPerSecond).toInt(), 1.0))

    @Test
    fun `nearest-rank percentile, and an empty sample is not a number`() {
        assertNull(percentile(emptyList(), 99))
        assertEquals(5.0, percentile(listOf(5.0), 99))
        assertEquals(99.0, percentile((1..100).map { it.toDouble() }, 99), "p99 of 1..100 is the 99th")
        assertEquals(50.0, percentile((1..100).map { it.toDouble() }, 50))
        assertEquals(3.0, percentile(listOf(1.0, 2.0, 3.0), 99), "p99 of three samples is their maximum")
        assertEquals(1.0, percentile((1..100).map { it.toDouble() }, 0))
        assertEquals(100.0, percentile((1..100).map { it.toDouble() }, 100))
        assertEquals(
            10.0, percentile(List(99) { 10.0 } + listOf(500.0), 99),
            "one tail sample in 100 is outside p99: a phase needs 100 samples for it to count",
        )
        assertFailsWith<IllegalArgumentException> { percentile(listOf(1.0), 101) }
    }

    @Test
    fun `the rank is exact at awkward sample counts`() {
        // ceil(p * n / 100) in integer arithmetic: the rank is a count of
        // samples, not a floating product that lands one sample away.
        assertEquals(100.0, percentile((1..101).map { it.toDouble() }, 99), "ceil(99 * 101 / 100) = 100")
        assertEquals(986.0, percentile((1..995).map { it.toDouble() }, 99), "ceil(99 * 995 / 100) = 986")
        assertEquals(19.0, percentile((1..20).map { it.toDouble() }, 95), "ceil(95 * 20 / 100) = 19")
        assertEquals(7.0, percentile(listOf(7.0), 50), "one sample is every percentile of itself")
    }

    @Test
    fun `median averages the two middle values of an even sample`() {
        assertNull(median(emptyList()))
        assertEquals(3.0, median(listOf(1.0, 3.0, 5.0)))
        assertEquals(20.0, median(listOf(10.0, 30.0)))
        assertEquals(3.0, median(listOf(4.0, 2.0)))
    }

    @Test
    fun `a phase's percentiles, throughput and batching ratio`() {
        val p = phase(1, List(90) { 10.0 } + List(10) { 500.0 }, List(4) { 100.0 }, 100, 2.0, 100, 25)
        assertEquals(500.0, p.tpot.p99)
        assertEquals(500.0, p.tpot.p95)
        assertEquals(10.0, p.tpot.p50)
        assertEquals(100.0, p.ttft.p99)
        assertEquals(50.0, p.aggregateTokensPerSecond)
        assertEquals(0.25, p.executionsPerRequest, "25 executions for 100 requests: the backend batched")
        val s = summarize(p)
        assertEquals(100, s.tpotSamples)
        assertEquals(4, s.ttftSamples)
        assertEquals(1, s.rounds)
    }

    @Test
    fun `one tail over its target fails the pair`() {
        val s = summarize(phase(1, List(90) { 10.0 } + List(10) { 500.0 }, List(4) { 100.0 }, 100, 2.0))
        assertTrue(meets(s, SloTarget(600.0, 0.2)), "500 ms p99 TPOT against 600, 100 ms TTFT against 200")
        assertFalse(meets(s, SloTarget(400.0, 0.2)), "the TPOT tail is over")
        assertFalse(meets(s, SloTarget(600.0, 0.05)), "the TTFT tail is over")
    }

    @Test
    fun `a phase that measured nothing meets nothing`() {
        val s = summarize(phase(1, emptyList(), emptyList(), 10, 1.0))
        assertNull(s.tpot.p99)
        assertNull(s.ttft.p99)
        assertFalse(meets(s, SloTarget(100.0, 1.0)))
    }

    @Test
    fun `a dimension with no samples is skipped, not failed`() {
        // A streamed turn measures TPOT but may report no first-token time.
        val s = summarize(phase(1, List(100) { 10.0 }, emptyList(), 100, 1.0))
        assertTrue(meets(s, SloTarget(50.0, 0.001)), "TPOT meets it; TTFT was not measured")
    }

    @Test
    fun `goodput is the highest concurrency that meets both targets`() {
        val ladder = mapOf(
            1 to summary(1, 10.0, 50.0, 100.0),
            2 to summary(2, 20.0, 60.0, 180.0),
            4 to summary(4, 90.0, 900.0, 300.0),
        )
        val targets = listOf(SloTarget(15.0, 1.0), SloTarget(50.0, 0.5), SloTarget(100.0, 1.0))
        val rows = goodputTable(ladder, targets).associateBy { it.target }
        assertEquals(1, rows.getValue(SloTarget(15.0, 1.0)).concurrency, "N=2's 20 ms misses a 15 ms target")
        assertEquals(100.0, rows.getValue(SloTarget(15.0, 1.0)).tokensPerSecond)
        assertEquals(2, rows.getValue(SloTarget(50.0, 0.5)).concurrency, "N=4 misses 50 ms and 0.5 s")
        assertEquals(180.0, rows.getValue(SloTarget(50.0, 0.5)).tokensPerSecond)
        assertEquals(4, rows.getValue(SloTarget(100.0, 1.0)).concurrency)
        assertEquals(300.0, rows.getValue(SloTarget(100.0, 1.0)).tokensPerSecond)
        assertEquals(90.0, rows.getValue(SloTarget(100.0, 1.0)).measuredTpotP99Ms)
    }

    @Test
    fun `an impossible target reports zero, not silence`() {
        val ladder = mapOf(1 to summary(1, 10.0, 50.0, 100.0))
        val row = goodputTable(ladder, listOf(SloTarget(5.0, 0.01))).single()
        assertEquals(0, row.concurrency)
        assertEquals(0.0, row.tokensPerSecond)
        assertNull(row.measuredTpotP99Ms)
    }

    @Test
    fun `rounds merge by median and their sample counts add up`() {
        val merged = mergeRounds(
            listOf(
                phase(2, List(10) { 10.0 }, listOf(5.0), 10, 1.0, 10, 5, round = 0),
                phase(2, List(10) { 30.0 }, listOf(7.0), 10, 1.0, 10, 5, round = 1),
                phase(1, List(10) { 8.0 }, listOf(4.0), 10, 1.0, 10, 10),
            ),
        )
        assertEquals(listOf(1, 2), merged.keys.toList())
        assertEquals(20.0, merged.getValue(2).tpot.p99, "the median of the two rounds' p99")
        assertEquals(2, merged.getValue(2).rounds)
        assertEquals(20, merged.getValue(2).tpotSamples, "counts add; a p99 of medians is not a p99")
        assertEquals(6.0, merged.getValue(2).ttft.p99)
    }

    private fun run(label: String, wait: Long?, tpotP99: Double, goodput: Double, comparable: Boolean = true) =
        SloRun(
            label = label,
            cohortWaitMicroseconds = wait,
            prefillBudget = null,
            summaries = mapOf(4 to summary(4, tpotP99, 100.0, goodput)),
            comparable = comparable,
            whyNot = if (comparable) null else "a request was refused",
        )

    @Test
    fun `the recommendation takes the higher goodput, and a tie the smaller wait`() {
        val a = run("cw=0", 0, 10.0, 180.0)
        val b = run("cw=20000", 20000, 20.0, 300.0)
        val target = SloTarget(50.0, 0.5)
        assertEquals(b, recommend(listOf(a, b), target), "the higher goodput wins")
        val tie = run("cw=0 tie", 0, 20.0, 300.0)
        assertEquals(tie, recommend(listOf(b, tie), target), "a tie goes to the smaller cohort wait")
    }

    @Test
    fun `a run with a refusal or a busy GPU is never recommended however fast`() {
        val good = run("cw=0", 0, 10.0, 180.0)
        val refused = run("cw=5000", 5000, 10.0, 900.0, comparable = false)
        val target = SloTarget(50.0, 0.5)
        assertEquals(good, recommend(listOf(good, refused), target))
        assertNull(recommend(listOf(refused), target))
    }

    @Test
    fun `a target nothing met recommends nothing`() {
        val none = SloRun("cw=0", 0, null, mapOf(4 to summarize(phase(4, emptyList(), emptyList(), 0, 1.0))))
        assertNull(recommend(listOf(none), SloTarget(50.0, 0.5)))
    }
}
