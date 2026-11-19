package brooklyn

import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class EvalTest {
    private fun episode(
        id: String, category: String, pass: Boolean, facts: String, invented: List<String> = emptyList(), extra: List<String> = emptyList(),
        calls: Int, valid: Int, tokens: Int, ms: Long, turnMs: List<Long>, stop: String = "answer", answer: String = "a",
    ): JsonObject = parseJson(
        toJson(
            linkedMapOf(
                "id" to id, "category" to category, "stop" to stop, "answer" to answer,
                "totals" to linkedMapOf("turns" to turnMs.size, "calls" to calls, "valid_calls" to valid, "calls_without_error" to valid, "prompt_tokens" to 1000, "completion_tokens" to tokens),
                "turns" to turnMs.map { linkedMapOf("completion_tokens" to tokens / turnMs.size, "timing" to linkedMapOf("ms" to it)) },
                "timing" to linkedMapOf("ms" to ms),
                "verdict" to linkedMapOf(
                    "id" to id, "scored" to true, "pass" to pass, "facts" to facts,
                    "invented" to invented.ifEmpty { null }, "extra" to extra.ifEmpty { null }, "calls" to calls,
                ),
            ),
        ),
    ) as JsonObject

    private val eps = listOf(
        episode("t1", "lookup", true, "2/2", calls = 2, valid = 2, tokens = 400, ms = 10_000, turnMs = listOf(4_000, 6_000)),
        episode("t2", "lookup", false, "1/2", invented = listOf("grade B"), calls = 3, valid = 2, tokens = 600, ms = 20_000, turnMs = listOf(5_000, 5_000, 10_000)),
        episode("t3", "filter", false, "0/1", extra = listOf("x", "y"), calls = 0, valid = 0, tokens = 200, ms = 30_000, turnMs = listOf(30_000), stop = "length"),
    )

    @Test
    fun metricsCountWhatTheVerifierAndTheEpisodesSay() {
        val m = evalMetrics(eps, 20.0)
        assertEquals(3L, m["scored"])
        assertEquals(1L, m["passed"])
        assertEquals(1.0 / 3, m["task_success"] as Double, 1e-12)
        assertEquals(3.0 / 5, m["fact_recall"] as Double, 1e-12)
        // 3 facts stated, 1 invented item, 2 extra restaurants.
        assertEquals(3.0 / 6, m["fact_precision"] as Double, 1e-12)
        assertEquals(1.0 / 3, m["invented_rate"] as Double, 1e-12)
        assertEquals(4.0 / 5, m["schema_valid_rate"] as Double, 1e-12)
        assertEquals(5.0 / 3, m["calls_per_task"] as Double, 1e-12)
        assertEquals(1L, m["tasks_without_calls"])
        assertEquals(1200L, m["completion_tokens"])
        assertEquals(60.0, m["tokens_per_s"] as Double, 1e-12)
        assertEquals(1200.0 / 60.0, m["stream_tokens_per_s"] as Double, 1e-12)
        assertEquals(20_000.0, m["episode_ms_median"])
        assertEquals(mapOf("answer" to 2L, "length" to 1L), m["stops"])
        assertEquals(mapOf("filter" to "0/1", "lookup" to "1/2"), m["per_category"])
        assertNull(evalMetrics(eps, null)["tokens_per_s"])
    }

    @Test
    fun spreadCountsVerdictsThatDiffer() {
        val again = listOf(
            eps[0],
            episode("t2", "lookup", true, "2/2", calls = 3, valid = 2, tokens = 600, ms = 1, turnMs = listOf(5_000, 5_000, 10_000), answer = "b"),
            episode("t3", "filter", false, "0/1", extra = listOf("x", "y"), calls = 0, valid = 0, tokens = 200, ms = 1, turnMs = listOf(30_000), stop = "length"),
        )
        val s = spread(eps, again)
        assertEquals(3L, s["tasks"])
        assertEquals(1L, s["passed_a"])
        assertEquals(2L, s["passed_b"])
        assertEquals(1L, s["verdicts_differ"])
        assertEquals(1L, s["fail_to_pass"])
        // Times are not part of the comparison: t1 and t3 are the same episodes.
        assertEquals(2L, s["identical_trajectories"])
        assertEquals(2L, s["identical_answers"])
        assertEquals(listOf("t2"), s["differing_tasks"])
    }

    @Test
    fun allTasksAreSampledWhenNIsTheirCount() {
        val tasks = (1..9).map { i ->
            Task("t$i", "dev", listOf("lookup", "filter", "menu")[i % 3], "k", "q$i", emptyList(), emptyList(), null, emptyList(), "", true, 0, "2026-11-13")
        }
        val s = sampleTasks(tasks, tasks.size, 6)
        assertEquals(tasks.map { it.id }.toSet(), s.map { it.id }.toSet())
        assertEquals(s.map { it.id }, sampleTasks(tasks, tasks.size, 6).map { it.id })
        // Categories alternate.
        assertEquals(3, s.take(3).map { it.category }.toSet().size)
    }

    @Test fun aCheckpointUnderBuildDataFollowsTheDataDirectory() {
        // BROOKLYN_DATA is unset under the test task: build/data is the data directory.
        val home = java.nio.file.Paths.get("/ex")
        assertEquals(home.resolve("build/data/train/merged"), ckptDir(home, "build/data/train/merged"))
        assertEquals(home.resolve("ckpt/other"), ckptDir(home, "ckpt/other"))
    }
}
