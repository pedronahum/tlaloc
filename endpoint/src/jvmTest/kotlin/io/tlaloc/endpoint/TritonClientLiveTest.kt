package io.tlaloc.endpoint

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * Against a running Triton: `TLALOC_TEST_TRITON=host:grpcPort:model` and
 * `TLALOC_TEST_TRITON_FIXTURE`, a JSON object of `[prompt ids, greedy ids, ...]` entries (the
 * `tiny_ripple/greedy.json` fixture, which triton/verify.sh serves). Skips without them.
 */
class TritonClientLiveTest {
    private val target = System.getenv("TLALOC_TEST_TRITON")?.split(":")
    private val fixture = System.getenv("TLALOC_TEST_TRITON_FIXTURE")

    private fun cases(): List<Pair<IntArray, IntArray>> {
        val o = parseJson(File(fixture!!).readText()) as JsonObject
        return o.fields.values.map { v ->
            val e = (v as JsonArray).elements
            (e[0] as JsonArray).asIntList("prompt").toIntArray() to (e[1] as JsonArray).asIntList("ids").toIntArray()
        }
    }

    private fun client(): TritonClient {
        assumeTrue(target != null && fixture != null, "set TLALOC_TEST_TRITON and TLALOC_TEST_TRITON_FIXTURE")
        return TritonClient(target!![0], target[1].toInt(), target[2])
    }

    private fun generated(c: TritonClient, id: Long, prompt: IntArray, n: Int): IntArray {
        val out = ArrayList<Int>()
        c.generate(id, prompt, start = true, parameters = mapOf("max_tokens" to TritonParameter.Int64(n.toLong()))).use { g ->
            while (true) out += (g.next(60_000) ?: break).toList()
        }
        c.end(id)
        return out.take(n).toIntArray()
    }

    @Test fun aGenerationStreamsTheGreedyIds() {
        client().use { c ->
            for ((i, case) in cases().take(2).withIndex()) {
                val (prompt, ids) = case
                assertContentEquals(ids, generated(c, 9_100L + i, prompt, ids.size))
            }
        }
    }

    @Test fun sequencesShareOneConnection() {
        client().use { c ->
            val cases = cases()
            val pool = Executors.newFixedThreadPool(cases.size)
            val results = cases.mapIndexed { i, (prompt, ids) -> pool.submit<IntArray> { generated(c, 9_200L + i, prompt, ids.size) } }
            for ((i, r) in results.withIndex()) assertContentEquals(cases[i].second, r.get())
            pool.shutdown()
        }
    }

    @Test fun aCancelledGenerationLeavesTheSequenceToEnd() {
        client().use { c ->
            val (prompt, ids) = cases().first()
            c.generate(9_300L, prompt, start = true, parameters = mapOf("max_tokens" to TritonParameter.Int64(2_000))).use { g ->
                assertTrue(g.next(60_000) != null)
            } // closed after the first step: cancelled
            c.end(9_300L)
            assertContentEquals(ids, generated(c, 9_301L, prompt, ids.size))
        }
    }
}
