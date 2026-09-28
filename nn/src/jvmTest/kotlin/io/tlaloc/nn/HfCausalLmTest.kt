package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.inference.HfCheckpoint
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Qwen3-0.6B read into a [CausalLM] predicts what transformers generated
 * (the fixture `qwen3_0_6b_greedy.json`, transformers f32 eager attention),
 * and survives a save and reload.
 *
 * Self-skips without the checkpoint in the Hugging Face cache (or at
 * TLALOC_QWEN3_CHECKPOINT).
 */
class HfCausalLmTest {

    private val fixture = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "qwen3_0_6b_greedy.json")

    private fun checkpointDir(): Path? {
        System.getenv("TLALOC_QWEN3_CHECKPOINT")?.let { return Path.of(it) }
        val snapshots = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
        val rev = (parseJson(Files.readString(fixture)) as JsonObject).str("revision")
        return snapshots.resolve(rev).takeIf { Files.isRegularFile(it.resolve("model.safetensors")) }
    }

    /** The logits for [ids] `[1, n]`, from the traced forward. */
    private fun logits(model: CausalLM, ids: IntArray): FloatArray {
        val input = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(1, ids.size), I32)
        val params = model.parameters
        var out: Tracer<Shape>? = null
        captureN(listOf(input) + params.map { it.tensor }) { leaves ->
            val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
            model.forward(leaves[0], Params { byKey.getValue(it) }).also { out = it }
        }
        return FloatArray(out!!.size) { out!!.peek(it) }
    }

    @Test
    fun qwen3PredictsTransformersGreedyTokensAndRoundTripsThroughSave() {
        val dir = checkpointDir()
        assumeTrue(dir != null, "no Qwen/Qwen3-0.6B checkpoint in the Hugging Face cache")
        val prompt = ((parseJson(Files.readString(fixture)) as JsonObject).arr("prompts").elements[0] as JsonObject)
        fun ints(k: String) = prompt.arr(k).elements.map { (it as JsonNumber).value.toInt() }
        val promptIds = ints("promptTokens")
        val generated = ints("generatedTokens")
        val topValues = prompt.arr("step1TopKValues").elements.map { (it as JsonNumber).value.toFloat() }
        val topIds = ints("step1TopKIndices")

        val loaded = HfCausalLm.load(dir!!)
        val model = loaded.model
        assertTrue(model.head == null, "Qwen3-0.6B ties its head to the embedding table")
        assertTrue(model.blocks.all { it.attn.qNorm != null })

        // Teacher forcing: feeding transformers' own greedy tokens, the argmax at
        // every position from the last prompt token on must be the next one.
        val ids = (promptIds + generated.dropLast(1)).toIntArray()
        val v = loaded.config.vocabSize
        val out = logits(model, ids)
        val predicted = (promptIds.size - 1 until ids.size).map { pos ->
            var best = 0
            for (c in 1 until v) if (out[pos * v + c] > out[pos * v + best]) best = c
            best
        }
        assertEquals(generated, predicted)
        val first = (promptIds.size - 1) * v
        var worst = 0f
        for ((id, value) in topIds.zip(topValues)) worst = maxOf(worst, abs(out[first + id] - value))
        println("[hf-causal-lm] Qwen3-0.6B: 16/16 greedy tokens; top-20 step-1 logits max |diff| $worst")
        assertTrue(worst < 2e-3f, "step-1 logits differ from transformers by $worst")

        val saved = Files.createTempDirectory("tlaloc-qwen3-save-")
        try {
            loaded.save(dir, saved)
            HfCheckpoint.open(saved).use { assertEquals(loaded.config, it.config) }
            val reloaded = HfCausalLm.load(saved).model
            for ((a, b) in model.parameters.zip(reloaded.parameters)) {
                assertEquals(a.key, b.key)
                // The checkpoint is bf16, so a bf16 save is exact.
                assertContentEquals(a.tensor.hostF32(), b.tensor.hostF32(), a.key)
            }
            // Sharded write: a small shard size splits the file and the index reads back.
            val sharded = Files.createTempDirectory("tlaloc-qwen3-sharded-")
            try {
                loaded.save(dir, sharded, shardBytes = 400_000_000L)
                assertTrue(Files.isRegularFile(sharded.resolve("model.safetensors.index.json")))
                val again = HfCausalLm.load(sharded).model
                assertContentEquals(
                    model.parameters.last().tensor.hostF32(),
                    again.parameters.last().tensor.hostF32(),
                )
            } finally {
                sharded.toFile().deleteRecursively()
            }
        } finally {
            saved.toFile().deleteRecursively()
        }
    }
}
