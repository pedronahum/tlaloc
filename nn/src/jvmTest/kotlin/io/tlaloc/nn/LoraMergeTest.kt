package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.inference.HfDecoderConfig
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Merging for export: [Lora.merge] of a trained adapter, written with
 * [HfCausalLm.save] and read back with [HfCausalLm.load], is an ordinary
 * checkpoint whose logits are the adapted model's.
 */
class LoraMergeTest {

    private val tmp = Files.createTempDirectory("tlaloc-lora-merge-")

    @AfterTest
    fun cleanup() {
        tmp.toFile().deleteRecursively()
    }

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

    private fun maxDiff(a: FloatArray, b: FloatArray) = a.indices.maxOf { abs(a[it] - b[it]) }

    @Test
    fun aTrainedAdapterMergedSavedAndReloadedGivesTheAdaptedLogits() {
        val json = """{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 16,
            "intermediate_size": 24, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "num_hidden_layers": 2, "vocab_size": 13, "rms_norm_eps": 1e-06, "rope_theta": 10000,
            "tie_word_embeddings": true, "hidden_act": "silu"}"""
        val src = Files.createDirectories(tmp.resolve("src"))
        Files.writeString(src.resolve("config.json"), json)
        val config = HfDecoderConfig.parse(json)
        val base = CausalLM.llama(
            CausalLmConfig(13, 16, 2, 4, 24, numKvHeads = 2, headDim = 4, normEps = 1e-6f, tiedEmbeddings = true, qkNorm = true, initStd = 0.3f),
            RandomKey.fromSeed(8),
        )
        var model = Lora.apply(base, LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(9))
        val tokens = intArrayOf(1, 5, 2, 7, 3, 12, 0, 4)
        val input = DTensor<Shape, I32>(HostI32Storage(tokens.copyOfRange(0, 7)), intArrayOf(1, 7), I32)
        val targets = oneHot(tokens.copyOfRange(1, 8), 13, intArrayOf(1, 7))
        val step = capture(model, listOf(input), listOf(targets), Lora.frozen) { l, t -> crossEntropy(l, t[0]) }
        val opt = AdamW(learningRate = 0.05f)
        var state = opt.initialState()
        repeat(15) {
            val r = step.run(model, listOf(input, targets))
            val (m, s) = opt.step(model, r.gradients, state, Lora.frozen)
            model = m
            state = s
        }

        val merged = Lora.merge(model)
        HfCausalLm(merged, config).save(src, tmp.resolve("out"), dtype = F32)
        val reloaded = HfCausalLm.load(tmp.resolve("out")).model
        for ((a, b) in merged.parameters.zip(reloaded.parameters)) {
            assertEquals(a.key, b.key)
            assertContentEquals(a.tensor.hostF32(), b.tensor.hostF32(), a.key)
        }
        val adapted = logits(model, tokens)
        val fromDisk = logits(reloaded, tokens)
        assertContentEquals(logits(merged, tokens), fromDisk)
        val diff = maxDiff(adapted, fromDisk)
        val moved = maxDiff(adapted, logits(base, tokens))
        println("[lora-merge] tiny Qwen3: reloaded merged checkpoint vs adapted model, logits max |diff| $diff (training moved them by $moved)")
        assertTrue(moved > 0.1f)
        assertTrue(diff < 1e-4f, "merged checkpoint differs from the adapted model by $diff")
    }

    @Test
    fun qwen3MergedAndUnmergedLogitsAgree() {
        val dir = System.getenv("TLALOC_QWEN3_CHECKPOINT")?.let { Path.of(it) }
            ?: Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots").let { s ->
                if (!Files.isDirectory(s)) null
                else Files.list(s).use { l -> l.filter { Files.isRegularFile(it.resolve("model.safetensors")) }.findFirst().orElse(null) }
            }
        assumeTrue(dir != null, "no Qwen/Qwen3-0.6B checkpoint in the Hugging Face cache")
        val base = HfCausalLm.load(dir!!).model
        val adapted = Lora.apply(base, LoraConfig(8, 16f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(1)).let { m ->
            // B drawn from U(±0.01): an adapter of the size a short fine-tune produces.
            m.withParameters(Lora.adapterParameters(m).filter { it.key.endsWith("lora_B") }.associate { p ->
                val n = p.tensor.dims.fold(1) { a, d -> a * d }
                val u = uniformFloats(RandomKey.fromSeed(p.key.hashCode().toLong()), n)
                p.key to DTensor<Shape, F32>(HostF32Storage(FloatArray(n) { 0.02f * u[it] - 0.01f }), p.tensor.dims, F32)
            })
        }
        val ids = intArrayOf(576, 6722, 315, 9625, 374)
        val a = logits(adapted, ids)
        val merged = Lora.merge(adapted)
        assertFalse(Lora.hasAdapters(merged))
        val m = logits(merged, ids)
        val diff = maxDiff(a, m)
        val moved = maxDiff(a, logits(base, ids))
        val scale = a.maxOf { abs(it) }
        println("[lora-merge] Qwen3-0.6B, adapters on all 196 linear layers: merged vs unmerged logits max |diff| $diff; " +
            "the adapters move them by up to $moved; largest |logit| $scale")
        assertTrue(moved > 0.5f, "the adapters should move the logits (moved $moved)")
        assertTrue(diff < 5e-4f, "merged and unmerged logits differ by $diff")
    }
}
