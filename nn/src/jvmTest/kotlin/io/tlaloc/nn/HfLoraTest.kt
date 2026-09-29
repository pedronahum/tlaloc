package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.inference.HfDecoderConfig
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * LoRA on the Hugging Face families [HfCausalLm] reads: Qwen3-0.6B and
 * TinyLlama-1.1B. The adapted model's logits equal the base model's bit for
 * bit at initialization, and the target names select the layers PEFT's
 * would. Each checkpoint test skips by name without its checkpoint.
 */
class HfLoraTest {

    private fun qwen3(): Path? {
        System.getenv("TLALOC_QWEN3_CHECKPOINT")?.let { return Path.of(it) }
        val snapshots = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
        if (!Files.isDirectory(snapshots)) return null
        return Files.list(snapshots).use { s -> s.filter { Files.isRegularFile(it.resolve("model.safetensors")) }.findFirst().orElse(null) }
    }

    private fun tinyLlama(): Path? {
        System.getenv("TLALOC_TINYLLAMA_CHECKPOINT")?.let { return Path.of(it) }
        return Path.of(System.getProperty("user.home"), ".cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0")
            .takeIf { Files.isRegularFile(it.resolve("model.safetensors")) }
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

    private fun checkFamily(dir: Path, config: LoraConfig, expectedLayers: Int, ids: IntArray) {
        val loaded = HfCausalLm.load(dir)
        val base = loaded.model
        val adapted = Lora.apply(base, config, RandomKey.fromSeed(0))
        assertEquals(expectedLayers, Lora.adapterParameters(adapted).size / 2)
        val trainable = Lora.frozen.trainable(adapted).sumOf { p -> p.tensor.dims.fold(1L) { a, d -> a * d } }
        val total = adapted.parameters.sumOf { p -> p.tensor.dims.fold(1L) { a, d -> a * d } }
        val a = logits(base, ids)
        val b = logits(adapted, ids)
        var differing = 0
        for (i in a.indices) if (a[i].toRawBits() != b[i].toRawBits()) differing++
        println("[hf-lora] ${loaded.config.family.id}: ${expectedLayers} adapted layers, %,d trainable of %,d parameters (%.3f %%); %d of %d logits differ from the base model"
            .format(trainable, total, 100.0 * trainable / total, differing, a.size))
        assertEquals(0, differing, "the adapted model must compute the base model's logits exactly at initialization")
    }

    @Test
    fun qwen3WithAttentionAdaptersComputesTheBaseModelsLogits() {
        val dir = qwen3()
        assumeTrue(dir != null, "no Qwen/Qwen3-0.6B checkpoint in the Hugging Face cache")
        // 28 blocks x q, k, v, o.
        checkFamily(dir!!, LoraConfig(8, 16f, LoraConfig.ATTENTION), 112, intArrayOf(576, 6722, 315, 9625, 374))
    }

    @Test
    fun tinyLlamaWithAllLinearAdaptersComputesTheBaseModelsLogits() {
        val dir = tinyLlama()
        assumeTrue(dir != null, "no TinyLlama-1.1B-Chat-v1.0 checkpoint in ~/.cache/tlaloc-checkpoints")
        // 22 blocks x q, k, v, o, gate, up, down.
        checkFamily(dir!!, LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), 154, intArrayOf(1, 450, 7483, 310, 3444, 338))
    }

    @Test
    fun savingAModelWithAdaptersIsRefused() {
        val config = HfDecoderConfig.parse(
            """{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 16,
               "intermediate_size": 24, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
               "num_hidden_layers": 2, "vocab_size": 13, "rms_norm_eps": 1e-06, "rope_theta": 10000,
               "tie_word_embeddings": true, "hidden_act": "silu"}""",
        )
        val model = CausalLM.llama(
            CausalLmConfig(13, 16, 2, 4, 24, numKvHeads = 2, headDim = 4, normEps = 1e-6f, tiedEmbeddings = true, qkNorm = true),
            RandomKey.fromSeed(1),
        )
        val adapted = Lora.apply(model, LoraConfig(2, 4f, LoraConfig.ATTENTION), RandomKey.fromSeed(2))
        val out = Files.createTempDirectory("tlaloc-lora-refused-")
        try {
            val e = assertFailsWith<IllegalArgumentException> {
                HfCausalLm(adapted, config).save(out, out.resolve("x"))
            }
            assertTrue("LoRA adapters" in e.message!!, e.message)
            assertTrue(Files.list(out).use { it.count() } == 0L, "a refused save must write nothing")
        } finally {
            out.toFile().deleteRecursively()
        }
    }
}
