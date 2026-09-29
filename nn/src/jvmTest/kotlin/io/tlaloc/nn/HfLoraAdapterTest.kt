package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.io.parseJson
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.inference.HfDecoderConfig
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [HfLoraAdapter]: PEFT's file names, tensor names, layouts and config;
 * a save and load that reproduces every adapter tensor; refusals of what
 * PEFT expresses and Tlaloc does not compute; and, when a Python with
 * torch, transformers and peft is available, parity with PEFT itself in
 * both directions on a tiny random Qwen3 checkpoint (no download).
 */
class HfLoraAdapterTest {

    private val tmp = Files.createTempDirectory("tlaloc-lora-adapter-")

    @AfterTest
    fun cleanup() {
        tmp.toFile().deleteRecursively()
    }

    private val configJson = """{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 16,
        "intermediate_size": 24, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
        "num_hidden_layers": 2, "vocab_size": 13, "rms_norm_eps": 1e-06, "rope_theta": 10000,
        "max_position_embeddings": 64, "tie_word_embeddings": true, "hidden_act": "silu",
        "attention_bias": false, "torch_dtype": "float32"}"""

    private val lm = CausalLmConfig(13, 16, 2, 4, 24, numKvHeads = 2, headDim = 4, ropeTheta = 10000f,
        normEps = 1e-6f, tiedEmbeddings = true, qkNorm = true, initStd = 0.3f)

    private val ids = intArrayOf(1, 5, 2, 7, 3, 12, 0, 4)

    /** A tiny random Qwen3 written as a Hugging Face checkpoint (f32) into `tmp/base`. */
    private fun baseCheckpoint(): HfCausalLm {
        val src = Files.createDirectories(tmp.resolve("src"))
        Files.writeString(src.resolve("config.json"), configJson)
        val config = HfDecoderConfig.parse(configJson)
        // Norm weights away from 1 so a transposed or misnamed norm would show.
        val model = CausalLM.llama(lm, RandomKey.fromSeed(17)).let { m ->
            m.withParameters(m.parameters.filter { "Norm" in it.key || it.key == "norm.weight" }.associate { p ->
                p.key to randomLike(p.tensor.dims, p.key.hashCode().toLong(), 0.5f, 1.5f)
            })
        }
        val hf = HfCausalLm(model, config)
        hf.save(src, tmp.resolve("base"), dtype = F32)
        return HfCausalLm.load(tmp.resolve("base"))
    }

    private fun randomLike(dims: IntArray, seed: Long, lo: Float, hi: Float): DTensor<*, F32> {
        val n = dims.fold(1) { a, d -> a * d }
        val u = uniformFloats(RandomKey.fromSeed(seed), n)
        return DTensor<Shape, F32>(HostF32Storage(FloatArray(n) { lo + (hi - lo) * u[it] }), dims.copyOf(), F32)
    }

    /** [model] with every adapter's A and B replaced by random values, so the adapter is not the identity. */
    private fun randomized(model: CausalLM): CausalLM =
        model.withParameters(Lora.adapterParameters(model).associate { p ->
            p.key to randomLike(p.tensor.dims, p.key.hashCode().toLong(), -0.3f, 0.3f)
        })

    private fun logits(model: CausalLM): FloatArray {
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
    fun savesPeftNamesLayoutsAndConfig() {
        val base = baseCheckpoint()
        val adapted = randomized(Lora.apply(base.model, LoraConfig(4, 8f, LoraConfig.ATTENTION), RandomKey.fromSeed(1)))
        val dir = HfLoraAdapter.save(HfCausalLm(adapted, base.config), tmp.resolve("adapter"), "Qwen/Qwen3-0.6B")

        val config = parseJson(Files.readString(dir.resolve("adapter_config.json"))) as JsonObject
        assertEquals("LORA", config.str("peft_type"))
        assertEquals("CAUSAL_LM", config.str("task_type"))
        assertEquals("Qwen/Qwen3-0.6B", config.str("base_model_name_or_path"))
        assertEquals(4, (config["r"] as JsonNumber).asInt("r"))
        assertEquals(8.0, (config["lora_alpha"] as JsonNumber).value)
        assertEquals(listOf("q_proj", "k_proj", "v_proj", "o_proj"),
            (config["target_modules"] as JsonArray).elements.map { (it as io.tlaloc.core.io.JsonString).value })

        SafetensorsFile.open(dir.resolve("adapter_model.safetensors")).use { f ->
            assertEquals(16, f.names.size)
            val a = f.load("base_model.model.model.layers.1.self_attn.k_proj.lora_A.weight")
            val b = f.load("base_model.model.model.layers.1.self_attn.k_proj.lora_B.weight")
            assertContentEquals(intArrayOf(4, 16), a.dims) // [r, in]
            assertContentEquals(intArrayOf(8, 4), b.dims)  // [out = kvHeads * headDim, r]
            val lora = adapted.blocks[1].attn.k.lora!!
            // PEFT's lora_A.weight[k, i] is A[i, k].
            assertEquals(lora.a.hostF32()[3 * 4 + 2], a.toF32Array()[2 * 16 + 3])
            assertEquals(lora.b.hostF32()[1 * 8 + 5], b.toF32Array()[5 * 4 + 1])
        }
    }

    @Test
    fun loadReproducesEveryAdapterTensorAndTheLogits() {
        val base = baseCheckpoint()
        val adapted = randomized(Lora.apply(base.model, LoraConfig(3, 5f, LoraConfig.ALL_LINEAR, useRslora = true), RandomKey.fromSeed(2)))
        HfLoraAdapter.save(HfCausalLm(adapted, base.config), tmp.resolve("adapter"), tmp.resolve("base").toString())
        assertEquals(LoraConfig(3, 5f, listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), 0f, true),
            HfLoraAdapter.readConfig(tmp.resolve("adapter")))
        val loaded = HfLoraAdapter.load(base, tmp.resolve("adapter")).model
        assertEquals(adapted.parameters.map { it.key }, loaded.parameters.map { it.key })
        for ((a, b) in adapted.parameters.zip(loaded.parameters)) assertContentEquals(a.tensor.hostF32(), b.tensor.hostF32(), a.key)
        assertContentEquals(logits(adapted), logits(loaded))
    }

    @Test
    fun partialTargetsAreWrittenAsFullModuleNames() {
        val base = baseCheckpoint()
        val adapted = Lora.apply(base.model, LoraConfig(2, 2f, listOf("q_proj", "model.layers.1.mlp.down_proj")), RandomKey.fromSeed(3))
        HfLoraAdapter.save(HfCausalLm(adapted, base.config), tmp.resolve("adapter"), "base")
        assertEquals(listOf("q_proj", "model.layers.1.mlp.down_proj"), HfLoraAdapter.readConfig(tmp.resolve("adapter")).targetModules)
        val loaded = HfLoraAdapter.load(base, tmp.resolve("adapter")).model
        assertEquals(Lora.matchingLayers(adapted, listOf("q_proj", "model.layers.1.mlp.down_proj")),
            listOf("blocks.0.attn.q", "blocks.1.attn.q", "blocks.1.mlp.down"))
        assertEquals(Lora.adapterParameters(adapted).map { it.key }, Lora.adapterParameters(loaded).map { it.key })
    }

    @Test
    fun whatLoraAdapterDoesNotComputeIsRefusedByName() {
        val base = baseCheckpoint()
        val adapted = Lora.apply(base.model, LoraConfig(2, 2f, listOf("q_proj")), RandomKey.fromSeed(3))
        val dir = HfLoraAdapter.save(HfCausalLm(adapted, base.config), tmp.resolve("adapter"), "base")
        val original = Files.readString(dir.resolve("adapter_config.json"))
        for ((edit, name) in listOf(
            "\"use_dora\": false" to "\"use_dora\": true",
            "\"rank_pattern\": {}" to "\"rank_pattern\": {\"q_proj\": 4}",
            "\"bias\": \"none\"" to "\"bias\": \"all\"",
            "\"modules_to_save\": null" to "\"modules_to_save\": [\"lm_head\"]",
        )) {
            Files.writeString(dir.resolve("adapter_config.json"), original.replace(edit, name))
            val e = assertFailsWith<IllegalArgumentException> { HfLoraAdapter.load(base, dir) }
            assertTrue("does not compute" in e.message!!, e.message)
        }
        Files.writeString(dir.resolve("adapter_config.json"), original)
        assertFailsWith<IllegalArgumentException> { HfLoraAdapter.load(HfCausalLm(adapted, base.config), dir) }
        assertFailsWith<IllegalArgumentException> { HfLoraAdapter.save(base, tmp.resolve("none"), "base") }
    }

    // ------------------------------------------------------------------
    // Parity with PEFT itself.
    // ------------------------------------------------------------------

    private fun peftPython(): String? {
        val candidates = listOfNotNull(
            System.getenv("TLALOC_PEFT_PYTHON"),
            Path.of(System.getProperty("user.home"), ".local/venvs/peft/bin/python").toString(),
        )
        return candidates.firstOrNull { py ->
            Files.isExecutable(Path.of(py)) && runCatching {
                val p = ProcessBuilder(py, "-c", "import torch, transformers, peft").redirectErrorStream(true).start()
                p.waitFor(120, TimeUnit.SECONDS) && p.exitValue() == 0
            }.getOrDefault(false)
        }
    }

    private fun runPeft(py: String, mode: String, adapterDir: Path): JsonObject {
        val idsFile = tmp.resolve("ids.json")
        Files.writeString(idsFile, ids.joinToString(",", "[", "]"))
        val out = tmp.resolve("peft-$mode.json")
        val script = Path.of("..", "harness", "python", "peft_lora_parity.py").toAbsolutePath().normalize()
        val p = ProcessBuilder(py, script.toString(), mode, tmp.resolve("base").toString(), adapterDir.toString(), idsFile.toString(), out.toString())
            .redirectErrorStream(true).start()
        val log = p.inputStream.bufferedReader().readText()
        assertTrue(p.waitFor(600, TimeUnit.SECONDS) && p.exitValue() == 0, "peft_lora_parity.py $mode failed:\n$log")
        return parseJson(Files.readString(out)) as JsonObject
    }

    private fun floats(o: JsonObject, key: String): FloatArray =
        o.arr(key).elements.map { (it as JsonNumber).value.toFloat() }.toFloatArray()

    private fun maxDiff(a: FloatArray, b: FloatArray): Float {
        assertEquals(a.size, b.size)
        return a.indices.maxOf { abs(a[it] - b[it]) }
    }

    @Test
    fun peftReadsATlalocAdapterAndComputesTheSameLogits() {
        val py = peftPython()
        assumeTrue(py != null, "no Python with torch, transformers and peft (set TLALOC_PEFT_PYTHON, or ~/.local/venvs/peft)")
        val base = baseCheckpoint()
        val adapted = randomized(Lora.apply(base.model, LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(4)))
        HfLoraAdapter.save(HfCausalLm(adapted, base.config), tmp.resolve("adapter"), tmp.resolve("base").toString())
        val peft = runPeft(py!!, "read", tmp.resolve("adapter"))
        val tlaloc = logits(adapted)
        val tlalocMerged = logits(Lora.merge(adapted))
        val moved = maxDiff(tlaloc, logits(base.model))
        val adaptedDiff = maxDiff(tlaloc, floats(peft, "adapted"))
        val mergedDiff = maxDiff(tlalocMerged, floats(peft, "merged"))
        println("[peft-parity] Tlaloc adapter read by PEFT: logits max |diff| $adaptedDiff; merged $mergedDiff; the adapter moves the logits by up to $moved")
        assertTrue(moved > 0.1f, "the adapter should change the logits")
        assertTrue(adaptedDiff < 1e-4f, "PEFT and Tlaloc adapted logits differ by $adaptedDiff")
        assertTrue(mergedDiff < 1e-4f, "PEFT and Tlaloc merged logits differ by $mergedDiff")
    }

    @Test
    fun tlalocReadsAPeftAdapterAndComputesTheSameLogits() {
        val py = peftPython()
        assumeTrue(py != null, "no Python with torch, transformers and peft (set TLALOC_PEFT_PYTHON, or ~/.local/venvs/peft)")
        val base = baseCheckpoint()
        val peft = runPeft(py!!, "write", tmp.resolve("peft-adapter"))
        val loaded = HfLoraAdapter.load(base, tmp.resolve("peft-adapter")).model
        assertEquals(8, Lora.adapterParameters(loaded).size) // 2 layers x q, v x A, B
        val diff = maxDiff(logits(loaded), floats(peft, "adapted"))
        val moved = maxDiff(logits(loaded), logits(base.model))
        println("[peft-parity] PEFT adapter read by Tlaloc: logits max |diff| $diff; the adapter moves the logits by up to $moved")
        assertTrue(moved > 1e-3f, "PEFT's random-B adapter should change the logits")
        assertTrue(diff < 1e-4f, "PEFT and Tlaloc logits differ by $diff")
    }
}
