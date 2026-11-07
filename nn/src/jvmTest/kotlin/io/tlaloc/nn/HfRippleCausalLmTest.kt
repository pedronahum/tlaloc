package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Files
import java.nio.file.Path
import java.util.Random
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A tiny Ripple checkpoint (four sliding layers with a window of 4, one full layer without RoPE,
 * value residual, output gate, sandwich norms, FFN clamp, embedding scale) read into a
 * [CausalLM]: its logits are the serving graph's, its gradients are finite differences of its
 * loss, and it saves, reloads and takes LoRA adapters on the gate projection under Ripple's
 * module names.
 */
class HfRippleCausalLmTest {
    private val dir: Path = Files.createTempDirectory("tiny-ripple-nn")

    init {
        writeTinyRipple(dir)
    }

    @AfterTest fun remove() {
        dir.toFile().deleteRecursively()
    }

    private fun idsTensor(ids: IntArray) = DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(1, ids.size), I32)

    /** `[n, vocab]` logits of [ids] from the traced forward. */
    private fun logits(model: CausalLM, ids: IntArray): FloatArray {
        val params = model.parameters
        var out: Tracer<Shape>? = null
        captureN(listOf(idsTensor(ids)) + params.map { it.tensor }) { leaves ->
            val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
            model.forward(leaves[0], Params { byKey.getValue(it) }).also { out = it }
        }
        return FloatArray(out!!.size) { out!!.peek(it) }
    }

    /** The serving prefill graph's logits for the last token of [ids], the whole prompt in one call. */
    private fun servedLastLogits(ids: IntArray): FloatArray {
        val (config, staged) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 32
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val bucket = DecodeBucket(1, context)
        val spec = HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = ids.size)
        val prefill = HfDecoderGraph.build(spec, config)
        val pools = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val out = DxirInterpreter.evalFunction(prefill, buildList {
            add(FloatArray(ids.size) { ids[it].toFloat() })
            add(FloatArray(ids.size) { it.toFloat() })
            add(FloatArray(context / bs) { (1 + it).toFloat() })
            add(floatArrayOf(ids.size.toFloat()))
            add(FloatArray(ids.size) { (bs + it).toFloat() })
            addAll(pools)
            addAll(staged)
        })
        return out[0]
    }

    @Test fun theCheckpointReadsWithEveryRippleFeature() {
        val lm = HfCausalLm.load(dir)
        val m = lm.model
        assertEquals(EMBED_SCALE.toFloat(), m.embedScale)
        val b0 = m.blocks[0]
        assertEquals(WINDOW, b0.attn.window)
        assertTrue(b0.attn.rope != null && b0.attn.gate != null && b0.valueResLambda != null)
        assertTrue(b0.attnOutNorm != null && b0.mlpOutNorm != null)
        assertEquals(CLAMP.toFloat(), (b0.mlp as SwiGLU).clamp)
        assertEquals(null, m.blocks[4].attn.window)
        assertEquals(null, m.blocks[4].attn.rope)
    }

    @Test fun logitsAreTheServingGraphs() {
        val lm = HfCausalLm.load(dir)
        val prompt = intArrayOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41, 8, 9)
        val all = logits(lm.model, prompt)
        var worst = 0f
        for (n in listOf(3, 7, prompt.size)) {
            val want = servedLastLogits(prompt.copyOf(n))
            for (v in 0 until VOCAB) {
                val got = all[(n - 1) * VOCAB + v]
                worst = maxOf(worst, abs(got - want[v]))
                assertTrue(abs(got - want[v]) <= 2e-4f, "prefix $n logit $v: $got vs the serving graph's ${want[v]}")
            }
        }
        println("tiny Ripple CausalLM against the serving graph: worst |logit difference| $worst")
    }

    @Test fun gradientsAreFiniteDifferencesOfTheLoss() {
        val lm = HfCausalLm.load(dir)
        val ids = intArrayOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22)
        val next = IntArray(ids.size) { if (it + 1 < ids.size) ids[it + 1] else -1 }
        val step = captureCausalLm(lm.model, idsTensor(ids), idsTensor(next), chunk = 4)
        val got = step.run(lm.model, listOf(idsTensor(ids), idsTensor(next)))
        fun loss(model: CausalLM) = step.run(model, listOf(idsTensor(ids), idsTensor(next))).loss.toDouble()
        val byKey = lm.model.parameters.associate { it.key to it.tensor }
        // Small enough for the scaled embeddings through RMSNorm; the f32 loss's rounding is
        // about 1e-3 of a difference over it.
        val eps = 1e-3f
        for ((key, index) in listOf(
            "blocks.0.valueResLambda" to 0,
            "blocks.1.attn.g.w" to 17,
            "blocks.4.attn.q.w" to 5,
            "blocks.2.attnOutNorm.weight" to 3,
            "blocks.3.mlp.up.w" to 40,
            "blocks.0.mlpOutNorm.weight" to 7,
            "embed.table" to 4 * HIDDEN + 2,
        )) {
            val t = byKey.getValue(key)
            fun shifted(d: Float): CausalLM {
                val v = t.hostF32().copyOf().also { it[index] += d }
                return lm.model.withParameters(mapOf(key to DTensor<Shape, F32>(HostF32Storage(v), t.dims.copyOf(), F32)))
            }
            val fd = (loss(shifted(eps)) - loss(shifted(-eps))) / (2 * eps)
            val g = got.gradients.getValue(key).hostF32()[index].toDouble()
            assertTrue(abs(fd - g) <= 2e-3 + 3e-2 * abs(fd), "d loss / d $key[$index]: reverse mode $g, finite difference $fd")
        }
    }

    @Test fun itSavesUnderRipplesNamesAndReloads() {
        val lm = HfCausalLm.load(dir)
        val out = Files.createTempDirectory("tiny-ripple-saved")
        try {
            lm.save(dir, out, dtype = F32)
            val names = SafetensorsFile.open(out.resolve("model.safetensors")).use { it.loadAll().keys }
            val source = SafetensorsFile.open(dir.resolve("model.safetensors")).use { it.loadAll().keys }
            assertEquals(source, names)
            val prompt = intArrayOf(3, 5, 9, 12, 33, 7)
            val a = logits(lm.model, prompt)
            val b = logits(HfCausalLm.load(out).model, prompt)
            for (i in a.indices) assertEquals(a[i], b[i], "logit $i")
        } finally {
            out.toFile().deleteRecursively()
        }
    }

    @Test fun loraTargetsTheGateProjection() {
        val lm = HfCausalLm.load(dir)
        val targets = listOf("q_proj", "v_proj", "g_proj", "up_proj")
        assertTrue(Lora.matchingLayers(lm.model, listOf("g_proj")).size == LAYERS)
        val adapted = Lora.apply(lm.model, LoraConfig(rank = 2, alpha = 4f, targetModules = targets), RandomKey.fromSeed(3))
        // B starts at zero: the adapted model computes what the base does, and keeps its Ripple layers.
        val prompt = intArrayOf(3, 5, 9, 12, 33, 7)
        val a = logits(lm.model, prompt)
        val b = logits(adapted, prompt)
        for (i in a.indices) assertEquals(a[i], b[i], 1e-6f, "logit $i")
        assertEquals(lm.model.blocks[0].attn.window, adapted.blocks[0].attn.window)
        assertTrue(adapted.blocks[0].valueResLambda != null && (adapted.blocks[0].mlp as SwiGLU).clamp != null)
        assertTrue(Lora.adapterParameters(adapted).any { "attn.g." in it.key })
    }

    private companion object {
        const val VOCAB = 64
        const val HIDDEN = 24
        const val INTER = 48
        const val LAYERS = 5
        const val HEADS = 6
        const val KV = 3
        const val HD = 8
        const val WINDOW = 4
        const val CLAMP = 0.75
        const val EMBED_SCALE = 4.898979485566356

        val CONFIG = """
            {"architectures": ["RicursiveForCausalLM"], "model_type": "ricursive", "dtype": "float32",
             "vocab_size": $VOCAB, "hidden_size": $HIDDEN, "intermediate_size": $INTER, "num_hidden_layers": $LAYERS,
             "num_attention_heads": $HEADS, "num_key_value_heads": $KV, "head_dim": $HD,
             "max_position_embeddings": 128, "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": $WINDOW,
             "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
             "hybrid_layer_pattern": [1, 1, 1, 1, 0], "is_hybrid_swa": true, "hidden_act": "silu",
             "use_gated_attention": true, "gate_full_precision": true, "use_valres": true, "ffn_clamp_threshold": $CLAMP,
             "embed_scale": $EMBED_SCALE, "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": [2],
             "pad_token_id": 0, "use_cache": true}
        """.trimIndent()

        fun writeTinyRipple(dir: Path) {
            val rng = Random(20261106)
            fun linear(o: Int, i: Int) = FloatArray(o * i) { (0.25 * rng.nextGaussian()).toFloat() }
            fun gain(n: Int) = FloatArray(n) { (1.0 + 0.3 * rng.nextGaussian()).toFloat() }
            val w = ArrayList<SafetensorsTensor>()
            fun put(name: String, data: FloatArray, vararg dims: Int) {
                w += SafetensorsTensor(name, F32, dims, HostF32Storage(data))
            }
            put("model.embed_tokens.weight", linear(VOCAB, HIDDEN), VOCAB, HIDDEN)
            put("lm_head.weight", linear(VOCAB, HIDDEN), VOCAB, HIDDEN)
            put("model.norm.weight", gain(HIDDEN), HIDDEN)
            for (l in 0 until LAYERS) {
                val p = "model.layers.$l."
                put(p + "self_attn.q_proj.weight", linear(HEADS * HD, HIDDEN), HEADS * HD, HIDDEN)
                put(p + "self_attn.k_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
                put(p + "self_attn.v_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
                put(p + "self_attn.g_proj.weight", linear(HEADS * HD, HIDDEN), HEADS * HD, HIDDEN)
                put(p + "self_attn.o_proj.weight", linear(HIDDEN, HEADS * HD), HIDDEN, HEADS * HD)
                put(p + "self_attn.q_norm.weight", gain(HD), HD)
                put(p + "self_attn.k_norm.weight", gain(HD), HD)
                put(p + "self_attn.value_res_lambda", floatArrayOf(rng.nextGaussian().toFloat()), 1)
                put(p + "mlp.gate_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
                put(p + "mlp.up_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
                put(p + "mlp.down_proj.weight", linear(HIDDEN, INTER), HIDDEN, INTER)
                for (n in listOf("input", "post_attention", "pre_feedforward", "post_feedforward")) put(p + n + "_layernorm.weight", gain(HIDDEN), HIDDEN)
            }
            SafetensorsFileWriter.write(dir.resolve("model.safetensors"), w)
            Files.writeString(dir.resolve("config.json"), CONFIG)
        }
    }
}
