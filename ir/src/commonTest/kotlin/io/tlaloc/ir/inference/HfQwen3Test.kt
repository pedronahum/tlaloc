package io.tlaloc.ir.inference

import io.tlaloc.core.io.JsonException
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The Qwen3 family: its config, its tensor names, and its decode graph against
 * an independent forward pass written out in plain Kotlin. No checkpoint; the
 * real-weights claim is `HfQwen3RealParityTest` (jvmTest).
 *
 * The config text is Qwen/Qwen3-0.6B's own `config.json`, verbatim.
 */
class HfQwen3Test {

    private val qwen3ConfigJson = """
        {
          "architectures": [
            "Qwen3ForCausalLM"
          ],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 1024,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "max_position_embeddings": 40960,
          "max_window_layers": 28,
          "model_type": "qwen3",
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06,
          "rope_scaling": null,
          "rope_theta": 1000000,
          "sliding_window": null,
          "tie_word_embeddings": true,
          "torch_dtype": "bfloat16",
          "transformers_version": "4.51.0",
          "use_cache": true,
          "use_sliding_window": false,
          "vocab_size": 151936
        }
    """.trimIndent()

    // ------------------------------------------------------------- config

    @Test
    fun theRealQwen3ConfigParsesAsTheQwen3Family() {
        val c = HfDecoderConfig.parse(qwen3ConfigJson)
        assertEquals(HfModelFamily.Qwen3, c.family)
        assertEquals(1024, c.hiddenSize)
        assertEquals(16, c.numHeads)
        assertEquals(8, c.numKvHeads)
        // head_dim is stated and is not hidden / heads (64).
        assertEquals(128, c.headDim)
        assertEquals(2048, c.qProjOut)
        assertEquals(1e6, c.ropeTheta)
        assertTrue(c.tieWordEmbeddings)
        assertEquals(28, c.numLayers)
        for (l in 0 until c.numLayers) {
            assertEquals(DecoderLayerSpec(qkNorm = true), c.layer(l), "layer $l")
        }
        assertTrue(c.unsupportedFeatures().isEmpty(), "${c.unsupportedFeatures()}")
        c.toDecodeModelShape(numBlocks = 4, blockSize = 16)
    }

    @Test
    fun aQwen3LayerHasElevenTensorsAndTheCheckpointHas311() {
        val c = HfDecoderConfig.parse(qwen3ConfigJson)
        assertEquals(11, c.layer(0).parts.size)
        val roles = HfDecoderNames.roles(c)
        // 1 embedding + 28 x 11 + final norm + head = 311: the tensor count of
        // the real file, which stores lm_head.weight although it is tied.
        assertEquals(311, roles.size)
        assertEquals(
            "model.layers.3.self_attn.q_norm.weight",
            HfDecoderNames.hfName(DecoderWeightRole.Layer(3, DecoderLayerPart.Q_NORM), c.family),
        )
        assertEquals(
            DecoderWeightRole.Layer(27, DecoderLayerPart.K_NORM),
            HfDecoderNames.role("model.layers.27.self_attn.k_norm.weight", c.family),
        )
        assertEquals(
            listOf(128),
            HfDecoderNames.expectedDims(DecoderWeightRole.Layer(0, DecoderLayerPart.Q_NORM), c).toList(),
        )
        // q_proj is [heads * head_dim, hidden] = [2048, 1024]: wider than hidden.
        assertEquals(
            listOf(2048, 1024),
            HfDecoderNames.expectedDims(DecoderWeightRole.Layer(0, DecoderLayerPart.Q_PROJ), c).toList(),
        )
        assertTrue(!HfDecoderNames.isTransposedLinear(DecoderWeightRole.Layer(0, DecoderLayerPart.K_NORM)))
    }

    @Test
    fun anUnknownConfigKeyIsRefusedByName() {
        val json = qwen3ConfigJson.replace("\"use_cache\": true,", "\"use_cache\": true, \"query_pre_attn_scalar\": 256,")
        val e = assertFailsWith<JsonException> { HfDecoderConfig.parse(json) }
        assertTrue("'query_pre_attn_scalar'" in e.message!!, e.message!!)
        assertTrue("qwen3" in e.message!!, e.message!!)
    }

    @Test
    fun aKeyAnotherFamilyReadsIsStillUnknownToLlama() {
        // layer_types is a Qwen3 key; a Llama config carrying it is refused.
        val json = """
            {"architectures":["LlamaForCausalLM"],"hidden_size":64,"intermediate_size":128,
             "num_hidden_layers":2,"num_attention_heads":4,"vocab_size":100,
             "layer_types":["full_attention","full_attention"]}
        """.trimIndent()
        val e = assertFailsWith<JsonException> { HfDecoderConfig.parse(json) }
        assertTrue("'layer_types'" in e.message!!, e.message!!)
    }

    @Test
    fun slidingLayersAreReadTheWayTransformersReadsThemAndRefusedByTheGraph() {
        val json = qwen3ConfigJson
            .replace("\"use_sliding_window\": false", "\"use_sliding_window\": true")
            .replace("\"sliding_window\": null", "\"sliding_window\": 4096")
            .replace("\"max_window_layers\": 28", "\"max_window_layers\": 20")
        val c = HfDecoderConfig.parse(json)
        assertEquals(AttentionKind.FULL, c.layer(19).attention)
        assertEquals(AttentionKind.SLIDING, c.layer(20).attention)
        assertEquals(4096, c.layer(27).slidingWindow)
        val e = assertFailsWith<JsonException> { c.toDecodeModelShape(4, 16) }
        assertTrue("layer 20: sliding-window attention (window 4096)" in e.message!!, e.message!!)
        assertTrue("refused BY NAME" in e.message!!, e.message!!)
        // A reduced copy that keeps only full layers builds.
        c.copy(numLayers = 20).toDecodeModelShape(4, 16)
    }

    @Test
    fun explicitLayerTypesWinAndAnUnknownTypeIsRefused() {
        val types = List(28) { if (it % 2 == 1) "\"sliding_attention\"" else "\"full_attention\"" }
        val json = qwen3ConfigJson
            .replace("\"use_sliding_window\": false", "\"use_sliding_window\": true")
            .replace("\"sliding_window\": null", "\"sliding_window\": 512")
            .replace("\"use_cache\": true,", "\"use_cache\": true, \"layer_types\": [${types.joinToString(",")}],")
        val c = HfDecoderConfig.parse(json)
        assertEquals(AttentionKind.FULL, c.layer(0).attention)
        assertEquals(AttentionKind.SLIDING, c.layer(1).attention)
        assertEquals(512, c.layer(1).slidingWindow)

        val bad = json.replaceFirst("\"full_attention\"", "\"chunked_attention\"")
        val e = assertFailsWith<JsonException> { HfDecoderConfig.parse(bad) }
        assertTrue("chunked_attention" in e.message!!, e.message!!)
    }

    @Test
    fun theGemmaStyleSettingsAreRecordedAndRefusedByName() {
        val base = HfDecoderConfig.parse(qwen3ConfigJson).copy(numLayers = 2)
        val capped = base.copy(finalLogitSoftcap = 30.0, attnLogitSoftcap = 50.0)
        val e1 = assertFailsWith<JsonException> { capped.toDecodeModelShape(4, 16) }
        assertTrue("final_logit_softcapping 30.0" in e1.message!!, e1.message!!)
        assertTrue("attn_logit_softcapping 50.0" in e1.message!!, e1.message!!)

        val gemmaLayer = DecoderLayerSpec(
            rope = false, postAttentionOutputNorm = true, postFeedforwardNorm = true,
        )
        val e2 = assertFailsWith<JsonException> {
            base.copy(layers = listOf(DecoderLayerSpec(qkNorm = true), gemmaLayer)).toDecodeModelShape(4, 16)
        }
        assertTrue("layer 1: a layer without RoPE" in e2.message!!, e2.message!!)
        assertTrue("post-attention output norm" in e2.message!!, e2.message!!)
        assertTrue("post-feedforward norm" in e2.message!!, e2.message!!)

        val gelu = HfDecoderConfig.parse(qwen3ConfigJson.replace("\"silu\"", "\"gelu_pytorch_tanh\""))
        val e3 = assertFailsWith<JsonException> { gelu.toDecodeModelShape(4, 16) }
        assertTrue("hidden_act 'gelu_pytorch_tanh'" in e3.message!!, e3.message!!)
    }

    @Test
    fun theTransformers5RopeParametersFormIsRead() {
        val json = qwen3ConfigJson
            .replace("\"rope_theta\": 1000000,", "\"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 5000000},")
        val c = HfDecoderConfig.parse(json)
        assertEquals(5e6, c.ropeTheta)
        assertNull(c.ropeScalingType)
        val scaled = json.replace("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"")
        assertEquals("yarn", HfDecoderConfig.parse(scaled).ropeScalingType)
    }

    // ---------------------------------------------- graph vs plain Kotlin

    /**
     * A small Qwen3: GQA 4/2, head_dim 8 (not hidden/heads = 4), tied head,
     * theta 1e6, two layers, random weights.
     */
    private val small = HfDecoderConfig(
        architecture = "Qwen3ForCausalLM", modelType = "qwen3",
        hiddenSize = 16, intermediateSize = 24, numLayers = 2,
        numHeads = 4, numKvHeads = 2, headDim = 8, vocabSize = 29,
        rmsNormEps = 1e-6, ropeTheta = 1e6, maxPositionEmbeddings = 64,
        tieWordEmbeddings = true, attentionBias = false, torchDtype = "float32",
        ropeScalingType = null, family = HfModelFamily.Qwen3,
    )

    /** Staged weights in slot order, math layout; the head is the table transposed. */
    private val weights: Map<DecoderWeightRole, FloatArray> = run {
        val rng = Random(20260925)
        val slots = HfDecoderGraph.weightSlots(small)
        val roles = HfDecoderGraph.weightRoles(small)
        val m = LinkedHashMap<DecoderWeightRole, FloatArray>()
        for (i in roles.indices) {
            val n = slots[i].type.dims.fold(1) { a, b -> a * b }
            m[roles[i]] = if (slots[i].type.dims.size == 1) {
                FloatArray(n) { 1f + 0.5f * (rng.nextFloat() - 0.5f) }
            } else {
                FloatArray(n) { 0.8f * (rng.nextFloat() - 0.5f) }
            }
        }
        val e = m.getValue(DecoderWeightRole.EmbedTokens)
        val v = small.vocabSize
        val d = small.hiddenSize
        m[DecoderWeightRole.LmHead] = FloatArray(v * d) { i -> e[(i % v) * d + i / v] }
        m
    }

    private val prompt = intArrayOf(3, 17, 0, 28, 11, 5, 9)

    @Test
    fun theQwen3DecodeGraphMatchesAPlainKotlinForwardPass() {
        val graph = decodeLoop()
        val ref = reference(qkNorm = true)
        var worst = 0.0
        for (pos in prompt.indices) {
            val denom = max(1.0, ref[pos].maxOf { abs(it) })
            for (v in 0 until small.vocabSize) {
                worst = max(worst, abs(graph[pos][v] - ref[pos][v]) / denom)
            }
        }
        assertTrue(worst < 1e-5, "graph vs plain Kotlin: worst relative difference $worst")
    }

    /** Negative control: the same forward pass without the q/k norms is far from the graph. */
    @Test
    fun theComparisonSeesTheQkNorms() {
        val graph = decodeLoop()
        val noNorm = reference(qkNorm = false)
        var moved = 0.0
        for (pos in 1 until prompt.size) {
            for (v in 0 until small.vocabSize) moved = max(moved, abs(graph[pos][v] - noNorm[pos][v]))
        }
        assertTrue(moved > 1e-2, "dropping the q/k norms moved the logits by only $moved")
    }

    /** Logits at every position, one decode step per token, identity block table. */
    private fun decodeLoop(): List<FloatArray> {
        val blockSize = 4
        val numBlocks = 4
        val model = small.toDecodeModelShape(numBlocks = numBlocks, blockSize = blockSize)
        val spec = HfDecoderGraph.spec(small, model, DecodeBucket(1, numBlocks * blockSize))
        val fn = HfDecoderGraph.build(spec, small)
        val staged = HfDecoderGraph.weightRoles(small).map { weights.getValue(it) }
        var pools = List(2 * small.numLayers) {
            FloatArray(numBlocks * blockSize * small.numKvHeads * small.headDim)
        }
        return prompt.indices.map { pos ->
            val out = DxirInterpreter.evalFunction(
                fn,
                buildList {
                    add(floatArrayOf(prompt[pos].toFloat()))
                    add(floatArrayOf(pos.toFloat()))
                    add(FloatArray(numBlocks) { it.toFloat() })
                    add(floatArrayOf((pos + 1).toFloat()))
                    add(floatArrayOf(pos.toFloat()))
                    addAll(pools)
                    addAll(staged)
                },
            )
            pools = out.drop(1)
            out[0]
        }
    }

    /**
     * Qwen3's forward pass as transformers writes it, in doubles, with full
     * causal attention over a plain list of past K/V: embed; per layer
     * RMSNorm, q/k/v, per-head RMSNorm of q and k, RoPE (rotate_half),
     * softmax(q.k / sqrt(head_dim)) v with GQA, o_proj, residual, RMSNorm,
     * SiLU(gate) * up, down_proj, residual; final RMSNorm and the head.
     */
    private fun reference(qkNorm: Boolean): List<DoubleArray> {
        val c = small
        val d = c.hiddenSize
        val hd = c.headDim
        val half = hd / 2
        fun w(role: DecoderWeightRole) = weights.getValue(role)
        fun lw(l: Int, p: DecoderLayerPart) = w(DecoderWeightRole.Layer(l, p))
        fun matvec(x: DoubleArray, m: FloatArray, out: Int) =
            DoubleArray(out) { o -> var s = 0.0; for (i in x.indices) s += x[i] * m[i * out + o]; s }
        fun rms(x: DoubleArray, g: FloatArray, off: Int = 0, n: Int = x.size): DoubleArray {
            var ss = 0.0
            for (i in 0 until n) ss += x[off + i] * x[off + i]
            val r = 1.0 / sqrt(ss / n + c.rmsNormEps)
            return DoubleArray(n) { x[off + it] * r * g[it] }
        }
        fun rope(x: DoubleArray, pos: Int): DoubleArray {
            val out = x.copyOf()
            for (h in 0 until x.size / hd) {
                for (i in 0 until half) {
                    val a = pos * c.ropeTheta.pow(-2.0 * i / hd)
                    val x1 = x[h * hd + i]
                    val x2 = x[h * hd + half + i]
                    out[h * hd + i] = x1 * cos(a) - x2 * sin(a)
                    out[h * hd + half + i] = x2 * cos(a) + x1 * sin(a)
                }
            }
            return out
        }
        fun headNorm(x: DoubleArray, g: FloatArray): DoubleArray {
            val out = DoubleArray(x.size)
            for (h in 0 until x.size / hd) rms(x, g, h * hd, hd).copyInto(out, h * hd)
            return out
        }

        val keys = List(c.numLayers) { ArrayList<DoubleArray>() }
        val values = List(c.numLayers) { ArrayList<DoubleArray>() }
        val result = ArrayList<DoubleArray>()
        for ((pos, tok) in prompt.withIndex()) {
            val e = w(DecoderWeightRole.EmbedTokens)
            var x = DoubleArray(d) { e[tok * d + it].toDouble() }
            for (l in 0 until c.numLayers) {
                val hn = rms(x, lw(l, DecoderLayerPart.INPUT_LAYERNORM))
                var q = matvec(hn, lw(l, DecoderLayerPart.Q_PROJ), c.qProjOut)
                var k = matvec(hn, lw(l, DecoderLayerPart.K_PROJ), c.kvProjOut)
                val v = matvec(hn, lw(l, DecoderLayerPart.V_PROJ), c.kvProjOut)
                if (qkNorm) {
                    q = headNorm(q, lw(l, DecoderLayerPart.Q_NORM))
                    k = headNorm(k, lw(l, DecoderLayerPart.K_NORM))
                }
                q = rope(q, pos)
                k = rope(k, pos)
                keys[l] += k
                values[l] += v
                val att = DoubleArray(c.qProjOut)
                for (h in 0 until c.numHeads) {
                    val kh = h / c.gqaGroup
                    val scores = DoubleArray(pos + 1) { j ->
                        var s = 0.0
                        for (i in 0 until hd) s += q[h * hd + i] * keys[l][j][kh * hd + i]
                        s / sqrt(hd.toDouble())
                    }
                    val mx = scores.max()
                    val p = scores.map { exp(it - mx) }
                    val z = p.sum()
                    for (j in 0..pos) {
                        for (i in 0 until hd) att[h * hd + i] += p[j] / z * values[l][j][kh * hd + i]
                    }
                }
                val o = matvec(att, lw(l, DecoderLayerPart.O_PROJ), d)
                x = DoubleArray(d) { x[it] + o[it] }
                val hn2 = rms(x, lw(l, DecoderLayerPart.POST_ATTENTION_LAYERNORM))
                val g = matvec(hn2, lw(l, DecoderLayerPart.GATE_PROJ), c.intermediateSize)
                val u = matvec(hn2, lw(l, DecoderLayerPart.UP_PROJ), c.intermediateSize)
                val act = DoubleArray(g.size) { g[it] / (1 + exp(-g[it])) * u[it] }
                val down = matvec(act, lw(l, DecoderLayerPart.DOWN_PROJ), d)
                x = DoubleArray(d) { x[it] + down[it] }
            }
            val hf = rms(x, w(DecoderWeightRole.FinalNorm))
            result += matvec(hf, w(DecoderWeightRole.LmHead), c.vocabSize)
        }
        return result
    }
}
