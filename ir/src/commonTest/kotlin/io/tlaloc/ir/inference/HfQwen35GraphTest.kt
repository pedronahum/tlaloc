package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.io.JsonException
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The qwen3_5 family without a checkpoint: the config parser, the signature
 * with linear-attention state, and the graph's consistency over a tiny model
 * with random weights (three Gated DeltaNet layers and one gated attention
 * layer). Parity with transformers is HfQwen35RealParityTest (jvmTest).
 */
class HfQwen35GraphTest {

    private fun configJson(textExtra: String = "", outerExtra: String = ""): String = """
        {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
         "image_token_id": 29, "video_token_id": 30, "tie_word_embeddings": false$outerExtra,
         "vision_config": {"depth": 1},
         "text_config": {
           "model_type": "qwen3_5_text", "hidden_size": 16, "intermediate_size": 24,
           "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
           "vocab_size": 31, "rms_norm_eps": 1e-6, "max_position_embeddings": 64,
           "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
           "full_attention_interval": 4, "attn_output_gate": true,
           "linear_conv_kernel_dim": 4, "linear_key_head_dim": 4, "linear_num_key_heads": 2,
           "linear_num_value_heads": 4, "linear_value_head_dim": 4, "mamba_ssm_dtype": "float32",
           "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false, "hidden_act": "silu",
           "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                               "mrope_interleaved": true, "mrope_section": [1, 1, 0]}$textExtra
         }}
    """.trimIndent()

    private val config: HfDecoderConfig = HfDecoderConfig.parse(configJson()).copy(weightDType = F32)

    @Test
    fun theParserReadsTheHybridLayout() {
        assertEquals(HfModelFamily.Qwen3_5, config.family)
        assertEquals(listOf(0, 1, 3), config.linearLayers)
        assertEquals(TokenMixer.ATTENTION, config.layer(2).mixer)
        assertTrue(config.layer(2).queryGate && config.layer(2).qkNorm)
        assertEquals(4, config.rotaryDim)
        assertEquals(LinearAttentionConfig(2, 4, 4, 4, 4), config.linearAttention)
        assertEquals(mapOf(29 to "image_token_id", 30 to "video_token_id"), config.refusedTokenIds)
        assertTrue(config.layerNormGainPlusOne && config.qkNormGainPlusOne && config.finalNormGainPlusOne)
        // The q projection carries the gate: [2 * heads * headDim, hidden].
        assertEquals(listOf(64, 16), HfDecoderNames.expectedDims(DecoderWeightRole.Layer(2, DecoderLayerPart.Q_PROJ), config).toList())
        assertEquals(listOf(32, 1, 4), HfDecoderNames.expectedDims(DecoderWeightRole.Layer(0, DecoderLayerPart.CONV1D), config).toList())
    }

    @Test
    fun whatTheGraphDoesNotImplementIsRefusedByName() {
        assertTrue("attn_output_gate" in assertFailsWith<JsonException> {
            HfDecoderConfig.parse(configJson().replace("\"attn_output_gate\": true", "\"attn_output_gate\": false"))
        }.message!!)
        assertTrue("mamba_ssm_dtype" in assertFailsWith<JsonException> {
            HfDecoderConfig.parse(configJson().replace("\"mamba_ssm_dtype\": \"float32\"", "\"mamba_ssm_dtype\": \"bfloat16\""))
        }.message!!)
        assertTrue("output_gate_type" in assertFailsWith<JsonException> {
            HfDecoderConfig.parse(configJson(",\"output_gate_type\": \"sigmoid\""))
        }.message!!)
        // Qwen3.8 states the default; it reads.
        assertEquals(config.linearLayers, HfDecoderConfig.parse(configJson(",\"output_gate_type\": \"swish\", \"partial_rotary_factor\": 0.5")).linearLayers)
        assertTrue("mlp_only_layers" in assertFailsWith<JsonException> {
            HfDecoderConfig.parse(configJson(",\"mlp_only_layers\": [1]"))
        }.message!!)
        assertTrue("state slots" in assertFailsWith<JsonException> {
            config.toDecodeModelShape(numBlocks = 4, blockSize = 4)
        }.message!!)
    }

    private val bs = 4
    private val context = 32
    private val model = config.toDecodeModelShape(numBlocks = 1 + 2 * (context / bs), blockSize = bs, stateSlots = 3)

    @Test
    fun theSignatureCarriesStateSlotsAndAStatePoolPairPerLinearLayer() {
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(2, context))
        assertEquals(DecodeSlotRole.STATE_SLOTS, spec.inputs[5].role)
        assertEquals(DxirType(I32, listOf(2)), spec.inputs[5].type)
        assertEquals(6, spec.kvPoolInputBase)
        val pools = spec.inputs.drop(spec.kvPoolInputBase).take(8)
        assertEquals(listOf("convState0", "recurrentState0", "convState1", "recurrentState1", "keyCache2", "valueCache2"), pools.take(6).map { it.name })
        assertEquals(DxirType(F32, listOf(3, 3, 32)), pools[0].type)
        assertEquals(DxirType(F32, listOf(3, 4, 4, 4)), pools[1].type)
        assertEquals(DecodeSlotRole.STATE_POOL_IN, pools[0].role)
        assertEquals(DecodeSlotRole.KV_POOL_IN, pools[4].role)
        for ((i, o) in spec.donationPairs) assertEquals(spec.inputs[i].type, spec.outputs[o].type)
        assertTrue(spec.executableCacheKey("m").endsWith("/ls3n3"))
        // Without linear layers nothing changes.
        assertNull(config.copy(layers = List(4) { config.layer(2) }).linearStatePool(3))
    }

    @Test
    fun quantizedProjectionsKeepTheSmallOnesBesideThemInF32() {
        val q = config.copy(weightDType = io.tlaloc.core.BF16, weightQuant = WeightQuant.FP8)
        val slots = HfDecoderGraph.weightSlots(q).associateBy { it.name }
        // q/k/v and z are one e4m3fn weight; b and a are f32, so their matmuls take the
        // f32 input and XLA has no bf16 dot on the same input to merge them with.
        assertEquals(io.tlaloc.core.F8E4M3FN, slots.getValue("inProj0").type.dtype)
        assertEquals(F32, slots.getValue("inProjB0").type.dtype)
        assertEquals(F32, slots.getValue("inProjA0").type.dtype)
        // Unquantized, b and a are in the fused weight.
        assertTrue("inProjB0" !in HfDecoderGraph.weightSlots(config).map { it.name })
    }

    @Test
    fun speculativeEntriesBuildForOneAndThreeDrafts() {
        for (k in listOf(1, 3)) {
            val c = config.copy(mtpLayers = 1, mtpDraftTokens = k)
            val shape = c.toDecodeModelShape(numBlocks = 9, blockSize = 4, stateSlots = 2 * (k + 2))
            val decode = HfDecoderGraph.spec(c, shape, DecodeBucket(2, 16))
            assertEquals(1 + k, decode.tokensPerSeq)
            assertEquals(listOf("nextTokens", "accepted", "drafts"), decode.outputs.take(3).map { it.name })
            assertEquals(listOf(2, k), decode.draftsType.dims)
            // build() checks the graph against the signature.
            HfDecoderGraph.build(decode, c)
            HfDecoderGraph.build(HfDecoderGraph.spec(c, shape, DecodeBucket(1, 16), DecodeGraphKind.PREFILL, prefillChunk = 8), c)
        }
    }

    @Test
    fun fusedPagedAttentionMarksEveryAttentionOpAndKeysTheExecutable() {
        val plain = config.toDecodeModelShape(numBlocks = 9, blockSize = 4, stateSlots = 2)
        val fused = config.toDecodeModelShape(numBlocks = 9, blockSize = 4, stateSlots = 2, fusedPagedAttention = true)
        fun marks(shape: DecodeModelShape) = HfDecoderGraph.build(HfDecoderGraph.spec(config, shape, DecodeBucket(2, 16)), config)
            .body.filterIsInstance<io.tlaloc.ir.DxirOp>().filter { it.op == io.tlaloc.ir.OpKind.PAGED_ATTENTION }
            .map { it.attrs[io.tlaloc.ir.PagedAttentionAttrs.FUSED_KERNEL] == true }
        assertTrue(marks(fused).isNotEmpty() && marks(fused).all { it })
        assertTrue(marks(plain).none { it })
        val key = { shape: DecodeModelShape -> HfDecoderGraph.spec(config, shape, DecodeBucket(2, 16)).executableCacheKey("m") }
        assertTrue(key(plain) != key(fused))
    }

    private val weights: List<FloatArray> = run {
        val rng = Random(20261003)
        HfDecoderGraph.weightSlots(config).map { slot ->
            val n = slot.type.dims.fold(1) { a, b -> a * b }
            when {
                slot.name.startsWith("aLog") || slot.name.startsWith("dtBias") -> FloatArray(n) { rng.nextFloat() - 0.5f }
                slot.type.dims.size == 1 -> FloatArray(n) { 0.5f * (rng.nextFloat() - 0.5f) }
                else -> FloatArray(n) { 0.9f * (rng.nextFloat() - 0.5f) }
            }
        }
    }

    private val graphs = HashMap<Triple<DecodeGraphKind, Int, Int>, DxirFunction>()

    private fun graph(kind: DecodeGraphKind, batch: Int): DxirFunction = graphs.getOrPut(Triple(kind, batch, context)) {
        HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(batch, context), kind), config)
    }

    /** Sequences with their pages and state slot, over shared pools. */
    private inner class Server {
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context))
        var pools: List<FloatArray> = (0 until config.numLayers).flatMap { l ->
            spec.poolTypesOf(l).toList().map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
        }
        private var nextPage = 1

        inner class Seq(val slot: Int) {
            val pages = IntArray(context / bs) { nextPage++ }
            var length = 0
        }

        /** One call: each (sequence, tokens) a row, prefill when [kind] is PREFILL. Returns each row's logits. */
        fun call(kind: DecodeGraphKind, batch: Int, rows: List<Pair<Seq, IntArray>>): List<FloatArray> {
            val t = if (kind == DecodeGraphKind.DECODE) 1 else context
            fun intRows(f: (Int, Seq, IntArray, Int) -> Int, pad: Int, width: Int = t) = FloatArray(batch * width) { k ->
                val r = k / width
                val j = k % width
                if (r >= rows.size) return@FloatArray pad.toFloat()
                val (s, toks) = rows[r]
                val off = j - (width - toks.size)
                if (off < 0) pad.toFloat() else f(off, s, toks, j).toFloat()
            }
            val ins = buildList {
                add(intRows({ o, _, toks, _ -> toks[o] }, 0))
                add(intRows({ o, s, _, _ -> s.length + o }, 0))
                add(FloatArray(batch * (context / bs)) { k -> rows.getOrNull(k / (context / bs))?.first?.pages?.get(k % (context / bs))?.toFloat() ?: 0f })
                add(FloatArray(batch) { r -> rows.getOrNull(r)?.let { (s, toks) -> (s.length + toks.size).toFloat() } ?: 1f })
                add(intRows({ o, s, _, _ -> s.pages[(s.length + o) / bs] * bs + (s.length + o) % bs }, -1))
                add(FloatArray(batch) { r -> rows.getOrNull(r)?.first?.slot?.toFloat() ?: 0f })
                addAll(pools)
                addAll(weights)
            }
            val out = DxirInterpreter.evalFunction(graph(kind, batch), ins)
            pools = out.drop(1)
            for ((s, toks) in rows) s.length += toks.size
            val v = config.vocabSize
            return rows.indices.map { r -> out[0].copyOfRange(r * v, (r + 1) * v) }
        }
    }

    private val prompt = IntArray(10) { (it * 7 + 3) % 27 }

    private fun close(a: FloatArray, b: FloatArray, what: String) {
        for (i in a.indices) {
            assertTrue(abs(a[i] - b[i]) <= 2e-5f * maxOf(1f, abs(a[i])), "$what: logit $i ${a[i]} vs ${b[i]}")
        }
    }

    @Test
    fun aPromptSplitAcrossCallsOrDecodedTokenByTokenGivesTheLogitsOfOneCall() {
        val whole = Server().let { s -> s.call(DecodeGraphKind.PREFILL, 1, listOf(s.Seq(1) to prompt)).single() }
        val split = Server().let { s ->
            val q = s.Seq(2)
            s.call(DecodeGraphKind.PREFILL, 1, listOf(q to prompt.copyOfRange(0, 6)))
            s.call(DecodeGraphKind.PREFILL, 1, listOf(q to prompt.copyOfRange(6, 10))).single()
        }
        val stepped = Server().let { s ->
            val q = s.Seq(0)
            s.call(DecodeGraphKind.PREFILL, 1, listOf(q to prompt.copyOfRange(0, 6)))
            (6 until 10).map { s.call(DecodeGraphKind.DECODE, 1, listOf(q to intArrayOf(prompt[it]))).single() }.last()
        }
        close(whole, split, "two prefill calls")
        close(whole, stepped, "prefill then decode")
    }

    @Test
    fun aRowGetsTheSameLogitsInABatchAsAlone() {
        val other = IntArray(7) { (it * 5 + 1) % 29 }
        val alone = Server().let { s -> s.call(DecodeGraphKind.PREFILL, 1, listOf(s.Seq(1) to prompt)).single() }
        val (together, _) = Server().let { s ->
            val a = s.Seq(2)
            val b = s.Seq(0)
            s.call(DecodeGraphKind.PREFILL, 2, listOf(a to prompt, b to other)).let { it[0] to it[1] }
        }
        for (i in alone.indices) assertEquals(alone[i], together[i], "logit $i")
        // A slot reused by a new sequence starts from zero state: the same prompt in slot 1
        // after another sequence used it gets the logits of a fresh server.
        val reused = Server().let { s ->
            s.call(DecodeGraphKind.PREFILL, 1, listOf(s.Seq(1) to other))
            s.call(DecodeGraphKind.PREFILL, 1, listOf(s.Seq(1) to prompt)).single()
        }
        for (i in alone.indices) assertEquals(alone[i], reused[i], "reused slot, logit $i")
    }
}
