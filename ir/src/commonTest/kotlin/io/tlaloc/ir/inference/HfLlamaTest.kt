package io.tlaloc.ir.inference

import io.tlaloc.core.BF16
import io.tlaloc.core.io.JsonException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.478 (H3c-1) — the PURE half of HF Llama ingestion: config parsing, the
 * name bijection, and the dims each role must have. No filesystem, no
 * checkpoint; the real-checkpoint claims are `HfLlamaCheckpointTest` (jvmTest).
 *
 * The config text below is **TinyLlama-1.1B-Chat-v1.0's own `config.json`,
 * verbatim** — copied out of the downloaded checkpoint rather than invented,
 * so the defaulting rules are exercised against a shape that really ships
 * (no `head_dim` key; GQA 32/4; `rope_scaling: null`).
 */
class HfLlamaTest {

    private val tinyLlamaConfigJson = """
        {
          "architectures": ["LlamaForCausalLM"],
          "attention_bias": false,
          "bos_token_id": 1,
          "eos_token_id": 2,
          "hidden_act": "silu",
          "hidden_size": 2048,
          "initializer_range": 0.02,
          "intermediate_size": 5632,
          "max_position_embeddings": 2048,
          "model_type": "llama",
          "num_attention_heads": 32,
          "num_hidden_layers": 22,
          "num_key_value_heads": 4,
          "pretraining_tp": 1,
          "rms_norm_eps": 1e-05,
          "rope_scaling": null,
          "rope_theta": 10000.0,
          "tie_word_embeddings": false,
          "torch_dtype": "bfloat16",
          "transformers_version": "4.35.0",
          "use_cache": true,
          "vocab_size": 32000
        }
    """.trimIndent()

    @Test
    fun realTinyLlamaConfigParses() {
        val c = HfLlamaConfig.parse(tinyLlamaConfigJson)
        assertEquals("LlamaForCausalLM", c.architecture)
        assertEquals("llama", c.modelType)
        assertEquals(2048, c.hiddenSize)
        assertEquals(5632, c.intermediateSize)
        assertEquals(22, c.numLayers)
        assertEquals(32, c.numHeads)
        assertEquals(4, c.numKvHeads)
        // No head_dim key: the quotient rule.
        assertEquals(64, c.headDim)
        assertEquals(32000, c.vocabSize)
        assertEquals(1e-5, c.rmsNormEps)
        assertEquals(10000.0, c.ropeTheta)
        assertEquals(2048, c.maxPositionEmbeddings)
        assertFalse(c.tieWordEmbeddings)
        assertFalse(c.attentionBias)
        assertEquals("bfloat16", c.torchDtype)
        assertEquals(BF16, c.storageDType)
        assertNull(c.ropeScalingType)

        // The derived widths the name mapping is built on.
        assertEquals(2048, c.qProjOut)
        assertEquals(256, c.kvProjOut)
        assertEquals(8, c.gqaGroup)
        assertFalse(c.isMultiHead)
    }

    @Test
    fun headDimPrefersTheExplicitKeyOverTheQuotient() {
        // Llama-3.2-shaped: head_dim is stated and is NOT hidden/heads.
        val json = """
            {"architectures":["LlamaForCausalLM"],"hidden_size":2048,"intermediate_size":8192,
             "num_hidden_layers":16,"num_attention_heads":32,"num_key_value_heads":8,
             "head_dim":128,"vocab_size":128256,"tie_word_embeddings":true}
        """.trimIndent()
        val c = HfLlamaConfig.parse(json)
        assertEquals(128, c.headDim)
        assertEquals(2048 / 32, 64)
        // The mapping uses head_dim, so q_proj is WIDER than hidden.
        assertEquals(32 * 128, c.qProjOut)
        assertEquals(8 * 128, c.kvProjOut)
        assertTrue(c.tieWordEmbeddings)
        // The defaults for the keys this config omits.
        assertEquals(1e-6, c.rmsNormEps)
        assertEquals(10000.0, c.ropeTheta)
    }

    @Test
    fun numKvHeadsDefaultsToNumHeadsWhichIsPlainMha() {
        val json = """
            {"architectures":["LlamaForCausalLM"],"hidden_size":64,"intermediate_size":128,
             "num_hidden_layers":2,"num_attention_heads":4,"vocab_size":100}
        """.trimIndent()
        val c = HfLlamaConfig.parse(json)
        assertEquals(4, c.numKvHeads)
        assertTrue(c.isMultiHead)
        assertEquals(1, c.gqaGroup)
    }

    @Test
    fun theNameBijectionRoundTrips() {
        val c = HfLlamaConfig.parse(tinyLlamaConfigJson)
        val roles = HfLlamaNames.roles(c)
        // 1 embedding + 22 layers * 9 + final norm + head = 201, which is
        // exactly the tensor count of the real checkpoint.
        assertEquals(1 + 22 * 9 + 1 + 1, roles.size)
        assertEquals(201, roles.size)
        for (role in roles) {
            val name = HfLlamaNames.hfName(role)
            assertEquals(role, HfLlamaNames.role(name), "round trip failed for $name")
        }
        // Distinct names, i.e. the map really is injective.
        assertEquals(roles.size, roles.map { HfLlamaNames.hfName(it) }.toSet().size)
    }

    @Test
    fun theSpellingsAreTheOnesHuggingFaceWrites() {
        assertEquals("model.embed_tokens.weight", HfLlamaNames.hfName(LlamaWeightRole.EmbedTokens))
        assertEquals("model.norm.weight", HfLlamaNames.hfName(LlamaWeightRole.FinalNorm))
        assertEquals("lm_head.weight", HfLlamaNames.hfName(LlamaWeightRole.LmHead))
        assertEquals(
            "model.layers.7.self_attn.q_proj.weight",
            HfLlamaNames.hfName(LlamaWeightRole.Layer(7, LlamaLayerPart.Q_PROJ)),
        )
        assertEquals(
            "model.layers.0.mlp.down_proj.weight",
            HfLlamaNames.hfName(LlamaWeightRole.Layer(0, LlamaLayerPart.DOWN_PROJ)),
        )
        assertEquals(
            "model.layers.21.post_attention_layernorm.weight",
            HfLlamaNames.hfName(LlamaWeightRole.Layer(21, LlamaLayerPart.POST_ATTENTION_LAYERNORM)),
        )
    }

    @Test
    fun unknownNamesParseToNullRatherThanThrowing() {
        // A real vintage artifact: transformers < 4.36 persisted this buffer.
        assertNull(HfLlamaNames.role("model.layers.0.self_attn.rotary_emb.inv_freq"))
        assertNull(HfLlamaNames.role("model.layers.x.mlp.up_proj.weight"))
        assertNull(HfLlamaNames.role("model.layers..mlp.up_proj.weight"))
        assertNull(HfLlamaNames.role("model.layers.0.mlp.up_proj.bias"))
        assertNull(HfLlamaNames.role("model.layers.0"))
        assertNull(HfLlamaNames.role(""))
        assertNull(HfLlamaNames.role("model.embed_tokens.weight.extra"))
    }

    @Test
    fun expectedDimsEncodeTheTransposedLinearConvention() {
        val c = HfLlamaConfig.parse(tinyLlamaConfigJson)
        fun dims(p: LlamaLayerPart) =
            HfLlamaNames.expectedDims(LlamaWeightRole.Layer(3, p), c).toList()

        // These four are the ones the real checkpoint was checked against, and
        // the ones a [in, out] convention would reverse.
        assertEquals(listOf(256, 2048), dims(LlamaLayerPart.K_PROJ))
        assertEquals(listOf(256, 2048), dims(LlamaLayerPart.V_PROJ))
        assertEquals(listOf(5632, 2048), dims(LlamaLayerPart.GATE_PROJ))
        assertEquals(listOf(2048, 5632), dims(LlamaLayerPart.DOWN_PROJ))
        // Square, so these prove nothing on their own — recorded so a reader
        // knows why the rectangular ones above carry the claim.
        assertEquals(listOf(2048, 2048), dims(LlamaLayerPart.Q_PROJ))
        assertEquals(listOf(2048, 2048), dims(LlamaLayerPart.O_PROJ))

        assertEquals(listOf(2048), dims(LlamaLayerPart.INPUT_LAYERNORM))
        assertEquals(
            listOf(32000, 2048),
            HfLlamaNames.expectedDims(LlamaWeightRole.EmbedTokens, c).toList(),
        )
        assertEquals(
            listOf(32000, 2048),
            HfLlamaNames.expectedDims(LlamaWeightRole.LmHead, c).toList(),
        )

        assertTrue(HfLlamaNames.isTransposedLinear(LlamaWeightRole.LmHead))
        assertTrue(HfLlamaNames.isTransposedLinear(LlamaWeightRole.Layer(0, LlamaLayerPart.Q_PROJ)))
        assertFalse(HfLlamaNames.isTransposedLinear(LlamaWeightRole.EmbedTokens))
        assertFalse(HfLlamaNames.isTransposedLinear(LlamaWeightRole.FinalNorm))
        assertFalse(
            HfLlamaNames.isTransposedLinear(LlamaWeightRole.Layer(0, LlamaLayerPart.INPUT_LAYERNORM)),
        )
    }

    @Test
    fun theConfigBecomesADecodeModelShape() {
        val c = HfLlamaConfig.parse(tinyLlamaConfigJson)
        val s = c.toDecodeModelShape(numBlocks = 64, blockSize = 16)
        assertEquals(32000, s.vocabSize)
        assertEquals(2048, s.hiddenSize)
        assertEquals(32, s.numHeads)
        assertEquals(4, s.numKvHeads)
        assertEquals(64, s.headDim)
        assertEquals(22, s.numLayers)
        assertEquals(64, s.numBlocks)
        assertEquals(16, s.blockSize)
        assertNull(s.kvQuant)
    }

    @Test
    fun aScaledRopeIsRefusedByName() {
        val json = tinyLlamaConfigJson.replace(
            "\"rope_scaling\": null",
            "\"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 8.0}",
        )
        val c = HfLlamaConfig.parse(json)
        assertEquals("llama3", c.ropeScalingType)
        val e = assertFailsWith<JsonException> { c.toDecodeModelShape(4, 4) }
        assertTrue("llama3" in e.message!!, e.message!!)
        assertTrue("refused BY NAME" in e.message!!, e.message!!)
    }

    @Test
    fun theLegacyRopeScalingSpellingIsCaughtToo() {
        // transformers < 4.43 wrote "type", not "rope_type".
        val json = tinyLlamaConfigJson.replace(
            "\"rope_scaling\": null",
            "\"rope_scaling\": {\"type\": \"linear\", \"factor\": 4.0}",
        )
        assertEquals("linear", HfLlamaConfig.parse(json).ropeScalingType)
        // …and HF's own way of writing "no scaling at all" is not a refusal.
        val unscaled = tinyLlamaConfigJson.replace(
            "\"rope_scaling\": null",
            "\"rope_scaling\": {\"rope_type\": \"default\"}",
        )
        assertNull(HfLlamaConfig.parse(unscaled).ropeScalingType)
    }

    @Test
    fun attentionBiasIsRefusedByName() {
        val json = tinyLlamaConfigJson.replace("\"attention_bias\": false", "\"attention_bias\": true")
        val c = HfLlamaConfig.parse(json)
        assertTrue(c.attentionBias)
        val e = assertFailsWith<JsonException> { c.toDecodeModelShape(4, 4) }
        assertTrue("attention_bias" in e.message!!, e.message!!)
    }

    @Test
    fun aForeignArchitectureIsRefusedUnlessTheCallerOptsOut() {
        val json = tinyLlamaConfigJson.replace("LlamaForCausalLM", "MixtralForCausalLM")
        val e = assertFailsWith<JsonException> { HfLlamaConfig.parse(json) }
        assertTrue("MixtralForCausalLM" in e.message!!, e.message!!)
        // Opting out is explicit and gives the same numbers.
        val c = HfLlamaConfig.parse(json, strictArchitecture = false)
        assertEquals("MixtralForCausalLM", c.architecture)
        assertEquals(22, c.numLayers)
    }

    @Test
    fun aMissingShapeKeyIsRefusedByNameRatherThanDefaulted() {
        for (key in listOf(
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "vocab_size",
        )) {
            val json = tinyLlamaConfigJson.replace("\"$key\":", "\"${key}_typo\":")
            val e = assertFailsWith<JsonException>("$key should be required") {
                HfLlamaConfig.parse(json)
            }
            assertTrue(key in e.message!!, "${e.message}")
        }
    }

    @Test
    fun anIndivisibleGqaGroupingIsRefused() {
        val json = tinyLlamaConfigJson.replace("\"num_key_value_heads\": 4", "\"num_key_value_heads\": 5")
        val e = assertFailsWith<IllegalArgumentException> { HfLlamaConfig.parse(json) }
        assertTrue("num_key_value_heads" in e.message!! || "divide" in e.message!!, e.message!!)
    }

    @Test
    fun aHeadDimWithNoQuotientToFallBackOnIsRefused() {
        val json = """
            {"architectures":["LlamaForCausalLM"],"hidden_size":100,"intermediate_size":128,
             "num_hidden_layers":2,"num_attention_heads":3,"vocab_size":100}
        """.trimIndent()
        val e = assertFailsWith<JsonException> { HfLlamaConfig.parse(json) }
        assertTrue("head_dim" in e.message!!, e.message!!)
    }
}
