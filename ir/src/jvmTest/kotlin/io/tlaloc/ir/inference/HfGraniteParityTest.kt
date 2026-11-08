package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import java.util.Random
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The granite family's decode graph against [Reference], a dense forward pass in doubles, on a
 * tiny random checkpoint whose four scalars are all away from 1 (Granite 3's style: embeddings
 * times 12, residual adds times 0.22, logits divided by 8) and whose attention scale is not
 * `1 / sqrt(head_dim)`. [theReferenceIsTransformersGraniteCode] checks the reference against
 * transformers' `modeling_granite.py` run on the same checkpoint.
 */
class HfGraniteParityTest {

    private val dir: Path = Files.createTempDirectory("tiny-granite")
    private val weights: Map<String, FloatArray> = writeTinyGranite(dir)

    @AfterTest
    fun removeCheckpoint() {
        dir.toFile().deleteRecursively()
    }

    @Test
    fun theTinyCheckpointReadsAsGranite() {
        HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config
            assertEquals(HfModelFamily.Granite, c.family)
            assertEquals(EMBEDDING_MULTIPLIER, c.embedScale)
            assertEquals(ATTENTION_MULTIPLIER, c.attentionScale)
            assertEquals(RESIDUAL_MULTIPLIER, c.residualMultiplier)
            assertEquals(1.0 / LOGITS_SCALING, c.logitMultiplier)
            assertEquals(DecoderLayerSpec(), c.layer(0))
            assertEquals(emptyList(), ckpt.verifyInventory())
        }
    }

    /**
     * The committed `tiny_granite` fixture (the transformers oracle reads it) is this checkpoint.
     * TLALOC_WRITE_TINY_GRANITE=1 writes it.
     */
    @Test
    fun theCommittedFixtureIsThisCheckpoint() {
        val fixture = Path.of("src/jvmTest/resources/io/tlaloc/ir/inference/tiny_granite")
        if (System.getenv("TLALOC_WRITE_TINY_GRANITE") == "1") writeTinyGranite(fixture.also { Files.createDirectories(it) })
        for (f in listOf("config.json", "model.safetensors")) {
            assertTrue(Files.readAllBytes(dir.resolve(f)).contentEquals(Files.readAllBytes(fixture.resolve(f))), "$f differs from $fixture/$f")
        }
    }

    /**
     * `tiny_granite/greedy.json`, what triton/verify.sh's Granite lane expects of the served model:
     * for each prompt, its ids and [GREEDY_NEW] greedy ids by [Reference], and the smallest top-1
     * margin among those steps. TLALOC_WRITE_TINY_GRANITE=1 writes it.
     */
    @Test
    fun theGreedyFixtureIsTheReferences() {
        val reference = Reference(weights)
        val out = StringBuilder("{")
        for ((i, prompt) in GREEDY_PROMPTS.withIndex()) {
            val seq = ArrayList(prompt)
            var margin = Double.MAX_VALUE
            repeat(GREEDY_NEW) {
                val l = reference.logits(seq).last()
                val order = l.indices.sortedByDescending { l[it] }
                margin = minOf(margin, l[order[0]] - l[order[1]])
                seq += order[0]
            }
            assertTrue(margin > 0.02, "prompt $i: a top-1 margin of $margin is too close for a device's rounding")
            if (i > 0) out.append(',')
            out.append("\"p$i\": [${prompt.joinToString(",", "[", "]")}, ${seq.drop(prompt.size).joinToString(",", "[", "]")}, $margin]")
        }
        out.append("}\n")
        val file = Path.of("src/jvmTest/resources/io/tlaloc/ir/inference/tiny_granite/greedy.json")
        if (System.getenv("TLALOC_WRITE_TINY_GRANITE") == "1") Files.writeString(file, out)
        assertGreedyFixture(Files.readString(file), out.toString())
    }

    /**
     * webAI's TwIL-8B (balanced and verbose share the file) and its multipliers, read from the
     * local Hugging Face cache (its licence is non-commercial, so the repo keeps no copy).
     */
    @Test
    fun theTwil8bConfigReadsAsGranite() {
        val snapshots = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--webAI-Official--TwIL-8B/snapshots")
        val file = if (Files.isDirectory(snapshots)) Files.list(snapshots).use { s -> s.map { it.resolve("config.json") }.filter { Files.isRegularFile(it) }.findFirst().orElse(null) } else null
        assumeTrue(file != null, "webAI-Official/TwIL-8B is not in the local Hugging Face cache")
        val c = HfDecoderConfig.parse(Files.readString(file!!))
        assertEquals(HfModelFamily.Granite, c.family)
        assertEquals(40, c.numLayers)
        assertEquals(4096, c.hiddenSize)
        assertEquals(12800, c.intermediateSize)
        assertEquals(32, c.numHeads)
        assertEquals(8, c.numKvHeads)
        assertEquals(128, c.headDim)
        assertEquals(100352, c.vocabSize)
        assertEquals(1e7, c.ropeTheta)
        assertEquals(131072, c.maxPositionEmbeddings)
        assertEquals(null, c.ropeScalingType)
        assertEquals(null, c.embedScale)
        assertEquals(0.0078125, c.attentionScale)
        assertEquals(1.0, c.residualMultiplier)
        assertEquals(1.0, c.logitMultiplier)
        assertEquals(io.tlaloc.core.BF16, c.weightDType)
        assertTrue(!c.tieWordEmbeddings)
    }

    @Test
    fun anUnreadKeyOrABadMultiplierIsRefusedByName() {
        val unknown = CONFIG.replaceFirst("\"use_cache\"", "\"use_new_thing\": true, \"use_cache\"")
        assertTrue("use_new_thing" in assertFailsWith<JsonException> { HfDecoderConfig.parse(unknown) }.message!!)
        val zero = CONFIG.replace("\"logits_scaling\": $LOGITS_SCALING", "\"logits_scaling\": 0")
        assertTrue("logits_scaling" in assertFailsWith<JsonException> { HfDecoderConfig.parse(zero) }.message!!)
    }

    /** Turning off any one multiplier moves the logits far past [TOL]. */
    @Test
    fun eachMultiplierMovesTheLogits() {
        val ids = PROMPTS[1]
        val base = Reference(weights).logits(ids)
        for ((what, other) in listOf(
            "embedding_multiplier" to Reference(weights, embedding = false),
            "attention_multiplier" to Reference(weights, attention = false),
            "residual_multiplier" to Reference(weights, residual = false),
            "logits_scaling" to Reference(weights, logits = false),
        )) {
            val o = other.logits(ids).last()
            val moved = o.indices.maxOf { abs(o[it] - base.last()[it]) }
            assertTrue(moved > 100 * TOL, "$what moves the last logits only $moved")
        }
    }

    /**
     * [Reference] against transformers' own `GraniteForCausalLM` run on this checkpoint
     * (`harness/python/granite_modeling_fixture.py`), every logit at every position.
     */
    @Test
    fun theReferenceIsTransformersGraniteCode() {
        val text = javaClass.getResourceAsStream("granite_modeling_logits.json")?.use { it.readBytes().decodeToString() }
        assumeTrue(text != null, "no granite_modeling_logits.json: run harness/python/granite_modeling_fixture.py")
        val fixture = parseJson(text!!) as JsonObject
        val reference = Reference(weights)
        var worst = 0.0
        for (p in (fixture.fields.getValue("prompts") as JsonArray).elements.map { it as JsonObject }) {
            val ids = (p.fields.getValue("ids") as JsonArray).asIntList("ids")
            val want = (p.fields.getValue("logits") as JsonArray).elements.map { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
            val got = reference.logits(ids)
            assertEquals(want.size, got.size)
            for (t in want.indices) for (v in want[t].indices) {
                val e = abs(got[t][v] - want[t][v])
                worst = maxOf(worst, e)
                assertTrue(e <= TOL, "prompt $ids, position $t, logit $v: ${got[t][v]} vs modeling_granite.py's ${want[t][v]}")
            }
        }
        println("tiny Granite against modeling_granite.py: worst |logit difference| $worst")
    }

    /** Whichever the prefill's piece size, the ids and every logit are the reference's. */
    @Test
    fun greedyDecodingMatchesTheReferenceLogitForLogit() {
        for (piece in listOf(36, 5)) check(piece)
    }

    private fun check(chunk: Int) {
        val (config, staged) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs)
        val bucket = DecodeBucket(1, context)
        val prefill = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket, DecodeGraphKind.PREFILL, prefillChunk = chunk), config)
        val decode = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, bucket), config)
        val poolTypes = HfDecoderGraph.spec(config, model, bucket).inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN }.map { it.type }
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val reference = Reference(weights)
        var worst = 0.0
        for (prompt in PROMPTS) {
            var pools: List<FloatArray> = poolTypes.map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
            var last = FloatArray(0)
            for (from in prompt.indices step chunk) {
                val n = minOf(chunk, prompt.size - from)
                val pad = chunk - n
                val out = DxirInterpreter.evalFunction(prefill, buildList {
                    add(FloatArray(chunk) { if (it < pad) 0f else prompt[from + it - pad].toFloat() })
                    add(FloatArray(chunk) { if (it < pad) 0f else (from + it - pad).toFloat() })
                    add(tables)
                    add(floatArrayOf((from + n).toFloat()))
                    add(FloatArray(chunk) { if (it < pad) -1f else (bs + from + it - pad).toFloat() })
                    addAll(pools)
                    addAll(staged)
                })
                pools = out.drop(1)
                last = out[0]
            }
            val seq = ArrayList(prompt)
            repeat(MAX_NEW) { step ->
                val want = reference.logits(seq).last()
                for (v in want.indices) {
                    val e = abs(last[v] - want[v])
                    worst = maxOf(worst, e)
                    assertTrue(e <= TOL, "pieces of $chunk, step $step, logit $v: ${last[v]} vs ${want[v]}")
                }
                val next = want.indices.maxByOrNull { want[it] }!!
                assertEquals(next, last.indices.maxByOrNull { last[it] }!!, "pieces of $chunk, step $step")
                seq += next
                if (step < MAX_NEW - 1) {
                    val pos = seq.size - 1
                    val out = DxirInterpreter.evalFunction(decode, buildList {
                        add(floatArrayOf(next.toFloat()))
                        add(floatArrayOf(pos.toFloat()))
                        add(tables)
                        add(floatArrayOf((pos + 1).toFloat()))
                        add(floatArrayOf((bs + pos).toFloat()))
                        addAll(pools)
                        addAll(staged)
                    })
                    pools = out.drop(1)
                    last = out[0]
                }
            }
        }
        println("tiny Granite parity, prefill pieces of $chunk: ids exact, worst |logit difference| $worst")
    }

    /** A sequence's logits from its start; the flags turn a multiplier off (to 1, or to 1 / sqrt(head_dim)). */
    internal class Reference(
        private val w: Map<String, FloatArray>,
        private val embedding: Boolean = true,
        private val attention: Boolean = true,
        private val residual: Boolean = true,
        private val logits: Boolean = true,
    ) {
        private fun rms(x: DoubleArray, gain: FloatArray): DoubleArray {
            val r = 1.0 / sqrt(x.sumOf { it * it } / x.size + EPS)
            return DoubleArray(x.size) { x[it] * r * gain[it] }
        }

        /** `x @ W^T` for W stored `[out, in]`. */
        private fun proj(x: DoubleArray, name: String, out: Int): DoubleArray {
            val m = w.getValue(name)
            return DoubleArray(out) { o -> var s = 0.0; for (i in x.indices) s += x[i] * m[o * x.size + i]; s }
        }

        /** One head's vector rotated at [pos], with rotate_half. */
        private fun rope(x: DoubleArray, pos: Int): DoubleArray {
            val half = x.size / 2
            return DoubleArray(x.size) { i ->
                val a = pos / THETA.pow(2.0 * (i % half) / x.size)
                val rot = if (i < half) -x[i + half] else x[i - half]
                x[i] * cos(a) + rot * sin(a)
            }
        }

        fun logits(ids: List<Int>): List<DoubleArray> {
            val n = ids.size
            val table = w.getValue("model.embed_tokens.weight")
            val scale = if (attention) ATTENTION_MULTIPLIER else 1.0 / sqrt(HD.toDouble())
            val rm = if (residual) RESIDUAL_MULTIPLIER else 1.0
            var h = List(n) { t ->
                DoubleArray(HIDDEN) { table[ids[t] * HIDDEN + it].toDouble() * if (embedding) EMBEDDING_MULTIPLIER else 1.0 }
            }
            for (l in 0 until LAYERS) {
                val p = "model.layers.$l."
                val x = h.map { rms(it, w.getValue(p + "input_layernorm.weight")) }
                fun heads(v: DoubleArray, count: Int) = List(count) { v.copyOfRange(it * HD, (it + 1) * HD) }
                val q = List(n) { t -> heads(proj(x[t], p + "self_attn.q_proj.weight", HEADS * HD), HEADS).map { rope(it, t) } }
                val k = List(n) { t -> heads(proj(x[t], p + "self_attn.k_proj.weight", KV * HD), KV).map { rope(it, t) } }
                val v = List(n) { t -> heads(proj(x[t], p + "self_attn.v_proj.weight", KV * HD), KV) }
                val attended = List(n) { t ->
                    val flat = DoubleArray(HEADS * HD)
                    for (head in 0 until HEADS) {
                        val kh = head / (HEADS / KV)
                        val scores = DoubleArray(t + 1) { s ->
                            var dot = 0.0
                            for (i in 0 until HD) dot += q[t][head][i] * k[s][kh][i]
                            dot * scale
                        }
                        val top = scores.max()
                        val sum = scores.sumOf { exp(it - top) }
                        for (s in scores.indices) {
                            val a = exp(scores[s] - top) / sum
                            for (i in 0 until HD) flat[head * HD + i] += a * v[s][kh][i]
                        }
                    }
                    proj(flat, p + "self_attn.o_proj.weight", HIDDEN)
                }
                val hA = List(n) { t -> DoubleArray(HIDDEN) { h[t][it] + rm * attended[t][it] } }
                h = List(n) { t ->
                    val xm = rms(hA[t], w.getValue(p + "post_attention_layernorm.weight"))
                    val g = proj(xm, p + "mlp.gate_proj.weight", INTER)
                    val u = proj(xm, p + "mlp.up_proj.weight", INTER)
                    val act = DoubleArray(INTER) { i -> g[i] / (1.0 + exp(-g[i])) * u[i] }
                    val out = proj(act, p + "mlp.down_proj.weight", HIDDEN)
                    DoubleArray(HIDDEN) { hA[t][it] + rm * out[it] }
                }
            }
            val div = if (logits) LOGITS_SCALING else 1.0
            return h.map { r -> proj(rms(r, w.getValue("model.norm.weight")), "lm_head.weight", VOCAB).map { it / div }.toDoubleArray() }
        }
    }

    internal companion object {
        const val VOCAB = 64
        const val HIDDEN = 32
        const val INTER = 48
        const val LAYERS = 3
        const val HEADS = 4
        const val KV = 2
        const val HD = 8
        const val THETA = 10000.0
        const val EPS = 1e-5
        const val EMBEDDING_MULTIPLIER = 12.0
        const val ATTENTION_MULTIPLIER = 0.0625
        const val RESIDUAL_MULTIPLIER = 0.22
        const val LOGITS_SCALING = 8.0
        const val MAX_NEW = 8
        const val TOL = 2e-4
        const val GREEDY_NEW = 12
        // Chosen for their top-1 margins: the logits divided by 8 leave many steps closer than a device's rounding.
        val GREEDY_PROMPTS = listOf(
            listOf(49, 18, 55, 4, 18, 7, 4, 54, 43, 19, 41),
            listOf(10, 2, 18, 37, 54, 19),
            listOf(27, 22, 43, 23, 9, 44),
            listOf(48, 33, 59, 38),
            listOf(
                33, 31, 55, 19, 40, 48, 2, 51, 27, 2, 28, 34, 50, 7, 23, 31, 46, 4, 35, 37, 14, 46, 56, 53, 6, 37, 53, 19, 11, 28,
            ),
        )
        val PROMPTS = listOf(
            listOf(3, 5, 9, 12, 33, 7),
            listOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41, 42, 43, 8, 9, 10, 44, 45),
        )

        /** The tiny checkpoint in [dir]; returns its tensors. */
        fun writeTinyGranite(dir: Path): Map<String, FloatArray> {
            val rng = Random(20261107)
            fun linear(o: Int, i: Int) = FloatArray(o * i) { (0.25 * rng.nextGaussian()).toFloat() }
            fun gain(n: Int) = FloatArray(n) { (1.0 + 0.3 * rng.nextGaussian()).toFloat() }
            val w = LinkedHashMap<String, Pair<IntArray, FloatArray>>()
            fun put(name: String, data: FloatArray, vararg dims: Int) { w[name] = dims to data }
            put("model.embed_tokens.weight", linear(VOCAB, HIDDEN), VOCAB, HIDDEN)
            put("lm_head.weight", linear(VOCAB, HIDDEN), VOCAB, HIDDEN)
            put("model.norm.weight", gain(HIDDEN), HIDDEN)
            for (l in 0 until LAYERS) {
                val p = "model.layers.$l."
                put(p + "self_attn.q_proj.weight", linear(HEADS * HD, HIDDEN), HEADS * HD, HIDDEN)
                put(p + "self_attn.k_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
                put(p + "self_attn.v_proj.weight", linear(KV * HD, HIDDEN), KV * HD, HIDDEN)
                put(p + "self_attn.o_proj.weight", linear(HIDDEN, HEADS * HD), HIDDEN, HEADS * HD)
                put(p + "mlp.gate_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
                put(p + "mlp.up_proj.weight", linear(INTER, HIDDEN), INTER, HIDDEN)
                put(p + "mlp.down_proj.weight", linear(HIDDEN, INTER), HIDDEN, INTER)
                put(p + "input_layernorm.weight", gain(HIDDEN), HIDDEN)
                put(p + "post_attention_layernorm.weight", gain(HIDDEN), HIDDEN)
            }
            SafetensorsFileWriter.write(
                dir.resolve("model.safetensors"),
                w.map { (name, t) -> SafetensorsTensor(name, F32, t.first, HostF32Storage(t.second)) },
            )
            Files.writeString(dir.resolve("config.json"), CONFIG)
            return w.mapValues { it.value.second }
        }

        val CONFIG = """
            {"architectures": ["GraniteForCausalLM"], "model_type": "granite", "dtype": "float32",
             "vocab_size": $VOCAB, "hidden_size": $HIDDEN, "intermediate_size": $INTER, "num_hidden_layers": $LAYERS,
             "num_attention_heads": $HEADS, "num_key_value_heads": $KV, "max_position_embeddings": 128,
             "rms_norm_eps": $EPS, "rope_parameters": {"rope_theta": $THETA, "rope_type": "default"}, "hidden_act": "silu",
             "attention_bias": false, "mlp_bias": false, "attention_dropout": 0.0,
             "embedding_multiplier": $EMBEDDING_MULTIPLIER, "attention_multiplier": $ATTENTION_MULTIPLIER,
             "residual_multiplier": $RESIDUAL_MULTIPLIER, "logits_scaling": $LOGITS_SCALING,
             "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "use_cache": true}
        """.trimIndent()
    }
}
