package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import org.junit.jupiter.api.Assumptions.assumeTrue
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.ir.passes.DxirInterpreter
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
 * The ricursive family's decode graph against [Reference], a dense forward pass in doubles
 * written from the arithmetic of the checkpoint's modeling code, on a tiny random
 * checkpoint: four sliding layers with a window of 8 and one full layer without RoPE, six
 * query heads over three KV heads. The reference is this repo's reading of that code, not
 * the code itself, so [eachFeatureMovesTheLogits] shows the comparison would see a feature
 * dropped; it cannot show the reading is right, which the real checkpoint's outputs do.
 */
class HfRippleParityTest {

    private val dir: Path = Files.createTempDirectory("tiny-ripple")
    private val weights: Map<String, FloatArray> = writeTinyRipple(dir)

    @AfterTest
    fun removeCheckpoint() {
        dir.toFile().deleteRecursively()
    }

    @Test
    fun theTinyCheckpointReadsAsRicursive() {
        HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config
            assertEquals(HfModelFamily.Ricursive, c.family)
            assertEquals(CLAMP, c.ffnClampThreshold)
            assertEquals(EMBED_SCALE, c.embedScale)
            assertEquals(AttentionKind.SLIDING, c.layer(0).attention)
            assertEquals(WINDOW, c.layer(0).slidingWindow)
            assertTrue(c.layer(0).rope && c.layer(0).valueResidual && c.layer(0).attentionOutputGate)
            assertEquals(AttentionKind.FULL, c.layer(4).attention)
            assertTrue(!c.layer(4).rope)
            // Every tensor is read.
            assertEquals(emptyList(), ckpt.verifyInventory())
        }
    }

    /** The committed `tiny_ripple` fixture (other modules' tests and the modeling-code oracle read it) is this checkpoint. */
    @Test
    fun theCommittedFixtureIsThisCheckpoint() {
        val fixture = Path.of("src/jvmTest/resources/io/tlaloc/ir/inference/tiny_ripple")
        for (f in listOf("config.json", "model.safetensors")) {
            assertTrue(Files.readAllBytes(dir.resolve(f)).contentEquals(Files.readAllBytes(fixture.resolve(f))), "$f differs from $fixture/$f")
        }
    }

    /**
     * `tiny_ripple/greedy.json`, what triton/verify.sh's Ripple lane expects of the served model:
     * for each prompt, its ids and [GREEDY_NEW] greedy ids by [Reference], and the smallest top-1
     * margin among those steps (large enough that TF32 dots do not change a choice).
     * TLALOC_WRITE_TINY_RIPPLE_GREEDY=1 writes it.
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
            assertTrue(margin > 0.015, "prompt $i: a top-1 margin of $margin is too close for a device's rounding")
            if (i > 0) out.append(',')
            out.append("\"p$i\": [${prompt.joinToString(",", "[", "]")}, ${seq.drop(prompt.size).joinToString(",", "[", "]")}, $margin]")
        }
        out.append("}\n")
        val file = Path.of("src/jvmTest/resources/io/tlaloc/ir/inference/tiny_ripple/greedy.json")
        if (System.getenv("TLALOC_WRITE_TINY_RIPPLE_GREEDY") == "1") Files.writeString(file, out)
        assertEquals(out.toString(), Files.readString(file))
    }

    /** The released instruct and Base configs (revisions e27e1717 and 5d72c8d4) read as the same 65B layout. */
    @Test
    fun theReleasedConfigsReadAsRicursive() {
        for (name in listOf("ripple_65b_preview_config.json", "ripple_65b_base_preview_config.json")) {
            val c = HfDecoderConfig.parse(Files.readString(Path.of("src/jvmTest/resources/io/tlaloc/ir/inference", name)))
            assertEquals(HfModelFamily.Ricursive, c.family, name)
            assertEquals(80, c.numLayers)
            assertEquals(5120, c.hiddenSize)
            assertEquals(40960, c.intermediateSize)
            assertEquals(100352, c.vocabSize)
            assertEquals(20.0, c.ffnClampThreshold)
            for (l in 0 until c.numLayers) {
                val s = c.layer(l)
                val full = l % 5 == 4
                assertEquals(if (full) AttentionKind.FULL else AttentionKind.SLIDING, s.attention, "$name layer $l")
                assertEquals(if (full) null else 1024, s.slidingWindow)
                assertEquals(!full, s.rope)
                assertTrue(s.valueResidual && s.attentionOutputGate && s.postAttentionOutputNorm && s.postFeedforwardNorm)
            }
        }
    }

    @Test
    fun anUnreadKeyOrABf16GateIsRefusedByName() {
        val unknown = CONFIG.replaceFirst("\"use_cache\"", "\"use_new_thing\": true, \"use_cache\"")
        assertTrue("use_new_thing" in assertFailsWith<JsonException> { HfDecoderConfig.parse(unknown) }.message!!)
        val low = CONFIG.replace("\"gate_full_precision\": true", "\"gate_full_precision\": false")
        assertTrue("gate_full_precision" in assertFailsWith<JsonException> { HfDecoderConfig.parse(low) }.message!!)
    }

    /** The clamp cuts values, and turning off any one feature moves the logits far past [TOL]. */
    @Test
    fun eachFeatureMovesTheLogits() {
        val ids = PROMPTS[1]
        val full = Reference(weights)
        full.logits(ids)
        assertTrue(full.clampedGate > 0 && full.clampedUp > 0, "the clamp never fired")
        val base = full.logits(ids)
        for ((what, other) in listOf(
            "clamp" to Reference(weights, clamp = false),
            "value residual" to Reference(weights, valueResidual = false),
            "embed scale" to Reference(weights, embedScale = false),
            "gate" to Reference(weights, gate = false),
            "window" to Reference(weights, window = false),
            "NoPE" to Reference(weights, nope = false),
        )) {
            val moved = other.logits(ids).last().indices.maxOf { abs(other.logits(ids).last()[it] - base.last()[it]) }
            assertTrue(moved > 100 * TOL, "$what moves the last logits only $moved")
        }
    }

    /**
     * [Reference] against Ricursive's own `modeling_ricursive.py` run on this checkpoint
     * (`harness/python/ripple_modeling_fixture.py`), every logit at every position: the check
     * that the reading of that code is right, which the other tests take as given.
     */
    @Test
    fun theReferenceIsRicursivesModelingCode() {
        val text = javaClass.getResourceAsStream("ripple_modeling_logits.json")?.use { it.readBytes().decodeToString() }
        assumeTrue(
            text != null,
            "no ripple_modeling_logits.json: run harness/python/ripple_modeling_fixture.py (it runs the checkpoint's modeling code)",
        )
        val fixture = parseJson(text!!) as JsonObject
        val reference = Reference(weights)
        var worst = 0.0
        for (p in fixture.arr("prompts").elements.map { it as JsonObject }) {
            val ids = p.arr("ids").asIntList("ids")
            val want = p.arr("logits").elements.map { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
            val got = reference.logits(ids)
            assertEquals(want.size, got.size)
            for (t in want.indices) for (v in want[t].indices) {
                val e = abs(got[t][v] - want[t][v])
                worst = maxOf(worst, e)
                assertTrue(e <= TOL, "prompt $ids, position $t, logit $v: ${got[t][v]} vs modeling_ricursive.py's ${want[t][v]}")
            }
        }
        println("tiny Ripple against modeling_ricursive.py: worst |logit difference| $worst")
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
            // Right-aligned pieces, each row of a piece at its position.
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
        println("tiny Ripple parity, prefill pieces of $chunk: ids exact, worst |logit difference| $worst")
    }

    /** A sequence's logits from its start, the mask giving each window; the flags turn a feature off. */
    internal class Reference(
        private val w: Map<String, FloatArray>,
        private val clamp: Boolean = true,
        private val valueResidual: Boolean = true,
        private val embedScale: Boolean = true,
        private val gate: Boolean = true,
        private val window: Boolean = true,
        private val nope: Boolean = true,
    ) {
        /** The output of each layer, each row, in the last [logits] call. */
        var layerOutputs: List<List<DoubleArray>> = emptyList()
        var clampedGate = 0
        var clampedUp = 0

        private fun rms(x: DoubleArray, gain: FloatArray): DoubleArray {
            val r = 1.0 / sqrt(x.sumOf { it * it } / x.size + EPS)
            return DoubleArray(x.size) { x[it] * r * gain[it] }
        }

        /** `x @ W^T` for W stored `[out, in]`. */
        private fun proj(x: DoubleArray, name: String, out: Int): DoubleArray {
            val m = w.getValue(name)
            return DoubleArray(out) { o -> var s = 0.0; for (i in x.indices) s += x[i] * m[o * x.size + i]; s }
        }

        private fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))

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
            val x0 = List(n) { t ->
                DoubleArray(HIDDEN) { table[ids[t] * HIDDEN + it].toDouble() * if (embedScale) EMBED_SCALE else 1.0 }
            }
            var h = x0
            val outputs = ArrayList<List<DoubleArray>>()
            for (l in 0 until LAYERS) {
                val p = "model.layers.$l."
                val sliding = l % 5 != 4
                val x = h.map { rms(it, w.getValue(p + "input_layernorm.weight")) }
                val blend = sigmoid(w.getValue(p + "self_attn.value_res_lambda")[0].toDouble())
                fun heads(v: DoubleArray, count: Int) = List(count) { v.copyOfRange(it * HD, (it + 1) * HD) }
                val rotate = sliding || !nope
                val q = List(n) { t ->
                    heads(proj(x[t], p + "self_attn.q_proj.weight", HEADS * HD), HEADS)
                        .map { rms(it, w.getValue(p + "self_attn.q_norm.weight")) }
                        .map { if (rotate) rope(it, t) else it }
                }
                val k = List(n) { t ->
                    heads(proj(x[t], p + "self_attn.k_proj.weight", KV * HD), KV)
                        .map { rms(it, w.getValue(p + "self_attn.k_norm.weight")) }
                        .map { if (rotate) rope(it, t) else it }
                }
                val v = List(n) { t ->
                    val input = if (valueResidual) DoubleArray(HIDDEN) { x0[t][it] + blend * (x[t][it] - x0[t][it]) } else x[t]
                    heads(proj(input, p + "self_attn.v_proj.weight", KV * HD), KV)
                }
                val attended = List(n) { t ->
                    val first = if (sliding && window) maxOf(0, t - WINDOW + 1) else 0
                    val flat = DoubleArray(HEADS * HD)
                    for (head in 0 until HEADS) {
                        val kh = head / (HEADS / KV)
                        val scores = DoubleArray(t - first + 1) { s ->
                            var dot = 0.0
                            for (i in 0 until HD) dot += q[t][head][i] * k[first + s][kh][i]
                            dot / sqrt(HD.toDouble())
                        }
                        val top = scores.max()
                        val sum = scores.sumOf { exp(it - top) }
                        for (s in scores.indices) {
                            val a = exp(scores[s] - top) / sum
                            for (i in 0 until HD) flat[head * HD + i] += a * v[first + s][kh][i]
                        }
                    }
                    if (gate) {
                        val g = proj(x[t], p + "self_attn.g_proj.weight", HEADS * HD)
                        for (i in flat.indices) flat[i] *= sigmoid(g[i])
                    }
                    rms(proj(flat, p + "self_attn.o_proj.weight", HIDDEN), w.getValue(p + "post_attention_layernorm.weight"))
                }
                val hA = List(n) { t -> DoubleArray(HIDDEN) { h[t][it] + attended[t][it] } }
                h = List(n) { t ->
                    val xm = rms(hA[t], w.getValue(p + "pre_feedforward_layernorm.weight"))
                    val g = proj(xm, p + "mlp.gate_proj.weight", INTER)
                    val u = proj(xm, p + "mlp.up_proj.weight", INTER)
                    val act = DoubleArray(INTER) { i ->
                        var gi = g[i]
                        var ui = u[i]
                        if (gi > CLAMP) clampedGate++
                        if (abs(ui) > CLAMP) clampedUp++
                        if (clamp) {
                            gi = minOf(gi, CLAMP)
                            ui = ui.coerceIn(-CLAMP, CLAMP)
                        }
                        gi * sigmoid(gi) * ui
                    }
                    val out = rms(proj(act, p + "mlp.down_proj.weight", HIDDEN), w.getValue(p + "post_feedforward_layernorm.weight"))
                    DoubleArray(HIDDEN) { hA[t][it] + out[it] }
                }
                outputs += h
            }
            layerOutputs = outputs
            return h.map { proj(rms(it, w.getValue("model.norm.weight")), "lm_head.weight", VOCAB) }
        }
    }

    internal companion object {
        const val VOCAB = 64
        const val HIDDEN = 24
        const val INTER = 48
        const val LAYERS = 5
        const val HEADS = 6
        const val KV = 3
        const val HD = 8
        const val WINDOW = 8
        const val THETA = 10000.0
        const val EPS = 1e-6
        const val CLAMP = 0.75
        const val EMBED_SCALE = 4.898979485566356
        const val MAX_NEW = 8
        const val TOL = 2e-4
        const val GREEDY_NEW = 12
        val GREEDY_PROMPTS = listOf(
            listOf(3, 5, 9, 12, 33, 7),
            listOf(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            listOf(44, 33, 22, 11),
            listOf(5, 50, 5, 50, 5, 50, 5),
            listOf(8, 16, 24, 32, 40, 48, 56),
            // Long enough that the draft's ring (28 positions in verify.sh's artifact) wraps.
            List(34) { (7 * it + 2) % 56 },
        )
        val PROMPTS = listOf(
            listOf(3, 5, 9, 12, 33, 7),
            listOf(3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41, 42, 43, 8, 9, 10, 44, 45),
        )

        /** The tiny checkpoint in [dir]; returns its tensors. */
        fun writeTinyRipple(dir: Path): Map<String, FloatArray> {
            val rng = Random(20261105)
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
            SafetensorsFileWriter.write(
                dir.resolve("model.safetensors"),
                w.map { (name, t) -> SafetensorsTensor(name, F32, t.first, HostF32Storage(t.second)) },
            )
            Files.writeString(dir.resolve("config.json"), CONFIG)
            return w.mapValues { it.value.second }
        }

        val CONFIG = """
            {"architectures": ["RicursiveForCausalLM"], "model_type": "ricursive", "dtype": "float32",
             "vocab_size": $VOCAB, "hidden_size": $HIDDEN, "intermediate_size": $INTER, "num_hidden_layers": $LAYERS,
             "num_attention_heads": $HEADS, "num_key_value_heads": $KV, "head_dim": $HD,
             "max_position_embeddings": 128, "rms_norm_eps": $EPS, "rope_theta": $THETA, "sliding_window": $WINDOW,
             "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
             "hybrid_layer_pattern": [1, 1, 1, 1, 0], "is_hybrid_swa": true, "hidden_act": "silu",
             "use_gated_attention": true, "gate_full_precision": true, "use_valres": true, "ffn_clamp_threshold": $CLAMP,
             "embed_scale": $EMBED_SCALE, "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": [2],
             "pad_token_id": 0, "use_cache": true}
        """.trimIndent()
    }
}
