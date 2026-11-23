package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.logSoftmax
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.hostF32
import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Shape
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [InklingLoop.forward] against transformers 5.24.0's modeling code on `tiny_inkling` (written by
 * `harness/python/inkling_modeling_fixture.py`: eight layers in blocks of two, layers 2-5 looped
 * three times (two blocks, so later passes' messages mix two states), full attention at layers 3 and 7, a window of 8 and a relative extent of 6 shorter
 * than the 19-token prompt, log scaling from position 3, clipping thresholds that clip). The logits
 * are compared at every position.
 */
class InklingLoopTest {
    private val dir: Path = Path.of("src/jvmTest/resources/io/tlaloc/nn/tiny_inkling")

    private fun logits(ck: InklingCheckpoint, weights: Map<String, DTensor<*, *>>, ids: IntArray): FloatArray {
        val keys = weights.keys.toList()
        var out: Tracer<*>? = null
        captureN(listOf(DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(ids.size), I32)) + weights.values) { leaves ->
            val w = InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to leaves[1 + i] })
            InklingLoop.forward(ck.config, leaves[0], w).also { out = it }
        }
        return FloatArray(out!!.size) { out!!.peek(it) }
    }

    @Test
    fun theTinyCheckpointReadsAsLooped() {
        InklingCheckpoint(dir).use { ck ->
            val c = ck.config
            assertEquals(listOf(true, true, true, false, true, true, true, false), c.sliding)
            assertEquals(Triple(2, 6, 3), Triple(c.loopStart, c.loopEnd, c.passes))
            assertEquals(5, c.gateRank)
            assertEquals(128, c.unpaddedVocabSize)
        }
    }

    /** NVFP4 and MXFP8 expert stacks decoded as transformers' `Inkling2LoopDequantizeExperts` decodes them (to bf16). */
    @Test
    fun quantizedExpertsDecodeAsTransformersDoes() {
        val want = parseJson(Files.readString(dir.resolve("quantized_experts.json"))) as JsonObject
        val e = (want["experts"] as JsonNumber).value.toInt()
        val rows = (want["rows"] as JsonNumber).value.toInt()
        val cols = (want["cols"] as JsonNumber).value.toInt()
        io.tlaloc.core.io.SafetensorsFile.open(dir.resolve("quantized_experts.safetensors")).use { f ->
            fun load(n: String) = if (n in f.names) f.load(n) else null
            for ((name, key) in listOf("nv.w" to "nvfp4", "mx.w" to "mxfp8")) {
                val got = InklingCheckpoint.dequantizeExperts(name, e, rows, cols, ::load)
                val exp = (want[key] as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }
                assertEquals(exp.size, got.size)
                // The op returns the model's dtype, bf16 here (as the released model loads): compare at bf16.
                for (i in exp.indices) {
                    assertEquals(exp[i], io.tlaloc.core.bf16BitsToFloat(io.tlaloc.core.floatToBf16Bits(got[i])), "$key element $i")
                }
            }
        }
    }

    /**
     * Token by token through [InklingLoop.decodeStep] (caches of 32 rows, the state threaded from
     * step to step): each step's logits equal the modeling code's at that position.
     */
    @Test
    fun decodeStepsMatchTheModelingCode() {
        val fixture = parseJson(Files.readString(dir.resolve("logits.json"))) as JsonObject
        val p = (fixture["prompts"] as JsonArray).elements[1] as JsonObject
        val ids = (p["ids"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }.toIntArray()
        val want = (p["logits"] as JsonArray).elements.map { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
        val ctx = 32
        InklingCheckpoint(dir).use { ck ->
            val c = ck.config
            val weights = ck.hostWeights()
            val keys = weights.keys.toList()
            var state: List<DTensor<*, *>> = InklingLoop.stateDims(c, ctx).map { d ->
                DTensor<Shape, io.tlaloc.core.F32>(io.tlaloc.core.HostF32Storage(FloatArray(d.fold(1) { a, b -> a * b })), d, io.tlaloc.core.F32)
            }
            var worst = 0.0
            for ((pos, id) in ids.withIndex()) {
                var outs: List<Tracer<Shape>> = emptyList()
                val inputs = listOf<DTensor<*, *>>(
                    DTensor<Shape, I32>(HostI32Storage(intArrayOf(id)), intArrayOf(1), I32),
                    DTensor<Shape, io.tlaloc.core.F32>(io.tlaloc.core.HostF32Storage(floatArrayOf(pos.toFloat())), intArrayOf(1), io.tlaloc.core.F32),
                ) + weights.values + state
                io.tlaloc.autograd.captureMulti(inputs) { leaves ->
                    val w = InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to leaves[2 + i] })
                    InklingLoop.decodeStep(c, ctx, leaves[0], leaves[1], leaves.drop(2 + keys.size), w).also { outs = it }
                }
                val logits = outs[0]
                for (v in 0 until logits.size) worst = maxOf(worst, abs(want[pos][v] - logits.peek(v)))
                state = outs.drop(1).map { o -> DTensor<Shape, io.tlaloc.core.F32>(io.tlaloc.core.HostF32Storage(FloatArray(o.size) { o.peek(it) }), o.dims, io.tlaloc.core.F32) }
            }
            println("tiny_inkling decode steps, ${ids.size} tokens: worst logit difference $worst")
            assertTrue(worst < 5e-4, "worst logit difference $worst")
        }
    }

    /**
     * The mean next-token cross entropy of the 19-token prompt and its gradient, through the
     * compiler's reverse transform, against torch autograd on the modeling code (`grads.json`), for
     * eleven weights of different kinds: embedding, unembedding, a linear, a router, a message gate,
     * a block query, the message queries, a relative-bias bank, convolution taps, expert weights.
     */
    @Test
    fun gradientsMatchTorch() {
        val g = parseJson(Files.readString(dir.resolve("grads.json"))) as JsonObject
        val ids = (g["ids"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }.toIntArray()
        InklingCheckpoint(dir).use { ck ->
            val c = ck.config
            val weights = ck.hostWeights()
            val keys = weights.keys.toList()
            val t = ids.size
            val v = c.unpaddedVocabSize
            val target = FloatArray(t * v).also { for (i in 0 until t - 1) it[i * v + ids[i + 1]] = 1f }
            val fn = captureN(listOf(DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(t), I32)) + weights.values) { leaves ->
                val w = InklingCheckpoint.Leaves(keys.withIndex().associate { (i, k) -> k to leaves[1 + i] })
                val lp = InklingLoop.forward(c, leaves[0], w).logSoftmax(1)
                val onehot = lp.constant<Shape>(target, intArrayOf(t, v))
                (lp * onehot).sum() * (-1f / (t - 1))
            }
            val gradient = io.tlaloc.ir.passes.DxirReverseTransform.apply(fn, includeForward = true)
            @Suppress("UNCHECKED_CAST")
            val inputs = listOf(FloatArray(t) { ids[it].toFloat() }) + weights.values.map { (it as DTensor<*, io.tlaloc.core.F32>).hostF32() }
            val out = io.tlaloc.ir.passes.DxirInterpreter.evalFunction(gradient, inputs)
            val loss = out[0][0]
            println("tiny_inkling loss $loss, torch ${(g["loss"] as JsonNumber).value}")
            assertEquals((g["loss"] as JsonNumber).value, loss.toDouble(), 1e-5)
            fun ours(key: String) = out[2 + keys.indexOf(key).also { require(it >= 0) { key } }]
            fun torch(name: String): Pair<List<Int>, FloatArray> {
                val o = (g["grads"] as JsonObject)[name] as JsonObject
                return (o["shape"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() } to
                    (o["values"] as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }.toFloatArray()
            }
            // Each torch gradient in our layout.
            fun transposed(a: FloatArray, rows: Int, cols: Int) = FloatArray(a.size) { i -> a[(i % rows) * cols + i / rows] }
            val cases = mapOf(
                "model.llm.embed.weight" to torch("model.embed_tokens.weight").second,
                "model.llm.layers.0.attn.rel_logits_proj.proj" to torch("model.layers.0.self_attn.relative_position_bias.proj").second,
                "model.llm.layers.1.attn_sconv.weight" to torch("model.layers.1.post_attention_convolution.conv1d.weight").second,
                "model.llm.layers.2.controllers.attn.down_weight" to torch("model.layers.2.q_message_gate.down_proj.weight").let { (d, x) -> transposed(x, d[0], d[1]) },
                "model.llm.attn_res.attn_queries.2" to torch("model.layers.3.attention_input.query").second,
                "model.llm.layers.3.attn.wq_du.weight" to torch("model.layers.3.self_attn.q_proj.linear.weight").let { (d, x) -> transposed(x, d[0], d[1]) },
                "model.llm.layers.4.mlp.gate.weight" to torch("model.layers.4.mlp.gate.weight").let { (d, x) -> transposed(x, d[0], d[1]) },
                "model.llm.layers.7.attn.k_sconv.static_w" to torch("model.layers.7.self_attn.k_proj.sconv.static_weight").second,
                // [L, 4, H] (query, key, value, mlp) -> the checkpoint's [2L, H] query/mlp rows.
                "model.llm.middle_cycle_cross_pass_queries" to torch("model.message_input.query").let { (d, x) ->
                    val h = d[2]
                    FloatArray(2 * d[0] * h) { i -> val row = i / h; x[((row / 2) * 4 + if (row % 2 == 0) 0 else 3) * h + i % h] }
                },
                // [E, H, I] -> [E, I, H].
                "experts.2.down" to torch("model.layers.2.mlp.experts.down_proj").let { (d, x) ->
                    val (e, h, i) = Triple(d[0], d[1], d[2])
                    FloatArray(x.size) { k -> val ex = k / (i * h); val r = k % (i * h); x[ex * h * i + (r % h) * i + r / h] }
                },
                // [V, H] unpadded -> the padded rows get no gradient.
                "model.llm.unembed.weight" to torch("lm_head.weight").second.let { x -> x + FloatArray(ours("model.llm.unembed.weight").size - x.size) },
            )
            for ((key, want) in cases) {
                val got = ours(key)
                assertEquals(want.size, got.size, key)
                val scale = want.maxOf { abs(it) }.toDouble()
                var worst = 0.0
                for (i in want.indices) worst = maxOf(worst, abs((want[i] - got[i]).toDouble()))
                println("  d/d $key: worst difference $worst (largest gradient $scale)")
                assertTrue(worst <= 5e-4 * maxOf(scale, 1e-3), "$key: worst difference $worst, largest gradient $scale")
            }
        }
    }

    @Test
    fun logitsMatchTheModelingCode() {
        val fixture = parseJson(Files.readString(dir.resolve("logits.json"))) as JsonObject
        InklingCheckpoint(dir).use { ck ->
            val weights = ck.hostWeights()
            for (p in (fixture["prompts"] as JsonArray).elements) {
                p as JsonObject
                val ids = (p["ids"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }.toIntArray()
                val want = (p["logits"] as JsonArray).elements.flatMap { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
                val got = logits(ck, weights, ids)
                assertEquals(want.size, got.size)
                var worst = 0.0
                for (i in want.indices) worst = maxOf(worst, abs(want[i] - got[i]))
                println("tiny_inkling ${ids.size} tokens: worst logit difference $worst")
                assertTrue(worst < 5e-4, "worst logit difference $worst over ${ids.size} tokens")
            }
        }
    }
}
