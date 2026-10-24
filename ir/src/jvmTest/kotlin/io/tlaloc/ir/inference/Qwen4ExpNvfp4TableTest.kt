package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import io.tlaloc.core.f8e4m3fnToFloat
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.Nvfp4EmbeddingAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The n-gram table as NVFP4 ([HfDecoderConfig.pleTableQuant]): an FP8 table rounded
 * shard by shard ([PleTableNvfp4.Fp8Rows]) gives [Nvfp4Quantizer]'s codes and scales of
 * the widened table; and the tiny Qwen4-Exp checkpoint with its table as NVFP4
 * ([io.tlaloc.ir.OpKind.NVFP4_EMBEDDING]) gives, in the interpreter, the logits of the
 * same model with an f32 table holding the NVFP4 table's values, bit for bit.
 */
class Qwen4ExpNvfp4TableTest {

    private val dir: Path = Path.of(javaClass.getResource("tiny_qwen4_exp/config.json")!!.toURI()).parent
    private val fixture = parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject

    @Test
    fun fp8RowsRoundAsTheQuantizerDoes() {
        val rnd = Random(7)
        for ((dim, scale) in listOf(160 to 0.0123f, 4 to 3.5e-4f, 32 to 1f)) {
            val rows = 300
            // Every code but NaN, and some groups all zero.
            val src = ByteArray(rows * dim) { i ->
                if ((i / 16) % 37 == 5) 0 else {
                    var b: Int
                    do { b = rnd.nextInt(256) } while (b and 0x7F == 0x7F)
                    b.toByte()
                }
            }
            val q = PleTableNvfp4.Fp8Rows(scale, dim)
            val codes = ByteArray(rows * dim / 2)
            val scales = ByteArray(rows * (dim / q.group))
            q.round(src, 0, rows, codes, 0, scales, 0)
            val widened = FloatArray(src.size) { f8e4m3fnToFloat(src[it]) * scale }
            val want = Nvfp4Quantizer.quantize(widened, rows, dim, q.group, scale / 6f)
            assertContentEquals(want.codes, codes, "codes, rows of $dim")
            assertContentEquals(want.scales, scales, "group scales, rows of $dim")
            assertEquals(want.scale2, q.scale2)
            // The largest value of the table is kept exactly: 448 s is 6 x e4m3(448) x s / 6.
            val max = Nvfp4EmbeddingAttrs.reference(
                Nvfp4EmbeddingAttrs.Parsed(1, 16, 16, 1),
                FloatArray(8) { 0x77.toFloat() },
                floatArrayOf(io.tlaloc.core.floatToF8e4m3fn(448f).toInt().and(0xFF).toFloat()),
                scale / 6f, intArrayOf(0),
            )[0]
            assertEquals(448f * scale, max, 448f * scale * 1e-6f)
        }
    }

    @Test
    fun anNvfp4TableGivesTheLogitsOfItsWidenedTable() {
        val (fp4, fp4Weights, f32, f32Weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32)
            val q = c.copy(pleTableQuant = WeightQuant.NVFP4)
            val qw = HfStagedWeights.stage(ckpt, q)
            // The f32 model's table: the NVFP4 table's values.
            val qSlots = HfDecoderGraph.weightSlots(q)
            val l = c.ple!!.layers.single()
            fun at(name: String) = qSlots.indexOfFirst { it.name == name }
            val codes = at("pleTable$l")
            val table = qSlots[codes].type.dims
            assertEquals(listOf(c.ple!!.tableRows(0), c.ple!!.headDim / 2), table)
            val p = Nvfp4EmbeddingAttrs.Parsed(table[0], 2 * table[1], 2 * table[1] / qSlots[at("pleTable${l}Fp4Scale")].type.dims[1], table[0])
            val widened = Nvfp4EmbeddingAttrs.reference(p, qw[codes], qw[at("pleTable${l}Fp4Scale")], qw[at("pleTable${l}Scale2")][0], IntArray(table[0]) { it })
            val w = HfStagedWeights.stage(ckpt, c).toMutableList()
            val f32Table = HfDecoderGraph.weightSlots(c).indexOfFirst { it.name == "pleTable$l" }
            val stored = w[f32Table]
            assertEquals(stored.size, widened.size)
            var err = 0.0
            for (i in stored.indices) err = maxOf(err, abs(stored[i] - widened[i]).toDouble())
            println("tiny Qwen4-Exp n-gram table as NVFP4: worst |value - stored| $err, the stored values up to ${stored.maxOf { abs(it) }}")
            w[f32Table] = widened
            Quad(q, qw, c, w)
        }
        var worstTransformers = 0.0
        for (p in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
            fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
            val prompt = ints("promptTokens")
            val steps = ints("generatedTokens").size
            val a = greedy(fp4, fp4Weights, prompt, steps)
            val b = greedy(f32, f32Weights, prompt, steps)
            assertEquals(b.first, a.first, "ids for prompt $prompt")
            for (s in a.second.indices) assertContentEquals(b.second[s], a.second[s], "logits of step $s for prompt $prompt")
            val wantLogits = (p["logits"] as JsonArray).elements.map { row -> (row as JsonArray).elements.map { (it as JsonNumber).value } }
            for (s in a.second.indices) for (v in a.second[s].indices) worstTransformers = maxOf(worstTransformers, abs(a.second[s][v] - wantLogits[s][v]))
            println("prompt of ${prompt.size}: ids ${a.first} (transformers ${ints("generatedTokens")})")
        }
        println("tiny Qwen4-Exp with an NVFP4 n-gram table: logits equal the widened table's; worst |logit - transformers| $worstTransformers")
        val g = HfDecoderGraph.build(HfDecoderGraph.spec(fp4, fp4.toDecodeModelShape(numBlocks = 10, blockSize = 4, stateSlots = 2), DecodeBucket(1, 36)), fp4)
        val ops = g.body.filterIsInstance<io.tlaloc.ir.DxirOp>()
        assertTrue(ops.any { it.op == OpKind.NVFP4_EMBEDDING } && ops.none { it.op == OpKind.EMBEDDING && it.operands[0].type.dims[0] == fp4.ple!!.tableRows(0) })
    }

    private data class Quad(val a: HfDecoderConfig, val aw: List<FloatArray>, val b: HfDecoderConfig, val bw: List<FloatArray>)

    /** Greedy ids and every step's logits for one prompt, prefilled whole and decoded. */
    private fun greedy(config: HfDecoderConfig, weights: List<FloatArray>, prompt: List<Int>, steps: Int): Pair<List<Int>, List<FloatArray>> {
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs, stateSlots = 2)
        fun build(kind: DecodeGraphKind): DxirFunction =
            HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
        val prefill = build(DecodeGraphKind.PREFILL)
        val decode = build(DecodeGraphKind.DECODE)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context))
        var pools: List<FloatArray> = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN || it.role == DecodeSlotRole.STATE_POOL_IN }
            .map { FloatArray(it.type.dims.fold(1) { a, b -> a * b }) }
        val pad = context - prompt.size
        val tables = FloatArray(context / bs) { (1 + it).toFloat() }
        val out = DxirInterpreter.evalFunction(prefill, buildList {
            add(FloatArray(context) { if (it < pad) 0f else prompt[it - pad].toFloat() })
            add(FloatArray(context) { if (it < pad) 0f else (it - pad).toFloat() })
            add(tables)
            add(floatArrayOf(prompt.size.toFloat()))
            add(FloatArray(context) { if (it < pad) -1f else (bs + it - pad).toFloat() })
            add(floatArrayOf(1f))
            addAll(pools)
            addAll(weights)
        })
        pools = out.drop(1)
        val logits = arrayListOf(out[0])
        fun argmax(v: FloatArray) = v.indices.maxByOrNull { v[it] }!!
        val got = arrayListOf(argmax(out[0]))
        while (got.size < steps) {
            val pos = prompt.size + got.size - 1
            val step = DxirInterpreter.evalFunction(decode, buildList {
                add(floatArrayOf(got.last().toFloat()))
                add(floatArrayOf(pos.toFloat()))
                add(tables)
                add(floatArrayOf((pos + 1).toFloat()))
                add(floatArrayOf((bs + pos).toFloat()))
                add(floatArrayOf(1f))
                addAll(pools)
                addAll(weights)
            })
            pools = step.drop(1)
            logits += step[0]
            got += argmax(step[0])
        }
        return got to logits
    }
}
