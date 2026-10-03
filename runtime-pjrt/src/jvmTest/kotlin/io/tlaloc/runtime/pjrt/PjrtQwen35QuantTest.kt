package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.I8
import io.tlaloc.core.floatToF8e4m3fn
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.ir.inference.WeightQuant
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Qwen3.5-0.8B with its projections quantized per output channel (int8, and
 * f8 e4m3fn), the rest f32, or with e4m3fn KV pools, greedy on the device
 * against transformers' f32 fixture: the first token is transformers', the first step's logits stay
 * within [TOL] of the largest, and the number of the 16 ids that match is
 * reported.
 */
class PjrtQwen35QuantTest {

    private val fixturePath = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "qwen3_5_0_8b_greedy.json")

    private fun run(quant: WeightQuant, kvDtype: io.tlaloc.core.DType? = null) {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val fixture = parseJson(Files.readString(fixturePath)) as JsonObject
        val dir = Path.of(
            System.getProperty("user.home"), ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots",
            (fixture["revision"] as JsonString).value,
        )
        assumeTrue(Files.isRegularFile(dir.resolve("config.json")), "no Qwen/Qwen3.5-0.8B checkpoint")
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, weightQuant = quant)
            val slots = HfDecoderGraph.weightSlots(c)
            c to HfStagedWeights.stage(ckpt, c).mapIndexed { i, v ->
                when (slots[i].type.dtype) {
                    I8 -> ByteArray(v.size) { v[it].toInt().toByte() }
                    F8E4M3FN -> ByteArray(v.size) { floatToF8e4m3fn(v[it]) }
                    else -> v
                }
            }
        }
        val bs = 8
        val nb = 6
        val model = config.toDecodeModelShape(numBlocks = nb, blockSize = bs, stateSlots = 2, kvDtype = kvDtype)
        val ctx = nb * bs
        fun build(kind: DecodeGraphKind, c: Int) = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, c), kind), config)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, ctx))
        val decode = build(DecodeGraphKind.DECODE, ctx)
        fun argmax(v: FloatArray): Int { var b = 0; for (i in v.indices) if (v[i] > v[b]) b = i; return b }
        TestBackend.session().use { s ->
            for (p in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
                fun ints(k: String) = (p[k] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
                val prompt = ints("promptTokens")
                val want = ints("generatedTokens")
                val top = (p["step1TopKIndices"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
                val topV = (p["step1TopKValues"] as JsonArray).elements.map { (it as JsonNumber).value }
                var pools: List<Any> = (0 until config.numLayers).flatMap { l ->
                    spec.poolTypesOf(l).toList().map {
                        val n = it.dims.fold(1) { a, b -> a * b }
                        if (it.dtype == F8E4M3FN) ByteArray(n) else FloatArray(n)
                    }
                }
                val c = ((prompt.size + bs - 1) / bs) * bs
                val pad = c - prompt.size
                val out = s.runOnHost(build(DecodeGraphKind.PREFILL, c), buildList {
                    add(IntArray(c) { if (it < pad) 0 else prompt[it - pad] })
                    add(IntArray(c) { if (it < pad) 0 else it - pad })
                    add(IntArray(c / bs) { it })
                    add(intArrayOf(prompt.size))
                    add(IntArray(c) { if (it < pad) -1 else it - pad })
                    add(intArrayOf(1))
                    addAll(pools)
                    addAll(weights)
                })
                val first = out[0] as FloatArray
                val denom = first.maxOf { abs(it) }
                var worst = 0.0
                for ((j, v) in top.withIndex()) worst = maxOf(worst, abs(first[v] - topV[j]) / denom)
                pools = out.drop(1)
                val got = arrayListOf(argmax(first))
                while (got.size < want.size) {
                    val pos = prompt.size + got.size - 1
                    val step = s.runOnHost(decode, buildList {
                        add(intArrayOf(got.last()))
                        add(intArrayOf(pos))
                        add(IntArray(ctx / bs) { it })
                        add(intArrayOf(pos + 1))
                        add(intArrayOf(pos))
                        add(intArrayOf(1))
                        addAll(pools)
                        addAll(weights)
                    })
                    pools = step.drop(1)
                    got += argmax(step[0] as FloatArray)
                }
                val same = got.zip(want).takeWhile { it.first == it.second }.size
                val label = quant.tag + if (kvDtype == null) "" else "-kv${kvDtype.name}"
                println("[qwen35-$label] ${(p["kind"] as JsonString).value}: $same of ${want.size} ids equal transformers' before the first difference; step-1 top-20 logits within $worst of the largest")
                assertEquals(want[0], got[0], "$label: the first token")
                assertTrue(worst < TOL, "$label: step-1 logits off by $worst")
            }
        }
    }

    @Test
    fun int8Weights() = run(WeightQuant.INT8)

    @Test
    fun fp8Weights() = run(WeightQuant.FP8)

    @Test
    fun fp8Kv() = run(WeightQuant.NONE, F8E4M3FN)

    @Test
    fun fp8WeightsAndKv() = run(WeightQuant.FP8, F8E4M3FN)

    private companion object {
        const val TOL = 0.05
    }
}
