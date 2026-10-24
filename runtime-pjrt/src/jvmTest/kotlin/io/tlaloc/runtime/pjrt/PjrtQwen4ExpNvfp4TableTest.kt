package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.U8
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.Nvfp4EmbeddingAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.inference.DecodeBucket
import io.tlaloc.ir.inference.DecodeGraphKind
import io.tlaloc.ir.inference.DecodeSlotRole
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderGraph
import io.tlaloc.ir.inference.HfStagedWeights
import io.tlaloc.ir.inference.WeightQuant
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The n-gram table as NVFP4 on the device: NVFP4_EMBEDDING's emitted form equals its
 * reference bit for bit on rows of Qwen3.8-Flash-Next's width (160 values, 10 group
 * scales), and the tiny Qwen4-Exp checkpoint with its table as NVFP4 gives the
 * interpreter's greedy ids, every logit within 1e-4, prefilled and decoded.
 */
class PjrtQwen4ExpNvfp4TableTest {

    private val dir = Path.of("..", "ir", "src", "jvmTest", "resources", "io", "tlaloc", "ir", "inference", "tiny_qwen4_exp")

    @Test
    fun nvfp4EmbeddingMatchesTheReferenceBitForBit() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val v = 5000; val d = 160; val n = 64; val heads = 16
        val fn = DxirBuilder.function("table") {
            val c = param("codes", DxirType(U8, listOf(v, d / 2)))
            val s = param("scales", DxirType(U8, listOf(v, d / 16)))
            val s2 = param("scale2", DxirType(F32, listOf(1)))
            val ids = param("ids", DxirType(I32, listOf(n, heads)))
            listOf(op(OpKind.NVFP4_EMBEDDING, listOf(c, s, s2, ids), DxirType(F32, listOf(n, heads, d))))
        }
        val rnd = Random(3)
        val codes = ByteArray(v * d / 2) { rnd.nextInt(256).toByte() }
        val scales = ByteArray(v * d / 16) { var b: Int; do { b = rnd.nextInt(256) } while (b and 0x7F == 0x7F); b.toByte() }
        val scale2 = floatArrayOf(1.7e-3f)
        val ids = IntArray(n * heads) { rnd.nextInt(v) }
        val p = Nvfp4EmbeddingAttrs.parse(fn.returns[0] as io.tlaloc.ir.DxirOp, "test")
        val want = Nvfp4EmbeddingAttrs.reference(
            p, FloatArray(codes.size) { (codes[it].toInt() and 0xFF).toFloat() }, FloatArray(scales.size) { (scales[it].toInt() and 0xFF).toFloat() },
            scale2[0], ids,
        )
        val got = TestBackend.session().use { s -> s.runOnHost(fn, listOf(codes, scales, scale2, ids))[0] as FloatArray }
        assertContentEquals(want, got)
        println("[pjrt-nvfp4-embedding] $n x $heads rows of $d == reference bit for bit")
    }

    @Test
    fun theTinyCheckpointWithAnNvfp4TableMatchesTheInterpreter() {
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val fixture = parseJson(Files.readString(dir.resolve("greedy.json"))) as JsonObject
        val (config, weights) = HfCheckpoint.open(dir).use { ckpt ->
            val c = ckpt.config.copy(weightDType = F32, pleTableQuant = WeightQuant.NVFP4)
            c to HfStagedWeights.stage(ckpt, c)
        }
        val bs = 4
        val context = 36
        val model = config.toDecodeModelShape(numBlocks = 1 + context / bs, blockSize = bs, stateSlots = 2)
        fun build(kind: DecodeGraphKind): DxirFunction = HfDecoderGraph.build(HfDecoderGraph.spec(config, model, DecodeBucket(1, context), kind), config)
        val prefill = build(DecodeGraphKind.PREFILL)
        val decode = build(DecodeGraphKind.DECODE)
        val spec = HfDecoderGraph.spec(config, model, DecodeBucket(1, context))
        val poolTypes = spec.inputs.filter { it.role == DecodeSlotRole.KV_POOL_IN || it.role == DecodeSlotRole.STATE_POOL_IN }.map { it.type }
        var worst = 0.0
        TestBackend.session().use { s ->
            for (pr in (fixture["prompts"] as JsonArray).elements.map { it as JsonObject }) {
                val prompt = (pr["promptTokens"] as JsonArray).elements.map { (it as JsonNumber).value.toInt() }
                val steps = (pr["generatedTokens"] as JsonArray).elements.size
                val pad = context - prompt.size
                val tables = FloatArray(context / bs) { (1 + it).toFloat() }
                val first = buildList {
                    add(FloatArray(context) { if (it < pad) 0f else prompt[it - pad].toFloat() })
                    add(FloatArray(context) { if (it < pad) 0f else (it - pad).toFloat() })
                    add(tables)
                    add(floatArrayOf(prompt.size.toFloat()))
                    add(FloatArray(context) { if (it < pad) -1f else (bs + it - pad).toFloat() })
                    add(floatArrayOf(1f))
                }
                var hostPools: List<FloatArray> = poolTypes.map { FloatArray(it.dims.fold(1) { a, b -> a * b }) }
                var devPools = hostPools
                val a = DxirInterpreter.evalFunction(prefill, first + hostPools + weights)
                val b = s.dev(prefill, first + devPools + weights)
                hostPools = a.drop(1); devPools = b.drop(1)
                fun argmax(v: FloatArray) = v.indices.maxByOrNull { v[it] }!!
                fun compare(x: FloatArray, y: FloatArray, step: Int) {
                    for (i in x.indices) worst = maxOf(worst, abs(x[i] - y[i]).toDouble())
                    assertEquals(argmax(x), argmax(y), "step $step of the prompt of ${prompt.size}")
                }
                compare(a[0], b[0], 0)
                val ids = arrayListOf(argmax(a[0]))
                while (ids.size < steps) {
                    val pos = prompt.size + ids.size - 1
                    val inputs = listOf(
                        floatArrayOf(ids.last().toFloat()), floatArrayOf(pos.toFloat()), tables,
                        floatArrayOf((pos + 1).toFloat()), floatArrayOf((bs + pos).toFloat()), floatArrayOf(1f),
                    )
                    val x = DxirInterpreter.evalFunction(decode, inputs + hostPools + weights)
                    val y = s.dev(decode, inputs + devPools + weights)
                    hostPools = x.drop(1); devPools = y.drop(1)
                    compare(x[0], y[0], ids.size)
                    ids += argmax(x[0])
                }
                println("[pjrt-nvfp4-table] prompt of ${prompt.size}: ids $ids on both")
            }
        }
        println("[pjrt-nvfp4-table] tiny Qwen4-Exp with an NVFP4 n-gram table: device ids == interpreter's, worst |logit difference| $worst")
        assertTrue(worst <= 1e-4, "worst logit difference $worst")
    }

    /** [fn] on the device with each input as its parameter's dtype holds it (codes as bytes), the results as floats. */
    private fun PjrtSession.dev(fn: DxirFunction, inputs: List<FloatArray>): List<FloatArray> =
        runOnHost(
            fn,
            fn.params.zip(inputs).map { (p, a) ->
                when (p.type.dtype) {
                    U8 -> ByteArray(a.size) { a[it].toInt().toByte() }
                    I32 -> IntArray(a.size) { a[it].toInt() }
                    else -> a
                }
            },
        ).map { r ->
            when (r) {
                is IntArray -> FloatArray(r.size) { r[it].toFloat() }
                is FloatArray -> r
                else -> error("unexpected result ${r::class.simpleName}")
            }
        }
}
