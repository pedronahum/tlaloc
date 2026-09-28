package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TransformerLayersTest {

    private val config = CausalLmConfig(
        vocabSize = 11, dModel = 8, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 12, initStd = 0.3f,
    )

    private fun ids(values: IntArray, batch: Int, seq: Int): DTensor<*, I32> =
        DTensor<Shape, I32>(HostI32Storage(values), intArrayOf(batch, seq), I32)

    private fun f32(dims: IntArray, seed: Long): DTensor<*, F32> {
        val n = dims.fold(1) { a, d -> a * d }
        val u = uniformFloats(RandomKey.fromSeed(seed), n)
        return DTensor<Shape, F32>(HostF32Storage(FloatArray(n) { 2f * u[it] - 1f }), dims, F32)
    }

    /** Runs [layer]'s forward on [input] through the captured graph and the interpreter. */
    private fun <M> forward(layer: M, input: DTensor<*, *>): FloatArray where M : Layer, M : Trainable<M> {
        val params = layer.parameters
        val fn = captureN(listOf(input) + params.map { it.tensor }) { leaves ->
            val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
            layer.forward(leaves[0], Params { byKey.getValue(it) })
        }
        val values = listOf(values(input)) + params.map { it.tensor.hostF32() }
        return DxirInterpreter.evalFunction(fn, values).single()
    }

    @Suppress("UNCHECKED_CAST")
    private fun values(t: DTensor<*, *>): FloatArray =
        if (t.dtype == I32) (t.storage as HostI32Storage).data.let { d -> FloatArray(d.size) { d[it].toFloat() } }
        else (t as DTensor<*, F32>).hostF32()

    @Test
    fun causalLmParameterKeysFollowTheModuleTree() {
        val model = CausalLM.llama(config, RandomKey.fromSeed(1))
        val keys = model.parameters.map { it.key }
        assertEquals("embed.table", keys.first())
        assertEquals("head.w", keys.last())
        assertTrue("blocks.1.attn.k.w" in keys && "blocks.0.mlp.down.w" in keys && "norm.weight" in keys, "$keys")
        assertEquals(1 + 2 * 9 + 2, keys.size)
        val k = model.parameters.first { it.key == "blocks.0.attn.k.w" }.tensor
        assertContentEquals(intArrayOf(8, 2 * 2), k.dims, "2 kv heads of headDim 2")

        val replaced = f32(intArrayOf(8), 5)
        val updated = model.withParameters(mapOf("blocks.1.mlpNorm.weight" to replaced))
        for ((a, b) in model.parameters.zip(updated.parameters)) {
            if (a.key == "blocks.1.mlpNorm.weight") assertTrue(b.tensor === replaced)
            else assertTrue(a.tensor === b.tensor, a.key)
        }
        assertFailsWith<IllegalArgumentException> { model.withParameters(mapOf("blocks.2.attn.q.w" to replaced)) }
        assertFailsWith<IllegalArgumentException> { model.withParameters(mapOf("blocks.0.attn.x.w" to replaced)) }

        val tied = CausalLM.llama(config.copy(tiedEmbeddings = true), RandomKey.fromSeed(1))
        assertTrue(tied.parameters.none { it.key.startsWith("head.") })
    }

    @Test
    fun aTokenDoesNotChangeTheLogitsOfEarlierPositions() {
        val model = CausalLM.llama(config, RandomKey.fromSeed(2))
        val a = forward(model, ids(intArrayOf(1, 2, 3, 4, 5), 1, 5))
        val b = forward(model, ids(intArrayOf(1, 2, 3, 9, 0), 1, 5))
        val v = config.vocabSize
        for (i in 0 until 3 * v) assertEquals(a[i], b[i], "position ${i / v} sees a later token")
        assertTrue((3 * v until 5 * v).any { a[it] != b[it] }, "later positions must see the change")

        val bidirectional = CausalLM(
            model.embed,
            model.blocks.map { TransformerBlock(it.attnNorm, it.attn.withCausal(false), it.mlpNorm, it.mlp) },
            model.norm, model.head,
        )
        val c = forward(bidirectional, ids(intArrayOf(1, 2, 3, 4, 5), 1, 5))
        val d = forward(bidirectional, ids(intArrayOf(1, 2, 3, 9, 0), 1, 5))
        assertTrue((0 until v).any { c[it] != d[it] }, "without the mask position 0 sees later tokens")
    }

    @Test
    fun groupedQueryAttentionEqualsFullAttentionWithRepeatedKeyValueHeads() {
        val gqa = MultiHeadAttention(8, 4, RandomKey.fromSeed(3), numKvHeads = 2, rope = RotaryEmbedding(2))
        // Key/value head j serves query heads 2j and 2j+1: repeat its columns.
        fun repeated(w: DTensor<*, F32>): DTensor<*, F32> {
            val src = w.hostF32()
            val rows = w.dims[0]
            val hd = gqa.headDim
            val out = FloatArray(rows * 4 * hd)
            for (r in 0 until rows) for (h in 0 until 4) for (c in 0 until hd) {
                out[r * 4 * hd + h * hd + c] = src[r * 2 * hd + (h / 2) * hd + c]
            }
            return DTensor<Shape, F32>(HostF32Storage(out), intArrayOf(rows, 4 * hd), F32)
        }
        val mha = MultiHeadAttention(
            gqa.q, Dense(repeated(gqa.k.w), null), Dense(repeated(gqa.v.w), null), gqa.o,
            numHeads = 4, numKvHeads = 4, rope = RotaryEmbedding(2),
        )
        val x = f32(intArrayOf(2, 5, 8), 4)
        val a = forward(gqa, x)
        val b = forward(mha, x)
        for (i in a.indices) assertEquals(a[i], b[i], 1e-6f, "element $i")
    }

    @Test
    fun denseAppliesToTheLastAxisOfAnyRank() {
        val dense = Dense(8, 3, RandomKey.fromSeed(5))
        val x = f32(intArrayOf(2, 5, 8), 6)
        val y3 = forward(dense, x)
        val y2 = forward(dense, DTensor<Shape, F32>(HostF32Storage(x.hostF32()), intArrayOf(10, 8), F32))
        assertContentEquals(y2, y3)
    }

    @Test
    fun normsProduceUnitStatisticsWithDefaultWeights() {
        val x = f32(intArrayOf(3, 16), 7)
        val ln = forward(LayerNorm(16), x)
        val rms = forward(RMSNorm(16, eps = 0f), x)
        for (r in 0 until 3) {
            val row = ln.copyOfRange(r * 16, r * 16 + 16)
            val mean = row.average()
            val variance = row.sumOf { (it - mean) * (it - mean) } / 16
            assertEquals(0.0, mean, 1e-6)
            assertEquals(1.0, variance, 1e-3)
            val rrow = rms.copyOfRange(r * 16, r * 16 + 16)
            assertEquals(1.0, sqrt(rrow.sumOf { it.toDouble() * it } / 16), 1e-5)
        }
    }

    @Test
    fun adamWShrinksBeforeTheAdamUpdateAndHonoursTheDecayFilter() {
        val p = listOf(
            NamedParameter("w", DTensor<Shape, F32>(HostF32Storage(floatArrayOf(1f)), intArrayOf(1), F32)),
            NamedParameter("b", DTensor<Shape, F32>(HostF32Storage(floatArrayOf(1f)), intArrayOf(1), F32)),
        )
        val g = floatArrayOf(0.5f)
        val grads = mapOf(
            "w" to DTensor<Shape, F32>(HostF32Storage(g), intArrayOf(1), F32),
            "b" to DTensor<Shape, F32>(HostF32Storage(g), intArrayOf(1), F32),
        )
        val opt = AdamW(learningRate = 0.1f, weightDecay = 0.1f, eps = 0f, decay = { it != "b" })
        val out = opt.step(p, grads, opt.initialState())
        // First bias-corrected Adam step moves by exactly lr: m̂/√v̂ = g/|g| = 1.
        assertEquals(1f * (1f - 0.1f * 0.1f) - 0.1f, out.params.getValue("w").hostF32()[0], 1e-6f)
        assertEquals(1f - 0.1f, out.params.getValue("b").hostF32()[0], 1e-6f)

        val saved = opt.saveState(out.state)
        assertEquals("AdamW", saved.kind)
        assertEquals(1, opt.loadState(saved).stepCount)
        val noDecay = AdamW(learningRate = 0.1f, weightDecay = 0f, eps = 0f).step(p, grads, opt.initialState())
        val adam = Adam(learningRate = 0.1f, eps = 0f).step(p, grads, Adam().initialState())
        assertContentEquals(adam.params.getValue("w").hostF32(), noDecay.params.getValue("w").hostF32())
    }

    @Test
    fun crossEntropyMatchesTheHandValueAndSkipsIgnoredRows() {
        val logits = floatArrayOf(1f, 2f, 0.5f, -1f, 0f, 3f, 0.2f, 0.2f, 0.2f)
        val x = DTensor<Shape, F32>(HostF32Storage(logits), intArrayOf(3, 3), F32)
        val targets = oneHot(intArrayOf(1, -100, 0), 3, ignoreIndex = -100)
        assertContentEquals(floatArrayOf(0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f), targets.hostF32())
        val fn = captureN(listOf(x, targets)) { l -> crossEntropy(l[0], l[1]) }
        val loss = DxirInterpreter.evalFunction(fn, listOf(logits, targets.hostF32())).single()[0]
        fun nll(row: Int, target: Int): Double {
            val r = logits.copyOfRange(row * 3, row * 3 + 3).map { it.toDouble() }
            return -(r[target] - ln(r.sumOf { exp(it) }))
        }
        assertEquals(((nll(0, 1) + nll(2, 0)) / 2).toFloat(), loss, 1e-6f)
    }

    @Test
    fun oneCapturedStepServesEveryTargetBatch() {
        val model = CausalLM.llama(config, RandomKey.fromSeed(8))
        val input = ids(intArrayOf(1, 2, 3, 4, 5, 6), 2, 3)
        val t1 = oneHot(intArrayOf(2, 3, 4, 5, 6, 7), config.vocabSize, intArrayOf(2, 3))
        val t2 = oneHot(intArrayOf(9, 9, 1, 0, 0, 10), config.vocabSize, intArrayOf(2, 3))
        val step = capture(model, listOf(input), targets = listOf(t1)) { logits, t -> crossEntropy(logits, t[0]) }
        assertEquals(2, step.inputCount)
        for (t in listOf(t1, t2)) {
            val baked = capture(model, listOf(input)) { logits ->
                crossEntropy(logits, logits.constantTargets(t))
            }.run(model, listOf(input))
            val rebound = step.run(model, listOf(input, t))
            assertEquals(baked.loss, rebound.loss)
            for (k in baked.gradients.keys) {
                assertContentEquals(baked.gradients.getValue(k).hostF32(), rebound.gradients.getValue(k).hostF32(), k)
            }
        }
        assertFailsWith<IllegalArgumentException> { step.run(model, listOf(input)) }
    }

    private fun Tracer<Shape>.constantTargets(t: DTensor<*, F32>): Tracer<Shape> =
        constant(t.hostF32(), t.dims)

    @Test
    fun mixedPrecisionCaptureStaysCloseToF32() {
        val model = CausalLM.llama(config, RandomKey.fromSeed(9))
        val input = ids(intArrayOf(1, 2, 3, 4, 5, 6), 2, 3)
        val t = oneHot(intArrayOf(2, 3, 4, 5, 6, 7), config.vocabSize, intArrayOf(2, 3))
        val full = capture(model, listOf(input), targets = listOf(t)) { l, ts -> crossEntropy(l, ts[0]) }
            .run(model, listOf(input, t))
        val mixed = capture(model, listOf(input), targets = listOf(t), precision = Precision.MIXED_BF16) { l, ts ->
            crossEntropy(l, ts[0])
        }.run(model, listOf(input, t))
        assertTrue(abs(full.loss - mixed.loss) < 2e-2f * full.loss, "f32 ${full.loss} vs bf16 ${mixed.loss}")
        for (k in full.gradients.keys) {
            assertTrue(mixed.gradients.getValue(k).hostF32().all { it.isFinite() }, k)
        }
    }

    @Test
    fun tiedEmbeddingGradientsMatchFiniteDifferences() {
        val model = CausalLM.llama(config.copy(tiedEmbeddings = true, numLayers = 1), RandomKey.fromSeed(10))
        val input = ids(intArrayOf(1, 2, 3, 4, 5, 6), 2, 3)
        val t = oneHot(intArrayOf(2, 3, 4, 5, 6, 7), config.vocabSize, intArrayOf(2, 3))
        val step = capture(model, listOf(input), targets = listOf(t)) { l, ts -> crossEntropy(l, ts[0]) }
        val analytic = step.run(model, listOf(input, t)).gradients
        val h = 1e-2f
        for (key in listOf("embed.table", "blocks.0.attn.q.w", "blocks.0.mlp.up.w", "norm.weight")) {
            val base = model.parameters.first { it.key == key }.tensor
            val values = base.hostF32()
            for (i in listOf(0, values.size / 3, values.size - 1, 13 % values.size)) {
                fun lossAt(delta: Float): Float {
                    val moved = values.copyOf().also { it[i] += delta }
                    val m = model.withParameters(mapOf(key to DTensor<Shape, F32>(HostF32Storage(moved), base.dims.copyOf(), F32)))
                    return step.run(m, listOf(input, t)).loss
                }
                val fd = (lossAt(h) - lossAt(-h)) / (2 * h)
                val g = analytic.getValue(key).hostF32()[i]
                assertTrue(abs(fd - g) <= 5e-3f * max(1f, abs(fd)), "$key[$i]: analytic $g vs finite difference $fd")
            }
        }
    }

    @Test
    fun aCausalLmRoundTripsThroughACheckpoint() {
        val model = CausalLM.llama(config, RandomKey.fromSeed(11))
        val restored = ModelCheckpoint.decode(ModelCheckpoint.encode(model)).restore(CausalLM.llama(config, RandomKey.fromSeed(12)))
        for ((a, b) in model.parameters.zip(restored.parameters)) {
            assertEquals(a.key, b.key)
            assertContentEquals(a.tensor.hostF32(), b.tensor.hostF32(), a.key)
        }
    }
}
