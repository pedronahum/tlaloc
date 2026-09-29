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
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * LoRA on `Dense` and on a tiny Qwen3-shaped [CausalLM] (grouped key/value
 * heads, per-head q/k norm, tied embeddings), all on the interpreter:
 * the adapted model's output equals the base model's bit for bit at
 * initialization; the adapter gradients agree with f64 central differences
 * of an f64 implementation of the same function; training changes the
 * adapters and nothing else; and the merged model computes what the
 * adapted one does.
 */
class LoraTest {

    private val config = CausalLmConfig(
        vocabSize = 13, dModel = 16, numLayers = 2, numHeads = 4, numKvHeads = 2, headDim = 4,
        ffHidden = 24, normEps = 1e-6f, tiedEmbeddings = true, qkNorm = true, initStd = 0.3f,
    )

    private fun base(): CausalLM = CausalLM.llama(config, RandomKey.fromSeed(11))

    private fun ids(values: IntArray, batch: Int, seq: Int): DTensor<*, I32> =
        DTensor<Shape, I32>(HostI32Storage(values), intArrayOf(batch, seq), I32)

    private fun f32(dims: IntArray, seed: Long, lo: Float = -1f, hi: Float = 1f): DTensor<*, F32> {
        val n = dims.fold(1) { a, d -> a * d }
        val u = uniformFloats(RandomKey.fromSeed(seed), n)
        return DTensor<Shape, F32>(HostF32Storage(FloatArray(n) { lo + (hi - lo) * u[it] }), dims, F32)
    }

    /** [layer]'s forward on [input], traced and run on the interpreter. */
    private fun <M> forward(layer: M, input: DTensor<*, *>): FloatArray where M : Layer, M : Trainable<M> {
        val params = layer.parameters
        val fn = captureN(listOf(input) + params.map { it.tensor }) { leaves ->
            val byKey = params.withIndex().associate { (i, p) -> p.key to leaves[1 + i] }
            layer.forward(leaves[0], Params { byKey.getValue(it) })
        }
        val first = if (input.dtype == I32) (input.storage as HostI32Storage).data.let { d -> FloatArray(d.size) { d[it].toFloat() } }
        else @Suppress("UNCHECKED_CAST") (input as DTensor<*, F32>).hostF32()
        return DxirInterpreter.evalFunction(fn, listOf(first) + params.map { it.tensor.hostF32() }).single()
    }

    private val tokens = ids(intArrayOf(1, 5, 2, 7, 3, 12, 0, 4, 9, 9, 6, 8), 2, 6)

    @Test
    fun anAdaptedModelComputesTheBaseModelsOutputExactly() {
        val m = base()
        val adapted = Lora.apply(m, LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(3))
        assertEquals(14, Lora.adapterParameters(adapted).size / 2)
        val expected = forward(m, tokens)
        val actual = forward(adapted, tokens)
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertEquals(expected[i].toRawBits(), actual[i].toRawBits(), "logit $i: ${expected[i]} vs ${actual[i]}")
        }
    }

    @Test
    fun targetsSelectLayersByHuggingFaceNameOrKeyPath() {
        val m = base()
        assertEquals(
            listOf("blocks.0.attn.q", "blocks.0.attn.k", "blocks.0.attn.v", "blocks.0.attn.o",
                "blocks.1.attn.q", "blocks.1.attn.k", "blocks.1.attn.v", "blocks.1.attn.o"),
            Lora.matchingLayers(m, LoraConfig.ATTENTION),
        )
        assertEquals(listOf("blocks.0.attn.q", "blocks.1.attn.q"), Lora.matchingLayers(m, listOf("q_proj")))
        assertEquals(listOf("blocks.1.mlp.down"), Lora.matchingLayers(m, listOf("model.layers.1.mlp.down_proj")))
        assertEquals(listOf("blocks.1.mlp.down"), Lora.matchingLayers(m, listOf("blocks.1.mlp.down")))
        assertEquals(listOf("blocks.0.mlp.gate", "blocks.1.mlp.gate"), Lora.matchingLayers(m, listOf("gate")))
        // A projection name is not a suffix of another one.
        assertEquals(emptyList(), Lora.matchingLayers(m, listOf("proj")))

        val adapted = Lora.apply(m, LoraConfig(2, 4f, listOf("q_proj", "v_proj")), RandomKey.fromSeed(1))
        val keys = Lora.adapterParameters(adapted).map { it.key }
        assertEquals(
            listOf("blocks.0.attn.q.lora_A", "blocks.0.attn.q.lora_B", "blocks.0.attn.v.lora_A", "blocks.0.attn.v.lora_B",
                "blocks.1.attn.q.lora_A", "blocks.1.attn.q.lora_B", "blocks.1.attn.v.lora_A", "blocks.1.attn.v.lora_B"),
            keys,
        )
        // Base parameters keep their keys and order; the adapters follow each layer's own.
        assertEquals(m.parameters.map { it.key }, adapted.parameters.map { it.key }.filterNot(Lora::isAdapterKey))
        assertEquals(keys, Lora.frozen.trainable(adapted).map { it.key })
    }

    @Test
    fun badTargetsAreRefusedByName() {
        val m = base()
        val none = assertFailsWith<IllegalArgumentException> {
            Lora.apply(m, LoraConfig(2, 4f, listOf("q_proj", "c_attn")), RandomKey.fromSeed(1))
        }
        assertTrue("'c_attn' matches no Dense layer" in none.message!!, none.message)
        // Tied embeddings: there is no lm_head layer to adapt.
        assertFailsWith<IllegalArgumentException> { Lora.apply(m, LoraConfig(2, 4f, listOf("lm_head")), RandomKey.fromSeed(1)) }
        val once = Lora.apply(m, LoraConfig(2, 4f, listOf("q_proj")), RandomKey.fromSeed(1))
        val twice = assertFailsWith<IllegalArgumentException> {
            Lora.apply(once, LoraConfig(2, 4f, listOf("q_proj")), RandomKey.fromSeed(1))
        }
        assertTrue("already has an adapter" in twice.message!!, twice.message)
    }

    @Test
    fun initializationIsPeftsKaimingUniformAAndZeroB() {
        val m = base()
        val adapted = Lora.apply(m, LoraConfig(4, 8f, listOf("o_proj", "down_proj")), RandomKey.fromSeed(5))
        val o = adapted.blocks[0].attn.o.lora!!
        assertContentEquals(intArrayOf(16, 4), o.a.dims)
        assertContentEquals(intArrayOf(4, 16), o.b.dims)
        assertEquals(2f, o.scale)
        val down = (adapted.blocks[1].mlp as SwiGLU).down.lora!!
        assertContentEquals(intArrayOf(24, 4), down.a.dims)
        for ((adapter, fanIn) in listOf(o to 16, down to 24)) {
            val bound = 1f / sqrt(fanIn.toFloat())
            val a = adapter.a.hostF32()
            assertTrue(a.all { abs(it) <= bound }, "A outside U(±1/√$fanIn)")
            assertTrue(a.count { abs(it) > bound / 2 } > a.size / 4, "A should fill its range")
            assertTrue(adapter.b.hostF32().all { it == 0f }, "B must start at zero")
        }
        // Distinct layers draw distinct A.
        assertFalse(o.a.hostF32().contentEquals(adapted.blocks[1].attn.o.lora!!.a.hostF32()))
        assertEquals(4f, LoraAdapter(o.a, o.b, 8f, useRslora = true).scale)
    }

    // ------------------------------------------------------------------
    // The gradient check. L = Σ C ⊙ (x·W + b + s·(x·A)·B) for one Dense;
    // an f64 implementation of L, central differences at h = 1e-6, against
    // the captured gradient (f32). Both A and B nonzero, so dA is not zero.
    // ------------------------------------------------------------------

    private fun f64Loss(x: DoubleArray, w: DoubleArray, b: DoubleArray, a: DoubleArray, bb: DoubleArray, c: DoubleArray,
                        rows: Int, inp: Int, out: Int, r: Int, s: Double): Double {
        var total = 0.0
        for (i in 0 until rows) {
            val xa = DoubleArray(r) { k -> (0 until inp).sumOf { p -> x[i * inp + p] * a[p * r + k] } }
            for (j in 0 until out) {
                var y = b[j]
                for (p in 0 until inp) y += x[i * inp + p] * w[p * out + j]
                for (k in 0 until r) y += s * xa[k] * bb[k * out + j]
                total += c[i * out + j] * y
            }
        }
        return total
    }

    private fun checkGradients(useRslora: Boolean) {
        val (rows, inp, out, r) = listOf(4, 5, 3, 2)
        val alpha = 6f
        val dense = Dense(f32(intArrayOf(inp, out), 21), f32(intArrayOf(out), 22))
            .withLora(LoraAdapter(f32(intArrayOf(inp, r), 23), f32(intArrayOf(r, out), 24), alpha, useRslora = useRslora))
        val x = f32(intArrayOf(rows, inp), 25)
        val cv = f32(intArrayOf(rows, out), 26).hostF32()
        val step = capture(dense, listOf(x), Lora.frozen) { y -> (y * y.constant<Shape>(cv, intArrayOf(rows, out))).sum<Shape>() }
        assertEquals(listOf("lora_A", "lora_B"), step.parameterKeys)
        assertEquals(listOf("w", "b"), step.frozenKeys)
        val grads = step.run(dense, listOf(x)).gradients

        fun d(t: DTensor<*, F32>) = t.hostF32().let { v -> DoubleArray(v.size) { v[it].toDouble() } }
        val xs = d(x); val w = d(dense.w); val b = d(dense.b!!); val c = DoubleArray(cv.size) { cv[it].toDouble() }
        val a = d(dense.lora!!.a); val bb = d(dense.lora!!.b)
        val s = if (useRslora) alpha / sqrt(r.toDouble()) else alpha.toDouble() / r
        val h = 1e-6
        var worst = 0.0
        for ((key, array) in listOf("lora_A" to a, "lora_B" to bb)) {
            val g = grads.getValue(key).hostF32()
            for (i in array.indices) {
                val keep = array[i]
                array[i] = keep + h
                val plus = f64Loss(xs, w, b, a, bb, c, rows, inp, out, r, s)
                array[i] = keep - h
                val minus = f64Loss(xs, w, b, a, bb, c, rows, inp, out, r, s)
                array[i] = keep
                val fd = (plus - minus) / (2 * h)
                val err = abs(g[i] - fd) / maxOf(1.0, abs(fd))
                worst = maxOf(worst, err)
                assertTrue(err < 1e-5, "$key[$i]: captured ${g[i]} vs f64 central difference $fd")
            }
        }
        println("[lora] rsLoRA=$useRslora: adapter gradients vs f64 central differences, worst relative error $worst")
    }

    @Test
    fun adapterGradientsMatchF64FiniteDifferences() = checkGradients(useRslora = false)

    @Test
    fun rsLoraGradientsMatchF64FiniteDifferences() = checkGradients(useRslora = true)

    // ------------------------------------------------------------------
    // Training on the tiny model: next-token cross entropy, AdamW on the
    // adapters only.
    // ------------------------------------------------------------------

    private fun train(adapted: CausalLM, steps: Int): Pair<CausalLM, List<Float>> {
        val next = intArrayOf(5, 2, 7, 3, 12, 0, 4, 9, 9, 6, 8, 1)
        val targets = oneHot(next, config.vocabSize, intArrayOf(2, 6))
        val step = capture(adapted, listOf(tokens), listOf(targets), Lora.frozen) { logits, t -> crossEntropy(logits, t[0]) }
        val opt = AdamW(learningRate = 0.05f, weightDecay = 0.01f)
        var state = opt.initialState()
        var m = adapted
        val losses = ArrayList<Float>()
        repeat(steps) {
            val r = step.run(m, listOf(tokens, targets))
            losses += r.loss
            val (nm, ns) = opt.step(m, r.gradients, state, Lora.frozen)
            m = nm
            state = ns
        }
        return m to losses
    }

    @Test
    fun trainingChangesTheAdaptersAndLeavesTheBaseWeightsBitIdentical() {
        val m = base()
        val adapted = Lora.apply(m, LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(3))
        val (trained, losses) = train(adapted, 25)
        println("[lora] tiny Qwen3-shaped model, 25 AdamW steps on the adapters: loss ${losses.first()} -> ${losses.last()}")
        assertTrue(losses.last() < losses.first() * 0.5f, "loss must at least halve: $losses")
        val before = adapted.parameters.associate { it.key to it.tensor.hostF32() }
        for (p in trained.parameters) {
            if (Lora.isAdapterKey(p.key)) continue
            assertContentEquals(before.getValue(p.key), p.tensor.hostF32(), "base weight ${p.key} changed")
        }
        assertTrue(Lora.adapterParameters(trained).filter { it.key.endsWith("lora_B") }.all { p -> p.tensor.hostF32().any { it != 0f } })
    }

    @Test
    fun theMergedModelComputesWhatTheAdaptedModelDoes() {
        val m = base()
        val (trained, _) = train(Lora.apply(m, LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(3)), 10)
        val merged = Lora.merge(trained)
        assertFalse(Lora.hasAdapters(merged))
        assertEquals(m.parameters.map { it.key }, merged.parameters.map { it.key })
        val adaptedOut = forward(trained, tokens)
        val mergedOut = forward(merged, tokens)
        val baseOut = forward(m, tokens)
        var worst = 0f
        var moved = 0f
        for (i in adaptedOut.indices) {
            worst = maxOf(worst, abs(adaptedOut[i] - mergedOut[i]))
            moved = maxOf(moved, abs(adaptedOut[i] - baseOut[i]))
        }
        println("[lora] merged vs adapted logits: max |diff| $worst (training moved them by up to $moved)")
        assertTrue(moved > 0.1f, "training should have moved the logits")
        assertTrue(worst < 1e-4f, "merged and adapted logits differ by $worst")
        // The merged weight is W + scale·A·B.
        val q = trained.blocks[0].attn.q
        val lora = q.lora!!
        val expected = q.w.hostF32().copyOf()
        val delta = lora.delta()
        for (i in expected.indices) expected[i] += delta[i]
        assertContentEquals(expected, merged.blocks[0].attn.q.w.hostF32())
        assertNull(merged.blocks[0].attn.q.lora)
    }

    @Test
    fun dropoutMasksTheAdapterInputAndIsOffWithoutAKey() {
        val dense = Dense(f32(intArrayOf(6, 3), 1), null)
            .withLora(LoraAdapter(f32(intArrayOf(6, 2), 2), f32(intArrayOf(2, 3), 3), 4f, dropout = 0.5f, dropoutKey = RandomKey.fromSeed(9)))
        val x = f32(intArrayOf(5, 6), 4)
        val withMask = forward(dense, x)
        val noKey = Dense(dense.w, null, Activation.Identity, dense.lora!!.withDropoutKey(null))
        val plain = forward(noKey, x)
        assertFalse(withMask.contentEquals(plain), "a keyed mask must change the output")
        val merged = forward(Lora.merge(noKey), x)
        for (i in plain.indices) assertTrue(abs(plain[i] - merged[i]) < 1e-5f)
        // Re-keying draws a new mask.
        val rekeyed = forward(Lora.withDropoutKey(Sequential(dense), RandomKey.fromSeed(10)), x)
        assertFalse(withMask.contentEquals(rekeyed))
    }

    @Test
    fun withParametersRoutesAdapterKeys() {
        val adapted = Lora.apply(base(), LoraConfig(2, 2f, listOf("q_proj")), RandomKey.fromSeed(1))
        val newB = f32(intArrayOf(2, 16), 7)
        val updated = adapted.withParameters(mapOf("blocks.1.attn.q.lora_B" to newB))
        assertContentEquals(newB.hostF32(), updated.blocks[1].attn.q.lora!!.b.hostF32())
        assertEquals(2f, updated.blocks[1].attn.q.lora!!.alpha)
        assertFailsWith<IllegalArgumentException> { base().withParameters(mapOf("blocks.1.attn.q.lora_B" to newB)) }
    }

    @Test
    fun mixedPrecisionCaptureTrainsTheAdaptersWithTheBaseFrozen() {
        val adapted = Lora.apply(base(), LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(3))
            .let { m -> m.withParameters(Lora.adapterParameters(m).filter { it.key.endsWith("lora_B") }.associate { p ->
                p.key to f32(p.tensor.dims, p.key.hashCode().toLong(), -0.2f, 0.2f) }) }
        val next = intArrayOf(5, 2, 7, 3, 12, 0, 4, 9, 9, 6, 8, 1)
        val targets = oneHot(next, config.vocabSize, intArrayOf(2, 6))
        fun step(precision: Precision) =
            capture(adapted, listOf(tokens), listOf(targets), Lora.frozen, precision = precision) { l, t -> crossEntropy(l, t[0]) }
        val full = step(Precision.F32).run(adapted, listOf(tokens, targets))
        val mixedStep = step(Precision.MIXED_BF16)
        val mixed = mixedStep.run(adapted, listOf(tokens, targets))
        assertEquals(Lora.frozen.trainable(adapted).map { it.key }, mixedStep.parameterKeys)
        val lossGap = abs(full.loss - mixed.loss)
        var worst = 0.0
        var scale = 0.0
        for ((key, g) in full.gradients) {
            val a = g.hostF32()
            val b = mixed.gradients.getValue(key).hostF32()
            for (i in a.indices) {
                worst = maxOf(worst, abs(a[i] - b[i]).toDouble())
                scale = maxOf(scale, abs(a[i]).toDouble())
            }
        }
        println("[lora] MIXED_BF16 vs F32 with the base frozen: loss ${mixed.loss} vs ${full.loss}; adapter gradients max |diff| $worst (largest $scale)")
        assertTrue(lossGap < 0.05f * full.loss, "bf16 loss ${mixed.loss} vs f32 ${full.loss}")
        assertTrue(worst < 0.1 * scale, "bf16 adapter gradients differ by $worst of $scale")
    }

    @Test
    fun anAdapterOutsideTheRebuiltContainersIsRefusedNotLeftBehind() {
        val gruLike = object : TrainableLayer<Nothing> {
            val inner = Dense(f32(intArrayOf(4, 3), 1), null).withLora(LoraAdapter(f32(intArrayOf(4, 2), 2), f32(intArrayOf(2, 3), 3), 2f))
            override val parameters = inner.parameters.map { NamedParameter("inner.${it.key}", it.tensor) }
            override fun withParameters(updated: Map<String, DTensor<*, F32>>): Nothing = error("unused")
            override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> = inner.forward(x, params)
        }
        val model = Sequential(gruLike)
        val e = assertFailsWith<IllegalArgumentException> { Lora.merge(model) }
        assertTrue("holds 1 adapters but only 0" in e.message!!, e.message)
        assertFailsWith<IllegalArgumentException> { Lora.inferenceMode(model) }
    }

    @Test
    fun inferenceModeTurnsDropoutOff() {
        val m = Lora.apply(base(), LoraConfig(2, 4f, listOf("q_proj"), dropout = 0.3f), RandomKey.fromSeed(1))
        assertTrue(m.blocks.all { it.attn.q.lora!!.dropoutKey != null })
        val eval = Lora.inferenceMode(m)
        assertTrue(eval.blocks.all { it.attn.q.lora!!.dropoutKey == null })
    }

    @Test
    fun dropoutWorksUnderMixedPrecision() {
        val m = Lora.apply(base(), LoraConfig(2, 4f, listOf("q_proj", "down_proj"), dropout = 0.2f), RandomKey.fromSeed(4))
        val targets = oneHot(intArrayOf(5, 2, 7, 3, 12, 0, 4, 9, 9, 6, 8, 1), config.vocabSize, intArrayOf(2, 6))
        val r = capture(m, listOf(tokens), listOf(targets), Lora.frozen, precision = Precision.MIXED_BF16) { l, t ->
            crossEntropy(l, t[0])
        }.run(m, listOf(tokens, targets))
        assertTrue(r.loss.isFinite())
        assertTrue(r.gradients.values.all { g -> g.hostF32().all { it.isFinite() } })
    }
}
