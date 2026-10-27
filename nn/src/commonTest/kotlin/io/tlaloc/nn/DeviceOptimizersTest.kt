package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [withDeviceSGD], [withDeviceMuon] and [withGradientAccumulation]: the device programs, run by
 * the interpreter, take the host optimizers' steps; [Muon] takes PyTorch's.
 */
class DeviceOptimizersTest {

    private val config = CausalLmConfig(
        vocabSize = 16, dModel = 16, numLayers = 1, numHeads = 2, numKvHeads = 1, ffHidden = 32, initStd = 0.1f,
    )

    private fun batch(k: Int, rows: Int = 2): Pair<DTensor<*, I32>, DTensor<*, F32>> {
        val ids = IntArray(rows * 4) { (3 * it + k) % config.vocabSize }
        val next = IntArray(ids.size) { (ids[it] + 1) % config.vocabSize }
        return DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(rows, 4), I32) to oneHot(next, config.vocabSize, intArrayOf(rows, 4))
    }

    private fun floats(t: DTensor<*, I32>): FloatArray =
        (t.storage as HostI32Storage).data.let { d -> FloatArray(d.size) { d[it].toFloat() } }

    private val start = CausalLM.llama(config, RandomKey.fromSeed(11))
    private val step = batch(0).let { (ids, t) -> capture(start, listOf(ids), targets = listOf(t)) { logits, tt -> crossEntropy(logits, tt[0]) } }

    /** The host loop: [CapturedStep.run], then [optimizer] on the host. Losses and final parameters. */
    private fun <S> hostLoop(optimizer: Optimizer<S>, steps: Int): Pair<List<Float>, List<FloatArray>> {
        var model = start
        var state = optimizer.initialState()
        val losses = (1..steps).map { s ->
            val (ids, t) = batch(s)
            val r = step.run(model, listOf(ids, t))
            val (next, st) = optimizer.step(model, r.gradients, state)
            model = next
            state = st
            r.loss
        }
        return losses to step.parameterTensors(model).map { it.hostF32() }
    }

    /** [device]'s program in the interpreter, its state threaded from step to step. */
    private fun deviceLoop(device: DeviceOptimizerStep, steps: Int): Pair<List<Float>, List<FloatArray>> {
        val nP = step.parameterKeys.size
        var params = step.parameterTensors(start).map { it.hostF32().copyOf() }
        var state = device.initialState()
        val losses = (1..steps).map { s ->
            val (ids, t) = batch(s)
            val outs = DxirInterpreter.evalFunction(
                device.function,
                listOf(floats(ids), t.hostF32()) + params + state + device.scalars(s).map { floatArrayOf(it) },
            )
            params = outs.subList(1, 1 + nP)
            state = outs.subList(1 + nP, outs.size)
            outs[0][0]
        }
        return losses to params
    }

    @Test
    fun theInterpretedDeviceSgdEqualsTheHostLoopBitForBit() {
        for (sgd in listOf(SGD(initialLearningRate = 0.05f, lrDecay = 0.1f, momentum = 0.9f), SGD(initialLearningRate = 0.05f))) {
            val device = step.withDeviceSGD(sgd)
            assertEquals(device.stateSlots.size, device.function.returns.size - 1 - step.parameterKeys.size)
            val (hostLosses, hostParams) = hostLoop(sgd, 3)
            val (devLosses, devParams) = deviceLoop(device, 3)
            assertEquals(hostLosses, devLosses)
            for (i in hostParams.indices) assertContentEquals(hostParams[i], devParams[i], step.parameterKeys[i])
        }
    }

    @Test
    fun theInterpretedDeviceMuonFollowsTheHostMuon() {
        // The embedding and the head stay on AdamW, as is usual; the attention and MLP matrices take Muon.
        val muon = Muon(learningRate = 0.02f, fallback = AdamW(learningRate = 1e-2f), useMuon = { key, _ -> "embed" !in key && "head" !in key })
        val device = step.withDeviceMuon(muon)
        assertTrue(device.muonParameters.isNotEmpty() && device.muonParameters.size < step.parameterKeys.size)
        val (hostLosses, hostParams) = hostLoop(muon, 3)
        val (devLosses, devParams) = deviceLoop(device, 3)
        for (s in hostLosses.indices) assertEquals(hostLosses[s], devLosses[s], 1e-5f)
        for (i in hostParams.indices) {
            val worst = hostParams[i].indices.maxOf { abs(hostParams[i][it] - devParams[i][it]) }
            assertTrue(worst < 1e-5f, "${step.parameterKeys[i]}: $worst")
        }
        assertTrue(hostLosses.last() < hostLosses.first(), "$hostLosses")
    }

    @Test
    fun muonTakesPyTorchsSteps() {
        // torch.optim.Muon (lr 0.02, weight decay 0.1, momentum 0.95, Nesterov) with its
        // Newton–Schulz iteration run in f32, and torch.optim.AdamW (lr 0.01, weight decay 0.01)
        // for the bias, three steps.
        val w1Want = floatArrayOf(-0.010623151f, 0.16716537f, 0.32078105f, 0.42921251f, 0.47758555f, 0.45702228f, 0.38178152f, 0.25312793f, 0.089311674f, -0.086647503f, -0.28806594f, -0.40944612f, -0.47682819f, -0.47944525f, -0.41529867f)
        val w2Want = floatArrayOf(0.3005459f, 0.29397386f, 0.27454856f, 0.24312052f, 0.20290269f, 0.15102333f, 0.092580467f, 0.030105878f, -0.02813793f, -0.091308281f, -0.15040675f, -0.20289117f, -0.23871627f, -0.27200416f, -0.29330501f, -0.30174887f, -0.2890451f, -0.27127713f, -0.24159032f, -0.20134652f, -0.14630578f, -0.090140171f, -0.03004967f, 0.03129144f)
        val biasWant = floatArrayOf(0.029228862f, 0.11569635f, 0.18410079f, 0.27338535f)

        fun t(dims: IntArray, f: (Int) -> Float): DTensor<*, F32> =
            DTensor<Shape, F32>(HostF32Storage(FloatArray(dims.fold(1) { a, b -> a * b }) { f(it) }), dims, F32)
        var params = listOf(
            NamedParameter("w1", t(intArrayOf(3, 5)) { (sin(it * 0.37) * 0.5).toFloat() }),
            NamedParameter("w2", t(intArrayOf(6, 4)) { (cos(it * 0.21) * 0.3).toFloat() }),
            NamedParameter("bias", t(intArrayOf(4)) { 0.1f * it }),
        )
        val muon = Muon(learningRate = 0.02f, weightDecay = 0.1f, fallback = AdamW(learningRate = 0.01f, weightDecay = 0.01f))
        var state = muon.initialState()
        for (s in 1..3) {
            val grads = mapOf(
                "w1" to t(intArrayOf(3, 5)) { (sin(it * 0.11 + s) * 0.2).toFloat() },
                "w2" to t(intArrayOf(6, 4)) { (cos(it * 0.13 + 2 * s) * 0.1).toFloat() },
                "bias" to t(intArrayOf(4)) { 0.05f * (it - s) },
            )
            val out = muon.step(params, grads, state)
            params = params.map { NamedParameter(it.key, out.params.getValue(it.key)) }
            state = out.state
        }
        // Five Newton–Schulz iterations amplify f32 rounding differences to a few 1e-6.
        for ((p, want) in params.zip(listOf(w1Want, w2Want, biasWant))) {
            val got = p.tensor.hostF32()
            for (i in want.indices) assertEquals(want[i], got[i], 1e-5f, "${p.key}[$i]")
        }
        // The state survives a checkpoint.
        val restored = muon.loadState(muon.saveState(state))
        assertEquals(state.momentum.keys, restored.momentum.keys)
        assertEquals(state.adam.stepCount, restored.adam.stepCount)
    }

    @Test
    fun theInterpretedDeviceAdamWOverLoraAdaptersEqualsTheHostLoopBitForBit() {
        val adapted = Lora.apply(start, LoraConfig(4, 8f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(3))
        val (ids0, t0) = batch(0)
        val lora = capture(adapted, listOf(ids0), listOf(t0), Lora.frozen) { logits, tt -> crossEntropy(logits, tt[0]) }
        val optimizer = AdamW(learningRate = 1e-2f, weightDecay = 0f)
        val device = lora.withDeviceAdamW(optimizer)
        val nP = lora.parameterKeys.size
        val frozen = Lora.frozen.frozen(adapted).map { it.tensor.hostF32() }
        var model = adapted
        var state = optimizer.initialState()
        var trained = Lora.frozen.trainable(adapted).map { it.tensor.hostF32().copyOf() }
        var moments = device.initialState()
        for (s in 1..4) {
            val (ids, t) = batch(s)
            val r = lora.run(model, listOf(ids, t))
            val (next, st) = optimizer.step(model, r.gradients, state, Lora.frozen)
            model = next
            state = st
            val outs = DxirInterpreter.evalFunction(
                device.function,
                listOf(floats(ids), t.hostF32()) + trained + frozen + moments + device.scalars(s).map { floatArrayOf(it) },
            )
            assertEquals(r.loss, outs[0][0], "step $s")
            trained = outs.subList(1, 1 + nP)
            moments = outs.subList(1 + nP, outs.size)
        }
        val host = Lora.frozen.trainable(model).map { it.tensor.hostF32() }
        for (i in 0 until nP) assertContentEquals(host[i], trained[i], lora.parameterKeys[i])
    }

    @Test
    fun accumulatedGradientsAreTheMeanOfTheMicrobatchGradientsBitForBit() {
        val acc = step.withGradientAccumulation(3)
        assertEquals(3 * step.inputCount, acc.inputCount)
        val barriers = acc.gradient.body.count { it is DxirOp && it.op == OpKind.OPTIMIZATION_BARRIER }
        assertEquals(2 * step.inputCount, barriers)

        val batches = (0 until 3).map { batch(it) }
        val results = batches.map { (ids, t) -> step.run(start, listOf(ids, t)) }
        val r = acc.run(start, batches.flatMap { listOf(it.first, it.second) })
        assertEquals((results[0].loss + results[1].loss + results[2].loss) / 3f, r.loss)
        for (key in step.parameterKeys) {
            val g = results.map { it.gradients.getValue(key).hostF32() }
            val want = FloatArray(g[0].size) { (g[0][it] + g[1][it] + g[2][it]) / 3f }
            assertContentEquals(want, r.gradients.getValue(key).hostF32(), key)
        }
        // Without the barriers the program computes the same values.
        val unordered = step.withGradientAccumulation(3, sequential = false).run(start, batches.flatMap { listOf(it.first, it.second) })
        for (key in step.parameterKeys) assertContentEquals(r.gradients.getValue(key).hostF32(), unordered.gradients.getValue(key).hostF32(), key)
        // The primal is the mean loss too.
        val primalLoss = DxirInterpreter.evalFunction(
            acc.primal, batches.flatMap { listOf(floats(it.first), it.second.hostF32()) } + step.parameterTensors(start).map { it.hostF32() },
        )[0][0]
        assertEquals(r.loss, primalLoss)
    }

    @Test
    fun accumulationMatchesTheFullBatchAndComposesWithRematAndTheDeviceOptimizer() {
        // Two microbatches of 2 rows against one batch of 4 rows: crossEntropy is a mean over
        // positions, so the mean of the two means is the full batch's mean.
        val (ids, t) = batch(5, rows = 4)
        val idv = (ids.storage as HostI32Storage).data
        val tv = t.hostF32()
        val half = idv.size / 2
        fun idsOf(a: Int) = DTensor<Shape, I32>(HostI32Storage(idv.copyOfRange(a, a + half)), intArrayOf(2, 4), I32)
        fun tOf(a: Int) = DTensor<Shape, F32>(HostF32Storage(tv.copyOfRange(a * config.vocabSize, (a + half) * config.vocabSize)), intArrayOf(2, 4, config.vocabSize), F32)
        val full = capture(start, listOf(ids), targets = listOf(t)) { logits, tt -> crossEntropy(logits, tt[0]) }.run(start, listOf(ids, t))
        val acc = step.withRemat().withGradientAccumulation(2)
        val r = acc.run(start, listOf(idsOf(0), tOf(0), idsOf(half), tOf(half)))
        assertEquals(full.loss, r.loss, 1e-6f)
        for (key in step.parameterKeys) {
            val a = full.gradients.getValue(key).hostF32()
            val b = r.gradients.getValue(key).hostF32()
            val worst = a.indices.maxOf { abs(a[it] - b[it]) }
            assertTrue(worst < 1e-6f, "$key: $worst")
        }
        assertFailsWith<IllegalArgumentException> { step.withGradientAccumulation(2).withRemat() }
        val device = acc.withDeviceSGD(SGD(0.05f, momentum = 0.9f))
        assertEquals(1 + 2 * step.parameterKeys.size, device.function.returns.size)
    }
}
