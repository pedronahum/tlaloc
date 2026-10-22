package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [withDeviceAdamW]'s program, run by the interpreter, takes the same steps as the host
 * [AdamW] loop over [CapturedStep.run]: the same loss and the same parameters, bit for bit,
 * because it is the same arithmetic in the same order.
 */
class DeviceAdamWTest {

    private val config = CausalLmConfig(
        vocabSize = 16, dModel = 16, numLayers = 1, numHeads = 2, numKvHeads = 1, ffHidden = 32, initStd = 0.1f,
    )

    private fun batch(k: Int): Pair<DTensor<*, I32>, DTensor<*, F32>> {
        val ids = IntArray(2 * 4) { (3 * it + k) % config.vocabSize }
        val next = IntArray(ids.size) { (ids[it] + 1) % config.vocabSize }
        return DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(2, 4), I32) to oneHot(next, config.vocabSize, intArrayOf(2, 4))
    }

    @Test
    fun theInterpretedDeviceStepEqualsTheHostLoopBitForBit() {
        val start = CausalLM.llama(config, RandomKey.fromSeed(11))
        val (ids0, t0) = batch(0)
        val step = capture(start, listOf(ids0), targets = listOf(t0)) { logits, t -> crossEntropy(logits, t[0]) }
        // Exempt norms from decay, so both kinds of parameter are covered.
        val optimizer = AdamW(learningRate = 1e-2f, weightDecay = 0.1f, decay = { "norm" !in it })
        val device = step.withDeviceAdamW(optimizer)
        val nP = step.parameterKeys.size
        assertEquals(1 + 3 * nP, device.function.returns.size)
        assertEquals(3 * nP, device.outputAliases.size)

        // Host loop.
        var model = start
        var state = optimizer.initialState()
        val hostLosses = ArrayList<Float>()
        // Device program, interpreted.
        var params = step.parameterTensors(start).map { it.hostF32().copyOf() }
        var m = params.map { FloatArray(it.size) }
        var v = params.map { FloatArray(it.size) }
        val deviceLosses = ArrayList<Float>()

        for (s in 1..3) {
            val (ids, targets) = batch(s)
            val idsF = (ids.storage as HostI32Storage).data.let { d -> FloatArray(d.size) { d[it].toFloat() } }
            val r = step.run(model, listOf(ids, targets))
            hostLosses += r.loss
            val (next, st) = optimizer.step(model, r.gradients, state)
            model = next
            state = st

            val outs = DxirInterpreter.evalFunction(
                device.function,
                listOf(idsF, targets.hostF32()) + params + m + v + device.scalars(s).map { floatArrayOf(it) },
            )
            deviceLosses += outs[0][0]
            params = outs.subList(1, 1 + nP)
            m = outs.subList(1 + nP, 1 + 2 * nP)
            v = outs.subList(1 + 2 * nP, 1 + 3 * nP)
        }
        assertEquals(hostLosses, deviceLosses)
        val hostParams = step.parameterTensors(model).map { it.hostF32() }
        for (i in 0 until nP) assertContentEquals(hostParams[i], params[i], step.parameterKeys[i])
        assertTrue(hostLosses.last() < hostLosses.first(), "$hostLosses")
    }
}
