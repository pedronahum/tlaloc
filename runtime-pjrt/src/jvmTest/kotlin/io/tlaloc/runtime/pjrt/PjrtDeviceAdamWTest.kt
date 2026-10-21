package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CapturedStep
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.DeviceAdamWStep
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.step
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.stablehlo.toStablehlo
import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The training step with AdamW inside the compiled program ([withDeviceAdamW]): parameters and
 * moments stay on the GPU, donated to the step and updated in place, and only the batch, four
 * scalars and the loss cross the host boundary. Against the loop that reads the gradients back
 * and runs [AdamW] on the host, over the same batches: the losses and the trained parameters
 * agree, and each loop is timed with [StepBenchmark].
 */
class PjrtDeviceAdamWTest {

    private val config = CausalLmConfig(
        vocabSize = 64, dModel = 128, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 256, initStd = 0.05f,
    )
    private val batch = 4
    private val seq = 16

    private fun batchAt(k: Int): Pair<DTensor<*, I32>, DTensor<*, F32>> {
        val ids = IntArray(batch * seq)
        val next = IntArray(batch * seq)
        for (r in 0 until batch) {
            var t = (r * 5 + k) % config.vocabSize
            for (s in 0 until seq) {
                ids[r * seq + s] = t
                t = (3 * t + 1 + r) % config.vocabSize
                next[r * seq + s] = t
            }
        }
        return DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(batch, seq), I32) to
            oneHot(next, config.vocabSize, intArrayOf(batch, seq))
    }

    /** The host-optimizer loop: run the gradient program, read every gradient back, AdamW on the host. */
    private class HostLoop(val session: PjrtSession, val step: CapturedStep, var model: CausalLM, val optimizer: AdamW) {
        var state = optimizer.initialState()
        fun step(ids: DTensor<*, I32>, targets: DTensor<*, F32>): Float {
            val idv = (ids.storage as HostI32Storage).data
            val outs = session.runOn(
                step.gradient,
                listOf(FloatArray(idv.size) { idv[it].toFloat() }, targets.hostF32()) + model.parameters.map { it.tensor.hostF32() },
            )
            val grads = step.parameterKeys.withIndex().associate { (j, key) ->
                val dims = step.primal.params[step.inputCount + j].type.dims.toIntArray()
                key to DTensor<Shape, F32>(HostF32Storage(outs[1 + step.inputCount + j]), dims, F32)
            }
            val (m, st) = optimizer.step(model, grads, state)
            model = m
            state = st
            return outs[0][0]
        }
    }

    /** The device loop: parameters and moments live in donated device buffers. */
    private class DeviceLoop(val session: PjrtSession, val device: DeviceAdamWStep, start: CausalLM) {
        val text = device.function.toStablehlo(outputAliases = device.outputAliases)
        var params: List<PjrtBuffer> = device.step.parameterTensors(start).map { t ->
            session.bufferFromHostF32(t.hostF32(), t.dims.toList())
        }
        var m: List<PjrtBuffer> = params.indices.map { i ->
            val dims = device.step.primal.params[device.step.inputCount + i].type.dims
            session.bufferFromHostF32(FloatArray(dims.fold(1) { a, b -> a * b }), dims)
        }
        var v: List<PjrtBuffer> = params.indices.map { i ->
            val dims = device.step.primal.params[device.step.inputCount + i].type.dims
            session.bufferFromHostF32(FloatArray(dims.fold(1) { a, b -> a * b }), dims)
        }
        var t = 0

        fun step(ids: DTensor<*, I32>, targets: DTensor<*, F32>): Float {
            t += 1
            val idsB = session.bufferFromHostI32((ids.storage as HostI32Storage).data, ids.dims.toList())
            val tgtB = session.bufferFromHostF32(targets.hostF32(), targets.dims.toList())
            val scalars = device.scalars(t).map { session.bufferFromHostF32(floatArrayOf(it), emptyList()) }
            val outs = session.executeStablehlo(text, listOf(idsB, tgtB) + params + m + v + scalars)
            (listOf(idsB, tgtB) + scalars + params + m + v).forEach { it.close() }
            val nP = params.size
            val loss = outs[0].toFloatArray(1)[0]
            outs[0].close()
            params = outs.subList(1, 1 + nP)
            m = outs.subList(1 + nP, 1 + 2 * nP)
            v = outs.subList(1 + 2 * nP, 1 + 3 * nP)
            return loss
        }

        fun parameters(): List<FloatArray> = params.mapIndexed { i, b ->
            b.toFloatArray(device.step.primal.params[device.step.inputCount + i].type.elementCount.toInt())
        }

        fun close() = (params + m + v).forEach { it.close() }
    }

    @Test
    fun theDeviceStepTrainsLikeTheHostLoopWithoutMovingParameters() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")

        val start = CausalLM.llama(config, RandomKey.fromSeed(5))
        val (ids0, t0) = batchAt(0)
        val captured = capture(start, listOf(ids0), targets = listOf(t0)) { logits, t -> crossEntropy(logits, t[0]) }
        val optimizer = AdamW(learningRate = 3e-3f, weightDecay = 0.01f)
        val device = captured.withDeviceAdamW(optimizer)
        val nParams = captured.parameterTensors(start).sumOf { it.hostF32().size }

        PjrtSession(target = PjrtTarget.Cuda, diskCache = null).use { session ->
            val host = HostLoop(session, captured, start, optimizer)
            val dev = DeviceLoop(session, device, start)
            try {
                val hostLosses = (0 until 20).map { host.step(batchAt(it % 4).first, batchAt(it % 4).second) }
                val devLosses = (0 until 20).map { dev.step(batchAt(it % 4).first, batchAt(it % 4).second) }
                println("[device-adamw] $nParams parameters; losses host ${hostLosses.first()} -> ${hostLosses.last()}, " +
                    "device ${devLosses.first()} -> ${devLosses.last()}")
                for (i in hostLosses.indices) {
                    assertTrue(
                        abs(hostLosses[i] - devLosses[i]) < 5e-3f * hostLosses[i],
                        "step $i: host loss ${hostLosses[i]}, device ${devLosses[i]}",
                    )
                }
                assertTrue(devLosses.last() < 0.7f * devLosses.first(), "the device loop did not train: $devLosses")
                // The two programs fuse the same gradient math differently and the GB10 runs f32
                // dots at TF32, so their gradients differ in low bits. Adam's step is close to
                // lr·sign(g): where a gradient is near zero its sign can differ, so each step can
                // move an element by up to 2·lr apart. The exact check is DeviceAdamWTest (the
                // interpreted program equals the host loop bit for bit).
                val hp = captured.parameterTensors(host.model).map { it.hostF32() }
                val dp = dev.parameters()
                var worst = 0f
                for (k in hp.indices) for (j in hp[k].indices) worst = maxOf(worst, abs(hp[k][j] - dp[k][j]))
                val bound = 2f * optimizer.learningRate * 20
                println("[device-adamw] worst |host - device| over the parameters after 20 steps: $worst (bound 2·lr·steps = $bound)")
                assertTrue(worst <= bound, "parameters differ by $worst > $bound")

                val bench = StepBenchmark(minWarmup = 5, maxWarmup = 10, timedSteps = 20)
                var k = 0
                val h = bench.run(compileCount = { session.cacheSize }, peakBytes = { session.memoryStats().second }) {
                    host.step(batchAt(k % 4).first, batchAt(k++ % 4).second)
                }
                val d = bench.run(compileCount = { session.cacheSize }, peakBytes = { session.memoryStats().second }) {
                    dev.step(batchAt(k % 4).first, batchAt(k++ % 4).second)
                }
                println("[device-adamw] " + h.toJson("host AdamW"))
                println("[device-adamw] " + d.toJson("device AdamW"))
            } finally {
                dev.close()
            }
        }
    }
}
