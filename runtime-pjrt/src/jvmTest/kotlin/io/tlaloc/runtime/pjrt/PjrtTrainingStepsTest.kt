package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.DeviceOptimizerStep
import io.tlaloc.nn.Muon
import io.tlaloc.nn.SGD
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.step
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.nn.withDeviceMuon
import io.tlaloc.nn.withDeviceSGD
import io.tlaloc.nn.withGradientAccumulation
import io.tlaloc.testing.Tolerances
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Training steps on the device: gradient accumulation ([withGradientAccumulation]) against the
 * full batch, with its peak memory, and the device SGD and Muon programs against their host loops.
 */
class PjrtTrainingStepsTest {

    private fun ids(rows: Int, seq: Int, vocab: Int, k: Int): Pair<DTensor<*, I32>, DTensor<*, F32>> {
        val ids = IntArray(rows * seq)
        val next = IntArray(rows * seq)
        for (r in 0 until rows) {
            var t = (r * 5 + k) % vocab
            for (s in 0 until seq) {
                ids[r * seq + s] = t
                t = (3 * t + 1 + r) % vocab
                next[r * seq + s] = t
            }
        }
        return DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(rows, seq), I32) to oneHot(next, vocab, intArrayOf(rows, seq))
    }

    private fun floats(t: DTensor<*, I32>): FloatArray =
        (t.storage as HostI32Storage).data.let { d -> FloatArray(d.size) { d[it].toFloat() } }

    private fun peak(fn: DxirFunction, inputs: List<FloatArray>): Pair<List<FloatArray>, Long> =
        TestBackend.session().use { s ->
            val out = s.runOn(fn, inputs)
            out to (s.memoryStats().second ?: error("the plugin reports no peak"))
        }

    @Test
    fun accumulationHoldsOneMicrobatchOfActivations() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val config = CausalLmConfig(vocabSize = 2048, dModel = 512, numLayers = 4, numHeads = 8, numKvHeads = 4, ffHidden = 1536, initStd = 0.02f)
        val model = CausalLM.llama(config, RandomKey.fromSeed(2))
        val seq = 256
        val micro = 4
        val rows = 8
        val (ids, t) = ids(rows, seq, config.vocabSize, 0)
        val loss = { logits: io.tlaloc.autograd.Tracer<Shape>, tt: List<io.tlaloc.autograd.Tracer<Shape>> -> crossEntropy(logits, tt[0]) }
        val full = capture(model, listOf(ids), targets = listOf(t), lossFn = loss)
        val mbRows = rows / micro
        val idv = floats(ids)
        val tv = t.hostF32()
        val n = mbRows * seq
        val (ids0, t0) = ids(mbRows, seq, config.vocabSize, 0)
        val step = capture(model, listOf(ids0), targets = listOf(t0), lossFn = loss)
        val params = step.parameterTensors(model).map { it.hostF32() }
        val mbInputs = (0 until micro).flatMap { k ->
            listOf(idv.copyOfRange(k * n, (k + 1) * n), tv.copyOfRange(k * n * config.vocabSize, (k + 1) * n * config.vocabSize))
        }

        val (fullOut, fullPeak) = peak(full.gradient, listOf(idv, tv) + params)
        val (seqOut, seqPeak) = peak(step.withGradientAccumulation(micro).gradient, mbInputs + params)
        // Without the barriers XLA:GPU merges matrix products of different microbatches that share
        // a weight; at the vendored XLA commit a CUDA session compiles them without v2 GEMM fusions,
        // which compute wrong parameter gradients for them (CompileOptions.withXlaGpuFixes).
        val (parOut, parPeak) = peak(step.withGradientAccumulation(micro, sequential = false).gradient, mbInputs + params)
        val mib = 1024.0 * 1024.0
        println(
            "PjrtTrainingStepsTest: peak ${"%.0f".format(fullPeak / mib)} MiB for $rows × $seq tokens at once, " +
                "${"%.0f".format(seqPeak / mib)} MiB as $micro microbatches in sequence, ${"%.0f".format(parPeak / mib)} MiB without the barriers",
        )
        val nP = step.parameterKeys.size
        for ((name, out) in listOf("sequential" to seqOut, "unordered" to parOut)) {
            Tolerances.assertClose(doubleArrayOf(out[0][0].toDouble()), doubleArrayOf(fullOut[0][0].toDouble()), 1e-4, "$name loss")
            for (j in 0 until nP) {
                val a = fullOut[1 + 1 + 1 + j] // loss, the two input gradients, then the parameters
                val b = out[1 + 2 * micro + j]
                Tolerances.assertClose(DoubleArray(b.size) { b[it].toDouble() }, DoubleArray(a.size) { a[it].toDouble() }, 2e-3, "$name ${step.parameterKeys[j]}")
            }
        }
        assertTrue(seqPeak < fullPeak * 0.6, "sequential accumulation peak $seqPeak should be well below the full batch's $fullPeak")
    }

    /** [device]'s program on the device, its state threaded through the host. Losses and final parameters. */
    private fun deviceLoop(s: PjrtSession, device: DeviceOptimizerStep, start: List<FloatArray>, batches: List<List<FloatArray>>): Pair<List<Float>, List<FloatArray>> {
        val nP = start.size
        var params = start
        var state = device.initialState()
        val losses = batches.mapIndexed { i, b ->
            val outs = s.runOn(device.function, b + params + state + device.scalars(i + 1).map { floatArrayOf(it) })
            params = outs.subList(1, 1 + nP)
            state = outs.subList(1 + nP, outs.size)
            outs[0][0]
        }
        return losses to params
    }

    @Test
    fun deviceAdamWOverLoraAdaptersTrainsLikeTheHostLoop() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val config = CausalLmConfig(vocabSize = 64, dModel = 128, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 256, initStd = 0.05f)
        val start = io.tlaloc.nn.Lora.apply(CausalLM.llama(config, RandomKey.fromSeed(5)), io.tlaloc.nn.LoraConfig(8, 16f, io.tlaloc.nn.LoraConfig.ALL_LINEAR), RandomKey.fromSeed(1))
        val data = (0 until 4).map { ids(4, 16, config.vocabSize, it) }
        val step = capture(start, listOf(data[0].first), listOf(data[0].second), io.tlaloc.nn.Lora.frozen) { logits, t -> crossEntropy(logits, t[0]) }
        val batches = (0 until 12).map { data[it % 4].let { (i, t) -> listOf(floats(i), t.hostF32()) } }
        val optimizer = AdamW(learningRate = 1e-2f, weightDecay = 0f)
        val device = step.withDeviceAdamW(optimizer)
        val frozen = io.tlaloc.nn.Lora.frozen.frozen(start).map { it.tensor.hostF32() }
        TestBackend.session().use { s ->
            var model = start
            var state = optimizer.initialState()
            val hostLosses = batches.map { b ->
                val outs = s.runOn(step.gradient, b + step.parameterTensors(model).map { it.hostF32() })
                val grads = step.parameterKeys.withIndex().associate { (j, key) ->
                    key to DTensor<Shape, F32>(HostF32Storage(outs[1 + step.inputCount + j]), step.primal.params[step.inputCount + j].type.dims.toIntArray(), F32)
                }
                val (m, st) = optimizer.step(model, grads, state, io.tlaloc.nn.Lora.frozen)
                model = m
                state = st
                outs[0][0]
            }
            var trained = io.tlaloc.nn.Lora.frozen.trainable(start).map { it.tensor.hostF32() }
            var moments = device.initialState()
            val devLosses = batches.mapIndexed { i, b ->
                val outs = s.runOn(device.function, b + trained + frozen + moments + device.scalars(i + 1).map { floatArrayOf(it) })
                trained = outs.subList(1, 1 + trained.size)
                moments = outs.subList(1 + trained.size, outs.size)
                outs[0][0]
            }
            println("[device-lora] losses host $hostLosses")
            println("[device-lora] losses device $devLosses")
            for (i in hostLosses.indices) {
                assertTrue(abs(hostLosses[i] - devLosses[i]) <= 1e-2f * hostLosses[i], "step $i: host ${hostLosses[i]}, device ${devLosses[i]}")
            }
        }
    }

    @Test
    fun deviceSgdAndMuonTrainLikeTheirHostLoops() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val config = CausalLmConfig(vocabSize = 64, dModel = 128, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 256, initStd = 0.05f)
        val start = CausalLM.llama(config, RandomKey.fromSeed(5))
        val data = (0 until 4).map { ids(4, 16, config.vocabSize, it) }
        val step = capture(start, listOf(data[0].first), targets = listOf(data[0].second)) { logits, t -> crossEntropy(logits, t[0]) }
        val batches = (0 until 12).map { data[it % 4].let { (i, t) -> listOf(floats(i), t.hostF32()) } }

        TestBackend.session().use { s ->
            for ((name, optimizer, device) in listOf(
                SGD(initialLearningRate = 0.2f, momentum = 0.9f).let { Triple("sgd", it, step.withDeviceSGD(it)) },
                Muon(learningRate = 0.02f, fallback = AdamW(learningRate = 3e-3f), useMuon = { key, _ -> "embed" !in key && "head" !in key }).let {
                    Triple("muon", it, step.withDeviceMuon(it))
                },
            )) {
                var model = start
                @Suppress("UNCHECKED_CAST")
                val opt = optimizer as io.tlaloc.nn.Optimizer<Any>
                var state = opt.initialState()
                val hostLosses = batches.map { b ->
                    val outs = s.runOn(step.gradient, b + step.parameterTensors(model).map { it.hostF32() })
                    val grads = step.parameterKeys.withIndex().associate { (j, key) ->
                        key to DTensor<Shape, F32>(HostF32Storage(outs[1 + step.inputCount + j]), step.primal.params[step.inputCount + j].type.dims.toIntArray(), F32)
                    }
                    val (m, st) = opt.step(model, grads, state)
                    model = m
                    state = st
                    outs[0][0]
                }
                val (devLosses, _) = deviceLoop(s, device, step.parameterTensors(start).map { it.hostF32() }, batches)
                println("[device-$name] losses host ${hostLosses.first()} -> ${hostLosses.last()}, device ${devLosses.first()} -> ${devLosses.last()}")
                for (i in hostLosses.indices) {
                    assertTrue(abs(hostLosses[i] - devLosses[i]) <= 1e-2f * hostLosses[i], "$name step $i: host ${hostLosses[i]}, device ${devLosses[i]}")
                }
                // The batches cycle through four, so compare the last cycle with the first.
                assertTrue(devLosses.takeLast(4).sum() < 0.9f * devLosses.take(4).sum(), "$name did not train: $devLosses")
            }
        }
    }
}
