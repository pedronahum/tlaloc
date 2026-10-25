/**
 * One training step of a small causal language model, three ways, each timed:
 *
 * [1] On the host. The captured gradient program runs in the interpreter and AdamW
 *     runs in Kotlin. No device.
 * [2] On the GPU, AdamW on the host. `session.runOn` uploads the batch and every
 *     parameter, runs the gradient program, and downloads the loss and every gradient;
 *     AdamW updates the parameters in Kotlin. Every step crosses the host twice with
 *     the whole model.
 * [3] On the GPU, AdamW in the program. `withDeviceAdamW` appends the update to the
 *     gradient program and donates the parameters and both moments to it, so they stay
 *     in device memory from step to step. Only the batch, four scalars and the loss
 *     cross the host.
 *
 * The three lanes train from the same start on the same batches. Each lane's losses are
 * checked against [1]'s; [2] and [3] differ from [1] by the GB10's TF32 dots, so the
 * check allows 1 %.
 */
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
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.StepBenchmark
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.stablehlo.toStablehlo
import kotlin.math.abs

val config = CausalLmConfig(
    vocabSize = 64, dModel = 128, numLayers = 2, numHeads = 4, numKvHeads = 2, ffHidden = 256, initStd = 0.05f,
)
const val BATCH = 4
const val SEQ = 16
const val CHECKED_STEPS = 10

/** Batch k of four: each row a sequence of a simple recurrence, the targets its next tokens. */
fun batchAt(k: Int): Pair<DTensor<*, I32>, DTensor<*, F32>> {
    val ids = IntArray(BATCH * SEQ)
    val next = IntArray(BATCH * SEQ)
    for (r in 0 until BATCH) {
        var t = (r * 5 + k % 4) % config.vocabSize
        for (s in 0 until SEQ) {
            ids[r * SEQ + s] = t
            t = (3 * t + 1 + r) % config.vocabSize
            next[r * SEQ + s] = t
        }
    }
    return DTensor<Shape, I32>(HostI32Storage(ids), intArrayOf(BATCH, SEQ), I32) to
        oneHot(next, config.vocabSize, intArrayOf(BATCH, SEQ))
}

interface Lane : AutoCloseable {
    /** One training step on batch k; returns the loss. */
    fun step(k: Int): Float
    override fun close() {}
}

/** [1] The interpreter runs the gradient program; AdamW runs in Kotlin. */
class HostLane(val captured: CapturedStep, var model: CausalLM, val optimizer: AdamW) : Lane {
    private var state = optimizer.initialState()
    override fun step(k: Int): Float {
        val (ids, targets) = batchAt(k)
        val out = captured.run(model, listOf(ids, targets))
        val (m, s) = optimizer.step(model, out.gradients, state)
        model = m
        state = s
        return out.loss
    }
}

/** [2] The GPU runs the gradient program; parameters go up and gradients come down every step. */
class GpuHostOptimizerLane(
    val session: PjrtSession,
    val captured: CapturedStep,
    var model: CausalLM,
    val optimizer: AdamW,
) : Lane {
    private var state = optimizer.initialState()
    override fun step(k: Int): Float {
        val (ids, targets) = batchAt(k)
        val idv = (ids.storage as HostI32Storage).data
        // The gradient program's inputs: the batch, then every parameter.
        val outs = session.runOn(
            captured.gradient,
            listOf(FloatArray(idv.size) { idv[it].toFloat() }, targets.hostF32()) +
                captured.parameterTensors(model).map { it.hostF32() },
        )
        // Its outputs: the loss, the inputs' gradients, then the parameters'.
        val grads = captured.parameterKeys.withIndex().associate { (j, key) ->
            val dims = captured.primal.params[captured.inputCount + j].type.dims.toIntArray()
            key to DTensor<Shape, F32>(HostF32Storage(outs[1 + captured.inputCount + j]), dims, F32)
        }
        val (m, s) = optimizer.step(model, grads, state)
        model = m
        state = s
        return outs[0][0]
    }
}

/** [3] AdamW is part of the program; parameters and moments stay in donated device buffers. */
class GpuDeviceOptimizerLane(val session: PjrtSession, val device: DeviceAdamWStep, start: CausalLM) : Lane {
    // The aliases mark each parameter and moment as donated to its updated value.
    private val text = device.function.toStablehlo(outputAliases = device.outputAliases)
    private fun zerosLike(i: Int): PjrtBuffer {
        val dims = device.step.primal.params[device.step.inputCount + i].type.dims
        return session.bufferFromHostF32(FloatArray(dims.fold(1) { a, b -> a * b }), dims)
    }
    private var params: List<PjrtBuffer> =
        device.step.parameterTensors(start).map { session.bufferFromHostF32(it.hostF32(), it.dims.toList()) }
    private var m = params.indices.map(::zerosLike)
    private var v = params.indices.map(::zerosLike)
    private var t = 0

    override fun step(k: Int): Float {
        t += 1
        val (ids, targets) = batchAt(k)
        val idsB = session.bufferFromHostI32((ids.storage as HostI32Storage).data, ids.dims.toList())
        val tgtB = session.bufferFromHostF32(targets.hostF32(), targets.dims.toList())
        // The learning rate and the bias corrections are inputs, so the program never changes.
        val scalars = device.scalars(t).map { session.bufferFromHostF32(floatArrayOf(it), emptyList()) }
        val outs = session.executeStablehlo(text, listOf(idsB, tgtB) + params + m + v + scalars)
        // The donated buffers were consumed by the call; close them, never pass them again.
        (listOf(idsB, tgtB) + scalars + params + m + v).forEach { it.close() }
        val n = params.size
        val loss = outs[0].toFloatArray(1)[0]
        outs[0].close()
        params = outs.subList(1, 1 + n)
        m = outs.subList(1 + n, 1 + 2 * n)
        v = outs.subList(1 + 2 * n, 1 + 3 * n)
        return loss
    }

    override fun close() = (params + m + v).forEach { it.close() }
}

/** Two seconds of steps first: an idle GPU's clocks are low, and the first timed steps would show it. */
fun warmClocks(lane: Lane) {
    val end = System.nanoTime() + 2_000_000_000L
    var k = 0
    while (System.nanoTime() < end) lane.step(k++)
}

fun main() {
    val start = CausalLM.llama(config, RandomKey.fromSeed(5))
    val (ids0, t0) = batchAt(0)
    val captured = capture(start, listOf(ids0), targets = listOf(t0)) { logits, t -> crossEntropy(logits, t[0]) }
    val optimizer = AdamW(learningRate = 3e-3f, weightDecay = 0.01f)
    val nParams = captured.parameterTensors(start).sumOf { it.hostF32().size }
    println("A ${config.numLayers}-layer causal LM, dModel ${config.dModel}, $nParams parameters; batch $BATCH × $SEQ tokens.")
    println()

    val reference = HostLane(captured, start, optimizer).let { lane -> List(CHECKED_STEPS) { lane.step(it) } }
    println("[1] host:  losses ${"%.4f".format(reference.first())} -> ${"%.4f".format(reference.last())} over $CHECKED_STEPS steps")
    val hostMs = StepBenchmark(minWarmup = 2, maxWarmup = 4, timedSteps = 5).run(compileCount = { 0 }) {
        HostLane(captured, start, optimizer).step(0)
    }.medianMs

    val skip = when {
        !PjrtBinaries.available -> "no PJRT plugin found"
        !PjrtBinaries.cudaAvailable -> "no CUDA device"
        else -> null
    }
    if (skip != null) {
        println("SKIP: lanes [2] and [3] need a GPU ($skip).")
        println("[1] median step: ${"%.2f".format(hostMs)} ms")
        return
    }

    PjrtSession(target = PjrtTarget.Cuda).use { session ->
        fun measure(label: String, make: () -> Lane): Double {
            make().use { lane ->
                val losses = List(CHECKED_STEPS) { lane.step(it) }
                val worst = losses.indices.maxOf { abs(losses[it] - reference[it]) / reference[it] }
                check(worst < 1e-2) { "$label: losses $losses differ from the host's $reference by $worst" }
                println("$label losses ${"%.4f".format(losses.first())} -> ${"%.4f".format(losses.last())}, " +
                    "at most ${"%.1e".format(worst)} from [1]")
            }
            make().use { lane ->
                warmClocks(lane)
                var k = 0
                val r = StepBenchmark(minWarmup = 10, maxWarmup = 40, timedSteps = 50).run(compileCount = { session.cacheSize }) {
                    lane.step(k++)
                }
                return r.medianMs
            }
        }
        val gpuHost = measure("[2] GPU, AdamW on the host:") { GpuHostOptimizerLane(session, captured, start, optimizer) }
        val device = captured.withDeviceAdamW(optimizer)
        val gpuDevice = measure("[3] GPU, AdamW in the program:") { GpuDeviceOptimizerLane(session, device, start) }

        println()
        println("median step                          ms    vs [1]")
        println("[1] host                       %8.2f    %6.1f×".format(hostMs, 1.0))
        println("[2] GPU, AdamW on the host     %8.2f    %6.1f×".format(gpuHost, hostMs / gpuHost))
        println("[3] GPU, AdamW in the program  %8.2f    %6.1f×".format(gpuDevice, hostMs / gpuDevice))
        println()
        println("[3] is %.1f× [2]: the parameters no longer cross the host.".format(gpuHost / gpuDevice))
    }
}
