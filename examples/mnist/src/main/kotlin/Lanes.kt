/**
 * Where the gradient graph actually RUNS.
 *
 * `:nn`'s `capture` hands back a [CapturedStep] holding two `DxirFunction`s:
 * the traced forward, and the gradient function the compiler's reverse
 * transform derived from it. A `DxirFunction` is just a program — it does not
 * know or care what executes it. That is a deliberate module boundary in
 * Tlaloc: **execution is a backend choice the caller makes**, which is why
 * `:nn` has no dependency on `:runtime-pjrt`.
 *
 * So this file is the whole "backend choice", and it is small:
 *
 *  - [HostLane] interprets the graph on the JVM ([DxirInterpreter]).
 *  - [GpuLane] emits the SAME graph to StableHLO, lets XLA compile it, and
 *    executes it on the GPU through PJRT.
 *
 * Nothing else in the example knows which one it got.
 *
 * ON PLUMBING (an honest note, because this example is also documentation):
 * the ~15 lines of [unpack] below mirror `CapturedStep.run`'s own output
 * indexing. `:nn` deliberately does not ship a `CapturedStep.runOn(session)`
 * convenience — adding one would put an `:nn` → `:runtime-pjrt` edge in the
 * module graph and undo the boundary above (the decision, and this exact
 * recipe, are recorded in docs/MODEL_LAYER_PLAN.md's F8 entry). Writing those
 * lines yourself IS the supported way to put a training step on the GPU, so
 * the example writes them in the open rather than hiding them.
 */
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.nn.CapturedStep
import io.tlaloc.nn.Trainable
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget

/** One evaluated training step: the scalar loss and one gradient per parameter key. */
class TrainStep(val loss: Float, val gradients: Map<String, DTensor<*, F32>>)

/**
 * A backend: given a compiled-or-interpreted program and its inputs, produce
 * its outputs. Deliberately the narrowest interface that can carry a whole
 * training loop.
 */
interface Lane : AutoCloseable {
    val description: String

    fun run(fn: DxirFunction, values: List<FloatArray>): List<FloatArray>

    override fun close() {}
}

/** The JVM lane: `:ir`'s reference interpreter walks the graph node by node. */
class HostLane : Lane {
    override val description = "host JVM (DxirInterpreter)"

    override fun run(fn: DxirFunction, values: List<FloatArray>): List<FloatArray> =
        DxirInterpreter.evalFunction(fn, values)
}

/**
 * The GPU lane: `DxirFunction` → StableHLO text → XLA compile → device
 * execution, all inside [PjrtSession.runOn].
 *
 * The session caches compiled executables keyed by the emitted MLIR text, so a
 * training loop that dispatches the same graph 300 times compiles ONCE. This
 * works only because the model's parameters enter the graph as PARAMETERS, not
 * as baked-in constants — every step re-binds new weight values into the same
 * compiled executable. [PjrtSession.cacheSize] is the receipt, and the example
 * prints it.
 */
class GpuLane : Lane {
    private val session = PjrtSession(target = PjrtTarget.Cuda)

    override val description = "GPU via PJRT/XLA (CUDA)"

    val compiledExecutables: Int get() = session.cacheSize

    override fun run(fn: DxirFunction, values: List<FloatArray>): List<FloatArray> =
        session.runOn(fn, values)

    override fun close() = session.close()

    companion object {
        /** Why the GPU lane is unavailable, or null when it is available. */
        fun unavailableReason(): String? = when {
            !PjrtBinaries.available ->
                "no PJRT plugin resolved (set TLALOC_PJRT_PLUGIN_PATH, or install a JAX CUDA plugin)"
            !PjrtBinaries.cudaAvailable ->
                "no CUDA device visible to nvidia-smi"
            else -> null
        }
    }
}

// ---------------------------------------------------------------------------
// Binding and unpacking — the only code that needs to know the calling
// convention of a captured gradient function.
// ---------------------------------------------------------------------------

/**
 * The gradient function's parameter list is `inputs ++ model.parameters`, in
 * that order. Binding a step means laying the live tensor values out the same
 * way — which is also why the loop below can keep ONE compiled executable: the
 * shapes never change, only the numbers.
 */
fun CapturedStep.bind(inputs: List<FloatArray>, model: Trainable<*>): List<FloatArray> =
    inputs + model.parameters.map { it.tensor.hostF32() }

/**
 * ...and the returns are `loss ++ inputGradients ++ parameterGradients`, again
 * in that order (`includeForward = true` on the reverse transform is what puts
 * the loss in slot 0, so one dispatch yields both the value and its gradient).
 */
fun CapturedStep.unpack(outs: List<FloatArray>): TrainStep {
    val grads = LinkedHashMap<String, DTensor<*, F32>>(parameterKeys.size)
    for ((j, key) in parameterKeys.withIndex()) {
        val dims = primal.params[inputCount + j].type.dims.toIntArray()
        grads[key] = DTensor<Shape, F32>(HostF32Storage(outs[1 + inputCount + j]), dims, F32)
    }
    return TrainStep(outs[0][0], grads)
}
