package io.tlaloc.maestro

import io.tlaloc.core.BufferHandle
import io.tlaloc.core.HandleRef
import io.tlaloc.core.Mesh
import io.tlaloc.core.ProgramScope

/**
 * Layer 2 §0.4.243+ — workflow composition.
 *
 * A [Workflow] is a typed DAG of [MaestroStep]s. The [workflow] builder
 * records the per-step composition order, validates input/output type
 * matches at compose time (Kotlin's checker does most of this directly),
 * and inserts synthetic reshard edges when adjacent steps' meshes differ
 * or when named-axis structure transposes.
 *
 * # Type-checking is Kotlin-native
 *
 * Step composition is by direct function call inside the builder:
 *
 * ```kotlin
 * Tlaloc.workflow("training") {
 *     val hidden = step(encodeStep, tokens)        // returns BufferHandle<…, M>
 *     val loss = step(decodeStep, hidden)          // signature must match
 * }
 * ```
 *
 * If `decodeStep`'s input type doesn't match `hidden`'s type, the call
 * site fails to compile via Kotlin's normal type checking — no plugin
 * involvement. Mesh mismatches are caught at compose time (runtime, but
 * before workflow execution) by inspecting the [Workflow.edges] reshard
 * metadata.
 *
 * # Runtime semantics
 *
 * v1 executes the steps in order via the recorded shim functions. The
 * stub executor (Layer 2.3) is a thin wrapper around [Workflow.run]; the
 * real Maestro executor consumes the [WorkflowDescriptor] (Layer 2.3).
 */
class Workflow(
    val name: String,
    /** Steps in the order they were composed. v1 is a linear chain. */
    val steps: List<MaestroStep<*, *>>,
    /** Composition edges, one per step transition. */
    val edges: List<WorkflowEdge>,
) {
    override fun toString(): String =
        "Workflow(name=$name, steps=${steps.size}, edges=${edges.size})"
}

/**
 * One transition in the workflow's step DAG. Records the source / target
 * step names plus reshard metadata (whether a mesh-reshard or transpose
 * is needed when crossing this edge).
 */
data class WorkflowEdge(
    val fromStep: String,
    val toStep: String,
    val reshardKind: ReshardKind,
    val fromMesh: String,
    val toMesh: String,
) {
    val isPassThrough: Boolean get() = reshardKind == ReshardKind.None
}

/**
 * Reshard kind for a [WorkflowEdge]. v1 distinguishes mesh-only reshards
 * (different mesh placement, same shape) from transpose reshards
 * (different named-axis structure). Layer 4 makes both produce real
 * StableHLO/SDY ops.
 */
enum class ReshardKind {
    /** No reshard needed; producer and consumer agree on placement + shape. */
    None,

    /**
     * Cross-mesh transition: producer outputs on Mesh A, consumer expects
     * Mesh B. v1 records this as metadata; Layer 4's emitter produces
     * `sdy.reshard` (or equivalent collective).
     */
    Mesh,

    /**
     * Named-axis transpose: shape stays the same but axis order/names
     * differ. v1 metadata-only; Layer 4 emits `stablehlo.transpose`.
     */
    Transpose,
}

/**
 * Builder for [Workflow]. Receives [ProgramScope] and is invoked from
 * inside a `workflow { … }` block. The DSL surface is just `step(...)`:
 * each call records a step, computes the edge from the previous step,
 * and returns the typed output handle.
 */
@io.tlaloc.core.WorldScope
class WorkflowBuilder internal constructor(val name: String) : ProgramScope {

    internal val steps = mutableListOf<MaestroStep<*, *>>()
    internal val edges = mutableListOf<WorkflowEdge>()
    internal var lastStepName: String? = null
    internal var lastStepMesh: String? = null
    internal var lastOutputDescriptor: TypeDescriptor? = null

    /**
     * Compose [step] into the workflow with [input]. Returns the typed
     * output handle. If the previous step's mesh / axis-name structure
     * disagrees with [step]'s expected input, a reshard edge is recorded.
     *
     * Type-level: the call site requires `In` to match the previous
     * step's `Out`. Kotlin's checker enforces this; a mismatch is a
     * compile error at the call site (not a runtime failure).
     */
    fun <In, Out> step(step: MaestroStep<In, Out>, input: In): Out {
        steps += step
        val previousName = lastStepName
        if (previousName != null) {
            val previousMesh = lastStepMesh ?: ""
            val currentMesh = step.manifest.meshRequirement
            val previousAxes = lastOutputDescriptor?.axisNames ?: emptyList()
            val currentAxes = step.manifest.inputs.singleOrNull()?.axisNames ?: emptyList()
            val reshardKind = when {
                previousMesh != currentMesh -> ReshardKind.Mesh
                previousAxes != currentAxes -> ReshardKind.Transpose
                else -> ReshardKind.None
            }
            edges += WorkflowEdge(
                fromStep = previousName,
                toStep = step.name,
                reshardKind = reshardKind,
                fromMesh = previousMesh,
                toMesh = currentMesh,
            )
        }
        lastStepName = step.name
        lastStepMesh = step.manifest.meshRequirement
        lastOutputDescriptor = step.manifest.outputs.singleOrNull()
        return step.shim(input)
    }

    /**
     * Bring an externally-provided tensor into the workflow as a typed
     * [BufferHandle]. v1 carries the tensor in [HandleRef.payload]; Layer 3
     * replaces with real device-buffer allocation.
     */
    fun <T : io.tlaloc.core.DTensor<*, *>, M : Mesh> seed(
        value: T,
        @Suppress("UNUSED_PARAMETER") mesh: M,
    ): BufferHandle<T, M> = BufferHandle(HandleRef(nativeId = nextNativeId(), payload = value))
}

/**
 * Open a [Workflow] composition block. Receiver must be a [ProgramScope]
 * (the [io.tlaloc.core.Tlaloc] singleton qualifies). Returns the assembled
 * [Workflow] after the block runs to completion.
 */
fun ProgramScope.workflow(name: String, body: WorkflowBuilder.() -> Unit): Workflow {
    val builder = WorkflowBuilder(name)
    builder.body()
    return Workflow(name, builder.steps.toList(), builder.edges.toList())
}
