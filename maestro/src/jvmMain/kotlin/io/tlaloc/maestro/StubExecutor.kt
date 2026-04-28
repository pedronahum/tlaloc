package io.tlaloc.maestro

/**
 * Layer 2 §0.4.243+ — in-process stub executor for [Workflow].
 *
 * The stub mimics the *contract* of Maestro's real Kubernetes-step
 * executor but skips the actual container launch + cluster dispatch.
 * Given an [Workflow] (in-memory), it walks the step DAG in composition
 * order, threads BufferHandles between steps, and returns the final output.
 *
 * # What this is and isn't
 *
 * - **Is**: a fixture that proves the workflow's step graph + shim
 *   functions compose correctly end-to-end. Useful in tests and as a
 *   "what would a real executor do" reference.
 * - **Isn't**: capable of consuming the JSON [MaestroDescriptor] directly.
 *   The shim functions are Kotlin closures (not serializable across the
 *   network); a real Maestro executor reads the descriptor's
 *   `tlaloc_artifact_uri`, downloads the StableHLO bytes, and dispatches
 *   them via `iree-compile` (Layer 3+). The stub takes the live
 *   [Workflow] object instead.
 *
 * Both paths exercise the same [Workflow]; the descriptor emission +
 * descriptor consumption are tested separately (see [MaestroDescriptor]
 * and `MaestroDescriptorTest`).
 */
class StubExecutor {

    /**
     * Run [workflow] against [seed] (the input to the first step). Returns
     * the final step's output. v1 supports linear chains only.
     */
    @Suppress("UNCHECKED_CAST")
    fun <In, Out> run(workflow: Workflow, seed: In): Out {
        require(workflow.steps.isNotEmpty()) { "cannot run a workflow with no steps" }
        var current: Any? = seed
        for (step in workflow.steps) {
            val shim = step.shim as (Any?) -> Any?
            current = shim(current)
        }
        return current as Out
    }
}
