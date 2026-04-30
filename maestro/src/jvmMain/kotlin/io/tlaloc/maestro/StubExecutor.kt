package io.tlaloc.maestro

/**
 * Layer 2 §0.4.243+ — in-process stub executor for [Workflow].
 *
 * # ⚠ Deprecated as of §0.4.249 (Layer 2.5.5)
 *
 * Layer 2.5 vendored Netflix/maestro and ships a real `TlalocStepRuntime`
 * (Java) that drives Tlaloc-typed steps through Maestro's actual lifecycle.
 * The in-process stub remains useful for unit tests that need to exercise
 * a [Workflow] without booting a Maestro instance, but it's no longer the
 * only option — vendored Maestro's `TlalocRunner` end-to-end test in
 * `third-party/maestro/maestro-tlaloc/src/test/java/.../TlalocRunnerEndToEndTest.java`
 * is the canonical replacement for "real-runtime-equivalent" coverage.
 *
 * Removal target: post-Layer-3 release once all in-flight tests have
 * migrated.
 *
 * # What this is and isn't (original docs)
 *
 * - **Is**: a fixture that proves the workflow's step graph + shim
 *   functions compose correctly end-to-end. Useful in tests and as a
 *   "what would a real executor do" reference.
 * - **Isn't**: capable of consuming the JSON [MaestroDescriptor] directly.
 *   The shim functions are Kotlin closures (not serializable across the
 *   network); a real Maestro executor (Layer 2.5's `TlalocStepRuntime`)
 *   reads the workflow's `params.tlaloc.artifact_uri`, downloads the
 *   StableHLO bytes, and dispatches them. The stub takes the live
 *   [Workflow] object instead.
 *
 * Both paths exercise the same [Workflow]; the descriptor emission +
 * descriptor consumption are tested separately (see [MaestroDescriptor]
 * and `MaestroDescriptorTest`).
 */
@Deprecated(
    message =
        "Layer 2 stub executor; replaced by vendored Maestro's TlalocStepRuntime + " +
            "TlalocRunner in §0.4.246–§0.4.247. Useful for unit tests; new code should " +
            "exercise workflows through vendored Maestro's test harnesses. Removal " +
            "target: post-Layer-3 release.",
    level = DeprecationLevel.WARNING,
)
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
