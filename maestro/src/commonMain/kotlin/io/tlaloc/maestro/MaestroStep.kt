package io.tlaloc.maestro

import io.tlaloc.core.Mesh

/**
 * Layer 2 §0.4.243+ — typed artifact for one `program { }` block.
 *
 * A [MaestroStep] is what the user names, ships, and Maestro's executor
 * loads. It bundles three things:
 *
 * 1. **Manifest** — fully-self-describing JSON-serializable metadata
 *    (input/output types, mesh requirement, sharding spec, content-
 *    addressed body hash). The manifest is what Maestro's scheduler
 *    inspects when planning a workflow.
 *
 * 2. **Body** — serialized StableHLO+SDY bytes. The text MLIR (UTF-8)
 *    that the runtime image's executor compiles via `iree-compile` (or
 *    PJRT, or another StableHLO consumer). v1 stores this as a `ByteArray`
 *    in memory; on-disk persistence to a content-addressed registry is
 *    Layer 3+.
 *
 * 3. **Shim** — a Kotlin function `(In) -> Out` that the in-process stub
 *    executor invokes. v1's shim re-traces the original Kotlin lambda
 *    against the real input — works for any Tracer-supported op, no
 *    runtime image required. Layer 3+ replaces this with a native dispatch
 *    of the compiled body.
 *
 * # Type-level [In] and [Out]
 *
 * v1 covers single-input / single-output programs (most common case in
 * the Layer 2 design). [In] and [Out] are typically [BufferHandle]s; users
 * of multi-input/output programs use Pair / Triple / data classes. v2 may
 * introduce a generated tuple type for higher-arity steps.
 *
 * @property name Step name (matches `program(name = ...)`).
 * @property manifest Self-describing metadata.
 * @property body StableHLO+SDY bytes.
 * @property mesh Phantom-typed mesh placement of the produced output.
 * @property shim Kotlin function run in process when the step is composed.
 */
class MaestroStep<In, Out>(
    val name: String,
    val manifest: ProgramManifest,
    val body: ByteArray,
    val mesh: Mesh,
    val shim: (In) -> Out,
) {
    /** Convenience: invoke the shim directly. Equivalent to `step.shim(input)`. */
    operator fun invoke(input: In): Out = shim(input)

    override fun toString(): String =
        "MaestroStep(name=$name, bodyHash=${manifest.bodyHash}, " +
            "inputs=${manifest.inputs.size}, outputs=${manifest.outputs.size})"
}
