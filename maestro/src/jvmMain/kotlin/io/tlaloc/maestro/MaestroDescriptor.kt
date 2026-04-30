package io.tlaloc.maestro

import java.util.Base64

/**
 * Layer 2 §0.4.243+ — Maestro-compatible workflow descriptor emitter.
 *
 * # ⚠ Deprecated as of §0.4.249 (Layer 2.5.5)
 *
 * Layer 2.5 vendors Netflix/maestro and registers `Tlaloc` as a real Maestro
 * step type. The masquerade approach this object implements — emitting every
 * Tlaloc step as `type: "Kubernetes"` because Maestro had no public extension
 * API for custom step types — is replaced by direct `type: "Tlaloc"` steps
 * served by `TlalocStepRuntime` inside the vendored Maestro tree.
 *
 * This object remains functional for one release cycle to give downstream
 * users a migration window. New code should:
 *
 * - Author workflows directly with `"type": "Tlaloc"` step JSONs (see
 *   `third-party/maestro/maestro-server/src/test/resources/samples/sample-tlaloc-*.json`
 *   for templates).
 * - Use `MaestroStep.body` (StableHLO bytes) + `MaestroStep.manifest` directly
 *   when populating a workflow JSON's `params.tlaloc.artifact_uri` /
 *   `manifest_ref` fields.
 *
 * Removal target: a future post-Layer-3 release once all in-flight workflows
 * have migrated.
 *
 * # Original (deprecated) conformance approach
 *
 * Netflix Maestro publishes no JSON Schema and no protobuf for its
 * step-definition format — authority is the Java model classes
 * (`maestro-common/src/main/java/com/netflix/maestro/models/definition/`)
 * plus 11 example workflow JSON files in `maestro-server/src/test/resources/samples/`.
 * Tlaloc's emitted descriptor conforms to that observed shape.
 *
 * Pre-Layer-2.5: Maestro defined exactly 9 step types, none of them
 * "Tlaloc"; this emitter shaped every Tlaloc step as a **Kubernetes** step
 * with three Tlaloc-specific params (`image`, `tlaloc_artifact_uri`,
 * `tlaloc_manifest`).
 *
 * Layer 2.5 removed the need for that masquerade by adding `TLALOC` as a
 * first-class enum entry in `StepType.java` inside vendored Maestro.
 *
 * # Schema documented in docs/maestro_descriptor.md
 *
 * The descriptor format is documented at `docs/maestro_descriptor.md`
 * with citations to the relevant Maestro Java model classes and example
 * workflows.
 */
@Deprecated(
    message =
        "Layer 2 masquerade emitter; replaced by first-class Tlaloc step type in §0.4.246. " +
            "Author workflows with `\"type\": \"Tlaloc\"` directly using the samples under " +
            "third-party/maestro/maestro-server/src/test/resources/samples/sample-tlaloc-*.json. " +
            "Removal target: post-Layer-3 release.",
    level = DeprecationLevel.WARNING,
)
object MaestroDescriptor {

    /** Default Tlaloc runtime image. Overridable per-call. */
    const val DEFAULT_RUNTIME_IMAGE = "tlaloc-runtime:0.0.1"

    /** Reshard runtime image; v1 placeholder. */
    const val DEFAULT_RESHARD_IMAGE = "tlaloc-reshard:0.0.1"

    /**
     * Emit a Maestro JSON descriptor for [workflow]. Returns the JSON
     * text; callers can write to disk or feed to a Maestro client.
     */
    fun emit(
        workflow: Workflow,
        owner: String = "tlaloc",
        runtimeImage: String = DEFAULT_RUNTIME_IMAGE,
        reshardImage: String = DEFAULT_RESHARD_IMAGE,
    ): String {
        val workflowId = "tlaloc_${workflow.name}"
        return buildString {
            append("{")
            append("\"properties\":{").append("\"owner\":").append(jsonString(owner)).append("},")
            append("\"workflow\":{")
            append("\"id\":").append(jsonString(workflowId)).append(',')
            append("\"name\":").append(jsonString(workflow.name)).append(',')
            append("\"steps\":")
            append("[")
            // Walk steps in order, interleaving reshard steps where edges
            // demand them. v1 linear-chain: emit step k, then reshard
            // (if needed), then step k+1.
            val stepDescriptors = mutableListOf<String>()
            for ((i, step) in workflow.steps.withIndex()) {
                val nextEdge = workflow.edges.firstOrNull { it.fromStep == step.name }
                val nextStepName = nextEdge?.toStep
                val reshardStepName = if (nextEdge != null && nextEdge.reshardKind != ReshardKind.None) {
                    "reshard_${step.name}_to_${nextEdge.toStep}"
                } else {
                    null
                }
                val effectiveSuccessor = reshardStepName ?: nextStepName
                stepDescriptors += stepDescriptor(
                    step = step,
                    successor = effectiveSuccessor,
                    runtimeImage = runtimeImage,
                )
                if (reshardStepName != null && nextEdge != null) {
                    stepDescriptors += reshardStepDescriptor(
                        reshardName = reshardStepName,
                        successor = nextEdge.toStep,
                        edge = nextEdge,
                        reshardImage = reshardImage,
                    )
                }
            }
            append(stepDescriptors.joinToString(","))
            append("]")
            append("}")
            append("}")
        }
    }

    private fun stepDescriptor(
        step: MaestroStep<*, *>,
        successor: String?,
        runtimeImage: String,
    ): String = buildString {
        append("{\"step\":{")
        append("\"id\":").append(jsonString(step.name)).append(',')
        append("\"type\":\"Kubernetes\",")
        append("\"params\":{")
        append("\"image\":{\"value\":").append(jsonString(runtimeImage)).append(",\"type\":\"STRING\"},")
        append("\"tlaloc_artifact_uri\":{\"value\":")
            .append(jsonString(artifactUri(step.body, step.manifest.bodyHash)))
            .append(",\"type\":\"STRING\"},")
        append("\"tlaloc_manifest\":{\"value\":")
            .append(jsonString(step.manifest.toJson()))
            .append(",\"type\":\"STRING\"}")
        append("},")
        append("\"transition\":")
        if (successor != null) {
            append("{\"successors\":{").append(jsonString(successor)).append(":\"true\"}}")
        } else {
            append("{}")
        }
        append("}}")
    }

    private fun reshardStepDescriptor(
        reshardName: String,
        successor: String,
        edge: WorkflowEdge,
        reshardImage: String,
    ): String = buildString {
        append("{\"step\":{")
        append("\"id\":").append(jsonString(reshardName)).append(',')
        append("\"type\":\"Kubernetes\",")
        append("\"params\":{")
        append("\"image\":{\"value\":").append(jsonString(reshardImage)).append(",\"type\":\"STRING\"},")
        append("\"reshard_kind\":{\"value\":").append(jsonString(edge.reshardKind.name)).append(",\"type\":\"STRING\"},")
        append("\"from_mesh\":{\"value\":").append(jsonString(edge.fromMesh)).append(",\"type\":\"STRING\"},")
        append("\"to_mesh\":{\"value\":").append(jsonString(edge.toMesh)).append(",\"type\":\"STRING\"}")
        append("},")
        append("\"transition\":{\"successors\":{").append(jsonString(successor)).append(":\"true\"}}")
        append("}}")
    }

    /**
     * Compose a `data:` URI for the artifact body. v1 inlines the bytes
     * as base64; Layer 3+ swaps in a real content-addressed registry
     * (`oci://<registry>/<image>@sha256:<hex>` or similar).
     */
    private fun artifactUri(body: ByteArray, bodyHash: String): String {
        val b64 = Base64.getEncoder().encodeToString(body)
        return "data:application/x-tlaloc-stablehlo;sha256=$bodyHash;base64,$b64"
    }

    private fun jsonString(s: String): String = buildString {
        append('"')
        for (c in s) {
            when (c) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (c.code < 0x20) append("\\u%04x".format(c.code)) else append(c)
            }
        }
        append('"')
    }
}
