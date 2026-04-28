package io.tlaloc.maestro

import java.util.Base64

/**
 * Layer 2 §0.4.243+ — Maestro-compatible workflow descriptor emitter.
 *
 * # Conformance approach
 *
 * Netflix Maestro publishes no JSON Schema and no protobuf for its
 * step-definition format — authority is the Java model classes
 * (`maestro-common/src/main/java/com/netflix/maestro/models/definition/`)
 * plus 11 example workflow JSON files in `maestro-server/src/test/resources/samples/`.
 * Tlaloc's emitted descriptor conforms to that observed shape.
 *
 * Maestro defines exactly 9 step types — none of them is "Tlaloc". There
 * is no public extension API for registering custom step types. So every
 * Tlaloc step in the emitted descriptor is shaped as a **Kubernetes**
 * step (one of Maestro's 5 leaf executable types) with three Tlaloc-
 * specific params:
 *
 * - `image`: the Tlaloc-runtime container image (e.g. `tlaloc-runtime:0.0.1`).
 *   Maestro's Kubernetes step launches this; the runtime image's
 *   entrypoint reads the artifact URI + manifest from its own params.
 * - `tlaloc_artifact_uri`: a `sha256:<hex>` URI pointing at the content-
 *   addressed StableHLO body (the manifest's [ProgramManifest.bodyHash]).
 *   v1 inlines the body as a base64-encoded `data:` URI; Layer 3 plugs in
 *   a real registry.
 * - `tlaloc_manifest`: the JSON-serialized [ProgramManifest], so the
 *   runtime image has the typed input/output descriptors at runtime.
 *
 * Reshard edges are emitted as synthetic Kubernetes steps with image
 * `tlaloc-reshard:0.0.1` between adjacent compute steps; their params
 * carry the source/target mesh names.
 *
 * # Schema documented in docs/maestro_descriptor.md
 *
 * The descriptor format is documented at `docs/maestro_descriptor.md`
 * with citations to the relevant Maestro Java model classes and example
 * workflows.
 */
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
