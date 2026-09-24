package io.tlaloc.maestro

import io.tlaloc.ir.DxirType

/**
 * Typed manifest for a [MaestroStep] artifact.
 *
 * The manifest is a fully-self-describing snapshot of a `program { }`
 * block: the input/output type structure (rank, dtype, named axes), the
 * mesh requirement, the placeholder sharding spec, and a content-
 * addressed hash of the StableHLO body bytes. Two semantically identical
 * programs produce manifests with identical [bodyHash]es.
 *
 * v1 keeps the JSON serialization minimal — hand-rolled, no
 * kotlinx.serialization dependency. The schema is small and stable.
 */
data class ProgramManifest(
    /** Step name; matches the `program(name = ...)` argument. */
    val name: String,
    /** Input tensor type descriptors, in argument order. */
    val inputs: List<TypeDescriptor>,
    /** Output tensor type descriptors, in return order. */
    val outputs: List<TypeDescriptor>,
    /**
     * Mesh requirement — the runtime [io.tlaloc.core.MeshSpec] name this
     * step expects. Cross-mesh transfers in a workflow inspect this field
     * and insert reshard steps when adjacent steps disagree.
     */
    val meshRequirement: String,
    /** SHA-256 hex of the StableHLO body bytes (content-addressed). */
    val bodyHash: String,
    /** Placeholder for cross-step Shardy propagation. Currently always empty. */
    val shardingSpec: List<String> = emptyList(),
    /**
     * Per-(vendor, arch) compile-decision tuples
     * recording what the kernel-selection pipeline picked for each device target.
     * Default empty for callers that haven't run the populator yet
     * (`Tlaloc.program { }` produces an empty list;
     * `populateBackendMatrix` fills it).
     */
    val backendMatrix: List<BackendTarget> = emptyList(),
) {
    /** JSON serialization. v1 is hand-rolled; keep schema strictly stable. */
    fun toJson(): String = buildString {
        append("{")
        append("\"name\":").append(jsonString(name)).append(',')
        append("\"inputs\":").append(inputs.toJsonArray()).append(',')
        append("\"outputs\":").append(outputs.toJsonArray()).append(',')
        append("\"meshRequirement\":").append(jsonString(meshRequirement)).append(',')
        append("\"bodyHash\":").append(jsonString(bodyHash)).append(',')
        append("\"shardingSpec\":").append(shardingSpec.toJsonStringArray()).append(',')
        append("\"backendMatrix\":").append(backendMatrix.toJsonBackendArray())
        append("}")
    }

    companion object {
        fun fromJson(json: String): ProgramManifest = ManifestJsonParser(json).parse()
    }
}

/**
 * Description of one tensor at a step boundary. Captures the same
 * structural shape carried by [DxirType]: dtype, dims (sentinel `-1`
 * for symbolic), and per-axis names.
 */
data class TypeDescriptor(
    val dtype: String,
    val dims: List<Int>,
    val axisNames: List<String?> = emptyList(),
) {
    init {
        require(axisNames.isEmpty() || axisNames.size == dims.size) {
            "axisNames size ${axisNames.size} must match dims size ${dims.size}"
        }
    }

    fun toJson(): String = buildString {
        append("{")
        append("\"dtype\":").append(jsonString(dtype)).append(',')
        append("\"dims\":").append(dims.joinToString(",", "[", "]")).append(',')
        append("\"axisNames\":")
        if (axisNames.isEmpty()) {
            append("[]")
        } else {
            append(axisNames.joinToString(",", "[", "]") { it?.let(::jsonString) ?: "null" })
        }
        append("}")
    }

    companion object {
        /** Project a [DxirType] into a transport-shaped descriptor. */
        fun fromDxirType(t: DxirType): TypeDescriptor = TypeDescriptor(
            dtype = t.dtype.name,
            dims = t.dims,
            axisNames = if (t.axisNames.isEmpty()) emptyList() else t.axisNames.toList(),
        )

        /**
         * Parse a single [TypeDescriptor] JSON object (inverse of [toJson]).
         * Needed by [io.tlaloc.maestro.SerializedBufferHandle]
         * for cross-pod descriptor parsing.
         */
        fun fromJson(json: String): TypeDescriptor =
            ManifestJsonParser(json).parseTypeDescriptor()
    }
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

private fun List<TypeDescriptor>.toJsonArray(): String =
    joinToString(",", "[", "]") { it.toJson() }

private fun List<String>.toJsonStringArray(): String =
    joinToString(",", "[", "]") { jsonString(it) }

@JvmName("toJsonBackendArray")
private fun List<BackendTarget>.toJsonBackendArray(): String =
    joinToString(",", "[", "]") { it.toJson() }
