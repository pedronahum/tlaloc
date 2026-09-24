package io.tlaloc.maestro

/**
 * Per-(vendor, arch) compile-decision tuple for a
 * single program manifest.
 *
 * One [BackendTarget] entry records what the kernel-selection pipeline (recognize →
 * coarsen → lowerKernelChoice → cost-model) decided for a specific
 * device target. A program is potentially compiled for many targets;
 * `ProgramManifest.backendMatrix` is the materialized list of
 * decisions, which downstream consumers (TlalocStepRuntime, scheduler,
 * IDE tooling) use to pick which artifact to load on a given pod.
 *
 * # Why structured, not free-form strings
 *
 * The per-target decision is a fixed-arity tuple: vendor, arch, kernel-name (or null for
 * decompose), kv-quant-dtype (or null), cost in microseconds (or null
 * if not yet estimated). String concatenation would re-encode the
 * tuple per consumer; structured carriage is one-edit-away from a
 * non-trivial schema bump.
 *
 * # Forward compat
 *
 * Adding a new field = bump the JSON schema + extend [ManifestJsonParser]
 * with the new key (with a sensible default). Removing a field is a
 * breaking change and should bump a manifest-format-version field
 * (the manifest does not carry one yet).
 *
 * @property vendor lower-case vendor string (`"nvidia"`, `"google"`,
 *   `"amd"`, `"aws"`, `"tlaloc"`). Matches `KernelTarget.vendor`.
 * @property arch device-specific identifier (`"h100"`, `"tpu_v5e"`,
 *   `"cpu_generic"`). Matches `KernelTarget.arch`.
 * @property kernelName the kernel identifier the kernel-choice lowering pass
 *   picked for this target — e.g. `"flash_attn_v3"`. `null` when the
 *   kernel registry returned no match and the pass decomposed.
 * @property kvQuantDtype the KV-quant dtype name tag (e.g.
 *   `"fp8_e4m3"`, `"int8"`) the KV-quant pass annotated. `null`
 *   when no quantization was applied (either user didn't request it or
 *   the kernel didn't support it).
 * @property costMicroseconds roofline-style time estimate from the
 *   roofline cost model. `null` when the populator wasn't asked to
 *   compute it. Useful for downstream scheduling.
 */
data class BackendTarget(
    val vendor: String,
    val arch: String,
    val kernelName: String? = null,
    val kvQuantDtype: String? = null,
    val costMicroseconds: Double? = null,
) {
    /**
     * JSON serialization — produces `{"vendor":"x","arch":"y", ...}`.
     * Matches the parser's expected schema in [ManifestJsonParser].
     */
    fun toJson(): String = buildString {
        append("{")
        append("\"vendor\":").append(jsonString(vendor)).append(',')
        append("\"arch\":").append(jsonString(arch)).append(',')
        append("\"kernelName\":")
        if (kernelName == null) append("null") else append(jsonString(kernelName))
        append(",\"kvQuantDtype\":")
        if (kvQuantDtype == null) append("null") else append(jsonString(kvQuantDtype))
        append(",\"costMicroseconds\":")
        if (costMicroseconds == null) append("null") else append(costMicroseconds.toString())
        append("}")
    }

    companion object {
        fun fromJson(json: String): BackendTarget = ManifestJsonParser(json).parseBackendTarget()
    }
}

/**
 * Local copy of the manifest's string-escaping rules — keeps
 * BackendTarget self-contained without exposing the
 * `private fun jsonString` from [ProgramManifest].
 */
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
