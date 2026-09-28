package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * One result of a `stablehlo.custom_call` that reuses an operand's buffer:
 * result [outputIndex] is written in place into operand [operandIndex].
 * Emitted as `#stablehlo.output_operand_alias`.
 */
@ExperimentalTlalocApi
data class OutputOperandAlias(val outputIndex: Int, val operandIndex: Int) {
    init {
        require(outputIndex >= 0 && operandIndex >= 0) {
            "OutputOperandAlias: indices must be non-negative (output $outputIndex, operand $operandIndex)"
        }
    }
}

/**
 * An opaque TPU kernel: a serialized Mosaic module plus the
 * `custom_call_config` fields libtpu reads next to it.
 *
 * This is what a Pallas `pallas_call` lowers to on TPU: JAX emits
 * `stablehlo.custom_call @tpu_custom_call` whose `backend_config` is the JSON
 * `{"custom_call_config": {"body": "<base64>", ...}}`, where `body` is the
 * Mosaic module as MLIR bytecode after `mosaic-serde{serialize=true}`.
 * [backendConfigJson] writes that JSON in the same field order and spacing
 * JAX uses, so a payload exported from JAX re-emits byte for byte.
 *
 * The payload is tied to the Mosaic serialization version it was written at
 * (see [provenance]); libtpu reads versions up to the one it was built with,
 * so a payload must be no newer than the libtpu that runs it.
 *
 * @property kernelName the `kernel_name` attribute on the custom call (the
 *   name profilers and XLA dumps show).
 * @property bodyBase64 the serialized Mosaic module, base64 (standard
 *   alphabet, padded).
 * @property customCallConfig the other `custom_call_config` fields, in emit
 *   order, e.g. `serialization_format = 1`, `needs_layout_passes = true`,
 *   `has_communication`, `collective_id`, `cost_estimate`. Values are
 *   Boolean, Int, Long, Double, String, List or Map<String, *>.
 * @property inputOutputAliases operand index → result index for results
 *   written in place into an operand's buffer (Pallas's
 *   `input_output_aliases`). Emitted as `output_operand_aliases`.
 * @property hasSideEffect emitted as the custom call's `has_side_effect`.
 * @property provenance where the payload came from: jax/jaxlib versions,
 *   Mosaic serialization version, generator. Not emitted.
 */
@ExperimentalTlalocApi
data class MosaicKernel(
    val kernelName: String,
    val bodyBase64: String,
    val customCallConfig: Map<String, Any> = DEFAULT_CONFIG,
    val inputOutputAliases: Map<Int, Int> = emptyMap(),
    val hasSideEffect: Boolean = false,
    val provenance: Map<String, String> = emptyMap(),
) {
    init {
        require(kernelName.isNotEmpty()) { "MosaicKernel: kernelName must not be empty" }
        require(bodyBase64.isNotEmpty()) { "MosaicKernel '$kernelName': bodyBase64 must not be empty" }
        require(bodyBase64.all { it in BASE64_ALPHABET }) {
            "MosaicKernel '$kernelName': bodyBase64 contains characters outside the standard base64 alphabet"
        }
        require("body" !in customCallConfig) {
            "MosaicKernel '$kernelName': 'body' is written from bodyBase64; it cannot also be a customCallConfig key"
        }
        require(inputOutputAliases.values.toSet().size == inputOutputAliases.size) {
            "MosaicKernel '$kernelName': two operands alias the same result: $inputOutputAliases"
        }
    }

    /** The aliases in the emitter's form, ordered by result index. */
    val outputOperandAliases: List<OutputOperandAlias>
        get() = inputOutputAliases.entries
            .sortedBy { it.value }
            .map { (operand, output) -> OutputOperandAlias(output, operand) }

    /**
     * The `backend_config` JSON for `tpu_custom_call`:
     * `{"custom_call_config": {"body": "<base64>", <customCallConfig...>}}`.
     */
    fun backendConfigJson(): String = buildString {
        append("{\"custom_call_config\": {\"body\": ")
        append(jsonString(bodyBase64))
        for ((k, v) in customCallConfig) {
            append(", ").append(jsonString(k)).append(": ").append(jsonValue(v))
        }
        append("}}")
    }

    /** The body is kilobytes of base64; printing it helps nobody. */
    override fun toString(): String =
        "MosaicKernel(kernelName=$kernelName, body=${bodyBase64.length} base64 chars, " +
            "customCallConfig=$customCallConfig, inputOutputAliases=$inputOutputAliases, " +
            "hasSideEffect=$hasSideEffect, provenance=$provenance)"

    companion object {
        /** The call target libtpu registers for Mosaic kernels. */
        const val CALL_TARGET: String = "tpu_custom_call"

        /** The fields JAX 0.10 writes for a TensorCore Pallas kernel with no
         * communication, cost estimate or memory-space overrides. */
        val DEFAULT_CONFIG: Map<String, Any> = linkedMapOf(
            "serialization_format" to 1,
            "needs_layout_passes" to true,
        )

        private const val BASE64_ALPHABET =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="

        /** A JSON string literal: quotes, backslashes and control characters
         * escaped per RFC 8259; everything else verbatim. */
        fun jsonString(s: String): String = buildString(s.length + 2) {
            append('"')
            for (c in s) {
                when {
                    c == '"' -> append("\\\"")
                    c == '\\' -> append("\\\\")
                    c == '\n' -> append("\\n")
                    c == '\r' -> append("\\r")
                    c == '\t' -> append("\\t")
                    c < ' ' -> append("\\u").append(c.code.toString(16).padStart(4, '0'))
                    else -> append(c)
                }
            }
            append('"')
        }

        /** A JSON value with JAX's separators (`", "` and `": "`). */
        fun jsonValue(v: Any?): String = when (v) {
            null -> "null"
            is Boolean -> v.toString()
            is Int, is Long, is Short, is Byte -> v.toString()
            is Double -> {
                require(v.isFinite()) { "MosaicKernel: JSON has no spelling for $v" }
                v.toString()
            }
            is Float -> {
                require(v.isFinite()) { "MosaicKernel: JSON has no spelling for $v" }
                v.toDouble().toString()
            }
            is String -> jsonString(v)
            is List<*> -> v.joinToString(", ", "[", "]") { jsonValue(it) }
            is Map<*, *> -> v.entries.joinToString(", ", "{", "}") { (k, value) ->
                require(k is String) { "MosaicKernel: JSON object keys must be strings; got $k" }
                "${jsonString(k)}: ${jsonValue(value)}"
            }
            else -> error("MosaicKernel: no JSON spelling for ${v::class.simpleName}")
        }
    }
}
