package io.tlaloc.ir.inference

/**
 * How the projection weights of a decoder's layers are stored on the device.
 *
 * [NONE] (the default) stages every weight in [HfDecoderConfig.weightDType].
 *
 * [INT8] is weight-only quantization, opt-in. Each layer's Linear weights
 * (q, k, v, o, the attention output gate, gate, up and down) are stored as
 * int8 codes with one f32 scale per output channel:
 *
 * ```
 *   scale[o] = max_i |W[o, i]| / 127        (1 when the row is all zero)
 *   code[o, i] = clamp(round_half_even(W[o, i] / scale[o]), -127, 127)
 * ```
 *
 * computed in f32 from the checkpoint's own values. In the graph a
 * projection widens the codes to the compute dtype (exact: every code is a
 * bf16 and an f32), multiplies as before into f32, and then multiplies each
 * output column by its scale: `(x @ code) * scale`, which is `x @ (code * scale)`
 * up to f32 rounding. The embedding table, the norms and the head keep
 * [HfDecoderConfig.weightDType].
 *
 * [FP8] is the same with e4m3fn codes: `scale[o] = max_i |W[o, i]| / 448`
 * and `code[o, i] = e4m3fn(W[o, i] / scale[o])` (round to nearest even).
 * Its log spacing keeps the small values of a row whose largest is far
 * larger, which matters for weights read from NVFP4 or block-FP8
 * checkpoints, whose scales vary along a row.
 *
 * Either halves the bytes a decode step reads for a bf16 model (a quarter for
 * f32) and changes the numerics: a quantized model is certified by how close
 * its outputs stay to the unquantized one, not by identity. Quantized: the
 * large projections (attention q/k/v/o and gate, the MLP, a Gated DeltaNet's
 * q/k/v, z and output projections, the shared expert's projections). The
 * small ones (in_proj_b/a, a router, a shared expert's gate) stay in the
 * weight dtype, as the quantized checkpoints keep them.
 */
enum class WeightQuant(val tag: String) {
    NONE("none"),
    INT8("int8"),
    FP8("fp8"),
    ;

    /** The dtype of this format's codes. */
    val codeDType: io.tlaloc.core.DType
        get() = when (this) {
            INT8 -> io.tlaloc.core.I8
            FP8 -> io.tlaloc.core.F8E4M3FN
            NONE -> error("WeightQuant.NONE has no codes")
        }

    companion object {
        /** The value whose [tag] is [tag], or a refusal naming the known tags. */
        fun parse(tag: String): WeightQuant =
            entries.firstOrNull { it.tag == tag.trim().lowercase() }
                ?: throw IllegalArgumentException(
                    "weightQuant must be one of ${entries.joinToString { it.tag }}, got '$tag'",
                )
    }
}
