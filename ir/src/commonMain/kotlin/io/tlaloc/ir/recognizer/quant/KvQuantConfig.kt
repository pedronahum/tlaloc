package io.tlaloc.ir.recognizer.quant

/**
 * KV-cache quantization configuration carried as
 * metadata on a FlashAttention `OpKind.COARSENED` op.
 *
 * # Why metadata, not type-system change
 *
 * Tlaloc's [io.tlaloc.core.DType] only has F32 / F64 / I32 / I64 / Bool —
 * no native int8 or fp8. A full type-system extension (adding I8, FP8,
 * FP4) is multi-week work that touches the FIR plugin, the StableHLO
 * emitter, and the interpreter. KV-quant for v1 ships as a *codegen
 * directive*: the COARSENED op carries a [KvQuantDtype] attr telling the
 * downstream kernel custom-call "materialize K and V as `<dtype>` at
 * runtime; perform on-the-fly dequantization inside the kernel."
 *
 * The DXIR-level types remain F32 (or whatever the user wrote);
 * runtime-level memory layout follows the attr.
 *
 * # Compatibility with kernel templates
 *
 * Each [io.tlaloc.ir.recognizer.kernel.KernelDescriptor] declares which
 * KV dtypes it supports via its `customCallAttrs["supported_kv_dtypes"]`
 * entry. The KV-quant pass only annotates COARSENED ops whose kernel
 * descriptor supports the requested dtype. Mismatches silently fall
 * through (no annotation, no error) — the user's request is best-effort.
 */
enum class KvQuantDtype(val nameTag: String, val bitsPerElement: Int) {
    F32("f32", 32),
    BF16("bf16", 16),
    FP8_E4M3("fp8_e4m3", 8),
    FP8_E5M2("fp8_e5m2", 8),
    INT8("int8", 8),
    INT4("int4", 4),
    ;

    /**
     * Whether a value of this dtype is a SMALL INTEGER
     * CODE read against a scale (`x ≈ code * scale`), which is what
     * [io.tlaloc.ir.inference.KvQuantPool]'s symmetric-absmax contract and
     * [io.tlaloc.ir.OpKind.DEQUANTIZE_KV] implement.
     *
     * False for the float formats, and that is a REFUSAL rather than a gap:
     * an fp8 value's "code" is a bit pattern with its own exponent field, so
     * an integer-code path would have to either store the pattern (making the
     * multiply meaningless) or round twice. fp8 KV-quant wants a narrow
     * [io.tlaloc.core.DType] the way bf16 has one.
     */
    val isIntegerCoded: Boolean get() = this == INT8 || this == INT4

    /**
     * The largest magnitude a code may take, for the integer-coded dtypes:
     * 127 for int8, 7 for int4 — the SYMMETRIC range, so `-128` is not used
     * even though int8 can hold it. Asymmetry buys one extra code and costs
     * the property that `quantize(-x) == -quantize(x)`, which is the property
     * every error bound in [io.tlaloc.ir.inference.KvQuantPool] is derived
     * from. Refuses by name for the float formats.
     */
    val codeMax: Int get() = when (this) {
        INT8 -> 127
        INT4 -> 7
        else -> error(
            "KvQuantDtype.codeMax: $nameTag is not an integer-coded dtype — see " +
                "isIntegerCoded for why fp8/bf16/f32 are refused here rather than approximated",
        )
    }
}

/**
 * Per-tensor / per-head scaling strategy. v1 supports two: a single
 * scalar per K/V tensor (cheapest, lowest accuracy) or a vector of
 * per-head scales (one scalar per attention head, the standard choice
 * for production inference).
 */
enum class KvScaleStrategy { PER_TENSOR, PER_HEAD }

/**
 * Bundled config. Goes onto the COARSENED op as `kv_quant_config: KvQuantConfig`.
 *
 * @property dtype the quantized dtype K and V should materialize as.
 * @property scaleStrategy where the per-element scale lives.
 */
data class KvQuantConfig(
    val dtype: KvQuantDtype,
    val scaleStrategy: KvScaleStrategy,
) {
    companion object {
        const val ATTR_KEY: String = "kv_quant_config"

        /**
         * Production-default: FP8 E4M3 with per-head scales. The
         * standard configuration for H100 / Hopper attention; matches
         * what the FlashAttention-3 reference impl ships.
         */
        val FP8_PER_HEAD = KvQuantConfig(KvQuantDtype.FP8_E4M3, KvScaleStrategy.PER_HEAD)

        /** Inference-cheap: int8 per-tensor. */
        val INT8_PER_TENSOR = KvQuantConfig(KvQuantDtype.INT8, KvScaleStrategy.PER_TENSOR)
    }
}
