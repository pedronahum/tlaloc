package io.tlaloc.ir.recognizer.quant

/**
 * Layer 3 §0.4.257+ — KV-cache quantization configuration carried as
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
