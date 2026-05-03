package io.tlaloc.ir.recognizer.kernel

/**
 * Layer 3 §0.4.253+ — FlashAttention kernel selector.
 *
 * Maps `(KernelTarget) → KernelDescriptor` for the recognized
 * FlashAttention compound. v1 ships per-target entries that match the
 * vendor-published fused kernels:
 *
 * | Target              | Kernel                          | Reference                                    |
 * |---------------------|---------------------------------|----------------------------------------------|
 * | nvidia/h100         | `flash_attn_v3`                 | Dao 2024, "FlashAttention-3"                 |
 * | nvidia/h200         | `flash_attn_v3`                 | same fp8/h200 kernel                         |
 * | nvidia/gb10         | `flash_attn_v3`                 | FA3 main branch sm_100 backend (Grace-Blackwell Spark) |
 * | nvidia/a100         | `flash_attn_v2`                 | Dao 2023, "FlashAttention-2"                 |
 * | nvidia/l40s         | `flash_attn_v2`                 | runs on Ada compute capability               |
 * | nvidia/b100, b200   | `cudnn_multi_head_attention`    | cuDNN fused-attention (Nvidia-official enterprise path on Blackwell) |
 * | amd/mi300x          | `flash_attn_amd`                | AMD-CK / Triton port                         |
 * | google/tpu_v4..v6e  | `tpu_pallas_flash_attention`    | Pallas reference impl                        |
 * | aws/trainium2       | `nki_flash_attention`           | NKI Neuron kernel                            |
 * | tlaloc/cpu_generic  | (null — decompose)              | always-available fallback                    |
 *
 * Other targets fall through to `null` (decompose).
 *
 * # Why GB10 ≠ B100/B200
 *
 * All three Blackwell SKUs share SM_100, but Tlaloc maps them to
 * different fused kernels:
 *
 * - GB10 (Grace-Blackwell Spark) is the dev/edge SKU. We pick
 *   `flash_attn_v3` because (a) FA3's main branch added sm_100 backends
 *   ahead of cuDNN's Blackwell MHA being broadly available, and (b) the
 *   dev-host audience this SKU targets benefits from FA3's full
 *   FP8 e4m3 + e5m2 KV-quant matrix.
 * - B100 / B200 (data-center Blackwell) target enterprise workloads
 *   where cuDNN's `multi_head_attention` is Nvidia's officially-supported
 *   fused-attention path. cuDNN MHA's KV-quant set is narrower
 *   (no fp8_e5m2 in the GA release) but is the conservative pick for
 *   production deployments.
 *
 * Either Blackwell variant can be remapped in a follow-up if the
 * literature shifts (e.g. FA4 ships with a stable cuDNN-equivalent
 * surface).
 *
 * # Backend-specific attrs
 *
 * The descriptor's `customCallAttrs` carries a small backend-config
 * dict per kernel — `softmax_scale` defaulting to `1/sqrt(head_dim)`
 * when the head-dim is inferable from the matched op's input types.
 * v1 leaves this empty; L3.4's cost model populates it during target
 * selection. Out-of-scope is causal-masking (a v2 kernel-selector
 * extension when the recognizer learns to detect `tril`-mask shape).
 */
internal val FlashAttentionKernel: KernelTemplate = KernelTemplate { _, target ->
    // Per-target supported_kv_dtypes (the [KvQuantDtype.nameTag] strings;
    // the L3.4d KV-quant pass reads this attr to decide whether to
    // annotate the COARSENED with a quantization directive).
    when (target) {
        KernelTarget.NVIDIA_H100,
        KernelTarget("nvidia", "h200"),
        KernelTarget.NVIDIA_GB10 ->
            KernelDescriptor(
                "flash_attn_v3", "nvidia", target.arch ?: "h100",
                customCallAttrs = mapOf(
                    "supported_kv_dtypes" to listOf("f32", "bf16", "fp8_e4m3", "fp8_e5m2", "int8"),
                ),
            )

        KernelTarget.NVIDIA_B100,
        KernelTarget.NVIDIA_B200 ->
            KernelDescriptor(
                "cudnn_multi_head_attention", "nvidia", target.arch ?: "b100",
                customCallAttrs = mapOf(
                    // cuDNN MHA on Blackwell GA: fp8_e4m3 yes, fp8_e5m2 not in
                    // the public surface yet. Keep conservative.
                    "supported_kv_dtypes" to listOf("f32", "bf16", "fp8_e4m3", "int8"),
                ),
            )

        KernelTarget.NVIDIA_A100,
        KernelTarget.NVIDIA_L40S ->
            KernelDescriptor(
                "flash_attn_v2", "nvidia", target.arch ?: "a100",
                customCallAttrs = mapOf(
                    "supported_kv_dtypes" to listOf("f32", "bf16", "int8"),
                ),
            )

        KernelTarget.AMD_MI300X ->
            KernelDescriptor(
                "flash_attn_amd", "amd", "mi300x",
                customCallAttrs = mapOf(
                    "supported_kv_dtypes" to listOf("f32", "bf16", "int8"),
                ),
            )

        KernelTarget.GOOGLE_TPU_V4,
        KernelTarget.GOOGLE_TPU_V5E,
        KernelTarget.GOOGLE_TPU_V5P,
        KernelTarget.GOOGLE_TPU_V6E ->
            KernelDescriptor(
                "tpu_pallas_flash_attention", "google", target.arch ?: "tpu_v5e",
                customCallAttrs = mapOf(
                    // TPU v6e adds FP8 (HBM3, sparsecore-coupled);
                    // older v4/v5 stay BF16 + INT8.
                    "supported_kv_dtypes" to if (target == KernelTarget.GOOGLE_TPU_V6E) {
                        listOf("f32", "bf16", "fp8_e4m3", "int8")
                    } else {
                        listOf("f32", "bf16", "int8")
                    },
                ),
            )

        KernelTarget.AWS_TRAINIUM2 ->
            KernelDescriptor(
                "nki_flash_attention", "aws", "trainium2",
                customCallAttrs = mapOf(
                    "supported_kv_dtypes" to listOf("f32", "bf16", "fp8_e4m3", "int8"),
                ),
            )

        else -> null  // CPU_GENERIC + everything else: force decompose.
    }
}
