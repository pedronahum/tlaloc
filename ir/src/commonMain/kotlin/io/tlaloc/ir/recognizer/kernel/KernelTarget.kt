package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * Layer 3 §0.4.253+ — identifies the device target a kernel template
 * picks against. Carries (vendor, arch) — the discriminator a vendor
 * uses to ship per-arch fused kernels (e.g., NVIDIA ships
 * `flash_attn_v3` for H100 but `flash_attn_v2` for A100).
 *
 * The strings are intentionally free-form rather than enum-typed —
 * vendors add architectures (h200, mi355x, tpu_v6e) outside Tlaloc's
 * release cadence, and an enum would force a release cycle for every
 * new arch. Templates pattern-match on these strings.
 *
 * # Common targets (string keys used across L3.3+ + L3.4 cost model)
 *
 * - `nvidia`: `h100`, `h200`, `a100`, `l40s`, `gb10`, `b100`, `b200`
 * - `amd`: `mi300x`, `mi355x`
 * - `google`: `tpu_v4`, `tpu_v5e`, `tpu_v5p`, `tpu_v6e`
 * - `aws`: `trainium2`, `trainium3`
 * - `tlaloc`: `cpu_generic` (the always-available decompose-only fallback)
 *
 * @property vendor "nvidia" / "amd" / "google" / "aws" / "tlaloc". Lower-case.
 * @property arch device-specific identifier ("h100", "tpu_v5e", "cpu_generic").
 *   `null` means "any arch from this vendor"; templates may use this to
 *   match a vendor-wide fallback (e.g. a portable CUDA kernel that runs
 *   on any sm_70+ arch).
 */
@ExperimentalTlalocApi
data class KernelTarget(
    val vendor: String,
    val arch: String?,
) {
    companion object {
        val NVIDIA_H100 = KernelTarget("nvidia", "h100")
        val NVIDIA_A100 = KernelTarget("nvidia", "a100")
        val NVIDIA_L40S = KernelTarget("nvidia", "l40s")
        // Blackwell (sm_100). GB10 is the Grace-Blackwell Spark dev/edge
        // SKU; B100/B200 are the data-center Blackwell SKUs. They share a
        // die family (SM_100) but Tlaloc's kernel registry maps them to
        // different fused kernels — see `FlashAttentionKernel.kt`.
        val NVIDIA_GB10 = KernelTarget("nvidia", "gb10")
        val NVIDIA_B100 = KernelTarget("nvidia", "b100")
        val NVIDIA_B200 = KernelTarget("nvidia", "b200")
        val AMD_MI300X = KernelTarget("amd", "mi300x")
        val GOOGLE_TPU_V4 = KernelTarget("google", "tpu_v4")
        val GOOGLE_TPU_V5E = KernelTarget("google", "tpu_v5e")
        val GOOGLE_TPU_V5P = KernelTarget("google", "tpu_v5p")
        val GOOGLE_TPU_V6E = KernelTarget("google", "tpu_v6e")
        val AWS_TRAINIUM2 = KernelTarget("aws", "trainium2")

        /**
         * Always-available, no-vendor-kernel target. KernelTemplates that
         * return `null` for [CPU_GENERIC] force the decompose path.
         */
        val CPU_GENERIC = KernelTarget("tlaloc", "cpu_generic")
    }
}
