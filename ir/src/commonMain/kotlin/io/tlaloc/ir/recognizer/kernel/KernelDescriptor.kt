package io.tlaloc.ir.recognizer.kernel

/**
 * Layer 3 §0.4.253+ — describes a vendor-fused kernel chosen for a
 * recognized pattern on a specific [KernelTarget].
 *
 * The descriptor is the bridge between L3 pattern recognition and
 * downstream lowering. After L3.3, each `OpKind.COARSENED` op may
 * carry an optional `kernel_descriptor` attribute (set when a kernel
 * template matched the target); StableHLO emit consumes that attr to
 * emit a `stablehlo.custom_call` instead of inlining the primal body.
 *
 * @property kernelName the vendor's kernel identifier — e.g.
 *   `flash_attn_v3`, `cudnn_multi_head_attention`, `tpu_attention`.
 *   Used as the `call_target_name` of the emitted custom call.
 * @property vendor "nvidia" / "amd" / "google" / "aws" / "tlaloc".
 * @property targetArch arch string this kernel targets (`h100`,
 *   `tpu_v5e`, etc.). May differ from the target's arch when the
 *   kernel is portable across a family.
 * @property customCallAttrs additional vendor-specific attributes
 *   passed via `stablehlo.custom_call`'s `backend_config` (e.g.
 *   `softmax_scale`, `is_causal`, `head_dim`). Free-form map.
 */
data class KernelDescriptor(
    val kernelName: String,
    val vendor: String,
    val targetArch: String,
    val customCallAttrs: Map<String, Any> = emptyMap(),
) {
    /**
     * Name of the attribute key the L3.3 lowering pass uses when
     * stashing this descriptor on a `OpKind.COARSENED` op for downstream
     * consumers. Centralised as a constant so renames are one-edit.
     */
    companion object {
        const val ATTR_KEY: String = "kernel_descriptor"
    }
}
