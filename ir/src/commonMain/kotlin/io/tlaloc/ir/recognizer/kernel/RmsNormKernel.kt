package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * KPTX v1.8 (§0.4.336) — RMS-norm kernel selector for the **KPTX tier**:
 * Tlaloc's own hand-written PTX kernels dispatched through
 * `KptxKernelRegistry` (runtime-pjrt) rather than a vendor library.
 *
 * Claims the eps-form RmsNorm COARSENED (operands `(x, eps)`, the shape
 * the LlamaDecoder's pre-attn / pre-MLP norms coarsen into) on the GB10
 * and lowers it to `stablehlo.custom_call @kptx_rms_norm` under the
 * typed-FFI convention (§0.4.332). The kernel's signature is positional
 * per the registry's inputs-then-outputs marshalling:
 * `(x_ptr, eps_ptr, out_ptr, n_cols)` — eps arrives as the coarsener's
 * `[rows, 1]` operand (matching the keepdims mean) and the kernel loads
 * `eps[row]` per CTA, so the same JIT'd kernel serves any eps value.
 *
 * Declines everything else:
 * - non-GB10 targets — vendor tiers keep their own templates
 *   ([FlashAttentionKernel]); KPTX is NVIDIA-only by construction
 *   (docs/KPTX_PLAN.md standing decisions).
 * - the no-eps form — the kernel signature expects the eps operand.
 * - non-rank-2 `x` — the launch shape is one CTA per token row over
 *   `[rows, cols]`.
 *
 * **Not in [defaultKernelTemplates] yet.** The KPTX lane is opt-in per
 * pipeline (tests and benchmarks pass an explicit registry); flipping
 * the default happens when the kptx-cuda lane graduates to the
 * production GPU path (v2+, after the DSL rewrite claims coarse ops
 * wholesale — KPTX plan task 15).
 */
@ExperimentalTlalocApi
val RmsNormKernel: KernelTemplate = KernelTemplate { coarsened, target ->
    val x = coarsened.operands.firstOrNull()
    when {
        target != KernelTarget.NVIDIA_GB10 -> null
        coarsened.operands.size != 2 -> null
        x == null || x.type.rank != 2 -> null
        else -> KernelDescriptor(
            kernelName = "kptx_rms_norm",
            vendor = "tlaloc",
            targetArch = "gb10",
            typedFfi = true,
        )
    }
}
