package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.ir.DxirOp

/**
 * Pattern-specific kernel selector.
 *
 * Given a coarsened op (an `OpKind.COARSENED` produced by the VJP
 * coarsener) and a [KernelTarget], a `KernelTemplate` returns either:
 *
 * - A [KernelDescriptor] — pick this vendor-fused kernel.
 * - `null` — no fused kernel for this target; force the decompose path.
 *
 * # Adding a kernel
 *
 * 1. Find or create the per-pattern file (`FlashAttentionKernel.kt`).
 * 2. Add a target → kernel-name mapping inside the template's `pickFor`.
 *
 * # Why a function and not a Map<KernelTarget, KernelDescriptor>?
 *
 * Per-target overrides commonly depend on the matched op's structure
 * (e.g., flash-attn-v3 only on H100 *and* head-dim ≤ 256; cudnn-MHA only
 * on rank-4 inputs). Static maps don't compose those constraints.
 * Functions do.
 */
@ExperimentalTlalocApi
fun interface KernelTemplate {
    /**
     * Pick a kernel for the given coarsened op + target, or return
     * `null` to force the decompose fallback.
     *
     * @param coarsened the `OpKind.COARSENED` op carrying the matched
     *   primal_body / gradient_body (from coarsening). The
     *   template can inspect operand types, primal_body structure, etc.
     * @param target the device descriptor.
     */
    fun pickFor(coarsened: DxirOp, target: KernelTarget): KernelDescriptor?
}
