package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind

/**
 * §0.4.349 — RoPE kernel selector for the KPTX tier. Claims the
 * recognizer's **SUB-form / cos-first** recombination
 * (`out = x_real·cos(θ) − x_imag·sin(θ)` — the LlamaDecoder shape) on
 * the GB10 and lowers it to `stablehlo.custom_call @kptx_rope` under
 * the typed-FFI convention. The KPTX kernel implements exactly that
 * form, so the template inspects the COARSENED's `primal_body`
 * structurally and declines the ADD-form / sin-first variants — those
 * decompose, which is always correct.
 *
 * Launch signature is positional per the registry marshalling:
 * `(x_real_ptr, x_imag_ptr, theta_ptr, out_ptr, n)` matching the
 * COARSENED's `(x_real, x_imag, theta)` operands.
 *
 * Like [RmsNormKernel], **not in [defaultKernelTemplates]** — the KPTX
 * lane stays opt-in per pipeline.
 */
@ExperimentalTlalocApi
val RopeKernel: KernelTemplate = KernelTemplate { coarsened, target ->
    val primal = coarsened.attrs["primal_body"] as? DxirFunction
    val ops = primal?.body?.filterIsInstance<DxirOp>().orEmpty()
    val recombine = ops.lastOrNull()
    val cosOp = ops.firstOrNull { it.op == OpKind.COS }
    val cosMulFirst = recombine?.operands?.getOrNull(0)?.let { first ->
        first is DxirOp && first.op == OpKind.MUL && cosOp != null &&
            first.operands.any { it.id == cosOp.id }
    } ?: false
    when {
        target != KernelTarget.NVIDIA_GB10 -> null
        coarsened.operands.size != 3 -> null
        recombine?.op != OpKind.SUB -> null
        !cosMulFirst -> null
        else -> KernelDescriptor(
            kernelName = "kptx_rope",
            vendor = "tlaloc",
            targetArch = "gb10",
            typedFfi = true,
        )
    }
}
