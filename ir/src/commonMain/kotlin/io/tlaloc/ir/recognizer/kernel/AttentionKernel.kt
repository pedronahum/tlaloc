package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi

/**
 * Attention kernel selector for the KPTX tier. Claims the
 * FlashAttention COARSENED — operands `(Q[T,D], Kᵀ[D,T], V[T,D])`,
 * single result `[T,D]` (the LlamaDecoder primal pre-transposes K, so
 * the second operand is already the score matmul's rhs) — on the GB10
 * and lowers it to `stablehlo.custom_call @kptx_attention` under the
 * typed-FFI convention with one scratch result: the `S[T,T]` score
 * matrix the three-stage launch chain (scores → row-softmax → output)
 * stages through, XLA-owned (see [KernelDescriptor.scratchResults]).
 *
 * The GQA/MQA COARSENED (broadcast-expanded K/V chains) is the planned
 * extension of the same kernel — the expansions are indexing, not new
 * math; it declines here until the chain grows head-index params.
 *
 * Like the other KPTX templates, **not in [defaultKernelTemplates]**.
 */
@ExperimentalTlalocApi
val AttentionKernel: KernelTemplate = KernelTemplate { coarsened, target ->
    val q = coarsened.operands.getOrNull(0)
    val kt = coarsened.operands.getOrNull(1)
    val v = coarsened.operands.getOrNull(2)
    val shapesMatch = q != null && kt != null && v != null &&
        q.type.rank == 2 && kt.type.rank == 2 && v.type.rank == 2 &&
        q.type.dims[1] == kt.type.dims[0] &&      // D agrees
        kt.type.dims[1] == q.type.dims[0] &&      // Kᵀ's cols = T
        v.type.dims == q.type.dims                // V is [T,D]
    when {
        target != KernelTarget.NVIDIA_GB10 -> null
        coarsened.operands.size != 3 -> null
        !shapesMatch -> null
        coarsened.numResults != 1 || coarsened.type.dims != q!!.type.dims -> null
        else -> KernelDescriptor(
            kernelName = "kptx_attention",
            vendor = "tlaloc",
            targetArch = "gb10",
            typedFfi = true,
            scratchResults = listOf(listOf(q.type.dims[0], q.type.dims[0])),
        )
    }
}
