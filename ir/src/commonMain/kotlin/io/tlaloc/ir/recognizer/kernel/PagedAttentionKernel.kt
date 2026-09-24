package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.F32
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.PagedAttentionAttrs

/**
 * The **paged-attention kernel selector**, and the
 * first claiming template that claims a *first-class op kind* rather than
 * a COARSENED splice.
 *
 * # Why this one is different
 *
 * Every earlier template ([FlashAttentionKernel], [RmsNormKernel],
 * [AttentionKernel], [RopeKernel], [CrossEntropyKernel]) claims an
 * `OpKind.COARSENED` produced by the VJP coarsener — a compile-time
 * artefact carrying a `primal_body`, whose *fallback* is decomposition
 * back into that body. `OpKind.PAGED_ATTENTION` is not that: it is a real
 * op with an interpreter arm and a gather-composed StableHLO emission of
 * its own. So the fallback here is **the op itself** — decline
 * and the graph emits exactly what it emitted before claiming existed.
 * That makes inference-side claiming strictly safer than the COARSENED
 * kind: there is no "neither annotated nor decomposed" hole to fall into.
 *
 * This is also where Tlaloc's differentiator meets the serving path.
 * Helion-style kernel libraries are *hand-invoked*: the model author
 * writes the call. Here the author writes `PAGED_ATTENTION`, the
 * recognizer's claiming pass sees a GB10 under it, and the fused kernel
 * appears — or does not, on a machine that has no KPTX tier, with the
 * same numbers either way.
 *
 * # What it claims
 *
 * `PAGED_ATTENTION(query, keyCache, valueCache, blockTables, seqLens)` on
 * [KernelTarget.NVIDIA_GB10], validated through [PagedAttentionAttrs] so
 * the template and the emitter cannot disagree about what a legal paged
 * attention is. Lowers to `stablehlo.custom_call @kptx_paged_attention`
 * under the typed-FFI convention with **one scratch result**: the
 * `S[numSeqs·numHeads, maxBlocksPerSeq·blockSize]` score matrix the
 * three-stage launch chain (scores → row softmax → output) stages
 * through, XLA-owned (see [KernelDescriptor.scratchResults]).
 *
 * `scale` rides [KernelDescriptor.customCallAttrs] so the emitted MLIR
 * names it. The v1 kernel *bakes* it into its PTX (the specialization
 * cache in `KptxKernels.pagedAttentionModule`), so a registration serves
 * one scale; the attr is there so a later handler that reads the frame's
 * decoded attrs can become scale-generic without an emit change.
 *
 * Declines:
 * - non-GB10 targets — the KPTX tier is NVIDIA-only by construction.
 * - anything [PagedAttentionAttrs] refuses, and any dtype other than F32:
 *   the kernel is the correctness tier's f32 loops (bf16 pools are not
 *   supported, as they are not in the op's own emission).
 *
 * **Not in [defaultInferenceKernelTemplates]** — the KPTX lane is opt-in
 * per pipeline, the standing convention for every KPTX template: claiming
 * a kernel whose launch chain was never registered with
 * `KptxKernelRegistry` would emit a custom_call XLA cannot resolve.
 */
@ExperimentalTlalocApi
val PagedAttentionKernel: KernelTemplate = KernelTemplate { node, target ->
    if (target != KernelTarget.NVIDIA_GB10) return@KernelTemplate null
    if (node.op != OpKind.PAGED_ATTENTION) return@KernelTemplate null
    val parsed = runCatching { PagedAttentionAttrs.parse(node, "PagedAttentionKernel") }
        .getOrNull() ?: return@KernelTemplate null
    val allF32 = node.operands.take(3).all { it.type.dtype == F32 } &&
        node.type.dtype == F32
    if (!allF32) return@KernelTemplate null
    KernelDescriptor(
        kernelName = "kptx_paged_attention",
        vendor = "tlaloc",
        targetArch = "gb10",
        customCallAttrs = mapOf("scale" to parsed.scale),
        typedFfi = true,
        scratchResults = listOf(
            listOf(
                parsed.numSeqs * parsed.numHeads,
                parsed.maxBlocksPerSeq * parsed.blockSize,
            ),
        ),
    )
}
