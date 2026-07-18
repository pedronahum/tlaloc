package io.tlaloc.ir.recognizer.kernel

/**
 * §0.4.351 — CrossEntropy kernel selector for the KPTX tier. Claims the
 * coarsened CE (`loss = Σ labels·log(softmax(logits))`, operands
 * `(logits, labels)`, scalar result) on the GB10 and lowers it to
 * `stablehlo.custom_call @kptx_cross_entropy` under the typed-FFI
 * convention **with a scratch result**: the §0.4.350 launch chain
 * stages its per-row pass through an XLA-owned `row_loss[rows]`
 * appended as result #1 (the op's scalar stays result #0; nothing
 * downstream references the scratch).
 *
 * Like the other KPTX templates, **not in [defaultKernelTemplates]** —
 * the lane stays opt-in per pipeline.
 */
val CrossEntropyKernel: KernelTemplate = KernelTemplate { coarsened, target ->
    val logits = coarsened.operands.firstOrNull()
    when {
        target != KernelTarget.NVIDIA_GB10 -> null
        coarsened.operands.size != 2 -> null
        logits == null || logits.type.rank != 2 -> null
        coarsened.numResults != 1 || coarsened.type.rank != 0 -> null
        else -> KernelDescriptor(
            kernelName = "kptx_cross_entropy",
            vendor = "tlaloc",
            targetArch = "gb10",
            typedFfi = true,
            scratchResults = listOf(listOf(logits.type.dims[0])),
        )
    }
}
