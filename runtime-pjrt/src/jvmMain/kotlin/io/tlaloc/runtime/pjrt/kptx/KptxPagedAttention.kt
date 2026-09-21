package io.tlaloc.runtime.pjrt.kptx

import io.tlaloc.kptx.KptxKernels
import io.tlaloc.kptx.emitPtx
import java.nio.file.Path

/**
 * §0.4.471 — Phase H4: the **runtime half** of paged-attention claiming.
 *
 * `PagedAttentionKernel` (in `:ir`) decides *that* a `PAGED_ATTENTION` op
 * on a GB10 becomes `stablehlo.custom_call @kptx_paged_attention`; this
 * object is what makes that name resolvable — it registers
 * `KptxKernels.pagedAttentionModule`'s three-stage launch chain with
 * [KptxKernelRegistry] under that target name.
 *
 * The two halves are deliberately in different modules: `:ir` must stay
 * Kotlin-Multiplatform and free of the FFM/CUDA world, and a graph that
 * merely *names* the kernel is a perfectly good artefact to ship to a
 * machine that resolves it differently. The pairing is the plugin's job,
 * which is exactly the H3 line ("the half vLLM calls, the half that does
 * the work").
 *
 * # Shape derivation, at dispatch
 *
 * Nothing is baked. Per call the handler reads the decoded frame's
 * buffer shapes:
 *
 * ```
 *   args = (query[S,H,D], keyCache[NB,BS,KH,D], valueCache[…],
 *           blockTables[S,MB] i32, seqLens[S] i32)
 *   rets = (out[S,H,D], S_scratch[S·H, MB·BS])        ← scratch is XLA-owned
 * ```
 *
 * and passes `(n_h, n_d, n_bs, n_kv, n_mb)` as trailing i32s. The grid is
 * `S·H` CTAs for stages 1 and 3 — one per (sequence, query head) — and
 * the same for the row softmax, whose rows are exactly those pairs.
 *
 * [scale] is the one value that *is* baked, into the PTX: see
 * `KptxKernels.pagedAttentionModule`. One registration therefore serves
 * one scale, which for a served model is one number for the whole run.
 * **Named deferral**: a scale-generic kernel, once `Stage` grows access
 * to the frame's decoded FFI attrs (the emitted call already carries
 * `scale` in its typed-FFI `backend_config`).
 */
object KptxPagedAttention {

    /** The custom-call target name `PagedAttentionKernel` emits. */
    const val TARGET: String = "kptx_paged_attention"

    /**
     * Register the paged-attention launch chain for [scale] on the PJRT
     * plugin at [pluginPath], under custom-call target [name].
     *
     * Process-permanent and single-shot per (plugin, name), like every
     * KPTX registration — a second call for a different scale needs a
     * different [name] until the deferral above closes.
     */
    fun register(
        pluginPath: Path,
        scale: Float,
        name: String = TARGET,
        blockThreads: Int = 256,
    ) {
        val ptx = KptxKernels.pagedAttentionModule(block = blockThreads, scale = scale).emitPtx()
        val block = KptxKernelRegistry.Dim3(blockThreads)
        // rows = numSeqs * numHeads — one CTA per (sequence, query head).
        val rows = { args: List<io.tlaloc.runtime.pjrt.ffm.XlaFfi.Buffer> ->
            KptxKernelRegistry.Dim3((args[0].dims[0] * args[0].dims[1]).toInt())
        }
        // (n_h, n_d, n_bs, n_kv, n_mb), every one read from a buffer shape.
        val shapeParams = { args: List<io.tlaloc.runtime.pjrt.ffm.XlaFfi.Buffer> ->
            intArrayOf(
                args[0].dims[1].toInt(), // numHeads
                args[0].dims[2].toInt(), // headDim
                args[1].dims[1].toInt(), // blockSize
                args[1].dims[2].toInt(), // numKvHeads
                args[3].dims[1].toInt(), // maxBlocksPerSeq
            )
        }
        KptxKernelRegistry.registerKernelChain(
            pluginPath, name, ptx,
            listOf(
                KptxKernelRegistry.Stage(
                    entryName = "kptx_paged_scores",
                    grid = { args, _ -> rows(args) },
                    block = block,
                    // q, keyCache, blockTables, seqLens, S
                    paramBuffers = { args, rets -> listOf(args[0], args[1], args[3], args[4], rets[1]) },
                    trailingI32Params = { args, _ -> shapeParams(args) },
                ),
                KptxKernelRegistry.Stage(
                    entryName = "kptx_paged_softmax",
                    grid = { args, _ -> rows(args) },
                    block = block,
                    paramBuffers = { _, rets -> listOf(rets[1]) },
                    // n_ctx = maxBlocksPerSeq * blockSize, the scratch row width.
                    trailingI32Params = { args, _ ->
                        intArrayOf((args[3].dims[1] * args[1].dims[1]).toInt())
                    },
                ),
                KptxKernelRegistry.Stage(
                    entryName = "kptx_paged_out",
                    grid = { args, _ -> rows(args) },
                    block = block,
                    // S, valueCache, blockTables, seqLens, out
                    paramBuffers = { args, rets -> listOf(rets[1], args[2], args[3], args[4], rets[0]) },
                    trailingI32Params = { args, _ -> shapeParams(args) },
                ),
            ),
        )
    }
}
