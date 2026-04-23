package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction

/**
 * §0.4.30 — return from [PhiCalculus.coarsenLeaf]. Either a `Success` carrying the
 * simplified mini-function produced by [PhiCalculus.apply] on the leaf's synthesised
 * sub-function (along with its post-simplification op count, used as the size source
 * by [SoiIdentification.identifyWithSizeLimit] when an engine is available), or a
 * `Failure` with a human-readable reason (the caller falls back to raw op count).
 *
 * C.3b will grow the `Success` variant with additional fields — a pre-computed
 * gradient body + `readsPrimalOperandIndices` set — for the `OpKind.COARSENED`
 * splice. For C.3a this is size-source-only.
 */
sealed class CoarsenResult {
    /**
     * Coarsening succeeded. [simplified] is the post-[PhiCalculus.apply] mini-function.
     * [size] is the op count of `simplified.body` — the authoritative replacement for
     * `RegionTreeNode.subtreeSize()` when the engine is available.
     */
    data class Success(
        val simplified: DxirFunction,
        val size: Int,
    ) : CoarsenResult()

    /**
     * Coarsening did not run or aborted. The caller (typically [SoiIdentification])
     * must fall back to the raw op count. Common reasons:
     *  - Leaf contains a region-bearing op (C.3a deliberately handles scalar leaves only).
     *  - Leaf is empty (nothing to coarsen).
     *  - `PhiCalculus.apply` threw (Symja-side failure, unsupported op kind, etc.).
     */
    data class Failure(val reason: String) : CoarsenResult()
}
