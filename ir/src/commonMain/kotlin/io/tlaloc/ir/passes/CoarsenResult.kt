package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirFunction

/**
 * Return type of [PhiCalculus.coarsenLeaf]. Either a `Success` carrying the
 * simplified mini-function produced by [PhiCalculus.apply] on the leaf's synthesised
 * sub-function (along with its post-simplification op count, used as the size source
 * by [SoiIdentification.identifyWithSizeLimit] when an engine is available), or a
 * `Failure` with a human-readable reason (the caller falls back to raw op count).
 *
 * The result is used as a size source only; the gradient body for the
 * `OpKind.COARSENED` splice is produced elsewhere.
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
     *  - Leaf contains a region-bearing op (only scalar leaves are handled).
     *  - Leaf is empty (nothing to coarsen).
     *  - `PhiCalculus.apply` threw (Symja-side failure, unsupported op kind, etc.).
     */
    data class Failure(val reason: String) : CoarsenResult()
}
