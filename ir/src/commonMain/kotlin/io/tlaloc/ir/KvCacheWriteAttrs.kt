package io.tlaloc.ir

import io.tlaloc.core.DType
import io.tlaloc.core.I32
import io.tlaloc.core.I64

/**
 * §0.4.466 — Phase H1b: the KV_CACHE_WRITE operand convention, shared by every
 * layer that touches the kind (the interpreter, the StableHLO emitter, the
 * renderer's refusal, the cost model). ONE parser, so no two layers can
 * disagree about what a legal cache write looks like — the [AllReduceAttrs]
 * (§0.4.460) / [PagedAttentionAttrs] (§0.4.465) precedent.
 *
 * Operands (see [OpKind.KV_CACHE_WRITE] for the full rationale):
 * ```
 *   0 cache       [numBlocks, blockSize, numKvHeads, headDim]   float
 *   1 newKv       [numTokens, numKvHeads, headDim]              float (same dtype)
 *   2 slotMapping [numTokens]                                   integer
 *   → out         [numBlocks, blockSize, numKvHeads, headDim]   (== cache's type)
 * ```
 *
 * **Attributes: NONE.** `blockSize`, `numKvHeads`, `headDim`, `numBlocks` and
 * `numTokens` are all derived from operand shapes, and are REFUSED as attrs BY
 * NAME: the house sentinel-dims rule forbids baking a dim-derived value where
 * an attr could then disagree with the operand it was derived from. This op
 * has nothing else to configure — the whole of its behaviour is in
 * `slotMapping`, which is a runtime tensor.
 *
 * **The flat-slot convention.** `slotMapping[i]` is a FLAT slot index into the
 * pool: `slot = blockIdx * blockSize + offset`, so the token lands at
 * `cache[slot / blockSize][slot % blockSize]`. This is vLLM's own convention,
 * and it is the one that makes the pool's row-major flattening
 * `[numBlocks * blockSize, numKvHeads, headDim]` the natural scatter operand —
 * one reshape, no index arithmetic on the block table.
 *
 * REJECTED alternative: a `[numTokens, 2]` (blockIdx, offset) pair tensor. It
 * reads more explicitly, but it forces every consumer to re-derive the flat
 * index, it makes the "is this slot live" test two comparisons instead of one,
 * and it diverges from what vLLM's Python scheduler already hands us — the
 * plugin (H3) would spend its first act flattening it back.
 *
 * **Padding slots.** A NEGATIVE `slotMapping` entry means "this token is
 * padding — do not write it", vLLM's `-1` convention for the slack lanes of a
 * bucketed batch. This is load-bearing for our static-shape story: decode
 * graphs are compiled per (batch, seq) bucket, so the tail of a short batch is
 * always padding, and a convention that *cannot* express "skip" would force a
 * recompile per real batch size.
 *
 * **Distinctness.** The non-negative slots must be DISTINCT. In serving they
 * are so by construction (the allocator hands each token its own slot), and
 * the requirement is what lets the emission be a plain scatter: StableHLO
 * leaves duplicate-index scatters with a non-commutative update computation
 * implementation-defined, and "replace" is about as non-commutative as it
 * gets. The interpreter CHECKS distinctness and refuses loudly rather than
 * silently picking a winner that the GPU would pick differently.
 */
object KvCacheWriteAttrs {

    /** The parsed, validated shape story. Every field is derived or checked. */
    data class Parsed(
        val numBlocks: Int,
        val blockSize: Int,
        val numKvHeads: Int,
        val headDim: Int,
        val numTokens: Int,
    ) {
        /** The pool seen as a flat slot array — what `slotMapping` indexes. */
        val numSlots: Int get() = numBlocks * blockSize

        /** Elements per slot: one token's worth of one pool. */
        val slotStride: Int get() = numKvHeads * headDim
    }

    /**
     * Parse and validate [op], refusing loudly with [layer] in the message.
     * Every refusal names the offending shape, so a hand-built graph fails
     * with the arity/rank story rather than an array-index crash inside a walk.
     */
    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.KV_CACHE_WRITE) {
            "$layer: KvCacheWriteAttrs.parse called on ${op.op} (a compiler bug)"
        }
        require(op.operands.size == 3) {
            "$layer: KV_CACHE_WRITE requires 3 operands (cache, newKv, slotMapping), " +
                "got ${op.operands.size}"
        }
        val c = op.operands[0].type
        val n = op.operands[1].type
        val s = op.operands[2].type

        require(c.rank == 4) {
            "$layer: KV_CACHE_WRITE cache must be rank-4 " +
                "[numBlocks, blockSize, numKvHeads, headDim], got ${c.dims}"
        }
        require(n.rank == 3) {
            "$layer: KV_CACHE_WRITE newKv must be rank-3 [numTokens, numKvHeads, headDim], got ${n.dims}"
        }
        require(s.rank == 1) {
            "$layer: KV_CACHE_WRITE slotMapping must be rank-1 [numTokens], got ${s.dims}"
        }
        require(isIntegral(s.dtype)) {
            "$layer: KV_CACHE_WRITE slotMapping must be an integer tensor, got ${s.dtype} " +
                "(a slot is an allocator index, never a differentiable value)"
        }
        require(n.dtype == c.dtype) {
            "$layer: KV_CACHE_WRITE newKv dtype ${n.dtype} must match the cache dtype ${c.dtype} " +
                "(a write never converts; H5's int8/fp8 KV-quant is a separate, named surface)"
        }

        val numBlocks = c.dims[0]
        val blockSize = c.dims[1]
        val numKvHeads = c.dims[2]
        val headDim = c.dims[3]
        val numTokens = n.dims[0]

        require(n.dims[1] == numKvHeads) {
            "$layer: KV_CACHE_WRITE newKv numKvHeads ${n.dims[1]} must match the cache's $numKvHeads"
        }
        require(n.dims[2] == headDim) {
            "$layer: KV_CACHE_WRITE newKv headDim ${n.dims[2]} must match the cache's $headDim"
        }
        require(s.dims[0] == numTokens) {
            "$layer: KV_CACHE_WRITE slotMapping has ${s.dims[0]} entries but newKv carries " +
                "$numTokens tokens — one slot per token, always"
        }
        require(numBlocks > 0 && blockSize > 0 && numKvHeads > 0 && headDim > 0) {
            "$layer: KV_CACHE_WRITE needs a positive pool shape, got ${c.dims}"
        }
        require(op.type.dims == c.dims && op.type.dtype == c.dtype) {
            "$layer: KV_CACHE_WRITE result type ${op.type} must equal the cache type $c — " +
                "the write is FUNCTIONAL: it returns an UPDATED POOL, it does not mutate in place"
        }

        for (forbidden in listOf(
            "block_size", "blockSize", "num_kv_heads", "numKvHeads",
            "head_dim", "headDim", "num_tokens", "numTokens", "num_blocks", "numBlocks",
        )) {
            require(forbidden !in op.attrs) {
                "$layer: KV_CACHE_WRITE attr '$forbidden' is refused by name — it is derived " +
                    "from the cache shape ${c.dims} / newKv shape ${n.dims}, and the house " +
                    "sentinel-dims rule forbids baking a dim-derived value into an attr that " +
                    "could then disagree with it. KV_CACHE_WRITE has NO attributes at all"
            }
        }

        return Parsed(
            numBlocks = numBlocks,
            blockSize = blockSize,
            numKvHeads = numKvHeads,
            headDim = headDim,
            numTokens = numTokens,
        )
    }

    private fun isIntegral(d: DType): Boolean = d == I32 || d == I64
}
