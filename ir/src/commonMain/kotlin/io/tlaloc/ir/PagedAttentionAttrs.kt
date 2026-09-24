package io.tlaloc.ir

import io.tlaloc.core.DType
import io.tlaloc.core.I32
import io.tlaloc.core.I64

/**
 * The PAGED_ATTENTION operand/attribute convention,
 * shared by every layer that touches the kind (the interpreter, the StableHLO
 * emitter, the renderer's refusal, the cost model). ONE parser, so the layers
 * cannot disagree about what a legal paged attention looks like (the same
 * pattern as [AllReduceAttrs]).
 *
 * Operands (see [OpKind.PAGED_ATTENTION] for the full rationale):
 * ```
 *   0 query      [numSeqs, numHeads, headDim]                    float
 *   1 keyCache   [numBlocks, blockSize, numKvHeads, headDim]     float
 *   2 valueCache [numBlocks, blockSize, numKvHeads, headDim]     float
 *   3 blockTables[numSeqs, maxBlocksPerSeq]                      integer
 *   4 seqLens    [numSeqs]                                       integer
 *   → out        [numSeqs, numHeads, headDim]                    float
 * ```
 *
 * Attributes: `scale: Number` — the softmax temperature, REQUIRED. It is the
 * only attribute, deliberately: every other quantity a paged-attention kernel
 * wants (`blockSize`, `numKvHeads`, the GQA `group`, `maxBlocksPerSeq`) is
 * DERIVED from operand shapes here. The sentinel-dims rule forbids
 * baking dim-derived values into attrs, and derivation additionally makes
 * attr-vs-operand disagreement unrepresentable.
 *
 * REJECTED alternative: `scale` as a sixth (scalar tensor) operand. It would
 * be a runtime value, which reads well — but the scale is a *compile-time
 * literal* of the model config (1/sqrt(headDim), or a config override), a
 * fused kernel needs it as a kernel constant, and a tensor operand would
 * force every claiming recognizer to prove constancy first.
 *
 * GQA: `numHeads` must be a multiple of `numKvHeads`; query head `h` reads kv
 * head `h / group`. Heads are grouped CONTIGUOUSLY per kv head — the
 * `[numSeqs, numKvHeads, group, headDim]` reshape convention. The interpreter
 * and the emitter both spell it that way, and the oracle test pins that they
 * agree with a dense attention built the same way.
 */
object PagedAttentionAttrs {

    /** The parsed, validated shape story. Every field is derived or checked. */
    data class Parsed(
        val numSeqs: Int,
        val numHeads: Int,
        val headDim: Int,
        val numBlocks: Int,
        val blockSize: Int,
        val numKvHeads: Int,
        val maxBlocksPerSeq: Int,
        val scale: Double,
    ) {
        /** GQA grouping: how many query heads share one kv head. */
        val group: Int get() = numHeads / numKvHeads

        /** The dense window width the gather-composed reference form materialises. */
        val maxContextLen: Int get() = maxBlocksPerSeq * blockSize
    }

    /**
     * Parse and validate [op], refusing loudly with [layer] in the message.
     * Every refusal names the offending shape so a hand-built graph fails with
     * the arity/rank story rather than an array-index crash deep in a walk.
     */
    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.PAGED_ATTENTION) {
            "$layer: PagedAttentionAttrs.parse called on ${op.op} (a compiler bug)"
        }
        require(op.operands.size == 5) {
            "$layer: PAGED_ATTENTION requires 5 operands (query, keyCache, valueCache, " +
                "blockTables, seqLens), got ${op.operands.size}"
        }
        val q = op.operands[0].type
        val k = op.operands[1].type
        val v = op.operands[2].type
        val t = op.operands[3].type
        val l = op.operands[4].type

        require(q.rank == 3) { "$layer: PAGED_ATTENTION query must be rank-3 [numSeqs, numHeads, headDim], got ${q.dims}" }
        require(k.rank == 4) {
            "$layer: PAGED_ATTENTION keyCache must be rank-4 " +
                "[numBlocks, blockSize, numKvHeads, headDim], got ${k.dims}"
        }
        require(v.dims == k.dims) {
            "$layer: PAGED_ATTENTION valueCache shape ${v.dims} must match keyCache ${k.dims} " +
                "(the two pools are indexed by the SAME block table)"
        }
        require(t.rank == 2) { "$layer: PAGED_ATTENTION blockTables must be rank-2 [numSeqs, maxBlocksPerSeq], got ${t.dims}" }
        require(l.rank == 1) { "$layer: PAGED_ATTENTION seqLens must be rank-1 [numSeqs], got ${l.dims}" }
        require(isIntegral(t.dtype)) { "$layer: PAGED_ATTENTION blockTables must be an integer tensor, got ${t.dtype}" }
        require(isIntegral(l.dtype)) { "$layer: PAGED_ATTENTION seqLens must be an integer tensor, got ${l.dtype}" }

        val numSeqs = q.dims[0]
        val numHeads = q.dims[1]
        val headDim = q.dims[2]
        val numBlocks = k.dims[0]
        val blockSize = k.dims[1]
        val numKvHeads = k.dims[2]
        val maxBlocksPerSeq = t.dims[1]

        require(k.dims[3] == headDim) {
            "$layer: PAGED_ATTENTION keyCache headDim ${k.dims[3]} must match query headDim $headDim"
        }
        require(t.dims[0] == numSeqs) {
            "$layer: PAGED_ATTENTION blockTables has ${t.dims[0]} rows but query has $numSeqs sequences"
        }
        require(l.dims[0] == numSeqs) {
            "$layer: PAGED_ATTENTION seqLens has ${l.dims[0]} entries but query has $numSeqs sequences"
        }
        require(numKvHeads > 0 && numHeads % numKvHeads == 0) {
            "$layer: PAGED_ATTENTION numHeads $numHeads must be a positive multiple of " +
                "numKvHeads $numKvHeads (GQA grouping is derived, never an attr)"
        }
        require(blockSize > 0 && numBlocks > 0 && maxBlocksPerSeq > 0) {
            "$layer: PAGED_ATTENTION needs positive numBlocks/blockSize/maxBlocksPerSeq, " +
                "got $numBlocks/$blockSize/$maxBlocksPerSeq"
        }
        require(op.type.dims == q.dims) {
            "$layer: PAGED_ATTENTION result shape ${op.type.dims} must match the query shape ${q.dims}"
        }

        val scale = (op.attrs["scale"] as? Number)?.toDouble()
            ?: error(
                "$layer: PAGED_ATTENTION requires a 'scale' attr (the softmax temperature — " +
                    "typically 1/sqrt(headDim)); got ${op.attrs["scale"]}",
            )
        require(scale.isFinite()) { "$layer: PAGED_ATTENTION scale must be finite, got $scale" }
        for (forbidden in listOf("block_size", "blockSize", "num_kv_heads", "numKvHeads")) {
            require(forbidden !in op.attrs) {
                "$layer: PAGED_ATTENTION attr '$forbidden' is refused by name — it is derived " +
                    "from keyCache's shape ${k.dims}, and the house sentinel-dims rule forbids " +
                    "baking a dim-derived value into an attr that could then disagree with it"
            }
        }

        return Parsed(
            numSeqs = numSeqs,
            numHeads = numHeads,
            headDim = headDim,
            numBlocks = numBlocks,
            blockSize = blockSize,
            numKvHeads = numKvHeads,
            maxBlocksPerSeq = maxBlocksPerSeq,
            scale = scale,
        )
    }

    private fun isIntegral(d: DType): Boolean = d == I32 || d == I64
}
