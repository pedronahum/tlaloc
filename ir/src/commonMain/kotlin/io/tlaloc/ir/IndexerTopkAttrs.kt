package io.tlaloc.ir

import io.tlaloc.core.I32
import io.tlaloc.core.I64

/**
 * [OpKind.INDEXER_TOPK]'s operands and attributes, one parser for every layer
 * that touches the kind, and the selection both interpreters run.
 * ```
 *   0 query       [R, H, D]                        float: the indexer's queries
 *   1 chunkKeys   [numBlocks, blockSize, 1, D]     float: a chunk's key at the slot of its last position
 *   2 blockTables [numTables, maxBlocksPerSeq]     integer
 *   3 seqLens     [R]                              integer
 *   -> keep       [R, maxBlocksPerSeq * blockSize / chunk]   I32
 * attrs: chunk: Int (divides blockSize), top_k: Int (>= 1), scale: Number
 * ```
 * The positions of a row are cut into chunks of `chunk`; with `L = seqLens[r]`
 * the first `L / chunk` are complete. A complete chunk `c` scores
 * `scale * sum over heads h of relu(query[r, h] · chunkKeys[position c * chunk + chunk - 1])`,
 * and the `top_k` highest are kept (all of them when there are no more;
 * equal scores keep the lower chunk). The incomplete chunk at the end, when
 * `L % chunk != 0`, is kept too. `keep` is 1 for a kept chunk and 0 for the
 * others, and is what [OpKind.PAGED_ATTENTION]'s keep operand reads. That is
 * transformers' `Qwen4ExpTextQSAIndexer` (query-selective sparse attention):
 * the chunk keys are its pooled block keys, the mean of the chunk's keys
 * normalized and rotated at the chunk's first position, which the graph
 * computes and writes where the chunk ends.
 *
 * Rows share block tables as [PagedAttentionAttrs] rows do: row `r` reads
 * table `r / (R / numTables)`.
 */
object IndexerTopkAttrs {

    data class Parsed(
        val rows: Int,
        val heads: Int,
        val dim: Int,
        val numBlocks: Int,
        val blockSize: Int,
        val numTables: Int,
        val maxBlocksPerSeq: Int,
        val chunk: Int,
        val topK: Int,
        val scale: Double,
    ) {
        val rowsPerTable: Int get() = rows / numTables
        val chunks: Int get() = maxBlocksPerSeq * blockSize / chunk
    }

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.INDEXER_TOPK) { "$layer: IndexerTopkAttrs.parse called on ${op.op}" }
        require(op.operands.size == 4) {
            "$layer: INDEXER_TOPK takes 4 operands (query, chunkKeys, blockTables, seqLens), got ${op.operands.size}"
        }
        val (q, k, t, l) = op.operands.map { it.type }
        require(q.rank == 3) { "$layer: INDEXER_TOPK query must be [R, H, D], got ${q.dims}" }
        val (rows, heads, dim) = q.dims
        require(k.rank == 4 && k.dims[2] == 1 && k.dims[3] == dim) {
            "$layer: INDEXER_TOPK chunkKeys must be [numBlocks, blockSize, 1, $dim], got ${k.dims}"
        }
        require(t.rank == 2 && (t.dtype == I32 || t.dtype == I64)) { "$layer: INDEXER_TOPK blockTables must be an integer [numTables, maxBlocksPerSeq], got ${t.dtype}${t.dims}" }
        require(l.rank == 1 && l.dims[0] == rows && (l.dtype == I32 || l.dtype == I64)) {
            "$layer: INDEXER_TOPK seqLens must be an integer [$rows], got ${l.dtype}${l.dims}"
        }
        require(t.dims[0] >= 1 && rows % t.dims[0] == 0) {
            "$layer: INDEXER_TOPK has ${t.dims[0]} block tables for $rows rows; one per row or per equal group of rows"
        }
        val chunk = (op.attrs["chunk"] as? Number)?.toInt() ?: error("$layer: INDEXER_TOPK needs a 'chunk' attr")
        val topK = (op.attrs["top_k"] as? Number)?.toInt() ?: error("$layer: INDEXER_TOPK needs a 'top_k' attr")
        val scale = (op.attrs["scale"] as? Number)?.toDouble() ?: error("$layer: INDEXER_TOPK needs a 'scale' attr")
        val blockSize = k.dims[1]
        require(chunk >= 1 && blockSize % chunk == 0) { "$layer: INDEXER_TOPK chunk $chunk must divide the block size $blockSize" }
        require(topK >= 1 && scale.isFinite()) { "$layer: INDEXER_TOPK top_k must be >= 1 and scale finite, got $topK and $scale" }
        val p = Parsed(rows, heads, dim, k.dims[0], blockSize, t.dims[0], t.dims[1], chunk, topK, scale)
        require(op.type.dtype == I32 && op.type.dims == listOf(rows, p.chunks)) {
            "$layer: INDEXER_TOPK result must be I32 [$rows, ${p.chunks}], got ${op.type.dtype}${op.type.dims}"
        }
        return p
    }

    /** The keep mask (`[R, chunks]` row-major, 0 or 1), with the scores summed in doubles. */
    fun reference(p: Parsed, query: FloatArray, chunkKeys: FloatArray, blockTables: IntArray, seqLens: IntArray): IntArray {
        val keep = IntArray(p.rows * p.chunks)
        val scores = DoubleArray(p.chunks)
        for (r in 0 until p.rows) {
            val len = seqLens[r]
            require(len in 0..p.maxBlocksPerSeq * p.blockSize) {
                "INDEXER_TOPK: seqLens[$r] = $len is outside [0, ${p.maxBlocksPerSeq * p.blockSize}]"
            }
            val table = r / p.rowsPerTable
            val complete = len / p.chunk
            for (c in 0 until complete) {
                val pos = c * p.chunk + p.chunk - 1
                val page = blockTables[table * p.maxBlocksPerSeq + pos / p.blockSize]
                require(page in 0 until p.numBlocks) { "INDEXER_TOPK: blockTables[$table] names page $page of ${p.numBlocks}" }
                val kOff = (page * p.blockSize + pos % p.blockSize) * p.dim
                var s = 0.0
                for (h in 0 until p.heads) {
                    val qOff = (r * p.heads + h) * p.dim
                    var dot = 0.0
                    for (j in 0 until p.dim) dot += query[qOff + j].toDouble() * chunkKeys[kOff + j]
                    if (dot > 0) s += dot
                }
                scores[c] = s * p.scale
            }
            val order = (0 until complete).sortedWith(compareByDescending<Int> { scores[it] }.thenBy { it })
            for (c in order.take(p.topK)) keep[r * p.chunks + c] = 1
            if (len % p.chunk != 0) keep[r * p.chunks + complete] = 1
        }
        return keep
    }
}
