/**
 * `Embedding` + `EmbeddingBag`, in DiffKT's exact
 * semantics: a trainable
 * `[numEmbeddings, embeddingSize]` table, gaussian-initialized by default;
 * `Embedding` gathers `(*) → (*, D)`, `EmbeddingBag` is embedding followed by
 * a per-bag reduction over axis 0 (`slice → reduce → concat`, DiffKT's own
 * spelling), bags addressed by linear offsets into the flattened indices with
 * the last bag running to the end.
 *
 * The AD route is the compiler route (see `Training.kt`): the forward is TRACE spellings only
 * (EMBEDDING, SLICE, SUM(axes), RESHAPE, CONCAT), the index input rides as an
 * I32-typed leaf/param, and `DxirReverseTransform` owns every gradient — the
 * table's arrives through `EmbeddingRule`'s fused EMBEDDING_GRAD dense scatter
 * (a dense gradient), the indices' as the ZEROS_LIKE
 * structural zero. Zero gradient math in this file.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.split
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.concat
import io.tlaloc.autograd.embedding
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.sum

/**
 * DiffKT's `Embedding(numEmbeddings, embeddingSize, random)`: a trainable
 * `[V, D]` table; `forward` gathers `table[indices, :]` — rank-1 `[N]` indices
 * produce `[N, D]`, rank-2 `[B, N]` produce `[B, N, D]` (DiffKT's
 * `(*) → (*, D)` contract at the ranks the trace spelling carries). The input
 * MUST be an I32 tensor — trace-time checked, mirroring DiffKT's own
 * `IllegalArgumentException` on a non-`IntTensor` input.
 *
 * [paddingIndex] is a Tlaloc EXTRA (DiffKT's Embedding has none;
 * the `:core` `embedding` op does): positions whose index equals it
 * produce exact-zero rows forward AND contribute exactly zero gradient to the
 * table (the attr rides the primal onto the fused EMBEDDING_GRAD). Negative =
 * none, the -1 sentinel.
 */
class Embedding(
    val table: DTensor<*, F32>,
    val paddingIndex: Int = -1,
) : TrainableLayer<Embedding> {

    init {
        require(table.dims.size == 2) {
            "Embedding: table must be rank-2 [numEmbeddings, embeddingSize] (got dims ${table.dims.toList()})"
        }
    }

    override val parameters: List<NamedParameter> = listOf(NamedParameter("table", table))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): Embedding {
        val unknown = updated.keys - setOf("table")
        require(unknown.isEmpty()) { "Embedding.withParameters: unknown keys $unknown (known: [table])" }
        return Embedding(updated["table"] ?: table, paddingIndex)
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.dtype == I32) {
            "Embedding must be called with an I32 index tensor (got ${x.dtype.name}) — " +
                "pass the indices as a DTensor<*, I32> capture input"
        }
        return params["table"].embedding(x, paddingIndex)
    }

    companion object {
        /** The DiffKT constructor surface: table drawn `gaussian()` (mean 0, variance 1) from [key]. */
        operator fun invoke(
            numEmbeddings: Int,
            embeddingSize: Int,
            key: RandomKey,
            paddingIndex: Int = -1,
        ): Embedding {
            require(numEmbeddings > 0 && embeddingSize > 0) {
                "Embedding: numEmbeddings=$numEmbeddings and embeddingSize=$embeddingSize must be positive"
            }
            val table = gaussianInit(key.split(1)[0], intArrayOf(numEmbeddings, embeddingSize))
            return Embedding(table, paddingIndex)
        }
    }
}

/**
 * DiffKT's `EmbeddingBag(numEmbeddings, embeddingSize, reduction, random)`:
 * embedding followed by a reduction over each BAG of embeddings along axis 0.
 * Bags are linear offsets into the flattened indices ([forwardBags]'s
 * `bagOffsets`, each the START of a bag, the last bag running to the end of
 * the indices) — result `[numBags, embeddingSize]`.
 *
 * The traced spelling is DiffKT's own, op for op: flatten indices → embed →
 * per bag `slice(start, end)` on axis 0 → [Reduction.reduce] → `concat` —
 * SLICE/SUM/RESHAPE/CONCAT through the registry rules, so the bag structure is
 * differentiated by the transform like everything else.
 *
 * DiffKT ships ONLY `Reduction.Sum` (`sum(0, keepDims = true)`); Mean/Max are
 * unimplemented upstream and are not offered here either — only Sum.
 */
class EmbeddingBag(
    val table: DTensor<*, F32>,
    val reduction: Reduction = Reduction.Sum,
) : TrainableLayer<EmbeddingBag> {

    init {
        require(table.dims.size == 2) {
            "EmbeddingBag: table must be rank-2 [numEmbeddings, embeddingSize] (got dims ${table.dims.toList()})"
        }
    }

    /**
     * The per-bag reduction: `[bagSize, D] → [1, D]`. DiffKT's sealed
     * companion class, with the one member DiffKT actually implements.
     */
    sealed class Reduction {
        abstract fun reduce(bag: Tracer<Shape>): Tracer<Shape>

        /** DiffKT's `x.sum(0, keepDims = true)` — spelled SUM(axes=[0]) + the keepdims RESHAPE. */
        object Sum : Reduction() {
            override fun reduce(bag: Tracer<Shape>): Tracer<Shape> =
                bag.sum<Shape>(intArrayOf(0)).reshape(intArrayOf(1, bag.dims[1]))
        }
    }

    override val parameters: List<NamedParameter> = listOf(NamedParameter("table", table))

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): EmbeddingBag {
        val unknown = updated.keys - setOf("table")
        require(unknown.isEmpty()) { "EmbeddingBag.withParameters: unknown keys $unknown (known: [table])" }
        return EmbeddingBag(updated["table"] ?: table, reduction)
    }

    /**
     * As in DiffKT, the single-input `Layer` form refuses — a bag call needs
     * `(indices, bagOffsets)`, exactly like DiffKT's vararg `invoke` throws
     * for its typed two-arg overload. Use [forwardBags] or [withOffsets].
     */
    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> =
        throw IllegalArgumentException(
            "EmbeddingBag must be called with indices AND bagOffsets — use forwardBags(indices, " +
                "bagOffsets, params) or withOffsets(bagOffsets) for the Layer view"
        )

    /** DiffKT's `invoke(indices: IntTensor, bagOffsets: IntTensor)`, traced. */
    fun forwardBags(indices: Tracer<Shape>, bagOffsets: IntArray, params: Params): Tracer<Shape> {
        require(indices.dtype == I32) {
            "EmbeddingBag must be called with an I32 index tensor (got ${indices.dtype.name})"
        }
        require(indices.rank == 1 || indices.rank == 2) {
            "EmbeddingBag: indices must be rank-1 [N] or rank-2 [B, N] (flattened, like DiffKT); " +
                "got ${indices.dims.toList()}"
        }
        val flat: Tracer<Shape> =
            if (indices.rank == 1) indices else indices.reshape(intArrayOf(indices.size))
        val n = flat.dims[0]
        require(bagOffsets.isNotEmpty()) { "EmbeddingBag: at least one bag offset required" }
        for (i in bagOffsets.indices) {
            val start = bagOffsets[i]
            val end = if (i + 1 == bagOffsets.size) n else bagOffsets[i + 1]
            require(start in 0 until n && end in (start + 1)..n) {
                "EmbeddingBag: bag $i spans [$start, $end) — offsets must be strictly increasing " +
                    "within [0, $n) (empty bags are a recorded narrowing; DiffKT's slice would " +
                    "admit them, our SLICE spelling refuses zero-extent slices)"
            }
        }
        val embedded: Tracer<Shape> = params["table"].embedding(flat)
        val bags = bagOffsets.indices.map { i ->
            val start = bagOffsets[i]
            val end = if (i + 1 == bagOffsets.size) n else bagOffsets[i + 1]
            reduction.reduce(embedded.slice(start, end, 0))
        }
        return concat(bags, 0)
    }

    /**
     * A single-input `Layer` view for one FIXED bag structure — what a
     * `Sequential` or [valueAndGradients] can hold. Consistent with the
     * caching contract of [CapturedStep]: the offsets shape the captured graph (they are SLICE
     * attrs), so a different bag structure is a different trace anyway; this
     * view just names that fact as a value.
     */
    fun withOffsets(bagOffsets: IntArray): WithOffsets = WithOffsets(this, bagOffsets.copyOf())

    /** The [withOffsets] adapter: parameters and keys pass through unchanged. */
    class WithOffsets internal constructor(
        val bag: EmbeddingBag,
        private val bagOffsets: IntArray,
    ) : TrainableLayer<WithOffsets> {
        override val parameters: List<NamedParameter> get() = bag.parameters
        override fun withParameters(updated: Map<String, DTensor<*, F32>>): WithOffsets =
            WithOffsets(bag.withParameters(updated), bagOffsets)
        override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> =
            bag.forwardBags(x, bagOffsets, params)
    }

    companion object {
        /** The DiffKT constructor surface (its own argument order): table drawn `gaussian()` from [key]. */
        operator fun invoke(
            numEmbeddings: Int,
            embeddingSize: Int,
            reduction: Reduction,
            key: RandomKey,
        ): EmbeddingBag {
            require(numEmbeddings > 0 && embeddingSize > 0) {
                "EmbeddingBag: numEmbeddings=$numEmbeddings and embeddingSize=$embeddingSize must be positive"
            }
            val table = gaussianInit(key.split(1)[0], intArrayOf(numEmbeddings, embeddingSize))
            return EmbeddingBag(table, reduction)
        }
    }
}
