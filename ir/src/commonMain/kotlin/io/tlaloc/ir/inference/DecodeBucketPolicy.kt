package io.tlaloc.ir.inference

/**
 * §0.4.467 — Phase H1c: the BUCKETING POLICY.
 *
 * A compiled decode step has static shapes. A serving loop does not have a
 * static batch size or a static context length — vLLM's scheduler hands the
 * executor whatever mix of sequences fits the token budget this step. The
 * bridge between the two is **bucketing**: round the request up to one of a
 * small, fixed ladder of shapes, compile one executable per ladder point, and
 * pad the request out to it. vLLM's own TPU backend does exactly this, and
 * bounded dynamism is a roadmap item for them too — so our static-shape story
 * is not a limitation we are working around, it is the same bet they made.
 *
 * ## The ladder
 *
 * Two independent axes, multiplied:
 *
 * - **Batch**: powers of two `1, 2, 4, 8, …` up to [maxBatch], with [maxBatch]
 *   itself appended when it is not already a power of two. The cap is always a
 *   ladder point, because the scheduler's `max_num_seqs` is the shape the
 *   steady state actually runs at and a request for it must not be refused.
 * - **Context**: powers of two from [minContext] up to [maxContext], each
 *   **aligned UP to a multiple of [blockSize]**, with the aligned [maxContext]
 *   appended. Alignment is not cosmetic: a context bucket names a
 *   `maxBlocksPerSeq = context / blockSize` block-table width, and a bucket
 *   that is not a whole number of pages would make that width a rounding
 *   decision taken twice, in two places, by two layers. Aligning once here
 *   makes `context / blockSize` exact everywhere downstream.
 *
 * REJECTED: a single fused (batch × context) ladder of "total token" buckets,
 * the shape a continuous-batching *prefill* scheduler thinks in. Decode is not
 * token-budgeted — one token per sequence, and the cost is dominated by the
 * KV window each sequence reads, so batch and context genuinely are two axes.
 * Folding them would either over-compile (every (b, c) pair a separate point
 * with no structure) or under-serve (a 1×4096 request and a 64×64 request are
 * not interchangeable even though both are 4096 "tokens").
 *
 * REJECTED: a geometric ladder with a ratio other than 2 (1.25×, say, as some
 * autotuners use). It halves the average padding waste and multiplies the
 * executable count by ~3, and every extra executable is a compile the serving
 * process pays for at startup and a slot in device memory. Powers of two cap
 * the worst-case waste at 2× and the ladder length at `log2(max)`; when a
 * deployment wants a tighter ladder, [DecodeBucketPolicy] takes the ladders it
 * is given — see the secondary constructor.
 *
 * REJECTED: rounding DOWN and running the request in two passes. It turns one
 * dispatch into a loop with a partial-batch tail, and the tail is exactly the
 * shape the ladder was supposed to avoid compiling.
 *
 * ## Refusals
 *
 * A request above [maxBatch] or [maxContext] is REFUSED BY NAME. It is not
 * clamped: clamping a context request silently truncates a sequence's history,
 * which is a wrong answer dressed as a slow one. The scheduler is the layer
 * that knows how to split an over-long request, and it can only do that if it
 * is told.
 */
class DecodeBucketPolicy private constructor(
    val maxBatch: Int,
    val maxContext: Int,
    val blockSize: Int,
    val minContext: Int,
    /** Ascending, distinct batch-size ladder points; last == [maxBatch]. */
    val batchLadder: List<Int>,
    /** Ascending, distinct context ladder points, each a multiple of
     *  [blockSize]; last == [maxContext] aligned up. */
    val contextLadder: List<Int>,
) {

    constructor(
        maxBatch: Int,
        maxContext: Int,
        blockSize: Int,
        minContext: Int = DEFAULT_MIN_CONTEXT,
    ) : this(
        maxBatch = maxBatch,
        maxContext = alignUp(requirePositive(maxContext, "maxContext"), requirePositive(blockSize, "blockSize")),
        blockSize = blockSize,
        minContext = minContext,
        batchLadder = powerOfTwoLadder(requirePositive(maxBatch, "maxBatch")),
        contextLadder = alignedPowerOfTwoLadder(
            from = requirePositive(minContext, "minContext"),
            to = alignUp(maxContext, blockSize),
            blockSize = blockSize,
        ),
    )

    init {
        require(batchLadder.isNotEmpty() && contextLadder.isNotEmpty()) {
            "DecodeBucketPolicy: both ladders must be non-empty (batch=$batchLadder, context=$contextLadder)"
        }
        require(batchLadder == batchLadder.sorted() && batchLadder.distinct() == batchLadder) {
            "DecodeBucketPolicy: batchLadder must be strictly ascending, got $batchLadder"
        }
        require(contextLadder == contextLadder.sorted() && contextLadder.distinct() == contextLadder) {
            "DecodeBucketPolicy: contextLadder must be strictly ascending, got $contextLadder"
        }
        require(contextLadder.all { it % blockSize == 0 }) {
            "DecodeBucketPolicy: every context bucket must be a whole number of pages of " +
                "blockSize $blockSize, got $contextLadder — a bucket that is not would make " +
                "maxBlocksPerSeq a rounding decision taken in two places"
        }
        require(batchLadder.last() == maxBatch) {
            "DecodeBucketPolicy: the batch ladder must end AT maxBatch $maxBatch (got " +
                "${batchLadder.last()}) — the scheduler's steady-state shape is never unreachable"
        }
        require(contextLadder.last() == maxContext) {
            "DecodeBucketPolicy: the context ladder must end AT maxContext $maxContext (got " +
                "${contextLadder.last()})"
        }
    }

    /** Every (batch, context) ladder point — one compiled executable each. */
    val allBuckets: List<DecodeBucket>
        get() = batchLadder.flatMap { b -> contextLadder.map { c -> DecodeBucket(b, c) } }

    /** How many executables a fully warmed serving process holds, per graph
     *  kind. The number a deployment actually has to look at before it starts
     *  tuning the ladder. */
    val executableCount: Int get() = batchLadder.size * contextLadder.size

    /**
     * The smallest ladder point that covers a step of [batch] sequences whose
     * longest context (INCLUDING the token about to be written — see
     * [DecodeGraphSpec]) is [context] positions.
     *
     * Refuses by name above the caps, and below 1: a zero-sequence step is not
     * a degenerate bucket, it is a scheduler that should not have dispatched.
     */
    fun bucketFor(batch: Int, context: Int): DecodeBucket {
        require(batch >= 1) {
            "DecodeBucketPolicy.bucketFor: batch $batch must be >= 1 — an empty step is a " +
                "scheduling bug, not a bucket"
        }
        require(context >= 1) {
            "DecodeBucketPolicy.bucketFor: context $context must be >= 1 — a sequence with no " +
                "live position has nothing to attend to, and seqLen 0 is the one shape the " +
                "PAGED_ATTENTION interpreter (zeros) and emission (NaN) disagree about"
        }
        require(batch <= maxBatch) {
            "DecodeBucketPolicy.bucketFor: batch $batch exceeds maxBatch $maxBatch — REFUSED, " +
                "not clamped: the scheduler owns splitting an over-large step, and can only do " +
                "it if it is told (batch ladder $batchLadder)"
        }
        require(context <= maxContext) {
            "DecodeBucketPolicy.bucketFor: context $context exceeds maxContext $maxContext — " +
                "REFUSED, not clamped: clamping would silently truncate a sequence's history, " +
                "which is a wrong answer dressed as a slow one (context ladder $contextLadder)"
        }
        return DecodeBucket(
            batch = batchLadder.first { it >= batch },
            maxContext = contextLadder.first { it >= context },
        )
    }

    /** Non-throwing sibling for a scheduler that wants to ask before it
     *  commits: null exactly where [bucketFor] would refuse. */
    fun bucketForOrNull(batch: Int, context: Int): DecodeBucket? =
        if (batch < 1 || context < 1 || batch > maxBatch || context > maxContext) {
            null
        } else {
            bucketFor(batch, context)
        }

    override fun toString(): String =
        "DecodeBucketPolicy(batch=$batchLadder, context=$contextLadder, blockSize=$blockSize, " +
            "executables=$executableCount)"

    companion object {
        /**
         * The smallest context bucket. 16 is vLLM's own floor and it is a
         * reasonable one: below it the executable count grows for buckets that
         * differ by a handful of KV rows, which no kernel notices.
         */
        const val DEFAULT_MIN_CONTEXT: Int = 16

        /**
         * A policy built from EXPLICIT ladders, for a deployment that has
         * profiled its traffic and wants something other than powers of two.
         * The invariants (ascending, distinct, page-aligned, ends at the cap)
         * are checked exactly as they are for the generated ladders — a
         * hand-written ladder gets no discount.
         */
        fun withLadders(
            batchLadder: List<Int>,
            contextLadder: List<Int>,
            blockSize: Int,
        ): DecodeBucketPolicy {
            requirePositive(blockSize, "blockSize")
            require(batchLadder.isNotEmpty() && contextLadder.isNotEmpty()) {
                "DecodeBucketPolicy.withLadders: ladders must be non-empty"
            }
            require(batchLadder.all { it >= 1 } && contextLadder.all { it >= 1 }) {
                "DecodeBucketPolicy.withLadders: every ladder point must be >= 1, got " +
                    "$batchLadder / $contextLadder"
            }
            return DecodeBucketPolicy(
                maxBatch = batchLadder.max(),
                maxContext = contextLadder.max(),
                blockSize = blockSize,
                minContext = contextLadder.min(),
                batchLadder = batchLadder,
                contextLadder = contextLadder,
            )
        }

        private fun requirePositive(v: Int, what: String): Int {
            require(v >= 1) { "DecodeBucketPolicy: $what must be >= 1, got $v" }
            return v
        }

        /** `x` rounded up to a multiple of `m`. */
        internal fun alignUp(x: Int, m: Int): Int = ((x + m - 1) / m) * m

        private fun powerOfTwoLadder(max: Int): List<Int> {
            val out = ArrayList<Int>()
            var v = 1
            while (v <= max) { out.add(v); v *= 2 }
            if (out.last() != max) out.add(max)
            return out
        }

        private fun alignedPowerOfTwoLadder(from: Int, to: Int, blockSize: Int): List<Int> {
            val out = LinkedHashSet<Int>()
            var v = alignUp(maxOf(from, blockSize), blockSize)
            while (v < to) { out.add(v); v = alignUp(v * 2, blockSize) }
            out.add(to)
            return out.toList().sorted()
        }
    }
}

/**
 * One ladder point: the static shape a compiled decode (or prefill) step runs
 * at. [maxContext] is a whole number of pages by construction — see
 * [DecodeBucketPolicy].
 */
data class DecodeBucket(val batch: Int, val maxContext: Int) {
    init {
        require(batch >= 1) { "DecodeBucket: batch must be >= 1, got $batch" }
        require(maxContext >= 1) { "DecodeBucket: maxContext must be >= 1, got $maxContext" }
    }

    /** The block-table width this bucket implies, given [blockSize]. */
    fun maxBlocksPerSeq(blockSize: Int): Int {
        require(blockSize >= 1) { "DecodeBucket: blockSize must be >= 1, got $blockSize" }
        require(maxContext % blockSize == 0) {
            "DecodeBucket: maxContext $maxContext is not a whole number of pages of blockSize " +
                "$blockSize — bucket ladders are page-aligned precisely so this division is exact"
        }
        return maxContext / blockSize
    }

    override fun toString(): String = "b$batch" + "x" + "c$maxContext"
}
