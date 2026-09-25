package io.tlaloc.ir.inference

/**
 * A second KV pool class for sliding-window layers, whose pages a sequence
 * RECYCLES as it advances, so that the pages it holds are bounded by the
 * window and not by its length.
 *
 * A sliding layer with window `W` reads, for a query at position `p`, only
 * the positions `p - W + 1 .. p` ([io.tlaloc.ir.PagedAttentionAttrs]). The
 * keys and values of every earlier position are never read again. The
 * layers listed in [layers] keep their KV in pools of this class: each
 * sequence holds a RING of at most [ringPages] pages, and logical block `b`
 * of the sequence (positions `b * blockSize .. (b + 1) * blockSize - 1`)
 * lives on ring page `b % ringPages`. From block `ringPages` on, each new
 * position is written over the position `ringPages * blockSize` before it,
 * which has left the window.
 *
 * ## Why the paged-attention op does not change
 *
 * The windowed layers get their own block table and slot mapping
 * ([DecodeSlotRole.WINDOW_BLOCK_TABLES], [DecodeSlotRole.WINDOW_SLOT_MAPPING]),
 * as wide as the full ones: entry `b` of a sequence's windowed block table
 * is the ring page of logical block `b`. The op reads no position outside
 * the window, and (below) no position in the window has been written over,
 * so every position it reads is at the slot the table names, holding that
 * position's keys and values. The positions it does not read are masked, as
 * they are in a full-history pool.
 *
 * ## How many pages the ring needs
 *
 * Position `p` is written to slot `p % blockSize` of ring page
 * `(p / blockSize) % ringPages`, so the ring is a ring of
 * `ringPages * blockSize` positions: two positions share a slot exactly when
 * they are a multiple of that apart. While block `b` is being filled, the
 * page it shares with block `b - ringPages` still holds that block's later
 * positions, and those are the ones still in the window.
 *
 * One call writes the keys and values of all its tokens before any row
 * attends (see [HfDecoderGraph]). A call that writes `n` tokens starting at
 * position `p0` therefore needs the positions
 * `max(0, p0 - W + 1) .. p0 + n - 1` on distinct slots at once, which is
 * `min(p0, W - 1) + n <= ringPages * blockSize`. A decode step (`n = 1`)
 * needs `ringPages * blockSize >= W`, so [minRingPages] is `ceil(W / blockSize)`
 * and a smaller ring is wrong. A prefill chunk needs more, so a runtime
 * splits a request into calls of at most [maxTokensPerCall] tokens. The
 * default ring ([defaultRingPages]) has one page more than the minimum,
 * which lets every call write at least `blockSize + 1` tokens.
 *
 * A ring as long as the largest context never wraps, and no ring needs to be
 * longer; [HfDecoderConfig.windowedKvPool] caps it there.
 *
 * @property window the largest sliding window of the [layers]; a layer with
 *   a smaller window reads a subset of what the ring holds.
 * @property layers the decoder layers whose pools are of this class, ascending.
 * @property numBlocks pages in each pool of this class; page 0 is the
 *   padding page, as in the full pools.
 * @property ringPages the most pages one sequence holds in this class.
 */
data class WindowedKvPool(
    val window: Int,
    val layers: List<Int>,
    val numBlocks: Int,
    val ringPages: Int,
) {
    init {
        require(window >= 1) { "WindowedKvPool: window must be >= 1, got $window" }
        require(layers.isNotEmpty()) { "WindowedKvPool: a windowed pool class with no layers" }
        require(layers.first() >= 0 && layers.zipWithNext().all { (a, b) -> a < b }) {
            "WindowedKvPool: layers must be distinct, ascending and >= 0, got $layers"
        }
        require(ringPages >= 1) { "WindowedKvPool: ringPages must be >= 1, got $ringPages" }
        require(numBlocks >= 1 + ringPages) {
            "WindowedKvPool: numBlocks $numBlocks cannot hold one ring of $ringPages pages " +
                "besides the padding page 0"
        }
    }

    /** The ring slot of logical block [block]: its page is the sequence's `ring[ringSlot(block)]`. */
    fun ringSlot(block: Int): Int = block % ringPages

    /** Pages a sequence of [length] positions holds in this class. */
    fun pagesHeld(length: Int, blockSize: Int): Int = minOf(ringPages, (length + blockSize - 1) / blockSize)

    /**
     * The most tokens one call may write when the first of them is at
     * position [start]: the positions the call needs at once,
     * `max(0, start - window + 1) .. start + n - 1`, must be on distinct
     * slots of the ring's `ringPages * blockSize`. At least 1 whenever the
     * ring is at least [minRingPages] long.
     */
    fun maxTokensPerCall(start: Int, blockSize: Int): Int =
        ringPages * blockSize - minOf(start, window - 1)

    companion object {
        /**
         * The fewest ring pages a decode step at any position needs: the
         * ring must hold `window` positions, `ceil(window / blockSize)` pages.
         */
        fun minRingPages(window: Int, blockSize: Int): Int {
            require(window >= 1 && blockSize >= 1) {
                "WindowedKvPool.minRingPages: window and blockSize must be >= 1, got $window, $blockSize"
            }
            return (window + blockSize - 1) / blockSize
        }

        /** [minRingPages] plus one page, so every call can write more than a page. */
        fun defaultRingPages(window: Int, blockSize: Int): Int = minRingPages(window, blockSize) + 1
    }
}
