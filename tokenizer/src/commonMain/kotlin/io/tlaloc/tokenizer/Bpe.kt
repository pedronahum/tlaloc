package io.tlaloc.tokenizer

/**
 * Open-addressing map from a token-id pair, packed `a shl 32 or b`, to the
 * merge's rank and the id of the merged token. No boxing: 440k merges take
 * three primitive arrays.
 */
internal class MergeTable(expected: Int) {
    private val capacity: Int = run {
        var c = 16
        while (c < expected * 2) c = c shl 1
        c
    }
    private val mask = capacity - 1
    private val keys = LongArray(capacity)
    private val ranks = IntArray(capacity) { -1 }
    private val merged = IntArray(capacity)

    private fun slot(key: Long): Int {
        var h = key * -0x61c8864680b583ebL
        h = h xor (h ushr 29)
        var i = h.toInt() and mask
        while (ranks[i] != -1 && keys[i] != key) i = (i + 1) and mask
        return i
    }

    /** Later entries replace earlier ones for the same pair, as a Rust HashMap collect does. */
    fun put(a: Int, b: Int, rank: Int, mergedId: Int) {
        val key = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
        val i = slot(key)
        keys[i] = key
        ranks[i] = rank
        merged[i] = mergedId
    }

    /** The slot for (a, b), or -1 when the pair does not merge. */
    fun find(a: Int, b: Int): Int {
        val i = slot((a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL))
        return if (ranks[i] == -1) -1 else i
    }

    fun rankAt(slot: Int): Int = ranks[slot]

    fun mergedAt(slot: Int): Int = merged[slot]
}

/**
 * The BPE model of Hugging Face tokenizers, for `continuing_subword_prefix`
 * and `end_of_word_suffix` empty and no dropout.
 *
 * A word starts as one symbol per code point (or per UTF-8 byte, through the
 * `<0xNN>` tokens, when [byteFallback] is set and the code point has no token).
 * Merges then apply lowest rank first, leftmost first among equal ranks,
 * until none applies. Candidates sit in a min-heap keyed by (rank, position)
 * over a doubly linked symbol list; a popped candidate whose pair no longer
 * exists is dropped. That keeps a 10,000-character word (SentencePiece-style
 * tokenizers see a whole line as one word) at O(n log n).
 */
internal class Bpe(
    val vocab: HashMap<String, Int>,
    private val merges: MergeTable,
    private val unkId: Int,
    private val fuseUnk: Boolean,
    private val byteFallback: Boolean,
    private val ignoreMerges: Boolean,
) {
    private val bmpIds = IntArray(0x10000) { -1 }
    private val astralIds = HashMap<Int, Int>()
    private val byteIds = IntArray(256) { -1 }
    private val cache: WordCache = newWordCache(CACHE_CAPACITY)

    init {
        for ((token, id) in vocab) {
            when {
                token.length == 1 -> bmpIds[token[0].code] = id
                token.length == 2 && token[0].isHighSurrogate() && token[1].isLowSurrogate() ->
                    astralIds[codePointAt(token, 0)] = id
            }
        }
        for (b in 0 until 256) {
            byteIds[b] = vocab["<0x" + HEX[b shr 4] + HEX[b and 15] + ">"] ?: -1
        }
    }

    /** Appends the ids of [word] to [out]. */
    fun tokenize(word: String, out: IntList) {
        if (word.isEmpty()) return
        if (ignoreMerges) {
            val whole = vocab[word]
            if (whole != null) { out.add(whole); return }
        }
        val cacheable = word.length < CACHE_MAX_WORD
        if (cacheable) {
            val hit = cache.get(word)
            if (hit != null) { out.addAll(hit); return }
        }
        val ids = mergeWord(word)
        if (cacheable) cache.put(word, ids)
        out.addAll(ids)
    }

    private fun mergeWord(word: String): IntArray {
        // Initial symbols, following tokenizers' BPE::merge_word, including
        // where a pending unknown is flushed relative to byte-fallback tokens.
        val sym = IntList(word.length + 4)
        var pendingUnk = false
        var i = 0
        while (i < word.length) {
            val cp = codePointAt(word, i)
            val width = if (cp >= 0x10000) 2 else 1
            val id = if (cp < 0x10000) bmpIds[cp] else astralIds[cp] ?: -1
            if (id >= 0) {
                if (pendingUnk) { sym.add(unkId); pendingUnk = false }
                sym.add(id)
            } else if (byteFallback && byteTokensFor(word, i, width, sym)) {
                // added by byteTokensFor
            } else if (unkId >= 0) {
                if (pendingUnk && !fuseUnk) sym.add(unkId)
                pendingUnk = true
            }
            i += width
        }
        if (pendingUnk) sym.add(unkId)
        val n = sym.size
        if (n <= 1) return sym.toIntArray()

        val ids = sym.toIntArray()
        val prev = IntArray(n) { it - 1 }
        val next = IntArray(n) { if (it + 1 < n) it + 1 else -1 }
        val alive = BooleanArray(n) { true }
        val heap = LongHeap(n)
        for (p in 0 until n - 1) {
            val s = merges.find(ids[p], ids[p + 1])
            if (s >= 0) heap.push(pack(merges.rankAt(s), p))
        }
        while (heap.size > 0) {
            val top = heap.pop()
            val rank = (top ushr 32).toInt()
            val pos = top.toInt()
            if (!alive[pos]) continue
            val right = next[pos]
            if (right < 0) continue
            val s = merges.find(ids[pos], ids[right])
            if (s < 0 || merges.rankAt(s) != rank) continue
            ids[pos] = merges.mergedAt(s)
            alive[right] = false
            val after = next[right]
            next[pos] = after
            if (after >= 0) prev[after] = pos
            val before = prev[pos]
            if (before >= 0) {
                val sp = merges.find(ids[before], ids[pos])
                if (sp >= 0) heap.push(pack(merges.rankAt(sp), before))
            }
            if (after >= 0) {
                val sn = merges.find(ids[pos], ids[after])
                if (sn >= 0) heap.push(pack(merges.rankAt(sn), pos))
            }
        }
        val out = IntList(n)
        var p = 0
        while (p >= 0) { out.add(ids[p]); p = next[p] }
        return out.toIntArray()
    }

    /** Adds the `<0xNN>` tokens of one code point's UTF-8 bytes, or nothing and false if one is missing. */
    private fun byteTokensFor(word: String, at: Int, width: Int, out: IntList): Boolean {
        val bytes = word.substring(at, at + width).encodeToByteArray()
        for (b in bytes) if (byteIds[b.toInt() and 0xFF] < 0) return false
        for (b in bytes) out.add(byteIds[b.toInt() and 0xFF])
        return true
    }

    private fun pack(rank: Int, pos: Int): Long = (rank.toLong() shl 32) or pos.toLong()

    companion object {
        private const val CACHE_CAPACITY = 10_000
        private const val CACHE_MAX_WORD = 256
        private val HEX = "0123456789ABCDEF".toCharArray()
    }
}

internal fun codePointAt(s: String, i: Int): Int {
    val c = s[i]
    if (c.isHighSurrogate() && i + 1 < s.length && s[i + 1].isLowSurrogate()) {
        return 0x10000 + ((c.code - 0xD800) shl 10) + (s[i + 1].code - 0xDC00)
    }
    return c.code
}

/** Binary min-heap of longs. */
internal class LongHeap(initial: Int) {
    private var a = LongArray(maxOf(4, initial))
    var size = 0
        private set

    fun push(v: Long) {
        if (size == a.size) a = a.copyOf(size * 2)
        var i = size++
        while (i > 0) {
            val parent = (i - 1) shr 1
            if (a[parent] <= v) break
            a[i] = a[parent]
            i = parent
        }
        a[i] = v
    }

    fun pop(): Long {
        val top = a[0]
        val last = a[--size]
        var i = 0
        while (true) {
            var c = 2 * i + 1
            if (c >= size) break
            if (c + 1 < size && a[c + 1] < a[c]) c++
            if (a[c] >= last) break
            a[i] = a[c]
            i = c
        }
        a[i] = last
        return top
    }
}

/** Growable int list without boxing. */
internal class IntList(initial: Int = 16) {
    private var a = IntArray(maxOf(4, initial))
    var size = 0
        private set

    fun add(v: Int) {
        if (size == a.size) a = a.copyOf(size * 2)
        a[size++] = v
    }

    fun addAll(vs: IntArray) {
        if (size + vs.size > a.size) a = a.copyOf(maxOf(size + vs.size, size * 2))
        vs.copyInto(a, size)
        size += vs.size
    }

    operator fun get(i: Int): Int = a[i]

    fun toIntArray(): IntArray = a.copyOf(size)
}
