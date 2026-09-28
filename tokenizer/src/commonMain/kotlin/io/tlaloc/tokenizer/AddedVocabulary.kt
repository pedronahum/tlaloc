package io.tlaloc.tokenizer

/** One entry of tokenizer.json's `added_tokens`. */
internal class AddedToken(val id: Int, val content: String, val special: Boolean, val normalized: Boolean)

/**
 * Finds added tokens in text: leftmost match first, longest among matches
 * starting at the same position, which is aho-corasick's LeftmostLongest as
 * tokenizers uses it.
 */
internal class TokenMatcher(tokens: List<Pair<String, Int>>) {
    private class Node {
        val children = HashMap<Char, Node>()
        var id = -1
    }

    private val root = Node()
    private val firstChars = HashSet<Char>()
    val isEmpty = tokens.isEmpty()

    init {
        for ((content, id) in tokens) {
            if (content.isEmpty()) continue
            var n = root
            for (c in content) n = n.children.getOrPut(c) { Node() }
            n.id = id
            firstChars.add(content[0])
        }
    }

    /** Matches in [text] as `[start, end, id]` triples. */
    fun find(text: String): IntArray {
        if (isEmpty) return IntArray(0)
        val out = IntList()
        var i = 0
        while (i < text.length) {
            if (text[i] !in firstChars) { i++; continue }
            var n: Node? = root
            var j = i
            var bestEnd = -1
            var bestId = -1
            while (j < text.length) {
                n = n!!.children[text[j]] ?: break
                j++
                if (n.id >= 0) { bestEnd = j; bestId = n.id }
            }
            if (bestEnd > 0) {
                out.add(i); out.add(bestEnd); out.add(bestId)
                i = bestEnd
            } else {
                i++
            }
        }
        return out.toIntArray()
    }
}

/** A stretch of input: either an added token ([id] >= 0) or text for the model. */
internal class Segment(val text: String, val atOrigin: Boolean, val id: Int)

/**
 * Cuts [text] at the added tokens [matcher] finds. A match [keep] rejects stays
 * part of the surrounding text: tokenizers drops special-token matches that way
 * under `encode_special_tokens`, after matching, so they still shadow any
 * shorter token inside them.
 */
internal fun splitOnTokens(
    text: String,
    atOrigin: Boolean,
    matcher: TokenMatcher,
    keep: (Int) -> Boolean,
    out: MutableList<Segment>,
    onText: (String, Boolean, MutableList<Segment>) -> Unit,
) {
    val m = matcher.find(text)
    var last = 0
    var k = 0
    while (k < m.size) {
        if (!keep(m[k + 2])) { k += 3; continue }
        if (m[k] > last) onText(text.substring(last, m[k]), atOrigin && last == 0, out)
        out.add(Segment(text.substring(m[k], m[k + 1]), atOrigin && m[k] == 0, m[k + 2]))
        last = m[k + 1]
        k += 3
    }
    if (last < text.length) onText(text.substring(last), atOrigin && last == 0, out)
}
