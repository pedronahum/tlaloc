package io.tlaloc.endpoint

import java.io.ByteArrayOutputStream

/**
 * HPACK (RFC 7541) for an HTTP/2 client: the decoder reads every representation a server may
 * send, with the dynamic table and Huffman-coded strings; the encoder writes each header as a
 * literal without indexing and without Huffman coding, which every decoder accepts.
 */
internal class HpackDecoder(private var maxTableSize: Int = 4096) {
    private val table = ArrayDeque<Pair<String, String>>() // newest first
    private var tableSize = 0

    fun decode(block: ByteArray): List<Pair<String, String>> {
        val out = ArrayList<Pair<String, String>>()
        var p = 0
        fun int(prefix: Int): Int {
            val mask = (1 shl prefix) - 1
            var v = block[p++].toInt() and mask
            if (v < mask) return v
            var shift = 0
            while (true) {
                val b = block[p++].toInt() and 0xff
                v += (b and 0x7f) shl shift
                if (b and 0x80 == 0) return v
                shift += 7
                if (shift > 28) throw Http2Exception("HPACK integer too large")
            }
        }
        fun string(): String {
            val huffman = block[p].toInt() and 0x80 != 0
            val n = int(7)
            if (p + n > block.size) throw Http2Exception("HPACK string runs past the header block")
            val bytes = if (huffman) huffmanDecode(block, p, n) else block.copyOfRange(p, p + n)
            p += n
            return String(bytes, Charsets.ISO_8859_1)
        }
        while (p < block.size) {
            val b = block[p].toInt() and 0xff
            when {
                b and 0x80 != 0 -> out += entry(int(7))
                b and 0xc0 == 0x40 -> {
                    val index = int(6)
                    val name = if (index == 0) string() else entry(index).first
                    val field = name to string()
                    out += field
                    add(field)
                }
                b and 0xe0 == 0x20 -> {
                    val size = int(5)
                    if (size > maxTableSize) throw Http2Exception("HPACK table size update $size above $maxTableSize")
                    maxTableSize = size
                    evict(0)
                }
                else -> { // literal without indexing (0000) or never indexed (0001)
                    val index = int(4)
                    val name = if (index == 0) string() else entry(index).first
                    out += name to string()
                }
            }
        }
        return out
    }

    private fun entry(index: Int): Pair<String, String> = when {
        index in 1..STATIC_TABLE.size -> STATIC_TABLE[index - 1]
        index - STATIC_TABLE.size - 1 in table.indices -> table[index - STATIC_TABLE.size - 1]
        else -> throw Http2Exception("HPACK index $index is outside the tables")
    }

    private fun add(field: Pair<String, String>) {
        val size = field.first.length + field.second.length + 32
        evict(size)
        if (size <= maxTableSize) {
            table.addFirst(field)
            tableSize += size
        }
    }

    private fun evict(room: Int) {
        while (tableSize + room > maxTableSize && table.isNotEmpty()) {
            val f = table.removeLast()
            tableSize -= f.first.length + f.second.length + 32
        }
    }
}

internal object HpackEncoder {
    fun encode(headers: List<Pair<String, String>>): ByteArray {
        val out = ByteArrayOutputStream()
        for ((name, value) in headers) {
            out.write(0) // literal without indexing, new name
            string(out, name)
            string(out, value)
        }
        return out.toByteArray()
    }

    private fun string(out: ByteArrayOutputStream, s: String) {
        val bytes = s.toByteArray(Charsets.ISO_8859_1)
        int(out, bytes.size, 7, 0)
        out.write(bytes)
    }

    fun int(out: ByteArrayOutputStream, value: Int, prefix: Int, flags: Int) {
        val mask = (1 shl prefix) - 1
        if (value < mask) {
            out.write(flags or value)
            return
        }
        out.write(flags or mask)
        var v = value - mask
        while (v >= 0x80) {
            out.write((v and 0x7f) or 0x80)
            v = v ushr 7
        }
        out.write(v)
    }
}

/** The decoding tree: node `n`'s children are `TREE[2n]` and `TREE[2n + 1]`; a leaf holds `-1 - symbol`. */
private val HUFFMAN_TREE: IntArray by lazy {
    val tree = IntArray(2 * 2 * 257) // at most 256 inner nodes
    var nodes = 1
    for (sym in 0..256) {
        val code = HUFFMAN_CODES[sym]
        val len = HUFFMAN_LENGTHS[sym].toInt()
        var node = 0
        for (i in len - 1 downTo 0) {
            val slot = 2 * node + ((code ushr i) and 1)
            node = if (i == 0) {
                tree[slot] = -1 - sym
                0
            } else {
                if (tree[slot] == 0) tree[slot] = nodes++
                tree[slot]
            }
        }
    }
    tree
}

internal fun huffmanDecode(src: ByteArray, off: Int, len: Int): ByteArray {
    val out = ByteArrayOutputStream(len * 8 / 5 + 1)
    var node = 0
    var depth = 0 // bits since the last symbol
    var allOnes = true
    for (k in off until off + len) {
        val b = src[k].toInt()
        for (i in 7 downTo 0) {
            val bit = (b ushr i) and 1
            val next = HUFFMAN_TREE[2 * node + bit]
            depth++
            allOnes = allOnes && bit == 1
            when {
                next < 0 -> {
                    val sym = -1 - next
                    if (sym == 256) throw Http2Exception("Huffman string holds EOS")
                    out.write(sym)
                    node = 0
                    depth = 0
                    allOnes = true
                }
                next == 0 -> throw Http2Exception("invalid Huffman code")
                else -> node = next
            }
        }
    }
    // Padding: fewer than 8 bits, all ones (a prefix of EOS).
    if (depth > 7 || !allOnes) throw Http2Exception("invalid Huffman padding")
    return out.toByteArray()
}

// RFC 7541 Appendix B: each symbol's Huffman code (right-aligned) and its length in bits; symbol 256 is EOS.
private val HUFFMAN_CODES = intArrayOf(
    0x1ff8, 0x7fffd8, 0xfffffe2, 0xfffffe3, 0xfffffe4, 0xfffffe5, 0xfffffe6, 0xfffffe7,
    0xfffffe8, 0xffffea, 0x3ffffffc, 0xfffffe9, 0xfffffea, 0x3ffffffd, 0xfffffeb, 0xfffffec,
    0xfffffed, 0xfffffee, 0xfffffef, 0xffffff0, 0xffffff1, 0xffffff2, 0x3ffffffe, 0xffffff3,
    0xffffff4, 0xffffff5, 0xffffff6, 0xffffff7, 0xffffff8, 0xffffff9, 0xffffffa, 0xffffffb,
    0x14, 0x3f8, 0x3f9, 0xffa, 0x1ff9, 0x15, 0xf8, 0x7fa,
    0x3fa, 0x3fb, 0xf9, 0x7fb, 0xfa, 0x16, 0x17, 0x18,
    0x0, 0x1, 0x2, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
    0x1e, 0x1f, 0x5c, 0xfb, 0x7ffc, 0x20, 0xffb, 0x3fc,
    0x1ffa, 0x21, 0x5d, 0x5e, 0x5f, 0x60, 0x61, 0x62,
    0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
    0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72,
    0xfc, 0x73, 0xfd, 0x1ffb, 0x7fff0, 0x1ffc, 0x3ffc, 0x22,
    0x7ffd, 0x3, 0x23, 0x4, 0x24, 0x5, 0x25, 0x26,
    0x27, 0x6, 0x74, 0x75, 0x28, 0x29, 0x2a, 0x7,
    0x2b, 0x76, 0x2c, 0x8, 0x9, 0x2d, 0x77, 0x78,
    0x79, 0x7a, 0x7b, 0x7ffe, 0x7fc, 0x3ffd, 0x1ffd, 0xffffffc,
    0xfffe6, 0x3fffd2, 0xfffe7, 0xfffe8, 0x3fffd3, 0x3fffd4, 0x3fffd5, 0x7fffd9,
    0x3fffd6, 0x7fffda, 0x7fffdb, 0x7fffdc, 0x7fffdd, 0x7fffde, 0xffffeb, 0x7fffdf,
    0xffffec, 0xffffed, 0x3fffd7, 0x7fffe0, 0xffffee, 0x7fffe1, 0x7fffe2, 0x7fffe3,
    0x7fffe4, 0x1fffdc, 0x3fffd8, 0x7fffe5, 0x3fffd9, 0x7fffe6, 0x7fffe7, 0xffffef,
    0x3fffda, 0x1fffdd, 0xfffe9, 0x3fffdb, 0x3fffdc, 0x7fffe8, 0x7fffe9, 0x1fffde,
    0x7fffea, 0x3fffdd, 0x3fffde, 0xfffff0, 0x1fffdf, 0x3fffdf, 0x7fffeb, 0x7fffec,
    0x1fffe0, 0x1fffe1, 0x3fffe0, 0x1fffe2, 0x7fffed, 0x3fffe1, 0x7fffee, 0x7fffef,
    0xfffea, 0x3fffe2, 0x3fffe3, 0x3fffe4, 0x7ffff0, 0x3fffe5, 0x3fffe6, 0x7ffff1,
    0x3ffffe0, 0x3ffffe1, 0xfffeb, 0x7fff1, 0x3fffe7, 0x7ffff2, 0x3fffe8, 0x1ffffec,
    0x3ffffe2, 0x3ffffe3, 0x3ffffe4, 0x7ffffde, 0x7ffffdf, 0x3ffffe5, 0xfffff1, 0x1ffffed,
    0x7fff2, 0x1fffe3, 0x3ffffe6, 0x7ffffe0, 0x7ffffe1, 0x3ffffe7, 0x7ffffe2, 0xfffff2,
    0x1fffe4, 0x1fffe5, 0x3ffffe8, 0x3ffffe9, 0xffffffd, 0x7ffffe3, 0x7ffffe4, 0x7ffffe5,
    0xfffec, 0xfffff3, 0xfffed, 0x1fffe6, 0x3fffe9, 0x1fffe7, 0x1fffe8, 0x7ffff3,
    0x3fffea, 0x3fffeb, 0x1ffffee, 0x1ffffef, 0xfffff4, 0xfffff5, 0x3ffffea, 0x7ffff4,
    0x3ffffeb, 0x7ffffe6, 0x3ffffec, 0x3ffffed, 0x7ffffe7, 0x7ffffe8, 0x7ffffe9, 0x7ffffea,
    0x7ffffeb, 0xffffffe, 0x7ffffec, 0x7ffffed, 0x7ffffee, 0x7ffffef, 0x7fffff0, 0x3ffffee,
    0x3fffffff,
)
private val HUFFMAN_LENGTHS = byteArrayOf(
    13, 23, 28, 28, 28, 28, 28, 28, 28, 24, 30, 28, 28, 30, 28, 28,
    28, 28, 28, 28, 28, 28, 30, 28, 28, 28, 28, 28, 28, 28, 28, 28,
    6, 10, 10, 12, 13, 6, 8, 11, 10, 10, 8, 11, 8, 6, 6, 6,
    5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 15, 6, 12, 10,
    13, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 8, 7, 8, 13, 19, 13, 14, 6,
    15, 5, 6, 5, 6, 5, 6, 6, 6, 5, 7, 7, 6, 6, 6, 5,
    6, 7, 6, 5, 5, 6, 7, 7, 7, 7, 7, 15, 11, 14, 13, 28,
    20, 22, 20, 20, 22, 22, 22, 23, 22, 23, 23, 23, 23, 23, 24, 23,
    24, 24, 22, 23, 24, 23, 23, 23, 23, 21, 22, 23, 22, 23, 23, 24,
    22, 21, 20, 22, 22, 23, 23, 21, 23, 22, 22, 24, 21, 22, 23, 23,
    21, 21, 22, 21, 23, 22, 23, 23, 20, 22, 22, 22, 23, 22, 22, 23,
    26, 26, 20, 19, 22, 23, 22, 25, 26, 26, 26, 27, 27, 26, 24, 25,
    19, 21, 26, 27, 27, 26, 27, 24, 21, 21, 26, 26, 28, 27, 27, 27,
    20, 24, 20, 21, 22, 21, 21, 23, 22, 22, 25, 25, 24, 24, 26, 23,
    26, 27, 26, 26, 27, 27, 27, 27, 27, 28, 27, 27, 27, 27, 27, 26,
    30,
)

// RFC 7541 Appendix A: the static table, indices 1 to 61.
private val STATIC_TABLE = arrayOf(
    ":authority" to "",
    ":method" to "GET",
    ":method" to "POST",
    ":path" to "/",
    ":path" to "/index.html",
    ":scheme" to "http",
    ":scheme" to "https",
    ":status" to "200",
    ":status" to "204",
    ":status" to "206",
    ":status" to "304",
    ":status" to "400",
    ":status" to "404",
    ":status" to "500",
    "accept-charset" to "",
    "accept-encoding" to "gzip, deflate",
    "accept-language" to "",
    "accept-ranges" to "",
    "accept" to "",
    "access-control-allow-origin" to "",
    "age" to "",
    "allow" to "",
    "authorization" to "",
    "cache-control" to "",
    "content-disposition" to "",
    "content-encoding" to "",
    "content-language" to "",
    "content-length" to "",
    "content-location" to "",
    "content-range" to "",
    "content-type" to "",
    "cookie" to "",
    "date" to "",
    "etag" to "",
    "expect" to "",
    "expires" to "",
    "from" to "",
    "host" to "",
    "if-match" to "",
    "if-modified-since" to "",
    "if-none-match" to "",
    "if-range" to "",
    "if-unmodified-since" to "",
    "last-modified" to "",
    "link" to "",
    "location" to "",
    "max-forwards" to "",
    "proxy-authenticate" to "",
    "proxy-authorization" to "",
    "range" to "",
    "referer" to "",
    "refresh" to "",
    "retry-after" to "",
    "server" to "",
    "set-cookie" to "",
    "strict-transport-security" to "",
    "transfer-encoding" to "",
    "user-agent" to "",
    "vary" to "",
    "via" to "",
    "www-authenticate" to "",
)
