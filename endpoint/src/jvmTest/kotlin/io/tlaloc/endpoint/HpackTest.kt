package io.tlaloc.endpoint

import kotlin.test.Test
import kotlin.test.assertEquals

/** RFC 7541 Appendix C's examples. */
class HpackTest {
    private fun hex(s: String): ByteArray = s.filter { !it.isWhitespace() }.chunked(2).map { it.toInt(16).toByte() }.toByteArray()

    @Test fun huffmanStrings() {
        assertEquals("www.example.com", String(hex("f1e3 c2e5 f23a 6ba0 ab90 f4ff").let { huffmanDecode(it, 0, it.size) }))
        assertEquals("no-cache", String(hex("a8eb 1064 9cbf").let { huffmanDecode(it, 0, it.size) }))
        assertEquals("custom-value", String(hex("25a8 49e9 5bb8 e8b4 bf").let { huffmanDecode(it, 0, it.size) }))
    }

    @Test fun requestsWithHuffmanAndTheDynamicTable() {
        val d = HpackDecoder()
        assertEquals(
            listOf(":method" to "GET", ":scheme" to "http", ":path" to "/", ":authority" to "www.example.com"),
            d.decode(hex("8286 8441 8cf1 e3c2 e5f2 3a6b a0ab 90f4 ff")),
        )
        assertEquals(
            listOf(":method" to "GET", ":scheme" to "http", ":path" to "/", ":authority" to "www.example.com", "cache-control" to "no-cache"),
            d.decode(hex("8286 84be 5886 a8eb 1064 9cbf")),
        )
        assertEquals(
            listOf(":method" to "GET", ":scheme" to "https", ":path" to "/index.html", ":authority" to "www.example.com", "custom-key" to "custom-value"),
            d.decode(hex("8287 85bf 4088 25a8 49e9 5ba9 7d7f 8925 a849 e95b b8e8 b4bf")),
        )
    }

    @Test fun responsesEvictFromTheDynamicTable() {
        // C.5: a 256-byte table, literals with indexing, no Huffman.
        val d = HpackDecoder(256)
        assertEquals(
            listOf(":status" to "302", "cache-control" to "private", "date" to "Mon, 21 Oct 2013 20:13:21 GMT", "location" to "https://www.example.com"),
            d.decode(hex("4803 3330 3258 0770 7269 7661 7465 611d 4d6f 6e2c 2032 3120 4f63 7420 3230 3133 2032 303a 3133 3a32 3120 474d 546e 1768 7474 7073 3a2f 2f77 7777 2e65 7861 6d70 6c65 2e63 6f6d")),
        )
        assertEquals(
            listOf(":status" to "307", "cache-control" to "private", "date" to "Mon, 21 Oct 2013 20:13:21 GMT", "location" to "https://www.example.com"),
            d.decode(hex("4803 3330 37c1 c0bf")),
        )
    }

    @Test fun theEncoderRoundTrips() {
        val h = listOf(":method" to "POST", ":path" to "/inference.GRPCInferenceService/ModelStreamInfer", "te" to "trailers", "x".repeat(300) to "y")
        assertEquals(h, HpackDecoder().decode(HpackEncoder.encode(h)))
    }
}
