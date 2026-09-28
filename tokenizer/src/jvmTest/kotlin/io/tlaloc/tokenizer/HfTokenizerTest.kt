package io.tlaloc.tokenizer

import io.tlaloc.core.io.jsonQuote
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Behaviour that needs no checkpoint: a byte-level BPE built inline, the UTF-8 rules, and the refusals. */
class HfTokenizerTest {

    /** An ASCII character's id: its byte value, in the vocabulary below. */
    private fun byteId(c: Char): Int = c.code

    /** Vocabulary: the 256 byte characters as ids 0..255 (by byte value), then merged tokens from 256. */
    private fun tokenizerJson(
        merges: List<Pair<String, String>> = listOf("a" to "b", "ab" to "c", "b" to "c", "a" to "a"),
        added: String = """
            {"id": 300, "content": "<a>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false},
            {"id": 301, "content": "<a><b>", "special": false, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false},
            {"id": 302, "content": "<ab>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false},
            {"id": 303, "content": "b>", "special": false, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false}
        """,
        normalizer: String = "null",
        postProcessor: String = "null",
        modelExtra: String = "",
    ): String {
        val vocab = LinkedHashMap<String, Int>()
        for (b in 0 until 256) vocab[ByteLevelTable.byteToChar[b].toString()] = b
        for ((a, b) in merges) vocab.getOrPut(a + b) { vocab.size }
        val vocabJson = vocab.entries.joinToString(",") { "${jsonQuote(it.key)}: ${it.value}" }
        val mergesJson = merges.joinToString(",") { "[${jsonQuote(it.first)}, ${jsonQuote(it.second)}]" }
        return """
            {"version": "1.0", "truncation": null, "padding": null,
             "added_tokens": [$added],
             "normalizer": $normalizer,
             "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
             "post_processor": $postProcessor,
             "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
             "model": {"type": "BPE", "dropout": null, "unk_token": null, "fuse_unk": false, "byte_fallback": false,
                       $modelExtra "vocab": {$vocabJson}, "merges": [$mergesJson]}}
        """.trimIndent()
    }

    private val tok = HfTokenizer.fromJson(tokenizerJson())

    @Test fun `merges apply lowest rank first, leftmost among equal ranks`() {
        val ab = tok.tokenToId("ab")!!
        val abc = tok.tokenToId("abc")!!
        val aa = tok.tokenToId("aa")!!
        // (a,b) outranks (b,c), so "abc" is ab+c, which merges to abc.
        assertContentEquals(intArrayOf(abc), tok.encode("abc"))
        assertContentEquals(intArrayOf(ab), tok.encode("ab"))
        // Equal ranks: the leftmost pair merges first.
        assertContentEquals(intArrayOf(aa, aa, byteId('a')), tok.encode("aaaaa"))
    }

    @Test fun `added tokens match leftmost-longest, and allowSpecial false keeps their text`() {
        assertContentEquals(intArrayOf(301), tok.encode("<a><b>"))
        assertContentEquals(intArrayOf(300, byteId('x')), tok.encode("<a>x"))
        // Special tokens as text: "<a>" becomes bytes; the non-special "<a><b>" still matches.
        val plain = tok.encode("<a>x", allowSpecial = false)
        assertTrue(300 !in plain)
        assertContentEquals(intArrayOf(301), tok.encode("<a><b>", allowSpecial = false))
        // A skipped special match still covers its text: "b>" inside "<ab>" is not found.
        val ab = tok.encode("<ab>", allowSpecial = false)
        assertTrue(302 !in ab && 303 !in ab, ab.toList().toString())
    }

    @Test fun `decode round-trips and skips special tokens`() {
        val text = "abc <a> héllo 👍"
        val ids = tok.encode(text)
        assertEquals(text, tok.decode(ids))
        assertEquals("abc  héllo 👍", tok.decode(ids, skipSpecialTokens = true))
        assertTrue(tok.isSpecial(300))
        assertEquals("<a>", tok.idToToken(300))
    }

    @Test fun `decode stream holds back an incomplete character`() {
        val stream = tok.decodeStream()
        val ids = tok.encode("é!")
        assertEquals(3, ids.size)
        assertNull(stream.step(ids[0]))
        assertEquals("é", stream.step(ids[1]))
        assertEquals("!", stream.step(ids[2]))
    }

    @Test fun `template post-processor adds its tokens only with addSpecialTokens`() {
        val pp = """{"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<a>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [], "special_tokens": {"<a>": {"id": "<a>", "ids": [300], "tokens": ["<a>"]}}}"""
        val t = HfTokenizer.fromJson(tokenizerJson(postProcessor = pp))
        assertContentEquals(intArrayOf(300, byteId('x')), t.encode("x"))
        assertContentEquals(intArrayOf(byteId('x')), t.encode("x", addSpecialTokens = false))
        assertContentEquals(intArrayOf(300), t.encode(""))
    }

    @Test fun `lossy UTF-8 replaces each maximal invalid subsequence once`() {
        fun lossy(vararg b: Int) = decodeUtf8Lossy(ByteArray(b.size) { b[it].toByte() })
        assertEquals("�", lossy(0xE2, 0x82))
        assertEquals("�A", lossy(0xF0, 0x9F, 0x98, 0x41))
        assertEquals("��", lossy(0xC0, 0x80))
        assertEquals("���", lossy(0xED, 0xA0, 0x80))
        assertEquals("€", lossy(0xE2, 0x82, 0xAC))
        assertNull(decodeUtf8Strict(byteArrayOf(0xE2.toByte(), 0x82.toByte()), 2))
        assertEquals("�", decodeUtf8Strict(byteArrayOf(0xEF.toByte(), 0xBF.toByte(), 0xBD.toByte()), 3))
    }

    @Test fun `unpaired surrogates are refused`() {
        assertFailsWith<IllegalArgumentException> { tok.encode("a\uD83Db") }
        assertFailsWith<IllegalArgumentException> { tok.encode("\uDC4D") }
    }

    @Test fun `unimplemented components are refused by name`() {
        fun refusal(json: String, config: String? = null) =
            assertFailsWith<UnsupportedTokenizerException> { HfTokenizer.fromJson(json, config) }.message!!
        assertTrue("Lowercase" in refusal(tokenizerJson(normalizer = """{"type": "Lowercase"}""")))
        assertTrue("dropout" in refusal(tokenizerJson(modelExtra = """"dropout": 0.1,""").replace(""""dropout": null, """, "")))
        val lstrip = """{"id": 300, "content": "<a>", "special": true, "normalized": false, "lstrip": true, "rstrip": false, "single_word": false}"""
        assertTrue("lstrip" in refusal(tokenizerJson(added = lstrip)))
        assertTrue("BertTokenizer" in refusal(tokenizerJson(), """{"tokenizer_class": "BertTokenizerFast"}"""))
        assertTrue("<missing>" in refusal(tokenizerJson(), """{"tokenizer_class": "TokenizersBackend", "eos_token": "<missing>"}"""))
    }

    @Test fun `chat templates are recognized by their exact source`() {
        val t = HfTokenizer.fromJson(tokenizerJson(), """{"tokenizer_class": "TokenizersBackend", "chat_template": "{{ messages }}"}""")
        assertNull(t.chatTemplate)
        assertFailsWith<IllegalStateException> { t.applyChatTemplate(listOf(ChatMessage("user", "hi"))) }
        val qwen = t.applyChatTemplate(
            listOf(ChatMessage("user", "hi")),
            ChatOptions(enableThinking = false),
            template = ChatTemplate.QWEN3,
        )
        assertEquals("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", qwen)
        assertFailsWith<IllegalArgumentException> {
            t.applyChatTemplate(listOf(ChatMessage("narrator", "hi")), template = ChatTemplate.QWEN3)
        }
    }
}
