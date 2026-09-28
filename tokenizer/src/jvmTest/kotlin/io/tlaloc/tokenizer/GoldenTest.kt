package io.tlaloc.tokenizer

import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Every id and string in the goldens (transformers 5.17 AutoTokenizer,
 * tokenizers 0.23 DecodeStream, apply_chat_template) reproduced exactly:
 * encode with and without special tokens and with special tokens split,
 * decode with and without skipping them, partial-id decodes, streaming
 * decode, and chat templates with the ids of their output.
 *
 * Each family self-skips when its checkpoint's tokenizer files are not in
 * the local cache.
 */
class GoldenTest {

    @Test fun qwen3() = check("qwen3")

    @Test fun museGlimmer() = check("muse_glimmer")

    @Test fun gpt2() = check("gpt2")

    @Test fun tinyllama() = check("tinyllama")

    @Test fun gemma4() = check("gemma4")

    private fun ints(v: JsonValue?): IntArray = (v as JsonArray).asIntList("ids").toIntArray()

    private fun check(family: String) {
        val golden = Checkpoints.golden(family)
        val dir = Checkpoints.directory(golden)
        assumeTrue(dir != null, "$family: tokenizer files for ${golden.str("checkpoint")} are not in the local cache")
        val tok = Checkpoints.tokenizer(family, dir!!)
        val failures = ArrayList<String>()
        var checks = 0
        fun expect(what: String, expected: Any?, actual: Any?) {
            checks++
            val same = if (expected is IntArray && actual is IntArray) expected.contentEquals(actual) else expected == actual
            if (!same) {
                val e = if (expected is IntArray) expected.toList() else expected
                val a = if (actual is IntArray) actual.toList() else actual
                failures += "$what\n  expected ${show(e)}\n  actual   ${show(a)}"
            }
        }

        for (section in listOf("cases", "random")) {
            for (c in golden.arr(section).elements.map { it as JsonObject }) {
                val text = (c["text"] as? JsonString)?.value ?: c.str("repeat").repeat(c.int("times"))
                val label = "$section ${show(text)}"
                expect("$label ids", ints(c["ids"]), tok.encode(text))
                c["idsNoSpecial"]?.let { expect("$label idsNoSpecial", ints(it), tok.encode(text, addSpecialTokens = false)) }
                c["idsSplitSpecial"]?.let {
                    expect("$label idsSplitSpecial", ints(it), tok.encode(text, addSpecialTokens = false, allowSpecial = false))
                }
                val ids = ints(c["ids"])
                (c["decode"] as? JsonString)?.let { expect("$label decode", it.value, tok.decode(ids)) }
                (c["decodeSkip"] as? JsonString)?.let { expect("$label decodeSkip", it.value, tok.decode(ids, skipSpecialTokens = true)) }
                (c["decodeSkipIsText"] as? JsonBool)?.let {
                    expect("$label decodeSkip == text", it.value, tok.decode(ids, skipSpecialTokens = true) == text)
                }
            }
        }
        for (d in golden.arr("decodes").elements.map { it as JsonObject }) {
            val ids = ints(d["ids"])
            expect("decode ${ids.toList()}", d.str("text"), tok.decode(ids))
            expect("decode skip ${ids.toList()}", d.str("textSkip"), tok.decode(ids, skipSpecialTokens = true))
        }
        for (s in golden.arr("streams").elements.map { it as JsonObject }) {
            val skip = (s["skip"] as JsonBool).value
            val stream = tok.decodeStream(skipSpecialTokens = skip)
            val chunks = ints(s["ids"]).map { stream.step(it) }
            val expected = s.arr("chunks").elements.map { if (it == JsonNull) null else (it as JsonString).value }
            expect("stream skip=$skip ${ints(s["ids"]).toList()}", expected, chunks)
        }
        for (c in golden.arr("chats").elements.map { it as JsonObject }) {
            val messages = c.arr("messages").elements.map {
                val m = it as JsonObject
                ChatMessage(m.str("role"), m.str("content"), (m["reasoning_content"] as? JsonString)?.value)
            }
            val vars = c.obj("vars")
            val options = ChatOptions(
                addGenerationPrompt = (c["addGenerationPrompt"] as JsonBool).value,
                enableThinking = (vars["enable_thinking"] as? JsonBool)?.value,
                currentDate = (vars["current_date"] as? JsonString)?.value,
                reasoningStrength = (vars["reasoning_strength"] as? JsonString)?.value,
                knowledgeCutoff = (vars["knowledge_cutoff"] as? JsonString)?.value,
            )
            val rendered = tok.applyChatTemplate(messages, options)
            expect("chat $messages $options", c.str("rendered"), rendered)
            expect("chat ids $messages $options", ints(c["ids"]), tok.encode(c.str("rendered"), addSpecialTokens = false))
        }

        println("$family: $checks checks, ${failures.size} mismatches")
        assertTrue(failures.isEmpty(), "$family: ${failures.size} of $checks differ:\n" + failures.take(15).joinToString("\n"))
    }

    private fun JsonObject.int(key: String): Int = (this[key] as io.tlaloc.core.io.JsonNumber).asInt(key)

    private fun show(v: Any?): String {
        val s = when (v) {
            is String -> buildString {
                append('"')
                for (ch in v) if (ch.code < 0x20 || ch.code in 0x7f..0xa0 || ch.isSurrogate()) append("\\u%04x".format(ch.code)) else append(ch)
                append('"')
            }
            else -> v.toString()
        }
        return if (s.length > 300) s.take(300) + "... (${s.length} chars)" else s
    }
}
