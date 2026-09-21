package io.tlaloc.core.io

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.468 (Phase H2) — the strict JSON reader that parses safetensors
 * headers. These tests exist because the header is UNTRUSTED INPUT: a tensor
 * name is an arbitrary JSON string, so the escapes, the control-character
 * rule and the duplicate-key refusal are all load-bearing, not decoration.
 */
class JsonTest {

    @Test
    fun parsesTheShapeOfASafetensorsHeader() {
        val v = parseJson(
            "{\"__metadata__\":{\"format\":\"pt\"}," +
                "\"model.layers.0.self_attn.q_proj.weight\":" +
                "{\"dtype\":\"BF16\",\"shape\":[4096,4096],\"data_offsets\":[0,33554432]}}",
        ) as JsonObject
        val e = v.obj("model.layers.0.self_attn.q_proj.weight")
        assertEquals("BF16", e.str("dtype"))
        assertEquals(listOf(4096, 4096), e.arr("shape").asIntList("shape"))
        assertEquals(listOf(0L, 33554432L), e.arr("data_offsets").asLongList("offs"))
    }

    @Test
    fun escapesInAStringAreDecoded() {
        val v = parseJson("{\"a\\\"b,{}\":\"x\\n\\t\\\\\\u0041\"}") as JsonObject
        assertEquals("x\n\t\\A", v.str("a\"b,{}"))
    }

    @Test
    fun aRawControlCharacterInAStringIsRefused() {
        val e = assertFailsWith<JsonException> { parseJson("{\"a\u0001\":1}") }
        assertTrue("control character" in e.message!!, e.message!!)
    }

    @Test
    fun duplicateKeysAreRefusedNotLastWins() {
        val e = assertFailsWith<JsonException> {
            parseJson("{\"w\":{\"dtype\":\"F32\"},\"w\":{\"dtype\":\"BF16\"}}")
        }
        assertTrue("duplicate key" in e.message!!, e.message!!)
    }

    @Test
    fun trailingContentIsRefused() {
        assertFailsWith<JsonException> { parseJson("{} {}") }
    }

    @Test
    fun trailingCommasAndUnquotedKeysAreRefused() {
        assertFailsWith<JsonException> { parseJson("{\"a\":1,}") }
        assertFailsWith<JsonException> { parseJson("{a:1}") }
        assertFailsWith<JsonException> { parseJson("[1,2,]") }
    }

    @Test
    fun nanAndInfinityLiteralsAreRefused() {
        assertFailsWith<JsonException> { parseJson("{\"a\":NaN}") }
        assertFailsWith<JsonException> { parseJson("{\"a\":Infinity}") }
    }

    @Test
    fun aLargeOffsetSurvivesAsAnExactLong() {
        // Past 2^53, where the Double value would have rounded.
        val v = parseJson("{\"o\":9007199254740993}") as JsonObject
        assertEquals(9007199254740993L, (v["o"] as JsonNumber).asLong("o"))
    }

    @Test
    fun anIntPast32BitsIsRefusedAsAnInt() {
        val v = parseJson("{\"d\":5000000000}") as JsonObject
        val e = assertFailsWith<JsonException> { (v["d"] as JsonNumber).asInt("d") }
        assertTrue("32 bits" in e.message!!, e.message!!)
    }

    @Test
    fun nestedStructureAndLiteralsParse() {
        val v = parseJson("{\"a\":[1,-2,3.5e2,true,false,null,{\"b\":[]}]}") as JsonObject
        val a = v.arr("a")
        assertEquals(7, a.size)
        assertEquals(350.0, (a[2] as JsonNumber).value)
        assertEquals(JsonBool(true), a[3])
        assertEquals(JsonNull, a[5])
        assertEquals(0, (a[6] as JsonObject).arr("b").size)
    }

    @Test
    fun missingFieldsAreNamed() {
        val v = parseJson("{\"dtype\":\"F32\"}") as JsonObject
        val e = assertFailsWith<JsonException> { v.arr("shape") }
        assertTrue("shape" in e.message!!, e.message!!)
    }
}
