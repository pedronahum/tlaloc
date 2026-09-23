package io.tlaloc.core.io

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.bf16BitsToFloat
import io.tlaloc.core.floatToBf16Bits
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

private class Builder {
        private val entries = ArrayList<Triple<String, String, List<Int>>>()
        private val payloads = ArrayList<ByteArray>()
        private var meta: String? = null

        fun add(name: String, dtype: String, dims: List<Int>, bytes: ByteArray): Builder {
            entries.add(Triple(name, dtype, dims))
            payloads.add(bytes)
            return this
        }

        fun metadata(json: String): Builder { meta = json; return this }

        fun build(): ByteArray {
            val sb = StringBuilder("{")
            var first = true
            if (meta != null) { sb.append("\"__metadata__\":").append(meta); first = false }
            var off = 0L
            for ((i, e) in entries.withIndex()) {
                if (!first) sb.append(',')
                first = false
                val end = off + payloads[i].size
                sb.append("\"").append(e.first).append("\":{\"dtype\":\"").append(e.second)
                    .append("\",\"shape\":[").append(e.third.joinToString(","))
                    .append("],\"data_offsets\":[").append(off).append(',').append(end).append("]}")
                off = end
            }
            sb.append('}')
            return assemble(sb.toString(), payloads.fold(ByteArray(0)) { a, b -> a + b })
        }
}

private fun assemble(headerJson: String, data: ByteArray): ByteArray {
        val h = headerJson.encodeToByteArray()
        val out = ByteArray(8 + h.size + data.size)
        var n = h.size.toLong()
        for (i in 0 until 8) { out[i] = (n and 0xFF).toByte(); n = n ushr 8 }
        h.copyInto(out, 8)
        data.copyInto(out, 8 + h.size)
        return out
    }

private fun f32Bytes(vararg v: Float): ByteArray {
        val out = ByteArray(v.size * 4)
        for ((i, x) in v.withIndex()) {
            var b = x.toRawBits()
            for (j in 0 until 4) { out[i * 4 + j] = (b and 0xFF).toByte(); b = b ushr 8 }
        }
        return out
    }

private fun i32Bytes(vararg v: Int): ByteArray {
        val out = ByteArray(v.size * 4)
        for ((i, x) in v.withIndex()) {
            var b = x
            for (j in 0 until 4) { out[i * 4 + j] = (b and 0xFF).toByte(); b = b ushr 8 }
        }
        return out
    }

private fun bf16Bytes(vararg v: Float): ByteArray {
        val out = ByteArray(v.size * 2)
        for ((i, x) in v.withIndex()) {
            val b = floatToBf16Bits(x).toInt()
            out[i * 2] = (b and 0xFF).toByte()
            out[i * 2 + 1] = ((b ushr 8) and 0xFF).toByte()
        }
        return out
    }

/**
 * §0.4.468 (Phase H2) — the safetensors reader against HAND-BUILT BYTES.
 *
 * These tests own the FORMAT claim: the byte layout, the little-endian
 * conventions, the dtype table and every refusal. They deliberately build
 * their files byte by byte rather than going through a writer of ours — a
 * reader tested only against its own writer certifies that the two agree,
 * not that either matches the format. The agreement with the REAL producer
 * (the Python `safetensors` library) is pinned separately, in
 * `benchmarks/.../LlamaSafetensorsParityTest`.
 */
class SafetensorsTest {

    // ---- the happy path ------------------------------------------------

    @Test
    fun readsMixedDtypesInRowMajorOrder() {
        val file = Builder()
            .metadata("{\"format\":\"pt\",\"note\":\"h2\"}")
            .add("w", "F32", listOf(2, 3), f32Bytes(1f, 2f, 3f, 4f, 5f, 6f))
            .add("g", "BF16", listOf(4), bf16Bytes(1.5f, -2.25f, 0f, 1e30f))
            .add("ids", "I32", listOf(2, 2), i32Bytes(7, -9, 2147483647, -2147483648))
            .add("eps", "F32", listOf(), f32Bytes(1e-6f))
            .build()

        val t = Safetensors.readAll(file)
        assertEquals(setOf("w", "g", "ids", "eps"), t.keys)

        val w = t.getValue("w")
        assertEquals(F32, w.dtype)
        assertEquals(listOf(2, 3), w.dims.toList())
        assertTrue(w.storage is HostF32Storage)
        assertEquals(listOf(1f, 2f, 3f, 4f, 5f, 6f), w.toF32Array().toList())

        val g = t.getValue("g")
        assertEquals(BF16, g.dtype)
        val gStore = g.storage
        assertTrue(gStore is HostBf16Storage)
        // bf16 storage carries RAW PATTERNS: compare bit for bit, then widen.
        assertEquals(
            listOf(floatToBf16Bits(1.5f), floatToBf16Bits(-2.25f), floatToBf16Bits(0f), floatToBf16Bits(1e30f)),
            gStore.data.toList(),
        )
        assertEquals(1.5f, g.toF32Array()[0])
        assertEquals(-2.25f, g.toF32Array()[1])

        val ids = t.getValue("ids")
        assertEquals(I32, ids.dtype)
        assertEquals(listOf(7, -9, 2147483647, -2147483648), (ids.storage as HostI32Storage).data.toList())

        val eps = t.getValue("eps")
        assertEquals(0, eps.dims.size)
        assertEquals(1, eps.size)
        assertEquals(1e-6f, eps.toF32Array()[0])
    }

    @Test
    fun metadataIsReadAndIsNotATensor() {
        val file = Builder()
            .metadata("{\"format\":\"pt\"}")
            .add("w", "F32", listOf(1), f32Bytes(3f))
            .build()
        val n = Safetensors.headerLength(file)
        val json = file.decodeToString(8, (8 + n).toInt())
        val h = Safetensors.parseHeader(json, 8 + n, (file.size - 8 - n))
        assertEquals(mapOf("format" to "pt"), h.metadata)
        assertEquals(setOf("w"), h.names)
    }

    @Test
    fun headerLengthIsLittleEndianU64() {
        // 0x0102 little-endian = 258.
        val prefix = byteArrayOf(2, 1, 0, 0, 0, 0, 0, 0)
        assertEquals(258L, Safetensors.headerLength(prefix))
    }

    @Test
    fun anEmptyTensorIsLegalAndOccupiesNoBytes() {
        val file = Builder()
            .add("empty", "F32", listOf(0, 4), ByteArray(0))
            .add("w", "F32", listOf(1), f32Bytes(5f))
            .build()
        val t = Safetensors.readAll(file)
        assertEquals(0, t.getValue("empty").toF32Array().size)
        assertEquals(5f, t.getValue("w").toF32Array()[0])
    }

    @Test
    fun signedZeroSurvivesTheRoundTrip() {
        val file = Builder().add("z", "F32", listOf(2), f32Bytes(-0.0f, 0.0f)).build()
        val v = Safetensors.readAll(file).getValue("z").toF32Array()
        // The house rule: pin signed zero on the raw bits, never on `== 0f`.
        assertEquals(Int.MIN_VALUE, v[0].toRawBits())
        assertEquals(0, v[1].toRawBits())
    }

    // ---- the refusals --------------------------------------------------

    @Test
    fun fp16IsRefusedByName() {
        val e = assertFailsWith<JsonException> { Safetensors.mapDType("F16") }
        assertTrue("F16" in e.message!! && "bf16" in e.message!!, e.message!!)
    }

    @Test
    fun i64IsRefusedByNameAndNotNarrowed() {
        val e = assertFailsWith<JsonException> { Safetensors.mapDType("I64") }
        assertTrue("HostI64Storage" in e.message!!, e.message!!)
    }

    @Test
    fun fp8IsRefusedByName() {
        val e = assertFailsWith<JsonException> { Safetensors.mapDType("F8_E4M3") }
        assertTrue("fp8 checkpoints are not supported" in e.message!!, e.message!!)
    }

    @Test
    fun unknownDtypeIsRefused() {
        assertFailsWith<JsonException> { Safetensors.mapDType("FLOAT32") }
    }

    @Test
    fun byteLengthDisagreeingWithShapeIsRefused() {
        // 6 f32 elements claimed, 5 elements' worth of bytes given.
        val header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,20]}}"
        val e = assertFailsWith<JsonException> { Safetensors.parseHeader(header, 0, 20) }
        assertTrue("24 bytes" in e.message!! && "20 bytes" in e.message!!, e.message!!)
    }

    @Test
    fun overlappingTensorsAreRefused() {
        val header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}," +
            "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}"
        val e = assertFailsWith<JsonException> { Safetensors.parseHeader(header, 0, 12) }
        assertTrue("overlaps" in e.message!!, e.message!!)
    }

    @Test
    fun aRangePastTheBufferIsRefused() {
        val header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}"
        val e = assertFailsWith<JsonException> { Safetensors.parseHeader(header, 0, 8) }
        assertTrue("only 8 bytes" in e.message!!, e.message!!)
    }

    @Test
    fun backwardsRangeIsRefused() {
        val header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,4]}}"
        assertFailsWith<JsonException> { Safetensors.parseHeader(header, 0, 16) }
    }

    @Test
    fun negativeDimIsRefused() {
        val header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[-1,4],\"data_offsets\":[0,0]}}"
        val e = assertFailsWith<JsonException> { Safetensors.parseHeader(header, 0, 0) }
        assertTrue("negative dim" in e.message!!, e.message!!)
    }

    @Test
    fun aFractionalShapeEntryIsNotAnInteger() {
        val header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[4.0],\"data_offsets\":[0,16]}}"
        val e = assertFailsWith<JsonException> { Safetensors.parseHeader(header, 0, 16) }
        assertTrue("integer literal" in e.message!!, e.message!!)
    }

    @Test
    fun anAbsurdHeaderLengthIsRefusedBeforeAllocating() {
        val prefix = ByteArray(8) { 0xFF.toByte() }  // u64 max, read as -1L
        val e = assertFailsWith<JsonException> { Safetensors.headerLength(prefix) }
        assertTrue("not a safetensors file" in e.message!!, e.message!!)
    }

    @Test
    fun aTruncatedFileIsRefused() {
        val file = Builder().add("w", "F32", listOf(4), f32Bytes(1f, 2f, 3f, 4f)).build()
        val chopped = file.copyOfRange(0, file.size - 5)
        val e = assertFailsWith<JsonException> { Safetensors.readAll(chopped) }
        assertTrue("data buffer" in e.message!! || "truncated" in e.message!!, e.message!!)
    }

    @Test
    fun missingTensorNamesWhatTheFileActuallyHolds() {
        val file = Builder().add("w", "F32", listOf(1), f32Bytes(1f)).build()
        val n = Safetensors.headerLength(file)
        val h = Safetensors.parseHeader(file.decodeToString(8, (8 + n).toInt()), 8 + n, -1)
        val e = assertFailsWith<JsonException> { h.entry("q_proj") }
        assertTrue("q_proj" in e.message!! && "w" in e.message!!, e.message!!)
    }

    @Test
    fun theWrongTypedAccessorRefusesRatherThanReinterpreting() {
        val file = Builder().add("g", "BF16", listOf(2), bf16Bytes(1f, 2f)).build()
        val g = Safetensors.readAll(file).getValue("g")
        val e = assertFailsWith<JsonException> { g.asF32<io.tlaloc.core.ScalarShape>() }
        assertTrue("is bf16, not f32" in e.message!!, e.message!!)
    }

    @Test
    fun toF32ArrayRefusesAnIntegerTensor() {
        val file = Builder().add("ids", "I32", listOf(2), i32Bytes(1, 2)).build()
        val ids = Safetensors.readAll(file).getValue("ids")
        val e = assertFailsWith<JsonException> { ids.toF32Array() }
        assertTrue("asI32()" in e.message!!, e.message!!)
    }

    @Test
    fun bf16WideningIsExactAcrossTheExponentRange() {
        // Every bf16 pattern widens to an f32 that narrows back to itself.
        val patterns = FloatArray(256) { bf16BitsToFloat(((it shl 7) or 0x0041).toShort()) }
        val file = Builder().add("g", "BF16", listOf(256), bf16Bytes(*patterns)).build()
        val got = Safetensors.readAll(file).getValue("g").toF32Array()
        for (i in patterns.indices) {
            if (patterns[i].isNaN()) continue
            assertEquals(patterns[i].toRawBits(), got[i].toRawBits(), "element $i")
        }
    }
}
