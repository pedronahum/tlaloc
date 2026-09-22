package io.tlaloc.core.io

import io.tlaloc.core.BF16
import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.core.floatArrayToBf16Bits
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.502 — the writer's own certification. The ORACLE here is the reader
 * that already existed (§0.4.468) and was certified independently against
 * bytes PyTorch wrote: writing through [SafetensorsWriter] and reading back
 * through [Safetensors] exercises two implementations of the format that
 * were written five months apart and share no code. The cross-IMPLEMENTATION
 * half — the reference Python `safetensors` library on both sides of the
 * exchange — is `SafetensorsWriterOracleTest` in jvmTest, which self-skips
 * when the oracle venv is absent.
 */
class SafetensorsWriterTest {

    private fun f32(name: String, dims: IntArray, vararg v: Float) =
        SafetensorsTensor(name, F32, dims, HostF32Storage(v))

    private fun headerOf(file: ByteArray): String {
        val n = Safetensors.headerLength(file).toInt()
        return file.decodeToString(8, 8 + n)
    }

    // ---- round trip, per dtype ------------------------------------------

    @Test
    fun everyWritableDtypeSurvivesAWriteAndARead() {
        val floats = floatArrayOf(1f, -2.5f, 3.25e12f, 0f)
        val file = SafetensorsWriter.encode(
            listOf(
                SafetensorsTensor("a.f32", F32, intArrayOf(2, 2), HostF32Storage(floats)),
                SafetensorsTensor("b.f64", F64, intArrayOf(3), HostF64Storage(doubleArrayOf(1.0, -0.5, 1e300))),
                SafetensorsTensor("c.i32", I32, intArrayOf(2), HostI32Storage(intArrayOf(-7, 2_000_000_000))),
                SafetensorsTensor(
                    "d.bf16",
                    BF16,
                    intArrayOf(4),
                    HostBf16Storage(floatArrayToBf16Bits(floatArrayOf(1f, -2f, 0.5f, 100f))),
                ),
            ),
        )
        val back = Safetensors.readAll(file)
        assertEquals(setOf("a.f32", "b.f64", "c.i32", "d.bf16"), back.keys)

        val a = back.getValue("a.f32")
        assertEquals(F32, a.dtype)
        assertContentEquals(intArrayOf(2, 2), a.dims)
        assertContentEquals(floats, (a.storage as HostF32Storage).data)

        val b = back.getValue("b.f64")
        assertEquals(F64, b.dtype)
        assertContentEquals(doubleArrayOf(1.0, -0.5, 1e300), (b.storage as HostF64Storage).data)

        val c = back.getValue("c.i32")
        assertEquals(I32, c.dtype)
        assertContentEquals(intArrayOf(-7, 2_000_000_000), (c.storage as HostI32Storage).data)

        // bf16 is certified on its BIT PATTERNS, not on widened floats: the
        // storage carries raw upper-16-bit buckets, and a writer that
        // widened-then-narrowed would pass a float comparison while having
        // rounded twice.
        val d = back.getValue("d.bf16")
        assertEquals(BF16, d.dtype)
        assertContentEquals(
            floatArrayToBf16Bits(floatArrayOf(1f, -2f, 0.5f, 100f)),
            (d.storage as HostBf16Storage).data,
        )
    }

    @Test
    fun signedZeroAndNaNAndInfinityKeepTheirExactBits() {
        // Not a float comparison: -0f == 0f and NaN != NaN, so this asserts
        // raw bits. A writer that normalised either would be lossy in a way
        // no `==` test would catch.
        val v = floatArrayOf(-0.0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)
        val back = Safetensors.readAll(SafetensorsWriter.encode(listOf(f32("t", intArrayOf(4), *v))))
        val got = (back.getValue("t").storage as HostF32Storage).data
        for (i in v.indices) assertEquals(v[i].toRawBits(), got[i].toRawBits(), "element $i")
    }

    @Test
    fun aRankZeroScalarRoundTrips() {
        val back = Safetensors.readAll(
            SafetensorsWriter.encode(listOf(f32("s", intArrayOf(), 42f))),
        )
        val s = back.getValue("s")
        assertEquals(0, s.dims.size)
        assertEquals(1, s.size)
        assertContentEquals(floatArrayOf(42f), (s.storage as HostF32Storage).data)
    }

    @Test
    fun aZeroElementTensorRoundTripsAndOccupiesNoBytes() {
        val file = SafetensorsWriter.encode(
            listOf(
                SafetensorsTensor("empty", F32, intArrayOf(0, 4), HostF32Storage(FloatArray(0))),
                f32("after", intArrayOf(1), 1f),
            ),
        )
        val back = Safetensors.readAll(file)
        assertContentEquals(intArrayOf(0, 4), back.getValue("empty").dims)
        assertEquals(0, (back.getValue("empty").storage as HostF32Storage).data.size)
        assertContentEquals(floatArrayOf(1f), (back.getValue("after").storage as HostF32Storage).data)
    }

    @Test
    fun metadataRoundTripsAndIsNotATensor() {
        val file = SafetensorsWriter.encode(
            listOf(f32("w", intArrayOf(1), 1f)),
            mapOf("format" to "tlaloc", "steps" to "600"),
        )
        val n = Safetensors.headerLength(file)
        val header = Safetensors.parseHeader(headerOf(file), 8 + n, file.size - 8 - n)
        assertEquals(mapOf("format" to "tlaloc", "steps" to "600"), header.metadata)
        assertEquals(setOf("w"), header.names)
    }

    // ---- the format's own invariants -------------------------------------

    @Test
    fun theDataBufferStartsOnAnEightByteBoundaryWhateverTheHeaderLength() {
        // Header length varies with the name, so this sweeps a name length
        // through a full alignment period and then some.
        for (len in 1..20) {
            val file = SafetensorsWriter.encode(listOf(f32("n".repeat(len), intArrayOf(1), 1f)))
            val n = Safetensors.headerLength(file)
            assertEquals(0L, (8 + n) % 8, "name length $len: data starts at ${8 + n}")
            val header = headerOf(file)
            assertTrue(header.endsWith("}") || header.trimEnd(' ').endsWith("}"), header)
            // The padding is ASCII SPACE, which is what the reference
            // implementation writes and what JSON tolerates as trailing
            // whitespace; anything else (NUL is the tempting choice) makes
            // the header unparseable by a strict reader.
            assertTrue(header.substring(header.trimEnd(' ').length).all { it == ' ' }, header)
            assertEquals(1, Safetensors.readAll(file).size)
        }
    }

    @Test
    fun everyTensorLandsOnAnOffsetAlignedToItsOwnElementWidth() {
        // This is the whole reason tensors are emitted widest-dtype-first: a
        // reader that builds a zero-copy view (torch.frombuffer,
        // np.ndarray(buffer=...)) needs it, and name order would not give it.
        val file = SafetensorsWriter.encode(
            listOf(
                SafetensorsTensor("z.bf16", BF16, intArrayOf(3), HostBf16Storage(ShortArray(3))),
                SafetensorsTensor("y.f64", F64, intArrayOf(2), HostF64Storage(DoubleArray(2))),
                SafetensorsTensor("x.i32", I32, intArrayOf(1), HostI32Storage(IntArray(1))),
                SafetensorsTensor("w.f32", F32, intArrayOf(5), HostF32Storage(FloatArray(5))),
            ),
        )
        val n = Safetensors.headerLength(file)
        val header = Safetensors.parseHeader(headerOf(file), 8 + n, file.size - 8 - n)
        for (e in header.entries.values) {
            val width = Safetensors.mapDType(e.wireDType).sizeBytes
            assertEquals(0L, (header.dataStart + e.begin) % width, "${e.name} at ${e.begin}")
        }
        // ... and the order in the header is the order that produces it.
        assertEquals(listOf("y.f64", "w.f32", "x.i32", "z.bf16"), header.entries.keys.toList())
    }

    @Test
    fun encodeIsAPureFunctionOfItsInputSoACheckpointIsHashable() {
        val a = listOf(
            f32("b", intArrayOf(2), 1f, 2f),
            f32("a", intArrayOf(1), 3f),
            SafetensorsTensor("c", F64, intArrayOf(1), HostF64Storage(doubleArrayOf(4.0))),
        )
        val shuffled = listOf(a[2], a[0], a[1])
        val one = SafetensorsWriter.encode(a, mapOf("k" to "v", "j" to "w"))
        val two = SafetensorsWriter.encode(shuffled, mapOf("j" to "w", "k" to "v"))
        assertContentEquals(one, two)
        assertContentEquals(one, SafetensorsWriter.encode(a, mapOf("k" to "v", "j" to "w")))
    }

    @Test
    fun aTensorNameCarryingJsonMetacharactersSurvives() {
        // The reason `jsonQuote` exists. A hand-concatenated header would
        // produce a file its own reader cannot parse — and tensor names are
        // caller data.
        val names = listOf(
            "model.layers.0.self_attn.q_proj.weight",
            "a\"quoted\"name",
            "back\\slash",
            "new\nline\ttab",
            "unicode-αβγ",
            "ctrl\u0001char",
        )
        val file = SafetensorsWriter.encode(
            names.mapIndexed { i, nm -> f32(nm, intArrayOf(1), i.toFloat()) },
        )
        val back = Safetensors.readAll(file)
        assertEquals(names.toSet(), back.keys)
        for ((i, nm) in names.withIndex()) {
            assertContentEquals(
                floatArrayOf(i.toFloat()),
                (back.getValue(nm).storage as HostF32Storage).data,
            )
        }
    }

    // ---- refusals, by name -----------------------------------------------

    @Test
    fun theReservedMetadataKeyCannotNameATensor() {
        val e = assertFailsWith<JsonException> {
            SafetensorsWriter.encode(listOf(f32("__metadata__", intArrayOf(1), 1f)))
        }
        assertTrue("__metadata__" in e.message!! && "reserved" in e.message!!, e.message!!)
    }

    @Test
    fun aDuplicateTensorNameIsRefusedByName() {
        val e = assertFailsWith<JsonException> {
            SafetensorsWriter.encode(listOf(f32("w", intArrayOf(1), 1f), f32("w", intArrayOf(1), 2f)))
        }
        assertTrue("duplicate" in e.message!! && "'w'" in e.message!!, e.message!!)
    }

    @Test
    fun aNegativeDimIsRefusedByName() {
        val e = assertFailsWith<JsonException> {
            SafetensorsWriter.encode(listOf(f32("w", intArrayOf(-1, 2), 1f)))
        }
        assertTrue("negative dim" in e.message!! && "'w'" in e.message!!, e.message!!)
    }

    @Test
    fun aShapeThatDisagreesWithTheStorageLengthIsRefusedByName() {
        val e = assertFailsWith<JsonException> {
            SafetensorsWriter.encode(listOf(f32("w", intArrayOf(3, 3), 1f, 2f)))
        }
        assertTrue("9 elements" in e.message!! && "storage holds 2" in e.message!!, e.message!!)
    }

    @Test
    fun aStorageClassThatDisagreesWithTheDtypeIsRefusedByName() {
        val e = assertFailsWith<JsonException> {
            SafetensorsWriter.encode(
                listOf(SafetensorsTensor("w", BF16, intArrayOf(2), HostF32Storage(floatArrayOf(1f, 2f)))),
            )
        }
        assertTrue("HostBf16Storage" in e.message!! && "HostF32Storage" in e.message!!, e.message!!)
    }

    @Test
    fun theDtypesWithNoHostStorageAreRefusedByName() {
        val i64 = assertFailsWith<JsonException> { SafetensorsWriter.wireDType(I64) }
        assertTrue("HostI64Storage" in i64.message!!, i64.message!!)
        val bool = assertFailsWith<JsonException> { SafetensorsWriter.wireDType(Bool) }
        assertTrue("bool" in bool.message!! && "no host storage" in bool.message!!, bool.message!!)
        // The writer's table and the reader's table agree about what a
        // Tlaloc checkpoint may contain — that agreement is the claim.
        for (dt in listOf(F32, F64, BF16, I32)) {
            assertEquals(dt, Safetensors.mapDType(SafetensorsWriter.wireDType(dt)))
        }
        assertTrue("F16 DType" in i64.message!!, i64.message!!)
    }
}
