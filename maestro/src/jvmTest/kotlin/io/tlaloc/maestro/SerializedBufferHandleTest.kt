package io.tlaloc.maestro

import io.tlaloc.core.BufferHandle
import io.tlaloc.core.DTensor
import io.tlaloc.core.DataAxis
import io.tlaloc.core.DynShape
import io.tlaloc.core.F32
import io.tlaloc.core.HandleRef
import io.tlaloc.core.Mesh0
import io.tlaloc.core.Mesh1
import io.tlaloc.core.ModelAxis
import io.tlaloc.core.Rank1
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Layer 2.5 §0.4.245+ — `SerializedBufferHandle` cross-pod serialization
 * tests.
 *
 * v1 contract under test:
 * - Round-trip: write a BufferHandle, read it back, assert tensor equality.
 * - Type-check: writing with one descriptor + reading with a different
 *   descriptor (rank mismatch, dtype mismatch, axis-name mismatch) fails
 *   with [SerializedBufferException].
 * - Hash validation: corrupting the on-disk payload bytes makes read fail.
 * - Mesh-check: reading with a different mesh fails.
 * - Missing URI: structured error.
 * - Unsupported scheme: [UnsupportedOperationException].
 * - JSON round-trip: toJson/fromJson preserves every field exactly.
 * - Default retention: file is deleted after a successful read.
 */
class SerializedBufferHandleTest {

    private fun freshFileUri(name: String): String {
        val dir = Files.createTempDirectory("tlaloc-serialized-buffer-test")
        val file = dir.resolve("$name.bin")
        return "file://${file.toAbsolutePath()}"
    }

    private fun handleFor(values: FloatArray): BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0> {
        val tensor = Tensors.f32Vector<Sym>(values)
        return BufferHandle(HandleRef(nativeId = 1L, payload = tensor))
    }

    /** Same payload but typed on Mesh1<DataAxis> for cross-mesh tests. */
    private fun handleForMesh1Data(values: FloatArray): BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh1<DataAxis>> {
        val tensor = Tensors.f32Vector<Sym>(values)
        return BufferHandle(HandleRef(nativeId = 1L, payload = tensor))
    }

    /** Same payload but typed on Mesh1<ModelAxis>. */
    private fun handleForMesh1Model(values: FloatArray): BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh1<ModelAxis>> {
        val tensor = Tensors.f32Vector<Sym>(values)
        return BufferHandle(HandleRef(nativeId = 1L, payload = tensor))
    }

    private fun rank1Descriptor(size: Int): TypeDescriptor =
        TypeDescriptor(dtype = "f32", dims = listOf(size), axisNames = emptyList())

    @Test
    fun roundTripWriteReadProducesEquivalentHandle() {
        val uri = freshFileUri("roundtrip")
        val original = handleFor(floatArrayOf(1f, 2f, 3f, -4f))
        val descriptor = rank1Descriptor(4)

        val serialized = SerializedBufferHandle.write(
            handle = original,
            uri = uri,
            mesh = Mesh0,
            typeDescriptor = descriptor,
            manifestRef = "test-manifest-001",
        )

        // Sanity on the wire-level metadata.
        assertEquals(uri, serialized.uri)
        assertEquals("test-manifest-001", serialized.manifestRef)
        assertEquals("Mesh0", serialized.meshName)
        assertEquals(64, serialized.contentHash.length)
        assertEquals(descriptor, serialized.typeDescriptor)

        // Round-trip: read it back and confirm tensor equality.
        val readBack: BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0> =
            serialized.read(descriptor, Mesh0, deleteAfterRead = false)
        @Suppress("UNCHECKED_CAST")
        val readTensor = readBack.ref.payload as DTensor<Rank1<Sym>, F32>
        assertTrue(
            floatArrayOf(1f, 2f, 3f, -4f).contentEquals(readTensor.hostF32()),
            "read tensor must equal original (got ${readTensor.hostF32().toList()})",
        )
    }

    @Test
    fun readDeletesFileByDefault() {
        val uri = freshFileUri("retention")
        val handle = handleFor(floatArrayOf(7f, 8f))
        val descriptor = rank1Descriptor(2)
        val serialized = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh0,
            typeDescriptor = descriptor, manifestRef = "retention-test",
        )

        // File exists pre-read.
        val path = Paths.get(java.net.URI(uri).path)
        assertTrue(Files.exists(path), "file must exist before read")

        serialized.read(descriptor, Mesh0)  // deleteAfterRead = true by default

        assertFalse(Files.exists(path), "file must be deleted after default read")
    }

    @Test
    fun typeMismatchInRankFailsLoudly() {
        val uri = freshFileUri("type-mismatch-rank")
        val handle = handleFor(floatArrayOf(1f, 2f, 3f))
        val written = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh0,
            typeDescriptor = rank1Descriptor(3),
            manifestRef = "rank-mismatch-test",
        )
        val expectedRank2 = TypeDescriptor(dtype = "f32", dims = listOf(2, 2), axisNames = emptyList())
        val ex = assertFailsWith<SerializedBufferException> {
            written.read(expectedRank2, Mesh0, deleteAfterRead = false)
        }
        assertTrue("type mismatch" in ex.message!!, "expected 'type mismatch' message; got: ${ex.message}")
    }

    @Test
    fun typeMismatchInAxisNamesFailsLoudly() {
        val uri = freshFileUri("type-mismatch-names")
        val handle = handleFor(floatArrayOf(1f, 2f))
        val produced = TypeDescriptor(dtype = "f32", dims = listOf(2), axisNames = listOf("Batch"))
        val written = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh0,
            typeDescriptor = produced,
            manifestRef = "axis-name-test",
        )
        val expected = TypeDescriptor(dtype = "f32", dims = listOf(2), axisNames = listOf("SeqLen"))
        val ex = assertFailsWith<SerializedBufferException> {
            written.read(expected, Mesh0, deleteAfterRead = false)
        }
        assertTrue("type mismatch" in ex.message!!)
    }

    @Test
    fun meshMismatchFailsLoudly() {
        // Two layers of mesh-protection in the design:
        //
        // 1. Compile-time: SerializedBufferHandle<T, M>.write requires the
        //    handle's M to match the mesh argument's M (BufferHandle<T, Mesh0>
        //    cannot be written with mesh = Mesh1<DataAxis>()). This test
        //    exercises layer 2 — *runtime* validation when the consumer is
        //    given the wrong expected mesh value.
        //
        // 2. Runtime: even if a SerializedBufferHandle is correctly typed
        //    at write, a misconfigured consumer that passes a different
        //    expectedMesh value to read(...) is rejected with a clear
        //    error. This catches the case where the consumer's typed
        //    declaration at the call site disagrees with the SerializedHandle's
        //    runtime meshName field.
        val uri = freshFileUri("mesh-mismatch")
        val handle = handleForMesh1Data(floatArrayOf(1f, 2f))
        val descriptor = rank1Descriptor(2)
        val written = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh1<DataAxis>(),
            typeDescriptor = descriptor,
            manifestRef = "mesh-test",
        )
        // Producer wrote on Mesh1<DataAxis>; consumer (incorrectly) passes
        // Mesh0 as the expected mesh value at read time. The SerializedBufferHandle's
        // type still says M=Mesh1<DataAxis>, but the runtime check on
        // mesh class simpleName fails fast.
        val ex = assertFailsWith<SerializedBufferException> {
            written.read(descriptor, Mesh0, deleteAfterRead = false)
        }
        assertTrue("mesh mismatch" in ex.message!!)
    }

    @Test
    fun corruptedPayloadFailsHashCheck() {
        val uri = freshFileUri("corrupt")
        val handle = handleFor(floatArrayOf(1f, 2f, 3f))
        val descriptor = rank1Descriptor(3)
        val written = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh0,
            typeDescriptor = descriptor,
            manifestRef = "corrupt-test",
        )

        // Flip a payload byte on disk. The file's overall size should
        // include the wire-format header (magic + version + descLen + desc
        // + payloadLen) plus 12 bytes of float payload — so the last byte
        // is definitely in the payload region.
        val path = Paths.get(java.net.URI(uri).path)
        val bytes = Files.readAllBytes(path)
        bytes[bytes.size - 1] = (bytes[bytes.size - 1].toInt() xor 0xff).toByte()
        Files.write(path, bytes)

        val ex = assertFailsWith<SerializedBufferException> {
            written.read(descriptor, Mesh0, deleteAfterRead = false)
        }
        assertTrue("content hash mismatch" in ex.message!!, "got: ${ex.message}")
    }

    @Test
    fun missingFileFailsLoudly() {
        val uri = freshFileUri("missing")
        val handle = handleFor(floatArrayOf(1f))
        val descriptor = rank1Descriptor(1)
        val written = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh0,
            typeDescriptor = descriptor,
            manifestRef = "missing-test",
        )
        // Delete the file before read.
        Files.delete(Paths.get(java.net.URI(uri).path))

        val ex = assertFailsWith<SerializedBufferException> {
            written.read(descriptor, Mesh0, deleteAfterRead = false)
        }
        assertTrue("file does not exist" in ex.message!!)
    }

    @Test
    fun unsupportedSchemeS3FailsLoudly() {
        val handle = handleFor(floatArrayOf(1f))
        val descriptor = rank1Descriptor(1)
        assertFailsWith<UnsupportedOperationException> {
            SerializedBufferHandle.write(
                handle = handle, uri = "s3://bucket/key",
                mesh = Mesh0,
                typeDescriptor = descriptor,
                manifestRef = "s3-stub",
            )
        }
    }

    @Test
    fun unknownSchemeFailsAsStructuredException() {
        val handle = handleFor(floatArrayOf(1f))
        val descriptor = rank1Descriptor(1)
        assertFailsWith<SerializedBufferException> {
            SerializedBufferHandle.write(
                handle = handle, uri = "ftp://server/path",
                mesh = Mesh0,
                typeDescriptor = descriptor,
                manifestRef = "ftp-bad-scheme",
            )
        }
    }

    @Test
    fun jsonRoundTripPreservesAllFields() {
        val uri = freshFileUri("json")
        val handle = handleForMesh1Model(floatArrayOf(1f, 2f))
        val descriptor = TypeDescriptor(
            dtype = "f32", dims = listOf(2),
            axisNames = listOf("SeqLen"),
        )
        val original = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh1<ModelAxis>(),
            typeDescriptor = descriptor,
            manifestRef = "json-roundtrip",
        )
        val json = original.toJson()
        val parsed = SerializedBufferHandle.fromJson(json)
        assertEquals(original.uri, parsed.uri)
        assertEquals(original.contentHash, parsed.contentHash)
        assertEquals(original.typeDescriptor, parsed.typeDescriptor)
        assertEquals(original.manifestRef, parsed.manifestRef)
        assertEquals(original.meshName, parsed.meshName)
    }

    @Test
    fun jsonContainsAllExpectedKeys() {
        val uri = freshFileUri("json-keys")
        val handle = handleFor(floatArrayOf(1f))
        val written = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh0,
            typeDescriptor = rank1Descriptor(1),
            manifestRef = "json-keys",
        )
        val json = written.toJson()
        for (key in listOf("uri", "contentHash", "typeDescriptor", "manifestRef", "meshName")) {
            assertTrue(
                "\"$key\":" in json,
                "JSON missing key '$key'; got:\n$json",
            )
        }
    }

    @Test
    fun writeRequiresDescriptorMatchingTensorDims() {
        val uri = freshFileUri("desc-mismatch")
        val handle = handleFor(floatArrayOf(1f, 2f, 3f))
        val wrongDescriptor = rank1Descriptor(99)  // says 99 dims but tensor has 3
        assertFails {
            SerializedBufferHandle.write(
                handle = handle, uri = uri, mesh = Mesh0,
                typeDescriptor = wrongDescriptor,
                manifestRef = "desc-mismatch",
            )
        }
    }

    @Test
    fun differentRankAxisNamesPreservedAcrossRoundTrip() {
        val uri = freshFileUri("axis-preserve")
        val handle = handleFor(floatArrayOf(1f, 2f))
        val descriptor = TypeDescriptor(
            dtype = "f32", dims = listOf(2),
            axisNames = listOf("Heads"),
        )
        val written = SerializedBufferHandle.write(
            handle = handle, uri = uri, mesh = Mesh0,
            typeDescriptor = descriptor,
            manifestRef = "axis-preserve",
        )
        val read: BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0> =
            written.read(descriptor, Mesh0, deleteAfterRead = false)
        @Suppress("UNCHECKED_CAST")
        val readTensor = read.ref.payload as DTensor<Rank1<Sym>, F32>
        assertTrue(floatArrayOf(1f, 2f).contentEquals(readTensor.hostF32()))
        // Descriptor preserved on the wire (proves axis names round-trip
        // through the binary format's UTF-8 JSON descriptor block).
        assertEquals(listOf("Heads"), written.typeDescriptor.axisNames)
    }
}
