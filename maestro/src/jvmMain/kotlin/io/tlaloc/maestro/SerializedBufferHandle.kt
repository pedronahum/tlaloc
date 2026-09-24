package io.tlaloc.maestro

import io.tlaloc.core.BufferHandle
import io.tlaloc.core.DTensor
import io.tlaloc.core.DynShape
import io.tlaloc.core.F32
import io.tlaloc.core.HandleRef
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Mesh
import io.tlaloc.core.hostF32
import java.net.URI
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Paths

/**
 * Typed cross-pod buffer-handle serialization.
 *
 * [BufferHandle] is in-process and references the materialized
 * [DTensor] via [HandleRef.payload]. To cross a step boundary in a real
 * Maestro deployment the handle's payload must be serialized to a content-
 * addressed object store and re-read on the consumer side with both
 * type and integrity validation.
 *
 * `SerializedBufferHandle` is the on-the-wire form:
 *
 * - **`uri`**: `file://` for tests; `s3://` and `gs://` are stubbed in v1 and
 *   throw [UnsupportedOperationException].
 * - **`contentHash`**: hex SHA-256 over the *payload bytes only*. The file
 *   header (magic, version, descriptor) is provenance metadata and not part
 *   of the integrity hash — a bit-flip in the descriptor JSON surfaces via
 *   the per-field structural validation, while a bit-flip in payload bytes
 *   surfaces via this hash.
 * - **`typeDescriptor`**: shape/dtype/axisNames — the producer's `TypeDescriptor`
 *   from its [ProgramManifest]. Consumer reads with an `expectedType` and
 *   any mismatch (rank, dtype, dims, axis names) fails loudly.
 * - **`manifestRef`**: the producer step's `manifest.bodyHash` — lineage
 *   pointer. It is stored for provenance; consumer validation does not require it.
 * - **`meshName`**: phantom-type-erased name of the mesh placement
 *   (`Mesh1`, `Mesh2`, etc.). Consumer validates against the expected mesh's
 *   class simple name.
 *
 * # Wire format
 *
 * Binary, little-endian, no padding:
 *
 * ```
 * [MAGIC "TLAL" 4B] [VERSION u32 = 1] [DESC_LEN u32] [DESC bytes UTF-8 JSON]
 *                                     [PAYLOAD_LEN u32] [PAYLOAD bytes F32 little-endian]
 * ```
 *
 * Future versions (e.g. supporting non-F32 dtypes) bump `VERSION`. v1
 * supports F32 only.
 *
 * # Lifecycle
 *
 * Default retention: `read(deleteAfterRead = true)` removes the on-disk
 * file once the consumer has materialized the BufferHandle. Tests use
 * `deleteAfterRead = false` for round-trip assertions.
 */
class SerializedBufferHandle<T : DTensor<*, *>, M : Mesh>(
    val uri: String,
    val contentHash: String,
    val typeDescriptor: TypeDescriptor,
    val manifestRef: String,
    val meshName: String,
) {

    init {
        require(contentHash.length == 64) {
            "contentHash must be a 64-char hex SHA-256 (got ${contentHash.length} chars: $contentHash)"
        }
        require(uri.isNotEmpty()) { "uri must be non-empty" }
        require(manifestRef.isNotEmpty()) { "manifestRef must be non-empty" }
        require(meshName.isNotEmpty()) { "meshName must be non-empty" }
    }

    /**
     * Materialize this serialized handle as an in-process [BufferHandle].
     * Validates the on-disk content against [contentHash], the on-disk
     * descriptor against [expectedType], and the carried [meshName]
     * against [expectedMesh]'s class name.
     *
     * Throws [SerializedBufferException] on any validation failure
     * (hash mismatch, type mismatch, mesh mismatch, missing URI,
     * unsupported scheme, malformed wire format).
     *
     * @param expectedType the consumer's expected [TypeDescriptor].
     *   Mismatches in dtype, dims, or axisNames fail loudly.
     * @param expectedMesh the consumer's expected [Mesh]. Mismatch on the
     *   class simple name (e.g. `Mesh0` vs `Mesh1`) fails loudly.
     * @param deleteAfterRead if true (default), the on-disk file is removed
     *   once successfully read. Tests pass false to enable round-trip
     *   assertions.
     */
    fun read(
        expectedType: TypeDescriptor,
        expectedMesh: Mesh,
        deleteAfterRead: Boolean = true,
    ): BufferHandle<T, M> {
        // 1. Mesh check (cheapest; before any IO).
        val expectedMeshName = expectedMesh::class.simpleName
            ?: error("expected mesh class has no simple name: ${expectedMesh::class}")
        if (meshName != expectedMeshName) {
            throw SerializedBufferException(
                "mesh mismatch: serialized handle on '$meshName' but consumer expects '$expectedMeshName' (uri=$uri)",
            )
        }

        // 2. Type-descriptor structural check.
        if (typeDescriptor != expectedType) {
            throw SerializedBufferException(
                "type mismatch: serialized handle has type $typeDescriptor but consumer expects $expectedType (uri=$uri)",
            )
        }

        // 3. Read bytes, parse wire format, validate hash + on-disk descriptor.
        val bytes = readScheme(uri)
        val (parsedDescriptor, payload, parsedContentHash) = parseWireFormat(bytes)
        if (parsedDescriptor != typeDescriptor) {
            throw SerializedBufferException(
                "on-disk descriptor diverges from header descriptor: on-disk=$parsedDescriptor, header=$typeDescriptor (uri=$uri)",
            )
        }
        if (parsedContentHash != contentHash) {
            throw SerializedBufferException(
                "content hash mismatch: expected $contentHash, got $parsedContentHash (uri=$uri)",
            )
        }

        // 4. Reconstruct DTensor + BufferHandle. DTensor's shape type
        //    parameter is phantom — using DynShape just to satisfy the bound;
        //    BufferHandle's T phantom carries the user-facing typed shape.
        val dims = typeDescriptor.dims.toIntArray()
        val tensor: DTensor<DynShape, F32> = DTensor(HostF32Storage(payload), dims, F32)
        val ref = HandleRef(nativeId = nextNativeId(), payload = tensor)

        // 5. Optional retention cleanup.
        if (deleteAfterRead) {
            cleanupScheme(uri)
        }

        return BufferHandle(ref)
    }

    /** JSON form for cross-pod passing through Maestro params. */
    fun toJson(): String = buildString {
        append("{")
        append("\"uri\":").append(jsonStringEscapeForBufferHandle(uri)).append(',')
        append("\"contentHash\":").append(jsonStringEscapeForBufferHandle(contentHash)).append(',')
        append("\"typeDescriptor\":").append(typeDescriptor.toJson()).append(',')
        append("\"manifestRef\":").append(jsonStringEscapeForBufferHandle(manifestRef)).append(',')
        append("\"meshName\":").append(jsonStringEscapeForBufferHandle(meshName))
        append("}")
    }

    override fun equals(other: Any?): Boolean = other is SerializedBufferHandle<*, *> &&
        uri == other.uri && contentHash == other.contentHash &&
        typeDescriptor == other.typeDescriptor && manifestRef == other.manifestRef &&
        meshName == other.meshName

    override fun hashCode(): Int =
        listOf(uri, contentHash, typeDescriptor, manifestRef, meshName).hashCode()

    override fun toString(): String =
        "SerializedBufferHandle(uri=$uri, contentHash=${contentHash.take(8)}…, " +
            "type=$typeDescriptor, mesh=$meshName)"

    companion object {
        const val MAGIC: Int = 0x4C414C54 // "TLAL" little-endian — 'T'=0x54 in low byte
        const val VERSION: Int = 1

        /**
         * Serialize [handle] (whose payload must be a [DTensor]<*, F32>) to
         * [uri] and return the typed [SerializedBufferHandle]. The [uri]'s
         * scheme determines the backing store:
         * - `file://path` — local filesystem.
         * - `s3://...` / `gs://...` — stubbed in v1 ([UnsupportedOperationException]).
         *
         * @param handle in-memory handle whose `ref.payload` is a [DTensor].
         * @param uri target URI; parent directory must exist for `file://`.
         * @param mesh phantom-typed mesh placement; the class simple name is
         *   stored in the serialized handle for consumer validation.
         * @param typeDescriptor descriptor for the buffer's type. Typically
         *   sourced from the producer step's `manifest.outputs[i]`.
         * @param manifestRef the producer step's `manifest.bodyHash`.
         */
        fun <T : DTensor<*, *>, M : Mesh> write(
            handle: BufferHandle<T, M>,
            uri: String,
            mesh: M,
            typeDescriptor: TypeDescriptor,
            manifestRef: String,
        ): SerializedBufferHandle<T, M> {
            val tensor = handle.ref.payload
                ?: throw SerializedBufferException(
                    "BufferHandle ref ${handle.ref.nativeId} has no payload — cannot serialize",
                )
            require(tensor is DTensor<*, *>) {
                "BufferHandle.ref.payload must be a DTensor, got ${tensor::class.simpleName}"
            }
            require(tensor.dtype == F32) {
                "v1 supports F32 payloads only, got ${tensor.dtype}"
            }
            // Validate descriptor matches the actual tensor.
            require(typeDescriptor.dtype.equals("f32", ignoreCase = true)) {
                "typeDescriptor.dtype must be F32 for v1 (got '${typeDescriptor.dtype}')"
            }
            require(typeDescriptor.dims == tensor.dims.toList()) {
                "typeDescriptor.dims ${typeDescriptor.dims} must match tensor.dims ${tensor.dims.toList()}"
            }

            // Serialize payload to little-endian F32 bytes. The require() check
            // above pinned tensor.dtype == F32 — refine the static type to the
            // F32 hostF32() extension's expected receiver shape.
            @Suppress("UNCHECKED_CAST")
            val floats = (tensor as DTensor<*, F32>).hostF32()
            val payload = ByteArray(floats.size * 4)
            val payloadBuf = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN)
            for (f in floats) payloadBuf.putFloat(f)

            val contentHash = sha256Hex(payload)
            val meshName = mesh::class.simpleName
                ?: error("mesh class has no simple name: ${mesh::class}")
            val bytes = encodeWireFormat(typeDescriptor, payload)
            writeScheme(uri, bytes)

            return SerializedBufferHandle(
                uri = uri,
                contentHash = contentHash,
                typeDescriptor = typeDescriptor,
                manifestRef = manifestRef,
                meshName = meshName,
            )
        }

        fun fromJson(json: String): SerializedBufferHandle<*, *> =
            SerializedBufferHandleJsonParser(json).parse()

        // ---- Internals -------------------------------------------------------

        private fun encodeWireFormat(descriptor: TypeDescriptor, payload: ByteArray): ByteArray {
            val descJson = descriptor.toJson().toByteArray(Charsets.UTF_8)
            val total = 4 + 4 + 4 + descJson.size + 4 + payload.size
            val out = ByteArray(total)
            val buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(MAGIC)
            buf.putInt(VERSION)
            buf.putInt(descJson.size)
            buf.put(descJson)
            buf.putInt(payload.size)
            buf.put(payload)
            return out
        }

        internal fun parseWireFormat(bytes: ByteArray): Triple<TypeDescriptor, FloatArray, String> {
            if (bytes.size < 16) {
                throw SerializedBufferException("file too short to be a wire-format buffer (${bytes.size} B)")
            }
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = buf.int
            if (magic != MAGIC) {
                throw SerializedBufferException(
                    "magic mismatch: expected ${"0x%08x".format(MAGIC)}, got ${"0x%08x".format(magic)}",
                )
            }
            val version = buf.int
            if (version != VERSION) {
                throw SerializedBufferException("version mismatch: expected $VERSION, got $version")
            }
            val descLen = buf.int
            require(descLen in 0..bytes.size) { "descLen $descLen out of range" }
            val descBytes = ByteArray(descLen)
            buf.get(descBytes)
            val descJson = descBytes.toString(Charsets.UTF_8)
            val descriptor = TypeDescriptor.fromJson(descJson)
            val payloadLen = buf.int
            require(payloadLen >= 0 && buf.remaining() >= payloadLen) {
                "payloadLen $payloadLen exceeds remaining buffer ${buf.remaining()}"
            }
            val payloadBytes = ByteArray(payloadLen)
            buf.get(payloadBytes)
            val contentHash = sha256Hex(payloadBytes)
            // Convert payload bytes → FloatArray (little-endian).
            val floats = FloatArray(payloadLen / 4)
            val pbuf = ByteBuffer.wrap(payloadBytes).order(ByteOrder.LITTLE_ENDIAN)
            for (i in floats.indices) floats[i] = pbuf.float
            return Triple(descriptor, floats, contentHash)
        }

        private fun writeScheme(uri: String, bytes: ByteArray) {
            val parsed = URI(uri)
            when (parsed.scheme) {
                "file" -> {
                    val path = Paths.get(parsed.path)
                    val parent = path.parent
                    if (parent != null && !Files.exists(parent)) {
                        Files.createDirectories(parent)
                    }
                    Files.write(path, bytes)
                }
                "s3", "gs" -> throw UnsupportedOperationException(
                    "v1 supports file:// only; '${parsed.scheme}://' stubbed for Layer 3+",
                )
                else -> throw SerializedBufferException(
                    "unsupported URI scheme '${parsed.scheme}' (got '$uri')",
                )
            }
        }

        private fun readScheme(uri: String): ByteArray {
            val parsed = try {
                URI(uri)
            } catch (e: Exception) {
                throw SerializedBufferException("malformed uri '$uri': ${e.message}")
            }
            return when (parsed.scheme) {
                "file" -> {
                    val path = Paths.get(parsed.path)
                    if (!Files.exists(path)) {
                        throw SerializedBufferException("file does not exist: $uri")
                    }
                    Files.readAllBytes(path)
                }
                "s3", "gs" -> throw UnsupportedOperationException(
                    "v1 supports file:// only; '${parsed.scheme}://' stubbed for Layer 3+",
                )
                else -> throw SerializedBufferException(
                    "unsupported URI scheme '${parsed.scheme}' (got '$uri')",
                )
            }
        }

        private fun cleanupScheme(uri: String) {
            val parsed = URI(uri)
            if (parsed.scheme == "file") {
                runCatching { Files.deleteIfExists(Paths.get(parsed.path)) }
            }
            // Other schemes: cleanup is the deployment's concern (lifecycle policy).
        }
    }
}

/**
 * Structured failure for [SerializedBufferHandle] operations. Distinct from
 * generic IO exceptions so callers (e.g. `TlalocRunner`) can pattern-match
 * type/hash/scheme violations vs. unrelated IO failures.
 */
class SerializedBufferException(message: String) : RuntimeException(message)

// Note: sha256Hex lives in Program.kt as internal — re-used here to avoid
// duplication. jsonStringEscape is file-private below (Program.kt + the
// other emitters use bare `jsonString` helpers; we keep ours scoped to
// avoid name collisions across the module).
private fun jsonStringEscapeForBufferHandle(s: String): String = buildString {
    append('"')
    for (c in s) {
        when (c) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (c.code < 0x20) append("\\u%04x".format(c.code)) else append(c)
        }
    }
    append('"')
}

/**
 * Targeted JSON parser for `SerializedBufferHandle.toJson()` output. Same
 * design philosophy as [ManifestJsonParser] — small, schema-specific, no
 * `kotlinx.serialization` dependency.
 */
internal class SerializedBufferHandleJsonParser(private val text: String) {
    private var pos = 0

    fun parse(): SerializedBufferHandle<*, *> {
        skipWs(); expect('{')
        var uri: String? = null
        var contentHash: String? = null
        var typeDescriptor: TypeDescriptor? = null
        var manifestRef: String? = null
        var meshName: String? = null
        while (true) {
            skipWs()
            if (peek() == '}') break
            val key = readString(); skipWs(); expect(':'); skipWs()
            when (key) {
                "uri" -> uri = readString()
                "contentHash" -> contentHash = readString()
                "typeDescriptor" -> typeDescriptor = readNestedTypeDescriptor()
                "manifestRef" -> manifestRef = readString()
                "meshName" -> meshName = readString()
                else -> error("unknown SerializedBufferHandle key: $key at pos $pos")
            }
            skipWs()
            if (peek() == ',') pos++ else break
        }
        skipWs(); expect('}')
        return SerializedBufferHandle<io.tlaloc.core.DTensor<*, *>, io.tlaloc.core.Mesh>(
            uri = requireNotNull(uri) { "missing 'uri'" },
            contentHash = requireNotNull(contentHash) { "missing 'contentHash'" },
            typeDescriptor = requireNotNull(typeDescriptor) { "missing 'typeDescriptor'" },
            manifestRef = requireNotNull(manifestRef) { "missing 'manifestRef'" },
            meshName = requireNotNull(meshName) { "missing 'meshName'" },
        )
    }

    private fun readNestedTypeDescriptor(): TypeDescriptor {
        // Locate the matching closing brace, slice the substring, delegate.
        skipWs()
        require(peek() == '{') { "expected '{' at pos $pos" }
        val start = pos
        var depth = 0
        while (pos < text.length) {
            when (text[pos]) {
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) { pos++; break }
                }
                '"' -> {
                    // Skip string content (including escaped quotes) so we
                    // don't accidentally count braces inside strings.
                    pos++
                    while (pos < text.length && text[pos] != '"') {
                        if (text[pos] == '\\') pos++
                        pos++
                    }
                }
            }
            pos++
        }
        val end = pos
        return TypeDescriptor.fromJson(text.substring(start, end))
    }

    private fun readString(): String {
        skipWs(); expect('"')
        val sb = StringBuilder()
        while (pos < text.length) {
            val c = text[pos]
            if (c == '"') { pos++; return sb.toString() }
            if (c == '\\' && pos + 1 < text.length) {
                when (val e = text[pos + 1]) {
                    '"' -> sb.append('"')
                    '\\' -> sb.append('\\')
                    'n' -> sb.append('\n')
                    'r' -> sb.append('\r')
                    't' -> sb.append('\t')
                    else -> sb.append(e)
                }
                pos += 2
            } else {
                sb.append(c); pos++
            }
        }
        error("unterminated string at pos $pos")
    }

    private fun skipWs() {
        while (pos < text.length && text[pos].isWhitespace()) pos++
    }
    private fun peek(): Char = if (pos < text.length) text[pos] else ' '
    private fun expect(c: Char) {
        if (pos >= text.length || text[pos] != c) error("expected '$c' at pos $pos, got '${peek()}'")
        pos++
    }
}
