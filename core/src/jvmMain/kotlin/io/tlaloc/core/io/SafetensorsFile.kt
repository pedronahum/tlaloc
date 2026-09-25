package io.tlaloc.core.io

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

// §0.4.468 (Phase H2) — the ONLY jvm-specific part of weight ingestion:
// opening a file and reading byte ranges out of it. Everything about the
// safetensors FORMAT lives in commonMain (Safetensors.kt); this file holds a
// FileChannel and nothing else that could be wrong.

/**
 * A safetensors checkpoint open on disk, read PER TENSOR.
 *
 * The header is read once at [open]; each [load] reads only that tensor's
 * `[begin, end)` range. REJECTED: reading the whole file into a ByteArray and
 * slicing — it doubles peak memory at the exact moment memory is scarce (a
 * 16 GB checkpoint would need 16 GB of byte array plus 16 GB of decoded
 * FloatArrays), and a serving process loading a 70B model is the intended
 * caller. REJECTED for v1: `FileChannel.map` (mmap) — it is the right answer
 * for a large f32 checkpoint whose storage could be a direct buffer view
 * rather than a copy, but Tlaloc's host storages are Kotlin primitive arrays,
 * so the decode copies regardless and mmap would only change WHICH copy the
 * OS makes. A zero-copy storage backed by a MemorySegment is a NAMED
 * DEFERRAL (and the FFM stack to do it already exists).
 */
class SafetensorsFile private constructor(
    val path: Path,
    private val channel: FileChannel,
    val header: SafetensorsHeader,
) : AutoCloseable {

    val names: Set<String> get() = header.names

    val metadata: Map<String, String> get() = header.metadata

    /** Decode one tensor by name, reading only its bytes. */
    fun load(name: String): LoadedTensor {
        val e = header.entry(name)
        val bytes = readRange(header.dataStart + e.begin, e.byteLength)
        return Safetensors.decode(e, bytes)
    }

    /** Decode every tensor, in header order. */
    fun loadAll(): Map<String, LoadedTensor> =
        header.entries.keys.associateWith { load(it) }

    /** The header entry of one tensor: its wire dtype, dims and byte range. */
    fun entry(name: String): SafetensorsEntry = header.entry(name)

    /**
     * Copy [length] raw bytes of tensor [name], starting [byteOffset] bytes
     * into it, into [into] at [offset]. No decoding: the bytes are the file's,
     * little-endian in the tensor's wire dtype. This is how a tensor larger
     * than a JVM array (a 2.7 GB bf16 embedding table) is read, a piece at a
     * time.
     */
    fun readBytes(name: String, byteOffset: Long, into: ByteArray, offset: Int = 0, length: Int = into.size - offset) {
        val e = header.entry(name)
        require(byteOffset >= 0 && length >= 0 && byteOffset + length <= e.byteLength) {
            "safetensors: bytes [$byteOffset, ${byteOffset + length}) are outside '$name' " +
                "(${e.byteLength} bytes)"
        }
        require(offset >= 0 && offset + length <= into.size) {
            "safetensors: $length bytes at $offset do not fit a ${into.size}-byte buffer"
        }
        val bb = ByteBuffer.wrap(into, offset, length)
        var pos = header.dataStart + e.begin + byteOffset
        while (bb.hasRemaining()) {
            val n = channel.read(bb, pos)
            if (n < 0) {
                throw JsonException("safetensors: $path ended at $pos while reading '$name'")
            }
            pos += n
        }
    }

    private fun readRange(fileOffset: Long, length: Long): ByteArray {
        if (length > Int.MAX_VALUE) {
            throw JsonException("safetensors: tensor range of $length bytes exceeds a JVM array")
        }
        val buf = ByteArray(length.toInt())
        val bb = ByteBuffer.wrap(buf)
        var pos = fileOffset
        while (bb.hasRemaining()) {
            // FileChannel.read is allowed to return a short read; a loop is
            // the difference between "works on my 4 KB test file" and
            // "works on a 9 GB shard".
            val n = channel.read(bb, pos)
            if (n < 0) {
                throw JsonException(
                    "safetensors: $path ended at ${pos} while reading a $length-byte tensor range " +
                        "starting at $fileOffset — the file is truncated relative to its header",
                )
            }
            pos += n
        }
        return buf
    }

    override fun close() = channel.close()

    companion object {
        /** Open [path], parse its header, and validate it against the file's real size. */
        fun open(path: Path): SafetensorsFile {
            val channel = FileChannel.open(path, StandardOpenOption.READ)
            try {
                val size = channel.size()
                if (size < Safetensors.HEADER_LENGTH_PREFIX_BYTES) {
                    throw JsonException(
                        "safetensors: $path is $size bytes, too short to hold a header length",
                    )
                }
                val prefix = ByteArray(Safetensors.HEADER_LENGTH_PREFIX_BYTES)
                readFully(channel, 0, prefix, path)
                val n = Safetensors.headerLength(prefix)
                val dataStart = Safetensors.HEADER_LENGTH_PREFIX_BYTES + n
                if (size < dataStart) {
                    throw JsonException(
                        "safetensors: $path is $size bytes but its header claims to end at $dataStart",
                    )
                }
                val json = ByteArray(n.toInt())
                readFully(channel, Safetensors.HEADER_LENGTH_PREFIX_BYTES.toLong(), json, path)
                val header = Safetensors.parseHeader(
                    String(json, StandardCharsets.UTF_8),
                    dataStart,
                    size - dataStart,
                )
                return SafetensorsFile(path, channel, header)
            } catch (t: Throwable) {
                channel.close()
                throw t
            }
        }

        private fun readFully(channel: FileChannel, at: Long, into: ByteArray, path: Path) {
            val bb = ByteBuffer.wrap(into)
            var pos = at
            while (bb.hasRemaining()) {
                val n = channel.read(bb, pos)
                if (n < 0) throw JsonException("safetensors: $path is truncated at $pos")
                pos += n
            }
        }
    }
}

/**
 * The SHARDED form: `model.safetensors.index.json`, which HF writes whenever a
 * checkpoint exceeds one file. The index is a JSON object with a `weight_map`
 * of tensor name -> shard filename (and an optional `metadata` object whose
 * `total_size` we deliberately do not trust — the shards' own headers are the
 * authority on where bytes live).
 *
 * Shards open LAZILY, on the first tensor that names them, and all of them
 * close together. REJECTED: opening every shard up front — a 70B checkpoint is
 * 30 files, and a caller that wants one tensor should pay for one file.
 * REJECTED: resolving shard paths anywhere but the index's own directory — a
 * `weight_map` value is an untrusted string, and a checkpoint that can name
 * `../../etc/...` is a checkpoint that can read arbitrary files; the filename
 * is required to be a bare name with no separator, refused by name otherwise.
 */
class SafetensorsIndex private constructor(
    val indexPath: Path,
    /** tensor name -> shard file name, verbatim from `weight_map`. */
    val weightMap: Map<String, String>,
    val metadata: Map<String, JsonValue>,
) : AutoCloseable {

    private val open = LinkedHashMap<String, SafetensorsFile>()

    val names: Set<String> get() = weightMap.keys

    /** The distinct shard files this index refers to, in first-mention order. */
    val shardNames: List<String> get() = weightMap.values.distinct()

    fun load(name: String): LoadedTensor {
        val shard = weightMap[name] ?: throw JsonException(
            "safetensors index: no tensor named '$name' in ${indexPath.fileName} " +
                "(${weightMap.size} entries across ${shardNames.size} shards)",
        )
        val file = open.getOrPut(shard) {
            SafetensorsFile.open(indexPath.parent.resolve(shard))
        }
        return file.load(name)
    }

    fun loadAll(): Map<String, LoadedTensor> = weightMap.keys.associateWith { load(it) }

    /** The header entry of one tensor, from the shard that holds it. */
    fun entry(name: String): SafetensorsEntry = shardOf(name).entry(name)

    /** [SafetensorsFile.readBytes] on the shard that holds [name]. */
    fun readBytes(name: String, byteOffset: Long, into: ByteArray, offset: Int = 0, length: Int = into.size - offset) =
        shardOf(name).readBytes(name, byteOffset, into, offset, length)

    private fun shardOf(name: String): SafetensorsFile {
        val shard = weightMap[name] ?: throw JsonException(
            "safetensors index: no tensor named '$name' in ${indexPath.fileName} " +
                "(${weightMap.size} entries across ${shardNames.size} shards)",
        )
        return open.getOrPut(shard) { SafetensorsFile.open(indexPath.parent.resolve(shard)) }
    }

    override fun close() {
        var first: Throwable? = null
        for (f in open.values) {
            try {
                f.close()
            } catch (t: Throwable) {
                if (first == null) first = t
            }
        }
        open.clear()
        if (first != null) throw first
    }

    companion object {
        fun open(indexPath: Path): SafetensorsIndex {
            val text = String(Files.readAllBytes(indexPath), StandardCharsets.UTF_8)
            val root = parseJson(text) as? JsonObject
                ?: throw JsonException("safetensors index: $indexPath is not a JSON object")
            val wm = root.obj("weight_map")
            val map = LinkedHashMap<String, String>()
            for ((tensor, v) in wm.fields) {
                val shard = (v as? JsonString)?.value
                    ?: throw JsonException("safetensors index: weight_map['$tensor'] is not a string")
                if (shard.isEmpty() || shard.contains('/') || shard.contains('\\') || shard == "." || shard == "..") {
                    throw JsonException(
                        "safetensors index: weight_map['$tensor'] = '$shard' is not a bare shard " +
                            "filename; an index may only name files beside itself",
                    )
                }
                map[tensor] = shard
            }
            val meta = (root["metadata"] as? JsonObject)?.fields ?: emptyMap()
            return SafetensorsIndex(indexPath.toAbsolutePath(), map, meta)
        }

        /**
         * Open a checkpoint DIRECTORY the way a serving loader would: the
         * sharded index if there is one, otherwise the single
         * `model.safetensors`. Returns a [WeightSource] so callers need not
         * care which shape they got.
         */
        fun openCheckpoint(dir: Path): WeightSource {
            val index = dir.resolve("model.safetensors.index.json")
            if (Files.isRegularFile(index)) return IndexWeightSource(open(index))
            val single = dir.resolve("model.safetensors")
            if (Files.isRegularFile(single)) return FileWeightSource(SafetensorsFile.open(single))
            throw JsonException(
                "safetensors: $dir holds neither model.safetensors.index.json nor model.safetensors",
            )
        }
    }
}

/** What a weight loader needs, independent of sharding. */
interface WeightSource : AutoCloseable {
    val names: Set<String>
    fun load(name: String): LoadedTensor

    /** The header entry of one tensor (wire dtype, dims, byte length), without reading it. */
    fun entry(name: String): SafetensorsEntry =
        throw UnsupportedOperationException("${this::class.simpleName} does not expose header entries")

    /** Raw bytes of a tensor, a range at a time; see [SafetensorsFile.readBytes]. */
    fun readBytes(name: String, byteOffset: Long, into: ByteArray, offset: Int, length: Int): Unit =
        throw UnsupportedOperationException("${this::class.simpleName} does not expose raw bytes")
}

private class FileWeightSource(private val f: SafetensorsFile) : WeightSource {
    override val names: Set<String> get() = f.names
    override fun load(name: String): LoadedTensor = f.load(name)
    override fun entry(name: String): SafetensorsEntry = f.entry(name)
    override fun readBytes(name: String, byteOffset: Long, into: ByteArray, offset: Int, length: Int) =
        f.readBytes(name, byteOffset, into, offset, length)
    override fun close() = f.close()
}

private class IndexWeightSource(private val i: SafetensorsIndex) : WeightSource {
    override val names: Set<String> get() = i.names
    override fun load(name: String): LoadedTensor = i.load(name)
    override fun entry(name: String): SafetensorsEntry = i.entry(name)
    override fun readBytes(name: String, byteOffset: Long, into: ByteArray, offset: Int, length: Int) =
        i.readBytes(name, byteOffset, into, offset, length)
    override fun close() = i.close()
}
