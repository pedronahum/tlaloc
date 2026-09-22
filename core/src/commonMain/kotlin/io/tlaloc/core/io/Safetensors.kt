package io.tlaloc.core.io

import io.tlaloc.core.BF16
import io.tlaloc.core.DType
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.TensorStorage

// §0.4.468 (Phase H2) — the safetensors reader: gap-list item 4's first half
// (INFERENCE_SERVING_AUDIT.md §2), weight ingestion.
//
// The WRITER is SafetensorsWriter.kt (§0.4.502), which states the format facts
// that are only a writer's problem — header padding and tensor ordering — and
// keeps its dtype table the exact inverse of [Safetensors.mapDType]'s. A test
// pins that inverse, because two tables that disagree about what a Tlaloc
// checkpoint may contain is the one way this pair can rot.
//
// THE FORMAT, stated once so no reader in this repo has to guess:
//
//   [0, 8)              u64 little-endian N — the header length in bytes
//   [8, 8+N)            UTF-8 JSON object — the header
//   [8+N, fileEnd)      the data buffer, tensors packed back to back
//
// The header maps tensor name -> {"dtype": "F32", "shape": [d0, d1, ...],
// "data_offsets": [begin, end)}, where begin/end are relative to the START OF
// THE DATA BUFFER (byte 8+N), not to the file. The reserved key
// "__metadata__" maps to a string->string map and is NOT a tensor. Element
// order is row-major (C order), every scalar little-endian.
//
// PLACEMENT (`:core`, package `io.tlaloc.core.io`, `commonMain`).
// The reader's output type IS [DTensor], and [HostF32Storage] /
// [HostBf16Storage] / [HostI32Storage] / [DType] all live in `:core`.
// REJECTED: a new `:io` module — it would name a `:core` type in every
// signature it has and be depended on by every module that loads weights: a
// module boundary with no API of its own. REJECTED: `:maestro` — that module
// is about the deployment MANIFEST (the program), and weights are not a
// program. REJECTED: putting the whole thing in `jvmMain` — the format
// decode is arithmetic over bytes with no filesystem in it, and KMP-common
// is where it can be tested without one. What IS jvm-only is opening a file;
// that is `core/src/jvmMain/.../SafetensorsFile.kt` and nothing else.
//
// UNTRUSTED INPUT. A checkpoint arrives from outside. Every quantity read
// from the header is validated against the file it claims to describe before
// a single byte is decoded: negative dims, an offset pair that runs backwards,
// a byte length that disagrees with dtype*elementCount, a range past the end
// of the buffer, and overlapping tensors are all refused BY NAME. The
// alternative — trusting the header and letting an index throw — reports a
// corrupt checkpoint as an ArrayIndexOutOfBoundsException from deep inside a
// decode loop, which is the same information with the diagnosis removed.

/** One tensor's header entry: what the file CLAIMS, before validation. */
data class SafetensorsEntry(
    val name: String,
    /** The wire dtype string, verbatim ("F32", "BF16", "F16", ...). */
    val wireDType: String,
    val dims: List<Int>,
    /** Byte range within the data buffer, `[begin, end)`. */
    val begin: Long,
    val end: Long,
) {
    val elementCount: Long get() = dims.fold(1L) { a, d -> a * d }
    val byteLength: Long get() = end - begin
}

/** A parsed header: the tensor entries, the metadata map, and where data starts. */
class SafetensorsHeader(
    val entries: Map<String, SafetensorsEntry>,
    val metadata: Map<String, String>,
    /** Absolute file offset of byte 0 of the data buffer (`8 + headerLength`). */
    val dataStart: Long,
    /** Total data-buffer length implied by the file, or -1 when unknown. */
    val dataLength: Long,
) {
    val names: Set<String> get() = entries.keys

    fun entry(name: String): SafetensorsEntry = entries[name]
        ?: throw JsonException(
            "safetensors: no tensor named '$name'; the file holds ${entries.size} " +
                "tensors (${entries.keys.take(8).joinToString(", ")}${if (entries.size > 8) ", ..." else ""})",
        )
}

/**
 * A decoded tensor: the runtime shape plus host storage. The phantom [Shape]
 * of a [DTensor] is a COMPILE-TIME fact and a checkpoint's rank is a RUNTIME
 * one, so this type does not pretend to know it — the caller brands the shape
 * at the point where it knows, via [asF32] / [asBf16] / [asI32] / [asF64].
 * REJECTED: returning `DTensor<Nothing, F32>` and letting subtyping paper over
 * it; it typechecks and it lies about what was proven.
 */
class LoadedTensor(
    val name: String,
    val dtype: DType,
    val dims: IntArray,
    val storage: TensorStorage,
) {
    val size: Int get() = if (dims.isEmpty()) 1 else dims.fold(1) { a, d -> a * d }

    private fun requireDType(want: DType) {
        if (dtype != want) {
            throw JsonException(
                "safetensors: tensor '$name' is ${dtype.name}, not ${want.name} — " +
                    "read it with the ${want.name} accessor only when the checkpoint stores it at that width",
            )
        }
    }

    /**
     * The branding site. A [DTensor]'s [Shape] parameter is phantom — it
     * appears in no field — so the runtime object is the same whatever S is;
     * [ScalarShape] is simply a concrete Shape the constructor can be called
     * with. The cast is unchecked BECAUSE the claim is the caller's: only the
     * caller knows what axes this checkpoint tensor's rank corresponds to.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <S : Shape, T : DType> brand(dt: T): DTensor<S, T> =
        DTensor<ScalarShape, T>(storage, dims.copyOf(), dt) as DTensor<S, T>

    fun <S : Shape> asF32(): DTensor<S, F32> {
        requireDType(F32)
        return brand(F32)
    }

    fun <S : Shape> asBf16(): DTensor<S, BF16> {
        requireDType(BF16)
        return brand(BF16)
    }

    fun <S : Shape> asI32(): DTensor<S, I32> {
        requireDType(I32)
        return brand(I32)
    }

    fun <S : Shape> asF64(): DTensor<S, F64> {
        requireDType(F64)
        return brand(F64)
    }

    /**
     * The tensor's values widened to f32, whatever width it is stored at.
     * This is the compute-in-f32 convention (Bf16.kt) applied at ingestion:
     * a bf16 weight widens EXACTLY (bf16 -> f32 loses nothing), so a graph
     * that runs in f32 on a bf16 checkpoint is reading the checkpoint's real
     * numbers. Refuses the integer dtypes by name — an index tensor is not
     * a number that wants widening.
     */
    fun toF32Array(): FloatArray = when (val s = storage) {
        is HostF32Storage -> s.data
        is HostBf16Storage -> io.tlaloc.core.bf16BitsToFloatArray(s.data)
        is HostF64Storage -> FloatArray(s.data.size) { s.data[it].toFloat() }
        else -> throw JsonException(
            "safetensors: tensor '$name' is ${dtype.name}; toF32Array() is for the " +
                "floating-point dtypes (F32/BF16/F64). Read an integer tensor with asI32().",
        )
    }
}

object Safetensors {

    /** Bytes of the u64 little-endian header-length prefix. */
    const val HEADER_LENGTH_PREFIX_BYTES: Int = 8

    /**
     * A sanity cap on the declared header length, checked BEFORE allocating
     * anything. 100 MB is orders of magnitude above any real checkpoint
     * header (Llama-3-70B's is ~100 KB) and far below what a corrupt or
     * hostile u64 would ask us to allocate. Refusing here is the difference
     * between a named error and an OutOfMemoryError.
     */
    const val MAX_HEADER_BYTES: Long = 100L * 1024 * 1024

    /** Read the u64 little-endian header length from the first 8 bytes. */
    fun headerLength(prefix: ByteArray): Long {
        require(prefix.size >= HEADER_LENGTH_PREFIX_BYTES) {
            "safetensors: need $HEADER_LENGTH_PREFIX_BYTES bytes to read the header length, got ${prefix.size}"
        }
        var n = 0L
        for (i in 0 until HEADER_LENGTH_PREFIX_BYTES) {
            n = n or ((prefix[i].toLong() and 0xFF) shl (8 * i))
        }
        if (n < 0 || n > MAX_HEADER_BYTES) {
            throw JsonException(
                "safetensors: declared header length $n is not in [0, $MAX_HEADER_BYTES] — " +
                    "the file is truncated, byte-swapped, or not a safetensors file",
            )
        }
        return n
    }

    /**
     * Map a wire dtype string to a Tlaloc [DType], refusing BY NAME every
     * width the host side cannot represent. The refusals are the point: a
     * loader that silently upcasts F16 to F32 produces numbers the producer
     * never wrote, and one that narrows I64 to I32 produces token ids that
     * are wrong only for large vocabularies.
     */
    fun mapDType(wire: String): DType = when (wire) {
        "F32" -> F32
        "F64" -> F64
        "BF16" -> BF16
        "I32" -> I32
        "F16" -> refuseDType(
            wire,
            "fp16 has no host representation in Tlaloc (bf16 does: BF16/HostBf16Storage, " +
                "§0.4.455). An fp16 checkpoint must be converted by its producer, or an " +
                "F16 DType + HostF16Storage must land first — narrowing or widening it " +
                "here would put numbers in the tensor that the checkpoint does not contain",
        )
        "I64" -> refuseDType(
            wire,
            "I64 is a Tlaloc DType but has no HostI64Storage, and narrowing to I32 " +
                "silently corrupts any value past 2^31 (HF stores token ids and position " +
                "buffers at this width). Add HostI64Storage before reading one",
        )
        "BOOL", "U8", "I8", "I16", "U16", "U32", "U64" -> refuseDType(
            wire,
            "no host storage at this width; the checkpoint's producer is the only layer " +
                "that can say what it should widen to",
        )
        "F8_E4M3", "F8_E5M2" -> refuseDType(
            wire,
            "fp8 is the KV-quant/weight-quant family, a NAMED DEFERRAL of Phase H5 " +
                "(int8/fp8) with a manifest slot already reserved (kvQuantDtype)",
        )
        else -> refuseDType(wire, "unknown safetensors dtype")
    }

    private fun refuseDType(wire: String, why: String): Nothing =
        throw JsonException("safetensors: dtype '$wire' is refused — $why")

    /**
     * Parse the header JSON. [dataLength] is the number of bytes available
     * after the header (-1 when the caller cannot know); when it is known,
     * every tensor's range is checked against it here rather than at decode
     * time.
     */
    fun parseHeader(json: String, dataStart: Long, dataLength: Long = -1L): SafetensorsHeader {
        val root = parseJson(json) as? JsonObject
            ?: throw JsonException("safetensors: header is not a JSON object")

        val metadata = LinkedHashMap<String, String>()
        val entries = LinkedHashMap<String, SafetensorsEntry>()

        for ((name, value) in root.fields) {
            if (name == "__metadata__") {
                val m = value as? JsonObject
                    ?: throw JsonException("safetensors: __metadata__ is not a JSON object")
                for ((k, v) in m.fields) {
                    metadata[k] = (v as? JsonString)?.value
                        ?: throw JsonException("safetensors: __metadata__['$k'] is not a string")
                }
                continue
            }
            val o = value as? JsonObject
                ?: throw JsonException("safetensors: entry '$name' is not a JSON object")
            val wire = o.str("dtype")
            val dims = o.arr("shape").asIntList("safetensors: '$name'.shape")
            if (dims.any { it < 0 }) {
                throw JsonException("safetensors: '$name' has a negative dim in ${dims}")
            }
            val offs = o.arr("data_offsets")
            if (offs.size != 2) {
                throw JsonException("safetensors: '$name'.data_offsets must have 2 entries, got ${offs.size}")
            }
            val pair = offs.asLongList("safetensors: '$name'.data_offsets")
            val (begin, end) = pair[0] to pair[1]
            if (begin < 0 || end < begin) {
                throw JsonException("safetensors: '$name' has a backwards or negative range [$begin, $end)")
            }
            entries[name] = SafetensorsEntry(name, wire, dims, begin, end)
        }

        // Validate each entry against its OWN claims and against the buffer.
        // Byte length must equal dtype width * element count exactly: a
        // mismatch means the header and the data disagree about the tensor,
        // and the header is the only thing telling us where the NEXT tensor
        // starts.
        for (e in entries.values) {
            val dt = mapDType(e.wireDType)
            val want = e.elementCount * dt.sizeBytes
            if (e.byteLength != want) {
                throw JsonException(
                    "safetensors: '${e.name}' declares shape ${e.dims} at ${e.wireDType} " +
                        "(= $want bytes) but its data_offsets span ${e.byteLength} bytes",
                )
            }
            if (dataLength >= 0 && e.end > dataLength) {
                throw JsonException(
                    "safetensors: '${e.name}' ends at ${e.end} but the data buffer is only $dataLength bytes",
                )
            }
        }

        // Overlap check. Tensors in a well-formed file are disjoint; an
        // overlap means two names alias one buffer, which is either a
        // corrupt header or a deliberate trick, and both deserve a refusal
        // rather than a silently shared FloatArray.
        val sorted = entries.values.filter { it.byteLength > 0 }.sortedBy { it.begin }
        for (i in 1 until sorted.size) {
            val prev = sorted[i - 1]
            val cur = sorted[i]
            if (cur.begin < prev.end) {
                throw JsonException(
                    "safetensors: '${cur.name}' [${cur.begin}, ${cur.end}) overlaps " +
                        "'${prev.name}' [${prev.begin}, ${prev.end})",
                )
            }
        }

        return SafetensorsHeader(entries, metadata, dataStart, dataLength)
    }

    /**
     * Decode one tensor from [bytes], where `bytes[0]` is the FIRST BYTE OF
     * THE TENSOR (the caller has already sliced `[begin, end)` out of the
     * data buffer). Splitting it this way is what lets the JVM entry point
     * read only the range it wants instead of the whole checkpoint.
     */
    fun decode(entry: SafetensorsEntry, bytes: ByteArray): LoadedTensor {
        val dt = mapDType(entry.wireDType)
        val n = entry.elementCount
        if (n > Int.MAX_VALUE) {
            throw JsonException(
                "safetensors: '${entry.name}' has $n elements, past the ${Int.MAX_VALUE}-element " +
                    "limit of a JVM array — sharded or streamed loading would be the fix",
            )
        }
        val count = n.toInt()
        if (bytes.size.toLong() != entry.byteLength) {
            throw JsonException(
                "safetensors: '${entry.name}' needs ${entry.byteLength} bytes, got ${bytes.size}",
            )
        }
        val storage: TensorStorage = when (dt) {
            F32 -> HostF32Storage(FloatArray(count) { Float.fromBits(leInt(bytes, it * 4)) })
            F64 -> HostF64Storage(DoubleArray(count) { Double.fromBits(leLong(bytes, it * 8)) })
            BF16 -> HostBf16Storage(ShortArray(count) { leShort(bytes, it * 2) })
            I32 -> HostI32Storage(IntArray(count) { leInt(bytes, it * 4) })
            else -> throw JsonException("safetensors: unreachable dtype ${dt.name}")
        }
        return LoadedTensor(entry.name, dt, entry.dims.toIntArray(), storage)
    }

    private fun leShort(b: ByteArray, o: Int): Short =
        (((b[o].toInt() and 0xFF)) or ((b[o + 1].toInt() and 0xFF) shl 8)).toShort()

    private fun leInt(b: ByteArray, o: Int): Int =
        (b[o].toInt() and 0xFF) or
            ((b[o + 1].toInt() and 0xFF) shl 8) or
            ((b[o + 2].toInt() and 0xFF) shl 16) or
            ((b[o + 3].toInt() and 0xFF) shl 24)

    private fun leLong(b: ByteArray, o: Int): Long {
        var v = 0L
        for (i in 0 until 8) v = v or ((b[o + i].toLong() and 0xFF) shl (8 * i))
        return v
    }

    /**
     * Read a whole in-memory safetensors file. Convenience for small files
     * and for tests; the JVM file entry point does NOT go through this (it
     * reads per-tensor ranges instead of materialising the checkpoint twice).
     */
    fun readAll(file: ByteArray): Map<String, LoadedTensor> {
        val n = headerLength(file)
        val dataStart = HEADER_LENGTH_PREFIX_BYTES + n
        if (file.size < dataStart) {
            throw JsonException(
                "safetensors: file is ${file.size} bytes but the header alone claims $dataStart",
            )
        }
        val json = file.decodeToString(HEADER_LENGTH_PREFIX_BYTES, dataStart.toInt())
        val header = parseHeader(json, dataStart, file.size - dataStart)
        return header.entries.mapValues { (_, e) ->
            val from = (dataStart + e.begin).toInt()
            decode(e, file.copyOfRange(from, from + e.byteLength.toInt()))
        }
    }
}
