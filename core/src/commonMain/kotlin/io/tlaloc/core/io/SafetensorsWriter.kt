package io.tlaloc.core.io

import io.tlaloc.core.BF16
import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.core.TensorStorage

// §0.4.502 (Tier 2 item 7) — the safetensors WRITER, the other half of
// §0.4.468's reader. Before this, Tlaloc could train a model and had no way
// to save the result: `Components.kt` said so in a comment ("minus
// store/load, out of scope v1") and `Safetensors.kt` was read-only.
//
// THE FORMAT is stated once, in Safetensors.kt's header comment, and this
// file is written against that statement rather than restating it. What is
// only a writer's problem, and is therefore recorded HERE:
//
//   HEADER PADDING. The header JSON is padded with ASCII spaces until
//   `8 + headerLength` is a multiple of 8. The reference implementation does
//   this (`safetensors` 0.8.0, observed: a 195-byte header written as 200
//   with five trailing spaces) and the reason is the next paragraph. Our own
//   reader does not require it — it decodes byte by byte — but a reader that
//   builds a zero-copy VIEW over the mapped buffer does, and producing files
//   only our own reader can use is not interoperability.
//
//   TENSOR ORDER. Tensors are emitted sorted by DESCENDING dtype width, then
//   by name. That is not an aesthetic choice and it is not the reference
//   implementation's order copied for its own sake: it is what makes every
//   tensor's data offset naturally aligned to its own element width. The data
//   buffer starts at a multiple of 8 (previous paragraph), so writing the
//   8-byte dtypes first, then the 4-byte ones, then the 2-byte ones leaves
//   every tensor beginning on a multiple of its element size — which is what
//   `torch.frombuffer` and `np.ndarray(buffer=...)` need to view the bytes
//   without copying. Sorting by NAME alone would be equally legal per the
//   format and would put an f64 tensor at an odd multiple of 4 the first time
//   a 2-byte tensor sorted ahead of it.
//
//   DETERMINISM. Because both the order and the padding are functions of the
//   input alone, `encode` is a pure function of (tensors, metadata): the same
//   model saved twice produces byte-identical files, which is what makes a
//   checkpoint hashable. Pinned by a test, not just asserted here.
//
// WHAT IS REFUSED, BY NAME. The writer refuses exactly the dtypes the reader
// refuses, from the same table and for the same reasons — F16 and the fp8
// pair have no host representation in Tlaloc at all (there is no `F16`
// DType), and I64/Bool are DTypes with no host storage class, so a tensor at
// those widths cannot be constructed to be written in the first place. A
// writer that silently widened bf16 to f32 to "help" would produce a
// checkpoint whose bytes its own producer never computed.
//
// NAMED DEFERRALS (recorded, not hidden):
//   - SHARDED output. A checkpoint larger than 2 GB cannot be encoded to a
//     single JVM ByteArray at all, and `encode` refuses by name rather than
//     throwing OutOfMemoryError from inside a copy loop. The sharded form
//     (`model.safetensors.index.json`) is already READ by
//     `SafetensorsIndex`; writing one needs a shard-assignment policy, which
//     is a decision, not a format detail.
//   - STREAMING output. `encode` materialises the whole file. The jvm entry
//     point (`SafetensorsFileWriter`) inherits that. A streaming writer that
//     computed the header first and then pushed tensors through a channel is
//     the right shape for a 70B checkpoint and is not needed by anything in
//     this repository today.

/**
 * One tensor on its way out: a name, a dtype, a runtime shape and host
 * storage. This is the writer's input element and deliberately NOT a
 * `DTensor` — a [DTensor]'s phantom [io.tlaloc.core.Shape] is a compile-time
 * fact that the file cannot carry, and a heterogeneous list of `DTensor<*,*>`
 * is exactly what a checkpoint is. [LoadedTensor] is the reader's mirror of
 * this type; [of] converts either.
 */
class SafetensorsTensor(
    val name: String,
    val dtype: DType,
    val dims: IntArray,
    val storage: TensorStorage,
) {
    val elementCount: Long get() = if (dims.isEmpty()) 1L else dims.fold(1L) { a, d -> a * d }

    companion object {
        fun of(name: String, t: DTensor<*, *>): SafetensorsTensor =
            SafetensorsTensor(name, t.dtype, t.dims.copyOf(), t.storage)

        fun of(t: LoadedTensor): SafetensorsTensor =
            SafetensorsTensor(t.name, t.dtype, t.dims.copyOf(), t.storage)
    }
}

object SafetensorsWriter {

    /**
     * The file's data buffer begins at `8 + headerLength`, and the header is
     * space-padded so that quantity is a multiple of this. See the file
     * comment for why a writer that skips this produces files only its own
     * reader can use.
     */
    const val ALIGNMENT: Int = 8

    /** The reserved header key; a tensor may not be called this. */
    const val METADATA_KEY: String = "__metadata__"

    /**
     * The inverse of [Safetensors.mapDType]: a Tlaloc [DType] to its wire
     * string, refusing BY NAME every dtype this project cannot write. The
     * refusals quote the reader's own reasons so the two tables cannot drift
     * into disagreeing about what a Tlaloc checkpoint may contain.
     */
    fun wireDType(dt: DType): String = when (dt) {
        F32 -> "F32"
        F64 -> "F64"
        BF16 -> "BF16"
        I32 -> "I32"
        I64 -> refuseDType(
            dt,
            "I64 is a Tlaloc DType but has no HostI64Storage, so no I64 tensor can be " +
                "constructed to write; the reader refuses to read one for the same reason",
        )
        Bool -> refuseDType(dt, "no host storage at this width")
        // DELIBERATELY NO `else`. [DType] is sealed, so this `when` is
        // exhaustive, and a new DType added to `:core` breaks THIS FILE at
        // compile time with the author's cursor on the decision they have to
        // make. An `else` arm would turn that into a runtime refusal nobody
        // sees until a checkpoint is being written.
    }

    private fun refuseDType(dt: DType, why: String): Nothing =
        throw JsonException(
            "safetensors writer: dtype '${dt.name}' is refused — $why. Tlaloc writes " +
                "F32, F64, BF16 and I32; fp16 and the fp8 pair have no host representation " +
                "in Tlaloc at all (there is no F16 DType)",
        )

    /**
     * Encode a complete safetensors file.
     *
     * Validates before it allocates: duplicate names, the reserved
     * `__metadata__` name, a negative dim, an element count that disagrees
     * with the storage's length, a storage class that disagrees with the
     * declared dtype, and a total size past the JVM array ceiling are each
     * refused by name. The alternative — writing the header we were told to
     * write and letting a copy throw — produces a file whose header describes
     * bytes that are not in it.
     */
    fun encode(
        tensors: List<SafetensorsTensor>,
        metadata: Map<String, String> = emptyMap(),
    ): ByteArray {
        val seen = HashSet<String>(tensors.size)
        for (t in tensors) {
            if (t.name == METADATA_KEY) {
                throw JsonException(
                    "safetensors writer: '$METADATA_KEY' is the format's reserved metadata key " +
                        "and cannot name a tensor",
                )
            }
            if (!seen.add(t.name)) {
                throw JsonException("safetensors writer: duplicate tensor name '${t.name}'")
            }
            if (t.dims.any { it < 0 }) {
                throw JsonException(
                    "safetensors writer: '${t.name}' has a negative dim in ${t.dims.toList()}",
                )
            }
            // Touches the dtype table BEFORE any sizing arithmetic, so an
            // unwritable dtype is reported as itself rather than as a
            // mismatched byte count.
            wireDType(t.dtype)
            val have = storageLength(t)
            if (have.toLong() != t.elementCount) {
                throw JsonException(
                    "safetensors writer: '${t.name}' declares shape ${t.dims.toList()} " +
                        "(= ${t.elementCount} elements) but its storage holds $have",
                )
            }
        }

        // The ordering decision, in one line. See the file comment.
        val ordered = tensors.sortedWith(
            compareByDescending<SafetensorsTensor> { it.dtype.sizeBytes }.thenBy { it.name },
        )

        var dataBytes = 0L
        val ranges = ArrayList<LongArray>(ordered.size)
        for (t in ordered) {
            val begin = dataBytes
            dataBytes += t.elementCount * t.dtype.sizeBytes
            ranges.add(longArrayOf(begin, dataBytes))
        }

        val header = buildHeader(ordered, ranges, metadata)
        val headerBytes = header.encodeToByteArray()
        val padded = headerBytes.size + ((ALIGNMENT - (ALIGNMENT + headerBytes.size) % ALIGNMENT) % ALIGNMENT)
        if (padded.toLong() > Safetensors.MAX_HEADER_BYTES) {
            throw JsonException(
                "safetensors writer: the header would be $padded bytes, past the " +
                    "${Safetensors.MAX_HEADER_BYTES}-byte cap the reader enforces — " +
                    "a checkpoint with this many tensors needs the sharded form, which is not supported",
            )
        }
        val total = ALIGNMENT.toLong() + padded + dataBytes
        if (total > Int.MAX_VALUE) {
            throw JsonException(
                "safetensors writer: the file would be $total bytes, past the " +
                    "${Int.MAX_VALUE}-byte ceiling of a JVM ByteArray — sharded or streamed " +
                    "writing would be needed and is not supported",
            )
        }

        val out = ByteArray(total.toInt())
        var n = padded.toLong()
        for (i in 0 until ALIGNMENT) {
            out[i] = (n and 0xFF).toByte()
            n = n ushr 8
        }
        headerBytes.copyInto(out, ALIGNMENT)
        for (i in headerBytes.size until padded) out[ALIGNMENT + i] = ' '.code.toByte()
        val dataStart = ALIGNMENT + padded
        for ((i, t) in ordered.withIndex()) {
            writeTensor(t, out, dataStart + ranges[i][0].toInt())
        }
        return out
    }

    private fun buildHeader(
        ordered: List<SafetensorsTensor>,
        ranges: List<LongArray>,
        metadata: Map<String, String>,
    ): String {
        val sb = StringBuilder()
        sb.append('{')
        var first = true
        if (metadata.isNotEmpty()) {
            sb.append(jsonQuote(METADATA_KEY)).append(":{")
            var m = true
            // Sorted so the header is a pure function of the map's CONTENT,
            // not of the iteration order of whatever map the caller passed.
            for (key in metadata.keys.sorted()) {
                if (!m) sb.append(',')
                m = false
                sb.append(jsonQuote(key)).append(':').append(jsonQuote(metadata.getValue(key)))
            }
            sb.append('}')
            first = false
        }
        for ((i, t) in ordered.withIndex()) {
            if (!first) sb.append(',')
            first = false
            sb.append(jsonQuote(t.name))
                .append(":{\"dtype\":").append(jsonQuote(wireDType(t.dtype)))
                .append(",\"shape\":[").append(t.dims.joinToString(",")).append(']')
                .append(",\"data_offsets\":[").append(ranges[i][0]).append(',')
                .append(ranges[i][1]).append("]}")
        }
        sb.append('}')
        return sb.toString()
    }

    /**
     * The storage's element count, and the place where a dtype/storage
     * disagreement is caught. A `HostF32Storage` under a declared `BF16`
     * would otherwise write four bytes per element into a two-byte-per-
     * element range and corrupt every tensor after it.
     */
    private fun storageLength(t: SafetensorsTensor): Int = when (t.dtype) {
        F32 -> (t.storage as? HostF32Storage)?.data?.size ?: mismatch(t, "HostF32Storage")
        F64 -> (t.storage as? HostF64Storage)?.data?.size ?: mismatch(t, "HostF64Storage")
        BF16 -> (t.storage as? HostBf16Storage)?.data?.size ?: mismatch(t, "HostBf16Storage")
        I32 -> (t.storage as? HostI32Storage)?.data?.size ?: mismatch(t, "HostI32Storage")
        else -> throw JsonException("safetensors writer: unreachable dtype ${t.dtype.name}")
    }

    private fun mismatch(t: SafetensorsTensor, want: String): Nothing =
        throw JsonException(
            "safetensors writer: '${t.name}' is declared ${t.dtype.name} but its storage is " +
                "${t.storage::class.simpleName ?: "an unknown storage"}, not $want — a device " +
                "tensor must be brought to the host before it can be written",
        )

    private fun writeTensor(t: SafetensorsTensor, out: ByteArray, at: Int) {
        when (val s = t.storage) {
            is HostF32Storage -> for (i in s.data.indices) putInt(out, at + i * 4, s.data[i].toRawBits())
            is HostF64Storage -> for (i in s.data.indices) putLong(out, at + i * 8, s.data[i].toRawBits())
            is HostI32Storage -> for (i in s.data.indices) putInt(out, at + i * 4, s.data[i])
            is HostBf16Storage -> for (i in s.data.indices) putShort(out, at + i * 2, s.data[i])
            else -> throw JsonException("safetensors writer: unreachable storage for '${t.name}'")
        }
    }

    private fun putShort(b: ByteArray, o: Int, v: Short) {
        val i = v.toInt()
        b[o] = (i and 0xFF).toByte()
        b[o + 1] = ((i ushr 8) and 0xFF).toByte()
    }

    private fun putInt(b: ByteArray, o: Int, v: Int) {
        b[o] = (v and 0xFF).toByte()
        b[o + 1] = ((v ushr 8) and 0xFF).toByte()
        b[o + 2] = ((v ushr 16) and 0xFF).toByte()
        b[o + 3] = ((v ushr 24) and 0xFF).toByte()
    }

    private fun putLong(b: ByteArray, o: Int, v: Long) {
        var x = v
        for (i in 0 until 8) {
            b[o + i] = (x and 0xFF).toByte()
            x = x ushr 8
        }
    }
}
