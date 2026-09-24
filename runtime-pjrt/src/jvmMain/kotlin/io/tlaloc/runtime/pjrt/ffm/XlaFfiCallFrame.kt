package io.tlaloc.runtime.pjrt.ffm

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.nio.charset.StandardCharsets

/**
 * Pure-Kotlin decoder for the typed-FFI
 * **`XLA_FFI_CallFrame`** delivered to handlers registered via
 * [PjrtFfiRegistry]. This is the pyptx-shim decode (call frame → stream +
 * device buffer pointers + attributes) done in FFM instead of C.
 *
 * All layouts from the jaxlib-0.10.0-shipped header
 * `jaxlib/include/xla/ffi/api/c_api.h` (FFI API 0.3), aarch64/x86_64
 * (8-byte pointers, 4-byte enums padded to 8-byte alignment):
 *
 * ```
 * XLA_FFI_CallFrame   struct_size(0) ext(8) api(16) ctx(24) stage(int 32)
 *                     args(inline 40) rets(inline 80) attrs(inline 120)
 *                     future(168)
 * XLA_FFI_Args/Rets   struct_size(+0) ext(+8) size(i64 +16) types(+24)
 *                     args|rets(+32)          — 40 bytes
 * XLA_FFI_Attrs       struct_size(+0) ext(+8) size(i64 +16) types(+24)
 *                     names(+32) attrs(+40)   — 48 bytes
 * XLA_FFI_Buffer      struct_size(0) ext(8) dtype(int 16) data(24)
 *                     rank(i64 32) dims(40)
 * XLA_FFI_ByteSpan    ptr(0) len(8)
 * XLA_FFI_Scalar      dtype(int 0) value(8)
 * XLA_FFI_Api         … api_version(16..39) internal_api(40)
 *                     Error_Create(48) Error_GetMessage(56)
 *                     Error_Destroy(64) Handler_Register(72)
 *                     Stream_Get(80) …
 * XLA_FFI_Stream_Get_Args struct_size(0) ext(8) ctx(16) stream(out 24)
 * ```
 */
object XlaFfi {

    // XLA_FFI_CallFrame offsets.
    internal const val OFF_FRAME_API = 16L
    internal const val OFF_FRAME_CTX = 24L
    internal const val OFF_FRAME_STAGE = 32L
    internal const val OFF_FRAME_ARGS = 40L
    internal const val OFF_FRAME_RETS = 80L
    internal const val OFF_FRAME_ATTRS = 120L
    internal const val FRAME_MIN_SIZE = 176L

    // Offsets *within* the inline Args/Rets/Attrs structs.
    internal const val OFF_SEQ_SIZE = 16L
    internal const val OFF_SEQ_TYPES = 24L
    internal const val OFF_SEQ_VALUES = 32L
    internal const val OFF_ATTRS_NAMES = 32L
    internal const val OFF_ATTRS_VALUES = 40L

    // XLA_FFI_Buffer offsets.
    internal const val OFF_BUF_DTYPE = 16L
    internal const val OFF_BUF_DATA = 24L
    internal const val OFF_BUF_RANK = 32L
    internal const val OFF_BUF_DIMS = 40L

    // XLA_FFI_Api fn-ptr offsets.
    internal const val OFF_API_ERROR_CREATE = 48L
    internal const val OFF_API_STREAM_GET = 80L

    /** XLA_FFI_Error_Code_INTERNAL (canonical absl error space). */
    const val ERROR_CODE_INTERNAL = 13

    // Enums.
    const val STAGE_EXECUTE = 3
    const val ARG_TYPE_BUFFER = 1
    const val ATTR_TYPE_ARRAY = 1
    const val ATTR_TYPE_DICTIONARY = 2
    const val ATTR_TYPE_SCALAR = 3
    const val ATTR_TYPE_STRING = 4

    // XLA_FFI_DataType values (subset Tlaloc touches; F32 is the §0.4.30x
    // runtime's only dtype today).
    const val DTYPE_S32 = 4
    const val DTYPE_S64 = 5
    const val DTYPE_F16 = 10
    const val DTYPE_F32 = 11
    const val DTYPE_F64 = 12
    const val DTYPE_BF16 = 16

    /** One `XLA_FFI_Buffer`: [dataAddress] is the raw **device** pointer
     * (a CUdeviceptr on the CUDA platform — feed it straight into the
     * `void**` kernel-params array), [dims] the logical shape. */
    data class Buffer(val dtype: Int, val dataAddress: Long, val dims: LongArray) {
        val elementCount: Long get() = dims.fold(1L) { a, b -> a * b }
        override fun equals(other: Any?): Boolean =
            other is Buffer && dtype == other.dtype && dataAddress == other.dataAddress &&
                dims.contentEquals(other.dims)
        override fun hashCode(): Int =
            31 * (31 * dtype + dataAddress.hashCode()) + dims.contentHashCode()
    }

    /** Scalar attribute (`XLA_FFI_AttrType_SCALAR`), decoded per dtype. */
    sealed interface AttrValue {
        data class I32(val value: Int) : AttrValue
        data class I64(val value: Long) : AttrValue
        data class F32(val value: Float) : AttrValue
        data class F64(val value: Double) : AttrValue
        data class Str(val value: String) : AttrValue
        /** ARRAY / DICTIONARY / unrecognised scalar dtypes — raw pointer
         * for callers that know the layout. */
        data class Raw(val attrType: Int, val pointer: MemorySegment) : AttrValue
    }

    /** Decoded view over a raw `XLA_FFI_CallFrame*`. Purely a *reader* —
     * the frame and everything it points to is owned by XLA and valid only
     * for the duration of the handler invocation; do not retain. */
    class Frame internal constructor(private val frame: MemorySegment) {

        val stage: Int = frame.get(JAVA_INT, OFF_FRAME_STAGE)

        val args: List<Buffer> by lazy { decodeBuffers(OFF_FRAME_ARGS, OFF_SEQ_VALUES) }
        val rets: List<Buffer> by lazy { decodeBuffers(OFF_FRAME_RETS, OFF_SEQ_VALUES) }
        val attrs: Map<String, AttrValue> by lazy { decodeAttrs() }

        /** The platform stream XLA is sequencing this program on — a
         * `CUstream` on the CUDA platform. Launch kernels **on this
         * stream**; never synchronize or switch contexts. */
        fun streamGet(): MemorySegment {
            val api = frame.get(ADDRESS, OFF_FRAME_API).reinterpret(256)
            val ctx = frame.get(ADDRESS, OFF_FRAME_CTX)
            val streamGetFn = PjrtFfm.LINKER.downcallHandle(
                api.get(ADDRESS, OFF_API_STREAM_GET).reinterpret(Long.MAX_VALUE),
                FunctionDescriptor.of(ADDRESS, ADDRESS),
            )
            Arena.ofConfined().use { scoped ->
                val getArgs = scoped.allocate(32)
                getArgs.set(JAVA_LONG, 0L, 32L)            // struct_size
                getArgs.set(ADDRESS, 8L, MemorySegment.NULL)
                getArgs.set(ADDRESS, 16L, ctx)
                getArgs.set(ADDRESS, 24L, MemorySegment.NULL)
                val err = streamGetFn.invokeExact(getArgs) as MemorySegment
                check(err.address() == 0L) {
                    "XLA_FFI_Stream_Get failed (XLA_FFI_Error* at 0x${err.address().toString(16)})"
                }
                return getArgs.get(ADDRESS, 24L)
            }
        }

        /** Builds an `XLA_FFI_Error*` via `XLA_FFI_Error_Create` — return
         * this from a handler to fail the execution with [message] instead
         * of crashing the JVM. XLA copies the message during the call.
         * `XLA_FFI_Error_Create_Args`: struct_size(0), ext(8), message(16),
         * errc(int 24) — 32 bytes. */
        fun createError(message: String, code: Int = ERROR_CODE_INTERNAL): MemorySegment {
            val api = frame.get(ADDRESS, OFF_FRAME_API).reinterpret(256)
            val createFn = PjrtFfm.LINKER.downcallHandle(
                api.get(ADDRESS, OFF_API_ERROR_CREATE).reinterpret(Long.MAX_VALUE),
                FunctionDescriptor.of(ADDRESS, ADDRESS),
            )
            Arena.ofConfined().use { scoped ->
                val createArgs = scoped.allocate(32)
                createArgs.set(JAVA_LONG, 0L, 32L)
                createArgs.set(ADDRESS, 8L, MemorySegment.NULL)
                createArgs.set(ADDRESS, 16L, scoped.allocateFrom(message))
                createArgs.set(JAVA_INT, 24L, code)
                return createFn.invokeExact(createArgs) as MemorySegment
            }
        }

        private fun decodeBuffers(seqOffset: Long, valuesOffset: Long): List<Buffer> {
            val size = frame.get(JAVA_LONG, seqOffset + OFF_SEQ_SIZE)
            if (size == 0L) return emptyList()
            val values = frame.get(ADDRESS, seqOffset + valuesOffset).reinterpret(size * 8)
            val types = frame.get(ADDRESS, seqOffset + OFF_SEQ_TYPES).reinterpret(size * 4)
            return List(size.toInt()) { i ->
                check(types.get(JAVA_INT, i * 4L) == ARG_TYPE_BUFFER) {
                    "arg/ret $i has non-BUFFER type ${types.get(JAVA_INT, i * 4L)}"
                }
                val buf = values.get(ADDRESS, i * 8L).reinterpret(48)
                val rank = buf.get(JAVA_LONG, OFF_BUF_RANK)
                val dims = if (rank == 0L) LongArray(0) else {
                    val dimsSeg = buf.get(ADDRESS, OFF_BUF_DIMS).reinterpret(rank * 8)
                    LongArray(rank.toInt()) { d -> dimsSeg.get(JAVA_LONG, d * 8L) }
                }
                Buffer(
                    dtype = buf.get(JAVA_INT, OFF_BUF_DTYPE),
                    dataAddress = buf.get(ADDRESS, OFF_BUF_DATA).address(),
                    dims = dims,
                )
            }
        }

        private fun decodeAttrs(): Map<String, AttrValue> {
            val base = OFF_FRAME_ATTRS
            val size = frame.get(JAVA_LONG, base + OFF_SEQ_SIZE)
            if (size == 0L) return emptyMap()
            val types = frame.get(ADDRESS, base + OFF_SEQ_TYPES).reinterpret(size * 4)
            val names = frame.get(ADDRESS, base + OFF_ATTRS_NAMES).reinterpret(size * 8)
            val values = frame.get(ADDRESS, base + OFF_ATTRS_VALUES).reinterpret(size * 8)
            val out = LinkedHashMap<String, AttrValue>()
            for (i in 0 until size.toInt()) {
                val name = readByteSpan(names.get(ADDRESS, i * 8L))
                val valuePtr = values.get(ADDRESS, i * 8L)
                out[name] = when (val attrType = types.get(JAVA_INT, i * 4L)) {
                    ATTR_TYPE_STRING -> AttrValue.Str(readByteSpan(valuePtr))
                    ATTR_TYPE_SCALAR -> {
                        val scalar = valuePtr.reinterpret(16)
                        val valueSeg = scalar.get(ADDRESS, 8L).reinterpret(8)
                        when (val dtype = scalar.get(JAVA_INT, 0L)) {
                            DTYPE_S32 -> AttrValue.I32(valueSeg.get(JAVA_INT, 0L))
                            DTYPE_S64 -> AttrValue.I64(valueSeg.get(JAVA_LONG, 0L))
                            DTYPE_F32 -> AttrValue.F32(valueSeg.get(JAVA_FLOAT, 0L))
                            DTYPE_F64 -> AttrValue.F64(valueSeg.get(JAVA_DOUBLE, 0L))
                            else -> AttrValue.Raw(dtype, valuePtr)
                        }
                    }
                    else -> AttrValue.Raw(attrType, valuePtr)
                }
            }
            return out
        }

        private fun readByteSpan(spanPtr: MemorySegment): String {
            val span = spanPtr.reinterpret(16)
            val len = span.get(JAVA_LONG, 8L)
            if (len == 0L) return ""
            val ptr = span.get(ADDRESS, 0L).reinterpret(len)
            val bytes = ByteArray(len.toInt()) { ptr.get(JAVA_BYTE, it.toLong()) }
            return String(bytes, StandardCharsets.UTF_8)
        }
    }

    /** Decodes the raw `XLA_FFI_CallFrame*` an [PjrtFfiRegistry.FfiExecuteHandler]
     * receives. */
    fun decode(framePtr: MemorySegment): Frame = Frame(framePtr.reinterpret(FRAME_MIN_SIZE))
}
