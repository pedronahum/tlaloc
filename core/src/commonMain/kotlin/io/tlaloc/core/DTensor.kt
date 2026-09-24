package io.tlaloc.core

interface TensorStorage {
    val sizeBytes: Long
    fun release()
}

class HostF32Storage(val data: FloatArray) : TensorStorage {
    override val sizeBytes: Long get() = data.size.toLong() * F32.sizeBytes
    override fun release() = Unit
}

class HostF64Storage(val data: DoubleArray) : TensorStorage {
    override val sizeBytes: Long get() = data.size.toLong() * F64.sizeBytes
    override fun release() = Unit
}

/**
 * Host storage for integer tensors. The first user is `embedding`'s
 * index operand (a `DTensor<Rank1<N>, I32>` of vocab slots): indices are DATA
 * at the host level, not shape, so they need a real storage class rather than
 * the float view the dxir interpreter uses internally.
 */
class HostI32Storage(val data: IntArray) : TensorStorage {
    override val sizeBytes: Long get() = data.size.toLong() * I32.sizeBytes
    override fun release() = Unit
}

/**
 * Host storage for bf16 tensors. [data] holds the RAW
 * upper-16-bit patterns of the f32 values (bf16 is the truncated top half of
 * binary32), NOT numeric Short values: a Short here is only a 16-bit bucket.
 * Narrowing f32 -> bf16 rounds to nearest-even ([floatToBf16Bits], matching
 * XLA's convention); widening bf16 -> f32 is exact ([bf16BitsToFloat]). Host
 * compute never happens at bf16 width — widen, use the f32 kernels, narrow
 * (the compute-in-f32-store-bf16 convention, Bf16.kt).
 */
class HostBf16Storage(val data: ShortArray) : TensorStorage {
    override val sizeBytes: Long get() = data.size.toLong() * BF16.sizeBytes
    override fun release() = Unit
}

class DTensor<S : Shape, T : DType>(
    val storage: TensorStorage,
    val dims: IntArray,
    val dtype: T,
) : AutoCloseable {

    init {
        require(dims.all { it >= 0 }) { "tensor dims must be non-negative: ${dims.toList()}" }
    }

    val rank: Int get() = dims.size

    val size: Int get() = if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }

    override fun close() = storage.release()

    override fun toString(): String =
        "DTensor(dtype=${dtype.name}, dims=${dims.toList()})"
}
