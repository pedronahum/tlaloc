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
 * §0.4.400 — host storage for integer tensors. The first user is `embedding`'s
 * index operand (a `DTensor<Rank1<N>, I32>` of vocab slots): indices are DATA
 * at the host level, not shape, so they need a real storage class rather than
 * the float view the dxir interpreter uses internally.
 */
class HostI32Storage(val data: IntArray) : TensorStorage {
    override val sizeBytes: Long get() = data.size.toLong() * I32.sizeBytes
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
