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
