package io.tlaloc.ir

import io.tlaloc.core.DType

data class DxirType(val dtype: DType, val dims: List<Int>) {
    val rank: Int get() = dims.size
    val isScalar: Boolean get() = dims.isEmpty()
    val elementCount: Long get() = if (dims.isEmpty()) 1L else dims.fold(1L) { acc, d -> acc * d }

    override fun toString(): String =
        if (dims.isEmpty()) dtype.name else "${dtype.name}[${dims.joinToString(",")}]"
}
