package io.tlaloc.ir

import io.tlaloc.core.DType

/**
 * SSA value type carried by every [DxirNode]: dtype + dim vector + optional
 * per-axis names.
 *
 * `axisNames` is the Layer 1 (§0.4.241+) extension that lets DXIR carry the
 * Kotlin-side `Named<N, A>` axis identifiers through to the StableHLO emitter
 * and SDY sharding anchors. When [axisNames] is empty (the default), the type
 * has no named axes — fully backwards-compatible with pre-Layer-1 DXIR. When
 * non-empty, it must align positionally with [dims]; individual entries may be
 * `null` to mark a single axis as unnamed within an otherwise-named type.
 */
data class DxirType(
    val dtype: DType,
    val dims: List<Int>,
    val axisNames: List<String?> = emptyList(),
) {
    init {
        require(axisNames.isEmpty() || axisNames.size == dims.size) {
            "DxirType axisNames size ${axisNames.size} must equal dims size ${dims.size} " +
                "(dtype=$dtype, dims=$dims, axisNames=$axisNames)"
        }
    }

    val rank: Int get() = dims.size
    val isScalar: Boolean get() = dims.isEmpty()
    val elementCount: Long get() = if (dims.isEmpty()) 1L else dims.fold(1L) { acc, d -> acc * d }

    /** True when the type carries at least one non-null axis name. */
    val hasNamedAxes: Boolean get() = axisNames.any { it != null }

    /** Axis name at [index], or null if unnamed / out of range. */
    fun axisNameOrNull(index: Int): String? = axisNames.getOrNull(index)

    override fun toString(): String =
        if (dims.isEmpty()) {
            dtype.name
        } else if (axisNames.isEmpty()) {
            "${dtype.name}[${dims.joinToString(",")}]"
        } else {
            val parts = dims.indices.joinToString(",") { i ->
                val n = axisNames[i]
                if (n != null) "${dims[i]}@$n" else dims[i].toString()
            }
            "${dtype.name}[$parts]"
        }
}
