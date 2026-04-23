package io.tlaloc.autograd

import io.tlaloc.ir.OpKind

class TapeEntry internal constructor(
    val id: Int,
    val op: OpKind?,
    val inputs: IntArray,
    val dims: IntArray,
    val value: FloatArray,
) {
    val isLeaf: Boolean get() = op == null
    val size: Int get() = value.size
}

class Tape {
    private val _entries = mutableListOf<TapeEntry>()
    val entries: List<TapeEntry> get() = _entries

    internal fun leaf(dims: IntArray, value: FloatArray): TapeEntry {
        val entry = TapeEntry(_entries.size, op = null, inputs = IntArray(0), dims = dims, value = value)
        _entries += entry
        return entry
    }

    internal fun op(op: OpKind, inputs: IntArray, dims: IntArray, value: FloatArray): TapeEntry {
        val entry = TapeEntry(_entries.size, op, inputs, dims, value)
        _entries += entry
        return entry
    }

    fun size(): Int = _entries.size
}

internal fun sizeOf(dims: IntArray): Int =
    if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }
