package io.tlaloc.autograd

import io.tlaloc.ir.OpKind

class TapeEntry internal constructor(
    val id: Int,
    val op: OpKind?,
    val inputs: IntArray,
    val dims: IntArray,
    val value: FloatArray,
    /**
     * §0.4.65 — marks a leaf as an opaque constant. The reverse walk short-circuits
     * any contribution targeting this entry (the user never asks for its gradient,
     * so there's no point materialising the VJP rule's dExp/dBase-like expression
     * for it). Only meaningful on leaves (`op == null`); ops carry their own
     * gradient-emission responsibilities and are never flagged constant.
     */
    val isConstant: Boolean = false,
) {
    val isLeaf: Boolean get() = op == null
    val size: Int get() = value.size
}

class Tape {
    private val _entries = mutableListOf<TapeEntry>()
    val entries: List<TapeEntry> get() = _entries

    internal fun leaf(dims: IntArray, value: FloatArray, isConstant: Boolean = false): TapeEntry {
        val entry = TapeEntry(_entries.size, op = null, inputs = IntArray(0), dims = dims, value = value, isConstant = isConstant)
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
