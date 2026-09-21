package io.tlaloc.autograd

import io.tlaloc.core.DType
import io.tlaloc.core.F32
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
    /**
     * §0.4.80 — attributes carried alongside the op, mirroring the dxir
     * DxirOp.attrs map. Most ops don't need attrs (their semantics are fully
     * determined by inputs + dims); BROADCAST needs `broadcast_dimensions` so
     * captures and StableHLO emission can reconstruct the dxir shape without
     * extra inference. Only meaningful when `op != null`.
     */
    val attrs: Map<String, Any> = emptyMap(),
    /**
     * §0.4.442 — the entry's element dtype, [F32] unless stated otherwise. The
     * tape's VALUE cache stays a FloatArray for every dtype (the dxir
     * interpreter's own float-encoded environment convention — its
     * EMBEDDING/EMBEDDING_GRAD arms read indices via `toInt()`); what the dtype
     * governs is the [io.tlaloc.ir.DxirType] that `Tape.toDxirFunction` stamps
     * on the reproduced node, so an I32 index leaf comes out an I32-typed
     * `DxirParam` and `DxirReverseTransform` emits its §0.4.419 ZEROS_LIKE
     * structural zero instead of trying to differentiate it. Float-encoded
     * integers are exact to 2²⁴ — the I32 leaf spelling asserts the bound.
     */
    val dtype: DType = F32,
) {
    val isLeaf: Boolean get() = op == null
    val size: Int get() = value.size
}

class Tape {
    private val _entries = mutableListOf<TapeEntry>()
    val entries: List<TapeEntry> get() = _entries

    internal fun leaf(
        dims: IntArray,
        value: FloatArray,
        isConstant: Boolean = false,
        dtype: DType = F32,
    ): TapeEntry {
        val entry = TapeEntry(
            _entries.size, op = null, inputs = IntArray(0), dims = dims, value = value,
            isConstant = isConstant, dtype = dtype,
        )
        _entries += entry
        return entry
    }

    internal fun op(
        op: OpKind,
        inputs: IntArray,
        dims: IntArray,
        value: FloatArray,
        attrs: Map<String, Any> = emptyMap(),
        dtype: DType = F32,
    ): TapeEntry {
        val entry = TapeEntry(_entries.size, op, inputs, dims, value, attrs = attrs, dtype = dtype)
        _entries += entry
        return entry
    }

    fun size(): Int = _entries.size
}

internal fun sizeOf(dims: IntArray): Int =
    if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }
