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

    /**
     * §0.4.458 (G1d) — [dtype] `null` (every pre-G1d op site) means PROPAGATE:
     * any [io.tlaloc.core.BF16] input makes the result BF16, and mixing a BF16
     * input with an F32 input REFUSES BY NAME — StableHLO's elementwise ops
     * demand one element type, so a mixed-operand tape entry would emit
     * invalid MLIR downstream; the loud trace-time refusal beats that. I32
     * inputs are exempt from the homogeneity check (an index operand — e.g.
     * EMBEDDING — carries no precision). Ops that pass [dtype] explicitly
     * (CAST, the dtype-preserving RESHAPE/SLICE) bypass propagation unchanged.
     *
     * The BF16 VALUE INVARIANT, enforced centrally here exactly like the
     * interpreter's `snapToBf16` in `evalNode` (§0.4.456): a BF16-typed
     * entry's FloatArray holds the f32-WIDENED FORMS OF BF16-ROUNDED numbers.
     * The op sites compute their forward in f32 through the host twins; this
     * one RNE snap at the entry's own output is the compute-in-f32,
     * round-once-at-op-boundary convention — so the tape's cached forward
     * agrees with what `DxirInterpreter` will compute for the reproduced
     * node BY CONSTRUCTION (same twins, same snap point). Idempotent, so
     * re-snapping an already-snapped array (RESHAPE of a bf16 entry, the
     * explicit CAST value) changes nothing.
     */
    internal fun op(
        op: OpKind,
        inputs: IntArray,
        dims: IntArray,
        value: FloatArray,
        attrs: Map<String, Any> = emptyMap(),
        dtype: DType? = null,
    ): TapeEntry {
        val resolved = dtype ?: run {
            var sawBf16 = false
            var sawF32 = false
            for (id in inputs) {
                when (_entries[id].dtype) {
                    io.tlaloc.core.BF16 -> sawBf16 = true
                    F32 -> sawF32 = true
                    else -> {}
                }
            }
            if (sawBf16) {
                require(!sawF32) {
                    "Tape.op($op): mixed BF16/F32 operands — bf16 compute is dtype-homogeneous " +
                        "(StableHLO elementwise ops take one element type). Cast the f32 operand " +
                        "with Tracer.cast(BF16) first; f32 CONSTANT leaves in a bf16 region are " +
                        "the known G1d limitation (spell them as cast leaves)."
                }
                io.tlaloc.core.BF16
            } else F32
        }
        val stored =
            if (resolved == io.tlaloc.core.BF16) {
                FloatArray(value.size) {
                    io.tlaloc.core.bf16BitsToFloat(io.tlaloc.core.floatToBf16Bits(value[it]))
                }
            } else value
        val entry = TapeEntry(_entries.size, op, inputs, dims, stored, attrs = attrs, dtype = resolved)
        _entries += entry
        return entry
    }

    fun size(): Int = _entries.size
}

internal fun sizeOf(dims: IntArray): Int =
    if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }
