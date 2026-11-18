package io.tlaloc.autograd

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.ir.OpKind

/** The value of an entry a symbolic tape did not compute. */
internal val NO_VALUE = FloatArray(0)

class TapeEntry internal constructor(
    val id: Int,
    val op: OpKind?,
    val inputs: IntArray,
    val dims: IntArray,
    val value: FloatArray,
    /**
     * Marks a leaf as an opaque constant. The reverse walk short-circuits
     * any contribution targeting this entry (the user never asks for its gradient,
     * so there's no point materialising the VJP rule's dExp/dBase-like expression
     * for it). Only meaningful on leaves (`op == null`); ops carry their own
     * gradient-emission responsibilities and are never flagged constant.
     */
    val isConstant: Boolean = false,
    /**
     * Attributes carried alongside the op, mirroring the dxir
     * DxirOp.attrs map. Most ops don't need attrs (their semantics are fully
     * determined by inputs + dims); BROADCAST needs `broadcast_dimensions` so
     * captures and StableHLO emission can reconstruct the dxir shape without
     * extra inference. Only meaningful when `op != null`.
     */
    val attrs: Map<String, Any> = emptyMap(),
    /**
     * The entry's element dtype, [F32] unless stated otherwise. The
     * tape's VALUE cache stays a FloatArray for every dtype (the dxir
     * interpreter's own float-encoded environment convention — its
     * EMBEDDING/EMBEDDING_GRAD arms read indices via `toInt()`); what the dtype
     * governs is the [io.tlaloc.ir.DxirType] that `Tape.toDxirFunction` stamps
     * on the reproduced node, so an I32 index leaf comes out an I32-typed
     * `DxirParam` and `DxirReverseTransform` emits its ZEROS_LIKE
     * structural zero instead of trying to differentiate it. Float-encoded
     * integers are exact to 2²⁴ — the I32 leaf spelling asserts the bound.
     */
    val dtype: DType = F32,
) {
    val isLeaf: Boolean get() = op == null
    val size: Int get() = dims.fold(1) { a, d -> a * d }

    /** False for an entry of a symbolic tape that is not a constant: it has no value. */
    val hasValue: Boolean get() = value.size == size
}

/**
 * The recorded program. A [symbolic] tape records dims, dtypes, ops and attrs but computes no
 * values: leaves keep no copy of their tensors (constants keep theirs, the graph needs them) and
 * ops skip their host computation. A capture of a large model at a long sequence then costs
 * the recording alone; reading a value ([Tracer.peek], [Tracer.toDTensor]) of a non-constant
 * entry refuses, and so does an op whose result shape depends on its operands' values.
 */
class Tape(val symbolic: Boolean = false) {
    /** [compute] on an eager tape; no value on a symbolic one. */
    internal inline fun value(compute: () -> FloatArray): FloatArray = if (symbolic) NO_VALUE else compute()

    /** Refuses on a symbolic tape: [what] needs the operands' values. */
    internal fun requireValues(what: String) {
        require(!symbolic) { "symbolic capture: $what needs its operands' values, which a symbolic tape does not compute" }
    }

    private val _entries = mutableListOf<TapeEntry>()
    val entries: List<TapeEntry> get() = _entries

    internal fun leaf(
        dims: IntArray,
        value: FloatArray,
        isConstant: Boolean = false,
        dtype: DType = F32,
    ): TapeEntry {
        // A symbolic tape records one leaf per distinct constant: a model at a long sequence
        // builds the same tables (RoPE's cos and sin) in every layer.
        val key = if (symbolic && isConstant) ConstantKey(dtype, dims, value) else null
        key?.let { k -> constants[k]?.let { return it } }
        val entry = TapeEntry(
            _entries.size, op = null, inputs = IntArray(0), dims = dims, value = value,
            isConstant = isConstant, dtype = dtype,
        )
        _entries += entry
        if (key != null) constants[key] = entry
        return entry
    }

    private class ConstantKey(val dtype: DType, val dims: IntArray, val value: FloatArray) {
        private val hash = 31 * (31 * dtype.hashCode() + dims.contentHashCode()) + value.contentHashCode()
        override fun hashCode(): Int = hash
        override fun equals(other: Any?): Boolean =
            other is ConstantKey && other.hash == hash && other.dtype == dtype &&
                other.dims.contentEquals(dims) && other.value.contentEquals(value)
    }

    private val constants = HashMap<ConstantKey, TapeEntry>()

    /**
     * [dtype] `null` (the default at most op sites) means PROPAGATE:
     * any [io.tlaloc.core.BF16] input makes the result BF16, and mixing a BF16
     * input with an F32 input REFUSES BY NAME — StableHLO's elementwise ops
     * demand one element type, so a mixed-operand tape entry would emit
     * invalid MLIR downstream; the loud trace-time refusal beats that. I32
     * inputs are exempt from the homogeneity check (an index operand — e.g.
     * EMBEDDING — carries no precision). Ops that pass [dtype] explicitly
     * (CAST, the dtype-preserving RESHAPE/SLICE) bypass propagation unchanged.
     *
     * The BF16 VALUE INVARIANT, enforced centrally here exactly like the
     * interpreter's `snapToBf16` in `evalNode`: a BF16-typed
     * entry's FloatArray holds the f32-WIDENED FORMS OF BF16-ROUNDED numbers.
     * The op sites compute their forward in f32 through the host twins; this
     * one RNE snap at the entry's own output is the compute-in-f32,
     * round-once-at-op-boundary convention — so the tape's cached forward
     * agrees with what `DxirInterpreter` will compute for the reproduced
     * node BY CONSTRUCTION (same twins, same snap point). Idempotent, so
     * re-snapping an already-snapped array (RESHAPE of a bf16 entry, the
     * explicit CAST value) changes nothing.
     */
    /** Replaces op entry [id] with a copy carrying [key] = true in its attributes; returns the copy. */
    internal fun mark(id: Int, key: String): TapeEntry {
        val e = _entries[id]
        require(e.op != null) { "Tape.mark: entry $id is a leaf; only an op's result can be marked" }
        val marked = TapeEntry(e.id, e.op, e.inputs, e.dims, e.value, e.isConstant, e.attrs + (key to true), e.dtype)
        _entries[id] = marked
        return marked
    }

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
                        "not supported (spell them as cast leaves)."
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
