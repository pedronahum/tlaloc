package io.tlaloc.ir

import io.tlaloc.core.DType
import io.tlaloc.core.I32
import io.tlaloc.core.I64

// The operand conventions of the two state-carrying ops of a Gated DeltaNet
// layer, [OpKind.CAUSAL_CONV1D] and [OpKind.GATED_DELTA_RULE]: one parser per
// op, shared by the interpreters, the emitter, the renderer's refusal and the
// cost model, and the row rules both ops apply to their token slots.

/**
 * How a row of a `[B, T]` token block maps onto a state pool, under the rules
 * shared by [OpKind.CAUSAL_CONV1D] and [OpKind.GATED_DELTA_RULE]:
 *
 * - a token is LIVE when its slot is `>= 0`; dead tokens (padding) come first
 *   in a row and live ones after them (the prefill convention: right-aligned);
 * - the live tokens of a row carry one slot, in `[0, numSlots)`, and no two
 *   rows carry the same slot;
 * - a row whose first live token is at position 0 starts from zero state;
 *   any other row starts from the pool at its slot. Only the first live token
 *   of a row may be at position 0.
 */
data class LinearStateRow(
    /** The row's slot, or -1 when the row has no live token. */
    val slot: Int,
    /** Index of the first live token; `T` when there is none. */
    val firstLive: Int,
    /** True when the row starts from zero state. */
    val reset: Boolean,
)

object LinearStateRows {

    /**
     * Read and check the rows of [slots] and [positions] (both `[b * t]`,
     * row-major). Refuses by name, with [what] in the message, every block
     * the rules above do not describe: the emitted form relies on them.
     */
    fun plan(slots: IntArray, positions: IntArray, b: Int, t: Int, numSlots: Int, what: String): List<LinearStateRow> {
        require(slots.size == b * t && positions.size == b * t) {
            "$what: ${slots.size} slots and ${positions.size} positions for a [$b, $t] token block"
        }
        val used = HashSet<Int>()
        return List(b) { row ->
            var first = t
            for (i in 0 until t) {
                if (slots[row * t + i] >= 0) {
                    first = i
                    break
                }
            }
            if (first == t) return@List LinearStateRow(-1, t, false)
            val slot = slots[row * t + first]
            require(slot < numSlots) {
                "$what: row $row names state slot $slot but the pool holds $numSlots slots"
            }
            for (i in first until t) {
                val s = slots[row * t + i]
                require(s >= 0) {
                    "$what: row $row has a padding token (slot $s) at $i after a live token at " +
                        "$first; padding comes first in a row"
                }
                require(s == slot) {
                    "$what: row $row's live tokens name slots $slot and $s; a row is one sequence " +
                        "and carries one slot"
                }
                require(i == first || positions[row * t + i] != 0) {
                    "$what: row $row has a live token at position 0 at $i, after its first live " +
                        "token; only a row's first token may start a sequence"
                }
            }
            require(used.add(slot)) {
                "$what: state slot $slot is named by more than one row; each sequence owns its slot"
            }
            LinearStateRow(slot, first, positions[row * t + first] == 0)
        }
    }

    internal fun isIntegral(d: DType): Boolean = d == I32 || d == I64
}

/**
 * [OpKind.CAUSAL_CONV1D]'s operands:
 * ```
 *   0 x          [B, T, C]      float
 *   1 weight     [K, C]         float
 *   2 convState  [S, K-1, C]    float (x's dtype)
 *   3 tokenSlots [B, T]         integer
 *   4 positions  [B, T]         integer
 *   -> y         [B, T, C]
 *   -> convState [S, K-1, C]
 * ```
 * No attributes: every size is an operand dim.
 */
object CausalConv1dAttrs {

    data class Parsed(val batch: Int, val tokens: Int, val channels: Int, val kernel: Int, val numSlots: Int)

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.CAUSAL_CONV1D) { "$layer: CausalConv1dAttrs.parse called on ${op.op}" }
        require(op.operands.size == 5) {
            "$layer: CAUSAL_CONV1D takes 5 operands (x, weight, convState, tokenSlots, positions), " +
                "got ${op.operands.size}"
        }
        require(op.attrs.isEmpty()) { "$layer: CAUSAL_CONV1D takes no attributes, got ${op.attrs.keys}" }
        val (x, w, st, sl, pos) = op.operands.map { it.type }
        require(x.rank == 3) { "$layer: CAUSAL_CONV1D x must be [B, T, C], got ${x.dims}" }
        val (b, t, c) = x.dims
        require(w.rank == 2 && w.dims[1] == c && w.dims[0] >= 1) {
            "$layer: CAUSAL_CONV1D weight must be [K, $c], got ${w.dims}"
        }
        val k = w.dims[0]
        require(st.rank == 3 && st.dims[1] == k - 1 && st.dims[2] == c && st.dims[0] >= 1) {
            "$layer: CAUSAL_CONV1D convState must be [S, ${k - 1}, $c], got ${st.dims}"
        }
        require(w.dtype == x.dtype && st.dtype == x.dtype) {
            "$layer: CAUSAL_CONV1D x, weight and convState share one dtype, got ${x.dtype}, ${w.dtype}, ${st.dtype}"
        }
        for ((name, ty) in listOf("tokenSlots" to sl, "positions" to pos)) {
            require(ty.dims == listOf(b, t) && LinearStateRows.isIntegral(ty.dtype)) {
                "$layer: CAUSAL_CONV1D $name must be integer [$b, $t], got $ty"
            }
        }
        require(op.types.size == 2 && op.types[0] == x && op.types[1] == st) {
            "$layer: CAUSAL_CONV1D returns (y $x, convState $st), got ${op.types}"
        }
        return Parsed(b, t, c, k, st.dims[0])
    }
}

/**
 * [OpKind.GATED_DELTA_RULE]'s operands:
 * ```
 *   0 query      [B, T, Hk, Dk]    float, L2-normalized and scaled by the caller
 *   1 key        [B, T, Hk, Dk]    float, L2-normalized by the caller
 *   2 value      [B, T, Hv, Dv]    float
 *   3 g          [B, T, Hv]        float, the log decay (<= 0)
 *   4 beta       [B, T, Hv]        float
 *   5 state      [S, Hv, Dk, Dv]   float
 *   6 tokenSlots [B, T]            integer
 *   7 positions  [B, T]            integer
 *   -> out       [B, T, Hv, Dv]
 *   -> state     [S, Hv, Dk, Dv]
 * ```
 * Value head `j` reads query/key head `j / (Hv / Hk)`. No attributes.
 */
object GatedDeltaRuleAttrs {

    data class Parsed(
        val batch: Int,
        val tokens: Int,
        val keyHeads: Int,
        val valueHeads: Int,
        val keyDim: Int,
        val valueDim: Int,
        val numSlots: Int,
    ) {
        val group: Int get() = valueHeads / keyHeads
    }

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.GATED_DELTA_RULE) { "$layer: GatedDeltaRuleAttrs.parse called on ${op.op}" }
        require(op.operands.size == 8) {
            "$layer: GATED_DELTA_RULE takes 8 operands (query, key, value, g, beta, state, " +
                "tokenSlots, positions), got ${op.operands.size}"
        }
        require(op.attrs.isEmpty()) { "$layer: GATED_DELTA_RULE takes no attributes, got ${op.attrs.keys}" }
        val ts = op.operands.map { it.type }
        val (q, k, v, g, beta) = ts
        val st = ts[5]
        require(q.rank == 4) { "$layer: GATED_DELTA_RULE query must be [B, T, Hk, Dk], got ${q.dims}" }
        val (b, t, hk, dk) = q.dims
        require(k == q) { "$layer: GATED_DELTA_RULE key must match the query $q, got $k" }
        require(v.rank == 4 && v.dims[0] == b && v.dims[1] == t) {
            "$layer: GATED_DELTA_RULE value must be [$b, $t, Hv, Dv], got ${v.dims}"
        }
        val hv = v.dims[2]
        val dv = v.dims[3]
        require(hv % hk == 0) {
            "$layer: GATED_DELTA_RULE has $hv value heads and $hk key heads; value head j reads key " +
                "head j / (Hv / Hk), so Hk must divide Hv"
        }
        for ((name, ty) in listOf("g" to g, "beta" to beta)) {
            require(ty.dims == listOf(b, t, hv)) { "$layer: GATED_DELTA_RULE $name must be [$b, $t, $hv], got ${ty.dims}" }
        }
        require(st.rank == 4 && st.dims.drop(1) == listOf(hv, dk, dv) && st.dims[0] >= 1) {
            "$layer: GATED_DELTA_RULE state must be [S, $hv, $dk, $dv], got ${st.dims}"
        }
        val dt = q.dtype
        require(listOf(v, g, beta, st).all { it.dtype == dt }) {
            "$layer: GATED_DELTA_RULE float operands share one dtype, got ${ts.take(6).map { it.dtype }}"
        }
        for ((name, ty) in listOf("tokenSlots" to ts[6], "positions" to ts[7])) {
            require(ty.dims == listOf(b, t) && LinearStateRows.isIntegral(ty.dtype)) {
                "$layer: GATED_DELTA_RULE $name must be integer [$b, $t], got $ty"
            }
        }
        require(op.types.size == 2 && op.types[0] == v && op.types[1] == st) {
            "$layer: GATED_DELTA_RULE returns (out $v, state $st), got ${op.types}"
        }
        return Parsed(b, t, hk, hv, dk, dv, st.dims[0])
    }
}
