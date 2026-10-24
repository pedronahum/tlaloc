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

    /**
     * Check a block's per-token write slots (`[b * t]`, row-major) against
     * its [rows]: a dead token writes nothing (-1), a live token writes the
     * state after it to its slot or nothing, and no slot is written twice.
     * A write slot may equal a row's read slot: every row's state is read
     * before any is written.
     */
    fun checkWrites(writes: IntArray, rows: List<LinearStateRow>, t: Int, numSlots: Int, what: String) {
        require(writes.size == rows.size * t) { "$what: ${writes.size} write slots for a [${rows.size}, $t] token block" }
        val used = HashSet<Int>()
        for ((row, r) in rows.withIndex()) {
            for (i in 0 until t) {
                val w = writes[row * t + i]
                if (w < 0) continue
                require(r.slot >= 0 && i >= r.firstLive) {
                    "$what: row $row writes slot $w after a padding token at $i; only live tokens write"
                }
                require(w < numSlots) { "$what: row $row writes slot $w but the pool holds $numSlots slots" }
                require(used.add(w)) { "$what: state slot $w is written by more than one token" }
            }
        }
    }

    /**
     * Check a committed call's `commit` rows (`[b * 3]`: base, target, count) against
     * its [rows]: a live row's base in the pool, its count in `0..recordTokens`, no
     * two rows sharing a base; with [records] a target in the pool, not its own base
     * and no other row's base or target.
     */
    fun checkCommit(commit: IntArray, rows: List<LinearStateRow>, numSlots: Int, recordTokens: Int, records: Boolean, what: String) {
        require(commit.size == rows.size * 3) { "$what: ${commit.size} commit values for ${rows.size} rows" }
        val used = HashSet<Int>()
        for ((row, r) in rows.withIndex()) {
            if (r.slot < 0) continue
            val base = commit[row * 3]
            val target = commit[row * 3 + 1]
            val count = commit[row * 3 + 2]
            require(base in 0 until numSlots) { "$what: row $row commits from slot $base but the pool holds $numSlots slots" }
            require(count in 0..recordTokens) { "$what: row $row applies $count records; a slot holds $recordTokens" }
            require(used.add(base)) { "$what: state slot $base is a base of more than one row" }
            if (records) {
                require(target in 0 until numSlots && target != base) {
                    "$what: row $row's target slot $target must be in the pool and not its base $base"
                }
                require(used.add(target)) { "$what: state slot $target is used by more than one row" }
            }
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
 *   5 writeSlots [B, T]         integer, optional
 *   -> y         [B, T, C]
 *   -> convState [S, K-1, C]
 * ```
 * Without `writeSlots` a row's state after its last token replaces the
 * state at its slot. With them the row's slot is only read, and the state
 * after token `i` is written at `writeSlots[b, i]` (-1: not written; see
 * [LinearStateRows.checkWrites]): a speculative step keeps the state after
 * each of its tokens, and the next step reads the one it accepted.
 * No attributes: every size is an operand dim.
 */
object CausalConv1dAttrs {

    data class Parsed(
        val batch: Int,
        val tokens: Int,
        val channels: Int,
        val kernel: Int,
        val numSlots: Int,
        /** Whether operand 5, `writeSlots`, is present. */
        val perTokenWrites: Boolean = false,
    )

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.CAUSAL_CONV1D) { "$layer: CausalConv1dAttrs.parse called on ${op.op}" }
        require(op.operands.size == 5 || op.operands.size == 6) {
            "$layer: CAUSAL_CONV1D takes 5 operands (x, weight, convState, tokenSlots, positions), " +
                "or 6 with writeSlots, got ${op.operands.size}"
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
        val ints = listOf("tokenSlots" to sl, "positions" to pos) +
            (if (op.operands.size == 6) listOf("writeSlots" to op.operands[5].type) else emptyList())
        for ((name, ty) in ints) {
            require(ty.dims == listOf(b, t) && LinearStateRows.isIntegral(ty.dtype)) {
                "$layer: CAUSAL_CONV1D $name must be integer [$b, $t], got $ty"
            }
        }
        require(op.types.size == 2 && op.types[0] == x && op.types[1] == st) {
            "$layer: CAUSAL_CONV1D returns (y $x, convState $st), got ${op.types}"
        }
        return Parsed(b, t, c, k, st.dims[0], op.operands.size == 6)
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
 *   8 writeSlots [B, T]            integer, optional
 *   -> out       [B, T, Hv, Dv]
 *   -> state     [S, Hv, Dk, Dv]
 * ```
 * Value head `j` reads query/key head `j / (Hv / Hk)`. `writeSlots` as for
 * [CausalConv1dAttrs]: with them the row's slot is only read and the state
 * after token `i` goes to `writeSlots[b, i]`.
 *
 * With [COMMIT] the state is committed once per call instead ([Commit]):
 * ```
 *   8 commit     [B, 3]                     integer: (base P, target Q, count c) per row
 *   9 records    [S, R, Hv, Dk + Dv + 4]    f32: per slot, R tokens' (k, kv, g)
 *   -> out, state, records
 * ```
 * A live row's starting state is `state[P]` with the first `c` records of slot P
 * applied (`S = S exp(g); S = S + k kvᵀ`), or zero for a row starting at
 * position 0. [Commit.FINAL] then runs the row's tokens and writes the state
 * after the last at P. [Commit.RECORDS] (R == T) writes the starting state at
 * Q (≠ P) and, per token `t`, `records[Q, t]` = (the token's key repeated per value
 * head, `kv = beta (v - kᵀS)`, `g`): a verify step leaves the state after any
 * number of its tokens to be had from Q, without writing one per token. The
 * slot value in `tokenSlots` is not read in either mode, only its liveness.
 */
object GatedDeltaRuleAttrs {

    /**
     * Boolean: emit the CUDA kernel tlaloc_gated_delta (triton/kernels/gated_delta.cu)
     * where it applies (f32, a few tokens per row). Interpreters ignore it.
     */
    const val FUSED_KERNEL = "fused_kernel"

    /** String: the committed form ([Commit]), with operands 8 and 9. */
    const val COMMIT = "commit"

    /** The two committed forms. */
    enum class Commit(val attr: String) {
        /** The state after the row's last token at the base slot (a prefill call). */
        FINAL("final"),

        /** The starting state at the target slot and a record per token (a verify step). */
        RECORDS("records"),
    }

    /** Floats per record: the key (Dk), kv (Dv), g, and 3 of padding (16-byte rows). */
    fun recordWidth(keyDim: Int, valueDim: Int): Int = keyDim + valueDim + 4

    data class Parsed(
        val batch: Int,
        val tokens: Int,
        val keyHeads: Int,
        val valueHeads: Int,
        val keyDim: Int,
        val valueDim: Int,
        val numSlots: Int,
        /** Whether operand 8, `writeSlots`, is present. */
        val perTokenWrites: Boolean = false,
        /** The committed form, or null. */
        val commit: Commit? = null,
        /** Tokens per slot of the records pool (with [commit]). */
        val recordTokens: Int = 0,
    ) {
        val group: Int get() = valueHeads / keyHeads
    }

    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.GATED_DELTA_RULE) { "$layer: GatedDeltaRuleAttrs.parse called on ${op.op}" }
        require(op.attrs.keys.all { it == FUSED_KERNEL || it == COMMIT }) {
            "$layer: GATED_DELTA_RULE takes only $FUSED_KERNEL and $COMMIT, got ${op.attrs.keys}"
        }
        val commitAttr = op.attrs[COMMIT]
        val commit = commitAttr?.let { a -> Commit.entries.firstOrNull { it.attr == a } }
        require(commitAttr == null || commit != null) {
            "$layer: GATED_DELTA_RULE $COMMIT is '${Commit.FINAL.attr}' or '${Commit.RECORDS.attr}', got '$commitAttr'"
        }
        if (commit != null) {
            require(op.operands.size == 10) {
                "$layer: a committed GATED_DELTA_RULE takes 10 operands (query, key, value, g, beta, state, " +
                    "tokenSlots, positions, commit, records), got ${op.operands.size}"
            }
        } else {
            require(op.operands.size == 8 || op.operands.size == 9) {
                "$layer: GATED_DELTA_RULE takes 8 operands (query, key, value, g, beta, state, " +
                    "tokenSlots, positions), or 9 with writeSlots, got ${op.operands.size}"
            }
        }
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
        require(listOf(v, g, beta).all { it.dtype == dt }) {
            "$layer: GATED_DELTA_RULE float operands share one dtype, got ${ts.take(5).map { it.dtype }}"
        }
        // The state pool may be bf16 under f32 operands: each row's state is
        // widened when read, carried in f32 over the row's tokens and rounded
        // to nearest even when written.
        require(st.dtype == dt || (dt == io.tlaloc.core.F32 && st.dtype == io.tlaloc.core.BF16)) {
            "$layer: GATED_DELTA_RULE state is ${st.dtype}; it takes the operands' $dt, or bf16 under f32"
        }
        val ints = listOf("tokenSlots" to ts[6], "positions" to ts[7]) +
            (if (ts.size == 9) listOf("writeSlots" to ts[8]) else emptyList())
        for ((name, ty) in ints) {
            require(ty.dims == listOf(b, t) && LinearStateRows.isIntegral(ty.dtype)) {
                "$layer: GATED_DELTA_RULE $name must be integer [$b, $t], got $ty"
            }
        }
        if (commit == null) {
            require(op.types.size == 2 && op.types[0] == v && op.types[1] == st) {
                "$layer: GATED_DELTA_RULE returns (out $v, state $st), got ${op.types}"
            }
            return Parsed(b, t, hk, hv, dk, dv, st.dims[0], ts.size == 9)
        }
        val c = ts[8]
        require(c.dims == listOf(b, 3) && c.dtype == ts[6].dtype) {
            "$layer: GATED_DELTA_RULE commit must be [$b, 3] (base, target, count) of tokenSlots' ${ts[6].dtype}, got $c"
        }
        val rec = ts[9]
        val width = recordWidth(dk, dv)
        require(rec.dtype == io.tlaloc.core.F32 && rec.rank == 4 && rec.dims[0] == st.dims[0] && rec.dims[2] == hv && rec.dims[3] == width) {
            "$layer: GATED_DELTA_RULE records must be f32 [${st.dims[0]}, R, $hv, $width], got $rec"
        }
        require(commit != Commit.RECORDS || rec.dims[1] == t) {
            "$layer: GATED_DELTA_RULE commit '${Commit.RECORDS.attr}' keeps one record per token: R must be $t, got ${rec.dims[1]}"
        }
        require(op.types.size == 3 && op.types[0] == v && op.types[1] == st && op.types[2] == rec) {
            "$layer: a committed GATED_DELTA_RULE returns (out $v, state $st, records $rec), got ${op.types}"
        }
        return Parsed(b, t, hk, hv, dk, dv, st.dims[0], false, commit, rec.dims[1])
    }
}
