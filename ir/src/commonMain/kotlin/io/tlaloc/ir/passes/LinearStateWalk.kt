package io.tlaloc.ir.passes

import io.tlaloc.ir.CausalConv1dAttrs
import io.tlaloc.ir.GatedDeltaRuleAttrs
import io.tlaloc.ir.LinearStateRows
import kotlin.math.exp

/**
 * The reference walks of [io.tlaloc.ir.OpKind.CAUSAL_CONV1D] and
 * [io.tlaloc.ir.OpKind.GATED_DELTA_RULE], shared by [DxirInterpreter] and
 * [DxirInterpreterF64]. Sums are taken in double; [round] is applied to every
 * value the walk stores (outputs and carried state), so the f32 interpreter
 * passes a rounding to f32 and carries the state at the device's width, and
 * the f64 interpreter passes the identity.
 */
internal object LinearStateWalk {

    /**
     * Returns `(y, convState')`. A live token `t` of a row whose first live
     * token is `f` reads the sequence `X(t-K+1) .. X(t)`, where `X(u)` is
     * `x[u]` for `u >= f` and the history before it otherwise: the row's slot
     * of [state], or zeros when the row starts at position 0. The new state
     * of a row is the last `K-1` values of its sequence.
     */
    fun causalConv1d(
        p: CausalConv1dAttrs.Parsed,
        x: DoubleArray,
        w: DoubleArray,
        state: DoubleArray,
        slots: IntArray,
        positions: IntArray,
        round: (Double) -> Double,
        writes: IntArray? = null,
    ): Pair<DoubleArray, DoubleArray> {
        val (b, t, c, k, s) = p
        val km1 = k - 1
        val rows = LinearStateRows.plan(slots, positions, b, t, s, "CAUSAL_CONV1D")
        if (writes != null) LinearStateRows.checkWrites(writes, rows, t, s, "CAUSAL_CONV1D")
        val y = DoubleArray(b * t * c)
        val out = state.copyOf()
        for ((row, r) in rows.withIndex()) {
            if (r.slot < 0) continue
            val f = r.firstLive
            val hist = DoubleArray(km1 * c)
            if (!r.reset) state.copyInto(hist, 0, r.slot * km1 * c, (r.slot + 1) * km1 * c)
            fun xAt(u: Int, ch: Int): Double =
                if (u >= f) x[(row * t + u) * c + ch] else hist[(km1 - (f - u)) * c + ch]
            for (tok in f until t) {
                for (ch in 0 until c) {
                    var acc = 0.0
                    for (j in 0 until k) acc += w[j * c + ch] * xAt(tok - km1 + j, ch)
                    y[(row * t + tok) * c + ch] = round(acc)
                }
            }
            // The state after token u is the last K-1 values of the sequence up to u.
            fun writeAfter(u: Int, slot: Int) {
                for (i in 0 until km1) {
                    for (ch in 0 until c) out[(slot * km1 + i) * c + ch] = round(xAt(u - km1 + 1 + i, ch))
                }
            }
            if (writes == null) {
                writeAfter(t - 1, r.slot)
            } else {
                for (tok in f until t) if (writes[row * t + tok] >= 0) writeAfter(tok, writes[row * t + tok])
            }
        }
        return y to out
    }

    /**
     * A committed [io.tlaloc.ir.OpKind.CAUSAL_CONV1D] ([CausalConv1dAttrs.COMMIT]):
     * returns `(y, convState', records')`. Each live row's starting history is the
     * last K-1 values of `state[base]` followed by the base slot's first `count`
     * records (zeros for a row at position 0); its tokens run from it as in
     * [causalConv1d]. FINAL writes the history after the last token at the base,
     * RECORDS the starting history at the target and the row's live inputs, in
     * order, as the target's first T records (zero for its dead tokens; records past
     * T are left as they are).
     */
    fun causalConv1dCommitted(
        p: CausalConv1dAttrs.Parsed,
        x: DoubleArray,
        w: DoubleArray,
        state: DoubleArray,
        slots: IntArray,
        positions: IntArray,
        commit: IntArray,
        records: DoubleArray,
        round: (Double) -> Double,
    ): Triple<DoubleArray, DoubleArray, DoubleArray> {
        val mode = requireNotNull(p.commit) { "LinearStateWalk: CAUSAL_CONV1D is not committed" }
        val b = p.batch; val t = p.tokens; val c = p.channels; val k = p.kernel
        val km1 = k - 1
        val rows = LinearStateRows.plan(slots, positions, b, t, p.numSlots, "CAUSAL_CONV1D")
        val keep = mode == GatedDeltaRuleAttrs.Commit.RECORDS
        LinearStateRows.checkCommit(commit, rows, p.numSlots, p.recordTokens, keep, "CAUSAL_CONV1D")
        val slotRecords = p.recordTokens * c
        val y = DoubleArray(b * t * c)
        val out = state.copyOf()
        val newRecords = records.copyOf()
        for ((row, r) in rows.withIndex()) {
            if (r.slot < 0) continue
            val base = commit[row * 3]
            val target = commit[row * 3 + 1]
            val count = commit[row * 3 + 2]
            val f = r.firstLive
            // The history, then the base's first count records: the start is the last K-1 of them.
            val hist = DoubleArray(km1 * c)
            if (!r.reset) {
                val seq = DoubleArray((km1 + count) * c)
                state.copyInto(seq, 0, base * km1 * c, (base + 1) * km1 * c)
                records.copyInto(seq, km1 * c, base * slotRecords, base * slotRecords + count * c)
                seq.copyInto(hist, 0, count * c, (count + km1) * c)
            }
            fun xAt(u: Int, ch: Int): Double =
                if (u >= f) x[(row * t + u) * c + ch] else hist[(km1 - (f - u)) * c + ch]
            for (tok in f until t) {
                for (ch in 0 until c) {
                    var acc = 0.0
                    for (j in 0 until k) acc += w[j * c + ch] * xAt(tok - km1 + j, ch)
                    y[(row * t + tok) * c + ch] = round(acc)
                }
            }
            if (keep) {
                for (i in 0 until km1 * c) out[target * km1 * c + i] = round(hist[i])
                // The step's T records; a records pool of more (a wider verify step's) keeps the rest.
                newRecords.fill(0.0, target * slotRecords, target * slotRecords + t * c)
                for (tok in f until t) {
                    for (ch in 0 until c) newRecords[target * slotRecords + (tok - f) * c + ch] = round(x[(row * t + tok) * c + ch])
                }
            } else {
                for (i in 0 until km1) {
                    for (ch in 0 until c) out[(base * km1 + i) * c + ch] = round(xAt(t - km1 + i, ch))
                }
            }
        }
        return Triple(y, out, newRecords)
    }

    /**
     * Returns `(out, state')`. Per live token and value head `h` (reading
     * key head `h / group`), with the row's state `S` `[Dk, Dv]`:
     * ```
     *   S = S * exp(g)
     *   S = S + k ⊗ (beta * (v - kᵀS))
     *   out = qᵀS
     * ```
     * transformers' `torch_recurrent_gated_delta_rule`, after its L2 norms
     * and query scale (which the caller applies).
     */
    fun gatedDeltaRule(
        p: GatedDeltaRuleAttrs.Parsed,
        q: DoubleArray,
        k: DoubleArray,
        v: DoubleArray,
        g: DoubleArray,
        beta: DoubleArray,
        state: DoubleArray,
        slots: IntArray,
        positions: IntArray,
        round: (Double) -> Double,
        writes: IntArray? = null,
    ): Pair<DoubleArray, DoubleArray> {
        val b = p.batch; val t = p.tokens; val hk = p.keyHeads; val hv = p.valueHeads
        val dk = p.keyDim; val dv = p.valueDim; val group = p.group
        val rows = LinearStateRows.plan(slots, positions, b, t, p.numSlots, "GATED_DELTA_RULE")
        if (writes != null) LinearStateRows.checkWrites(writes, rows, t, p.numSlots, "GATED_DELTA_RULE")
        val headState = dk * dv
        val slotState = hv * headState
        val out = DoubleArray(b * t * hv * dv)
        val newState = state.copyOf()
        val kv = DoubleArray(dv)
        for ((row, r) in rows.withIndex()) {
            if (r.slot < 0) continue
            val st = DoubleArray(slotState)
            if (!r.reset) state.copyInto(st, 0, r.slot * slotState, (r.slot + 1) * slotState)
            for (tok in r.firstLive until t) {
                val bt = row * t + tok
                for (h in 0 until hv) {
                    val kh = h / group
                    val qk = (bt * hk + kh) * dk
                    val base = h * headState
                    val decay = exp(g[bt * hv + h])
                    for (i in 0 until headState) st[base + i] *= decay
                    for (j in 0 until dv) {
                        var acc = 0.0
                        for (i in 0 until dk) acc += st[base + i * dv + j] * k[qk + i]
                        kv[j] = acc
                    }
                    val bta = beta[bt * hv + h]
                    val vo = (bt * hv + h) * dv
                    for (j in 0 until dv) kv[j] = (v[vo + j] - kv[j]) * bta
                    for (i in 0 until dk) {
                        val ki = k[qk + i]
                        for (j in 0 until dv) st[base + i * dv + j] = round(st[base + i * dv + j] + ki * kv[j])
                    }
                    for (j in 0 until dv) {
                        var acc = 0.0
                        for (i in 0 until dk) acc += st[base + i * dv + j] * q[qk + i]
                        out[vo + j] = round(acc)
                    }
                }
                val w = writes?.get(bt) ?: -1
                if (w >= 0) st.copyInto(newState, w * slotState)
            }
            if (writes == null) st.copyInto(newState, r.slot * slotState)
        }
        return out to newState
    }

    /**
     * A committed [io.tlaloc.ir.OpKind.GATED_DELTA_RULE] ([GatedDeltaRuleAttrs.Commit]):
     * returns `(out, state', records')`. Each live row starts from `state[base]` with
     * the first `count` records of the base slot applied (zero for a row at position
     * 0), then runs its tokens as [gatedDeltaRule] does; FINAL writes the state after
     * its last token at the base, RECORDS the starting state at the target and one
     * record per token (a dead token's record is zero, which applies as the identity).
     */
    fun gatedDeltaRuleCommitted(
        p: GatedDeltaRuleAttrs.Parsed,
        q: DoubleArray,
        k: DoubleArray,
        v: DoubleArray,
        g: DoubleArray,
        beta: DoubleArray,
        state: DoubleArray,
        slots: IntArray,
        positions: IntArray,
        commit: IntArray,
        records: DoubleArray,
        round: (Double) -> Double,
    ): Triple<DoubleArray, DoubleArray, DoubleArray> {
        val mode = requireNotNull(p.commit) { "LinearStateWalk: GATED_DELTA_RULE is not committed" }
        val b = p.batch; val t = p.tokens; val hk = p.keyHeads; val hv = p.valueHeads
        val dk = p.keyDim; val dv = p.valueDim; val group = p.group
        val rows = LinearStateRows.plan(slots, positions, b, t, p.numSlots, "GATED_DELTA_RULE")
        val keep = mode == GatedDeltaRuleAttrs.Commit.RECORDS
        LinearStateRows.checkCommit(commit, rows, p.numSlots, p.recordTokens, keep, "GATED_DELTA_RULE")
        val width = GatedDeltaRuleAttrs.recordWidth(dk, dv)
        val slotRecords = p.recordTokens * hv * width
        val headState = dk * dv
        val slotState = hv * headState
        val out = DoubleArray(b * t * hv * dv)
        val newState = state.copyOf()
        val newRecords = records.copyOf()
        val kv = DoubleArray(dv)
        for ((row, r) in rows.withIndex()) {
            if (r.slot < 0) continue
            val base = commit[row * 3]
            val target = commit[row * 3 + 1]
            val count = commit[row * 3 + 2]
            val st = DoubleArray(slotState)
            if (!r.reset) {
                state.copyInto(st, 0, base * slotState, (base + 1) * slotState)
                for (j in 0 until count) {
                    for (h in 0 until hv) {
                        val rec = base * slotRecords + (j * hv + h) * width
                        val decay = exp(records[rec + dk + dv])
                        val sb = h * headState
                        for (i in 0 until headState) st[sb + i] *= decay
                        for (i in 0 until dk) {
                            val ki = records[rec + i]
                            for (jj in 0 until dv) st[sb + i * dv + jj] = round(st[sb + i * dv + jj] + ki * records[rec + dk + jj])
                        }
                    }
                }
            }
            if (keep) {
                st.copyInto(newState, target * slotState)
                newRecords.fill(0.0, target * slotRecords, target * slotRecords + t * hv * width)
            }
            for (tok in r.firstLive until t) {
                val bt = row * t + tok
                for (h in 0 until hv) {
                    val kh = h / group
                    val qk = (bt * hk + kh) * dk
                    val sb = h * headState
                    val decay = exp(g[bt * hv + h])
                    for (i in 0 until headState) st[sb + i] *= decay
                    for (j in 0 until dv) {
                        var acc = 0.0
                        for (i in 0 until dk) acc += st[sb + i * dv + j] * k[qk + i]
                        kv[j] = acc
                    }
                    val bta = beta[bt * hv + h]
                    val vo = (bt * hv + h) * dv
                    for (j in 0 until dv) kv[j] = round((v[vo + j] - kv[j]) * bta)
                    for (i in 0 until dk) {
                        val ki = k[qk + i]
                        for (j in 0 until dv) st[sb + i * dv + j] = round(st[sb + i * dv + j] + ki * kv[j])
                    }
                    for (j in 0 until dv) {
                        var acc = 0.0
                        for (i in 0 until dk) acc += st[sb + i * dv + j] * q[qk + i]
                        out[vo + j] = round(acc)
                    }
                    if (keep) {
                        val rec = target * slotRecords + (tok * hv + h) * width
                        for (i in 0 until dk) newRecords[rec + i] = k[qk + i]
                        for (j in 0 until dv) newRecords[rec + dk + j] = kv[j]
                        newRecords[rec + dk + dv] = g[bt * hv + h]
                    }
                }
            }
            if (!keep) st.copyInto(newState, base * slotState)
        }
        return Triple(out, newState, newRecords)
    }
}
