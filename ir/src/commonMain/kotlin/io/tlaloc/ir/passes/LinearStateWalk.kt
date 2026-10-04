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
}
