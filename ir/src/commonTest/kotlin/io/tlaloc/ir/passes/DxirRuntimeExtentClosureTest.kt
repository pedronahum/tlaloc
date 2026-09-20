package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.399 — the runtime-extent family closes under differentiation.
 *
 * Until this slice the runtime-extent adjoint ops (SUM_TO §0.4.373, PAD_TO
 * §0.4.374, SLICE_LIKE Phase A2b) had NO VjpRule: any reverse transform over a
 * body containing one — which is what reverse-over-reverse IS, since the first
 * reverse pass emits them into its gradient body — failed loudly with
 * "no VJP rule registered for SUM_TO" (verified at §0.4.398 HEAD; the
 * forward-over-reverse HVP was never affected, §0.4.394's hessian route).
 *
 * The closure: each op's adjoint is its runtime-extent mirror —
 *   SUM_TO(value, template)         ⇄  BROADCAST_LIKE(upstream, template=value)
 *   PAD_TO(value, template, low)    ⇄  SLICE_AT(upstream, template=value, low)
 * with `value` becoming the SHAPE-ONLY template of its own adjoint (its runtime
 * dims are the target extents, unknowable under `grad {}`'s -1 sentinels), so
 * differentiating any number of times only ever alternates within the pair.
 * SLICE_LIKE stays ruleless: its window offset is a runtime SUM of prior
 * templates' extents, which no literal `low` can carry.
 */
class DxirRuntimeExtentClosureTest {

    private val scalar = DxirType(F32, emptyList())

    /** BROADCAST_LIKE is the numpy right-aligned stretch: pin the shape/value contract. */
    @Test
    fun broadcastLikeInterpreterPins() {
        // Equal-rank stretch U=[1,3] → T=[2,3].
        run {
            val fn = DxirBuilder.function("bl_row") {
                val v = param("v", DxirType(F32, listOf(1, 3)))
                val t = param("t", DxirType(F32, listOf(2, 3)))
                listOf(op(OpKind.BROADCAST_LIKE, listOf(v, t), DxirType(F32, listOf(2, 3))))
            }
            val v = floatArrayOf(1f, 2f, 3f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(6)))[0]
            assertEquals(6, out.size)
            val want = floatArrayOf(1f, 2f, 3f, 1f, 2f, 3f)
            assertTrue(want.indices.all { abs(out[it] - want[it]) < 1e-6f }, out.toList().toString())
        }
        // Equal-rank column stretch U=[2,1] → T=[2,3].
        run {
            val fn = DxirBuilder.function("bl_col") {
                val v = param("v", DxirType(F32, listOf(2, 1)))
                val t = param("t", DxirType(F32, listOf(2, 3)))
                listOf(op(OpKind.BROADCAST_LIKE, listOf(v, t), DxirType(F32, listOf(2, 3))))
            }
            val v = floatArrayOf(5f, -2f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(6)))[0]
            val want = floatArrayOf(5f, 5f, 5f, -2f, -2f, -2f)
            assertTrue(want.indices.all { abs(out[it] - want[it]) < 1e-6f }, out.toList().toString())
        }
        // Rank extension U=[2] → T=[3,2] (missing leading axis replicated).
        run {
            val fn = DxirBuilder.function("bl_lead") {
                val v = param("v", DxirType(F32, listOf(2)))
                val t = param("t", DxirType(F32, listOf(3, 2)))
                listOf(op(OpKind.BROADCAST_LIKE, listOf(v, t), DxirType(F32, listOf(3, 2))))
            }
            val v = floatArrayOf(7f, 9f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(6)))[0]
            val want = floatArrayOf(7f, 9f, 7f, 9f, 7f, 9f)
            assertTrue(want.indices.all { abs(out[it] - want[it]) < 1e-6f }, out.toList().toString())
        }
        // Identity fast path U == T: pass-through copy.
        run {
            val fn = DxirBuilder.function("bl_id") {
                val v = param("v", DxirType(F32, listOf(2, 2)))
                val t = param("t", DxirType(F32, listOf(2, 2)))
                listOf(op(OpKind.BROADCAST_LIKE, listOf(v, t), DxirType(F32, listOf(2, 2))))
            }
            val v = floatArrayOf(1f, 2f, 3f, 4f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(4)))[0]
            assertTrue(v.indices.all { abs(out[it] - v[it]) < 1e-6f }, out.toList().toString())
        }
    }

    /** SLICE_AT is the window at a literal offset: pin the contract, including low=0 identity. */
    @Test
    fun sliceAtInterpreterPins() {
        // Rank-2 interior window: v=[3,4], t=[2,2], low=[1,1] → rows 1..2, cols 1..2.
        run {
            val fn = DxirBuilder.function("sa_win") {
                val v = param("v", DxirType(F32, listOf(3, 4)))
                val t = param("t", DxirType(F32, listOf(2, 2)))
                listOf(
                    op(
                        OpKind.SLICE_AT, listOf(v, t), DxirType(F32, listOf(2, 2)),
                        attrs = mapOf("low" to listOf(1, 1)),
                    ),
                )
            }
            val v = FloatArray(12) { it.toFloat() }
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(4)))[0]
            val want = floatArrayOf(5f, 6f, 9f, 10f)
            assertTrue(want.indices.all { abs(out[it] - want[it]) < 1e-6f }, out.toList().toString())
        }
        // low = 0 everywhere with T == U: identity copy.
        run {
            val fn = DxirBuilder.function("sa_id") {
                val v = param("v", DxirType(F32, listOf(2, 2)))
                val t = param("t", DxirType(F32, listOf(2, 2)))
                listOf(
                    op(
                        OpKind.SLICE_AT, listOf(v, t), DxirType(F32, listOf(2, 2)),
                        attrs = mapOf("low" to listOf(0, 0)),
                    ),
                )
            }
            val v = floatArrayOf(4f, 3f, 2f, 1f)
            val out = DxirInterpreter.evalFunction(fn, listOf(v, FloatArray(4)))[0]
            assertTrue(v.indices.all { abs(out[it] - v[it]) < 1e-6f }, out.toList().toString())
        }
    }

    /**
     * THE closure pin: reverse-mode THROUGH a SUM_TO — the exact composition
     * that failed at §0.4.398 HEAD with "no VJP rule registered for SUM_TO".
     * A SUM_TO-containing scalar body is precisely the shape of a gradient
     * body (BroadcastRule and the binary rules emit SUM_TO), so this is the
     * reverse-over-reverse mechanism, minus the function-composition plumbing
     * the IR layer does not have.
     *
     * h(v[3,2], t[1,2], w[1,2]) = Σ (SUM_TO(v, t) ⊙ w):
     *   s[0,j] = Σ_k v[k,j];  loss = Σ_j s[0,j]·w[0,j]
     *   dv[k,j] = w[0,j]  — BROADCAST_LIKE(upstream·w, template=v), shape [3,2]
     *   dw[0,j] = Σ_k v[k,j]
     *   dt = 0            — the template is a pure shape source.
     */
    @Test
    fun reverseThroughSumTo() {
        val fn = DxirBuilder.function("sumto_body") {
            val v = param("v", DxirType(F32, listOf(3, 2)))
            val t = param("t", DxirType(F32, listOf(1, 2)))
            val w = param("w", DxirType(F32, listOf(1, 2)))
            val s = op(OpKind.SUM_TO, listOf(v, t), DxirType(F32, listOf(1, 2)))
            val p = op(OpKind.MUL, listOf(s, w), DxirType(F32, listOf(1, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val v = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val w = floatArrayOf(0.5f, -1.5f)
        val out = DxirInterpreter.evalFunction(grad, listOf(v, FloatArray(2), w))
        assertEquals(6, out[0].size, "dv must be broadcast back up to v's [3,2]")
        for (k in 0 until 3) for (j in 0 until 2) {
            assertTrue(
                abs(out[0][k * 2 + j] - w[j]) < 1e-6f,
                "dv[$k,$j] = ${out[0][k * 2 + j]}, want ${w[j]}",
            )
        }
        // dt: the template gets no gradient — a typed zero.
        assertTrue(out[1].all { abs(it) < 1e-6f }, "dt must be zero; got ${out[1].toList()}")
        // dw = col sums of v.
        assertTrue(abs(out[2][0] - 9f) < 1e-6f && abs(out[2][1] - 12f) < 1e-6f, out[2].toList().toString())
    }

    /**
     * The pair's other half: reverse-mode THROUGH a BROADCAST_LIKE, whose
     * adjoint is SUM_TO again — one more differentiation stays inside the pair.
     * h(u[1,2], t[3,2], w[3,2]) = Σ (BROADCAST_LIKE(u, t) ⊙ w):
     *   du[0,j] = Σ_k w[k,j] (keepdim — SUM_TO(upstream·w, template=u))
     *   dw[k,j] = u[0,j];  dt = 0.
     */
    @Test
    fun reverseThroughBroadcastLike() {
        val fn = DxirBuilder.function("bl_body") {
            val u = param("u", DxirType(F32, listOf(1, 2)))
            val t = param("t", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(3, 2)))
            val b = op(OpKind.BROADCAST_LIKE, listOf(u, t), DxirType(F32, listOf(3, 2)))
            val p = op(OpKind.MUL, listOf(b, w), DxirType(F32, listOf(3, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val u = floatArrayOf(2f, -3f)
        val w = FloatArray(6) { (it * 0.5f) - 1f }
        val out = DxirInterpreter.evalFunction(grad, listOf(u, FloatArray(6), w))
        assertEquals(2, out[0].size, "du must keep u's [1,2]")
        for (j in 0 until 2) {
            val want = (0 until 3).sumOf { k -> w[k * 2 + j].toDouble() }.toFloat()
            assertTrue(abs(out[0][j] - want) < 1e-5f, "du[0,$j] = ${out[0][j]}, want $want")
        }
        assertTrue(out[1].all { abs(it) < 1e-6f }, "dt must be zero; got ${out[1].toList()}")
        for (k in 0 until 3) for (j in 0 until 2) {
            assertTrue(abs(out[2][k * 2 + j] - u[j]) < 1e-6f, "dw[$k,$j] = ${out[2][k * 2 + j]}, want ${u[j]}")
        }
    }

    /**
     * PAD_TO's half: reverse-mode THROUGH a PAD_TO — previously the same loud
     * "no VJP rule" failure. h(u[2], t[4], w[4]) = Σ (PAD_TO(u, t, low=[1]) ⊙ w)
     * = u[0]·w[1] + u[1]·w[2]:
     *   du = [w[1], w[2]]  — SLICE_AT(upstream·w, template=u, low=[1])
     *   dw = [0, u0, u1, 0];  dt = 0.
     */
    @Test
    fun reverseThroughPadTo() {
        val fn = DxirBuilder.function("padto_body") {
            val u = param("u", DxirType(F32, listOf(2)))
            val t = param("t", DxirType(F32, listOf(4)))
            val w = param("w", DxirType(F32, listOf(4)))
            val pd = op(
                OpKind.PAD_TO, listOf(u, t), DxirType(F32, listOf(4)),
                attrs = mapOf("low" to listOf(1)),
            )
            val p = op(OpKind.MUL, listOf(pd, w), DxirType(F32, listOf(4)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val u = floatArrayOf(3f, -4f)
        val w = floatArrayOf(10f, 20f, 30f, 40f)
        val out = DxirInterpreter.evalFunction(grad, listOf(u, FloatArray(4), w))
        assertTrue(abs(out[0][0] - 20f) < 1e-6f && abs(out[0][1] - 30f) < 1e-6f, out[0].toList().toString())
        assertTrue(out[1].all { abs(it) < 1e-6f }, "dt must be zero; got ${out[1].toList()}")
        val wantDw = floatArrayOf(0f, 3f, -4f, 0f)
        assertTrue(wantDw.indices.all { abs(out[2][it] - wantDw[it]) < 1e-6f }, out[2].toList().toString())
    }

    /**
     * SLICE_AT's own reverse: its adjoint is PAD_TO again — the second pair is
     * closed too. h(v[4], t[2], w[2]) = Σ (SLICE_AT(v, t, low=[1]) ⊙ w)
     * = v[1]·w[0] + v[2]·w[1]:
     *   dv = [0, w0, w1, 0]  — PAD_TO(upstream·w, template=v, low=[1])
     *   dw = [v1, v2];  dt = 0.
     */
    @Test
    fun reverseThroughSliceAt() {
        val fn = DxirBuilder.function("sliceat_body") {
            val v = param("v", DxirType(F32, listOf(4)))
            val t = param("t", DxirType(F32, listOf(2)))
            val w = param("w", DxirType(F32, listOf(2)))
            val s = op(
                OpKind.SLICE_AT, listOf(v, t), DxirType(F32, listOf(2)),
                attrs = mapOf("low" to listOf(1)),
            )
            val p = op(OpKind.MUL, listOf(s, w), DxirType(F32, listOf(2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val v = floatArrayOf(1f, 2f, 3f, 4f)
        val w = floatArrayOf(5f, -7f)
        val out = DxirInterpreter.evalFunction(grad, listOf(v, FloatArray(2), w))
        val wantDv = floatArrayOf(0f, 5f, -7f, 0f)
        assertTrue(wantDv.indices.all { abs(out[0][it] - wantDv[it]) < 1e-6f }, out[0].toList().toString())
        assertTrue(out[1].all { abs(it) < 1e-6f }, "dt must be zero; got ${out[1].toList()}")
        assertTrue(abs(out[2][0] - 2f) < 1e-6f && abs(out[2][1] - 3f) < 1e-6f, out[2].toList().toString())
    }

    /** JVP⇄VJP cross-identity through BROADCAST_LIKE — each mode certifies the other. */
    @Test
    fun broadcastLikeJvpVjpCrossIdentity() {
        val fn = DxirBuilder.function("bl_chain") {
            val u = param("u", DxirType(F32, listOf(1, 3)))
            val t = param("t", DxirType(F32, listOf(4, 3)))
            val w = param("w", DxirType(F32, listOf(4, 3)))
            val b = op(OpKind.BROADCAST_LIKE, listOf(u, t), DxirType(F32, listOf(4, 3)))
            val p = op(OpKind.MUL, listOf(b, w), DxirType(F32, listOf(4, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val u = floatArrayOf(0.3f, -1.2f, 2.1f)
        val t = FloatArray(12)
        val w = FloatArray(12) { (it * 0.13f) - 1.1f }
        val vu = floatArrayOf(0.11f, -0.23f, 0.37f)
        val vt = FloatArray(12) { 0.31f * it } // shape-only operand: must not matter
        val vw = FloatArray(12) { (it * 0.017f) - 0.2f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(u, t, w))
        var dot = 0.0
        for (i in 0 until 3) dot += grads[0][i].toDouble() * vu[i]
        for (i in 0 until 12) dot += grads[1][i].toDouble() * vt[i]
        for (i in 0 until 12) dot += grads[2][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(u, t, w, vu, vt, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through BROADCAST_LIKE: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }

    /** JVP⇄VJP cross-identity through SLICE_AT. */
    @Test
    fun sliceAtJvpVjpCrossIdentity() {
        val fn = DxirBuilder.function("sa_chain") {
            val v = param("v", DxirType(F32, listOf(2, 5)))
            val t = param("t", DxirType(F32, listOf(2, 3)))
            val w = param("w", DxirType(F32, listOf(2, 3)))
            val s = op(
                OpKind.SLICE_AT, listOf(v, t), DxirType(F32, listOf(2, 3)),
                attrs = mapOf("low" to listOf(0, 2)),
            )
            val p = op(OpKind.MUL, listOf(s, w), DxirType(F32, listOf(2, 3)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val v = FloatArray(10) { (it * 0.21f) - 0.9f }
        val t = FloatArray(6)
        val w = FloatArray(6) { (it * 0.11f) - 0.3f }
        val vv = FloatArray(10) { (it * 0.031f) - 0.12f }
        val vt = FloatArray(6) { 0.19f * it }
        val vw = FloatArray(6) { (it * 0.023f) - 0.07f }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(v, t, w))
        var dot = 0.0
        for (i in 0 until 10) dot += grads[0][i].toDouble() * vv[i]
        for (i in 0 until 6) dot += grads[1][i].toDouble() * vt[i]
        for (i in 0 until 6) dot += grads[2][i].toDouble() * vw[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(v, t, w, vv, vt, vw),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through SLICE_AT: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }

    /**
     * The pair alternates, it never grows: reverse the [reverseThroughSumTo]
     * GRADIENT-shaped body once more. Its dv arm is a BROADCAST_LIKE, whose
     * rule emits SUM_TO — third-order stays inside the pair, and the values
     * still honour the math: g(v, t, w) = Σ (BROADCAST_LIKE(w, v) ⊙ v) with
     * w:[1,2], v:[3,2] gives dw[0,j] = Σ_k v[k,j] and dv[k,j] = w[0,j].
     */
    @Test
    fun thirdOrderStaysInsideThePair() {
        val fn = DxirBuilder.function("bl_grad_shaped") {
            val w = param("w", DxirType(F32, listOf(1, 2)))
            val v = param("v", DxirType(F32, listOf(3, 2)))
            val b = op(OpKind.BROADCAST_LIKE, listOf(w, v), DxirType(F32, listOf(3, 2)))
            val p = op(OpKind.MUL, listOf(b, v), DxirType(F32, listOf(3, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val w = floatArrayOf(1.5f, -2.5f)
        val v = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val out = DxirInterpreter.evalFunction(grad, listOf(w, v))
        // dw[0,j] = Σ_k v[k,j] — SUM_TO of the upstream·v back to w's [1,2].
        assertTrue(abs(out[0][0] - 9f) < 1e-5f && abs(out[0][1] - 12f) < 1e-5f, out[0].toList().toString())
        // dv: v appears BOTH as the template (shape-only: no gradient) and as
        // the MUL factor (dv[k,j] = broadcast(w)[k,j] = w[0,j]).
        for (k in 0 until 3) for (j in 0 until 2) {
            assertTrue(abs(out[1][k * 2 + j] - w[j]) < 1e-6f, "dv[$k,$j] = ${out[1][k * 2 + j]}, want ${w[j]}")
        }
    }
}
