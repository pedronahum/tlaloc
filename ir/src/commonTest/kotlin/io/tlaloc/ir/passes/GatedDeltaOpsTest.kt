package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.render.KotlinRenderRefusal
import io.tlaloc.ir.render.toKotlinSource
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [OpKind.CAUSAL_CONV1D] and [OpKind.GATED_DELTA_RULE]: the row rules, the
 * state threading across calls, and the refusals. Parity with transformers is
 * GatedDeltaOpsFixtureTest (jvmTest).
 */
class GatedDeltaOpsTest {

    private val hk = 1
    private val hv = 2
    private val dk = 2
    private val dv = 3
    private val c = 4
    private val k = 3
    private val s = 3

    private fun graph(b: Int, t: Int): DxirFunction = DxirBuilder.function("gdn") {
        val tX = DxirType(F32, listOf(b, t, c))
        val tConv = DxirType(F32, listOf(s, k - 1, c))
        val tQk = DxirType(F32, listOf(b, t, hk, dk))
        val tV = DxirType(F32, listOf(b, t, hv, dv))
        val tG = DxirType(F32, listOf(b, t, hv))
        val tState = DxirType(F32, listOf(s, hv, dk, dv))
        val tIdx = DxirType(I32, listOf(b, t))
        val ps = listOf(
            param("x", tX), param("w", DxirType(F32, listOf(k, c))), param("conv", tConv),
            param("q", tQk), param("k", tQk), param("v", tV), param("g", tG), param("beta", tG),
            param("state", tState), param("slots", tIdx), param("pos", tIdx),
        )
        val cv = opMulti(OpKind.CAUSAL_CONV1D, listOf(ps[0], ps[1], ps[2], ps[9], ps[10]), listOf(tX, tConv))
        val dr = opMulti(OpKind.GATED_DELTA_RULE, listOf(ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10]), listOf(tV, tState))
        listOf(cv.result(0), cv.result(1), dr.result(0), dr.result(1))
    }

    /** Per-token inputs of one sequence of [n] tokens, in token order. */
    private class Seq(n: Int, rnd: Random, c: Int, hk: Int, hv: Int, dk: Int, dv: Int) {
        val x = List(n) { FloatArray(c) { rnd.nextFloat() - 0.5f } }
        val q = List(n) { FloatArray(hk * dk) { rnd.nextFloat() - 0.5f } }
        val k = List(n) { FloatArray(hk * dk) { rnd.nextFloat() - 0.5f } }
        val v = List(n) { FloatArray(hv * dv) { rnd.nextFloat() - 0.5f } }
        val g = List(n) { FloatArray(hv) { -rnd.nextFloat() } }
        val beta = List(n) { FloatArray(hv) { rnd.nextFloat() } }
    }

    private val weight = FloatArray(k * c) { Random(1).nextFloat() - 0.5f + it * 0.01f }

    /**
     * Run [seq]'s tokens [from, until) as row 0 of a `[2, t]` block,
     * right-aligned; row 1 is padding. Returns the four results.
     */
    private fun call(
        seq: Seq, from: Int, until: Int, t: Int, slot: Int, conv: FloatArray, state: FloatArray,
    ): List<FloatArray> {
        val n = until - from
        val pad = t - n
        fun rows(width: Int, pick: (Int) -> FloatArray): FloatArray {
            val out = FloatArray(2 * t * width)
            for (i in 0 until n) pick(from + i).copyInto(out, (pad + i) * width)
            return out
        }
        val slots = FloatArray(2 * t) { if (it < t && it >= pad) slot.toFloat() else -1f }
        val pos = FloatArray(2 * t) { if (it < t && it >= pad) (from + it - pad).toFloat() else 0f }
        return DxirInterpreter.evalFunction(
            graph(2, t),
            listOf(
                rows(c) { seq.x[it] }, weight, conv, rows(hk * dk) { seq.q[it] }, rows(hk * dk) { seq.k[it] },
                rows(hv * dv) { seq.v[it] }, rows(hv) { seq.g[it] }, rows(hv) { seq.beta[it] },
                state, slots, pos,
            ),
        )
    }

    private val conv0 = FloatArray(s * (k - 1) * c) { 100f + it }
    private val state0 = FloatArray(s * hv * dk * dv) { 50f + it }

    @Test
    fun aSequenceSplitAcrossTwoCallsGivesTheOutputsAndStateOfOneCall() {
        val seq = Seq(5, Random(3), c, hk, hv, dk, dv)
        val whole = call(seq, 0, 5, 5, slot = 1, conv0, state0)
        val first = call(seq, 0, 3, 4, slot = 1, conv0, state0)
        val second = call(seq, 3, 5, 4, slot = 1, first[1], first[3])
        // Row 0's last two tokens: positions 3 and 4 of the whole call, the
        // last two rows of the second (right-aligned in a block of 4).
        for (tok in 0 until 2) {
            for (ch in 0 until c) {
                assertClose(whole[0][(3 + tok) * c + ch], second[0][(2 + tok) * c + ch], "conv y token ${3 + tok}")
            }
            for (j in 0 until hv * dv) {
                assertClose(whole[2][(3 + tok) * hv * dv + j], second[2][(2 + tok) * hv * dv + j], "delta out token ${3 + tok}")
            }
        }
        for (i in whole[1].indices) assertClose(whole[1][i], second[1][i], "conv pool $i")
        for (i in whole[3].indices) assertClose(whole[3][i], second[3][i], "state pool $i")
    }

    @Test
    fun aRowStartingAtPositionZeroIgnoresWhatItsSlotHeld() {
        val seq = Seq(3, Random(4), c, hk, hv, dk, dv)
        val a = call(seq, 0, 3, 3, slot = 2, conv0, state0)
        val b = call(seq, 0, 3, 3, slot = 2, FloatArray(conv0.size) { -7f }, FloatArray(state0.size) { 9f })
        for (r in listOf(0, 2)) for (i in a[r].indices) assertEquals(a[r][i], b[r][i], "result $r element $i")
    }

    @Test
    fun paddingWritesNothingAndOutputsZero() {
        val seq = Seq(2, Random(5), c, hk, hv, dk, dv)
        val r = call(seq, 0, 2, 4, slot = 0, conv0, state0)
        // Slots 1 and 2 are named by no live token: unchanged.
        for (slot in 1 until s) {
            val cs = (k - 1) * c
            for (i in slot * cs until (slot + 1) * cs) assertEquals(conv0[i], r[1][i], "conv pool slot $slot")
            val ss = hv * dk * dv
            for (i in slot * ss until (slot + 1) * ss) assertEquals(state0[i], r[3][i], "state pool slot $slot")
        }
        // Row 0's two padding tokens and all of row 1 output zero.
        for (tok in listOf(0, 1, 4, 5, 6, 7)) {
            for (ch in 0 until c) assertEquals(0f, r[0][tok * c + ch], "conv y token $tok")
            for (j in 0 until hv * dv) assertEquals(0f, r[2][tok * hv * dv + j], "delta out token $tok")
        }
    }

    private fun refusal(slots: FloatArray, pos: FloatArray): String {
        val t = 3
        val zeros = { n: Int -> FloatArray(n) }
        val ex = assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                graph(2, t),
                listOf(
                    zeros(2 * t * c), weight, conv0, zeros(2 * t * hk * dk), zeros(2 * t * hk * dk),
                    zeros(2 * t * hv * dv), zeros(2 * t * hv), zeros(2 * t * hv), state0, slots, pos,
                ),
            )
        }
        return ex.message ?: ""
    }

    @Test
    fun paddingAfterALiveTokenIsRefusedByName() {
        val msg = refusal(floatArrayOf(0f, -1f, 0f, -1f, -1f, -1f), FloatArray(6) { it.toFloat() })
        assertTrue("padding comes first" in msg, msg)
    }

    @Test
    fun twoSlotsInOneRowAndOneSlotInTwoRowsAreRefusedByName() {
        assertTrue("carries one slot" in refusal(floatArrayOf(0f, 0f, 1f, -1f, -1f, -1f), floatArrayOf(1f, 2f, 3f, 0f, 0f, 0f)))
        assertTrue("more than one row" in refusal(floatArrayOf(-1f, 1f, 1f, 1f, 1f, 1f), floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f)))
    }

    @Test
    fun aSecondSequenceStartInsideARowIsRefusedByName() {
        val msg = refusal(floatArrayOf(2f, 2f, 2f, -1f, -1f, -1f), floatArrayOf(4f, 0f, 1f, 0f, 0f, 0f))
        assertTrue("only a row's first token may start a sequence" in msg, msg)
    }

    @Test
    fun aSlotPastThePoolIsRefusedByName() {
        val msg = refusal(floatArrayOf(s.toFloat(), s.toFloat(), s.toFloat(), -1f, -1f, -1f), FloatArray(6))
        assertTrue("state slot $s" in msg, msg)
    }

    private fun lossFn(kind: OpKind): DxirFunction = DxirBuilder.function("gdnLoss") {
        val full = graph(1, 2)
        val ps = full.params.map { param(it.name, it.type) }
        val r = if (kind == OpKind.CAUSAL_CONV1D) {
            opMulti(kind, listOf(ps[0], ps[1], ps[2], ps[9], ps[10]), listOf(ps[0].type, ps[2].type))
        } else {
            opMulti(kind, listOf(ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10]), listOf(ps[5].type, ps[8].type))
        }
        listOf(op(OpKind.SUM, listOf(r.result(0)), DxirType(F32, emptyList())))
    }

    @Test
    fun bothTransformsAndTheRendererRefuseTheOpsByName() {
        for (kind in listOf(OpKind.CAUSAL_CONV1D, OpKind.GATED_DELTA_RULE)) {
            for (ex in listOf(
                assertFailsWith<IllegalStateException> { DxirReverseTransform.apply(lossFn(kind)) },
                assertFailsWith<IllegalStateException> { DxirForwardTransform.apply(lossFn(kind)) },
            )) {
                val msg = ex.message ?: ""
                assertTrue(kind.name in msg && "inference-only" in msg, "refusal must name $kind and why; got $msg")
            }
            // The renderer refuses every multi-result op before looking at its kind.
            val msg = assertFailsWith<KotlinRenderRefusal> { toKotlinSource(lossFn(kind)) }.message ?: ""
            assertTrue(kind.name in msg, "render refusal must name $kind; got $msg")
        }
    }

    private fun assertClose(a: Float, b: Float, what: String) =
        assertTrue(kotlin.math.abs(a - b) <= 1e-5f * maxOf(1f, kotlin.math.abs(a)), "$what: $a vs $b")
}
