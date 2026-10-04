package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [OpKind.CAUSAL_CONV1D] and [OpKind.GATED_DELTA_RULE] with `writeSlots`: a
 * call over several tokens that reads its row's slot and writes the state
 * after each token to its own slot holds, in those slots, the states a chain
 * of one-token calls reaches, gives the same outputs, and leaves the slot it
 * read as it was.
 */
class PerTokenStateWritesTest {

    private val b = 2
    private val hk = 1
    private val hv = 2
    private val dk = 2
    private val dv = 3
    private val c = 4
    private val k = 3
    private val s = 6
    private val convW = (k - 1) * c
    private val stateW = hv * dk * dv

    private fun graph(t: Int, perToken: Boolean): DxirFunction = DxirBuilder.function("gdn") {
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
        val writes = if (perToken) listOf(param("writes", tIdx)) else emptyList()
        val cv = opMulti(OpKind.CAUSAL_CONV1D, listOf(ps[0], ps[1], ps[2], ps[9], ps[10]) + writes, listOf(tX, tConv))
        val dr = opMulti(
            OpKind.GATED_DELTA_RULE, listOf(ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10]) + writes, listOf(tV, tState),
        )
        listOf(cv.result(0), cv.result(1), dr.result(0), dr.result(1))
    }

    private val rnd = Random(5)
    private val n = 5
    private val x = List(n) { FloatArray(c) { rnd.nextFloat() - 0.5f } }
    private val q = List(n) { FloatArray(hk * dk) { rnd.nextFloat() - 0.5f } }
    private val kk = List(n) { FloatArray(hk * dk) { rnd.nextFloat() - 0.5f } }
    private val v = List(n) { FloatArray(hv * dv) { rnd.nextFloat() - 0.5f } }
    private val g = List(n) { FloatArray(hv) { -rnd.nextFloat() } }
    private val beta = List(n) { FloatArray(hv) { rnd.nextFloat() } }
    private val weight = FloatArray(k * c) { rnd.nextFloat() - 0.5f }

    /** Tokens [from, until) of the sequence as row 0 (row 1 padding), reading slot 0. */
    private fun call(from: Int, until: Int, conv: FloatArray, state: FloatArray, writes: IntArray? = null): List<FloatArray> {
        val t = until - from
        fun rows(width: Int, pick: (Int) -> FloatArray) = FloatArray(b * t * width).also { out ->
            for (i in 0 until t) pick(from + i).copyInto(out, i * width)
        }
        val ins = mutableListOf(
            rows(c, x::get), weight, conv, rows(hk * dk, q::get), rows(hk * dk, kk::get), rows(hv * dv, v::get),
            rows(hv, g::get), rows(hv, beta::get), state,
            FloatArray(b * t) { if (it < t) 0f else -1f },
            FloatArray(b * t) { if (it < t) (from + it).toFloat() else 0f },
        )
        if (writes != null) ins += FloatArray(writes.size) { writes[it].toFloat() }
        return DxirInterpreter.evalFunction(graph(t, writes != null), ins)
    }

    @Test
    fun eachTokensStateLandsInItsSlotAsAChainOfOneTokenCallsLeavesIt() {
        val conv0 = FloatArray(s * convW) { rnd.nextFloat() - 0.5f }
        val state0 = FloatArray(s * stateW) { rnd.nextFloat() - 0.5f }
        val prefix = call(0, 2, conv0, state0)
        // The chain: tokens 2, 3, 4 one call each, threading slot 0.
        var conv = prefix[1]
        var state = prefix[3]
        val chain = (2 until n).map { j -> call(j, j + 1, conv, state).also { conv = it[1]; state = it[3] } }
        // One call over tokens 2..4, reading slot 0 and writing slots 3, 4, 5.
        val got = call(2, n, prefix[1], prefix[3], intArrayOf(3, 4, 5, -1, -1, -1))
        for ((j, ref) in chain.withIndex()) {
            val slot = 3 + j
            close(ref[1].copyOfRange(0, convW), got[1].copyOfRange(slot * convW, (slot + 1) * convW), "conv state after token ${2 + j}")
            close(ref[3].copyOfRange(0, stateW), got[3].copyOfRange(slot * stateW, (slot + 1) * stateW), "state after token ${2 + j}")
            close(ref[0].copyOfRange(0, c), got[0].copyOfRange(j * c, (j + 1) * c), "conv output of token ${2 + j}")
            close(ref[2].copyOfRange(0, hv * dv), got[2].copyOfRange(j * hv * dv, (j + 1) * hv * dv), "output of token ${2 + j}")
        }
        // The slot read, and the slots nobody wrote, are as they were.
        for (slot in 0 until 3) {
            close(prefix[1].copyOfRange(slot * convW, (slot + 1) * convW), got[1].copyOfRange(slot * convW, (slot + 1) * convW), "conv slot $slot")
            close(prefix[3].copyOfRange(slot * stateW, (slot + 1) * stateW), got[3].copyOfRange(slot * stateW, (slot + 1) * stateW), "slot $slot")
        }
    }

    @Test
    fun aSlotWrittenTwiceOrAWriteFromPaddingIsRefusedByName() {
        val conv = FloatArray(s * convW)
        val state = FloatArray(s * stateW)
        val twice = assertFailsWith<IllegalArgumentException> { call(2, n, conv, state, intArrayOf(3, 3, 5, -1, -1, -1)) }
        assertTrue("written by more than one token" in twice.message!!, twice.message)
        val padding = assertFailsWith<IllegalArgumentException> { call(2, n, conv, state, intArrayOf(3, 4, 5, 1, -1, -1)) }
        assertTrue("only live tokens write" in padding.message!!, padding.message)
    }

    private fun close(want: FloatArray, got: FloatArray, what: String) {
        assertEquals(want.size, got.size, what)
        for (i in want.indices) {
            assertTrue(kotlin.math.abs(want[i] - got[i]) <= 1e-6f * maxOf(1f, kotlin.math.abs(want[i])), "$what[$i]: ${got[i]} vs ${want[i]}")
        }
    }
}
