package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The StableHLO forms of [OpKind.CAUSAL_CONV1D] (row compaction, no loop) and
 * [OpKind.GATED_DELTA_RULE] (a while over the tokens for one token per row, the
 * chunked form for several) on the device, against
 * the interpreter, which GatedDeltaOpsFixtureTest checks against transformers.
 * Rows: a new sequence, a padding row, and a continuation behind padding; then
 * a decode call reading the pools the first call wrote.
 */
class PjrtGatedDeltaOpsTest {

    private class Dims(val b: Int, val hk: Int, val hv: Int, val dk: Int, val dv: Int, val c: Int, val k: Int, val s: Int)

    private fun graph(d: Dims, t: Int): DxirFunction = DxirBuilder.function("gdn_ops") {
        val tX = DxirType(F32, listOf(d.b, t, d.c))
        val tConv = DxirType(F32, listOf(d.s, d.k - 1, d.c))
        val tQk = DxirType(F32, listOf(d.b, t, d.hk, d.dk))
        val tV = DxirType(F32, listOf(d.b, t, d.hv, d.dv))
        val tG = DxirType(F32, listOf(d.b, t, d.hv))
        val tState = DxirType(F32, listOf(d.s, d.hv, d.dk, d.dv))
        val tIdx = DxirType(I32, listOf(d.b, t))
        val ps = listOf(
            param("x", tX), param("w", DxirType(F32, listOf(d.k, d.c))), param("conv", tConv),
            param("q", tQk), param("k", tQk), param("v", tV), param("g", tG), param("beta", tG),
            param("state", tState), param("slots", tIdx), param("pos", tIdx),
        )
        val cv = opMulti(OpKind.CAUSAL_CONV1D, listOf(ps[0], ps[1], ps[2], ps[9], ps[10]), listOf(tX, tConv))
        val dr = opMulti(
            OpKind.GATED_DELTA_RULE, listOf(ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10]), listOf(tV, tState),
        )
        listOf(cv.result(0), cv.result(1), dr.result(0), dr.result(1))
    }

    /** L2-normalized rows of [width], as the graph hands the op its q and k. */
    private fun unitRows(n: Int, width: Int, rnd: Random, scale: Float): FloatArray {
        val a = FloatArray(n * width) { rnd.nextFloat() - 0.5f }
        for (r in 0 until n) {
            var ss = 0f
            for (i in 0 until width) ss += a[r * width + i] * a[r * width + i]
            val inv = scale / kotlin.math.sqrt(ss + 1e-6f)
            for (i in 0 until width) a[r * width + i] *= inv
        }
        return a
    }

    private fun inputs(d: Dims, t: Int, rnd: Random, conv: FloatArray, state: FloatArray, slots: IntArray, pos: IntArray) =
        listOf(
            FloatArray(d.b * t * d.c) { rnd.nextFloat() - 0.5f },
            FloatArray(d.k * d.c) { rnd.nextFloat() - 0.5f },
            conv,
            unitRows(d.b * t * d.hk, d.dk, rnd, 1f / kotlin.math.sqrt(d.dk.toFloat())),
            unitRows(d.b * t * d.hk, d.dk, rnd, 1f),
            FloatArray(d.b * t * d.hv * d.dv) { rnd.nextFloat() - 0.5f },
            FloatArray(d.b * t * d.hv) { -rnd.nextFloat() },
            FloatArray(d.b * t * d.hv) { rnd.nextFloat() },
            state,
            FloatArray(slots.size) { slots[it].toFloat() },
            FloatArray(pos.size) { pos[it].toFloat() },
        )

    private fun compare(what: String, got: List<FloatArray>, want: List<FloatArray>, tol: Float): Float {
        var worst = 0f
        for ((r, name) in listOf("y", "convPool", "out", "statePool").withIndex()) {
            for (i in want[r].indices) {
                val e = abs(got[r][i] - want[r][i]) / maxOf(1f, abs(want[r][i]))
                worst = maxOf(worst, e)
                assertTrue(e <= tol, "$what $name[$i]: GPU ${got[r][i]} vs interpreter ${want[r][i]}")
            }
        }
        return worst
    }

    private fun runCase(d: Dims, t: Int, tol: Float) {
        val rnd = Random(11)
        var conv = FloatArray(d.s * (d.k - 1) * d.c) { rnd.nextFloat() - 0.5f }
        var state = FloatArray(d.s * d.hv * d.dk * d.dv) { (rnd.nextFloat() - 0.5f) * 0.3f }
        val padRow = d.b - 2
        // Row 0 new at slot 2; row b-1 continues slot 0 behind t/2 padding tokens.
        val slots1 = IntArray(d.b * t) { i ->
            val row = i / t; val tok = i % t
            when (row) { 0 -> 2; d.b - 1 -> if (tok < t / 2) -1 else 0; else -> -1 }
        }
        val pos1 = IntArray(d.b * t) { i ->
            val row = i / t; val tok = i % t
            when (row) { 0 -> tok; d.b - 1 -> if (tok < t / 2) 0 else 7 + tok; else -> 0 }
        }
        val slots2 = IntArray(d.b) { row -> when (row) { 0 -> 2; d.b - 1 -> 0; else -> -1 } }
        val pos2 = IntArray(d.b) { row -> when (row) { 0 -> t; d.b - 1 -> 7 + t; else -> 0 } }
        check(padRow >= 1)
        TestBackend.session().use { session ->
            var worst = 0f
            for ((tt, sl, ps) in listOf(Triple(t, slots1, pos1), Triple(1, slots2, pos2))) {
                val fn = graph(d, tt)
                val ins = inputs(d, tt, rnd, conv, state, sl, ps)
                val want = DxirInterpreter.evalFunction(fn, ins)
                val got = session.runOn(fn, ins)
                worst = maxOf(worst, compare("T=$tt", got, want, tol))
                conv = want[1]
                state = want[3]
            }
            println("[pjrt-gdn] B=${d.b} T=$t Hk=${d.hk} Hv=${d.hv} Dk=${d.dk} Dv=${d.dv}: worst relative error $worst on ${TestBackend.target}")
        }
    }

    @Test
    fun smallBlockMatchesTheInterpreterOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        runCase(Dims(b = 3, hk = 2, hv = 4, dk = 3, dv = 2, c = 6, k = 4, s = 4), t = 5, tol = 1e-5f)
    }

    @Test
    fun groupedHeadsOverSixtyFourTokensMatchTheInterpreterOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        runCase(Dims(b = 4, hk = 2, hv = 6, dk = 32, dv = 16, c = 160, k = 4, s = 5), t = 64, tol = 1e-4f)
    }

    /** Two hundred tokens: four chunks of 64, the last padded, so the chunk scan and its padding run. */
    @Test
    fun severalChunksWithPaddingMatchTheInterpreterOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        runCase(Dims(b = 3, hk = 2, hv = 4, dk = 16, dv = 8, c = 48, k = 4, s = 4), t = 200, tol = 1e-4f)
    }

    /**
     * Per-token state writes (`writeSlots`, a speculative step): row 0 reads
     * slot 2 and writes the state after each of its 4 tokens to slots 4..7,
     * row 1 is padding, row 2 reads slot 0 behind one padding token and writes
     * after its second and last tokens only. The GPU's unrolled recurrence and
     * windowed conv states against the interpreter.
     */
    @Test
    fun perTokenStateWritesMatchTheInterpreterOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val d = Dims(b = 3, hk = 2, hv = 4, dk = 8, dv = 4, c = 12, k = 4, s = 9)
        val t = 4
        val fn = DxirBuilder.function("gdn_per_token") {
            val tX = DxirType(F32, listOf(d.b, t, d.c))
            val tConv = DxirType(F32, listOf(d.s, d.k - 1, d.c))
            val tQk = DxirType(F32, listOf(d.b, t, d.hk, d.dk))
            val tV = DxirType(F32, listOf(d.b, t, d.hv, d.dv))
            val tG = DxirType(F32, listOf(d.b, t, d.hv))
            val tState = DxirType(F32, listOf(d.s, d.hv, d.dk, d.dv))
            val tIdx = DxirType(I32, listOf(d.b, t))
            val ps = listOf(
                param("x", tX), param("w", DxirType(F32, listOf(d.k, d.c))), param("conv", tConv),
                param("q", tQk), param("k", tQk), param("v", tV), param("g", tG), param("beta", tG),
                param("state", tState), param("slots", tIdx), param("pos", tIdx), param("writes", tIdx),
            )
            val cv = opMulti(OpKind.CAUSAL_CONV1D, listOf(ps[0], ps[1], ps[2], ps[9], ps[10], ps[11]), listOf(tX, tConv))
            val dr = opMulti(
                OpKind.GATED_DELTA_RULE, listOf(ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10], ps[11]), listOf(tV, tState),
            )
            listOf(cv.result(0), cv.result(1), dr.result(0), dr.result(1))
        }
        val rnd = Random(17)
        val conv = FloatArray(d.s * (d.k - 1) * d.c) { rnd.nextFloat() - 0.5f }
        val state = FloatArray(d.s * d.hv * d.dk * d.dv) { (rnd.nextFloat() - 0.5f) * 0.3f }
        val slots = intArrayOf(2, 2, 2, 2, -1, -1, -1, -1, -1, 0, 0, 0)
        val pos = intArrayOf(9, 10, 11, 12, 0, 0, 0, 0, 0, 5, 6, 7)
        val writes = intArrayOf(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, 8, 1)
        val ins = inputs(d, t, rnd, conv, state, slots, pos) + listOf(FloatArray(writes.size) { writes[it].toFloat() })
        val want = DxirInterpreter.evalFunction(fn, ins)
        TestBackend.session().use { session ->
            val got = session.runOn(fn, ins)
            val worst = compare("per-token", got, want, 1e-5f)
            println("[pjrt-gdn] per-token writes, B=${d.b} T=$t: worst relative error $worst on ${TestBackend.target}")
        }
    }

    /**
     * The committed conv ([io.tlaloc.ir.CausalConv1dAttrs.COMMIT]): a verify block (RECORDS,
     * T = R = 4) whose rows read base 2 with 3 records applied, are padding, and start at
     * position 0; then a prefill block (FINAL, T = 6) whose rows read base 5 with 2 records,
     * behind 2 padding tokens, and start at position 0. The device's gathers and window
     * against the interpreter, exactly.
     */
    @Test
    fun committedConvStatesMatchTheInterpreterOnTheDevice() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val b = 3; val c = 10; val k = 4; val s = 7; val r = 4
        fun graph(t: Int, mode: io.tlaloc.ir.GatedDeltaRuleAttrs.Commit) = DxirBuilder.function("conv_commit") {
            val tX = DxirType(F32, listOf(b, t, c))
            val tConv = DxirType(F32, listOf(s, k - 1, c))
            val tRec = DxirType(F32, listOf(s, r, c))
            val tIdx = DxirType(I32, listOf(b, t))
            val ps = listOf(
                param("x", tX), param("w", DxirType(F32, listOf(k, c))), param("conv", tConv), param("slots", tIdx),
                param("pos", tIdx), param("commit", DxirType(I32, listOf(b, 3))), param("records", tRec),
            )
            val cv = opMulti(OpKind.CAUSAL_CONV1D, ps, listOf(tX, tConv, tRec), mapOf(io.tlaloc.ir.CausalConv1dAttrs.COMMIT to mode.attr))
            listOf(cv.result(0), cv.result(1), cv.result(2))
        }
        val rnd = Random(23)
        var conv = FloatArray(s * (k - 1) * c) { rnd.nextFloat() - 0.5f }
        var records = FloatArray(s * r * c) { rnd.nextFloat() - 0.5f }
        val w = FloatArray(k * c) { rnd.nextFloat() - 0.5f }
        fun f(a: IntArray) = FloatArray(a.size) { a[it].toFloat() }
        val cases = listOf(
            Triple(4, io.tlaloc.ir.GatedDeltaRuleAttrs.Commit.RECORDS, Triple(
                intArrayOf(2, 2, 2, 2, -1, -1, -1, -1, 0, 0, 0, 0), intArrayOf(9, 10, 11, 12, 0, 0, 0, 0, 0, 1, 2, 3),
                intArrayOf(2, 5, 3, 0, 0, 0, 1, 4, 2),
            )),
            Triple(6, io.tlaloc.ir.GatedDeltaRuleAttrs.Commit.FINAL, Triple(
                intArrayOf(-1, -1, 5, 5, 5, 5, -1, -1, -1, -1, -1, -1, 6, 6, 6, 6, 6, 6),
                intArrayOf(0, 0, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5),
                intArrayOf(5, 2, 2, 0, 0, 0, 6, 3, 4),
            )),
        )
        TestBackend.session().use { session ->
            for ((t, mode, rows) in cases) {
                val (slots, pos, commit) = rows
                val fn = graph(t, mode)
                val ins = listOf(FloatArray(b * t * c) { rnd.nextFloat() - 0.5f }, w, conv, f(slots), f(pos), f(commit), records)
                val want = DxirInterpreter.evalFunction(fn, ins)
                val got = session.runOn(fn, ins)
                for ((i, name) in listOf("y", "convPool", "records").withIndex()) {
                    for (j in want[i].indices) {
                        val e = abs(got[i][j] - want[i][j]) / maxOf(1f, abs(want[i][j]))
                        assertTrue(e <= 1e-6f, "${mode.attr} $name[$j]: GPU ${got[i][j]} vs interpreter ${want[i][j]}")
                    }
                }
                conv = want[1]
                records = want[2]
            }
            println("[pjrt-gdn] committed conv, records and final: the device equals the interpreter on ${TestBackend.target}")
        }
    }
}
