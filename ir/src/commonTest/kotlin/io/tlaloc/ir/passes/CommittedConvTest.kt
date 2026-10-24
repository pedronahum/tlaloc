package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.CausalConv1dAttrs
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.GatedDeltaRuleAttrs
import io.tlaloc.ir.OpKind
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [OpKind.CAUSAL_CONV1D] committed once per call ([CausalConv1dAttrs.COMMIT]): a
 * verify step that keeps its starting history and its inputs as records leaves,
 * for every count of accepted tokens, the history a per-token write keeps after
 * that many tokens, bit for bit; chained verify steps and a prefill call after
 * them continue from it as the per-token form does.
 */
class CommittedConvTest {

    private val b = 2
    private val c = 3
    private val k = 4
    private val s = 4
    private val km1 = k - 1
    private val slotW = km1 * c

    private fun graph(t: Int, commit: GatedDeltaRuleAttrs.Commit?, r: Int): DxirFunction = DxirBuilder.function("conv") {
        val tX = DxirType(F32, listOf(b, t, c))
        val tConv = DxirType(F32, listOf(s, km1, c))
        val tIdx = DxirType(I32, listOf(b, t))
        val x = param("x", tX)
        val w = param("w", DxirType(F32, listOf(k, c)))
        val conv = param("conv", tConv)
        val slots = param("slots", tIdx)
        val pos = param("pos", tIdx)
        if (commit == null) {
            val cv = opMulti(OpKind.CAUSAL_CONV1D, listOf(x, w, conv, slots, pos), listOf(tX, tConv))
            listOf(cv.result(0), cv.result(1))
        } else {
            val tRec = DxirType(F32, listOf(s, r, c))
            val rows = param("commit", DxirType(I32, listOf(b, 3)))
            val rec = param("records", tRec)
            val cv = opMulti(
                OpKind.CAUSAL_CONV1D, listOf(x, w, conv, slots, pos, rows, rec), listOf(tX, tConv, tRec),
                mapOf(CausalConv1dAttrs.COMMIT to commit.attr),
            )
            listOf(cv.result(0), cv.result(1), cv.result(2))
        }
    }

    private val rnd = Random(11)
    private val n = 12
    private val xs = List(n) { FloatArray(c) { rnd.nextFloat() - 0.5f } }
    private val weight = FloatArray(k * c) { rnd.nextFloat() - 0.5f }

    /** Tokens [from, until) right-aligned in a row of [t] as row 0 at slot [slot]; row 1 padding. */
    private fun ins(from: Int, until: Int, t: Int, slot: Int, conv: FloatArray): MutableList<FloatArray> {
        val pad = t - (until - from)
        return mutableListOf(
            FloatArray(b * t * c).also { o -> for (i in from until until) xs[i].copyInto(o, (pad + i - from) * c) },
            weight, conv,
            FloatArray(b * t) { if (it < t && it >= pad) slot.toFloat() else -1f },
            FloatArray(b * t) { if (it < t && it >= pad) (from + it - pad).toFloat() else 0f },
        )
    }

    private fun plain(from: Int, until: Int, t: Int, conv: FloatArray, slot: Int = 0): List<FloatArray> =
        DxirInterpreter.evalFunction(graph(t, null, 0), ins(from, until, t, slot, conv))

    private fun committed(
        mode: GatedDeltaRuleAttrs.Commit,
        from: Int,
        until: Int,
        t: Int,
        conv: FloatArray,
        records: FloatArray,
        base: Int,
        target: Int,
        count: Int,
        r: Int = 3,
    ): List<FloatArray> = DxirInterpreter.evalFunction(
        graph(t, mode, r),
        ins(from, until, t, base, conv) + listOf(
            floatArrayOf(base.toFloat(), target.toFloat(), count.toFloat(), 0f, 0f, 0f), records,
        ),
    )

    private fun slot(a: FloatArray, i: Int) = a.copyOfRange(i * slotW, (i + 1) * slotW)

    @Test
    fun aVerifyStepsRecordsGiveTheHistoryAfterAnyCountBitForBit() {
        val conv0 = FloatArray(s * slotW) { rnd.nextFloat() - 0.5f }
        val rec0 = FloatArray(s * 3 * c)
        // A prompt of 5 tokens, then a verify step of 3 from slot 1 to slot 2.
        val pre = committed(GatedDeltaRuleAttrs.Commit.FINAL, 0, 5, 6, conv0, rec0, base = 1, target = 2, count = 0)
        val ref = plain(0, 5, 6, conv0, slot = 1)
        assertContentEquals(ref[0], pre[0], "prefill output")
        assertContentEquals(slot(ref[1], 1), slot(pre[1], 1), "prefill history at the base")
        val step = committed(GatedDeltaRuleAttrs.Commit.RECORDS, 5, 8, 3, pre[1], pre[2], base = 1, target = 2, count = 0)
        val oneShot = plain(5, 8, 3, pre[1], slot = 1)
        assertContentEquals(oneShot[0], step[0], "verify output")
        assertContentEquals(slot(pre[1], 1), slot(step[1], 2), "the starting history at the target")
        assertContentEquals(slot(pre[1], 1), slot(step[1], 1), "the base is not written")
        // For each count, the next token from the target with count records applied.
        var conv = pre[1]
        for (count in 1..3) {
            val perToken = plain(5, 5 + count, 3, pre[1], slot = 1)
            conv = perToken[1]
            val next = committed(GatedDeltaRuleAttrs.Commit.FINAL, 5 + count, 6 + count, 2, step[1], step[2], base = 2, target = 1, count = count)
            val want = plain(5 + count, 6 + count, 2, conv, slot = 1)
            assertContentEquals(want[0], next[0], "the token after $count accepted")
            assertContentEquals(slot(want[1], 1), slot(next[1], 2), "the history after ${count + 1} tokens")
        }
    }

    @Test
    fun chainedVerifyStepsAndAPrefillAfterThemEqualThePerTokenForm() {
        val conv0 = FloatArray(s * slotW)
        val rec0 = FloatArray(s * 3 * c) { rnd.nextFloat() }
        // Committed: prompt 0..2 (position 0: the stale records are not read), steps of 3 accepting 2 and 3, a prefill of 2.
        val a = committed(GatedDeltaRuleAttrs.Commit.FINAL, 0, 2, 4, conv0, rec0, base = 0, target = 3, count = 2)
        val v1 = committed(GatedDeltaRuleAttrs.Commit.RECORDS, 2, 5, 3, a[1], a[2], base = 0, target = 3, count = 0)
        val v2 = committed(GatedDeltaRuleAttrs.Commit.RECORDS, 4, 7, 3, v1[1], v1[2], base = 3, target = 0, count = 2)
        val p = committed(GatedDeltaRuleAttrs.Commit.FINAL, 7, 9, 4, v2[1], v2[2], base = 0, target = 3, count = 3)
        // Per token: the same tokens through one slot.
        val ra = plain(0, 2, 4, conv0)
        val r1 = plain(2, 4, 3, ra[1])
        val r2 = plain(4, 7, 3, r1[1])
        val rp = plain(7, 9, 4, r2[1])
        assertContentEquals(plain(4, 7, 3, r1[1])[0], v2[0], "the second step's output")
        assertContentEquals(rp[0], p[0], "the prefill's output")
        assertContentEquals(slot(rp[1], 0), slot(p[1], 0), "the history after the prefill")
    }

    @Test
    fun aCommitOutsideThePoolOrWithTooManyRecordsIsRefusedByName() {
        val conv = FloatArray(s * slotW)
        val rec = FloatArray(s * 3 * c)
        val many = assertFailsWith<IllegalArgumentException> {
            committed(GatedDeltaRuleAttrs.Commit.FINAL, 3, 4, 2, conv, rec, base = 1, target = 2, count = 4)
        }
        assertTrue("applies 4 records" in many.message!!, many.message)
        val same = assertFailsWith<IllegalArgumentException> {
            committed(GatedDeltaRuleAttrs.Commit.RECORDS, 3, 6, 3, conv, rec, base = 1, target = 1, count = 0)
        }
        assertTrue("not its base" in same.message!!, same.message)
        // Records for fewer tokens than the pool holds (a narrower verify step) are taken; for more, refused.
        committed(GatedDeltaRuleAttrs.Commit.RECORDS, 3, 5, 2, conv, rec, base = 1, target = 2, count = 0)
        val width = assertFailsWith<IllegalArgumentException> {
            committed(GatedDeltaRuleAttrs.Commit.RECORDS, 3, 7, 4, conv, rec, base = 1, target = 2, count = 0)
        }
        assertTrue("R must be at least 4" in width.message!!, width.message)
    }

    /**
     * A verify step of fewer tokens than the records pool holds (an entry of fewer drafts sharing
     * the pools of a wider one) writes its own records and leaves the rest of the target's as they
     * were; the next call applies them as from a pool of its own size.
     */
    @Test
    fun aStepOfFewerTokensThanThePoolKeepsTheRestOfTheRecords() {
        val conv = FloatArray(s * slotW) { 0.01f * (it % 17) }
        val wide = FloatArray(s * 5 * c) { 100f + it }
        val v = committed(GatedDeltaRuleAttrs.Commit.RECORDS, 3, 5, 2, conv, wide, base = 1, target = 2, count = 0, r = 5)
        val exact = committed(GatedDeltaRuleAttrs.Commit.RECORDS, 3, 5, 2, conv, FloatArray(s * 2 * c), base = 1, target = 2, count = 0, r = 2)
        assertContentEquals(exact[0], v[0], "the output")
        assertContentEquals(exact[1], v[1], "the histories")
        for (q in 0 until s) {
            for (j in 0 until 5) {
                val got = v[2].copyOfRange((q * 5 + j) * c, (q * 5 + j + 1) * c)
                val want = if (q == 2 && j < 2) exact[2].copyOfRange((q * 2 + j) * c, (q * 2 + j + 1) * c) else wide.copyOfRange((q * 5 + j) * c, (q * 5 + j + 1) * c)
                assertContentEquals(want, got, "slot $q record $j")
            }
        }
    }
}
