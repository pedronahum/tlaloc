package io.tlaloc.autograd

import io.tlaloc.core.Bounded
import io.tlaloc.core.DTensor
import io.tlaloc.core.DimBound
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Named
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

object MaxSeqT : DimBound(8)
object MaxBatchT : DimBound(3)
object OtherT : DimBound(8)

class BoundedProgramTest {

    private val hidden = 3
    private val seqLadder = BucketLadders(mapOf(MaxSeqT to listOf(1, 2, 4, 8)))

    private fun f32(dims: IntArray, f: (Int) -> Float): DTensor<Shape, F32> =
        DTensor(HostF32Storage(FloatArray(dims.fold(1) { a, b -> a * b }) { f(it) }), dims, F32)

    private fun values(n: Int, salt: Int) = FloatArray(n) { ((it * 37 + salt * 11) % 17 - 8) / 5f }

    /** Captures a hand-written program at one concrete shape and runs it in the interpreter. */
    private fun static(x: DTensor<Shape, F32>, f: (Tracer<Shape>) -> Tracer<*>): FloatArray {
        val fn = captureN(listOf(x)) { f(it.single()) }
        return DxirInterpreter.evalFunction(fn, listOf(x.hostF32())).single()
    }

    private fun assertClose(want: FloatArray, got: FloatArray, what: String, tol: Float = 1e-6f) {
        assertEquals(want.size, got.size, "$what: size")
        for (i in want.indices) {
            assertTrue(abs(want[i] - got[i]) <= tol * maxOf(1f, abs(want[i])), "$what [$i]: want ${want[i]} got ${got[i]}")
        }
    }

    @Test
    fun `specOf reads bounds from the type, positional and inside Named`() {
        val s = specOf<Rank3<Named<SeqLen, Bounded<MaxSeqT>>, Sym, Bounded<MaxBatchT>>>(F32, 5)
        assertEquals(listOf(AxisSpec.Bounded(MaxSeqT), AxisSpec.Fixed(5), AxisSpec.Bounded(MaxBatchT)), s.axes)
        assertEquals(listOf(MaxSeqT, MaxBatchT), s.bounds)
        val e = assertFailsWith<IllegalArgumentException> {
            specFromType(kotlin.reflect.typeOf<Rank2<Bounded<MaxSeqT>, Sym>>(), F32, intArrayOf(1, 2))
        }
        assertTrue("takes 1 fixed size(s); 2 given" in e.message!!, e.message)
    }

    @Test
    fun `a row-wise program matches the static program at every size, exact and bucketed`() {
        val p = boundedProgram(
            "rowwise",
            listOf(specOf<Rank2<Named<SeqLen, Bounded<MaxSeqT>>, Named<Hidden, Sym>>>(F32, hidden)),
            specOf<Rank2<Named<SeqLen, Bounded<MaxSeqT>>, Named<Hidden, Sym>>>(F32, hidden),
        ) { xs, _ -> (xs[0] * xs[0]).tanh() + 1f }
        for (n in 1..MaxSeqT.max) {
            val x = f32(intArrayOf(n, hidden)) { values(n * hidden, n)[it] }
            val want = static(x) { (it * it).tanh() + 1f }
            assertClose(want, p.run(listOf(x)).hostF32(), "exact n=$n")
            val b = p.runBucketed(listOf(x), seqLadder)
            assertEquals(listOf(n, hidden), b.dims.toList())
            assertClose(want, b.hostF32(), "bucketed n=$n")
        }
        // One trace per distinct size: the buckets 1, 2, 4, 8 are exact sizes too.
        assertEquals(MaxSeqT.max, p.traceCount)
    }

    @Test
    fun `a masked mean over the bounded axis matches the static mean at every size`() {
        val p = boundedProgram(
            "masked_mean",
            listOf(specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, hidden)),
            specOf<Rank1<Sym>>(F32, hidden),
        ) { xs, ctx ->
            val x = xs[0]
            val masked = x * x.broadcastAlong<Shape>(ctx.validMask(MaxSeqT), 0)
            masked.sum<Rank1<Sym>>(intArrayOf(0)) / ctx.validLength(MaxSeqT)
        }
        for (n in 1..MaxSeqT.max) {
            val x = f32(intArrayOf(n, hidden)) { values(n * hidden, 3 * n)[it] }
            val want = static(x) { it.mean<Shape>(intArrayOf(0)) }
            assertClose(want, p.run(listOf(x)).hostF32(), "exact n=$n")
            assertClose(want, p.runBucketed(listOf(x), seqLadder).hostF32(), "bucketed n=$n")
        }
        assertTrue(p.checkPadding(seqLadder).passed)
    }

    @Test
    fun `a masked softmax over the bounded axis matches the static softmax at every size`() {
        val p = boundedProgram(
            "masked_softmax",
            listOf(specOf<Rank1<Bounded<MaxSeqT>>>(F32)),
            specOf<Rank1<Bounded<MaxSeqT>>>(F32),
        ) { xs, ctx -> (xs[0] + (ctx.validMask(MaxSeqT) - 1f) * 1e9f).softmax() }
        for (n in 1..MaxSeqT.max) {
            val x = f32(intArrayOf(n)) { values(n, n)[it] }
            val want = static(x) { it.softmax() }
            assertClose(want, p.run(listOf(x)).hostF32(), "exact n=$n")
            assertClose(want, p.runBucketed(listOf(x), seqLadder).hostF32(), "bucketed n=$n")
        }
    }

    @Test
    fun `two bounds on one input, masked sum over one of them, every size pair`() {
        val ladders = BucketLadders(mapOf(MaxBatchT to listOf(1, 3), MaxSeqT to listOf(2, 8)))
        val p = boundedProgram(
            "two_bounds",
            listOf(specOf<Rank2<Bounded<MaxBatchT>, Bounded<MaxSeqT>>>(F32)),
            specOf<Rank1<Bounded<MaxBatchT>>>(F32),
        ) { xs, ctx ->
            val x = xs[0]
            (x * x.broadcastAlong<Shape>(ctx.validMask(MaxSeqT), 1)).sum<Shape>(intArrayOf(1))
        }
        for (b in 1..MaxBatchT.max) for (n in 1..MaxSeqT.max) {
            val x = f32(intArrayOf(b, n)) { values(b * n, b + 5 * n)[it] }
            val want = static(x) { it.sum<Shape>(intArrayOf(1)) }
            assertClose(want, p.run(listOf(x)).hostF32(), "exact $b x $n")
            assertClose(want, p.runBucketed(listOf(x), ladders).hostF32(), "bucketed $b x $n")
        }
        // Edges: batch {1, 2, 3} x seq {1, 2, 3, 8}, all in the cross product.
        assertEquals(12, p.edgeSizes(ladders).size)
        assertTrue(p.checkPadding(ladders).passed)
    }

    @Test
    fun `an I32 index input with a bounded axis, through an embedding`() {
        val vocab = 5
        val table = f32(intArrayOf(vocab, hidden)) { it * 0.25f - 1f }
        val p = boundedProgram(
            "embed",
            listOf(specOf<Rank2<Sym, Sym>>(F32, vocab, hidden), specOf<Rank1<Bounded<MaxSeqT>>>(I32)),
            specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, hidden),
        ) { xs, _ -> xs[0].embedding<Shape>(xs[1]).tanh() }
        for (n in 1..MaxSeqT.max) {
            val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(n) { (it * 3) % vocab }), intArrayOf(n), I32)
            val exact = p.run(listOf(table, ids)).hostF32()
            val want = FloatArray(n * hidden) { i ->
                kotlin.math.tanh(table.hostF32()[((i / hidden) * 3 % vocab) * hidden + i % hidden])
            }
            assertClose(want, exact, "exact n=$n", 1e-5f)
            assertClose(want, p.runBucketed(listOf(table, ids), seqLadder).hostF32(), "bucketed n=$n", 1e-5f)
        }
    }

    @Test
    fun `checkPadding catches a mean that ignores the mask`() {
        val p = boundedProgram(
            "unmasked_mean",
            listOf(specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, hidden)),
            specOf<Rank1<Sym>>(F32, hidden),
        ) { xs, _ -> xs[0].mean<Shape>(intArrayOf(0)) }
        val report = p.checkPadding(seqLadder)
        assertTrue(!report.passed, "an unmasked mean divides by the bucket size, not the length: $report")
        // The worst case is a size inside a bucket, never a bucket's own size (no padding there).
        val worst = report.worstSizes!![MaxSeqT]!!
        assertTrue(worst !in seqLadder.ladder(MaxSeqT), "worst at $worst")
        assertTrue(report.maxDifference > 0.01f, "$report")
        // A sum needs no mask: zero padding adds nothing.
        val sum = boundedProgram(
            "unmasked_sum",
            listOf(specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, hidden)),
            specOf<Rank1<Sym>>(F32, hidden),
        ) { xs, _ -> xs[0].sum<Shape>(intArrayOf(0)) }
        assertTrue(sum.checkPadding(seqLadder).passed)
    }

    /** `sum over real rows of (x·W - t + bias)^2 / len`; masked unless [masked] is false. */
    private fun mse(masked: Boolean, bias: Float = 0f) = boundedProgram(
        if (masked) "masked_mse" else "unmasked_mse",
        listOf(
            specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, hidden),
            specOf<Rank2<Sym, Sym>>(F32, hidden, hidden),
            specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, hidden),
        ),
        specOf<io.tlaloc.core.ScalarShape>(F32),
    ) { xs, ctx ->
        @Suppress("UNCHECKED_CAST")
        val r = ((xs[0] as Tracer<Rank2<Sym, Sym>>) matmul (xs[1] as Tracer<Rank2<Sym, Sym>>)) as Tracer<Shape> - xs[2] + bias
        val sq = if (masked) r * r * r.broadcastAlong<Shape>(ctx.validMask(MaxSeqT), 0) else r * r
        sq.sum() / ctx.validLength(MaxSeqT)
    }

    @Test
    fun `valueAndGrad gives the loss and the analytic gradients at every size, exact and bucketed`() {
        val step = mse(masked = true).valueAndGrad(listOf(1, 0))
        assertEquals(listOf(0, 2, 2), step.outputs.map { it.rank })
        val w = f32(intArrayOf(hidden, hidden)) { values(hidden * hidden, 99)[it] }
        for (n in 1..MaxSeqT.max) {
            val x = f32(intArrayOf(n, hidden)) { values(n * hidden, n)[it] }
            val t = f32(intArrayOf(n, hidden)) { values(n * hidden, 3 * n + 1)[it] }
            // Oracle: r = xW - t, L = sum(r^2)/n, dL/dW = (2/n) x^T r, dL/dx = (2/n) r W^T.
            val xv = x.hostF32(); val wv = w.hostF32(); val tv = t.hostF32()
            val r = FloatArray(n * hidden) { i ->
                val row = i / hidden; val col = i % hidden
                var acc = 0f
                for (k in 0 until hidden) acc += xv[row * hidden + k] * wv[k * hidden + col]
                acc - tv[i]
            }
            val loss = r.fold(0f) { a, v -> a + v * v } / n
            val dW = FloatArray(hidden * hidden) { i ->
                val a = i / hidden; val b = i % hidden
                var acc = 0f
                for (row in 0 until n) acc += xv[row * hidden + a] * r[row * hidden + b]
                2f * acc / n
            }
            val dX = FloatArray(n * hidden) { i ->
                val row = i / hidden; val a = i % hidden
                var acc = 0f
                for (b in 0 until hidden) acc += r[row * hidden + b] * wv[a * hidden + b]
                2f * acc / n
            }
            for ((label, outs) in listOf("exact" to step.runAll(listOf(x, w, t)), "bucketed" to step.runBucketedAll(listOf(x, w, t), seqLadder))) {
                assertEquals(3, outs.size)
                assertClose(floatArrayOf(loss), outs[0].hostF32(), "$label n=$n loss", 1e-5f)
                assertClose(dW, outs[1].hostF32(), "$label n=$n dW", 1e-5f)
                assertEquals(listOf(n, hidden), outs[2].dims.toList())
                assertClose(dX, outs[2].hostF32(), "$label n=$n dX", 1e-5f)
            }
        }
        assertTrue(step.checkPadding(seqLadder).passed)
    }

    @Test
    fun `checkPadding sees a loss that reads padded rows, in the value and the gradients`() {
        // With a bias, a padded row's residual is not zero, so an unmasked loss counts it.
        val bad = mse(masked = false, bias = 0.5f).valueAndGrad(listOf(1))
        val report = bad.checkPadding(seqLadder)
        assertTrue(!report.passed, "$report")
        assertTrue(mse(masked = true, bias = 0.5f).valueAndGrad(listOf(1)).checkPadding(seqLadder).passed)
    }

    @Test
    fun `valueAndGrad refuses a non-scalar output, an integer input and a second application`() {
        val vector = boundedProgram("v", listOf(specOf<Rank1<Bounded<MaxSeqT>>>(F32)), specOf<Rank1<Bounded<MaxSeqT>>>(F32)) { xs, _ -> xs[0] }
        assertTrue("one scalar output" in assertFailsWith<IllegalArgumentException> { vector.valueAndGrad(listOf(0)) }.message!!)
        val ints = boundedProgram(
            "i", listOf(specOf<Rank2<Sym, Sym>>(F32, 3, 2), specOf<Rank1<Bounded<MaxSeqT>>>(I32)), specOf<io.tlaloc.core.ScalarShape>(F32),
        ) { xs, _ -> xs[0].embedding<Shape>(xs[1]).sum() }
        assertTrue("not F32" in assertFailsWith<IllegalArgumentException> { ints.valueAndGrad(listOf(1)) }.message!!)
        val step = ints.valueAndGrad(listOf(0))
        assertTrue("already a gradient program" in assertFailsWith<IllegalArgumentException> { step.valueAndGrad(listOf(0)) }.message!!)
        assertTrue("read `outputs`" in assertFailsWith<IllegalStateException> { step.output }.message!!)
    }

    @Test
    fun `cache keys differ between programs that share a name, and repeat for one trace`() {
        val loss = mse(masked = true)
        val a = loss.valueAndGrad(listOf(0)).trace(mapOf(MaxSeqT to 4))
        val b = loss.valueAndGrad(listOf(1)).trace(mapOf(MaxSeqT to 4))
        assertTrue(a.cacheKey != b.cacheKey, "${a.cacheKey} vs ${b.cacheKey}")
        assertEquals(a.cacheKey, loss.valueAndGrad(listOf(0)).trace(mapOf(MaxSeqT to 4)).cacheKey)
        val one = boundedProgram("same", listOf(specOf<Rank1<Bounded<MaxSeqT>>>(F32)), specOf<Rank1<Bounded<MaxSeqT>>>(F32)) { xs, _ -> xs[0] * 2f }
        val two = boundedProgram("same", listOf(specOf<Rank1<Bounded<MaxSeqT>>>(F32)), specOf<Rank1<Bounded<MaxSeqT>>>(F32)) { xs, _ -> xs[0] * 3f }
        assertTrue(one.trace(mapOf(MaxSeqT to 2)).cacheKey != two.trace(mapOf(MaxSeqT to 2)).cacheKey)
    }

    @Test
    fun `checkPadding agrees on equal infinities and NaNs`() {
        val p = boundedProgram(
            "log_of_zero", listOf(specOf<Rank1<Bounded<MaxSeqT>>>(F32)), specOf<Rank1<Bounded<MaxSeqT>>>(F32),
        ) { xs, _ -> (xs[0] * 0f).log() }
        val report = p.checkPadding(seqLadder)
        assertTrue(report.passed, "-inf in both results is agreement: $report")
    }

    @Test
    fun `runs refuse sizes outside the bound, disagreeing sizes and wrong fixed axes`() {
        val p = boundedProgram(
            "pair",
            listOf(specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, 2), specOf<Rank1<Bounded<MaxSeqT>>>(F32)),
            specOf<Rank2<Bounded<MaxSeqT>, Sym>>(F32, 2),
        ) { xs, _ -> xs[0] * 2f }
        fun run(a: IntArray, b: IntArray) = p.run(listOf(f32(a) { 0f }, f32(b) { 0f }))
        run(intArrayOf(8, 2), intArrayOf(8))
        val over = assertFailsWith<IllegalArgumentException> { run(intArrayOf(9, 2), intArrayOf(9)) }
        assertTrue("outside 1..8" in over.message!!, over.message)
        val zero = assertFailsWith<IllegalArgumentException> { run(intArrayOf(0, 2), intArrayOf(0)) }
        assertTrue("outside 1..8" in zero.message!!, zero.message)
        val disagree = assertFailsWith<IllegalArgumentException> { run(intArrayOf(3, 2), intArrayOf(4)) }
        assertTrue("every axis of one bound has the same size" in disagree.message!!, disagree.message)
        val fixed = assertFailsWith<IllegalArgumentException> { run(intArrayOf(3, 5), intArrayOf(3)) }
        assertTrue("fixes it at 2" in fixed.message!!, fixed.message)
    }

    @Test
    fun `programs and ladders refuse inconsistent declarations`() {
        val noBound = assertFailsWith<IllegalArgumentException> {
            boundedProgram("x", listOf(specOf<Rank1<Sym>>(F32, 2)), specOf<Rank1<Sym>>(F32, 2)) { xs, _ -> xs[0] }
        }
        assertTrue("no input has a bounded axis" in noBound.message!!)
        val stray = assertFailsWith<IllegalArgumentException> {
            boundedProgram("x", listOf(specOf<Rank1<Bounded<MaxSeqT>>>(F32)), specOf<Rank1<Bounded<OtherT>>>(F32)) { xs, _ -> xs[0] }
        }
        assertTrue("no input has" in stray.message!!)
        val short = assertFailsWith<IllegalArgumentException> { BucketLadders(mapOf(MaxSeqT to listOf(2, 4))) }
        assertTrue("ends at 4, not at the bound 8" in short.message!!)
        assertEquals(listOf(16, 32, 64, 100), BucketLadders.powersOfTwo(listOf(object : DimBound(100) {})).ladders.values.single())
        assertEquals(listOf(1, 2, 4, 8), BucketLadders.powersOfTwo(listOf(MaxSeqT), minBucket = 1).ladder(MaxSeqT))
        assertEquals(listOf(8), BucketLadders.powersOfTwo(listOf(MaxSeqT)).ladder(MaxSeqT))
        val wrongShape = boundedProgram(
            "wrong_output", listOf(specOf<Rank1<Bounded<MaxSeqT>>>(F32)), specOf<Rank1<Sym>>(F32, 1),
        ) { xs, _ -> xs[0] }
        val e = assertFailsWith<IllegalArgumentException> { wrongShape.run(listOf(f32(intArrayOf(3)) { 0f })) }
        assertTrue("the output spec" in e.message!!, e.message)
    }
}
