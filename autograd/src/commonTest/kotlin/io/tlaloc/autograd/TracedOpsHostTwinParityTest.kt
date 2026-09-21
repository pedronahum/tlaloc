package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank3
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.broadcastLike
import io.tlaloc.core.ops.broadcastToLike
import io.tlaloc.core.ops.div
import io.tlaloc.core.ops.exp
import io.tlaloc.core.ops.log
import io.tlaloc.core.ops.matmul
import io.tlaloc.core.ops.mean
import io.tlaloc.core.ops.minus
import io.tlaloc.core.ops.neg
import io.tlaloc.core.ops.plus
import io.tlaloc.core.ops.pow
import io.tlaloc.core.ops.relu
import io.tlaloc.core.ops.sigmoid
import io.tlaloc.core.ops.sqrt
import io.tlaloc.core.ops.step
import io.tlaloc.core.ops.sum
import io.tlaloc.core.ops.tanh
import io.tlaloc.core.ops.times
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.447 — audit finding B (docs/AD_SINGLE_ENGINE_AUDIT.md): the pre-F4
 * TracedOps spellings no longer carry private FloatArray forward loops — they
 * route through the certified `io.tlaloc.core.ops` host twins. These pins
 * certify the migration changed NOTHING: for every migrated op the traced
 * forward value must be RAW-BIT-IDENTICAL (`Float.toRawBits`) to the host twin
 * applied directly to DTensors, on both a quarter-grid (exactly representable)
 * and an irrational-constant input set — and, where the math is exact, to a
 * hand-derived literal. If any of these flips, the trace and host paths have
 * diverged: investigate, never loosen.
 *
 * `bmm` is the named twin-gap (`:core` has no rank-3 batched matmul): its kept
 * private loop is pinned against hand-derived literals so any accidental edit
 * to the loop trips here.
 */
class TracedOpsHostTwinParityTest {

    // Quarter-grid values: exactly representable, exercise signs and zero.
    private val quarterA = floatArrayOf(0.25f, -1.75f, 2.5f, -0.5f, 3.25f, 0f)
    private val quarterB = floatArrayOf(1.5f, 0.75f, -2.25f, 4f, -0.25f, 1f)

    // Irrational constants (π, √2, e, γ, √3, ln 2) — nothing cancels exactly.
    private val irrA = floatArrayOf(3.1415927f, 1.4142135f, 2.7182817f, 0.57721566f, 1.7320508f, 0.6931472f)
    private val irrB = floatArrayOf(1.6180339f, 2.2360680f, 0.36787945f, 1.2020569f, 0.90929743f, 2.3025851f)

    private fun assertBitEqual(expected: FloatArray, actual: FloatArray, ctx: String) {
        assertEquals(expected.size, actual.size, "$ctx: size")
        for (i in expected.indices) {
            assertEquals(
                expected[i].toRawBits(),
                actual[i].toRawBits(),
                "$ctx[$i]: expected ${expected[i]} got ${actual[i]} (raw bits differ)",
            )
        }
    }

    private fun vec(data: FloatArray): DTensor<Rank1<Sym>, F32> = Tensors.f32Vector(data)

    /** Trace `f` over one rank-1 leaf and return the result entry's cached value. */
    private fun traced1(
        data: FloatArray,
        f: (Tracer<Rank1<Sym>>) -> Tracer<*>,
    ): FloatArray {
        val tape = Tape()
        return f(tape.traceLeaf(vec(data))).entry.value
    }

    /** Trace `f` over two rank-1 leaves and return the result entry's cached value. */
    private fun traced2(
        a: FloatArray,
        b: FloatArray,
        f: (Tracer<Rank1<Sym>>, Tracer<Rank1<Sym>>) -> Tracer<*>,
    ): FloatArray {
        val tape = Tape()
        return f(tape.traceLeaf(vec(a)), tape.traceLeaf(vec(b))).entry.value
    }

    // ------------------------------------------------------------------
    // Elementwise binaries
    // ------------------------------------------------------------------

    @Test
    fun elementwiseBinariesMatchHostTwinsBitForBit() {
        for ((a, b) in listOf(quarterA to quarterB, irrA to irrB)) {
            assertBitEqual((vec(a) + vec(b)).hostF32(), traced2(a, b) { x, y -> x + y }, "add")
            assertBitEqual((vec(a) - vec(b)).hostF32(), traced2(a, b) { x, y -> x - y }, "sub")
            assertBitEqual((vec(a) * vec(b)).hostF32(), traced2(a, b) { x, y -> x * y }, "mul")
            assertBitEqual((vec(a) / vec(b)).hostF32(), traced2(a, b) { x, y -> x / y }, "div")
        }
        // pow needs a positive base for irrational exponents.
        assertBitEqual(
            vec(irrA).pow(vec(irrB)).hostF32(),
            traced2(irrA, irrB) { x, y -> x.pow(y) },
            "pow",
        )
    }

    @Test
    fun addAndMatmulMatchHandDerivedQuarterGridValues() {
        // 0.25+1.5, -1.75+0.75, 2.5-2.25, -0.5+4, 3.25-0.25, 0+1 — all exact.
        assertBitEqual(
            floatArrayOf(1.75f, -1f, 0.25f, 3.5f, 3f, 1f),
            traced2(quarterA, quarterB) { x, y -> x + y },
            "add hand-derived",
        )
        // [[0.25, -1.75], [2.5, -0.5]] · [[1.5, 0.75], [-2.25, 4]] — exact.
        val tape = Tape()
        val a = tape.traceLeaf<Rank2<Sym, Sym>>(
            Tensors.f32Matrix(2, 2, floatArrayOf(0.25f, -1.75f, 2.5f, -0.5f)),
        )
        val b = tape.traceLeaf<Rank2<Sym, Sym>>(
            Tensors.f32Matrix(2, 2, floatArrayOf(1.5f, 0.75f, -2.25f, 4f)),
        )
        assertBitEqual(
            floatArrayOf(4.3125f, -6.8125f, 4.875f, -0.125f),
            (a matmul b).entry.value,
            "matmul hand-derived",
        )
    }

    // ------------------------------------------------------------------
    // Elementwise unaries
    // ------------------------------------------------------------------

    @Test
    fun elementwiseUnariesMatchHostTwinsBitForBit() {
        for (data in listOf(quarterA, irrA)) {
            assertBitEqual(vec(data).neg().hostF32(), traced1(data) { it.neg() }, "neg")
            assertBitEqual(vec(data).relu().hostF32(), traced1(data) { it.relu() }, "relu")
            assertBitEqual(vec(data).step().hostF32(), traced1(data) { it.step() }, "step")
            assertBitEqual(vec(data).exp().hostF32(), traced1(data) { it.exp() }, "exp")
            assertBitEqual(vec(data).tanh().hostF32(), traced1(data) { it.tanh() }, "tanh")
            assertBitEqual(vec(data).sigmoid().hostF32(), traced1(data) { it.sigmoid() }, "sigmoid")
        }
        // sqrt / log need non-negative / positive inputs.
        assertBitEqual(vec(irrA).sqrt().hostF32(), traced1(irrA) { it.sqrt() }, "sqrt")
        assertBitEqual(vec(irrA).log().hostF32(), traced1(irrA) { it.log() }, "log")
    }

    // ------------------------------------------------------------------
    // Matmul + the bmm twin-gap
    // ------------------------------------------------------------------

    @Test
    fun matmulMatchesHostTwinBitForBit() {
        val tape = Tape()
        val aD = Tensors.f32Matrix<Sym, Sym>(2, 3, irrA)
        val bD = Tensors.f32Matrix<Sym, Sym>(3, 2, irrB)
        val a = tape.traceLeaf(aD)
        val b = tape.traceLeaf(bD)
        assertBitEqual((aD matmul bD).hostF32(), (a matmul b).entry.value, "matmul")
    }

    @Test
    fun bmmTwinGapLoopMatchesHandDerivedValues() {
        // TWIN-GAP guard: `bmm` keeps its private loop (no rank-3 host twin) —
        // pin it against hand-derived per-batch products on exact inputs.
        val tape = Tape()
        val a = tape.traceLeaf<Rank3<Sym, Sym, Sym>>(
            Tensors.f32Tensor3(2, 2, 2, floatArrayOf(1f, 2f, 3f, 4f, 0.5f, -1f, 2f, 0.25f)),
        )
        val b = tape.traceLeaf<Rank3<Sym, Sym, Sym>>(
            Tensors.f32Tensor3(2, 2, 2, floatArrayOf(2f, 0f, 1f, 3f, -1f, 2f, 4f, 0.5f)),
        )
        assertBitEqual(
            floatArrayOf(4f, 6f, 10f, 12f, -4.5f, 0.5f, -1f, 4.125f),
            (a bmm b).entry.value,
            "bmm hand-derived",
        )
    }

    // ------------------------------------------------------------------
    // Reductions
    // ------------------------------------------------------------------

    @Test
    fun fullReductionsMatchHostTwinsBitForBit() {
        for (data in listOf(quarterA, irrA)) {
            assertBitEqual(vec(data).sum().hostF32(), traced1(data) { it.sum() }, "sum")
            assertBitEqual(vec(data).mean().hostF32(), traced1(data) { it.mean() }, "mean")
        }
        // Empty-input mean keeps the tape's structural-zero convention (0f).
        val emptyMean = traced1(FloatArray(0)) { it.mean() }
        assertEquals(1, emptyMean.size, "mean(empty) is scalar")
        assertTrue(emptyMean[0] == 0f, "mean(empty) must be exactly 0f, got ${emptyMean[0]}")
    }

    @Test
    fun axisReductionsMatchHostTwinsBitForBit() {
        val dims = intArrayOf(2, 3, 2)
        val data = FloatArray(12) { irrA[it % irrA.size] * (1 + it % 5) }
        val d = Tensors.f32Tensor3<Sym, Sym, Sym>(dims[0], dims[1], dims[2], data)
        for (axes in listOf(intArrayOf(0), intArrayOf(1), intArrayOf(0, 2), intArrayOf(2, 0))) {
            val tape = Tape()
            val t = tape.traceLeaf(d)
            val sortedAxes = axes.distinct().sorted().toIntArray()
            assertBitEqual(
                d.sum(*sortedAxes).hostF32(),
                t.sum<Shape>(axes).entry.value,
                "sum(${axes.toList()})",
            )
            assertBitEqual(
                d.mean(*sortedAxes).hostF32(),
                t.mean<Shape>(axes).entry.value,
                "mean(${axes.toList()})",
            )
        }
    }

    // ------------------------------------------------------------------
    // Broadcasts (pure replication — pins are copy-exactness)
    // ------------------------------------------------------------------

    @Test
    fun scalarBroadcastOperatorMatchesHostTwinComposition() {
        val tape = Tape()
        val row = tape.traceLeaf(vec(irrA))
        val s = tape.traceLeaf<ScalarShape>(Tensors.f32Scalar(1.4142135f))
        val got = (row + s).entry.value
        val expected = (vec(irrA) + broadcastLike(1.4142135f, vec(irrA))).hostF32()
        assertBitEqual(expected, got, "row + scalar (BROADCAST + ADD)")
    }

    @Test
    fun rowColInnerBatchAlongBroadcastsMatchHostTwinBitForBit() {
        val m = Tensors.f32Matrix<Sym, Sym>(2, 3, irrA)
        val rowD = vec(floatArrayOf(1.6180339f, 2.2360680f, 0.36787945f))
        val colD = vec(floatArrayOf(1.2020569f, 0.90929743f))
        run { // broadcastRow: [3] → [2, 3]
            val tape = Tape()
            val got = tape.traceLeaf(m).broadcastRow(tape.traceLeaf(rowD)).entry.value
            assertBitEqual(broadcastToLike(rowD, m).hostF32(), got, "broadcastRow")
            // Hand-derived: two copies of the row.
            assertBitEqual(rowD.hostF32() + rowD.hostF32(), got, "broadcastRow hand-derived")
        }
        run { // broadcastCol: [2] → [2, 3]
            val tape = Tape()
            val got = tape.traceLeaf(m).broadcastCol(tape.traceLeaf(colD)).entry.value
            val c = colD.hostF32()
            assertBitEqual(
                floatArrayOf(c[0], c[0], c[0], c[1], c[1], c[1]),
                got,
                "broadcastCol hand-derived",
            )
        }
        val t3 = Tensors.f32Tensor3<Sym, Sym, Sym>(2, 2, 3, FloatArray(12) { irrB[it % irrB.size] })
        run { // broadcastInner: [3] → [2, 2, 3]
            val tape = Tape()
            val got = tape.traceLeaf(t3).broadcastInner(tape.traceLeaf(rowD)).entry.value
            assertBitEqual(broadcastToLike(rowD, t3).hostF32(), got, "broadcastInner")
        }
        run { // broadcastBatch: [2, 3] → [2, 2, 3]
            val tape = Tape()
            val batchD = Tensors.f32Matrix<Sym, Sym>(2, 3, irrA)
            val got = tape.traceLeaf(t3).broadcastBatch(tape.traceLeaf(batchD)).entry.value
            assertBitEqual(broadcastToLike(batchD, t3).hostF32(), got, "broadcastBatch")
        }
        run { // broadcastAlong axis 1: [2] → [2, 2, 3]
            val tape = Tape()
            val got = tape.traceLeaf(t3).broadcastAlong<Shape>(tape.traceLeaf(colD), axis = 1).entry.value
            val aligned = DTensor<Shape, F32>(
                HostF32Storage(colD.hostF32().copyOf()), intArrayOf(1, 2, 1), F32,
            )
            assertBitEqual(broadcastToLike(aligned, t3).hostF32(), got, "broadcastAlong axis 1")
            // Hand-derived: c0 c0 c0 c1 c1 c1 per batch slab.
            val c = colD.hostF32()
            assertBitEqual(
                floatArrayOf(c[0], c[0], c[0], c[1], c[1], c[1], c[0], c[0], c[0], c[1], c[1], c[1]),
                got,
                "broadcastAlong hand-derived",
            )
        }
    }
}
