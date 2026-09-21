package io.tlaloc.nn

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.split
import io.tlaloc.autograd.captureN
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.embedding
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.442 — F6: Embedding + EmbeddingBag certification through the F1
 * `valueAndGradients` route. Every value on the quarter-integer grid, every
 * assertion `==` — embedding gathers copy, the scatter-add sums dyadics, and
 * the flat-order f32 accumulations are exact throughout.
 *
 * Table (V=4, D=2): t0 = [0.25, −0.5], t1 = [1, 0.75], t2 = [−1.25, 2],
 * t3 = [0.5, −0.25]; row sums −0.25, 1.75, 0.75, 0.25. Indices [2, 0, 2, 1, 0]
 * (counts: v0×2, v1×1, v2×2, v3×0 — the collision-count oracle).
 */
class EmbeddingLayersTest {

    private val tableData = floatArrayOf(0.25f, -0.5f, 1f, 0.75f, -1.25f, 2f, 0.5f, -0.25f)
    private fun table() = Tensors.f32Matrix<Sym, Sym>(4, 2, tableData)
    private fun indices() = Tensors.i32Vector<Sym>(intArrayOf(2, 0, 2, 1, 0))

    /**
     * L = Σ out: every upstream row is ones, so dTable[v, :] = count(v) — the
     * collision-count gradient, hand-exact. The indices input's gradient is
     * the §0.4.419 ZEROS_LIKE structural zero, shaped like the indices.
     */
    @Test
    fun collisionCountGradientsHandExact() {
        val result = valueAndGradients(Embedding(table()), listOf(indices())) { y -> y.sum() }

        // 2·t2 + 2·t0 + t1 row sums = 1.5 − 0.5 + 1.75
        assertEquals(2.75f, result.loss)
        assertContentEquals(
            floatArrayOf(2f, 2f, 1f, 1f, 2f, 2f, 0f, 0f),
            result.gradients["table"]!!.hostF32(),
        )
        assertEquals(1, result.inputGradients.size)
        assertContentEquals(intArrayOf(5), result.inputGradients[0].dims)
        assertContentEquals(FloatArray(5), result.inputGradients[0].hostF32())
    }

    /**
     * L = Σ (out ⊙ c) with a constant per-position weight grid: dTable[v, :] =
     * Σ_{p: idx[p]=v} c[p, :] — the WEIGHTED collision sum, discriminating
     * between positions the way the unweighted count cannot.
     */
    @Test
    fun weightedCollisionGradientsHandExact() {
        val c = floatArrayOf(1f, -0.5f, 2f, 0.25f, -1f, 0.75f, 0.5f, 2f, -0.25f, 1f)
        val result = valueAndGradients(Embedding(table()), listOf(indices())) { y ->
            (y * y.constant<Shape>(c, intArrayOf(5, 2))).sum()
        }

        assertEquals(2.3125f, result.loss)
        assertContentEquals(
            floatArrayOf(
                1.75f, 1.25f, // v0: c[p1] + c[p4]
                0.5f, 2f,     // v1: c[p3]
                0f, 0.25f,    // v2: c[p0] + c[p2]
                0f, 0f,       // v3: never gathered
            ),
            result.gradients["table"]!!.hostF32(),
        )
    }

    /**
     * paddingIndex rows exactly zero in BOTH directions: the forward output's
     * padded rows are exact zeros (pinned through the captured PRIMAL — the
     * interpreter's EMBEDDING arm reading the `padding_index` attr off the
     * captured graph), and the padded vocab row's gradient is exactly zero
     * (the attr rides the primal onto the fused EMBEDDING_GRAD).
     */
    @Test
    fun paddingIndexRowsExactlyZeroBothDirections() {
        val idx = Tensors.i32Vector<Sym>(intArrayOf(2, 0, 2, 1))

        // Forward: through the raw trace spelling, non-scalar result allowed.
        val primal = captureN(listOf(table(), idx), "padded") { leaves ->
            leaves[0].embedding<Shape>(leaves[1], 2)
        }
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(tableData, floatArrayOf(2f, 0f, 2f, 1f)),
        )
        assertContentEquals(
            floatArrayOf(0f, 0f, 0.25f, -0.5f, 0f, 0f, 1f, 0.75f),
            out[0],
        )

        // Backward: through the model route, L = Σ out.
        val result = valueAndGradients(Embedding(table(), paddingIndex = 2), listOf(idx)) { y -> y.sum() }
        assertEquals(1.5f, result.loss) // t0 + t1 row sums only
        assertContentEquals(
            floatArrayOf(1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f),
            result.gradients["table"]!!.hostF32(),
        )
    }

    /**
     * The trace-vs-hand-built-DXIR oracle: EMBEDDING → SUM built through the
     * `:nn` capture AND directly through [DxirBuilder] with the I32 index
     * param, both through the SAME reverse transform, elementwise equal — and
     * both against the hand values. Also the ROUTE PIN: the captured indices
     * param comes out I32-TYPED (the §0.4.442 dtype plumbing), which is
     * exactly what makes the transform emit its ZEROS_LIKE structural zero.
     */
    @Test
    fun tracedEmbeddingMatchesHandBuiltDxirThroughTheSameTransform() {
        val captured = capture(Embedding(table()), listOf(indices())) { y -> y.sum() }
        assertEquals(2, captured.primal.params.size) // indices first, then the table
        assertEquals(I32, captured.primal.params[0].type.dtype)
        assertEquals(F32, captured.primal.params[1].type.dtype)

        val hand = DxirBuilder.function("embeddingHand") {
            val pIdx = param("idx", DxirType(I32, listOf(5)))
            val pTable = param("table", DxirType(F32, listOf(4, 2)))
            val emb = op(OpKind.EMBEDDING, listOf(pTable, pIdx), DxirType(F32, listOf(5, 2)))
            val loss = op(OpKind.SUM, listOf(emb), DxirType(F32, emptyList()))
            listOf(loss)
        }

        val values = listOf(floatArrayOf(2f, 0f, 2f, 1f, 0f), tableData)
        val fromTrace = DxirInterpreter.evalFunction(DxirReverseTransform.apply(captured.primal), values)
        val fromHand = DxirInterpreter.evalFunction(DxirReverseTransform.apply(hand), values)

        assertEquals(fromHand.size, fromTrace.size)
        for (i in fromHand.indices) assertContentEquals(fromHand[i], fromTrace[i])

        assertContentEquals(FloatArray(5), fromTrace[0]) // dIdx: the structural zero
        assertContentEquals(floatArrayOf(2f, 2f, 1f, 1f, 2f, 2f, 0f, 0f), fromTrace[1])
    }

    /**
     * EmbeddingBag, DiffKT's Sum reduction, hand-exact through
     * `valueAndGradients` via the [EmbeddingBag.withOffsets] Layer view.
     * Bags over indices [2,0,2,1,0] at offsets [0,2,3]: [t2+t0], [t2],
     * [t1+t0] (the LAST bag runs to the end); L = Σ (bags ⊙ c) puts a
     * distinct upstream row on each bag, so dTable[v, :] sums the c-rows of
     * the bags that gathered v — hand-exact per position.
     */
    @Test
    fun embeddingBagSumReductionHandExact() {
        val model = EmbeddingBag(table(), EmbeddingBag.Reduction.Sum).withOffsets(intArrayOf(0, 2, 3))
        val c = floatArrayOf(1f, -0.5f, 2f, 0.25f, -1f, 0.75f)
        val result = valueAndGradients(model, listOf(indices())) { y ->
            (y * y.constant<Shape>(c, intArrayOf(3, 2))).sum()
        }

        // bags: [−1, 1.5], [−1.25, 2], [1.25, 0.25] ⊙ c → −1.75 − 2 − 1.0625
        assertEquals(-4.8125f, result.loss)
        assertContentEquals(
            floatArrayOf(
                0f, 0.25f,   // v0: c0 (bag 0) + c2 (bag 2)
                -1f, 0.75f,  // v1: c2
                3f, -0.25f,  // v2: c0 + c1
                0f, 0f,      // v3
            ),
            result.gradients["table"]!!.hostF32(),
        )
        assertContentEquals(FloatArray(5), result.inputGradients[0].hostF32())
    }

    /**
     * Rank-2 indices flatten through the dtype-preserving RESHAPE before
     * bagging (DiffKT's own `indices.flatten()`): [[2,0],[2,1]] at offsets
     * [0,3] → bags [t2+t0+t2], [t1]; L = Σ out. The input gradient comes back
     * in the RANK-2 index shape — the structural zero named by the reshaped
     * param.
     */
    @Test
    fun embeddingBagFlattensRank2Indices() {
        val idx = Tensors.i32Matrix<Sym, Sym>(2, 2, intArrayOf(2, 0, 2, 1))
        val model = EmbeddingBag(table()).withOffsets(intArrayOf(0, 3))
        val result = valueAndGradients(model, listOf(idx)) { y -> y.sum() }

        // (2·t2 + t0) + t1 row sums = (1.5 − 0.25) + 1.75
        assertEquals(3f, result.loss)
        assertContentEquals(
            floatArrayOf(1f, 1f, 1f, 1f, 2f, 2f, 0f, 0f),
            result.gradients["table"]!!.hostF32(),
        )
        assertContentEquals(intArrayOf(2, 2), result.inputGradients[0].dims)
        assertContentEquals(FloatArray(4), result.inputGradients[0].hostF32())
    }

    /**
     * The full-chain integration: Embedding → Flatten → Dense in a Sequential,
     * rank-2 indices [[2,0],[2,1]], L = Σ y. The table gradient arrives
     * THROUGH the dense weights (upstream row wᵀ scattered per position), the
     * dense gradients through the gathered rows — every value hand-exact.
     */
    @Test
    fun embeddingFlattenDenseChainHandExact() {
        val idx = Tensors.i32Matrix<Sym, Sym>(2, 2, intArrayOf(2, 0, 2, 1))
        val dense = Dense(
            Tensors.f32Matrix<Sym, Sym>(4, 1, floatArrayOf(0.5f, -1f, 2f, 0.25f)),
            Tensors.f32Vector<Sym>(floatArrayOf(0.5f)),
        )
        val model = Sequential(Embedding(table()), Flatten, dense)
        val result = valueAndGradients(model, listOf(idx)) { y -> y.sum() }

        assertEquals(-1.6875f, result.loss)
        assertEquals(listOf("0.table", "2.w", "2.b"), result.gradients.keys.toList())
        assertContentEquals(
            floatArrayOf(
                2f, 0.25f, // v0: upstream at flat position (0,1)
                2f, 0.25f, // v1: upstream at flat position (1,1)
                1f, -2f,   // v2: positions (0,0) + (1,0), each [0.5, −1]
                0f, 0f,    // v3
            ),
            result.gradients["0.table"]!!.hostF32(),
        )
        assertContentEquals(floatArrayOf(-2.5f, 4f, 1.25f, 0.25f), result.gradients["2.w"]!!.hostF32())
        assertContentEquals(floatArrayOf(2f), result.gradients["2.b"]!!.hostF32())
        assertContentEquals(FloatArray(4), result.inputGradients[0].hostF32())
    }

    /** The companion draws under the F2 key discipline, bit-exact vs the raw initializer. */
    @Test
    fun companionInitsBitExactUnderTheKeyDiscipline() {
        val key = RandomKey.fromSeed(42L)
        val expected = gaussianInit(key.split(1)[0], intArrayOf(4, 2)).hostF32()
        assertContentEquals(expected, Embedding(4, 2, key).table.hostF32())
        assertContentEquals(
            expected,
            EmbeddingBag(4, 2, EmbeddingBag.Reduction.Sum, key).table.hostF32(),
        )
    }

    /** Functional rebuilds and the DiffKT-parity refusals. */
    @Test
    fun surfaceContracts() {
        val emb = Embedding(table(), paddingIndex = 3)
        val updated = emb.withParameters(mapOf("table" to Tensors.f32Matrix<Sym, Sym>(4, 2, FloatArray(8))))
        assertContentEquals(tableData, emb.table.hostF32()) // original untouched
        assertContentEquals(FloatArray(8), updated.table.hostF32())
        assertEquals(3, updated.paddingIndex)
        assertFailsWith<IllegalArgumentException> { emb.withParameters(mapOf("w" to table())) }

        // The Layer-form EmbeddingBag call refuses, like DiffKT's vararg invoke.
        assertFailsWith<IllegalArgumentException> {
            valueAndGradients(EmbeddingBag(table()), listOf(indices())) { y -> y.sum() }
        }

        // Non-I32 input into Embedding refuses at trace time.
        assertFailsWith<IllegalArgumentException> {
            valueAndGradients(Embedding(table()), listOf(table())) { y -> y.sum() }
        }

        // The float-encoding exactness cap is loud, not a silent rounding.
        val huge = Tensors.i32Vector<Sym>(intArrayOf(16_777_216))
        val ex = assertFailsWith<IllegalArgumentException> {
            valueAndGradients(Embedding(table()), listOf(huge)) { y -> y.sum() }
        }
        assertTrue(ex.message!!.contains("2^24"))
    }

    /**
     * Re-bind through a held [CapturedStep]: one capture, two index bindings —
     * the SAME graph re-runs with new indices per step (indices are an
     * I32-typed param, never a baked constant), each loss hand-exact.
     */
    @Test
    fun capturedStepRebindsNewIndices() {
        val model = Embedding(table())
        val step = capture(model, listOf(indices())) { y -> y.sum() }
        assertEquals(2.75f, step.run(model, listOf(indices())).loss)
        val other = Tensors.i32Vector<Sym>(intArrayOf(3, 3, 1, 0, 1))
        // 2·t3 + 2·t1 + t0 row sums = 0.5 + 3.5 − 0.25
        val rebound = step.run(model, listOf(other))
        assertEquals(3.75f, rebound.loss)
        assertContentEquals(
            floatArrayOf(1f, 1f, 2f, 2f, 0f, 0f, 2f, 2f),
            rebound.gradients["table"]!!.hostF32(),
        )
    }

    /** The raw SLICE/CONCAT spellings round-trip vs a hand-built DXIR twin. */
    @Test
    fun tracedSliceConcatMatchesHandBuiltDxirThroughTheSameTransform() {
        val xData = floatArrayOf(1f, -0.5f, 2f, 0.25f, -1.25f, 0.75f)
        val x = Tensors.f32Matrix<Sym, Sym>(3, 2, xData)

        // concat(slice(x, 0, 1), slice(x, 1, 3)) ⊙ c → sum: the reverse walks
        // ConcatRule's per-part slices and SliceRule's zero-pad back.
        val c = floatArrayOf(2f, -1f, 0.5f, 4f, -0.25f, 1f)
        val primal = captureN(listOf(x), "sliceConcat") { leaves ->
            val a = leaves[0].slice<Shape>(0, 1, 0)
            val b = leaves[0].slice<Shape>(1, 3, 0)
            val cat = io.tlaloc.autograd.concat<Shape>(listOf(a, b), 0)
            (cat * cat.constant<Shape>(c, intArrayOf(3, 2))).sum()
        }
        val hand = DxirBuilder.function("sliceConcatHand") {
            val px = param("x", DxirType(F32, listOf(3, 2)))
            val a = op(
                OpKind.SLICE, listOf(px), DxirType(F32, listOf(1, 2)),
                attrs = mapOf(
                    "start_indices" to listOf(0, 0), "limit_indices" to listOf(1, 2), "strides" to listOf(1, 1),
                ),
            )
            val b = op(
                OpKind.SLICE, listOf(px), DxirType(F32, listOf(2, 2)),
                attrs = mapOf(
                    "start_indices" to listOf(1, 0), "limit_indices" to listOf(3, 2), "strides" to listOf(1, 1),
                ),
            )
            val cat = op(OpKind.CONCAT, listOf(a, b), DxirType(F32, listOf(3, 2)), attrs = mapOf("dimension" to 0))
            val cw = const(c.copyOf(), DxirType(F32, listOf(3, 2)))
            val prod = op(OpKind.MUL, listOf(cat, cw), DxirType(F32, listOf(3, 2)))
            val loss = op(OpKind.SUM, listOf(prod), DxirType(F32, emptyList()))
            listOf(loss)
        }

        val fromTrace = DxirInterpreter.evalFunction(DxirReverseTransform.apply(primal), listOf(xData))
        val fromHand = DxirInterpreter.evalFunction(DxirReverseTransform.apply(hand), listOf(xData))
        assertEquals(fromHand.size, fromTrace.size)
        for (i in fromHand.indices) assertContentEquals(fromHand[i], fromTrace[i])
        // The concat is the identity permutation here, so dx = c exactly.
        assertContentEquals(c, fromTrace[0])
    }
}
