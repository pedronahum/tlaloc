package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Stage D.1-S2 (§0.4.41) — scalar-index-into-rank-1 GATHER substrate.
 *
 * Exercises [DxirInterpreter]'s GATHER and SCATTER arms plus the new [GatherRule] in
 * [VjpRegistry]. Hand-built dxir primals only — the FIR `arr[i]` lowering lands in
 * S3, plugin-level tests there.
 *
 * Scope for S2: `GATHER(arr: rank-1, idx: scalar I32) → scalar` and
 * `SCATTER(base: rank-1, idx: scalar I32, value: scalar) → rank-1`. General stablehlo-
 * style gather/scatter (multi-dim `start_indices` + `slice_sizes`) is out of scope;
 * our surface matches what per-segment brachistochrone-style primals need.
 */
class GatherTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val rank1_4 = DxirType(F32, listOf(4))

    // --- Forward interpreter ----------------------------------------------

    @Test
    fun gatherReadsSlotFromRank1Array() {
        val fn = DxirBuilder.function("gather") {
            val arr = param("arr", rank1_4)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), f32s)
            listOf(g)
        }
        // arr = [10, 20, 30, 40], idx = 2 → arr[2] = 30
        val out = DxirInterpreter.evalFunction(
            fn, listOf(floatArrayOf(10f, 20f, 30f, 40f), floatArrayOf(2f)),
        )
        assertEquals(1, out.size)
        assertEquals(1, out[0].size)
        assertEquals(30f, out[0][0])
    }

    @Test
    fun gatherOutOfBoundsIsFailLoud() {
        val fn = DxirBuilder.function("gather") {
            val arr = param("arr", rank1_4)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), f32s)
            listOf(g)
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn, listOf(floatArrayOf(10f, 20f, 30f, 40f), floatArrayOf(7f)),
            )
        }
    }

    @Test
    fun scatterReplacesSlotNonDestructively() {
        val fn = DxirBuilder.function("scatter") {
            val base = param("base", rank1_4)
            val idx = param("idx", i32s)
            val v = param("v", f32s)
            val s = op(OpKind.SCATTER, listOf(base, idx, v), rank1_4)
            listOf(s)
        }
        // base = [1, 2, 3, 4], idx = 1, v = 99 → [1, 99, 3, 4]
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(
                floatArrayOf(1f, 2f, 3f, 4f),
                floatArrayOf(1f),
                floatArrayOf(99f),
            ),
        )
        assertEquals(1, out.size)
        assertEquals(4, out[0].size)
        assertEquals(floatArrayOf(1f, 99f, 3f, 4f).toList(), out[0].toList())
    }

    @Test
    fun scatterIntoZerosProducesOneHot() {
        // Expresses the "into zeros" variant GatherRule's adjoint uses:
        //   SCATTER(BROADCAST(0, rank-1), idx, value) → one-hot vector.
        val fn = DxirBuilder.function("onehot") {
            val idx = param("idx", i32s)
            val v = param("v", f32s)
            val z = const(0.0f, f32s)
            val zeroVec = op(
                OpKind.BROADCAST,
                listOf(z),
                rank1_4,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            val s = op(OpKind.SCATTER, listOf(zeroVec, idx, v), rank1_4)
            listOf(s)
        }
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(floatArrayOf(2f), floatArrayOf(7f)),
        )
        assertEquals(floatArrayOf(0f, 0f, 7f, 0f).toList(), out[0].toList())
    }

    @Test
    fun gatherAfterScatterRoundTripAtSameIndex() {
        // SCATTER then GATHER at the same index yields the written value; the other
        // slots remain at `base`'s value.
        val fn = DxirBuilder.function("roundtrip") {
            val base = param("base", rank1_4)
            val idx = param("idx", i32s)
            val v = param("v", f32s)
            val s = op(OpKind.SCATTER, listOf(base, idx, v), rank1_4)
            val g = op(OpKind.GATHER, listOf(s, idx), f32s)
            listOf(g)
        }
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(
                floatArrayOf(1f, 2f, 3f, 4f),
                floatArrayOf(2f),
                floatArrayOf(55f),
            ),
        )
        assertEquals(55f, out[0][0])
    }

    // --- Reverse-mode: GatherRule ------------------------------------------

    @Test
    fun gradientOfGatherIsOneHotAtIndex() {
        // f(arr, idx) = arr[idx].  d/d(arr) = one-hot at slot `idx` with value 1.
        //               d/d(idx) = 0 (non-differentiable Int).
        val primal = DxirBuilder.function("gather") {
            val arr = param("arr", rank1_4)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), f32s)
            listOf(g)
        }
        val grad = DxirReverseTransform.apply(primal)
        assertEquals(2, grad.returns.size, "2-param primal should yield 2 gradients")
        // Evaluate at arr = [10, 20, 30, 40], idx = 1. Expected d(arr) = [0, 1, 0, 0].
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(10f, 20f, 30f, 40f), floatArrayOf(1f)),
        )
        assertEquals(floatArrayOf(0f, 1f, 0f, 0f).toList(), out[0].toList())
        // d(idx) is a typed zero scalar — `DxirReverseTransform.zeroValueFor(I32) = 0`
        // lowered through `const(0, i32s)` → single-entry FloatArray `[0]`.
        assertEquals(1, out[1].size)
        assertEquals(0f, out[1][0])
    }

    @Test
    fun gradientOfScaledGatherScalesOneHot() {
        // f(arr, idx) = 3 · arr[idx].  d/d(arr) = 3 · one-hot at idx.
        val primal = DxirBuilder.function("scaled") {
            val arr = param("arr", rank1_4)
            val idx = param("idx", i32s)
            val three = const(3.0f, f32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), f32s)
            val y = op(OpKind.MUL, listOf(three, g), f32s)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(0f, 0f, 0f, 0f), floatArrayOf(2f)),
        )
        assertEquals(floatArrayOf(0f, 0f, 3f, 0f).toList(), out[0].toList())
    }

    @Test
    fun gradientOfSumOfTwoGathersAccumulatesContributions() {
        // f(arr, i0, i1) = arr[i0] + arr[i1].  d/d(arr) = one-hot[i0] + one-hot[i1].
        // When i0 != i1, result has two slots set to 1; when i0 == i1, the single
        // slot is set to 2 (gradAccum's outer ADD accumulation).
        val primal = DxirBuilder.function("twoGather") {
            val arr = param("arr", rank1_4)
            val i0 = param("i0", i32s)
            val i1 = param("i1", i32s)
            val g0 = op(OpKind.GATHER, listOf(arr, i0), f32s)
            val g1 = op(OpKind.GATHER, listOf(arr, i1), f32s)
            val s = op(OpKind.ADD, listOf(g0, g1), f32s)
            listOf(s)
        }
        val grad = DxirReverseTransform.apply(primal)

        // Distinct indices: expect [0, 1, 1, 0].
        val distinct = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(10f, 20f, 30f, 40f), floatArrayOf(1f), floatArrayOf(2f)),
        )
        assertEquals(floatArrayOf(0f, 1f, 1f, 0f).toList(), distinct[0].toList())

        // Same index: expect [0, 2, 0, 0].
        val same = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(10f, 20f, 30f, 40f), floatArrayOf(1f), floatArrayOf(1f)),
        )
        assertEquals(floatArrayOf(0f, 2f, 0f, 0f).toList(), same[0].toList())
    }

    @Test
    fun scatterAddChainHasInPlaceTagsAfterReverseTransform() {
        // §0.4.46 — a linear SCATTER_ADD accumulator chain emitted by GatherRule on
        // multiple gathers of the same `arr` should have every SCATTER_ADD tagged
        // `"in_place" = true`. Each SCATTER_ADD's base operand is single-use (either
        // the previous SCATTER_ADD's result, or the first's zero-BROADCAST) so
        // destructive in-place lowering is safe.
        val primal = DxirBuilder.function("threeGathers") {
            val arr = param("arr", rank1_4)
            val i0 = param("i0", i32s)
            val i1 = param("i1", i32s)
            val i2 = param("i2", i32s)
            val g0 = op(OpKind.GATHER, listOf(arr, i0), f32s)
            val g1 = op(OpKind.GATHER, listOf(arr, i1), f32s)
            val g2 = op(OpKind.GATHER, listOf(arr, i2), f32s)
            val s01 = op(OpKind.ADD, listOf(g0, g1), f32s)
            val sum = op(OpKind.ADD, listOf(s01, g2), f32s)
            listOf(sum)
        }
        val grad = DxirReverseTransform.apply(primal)
        val scatterAdds = grad.body
            .filterIsInstance<io.tlaloc.ir.DxirOp>()
            .filter { it.op == OpKind.SCATTER_ADD }
        assertEquals(3, scatterAdds.size, "expected three SCATTER_ADD ops in grad body (one per gather)")
        for (sa in scatterAdds) {
            assertEquals(
                true,
                sa.attrs["in_place"],
                "SCATTER_ADD id=${sa.id} should have in_place=true — its base is single-use in the chain",
            )
        }
    }

    // --- §0.4.111: rank-2 GATHER (row-indexing) -----------------------------

    private val rank2_3x4 = DxirType(F32, listOf(3, 4))

    @Test
    fun rank2GatherReadsRowFromMatrix() {
        // arr is a 3x4 matrix [[10..13], [20..23], [30..33]]. idx=1 picks the
        // middle row → [20, 21, 22, 23].
        val fn = DxirBuilder.function("rowGather") {
            val arr = param("arr", rank2_3x4)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, listOf(4)))
            listOf(g)
        }
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(
                floatArrayOf(10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f, 30f, 31f, 32f, 33f),
                floatArrayOf(1f),
            ),
        )
        assertEquals(1, out.size)
        assertEquals(4, out[0].size)
        assertEquals(floatArrayOf(20f, 21f, 22f, 23f).toList(), out[0].toList())
    }

    @Test
    fun rank2GatherFirstAndLastRowsRoundTrip() {
        // Boundary check: idx=0 and idx=N-1 both work. Pre-fix this would have
        // failed the `arrType.rank == 1` precondition.
        val fn = DxirBuilder.function("rowGatherBounds") {
            val arr = param("arr", rank2_3x4)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, listOf(4)))
            listOf(g)
        }
        val backing = floatArrayOf(
            10f, 11f, 12f, 13f,
            20f, 21f, 22f, 23f,
            30f, 31f, 32f, 33f,
        )
        val firstRow = DxirInterpreter.evalFunction(fn, listOf(backing, floatArrayOf(0f)))
        assertEquals(floatArrayOf(10f, 11f, 12f, 13f).toList(), firstRow[0].toList())
        val lastRow = DxirInterpreter.evalFunction(fn, listOf(backing, floatArrayOf(2f)))
        assertEquals(floatArrayOf(30f, 31f, 32f, 33f).toList(), lastRow[0].toList())
    }

    @Test
    fun rank2GatherOutOfBoundsIsFailLoud() {
        val fn = DxirBuilder.function("rowGather") {
            val arr = param("arr", rank2_3x4)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), DxirType(F32, listOf(4)))
            listOf(g)
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(FloatArray(12), floatArrayOf(7f)),
            )
        }
    }

    @Test
    fun gradientOfRank2GatherIsOneHotRow() {
        // f(arr, idx) = sum(arr[idx, :]). The gradient wrt arr is a 3x4 matrix
        // whose row [idx] is all ones, others zero. Pin the GatherRule + rank-2
        // SCATTER_ADD interpreter path together.
        val rank1_4 = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("rowSum") {
            val arr = param("arr", rank2_3x4)
            val idx = param("idx", i32s)
            val row = op(OpKind.GATHER, listOf(arr, idx), rank1_4)
            val s = op(OpKind.SUM, listOf(row), f32s)
            listOf(s)
        }
        val grad = DxirReverseTransform.apply(primal)
        // idx=1: expect grad_arr to have row 1 = [1,1,1,1], others 0.
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(FloatArray(12), floatArrayOf(1f)),
        )
        val expected = floatArrayOf(
            0f, 0f, 0f, 0f,
            1f, 1f, 1f, 1f,
            0f, 0f, 0f, 0f,
        )
        assertEquals(expected.toList(), out[0].toList())
    }

    @Test
    fun gradientOfTwoRank2GathersAccumulatesPerRow() {
        // f(arr, i0, i1) = sum(arr[i0, :]) + sum(arr[i1, :]). For distinct
        // indices, two rows of grad_arr go to all ones; for matching indices,
        // one row goes to all twos (gradAccum's outer ADD).
        val rank1_4 = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("twoRowSum") {
            val arr = param("arr", rank2_3x4)
            val i0 = param("i0", i32s)
            val i1 = param("i1", i32s)
            val r0 = op(OpKind.GATHER, listOf(arr, i0), rank1_4)
            val r1 = op(OpKind.GATHER, listOf(arr, i1), rank1_4)
            val s0 = op(OpKind.SUM, listOf(r0), f32s)
            val s1 = op(OpKind.SUM, listOf(r1), f32s)
            val total = op(OpKind.ADD, listOf(s0, s1), f32s)
            listOf(total)
        }
        val grad = DxirReverseTransform.apply(primal)

        val distinct = DxirInterpreter.evalFunction(
            grad,
            listOf(FloatArray(12), floatArrayOf(0f), floatArrayOf(2f)),
        )
        val expectedDistinct = floatArrayOf(
            1f, 1f, 1f, 1f,
            0f, 0f, 0f, 0f,
            1f, 1f, 1f, 1f,
        )
        assertEquals(expectedDistinct.toList(), distinct[0].toList())

        val same = DxirInterpreter.evalFunction(
            grad,
            listOf(FloatArray(12), floatArrayOf(1f), floatArrayOf(1f)),
        )
        val expectedSame = floatArrayOf(
            0f, 0f, 0f, 0f,
            2f, 2f, 2f, 2f,
            0f, 0f, 0f, 0f,
        )
        assertEquals(expectedSame.toList(), same[0].toList())
    }

    @Test
    fun gradientOfScaledRank2GatherScalesEachOneHotElement() {
        // f(arr, idx) = sum(arr[idx, :]) * 5. Gradient row should be [5,5,5,5]
        // at idx, others zero. Exercises chain rule MulRule ∘ GatherRule on the
        // rank-2 path.
        val rank1_4 = DxirType(F32, listOf(4))
        val primal = DxirBuilder.function("scaledRowSum") {
            val arr = param("arr", rank2_3x4)
            val idx = param("idx", i32s)
            val five = const(5.0f, f32s)
            val row = op(OpKind.GATHER, listOf(arr, idx), rank1_4)
            val s = op(OpKind.SUM, listOf(row), f32s)
            val out = op(OpKind.MUL, listOf(s, five), f32s)
            listOf(out)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(FloatArray(12), floatArrayOf(2f)),
        )
        val expected = floatArrayOf(
            0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f,
            5f, 5f, 5f, 5f,
        )
        assertEquals(expected.toList(), out[0].toList())
    }

    // --- §0.4.114: rank-2 SCATTER (row-replace user write path) -------------

    @Test
    fun rank2ScatterReplacesRowNonDestructively() {
        // Replace row 1 of a 3x4 matrix with [99, 99, 99, 99]; other rows unchanged.
        val rank1_4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rowScatter") {
            val base = param("b", rank2_3x4)
            val idx = param("i", i32s)
            val v = param("v", rank1_4)
            val s = op(OpKind.SCATTER, listOf(base, idx, v), rank2_3x4)
            listOf(s)
        }
        val backing = floatArrayOf(
            10f, 11f, 12f, 13f,
            20f, 21f, 22f, 23f,
            30f, 31f, 32f, 33f,
        )
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(backing, floatArrayOf(1f), floatArrayOf(99f, 99f, 99f, 99f)),
        )
        val expected = floatArrayOf(
            10f, 11f, 12f, 13f,
            99f, 99f, 99f, 99f,
            30f, 31f, 32f, 33f,
        )
        assertEquals(expected.toList(), out[0].toList())
    }

    @Test
    fun rank2ScatterFirstAndLastRowsRoundTrip() {
        val rank1_4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rowScatter") {
            val base = param("b", rank2_3x4)
            val idx = param("i", i32s)
            val v = param("v", rank1_4)
            val s = op(OpKind.SCATTER, listOf(base, idx, v), rank2_3x4)
            listOf(s)
        }
        val backing = FloatArray(12) // all zeros
        val newRow = floatArrayOf(1f, 2f, 3f, 4f)

        val firstReplaced = DxirInterpreter.evalFunction(
            fn, listOf(backing, floatArrayOf(0f), newRow),
        )
        assertEquals(
            floatArrayOf(1f, 2f, 3f, 4f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f).toList(),
            firstReplaced[0].toList(),
        )
        val lastReplaced = DxirInterpreter.evalFunction(
            fn, listOf(backing, floatArrayOf(2f), newRow),
        )
        assertEquals(
            floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f).toList(),
            lastReplaced[0].toList(),
        )
    }

    @Test
    fun rank2ScatterOutOfBoundsIsFailLoud() {
        val rank1_4 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("rowScatter") {
            val base = param("b", rank2_3x4)
            val idx = param("i", i32s)
            val v = param("v", rank1_4)
            val s = op(OpKind.SCATTER, listOf(base, idx, v), rank2_3x4)
            listOf(s)
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(FloatArray(12), floatArrayOf(7f), floatArrayOf(0f, 0f, 0f, 0f)),
            )
        }
    }

    // --- §0.4.41 helpers (rank-1 path) --------------------------------------

    @Test
    fun gradientOfGatherSquaredMatchesChainRule() {
        // f(arr, idx) = arr[idx]^2.  d/d(arr) = 2·arr[idx] · one-hot at idx.
        // Exercises MulRule composed with GatherRule (the adjoint of the inner gather
        // needs to be scaled by 2·arr[idx], then scattered into a one-hot).
        val primal = DxirBuilder.function("sqGather") {
            val arr = param("arr", rank1_4)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), f32s)
            val y = op(OpKind.MUL, listOf(g, g), f32s)
            listOf(y)
        }
        val grad = DxirReverseTransform.apply(primal)
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(10f, 20f, 30f, 40f), floatArrayOf(2f)),
        )
        // At idx=2, arr[2]=30. d/d(arr[2]) = 2·30 = 60; other slots zero.
        val expected = floatArrayOf(0f, 0f, 60f, 0f)
        for (i in expected.indices) {
            assertTrue(
                kotlin.math.abs(out[0][i] - expected[i]) < 1e-3f,
                "slot $i: expected ${expected[i]}, got ${out[0][i]}",
            )
        }
    }

    // --- §0.4.132: rank-3 GATHER / SCATTER / SCATTER_ADD (slice indexing) ---

    private val rank3_2x2x3 = DxirType(F32, listOf(2, 2, 3))
    private val rank2_2x3 = DxirType(F32, listOf(2, 3))

    @Test
    fun rank3GatherReadsSliceFromTensor() {
        // arr is a 2×2×3 tensor:
        //   slice 0 = [[1,2,3], [4,5,6]]
        //   slice 1 = [[7,8,9], [10,11,12]]
        // idx=1 picks the second slice → [7,8,9, 10,11,12] (rank-2 [2,3]).
        val fn = DxirBuilder.function("rank3Gather") {
            val arr = param("arr", rank3_2x2x3)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), rank2_2x3)
            listOf(g)
        }
        val backing = floatArrayOf(
            1f, 2f, 3f, 4f, 5f, 6f,           // slice 0
            7f, 8f, 9f, 10f, 11f, 12f,        // slice 1
        )
        val sliceOne = DxirInterpreter.evalFunction(fn, listOf(backing, floatArrayOf(1f)))
        assertEquals(1, sliceOne.size)
        assertEquals(6, sliceOne[0].size)
        assertEquals(floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f).toList(), sliceOne[0].toList())
        val sliceZero = DxirInterpreter.evalFunction(fn, listOf(backing, floatArrayOf(0f)))
        assertEquals(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f).toList(), sliceZero[0].toList())
    }

    @Test
    fun rank3GatherOutOfBoundsIsFailLoud() {
        val fn = DxirBuilder.function("rank3Gather") {
            val arr = param("arr", rank3_2x2x3)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), rank2_2x3)
            listOf(g)
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(FloatArray(12), floatArrayOf(5f)),
            )
        }
    }

    @Test
    fun rank3ScatterReplacesSliceInTensor() {
        // base 2×2×3 tensor of zeros; SCATTER replaces slice 1 with value.
        val fn = DxirBuilder.function("rank3Scatter") {
            val base = param("base", rank3_2x2x3)
            val idx = param("idx", i32s)
            val v = param("v", rank2_2x3)
            val s = op(OpKind.SCATTER, listOf(base, idx, v), rank3_2x2x3)
            listOf(s)
        }
        val baseBacking = FloatArray(12)
        val valueBacking = floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f)
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(baseBacking, floatArrayOf(1f), valueBacking),
        )
        assertEquals(1, out.size)
        // Slice 0 stays zero; slice 1 = value.
        val expected = floatArrayOf(
            0f, 0f, 0f, 0f, 0f, 0f,           // slice 0
            7f, 8f, 9f, 10f, 11f, 12f,        // slice 1 replaced
        )
        assertEquals(expected.toList(), out[0].toList())
    }

    @Test
    fun rank3ScatterAddAccumulatesSliceInTensor() {
        // base 2×2×3 of ones; SCATTER_ADD adds value into slice 0.
        val fn = DxirBuilder.function("rank3ScatterAdd") {
            val base = param("base", rank3_2x2x3)
            val idx = param("idx", i32s)
            val v = param("v", rank2_2x3)
            val s = op(OpKind.SCATTER_ADD, listOf(base, idx, v), rank3_2x2x3)
            listOf(s)
        }
        val baseBacking = FloatArray(12) { 1f }  // all ones
        val valueBacking = floatArrayOf(0.5f, 1f, 1.5f, 2f, 2.5f, 3f)
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(baseBacking, floatArrayOf(0f), valueBacking),
        )
        // Slice 0 = ones + value; slice 1 stays ones.
        val expected = floatArrayOf(
            1.5f, 2f, 2.5f, 3f, 3.5f, 4f,     // slice 0 += value
            1f, 1f, 1f, 1f, 1f, 1f,           // slice 1 unchanged
        )
        assertEquals(expected.toList(), out[0].toList())
    }

    @Test
    fun rank3GatherFollowedByScatterAddRoundTrips() {
        // Read slice 0 via GATHER, write it back via SCATTER_ADD into a zero base.
        // Result's slice 0 should equal the original input's slice 0.
        val fn = DxirBuilder.function("gatherThenScatterAdd") {
            val arr = param("arr", rank3_2x2x3)
            val idx = param("idx", i32s)
            val g = op(OpKind.GATHER, listOf(arr, idx), rank2_2x3)
            val zeroScalar = const(0f, f32s)
            val zeroBase = op(
                OpKind.BROADCAST,
                listOf(zeroScalar),
                rank3_2x2x3,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
            val sa = op(OpKind.SCATTER_ADD, listOf(zeroBase, idx, g), rank3_2x2x3)
            listOf(sa)
        }
        val backing = floatArrayOf(
            10f, 11f, 12f, 13f, 14f, 15f,
            20f, 21f, 22f, 23f, 24f, 25f,
        )
        val out = DxirInterpreter.evalFunction(fn, listOf(backing, floatArrayOf(0f)))
        // Slice 0 = original slice 0 of `backing`; slice 1 = zeros.
        val expected = floatArrayOf(
            10f, 11f, 12f, 13f, 14f, 15f,
            0f, 0f, 0f, 0f, 0f, 0f,
        )
        assertEquals(expected.toList(), out[0].toList())
    }
}
