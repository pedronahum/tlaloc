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
}
