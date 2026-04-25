package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
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
 * Stage B.0a interpreter tests for `OpKind.IF` and `OpKind.WHILE`. The substrate that
 * Stages B.1 / B.2 / B.3 use to validate themselves: every φ-calculus rewrite landing
 * in a future session must produce a primal whose [DxirInterpreter.evalFunction] result
 * agrees numerically with the pre-rewrite version. These tests pin the interpreter's
 * direct evaluation of hand-built IF / WHILE primals, before any φ-calculus lives.
 */
class DxirInterpreterTest {
    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    // ---- IF ----------------------------------------------------------------

    @Test
    fun ifThenBranchEvaluatesWhenPredicateTrue() {
        // f(x) = if (x > 0) -x else x
        val fn = DxirBuilder.function("absNeg") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifop = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val nx = op(OpKind.NEG, listOf(x), f32s)
                    yields(nx)
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifop)
        }
        // x = 3 → STEP(3) = 1 → take then → -3
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(3f)))
        assertEquals(1, out.size)
        assertEquals(-3f, out[0][0])
    }

    @Test
    fun ifElseBranchEvaluatesWhenPredicateFalse() {
        val fn = DxirBuilder.function("absNeg") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifop = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region {
                    val nx = op(OpKind.NEG, listOf(x), f32s)
                    yields(nx)
                },
                elseRegion = region { yields(x) },
            )
            listOf(ifop)
        }
        // x = -3 → STEP(-3) = 0 → take else → -3 (the input itself, unchanged)
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(-3f)))
        assertEquals(-3f, out[0][0])
        // x = 0 → STEP(0) = 0 (XLA convention: H(0) = 0) → take else → 0
        val outZero = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(0f)))
        assertEquals(0f, outZero[0][0])
    }

    @Test
    fun nestedIfEvaluatesCorrectly() {
        // f(x) = if (x > 0) (if (x > 0) x*x else 0) else x
        // Outer pred = inner pred for simplicity; result at x=3 = 9, at x=-3 = -3.
        val fn = DxirBuilder.function("nested") {
            val x = param("x", f32s)
            val outerPred = op(OpKind.STEP, listOf(x), boolS)
            val outer = ifOp(
                cond = outerPred,
                types = listOf(f32s),
                thenRegion = region {
                    val innerPred = op(OpKind.STEP, listOf(x), boolS)
                    val zero = const(0f, f32s)
                    val inner = ifOp(
                        cond = innerPred,
                        types = listOf(f32s),
                        thenRegion = region {
                            val sq = op(OpKind.MUL, listOf(x, x), f32s)
                            yields(sq)
                        },
                        elseRegion = region { yields(zero) },
                    )
                    yields(inner)
                },
                elseRegion = region { yields(x) },
            )
            listOf(outer)
        }
        val outPos = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(3f)))
        assertEquals(9f, outPos[0][0])
        val outNeg = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(-3f)))
        assertEquals(-3f, outNeg[0][0])
    }

    // ---- WHILE -------------------------------------------------------------

    /**
     * Helper: build a WHILE that doubles `x` exactly `n` times via an integer counter.
     * Models C5's canonical test (`f^[n](x) = 2^n · x`) but as a hand-built primal,
     * not via Stage B's φ-calculus.
     */
    private fun iterateNTimes(): io.tlaloc.ir.DxirFunction = DxirBuilder.function("doubleN") {
        val x = param("x", f32s)
        val n = param("n", i32s)
        val zero = const(0, i32s)
        val w = whileOp(
            inits = listOf(x, zero),
            cond = { args ->
                // args[1] is the counter `i`; predicate: i < n  →  STEP(n - i)
                val diff = op(OpKind.SUB, listOf(n, args[1]), i32s)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val two = const(2f, f32s)
                val newX = op(OpKind.MUL, listOf(args[0], two), f32s)
                val one = const(1, i32s)
                val newI = op(OpKind.ADD, listOf(args[1], one), i32s)
                yields(newX, newI)
            },
        )
        // Return the loop-carried x (result 0).
        listOf(w.result(0))
    }

    @Test
    fun whileSimpleIterationProducesExpectedFinalValue() {
        val fn = iterateNTimes()
        // x=3, n=5 → final x = 3 · 2^5 = 96
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(3f), floatArrayOf(5f)))
        assertEquals(96f, out[0][0])
    }

    @Test
    fun whileEmptyTripCountReturnsInitialValuesUnchanged() {
        val fn = iterateNTimes()
        // x=7, n=0 → cond evaluates to false on first iter → return x = 7
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(7f), floatArrayOf(0f)))
        assertEquals(7f, out[0][0])
    }

    @Test
    fun whileMultiLoopCarriedExposesAllResultsViaDxirOpResult() {
        // Two independent counters: x doubles each iter; y triples each iter. Loop runs
        // for `n` iterations driven by a third counter. Returns (x_final, y_final).
        val fn = DxirBuilder.function("twoCarried") {
            val xInit = param("x", f32s)
            val yInit = param("y", f32s)
            val n = param("n", i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(xInit, yInit, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(n, args[2]), i32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val two = const(2f, f32s)
                    val three = const(3f, f32s)
                    val one = const(1, i32s)
                    val nx = op(OpKind.MUL, listOf(args[0], two), f32s)
                    val ny = op(OpKind.MUL, listOf(args[1], three), f32s)
                    val ni = op(OpKind.ADD, listOf(args[2], one), i32s)
                    yields(nx, ny, ni)
                },
            )
            listOf(w.result(0), w.result(1))
        }
        // x=1, y=1, n=4 → x_final = 16, y_final = 81
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(floatArrayOf(1f), floatArrayOf(1f), floatArrayOf(4f)),
        )
        assertEquals(2, out.size)
        assertEquals(16f, out[0][0])
        assertEquals(81f, out[1][0])
    }

    @Test
    fun whileWithLandBreakHoistTerminatesOnBreakCondition() {
        // §0.4.50 Gap 3 — mirrors what the FIR break-hoist path emits. The body doubles
        // x and increments k; the cond is `LAND(k < 10, NOT(k > 2))` which terminates at
        // k = 3 even though k < 10. Expected: x doubles exactly 3 times → 2^3 = 8 · init.
        val fn = DxirBuilder.function("doubleUntilBreak") {
            val x = param("x", f32s)
            val cap = param("cap", i32s)     // outer cap, e.g. 10
            val brk = param("brk", i32s)     // break threshold, e.g. 2 (break when k > 2)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val capMinusK = op(OpKind.SUB, listOf(cap, args[1]), i32s)
                    val primary = op(OpKind.STEP, listOf(capMinusK), boolS)
                    val kMinusBrk = op(OpKind.SUB, listOf(args[1], brk), i32s)
                    val breakPred = op(OpKind.STEP, listOf(kMinusBrk), boolS)
                    val notBreak = op(OpKind.NOT, listOf(breakPred), boolS)
                    val pred = op(OpKind.LAND, listOf(primary, notBreak), boolS)
                    yields(pred)
                },
                body = { args ->
                    val two = const(2f, f32s)
                    val one = const(1, i32s)
                    val nx = op(OpKind.MUL, listOf(args[0], two), f32s)
                    val nk = op(OpKind.ADD, listOf(args[1], one), i32s)
                    yields(nx, nk)
                },
            )
            listOf(w.result(0))
        }
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(floatArrayOf(1f), floatArrayOf(10f), floatArrayOf(2f)),
        )
        assertEquals(8f, out[0][0])
    }

    // §0.4.84 — axis-aware SUM in the interpreter's bridge evalOp. Supports both
    // the pre-§0.4.84 all-dims collapse (no attrs, matches Tracer.sum()) and the
    // new partial-SUM path (`reduction_dims` attr) used by BroadcastRule's
    // reverse on non-scalar input.

    @Test
    fun sumWithoutReductionDimsCollapsesToScalar() {
        // Rank-2 input [[1, 2], [3, 4]], no attrs → sum = 10, scalar output.
        val f32_2x2 = DxirType(F32, listOf(2, 2))
        val fn = DxirBuilder.function("sum_all") {
            val x = param("x", f32_2x2)
            val s = op(OpKind.SUM, listOf(x), f32s)
            listOf(s)
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f, 4f)))
        assertEquals(1, out.size)
        assertEquals(1, out[0].size)
        assertEquals(10f, out[0][0])
    }

    @Test
    fun sumWithReductionDimsRank2Axis0ProducesRank1() {
        // Rank-2 [[1, 2], [3, 4]] summed over dim 0 → [4, 6] (rank-1, size-2).
        val f32_2x2 = DxirType(F32, listOf(2, 2))
        val f32_2 = DxirType(F32, listOf(2))
        val fn = DxirBuilder.function("sum_axis0") {
            val x = param("x", f32_2x2)
            val s = op(
                OpKind.SUM,
                listOf(x),
                f32_2,
                attrs = mapOf("reduction_dims" to listOf(0)),
            )
            listOf(s)
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f, 4f)))
        assertEquals(1, out.size)
        assertEquals(2, out[0].size)
        assertEquals(4f, out[0][0])  // 1 + 3
        assertEquals(6f, out[0][1])  // 2 + 4
    }

    @Test
    fun sumWithReductionDimsRank2Axis1ProducesRank1() {
        // Rank-2 [[1, 2], [3, 4]] summed over dim 1 → [3, 7] (rank-1, size-2).
        val f32_2x2 = DxirType(F32, listOf(2, 2))
        val f32_2 = DxirType(F32, listOf(2))
        val fn = DxirBuilder.function("sum_axis1") {
            val x = param("x", f32_2x2)
            val s = op(
                OpKind.SUM,
                listOf(x),
                f32_2,
                attrs = mapOf("reduction_dims" to listOf(1)),
            )
            listOf(s)
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f, 2f, 3f, 4f)))
        assertEquals(3f, out[0][0])  // 1 + 2
        assertEquals(7f, out[0][1])  // 3 + 4
    }

    @Test
    fun sumWithReductionDimsRank3DropsTwoAxes() {
        // Rank-3 [2, 2, 2] summed over dims [0, 2] → rank-1 [2].
        // Input: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        //   along axis 0: [[6, 8], [10, 12]]
        //   then along axis 2: [14, 22]
        val f32_2x2x2 = DxirType(F32, listOf(2, 2, 2))
        val f32_2 = DxirType(F32, listOf(2))
        val fn = DxirBuilder.function("sum_0_2") {
            val x = param("x", f32_2x2x2)
            val s = op(
                OpKind.SUM,
                listOf(x),
                f32_2,
                attrs = mapOf("reduction_dims" to listOf(0, 2)),
            )
            listOf(s)
        }
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)),
        )
        assertEquals(14f, out[0][0])
        assertEquals(22f, out[0][1])
    }

    @Test
    fun whileExceedingIterationCapErrorsLoudly() {
        // Build a WHILE with predicate always-true and trivial body (carries an unchanged
        // float). Should hit the cap and throw — proves the safety net works.
        val fn = DxirBuilder.function("infinite") {
            val xInit = param("x", f32s)
            val w = whileOp(
                inits = listOf(xInit),
                cond = { _ ->
                    val one = const(1f, f32s)
                    val pred = op(OpKind.STEP, listOf(one), boolS)
                    yields(pred)
                },
                body = { args -> yields(args[0]) }, // back-edge = arg → unchanged
            )
            listOf(w)
        }
        val ex = assertFailsWith<IllegalStateException> {
            DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f)))
        }
        assertTrue(
            ex.message?.contains("exceeded iteration cap") == true,
            "expected cap error message; got: ${ex.message}",
        )
    }

    // ---- §0.4.135: batched MATMUL ------------------------------------------

    @Test
    fun batchedRank3MatmulComputesPerBatchSlices() {
        // a: (B=2, M=2, K=3), b: (B=2, K=3, N=2). Per-batch matmul should produce
        // a (2, 2, 2) result. Pin both batches to known slice values.
        val aType = DxirType(F32, listOf(2, 2, 3))
        val bType = DxirType(F32, listOf(2, 3, 2))
        val outType = DxirType(F32, listOf(2, 2, 2))
        val fn = DxirBuilder.function("bmm") {
            val a = param("a", aType)
            val b = param("b", bType)
            val y = op(OpKind.MATMUL, listOf(a, b), outType)
            listOf(y)
        }
        // Batch 0: a0 = [[1,2,3],[4,5,6]], b0 = [[1,0],[0,1],[1,1]] →
        //   row 0 = [1+0+3, 0+2+3] = [4, 5]; row 1 = [4+0+6, 0+5+6] = [10, 11].
        // Batch 1: a1 = [[1,1,1],[2,2,2]], b1 = [[1,1],[1,1],[1,1]] →
        //   row 0 = [3, 3]; row 1 = [6, 6].
        val aBacking = floatArrayOf(
            // batch 0
            1f, 2f, 3f, 4f, 5f, 6f,
            // batch 1
            1f, 1f, 1f, 2f, 2f, 2f,
        )
        val bBacking = floatArrayOf(
            // batch 0
            1f, 0f, 0f, 1f, 1f, 1f,
            // batch 1
            1f, 1f, 1f, 1f, 1f, 1f,
        )
        val out = DxirInterpreter.evalFunction(fn, listOf(aBacking, bBacking))
        assertEquals(1, out.size)
        assertEquals(8, out[0].size)
        val expected = floatArrayOf(
            // batch 0 result (2x2 row-major): [[4,5],[10,11]]
            4f, 5f, 10f, 11f,
            // batch 1 result: [[3,3],[6,6]]
            3f, 3f, 6f, 6f,
        )
        assertEquals(expected.toList(), out[0].toList())
    }

    @Test
    fun batchedMatmulAgreesWithRank2OnBatchSizeOne() {
        // For B=1, the rank-3 batched form must agree with the rank-2 reference.
        // Both produce the same per-batch slice (since there's only one batch).
        val rank3A = DxirType(F32, listOf(1, 2, 3))
        val rank3B = DxirType(F32, listOf(1, 3, 2))
        val rank3Out = DxirType(F32, listOf(1, 2, 2))
        val fnRank3 = DxirBuilder.function("bmm1") {
            val a = param("a", rank3A)
            val b = param("b", rank3B)
            val y = op(OpKind.MATMUL, listOf(a, b), rank3Out)
            listOf(y)
        }
        val rank2A = DxirType(F32, listOf(2, 3))
        val rank2B = DxirType(F32, listOf(3, 2))
        val rank2Out = DxirType(F32, listOf(2, 2))
        val fnRank2 = DxirBuilder.function("mm") {
            val a = param("a", rank2A)
            val b = param("b", rank2B)
            val y = op(OpKind.MATMUL, listOf(a, b), rank2Out)
            listOf(y)
        }
        val aBacking = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val bBacking = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f)
        val outRank3 = DxirInterpreter.evalFunction(fnRank3, listOf(aBacking, bBacking))
        val outRank2 = DxirInterpreter.evalFunction(fnRank2, listOf(aBacking, bBacking))
        assertEquals(outRank2[0].toList(), outRank3[0].toList())
    }

    @Test
    fun batchedMatmulRejectsInnerDimMismatch() {
        // (B, M, K) x (B, K', N) with K != K' must fail loud.
        val aType = DxirType(F32, listOf(2, 2, 3))
        val bType = DxirType(F32, listOf(2, 4, 2))   // K'=4, but lhs.K=3
        val outType = DxirType(F32, listOf(2, 2, 2))
        val fn = DxirBuilder.function("bmmBad") {
            val a = param("a", aType)
            val b = param("b", bType)
            val y = op(OpKind.MATMUL, listOf(a, b), outType)
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(FloatArray(2 * 2 * 3), FloatArray(2 * 4 * 2)),
            )
        }
    }

    // ---- §0.4.136: rank-N TRANSPOSE -----------------------------------------

    @Test
    fun transposeRank3LastTwoAxesSwap() {
        // Rank-3 (B=2, M=2, K=3) with permutation [0, 2, 1] → output (B, K, M) = (2, 3, 2).
        // out[b, j, i] = in[b, i, j]. This is the canonical batched-transpose used by
        // a future batched MatmulRule.
        val inType = DxirType(F32, listOf(2, 2, 3))
        val outType = DxirType(F32, listOf(2, 3, 2))
        val fn = DxirBuilder.function("t3") {
            val x = param("x", inType)
            val t = op(
                OpKind.TRANSPOSE,
                listOf(x),
                outType,
                attrs = mapOf("permutation" to listOf(0, 2, 1)),
            )
            listOf(t)
        }
        // Batch 0: [[1, 2, 3], [4, 5, 6]]; expected transpose: [[1, 4], [2, 5], [3, 6]]
        // Batch 1: [[7, 8, 9], [10, 11, 12]]; expected transpose: [[7, 10], [8, 11], [9, 12]]
        val backing = floatArrayOf(
            1f, 2f, 3f, 4f, 5f, 6f,
            7f, 8f, 9f, 10f, 11f, 12f,
        )
        val out = DxirInterpreter.evalFunction(fn, listOf(backing))
        assertEquals(
            floatArrayOf(
                1f, 4f, 2f, 5f, 3f, 6f,
                7f, 10f, 8f, 11f, 9f, 12f,
            ).toList(),
            out[0].toList(),
        )
    }

    @Test
    fun transposeRank3GeneralPermutation() {
        // Rank-3 (A=2, B=3, C=2) with permutation [2, 0, 1] → output (C, A, B) = (2, 2, 3).
        // out[c, a, b] = in[a, b, c]. Tests the general (non-axis-swap) case.
        val inType = DxirType(F32, listOf(2, 3, 2))
        val outType = DxirType(F32, listOf(2, 2, 3))
        val fn = DxirBuilder.function("t3perm") {
            val x = param("x", inType)
            val t = op(
                OpKind.TRANSPOSE,
                listOf(x),
                outType,
                attrs = mapOf("permutation" to listOf(2, 0, 1)),
            )
            listOf(t)
        }
        // in[a, b, c] indexed at row-major offset a*6 + b*2 + c.
        // Use distinguishable values so we can pin per-element correctness.
        val backing = FloatArray(12) { (it + 1).toFloat() }
        // Manually compute expected: out[c, a, b] = in[a, b, c].
        val expected = FloatArray(12)
        for (c in 0 until 2) {
            for (a in 0 until 2) {
                for (b in 0 until 3) {
                    val outOff = c * 6 + a * 3 + b
                    val inOff = a * 6 + b * 2 + c
                    expected[outOff] = backing[inOff]
                }
            }
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(backing))
        assertEquals(expected.toList(), out[0].toList())
    }

    @Test
    fun transposeRank4LastTwoAxesSwap() {
        // Rank-4 (D0=2, D1=2, M=2, K=2) with permutation [0, 1, 3, 2] →
        // output (D0, D1, K, M). Two batch axes preserved; last two swapped.
        // out[d0, d1, j, i] = in[d0, d1, i, j].
        val inType = DxirType(F32, listOf(2, 2, 2, 2))
        val outType = DxirType(F32, listOf(2, 2, 2, 2))
        val fn = DxirBuilder.function("t4") {
            val x = param("x", inType)
            val t = op(
                OpKind.TRANSPOSE,
                listOf(x),
                outType,
                attrs = mapOf("permutation" to listOf(0, 1, 3, 2)),
            )
            listOf(t)
        }
        val backing = FloatArray(16) { (it + 1).toFloat() }
        val expected = FloatArray(16)
        for (d0 in 0 until 2) {
            for (d1 in 0 until 2) {
                for (j in 0 until 2) {
                    for (i in 0 until 2) {
                        val outOff = d0 * 8 + d1 * 4 + j * 2 + i
                        val inOff = d0 * 8 + d1 * 4 + i * 2 + j
                        expected[outOff] = backing[inOff]
                    }
                }
            }
        }
        val out = DxirInterpreter.evalFunction(fn, listOf(backing))
        assertEquals(expected.toList(), out[0].toList())
    }

    @Test
    fun transposeRank2StillWorksAfterGeneralisation() {
        // Sanity: the original rank-2 [1, 0] swap path still produces the same result
        // after the rank-N rewrite. Pre-§0.4.136 this was a hard-coded fast path.
        val inType = DxirType(F32, listOf(2, 3))
        val outType = DxirType(F32, listOf(3, 2))
        val fn = DxirBuilder.function("t2") {
            val x = param("x", inType)
            val t = op(
                OpKind.TRANSPOSE,
                listOf(x),
                outType,
                attrs = mapOf("permutation" to listOf(1, 0)),
            )
            listOf(t)
        }
        // [[1, 2, 3], [4, 5, 6]] transposed → [[1, 4], [2, 5], [3, 6]].
        val out = DxirInterpreter.evalFunction(
            fn,
            listOf(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)),
        )
        assertEquals(
            floatArrayOf(1f, 4f, 2f, 5f, 3f, 6f).toList(),
            out[0].toList(),
        )
    }

    @Test
    fun transposeRejectsInvalidPermutation() {
        // Permutation must be a valid 0..N-1 permutation. Length mismatch fails.
        val inType = DxirType(F32, listOf(2, 3, 4))
        val outType = DxirType(F32, listOf(2, 3, 4))
        val fn = DxirBuilder.function("tBad") {
            val x = param("x", inType)
            val t = op(
                OpKind.TRANSPOSE,
                listOf(x),
                outType,
                attrs = mapOf("permutation" to listOf(0, 1)),  // length 2, not 3
            )
            listOf(t)
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(fn, listOf(FloatArray(24)))
        }
    }

    @Test
    fun batchedMatmulRejectsBatchAxisMismatch() {
        // (B, M, K) x (B', K, N) with B != B' must fail loud.
        val aType = DxirType(F32, listOf(2, 2, 3))
        val bType = DxirType(F32, listOf(3, 3, 2))   // B'=3, lhs.B=2
        val outType = DxirType(F32, listOf(2, 2, 2))
        val fn = DxirBuilder.function("bmmBatchBad") {
            val a = param("a", aType)
            val b = param("b", bType)
            val y = op(OpKind.MATMUL, listOf(a, b), outType)
            listOf(y)
        }
        assertFailsWith<IllegalArgumentException> {
            DxirInterpreter.evalFunction(
                fn,
                listOf(FloatArray(2 * 2 * 3), FloatArray(3 * 3 * 2)),
            )
        }
    }
}
