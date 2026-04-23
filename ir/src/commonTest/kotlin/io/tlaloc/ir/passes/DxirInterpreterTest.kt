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
}
