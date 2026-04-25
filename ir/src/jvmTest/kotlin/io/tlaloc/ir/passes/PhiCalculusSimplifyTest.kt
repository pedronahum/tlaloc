package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * **D.1i Phase 1+2** — tests for `PhiCalculus.simplifyReturns`. The pass lifts
 * each return expression to a Symja `SymExpr`, runs `Simplify`, and lowers back.
 * Phase 1's surface covers integer-valued `DxirConst` + arithmetic ops on scalars
 * (the first four tests). Phase 2 (§0.4.104) widens the lift to fractional Float
 * payloads via `realLiteral`, fixing a silent truncation bug — `0.5f` previously
 * lifted as `rational(0)` because `0.5f.toLong() == 0L`. The Phase-2 tests below
 * exercise that path.
 *
 * Phase 3 (future session) will wire the pass into `TlalocIrGenerationExtension`
 * behind an opt-in property + benchmark perf delta. Phase 4 will add opaque-leaf
 * handling for non-arithmetic gradient ops (SUM, MEAN, MATMUL, GATHER) reusing
 * §0.4.52's `symOpaqueLeaves` mechanism.
 */
class PhiCalculusSimplifyTest {

    private val engine = SymjaEngine()
    private val f32 = DxirType(F32, emptyList())

    @Test
    fun simplifyReturnsCollapsesMulByOne() {
        // f(x) = 1 * x. Symja's Simplify recognises Times[1, x] = x.
        val fn = DxirBuilder.function("mul_by_one") {
            val x = param("x", f32)
            val one = const(1L, f32)
            val out = op(OpKind.MUL, listOf(one, x), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        // Pre-simplify: 1 op (MUL). Post: 0 ops — the result is just the param.
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertEquals(0, ops.size, "simplify should remove MUL by 1; body has ${ops.size} ops")
        // Numerical equivalence: f(5) = 5 either way.
        val out = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(5f)))
        assertEquals(5f, out.single()[0])
    }

    @Test
    fun simplifyReturnsFoldsAddIdentityAndDuplicateMuls() {
        // f(x) = x*1 + 1*x. Each MUL collapses to x; ADD becomes 2*x.
        val fn = DxirBuilder.function("dup_x") {
            val x = param("x", f32)
            val one = const(1L, f32)
            val a = op(OpKind.MUL, listOf(x, one), f32)
            val b = op(OpKind.MUL, listOf(one, x), f32)
            val sum = op(OpKind.ADD, listOf(a, b), f32)
            listOf(sum)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertTrue(ops.size <= 2, "expected ≤ 2 ops post-simplify (e.g., MUL by 2); got ${ops.size}")
        // Numerical: f(5) = 5 + 5 = 10.
        val out = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(5f)))
        assertEquals(10f, out.single()[0])
    }

    @Test
    fun simplifyReturnsBailsOutWhenLiftFails() {
        // f(x) = sum(x) — SUM is not in liftNode's supported set. The pass must
        // return the original function unchanged rather than throwing.
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("sum_rank1") {
            val x = param("x", rank1)
            val s = op(OpKind.SUM, listOf(x), f32)
            listOf(s)
        }
        val out = PhiCalculus.simplifyReturns(fn, engine)
        // Bailed: same body shape as the input (one SUM op).
        val ops = out.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size)
        assertEquals(OpKind.SUM, ops.single().op)
    }

    // ---------------------- Phase 2: fractional consts ----------------------

    @Test
    fun simplifyDoesNotTruncateFractionalFloatConst() {
        // f(x) = 0.5 * x. Phase 1's liftNode used `n.toLong()` which truncated
        // 0.5f to 0L, so the simplified body collapsed to 0 — a silent
        // correctness bug. Phase 2 lifts via `realLiteral`, preserving 0.5.
        val fn = DxirBuilder.function("half_x") {
            val x = param("x", f32)
            val half = const(0.5f, f32)
            val out = op(OpKind.MUL, listOf(half, x), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        // Numerical: f(4) = 2.0 (would have been 0.0 under Phase 1's truncation).
        val out = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(4f)))
        assertEquals(2.0f, out.single()[0])
    }

    @Test
    fun simplifyCollapsesMulByOneEvenWhenLiteralIsFloat() {
        // f(x) = 1.0f * x. The Float 1.0f round-trips through Long (1.0 == 1L.toDouble()),
        // so liftNumber promotes it to integer rational form — Symja's Times[1, x] → x
        // rule still fires. Verifies Phase 2 doesn't regress Phase 1's behavior.
        val fn = DxirBuilder.function("mul_by_one_float") {
            val x = param("x", f32)
            val one = const(1.0f, f32)
            val out = op(OpKind.MUL, listOf(one, x), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertEquals(0, ops.size, "1.0f * x should still collapse to x; got ${ops.size} ops")
        val out = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(7f)))
        assertEquals(7f, out.single()[0])
    }

    @Test
    fun simplifyFoldsFractionalConstantArithmetic() {
        // f(x) = (0.5 + 0.5) * x. Symja's evaluator folds Plus[0.5, 0.5] → 1.0;
        // Times[1.0, x] stays as 1.*x (real-domain Simplify is conservative).
        // The point: numerical equivalence holds and the body is no larger than
        // the input. f(6) = 6.
        val fn = DxirBuilder.function("half_plus_half_x") {
            val x = param("x", f32)
            val a = const(0.5f, f32)
            val b = const(0.5f, f32)
            val sum = op(OpKind.ADD, listOf(a, b), f32)
            val out = op(OpKind.MUL, listOf(sum, x), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val out = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(6f)))
        assertEquals(6f, out.single()[0])
        // Body should be at most the input size: 2 consts + 2 ops = 4 nodes.
        assertTrue(simplified.body.size <= 4, "body grew unexpectedly: ${simplified.body.size}")
    }

    @Test
    fun simplifyHandlesNegativeFractionalConst() {
        // f(x) = -0.25 + x. Symja keeps as x - 0.25 (or x + (-0.25)); numerical agreement
        // is what we verify. Pre-Phase-2: -0.25f.toLong() = 0L, body collapsed to 0 + x = x.
        val fn = DxirBuilder.function("neg_quarter_plus_x") {
            val x = param("x", f32)
            val negQuarter = const(-0.25f, f32)
            val out = op(OpKind.ADD, listOf(negQuarter, x), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val out = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(1f)))
        assertEquals(0.75f, out.single()[0])
    }

    // ---------------------- Phase 1: integer constants ----------------------

    @Test
    fun simplifyReturnsHandlesMultiReturnFunctions() {
        // valueAndGrad-style: returns (value, grad). f(x) = (x*1, 0+x).
        // After simplify: returns (x, x).
        val fn = DxirBuilder.function("mr") {
            val x = param("x", f32)
            val one = const(1L, f32)
            val zero = const(0L, f32)
            val value = op(OpKind.MUL, listOf(x, one), f32)
            val grad = op(OpKind.ADD, listOf(zero, x), f32)
            listOf(value, grad)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertEquals(0, ops.size, "both returns simplify to the param directly")
        val out = DxirInterpreter.evalFunction(simplified, listOf(floatArrayOf(7f)))
        assertEquals(2, out.size)
        assertEquals(7f, out[0][0])
        assertEquals(7f, out[1][0])
    }
}
