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
    fun simplifyReturnsKeepsOpaqueLeafIntact() {
        // f(x) = sum(x). Phase 4 (§0.4.107) lifts SUM as an opaque sentinel, so the
        // pass no longer bails out — it just returns a function whose body cones
        // the SUM op verbatim. Pre-Phase-4 (§0.4.103) this test asserted the bail-out
        // path; post-Phase-4 we assert the opaque-leaf round-trip preserves the op.
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("sum_rank1") {
            val x = param("x", rank1)
            val s = op(OpKind.SUM, listOf(x), f32)
            listOf(s)
        }
        val out = PhiCalculus.simplifyReturns(fn, engine)
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

    // ---------------------- Phase 4: opaque leaves ----------------------

    @Test
    fun simplifyCollapsesMulByOneAroundOpaqueLeaf() {
        // f(x) = SUM(x) * 1. SUM is non-arithmetic; Phase 4 lifts it as an opaque
        // sentinel. Symja sees `_leaf * 1` and simplifies to `_leaf`. Lower clones
        // the original SUM in. Result: a body with only the SUM op (the MUL and
        // const(1) are gone).
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("sum_times_one") {
            val x = param("x", rank1)
            val s = op(OpKind.SUM, listOf(x), f32)
            val one = const(1L, f32)
            val out = op(OpKind.MUL, listOf(s, one), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "MUL by 1 around opaque leaf should collapse to just the leaf")
        assertEquals(OpKind.SUM, ops.single().op)
    }

    @Test
    fun simplifyEliminatesOpaqueLeafUnderMulByZero() {
        // f(x) = SUM(x) * 0. Symja folds `_leaf * 0` to `0`. The SUM is no longer
        // referenced in the simplified output; the leaf is unused, so cloneOpaqueSubtree
        // must NOT be invoked for it. Resulting body has just the const(0) and no SUM.
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("sum_times_zero") {
            val x = param("x", rank1)
            val s = op(OpKind.SUM, listOf(x), f32)
            val zero = const(0L, f32)
            val out = op(OpKind.MUL, listOf(s, zero), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertTrue(
            ops.none { it.op == OpKind.SUM },
            "SUM should be DCE'd because mul-by-zero made the leaf unreferenced; ops=${ops.map { it.op }}",
        )
    }

    @Test
    fun simplifySharesOpaqueLeafAcrossDuplicateUses() {
        // f(x) = SUM(x) + SUM(x), where both ADD operands point to the SAME DxirOp
        // (shared subexpression in the input). Both lift to the same sentinel name
        // (keyed on node.id), so Symja sees `_leaf + _leaf` and folds to `2*_leaf`.
        // The cloned SUM appears once in the output body.
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("dup_sum") {
            val x = param("x", rank1)
            val s = op(OpKind.SUM, listOf(x), f32)
            val out = op(OpKind.ADD, listOf(s, s), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        // Body should contain exactly one SUM (from clone) plus one MUL (the 2*leaf).
        // No ADD remains.
        val sumCount = ops.count { it.op == OpKind.SUM }
        val mulCount = ops.count { it.op == OpKind.MUL }
        val addCount = ops.count { it.op == OpKind.ADD }
        assertEquals(1, sumCount, "exactly one SUM (the cloned leaf); ops=${ops.map { it.op }}")
        assertEquals(1, mulCount, "Symja should produce 2 * leaf; ops=${ops.map { it.op }}")
        assertEquals(0, addCount, "the ADD(leaf, leaf) should fold; ops=${ops.map { it.op }}")
    }

    @Test
    fun simplifyOpaqueLeafSelfReferenceRoundTrip() {
        // f(x) = SUM(x). Pure leaf with no surrounding arithmetic. Pass must clone
        // the SUM into the output unchanged. Pins the basic round-trip path.
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("just_sum") {
            val x = param("x", rank1)
            val s = op(OpKind.SUM, listOf(x), f32)
            listOf(s)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size)
        assertEquals(OpKind.SUM, ops.single().op)
    }

    @Test
    fun simplifyMixesArithmeticAndOpaqueLeaves() {
        // f(x, y) = SUM(x) * 1 + 0 * y. The SUM is opaque; the 0*y branch is pure
        // arithmetic Symja folds away; the MUL-by-1 around SUM collapses. Result
        // should be a body containing only the cloned SUM op.
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("mixed") {
            val x = param("x", rank1)
            val y = param("y", f32)
            val s = op(OpKind.SUM, listOf(x), f32)
            val one = const(1L, f32)
            val zero = const(0L, f32)
            val left = op(OpKind.MUL, listOf(s, one), f32)
            val right = op(OpKind.MUL, listOf(zero, y), f32)
            val out = op(OpKind.ADD, listOf(left, right), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected just the SUM after Symja folds the rest; ops=${ops.map { it.op }}")
        assertEquals(OpKind.SUM, ops.single().op)
    }

    @Test
    fun simplifyHandlesSymmetricallyShapedDistinctLeaves() {
        // f(x, y) = SUM(x) + SUM(y). DIFFERENT SUM nodes (different ids) → different
        // sentinels. Symja can't fold `_leafA + _leafB` (different variables), so the
        // body retains both SUMs and the ADD. This pins the "different leaves stay
        // separate" semantics — important to avoid accidentally over-simplifying when
        // two leaf subtrees happen to have the same shape but different operands.
        val rank1 = DxirType(F32, listOf(4))
        val fn = DxirBuilder.function("two_sums") {
            val x = param("x", rank1)
            val y = param("y", rank1)
            val sx = op(OpKind.SUM, listOf(x), f32)
            val sy = op(OpKind.SUM, listOf(y), f32)
            val out = op(OpKind.ADD, listOf(sx, sy), f32)
            listOf(out)
        }
        val simplified = PhiCalculus.simplifyReturns(fn, engine)
        val ops = simplified.body.filterIsInstance<DxirOp>()
        val sumCount = ops.count { it.op == OpKind.SUM }
        val addCount = ops.count { it.op == OpKind.ADD }
        assertEquals(2, sumCount, "both distinct SUMs survive; ops=${ops.map { it.op }}")
        assertEquals(1, addCount, "the ADD(sumX, sumY) survives; ops=${ops.map { it.op }}")
    }
}
