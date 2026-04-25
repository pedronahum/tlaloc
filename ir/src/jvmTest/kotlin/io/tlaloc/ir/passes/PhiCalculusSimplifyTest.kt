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
 * **D.1i Phase 1** — first-cut tests for `PhiCalculus.simplifyReturns`. The pass
 * lifts each return expression to a Symja `SymExpr`, runs `Simplify`, and lowers
 * back. Phase 1's lift surface (per `SymjaEngine.liftNode`) covers integer-valued
 * `DxirConst` + arithmetic ops on scalars; these tests use that subset.
 *
 * Phase 2 (future session) will widen the lift to handle fractional constants
 * (via `realLiteral` instead of integer `rational`) and wire the pass into
 * `TlalocIrGenerationExtension` behind an opt-in property. Phase 3+ will add
 * opaque-leaf handling for non-arithmetic gradient ops.
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
