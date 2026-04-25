package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * §0.4.123 — Phase 1 detector tests for the LAND-composed break-bearing WHILE.
 * Pure-IR tests on hand-built primals; no rewrite or gradient logic exercised yet
 * (those land in subsequent D.3i phases).
 */
class BreakBearingWhileTest {

    private val f32s = DxirType(F32, emptyList())
    private val i32s = DxirType(I32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    private fun findWhile(fn: io.tlaloc.ir.DxirFunction): DxirOp =
        fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.WHILE }

    @Test
    fun detectsLandComposedBreakBearingWhile() {
        // Build a WHILE whose cond region is `LAND(STEP(SUB(n, counter)), NOT(STEP(args[0])))`.
        // The natural cond is `STEP(SUB(n, counter))`; the break cond is `STEP(args[0])`
        // (i.e., break when args[0] > 0 — arbitrary but structurally well-formed).
        val fn = DxirBuilder.function("breakLoop") {
            val x = param("x", f32s)
            val n = const(5, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val brkInner = op(OpKind.STEP, listOf(args[0]), boolS)
                    val notBrk = op(OpKind.NOT, listOf(brkInner), boolS)
                    val land = op(OpKind.LAND, listOf(origCond, notBrk), boolS)
                    yields(land)
                },
                body = { args ->
                    val two = const(2f, f32s)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32s)
                    val one = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val whileOp = findWhile(fn)
        val pattern = BreakBearingWhile.detect(whileOp)
        assertNotNull(pattern, "detector should match LAND-composed break-bearing WHILE")
        // origCond is the STEP(SUB(n, counter)) node; breakCond is the STEP(args[0]) node.
        val origCondOp = pattern.origCond as DxirOp
        val breakCondOp = pattern.breakCond as DxirOp
        assertEquals(OpKind.STEP, origCondOp.op, "origCond should be STEP")
        assertEquals(OpKind.STEP, breakCondOp.op, "breakCond should be STEP (operand of NOT)")
        assertEquals(whileOp, pattern.whileOp, "pattern carries the original WHILE op")
    }

    @Test
    fun rejectsVanillaWhileWithoutLand() {
        // Standard simple-loop cond (no LAND, no NOT). Detector must return null.
        val fn = DxirBuilder.function("simpleLoop") {
            val x = param("x", f32s)
            val n = const(5, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val pred = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
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
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNull(pattern, "vanilla WHILE without LAND should not match")
    }

    @Test
    fun rejectsLandWhereSecondOperandIsNotNot() {
        // Cond = `LAND(origCond, anotherStep)` — second operand isn't NOT, so the
        // shape isn't break-bearing per the FIR-side hoist contract.
        val fn = DxirBuilder.function("badLand") {
            val x = param("x", f32s)
            val n = const(5, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val anotherStep = op(OpKind.STEP, listOf(args[0]), boolS) // not NOT-wrapped
                    val land = op(OpKind.LAND, listOf(origCond, anotherStep), boolS)
                    yields(land)
                },
                body = { args ->
                    val two = const(2f, f32s)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32s)
                    val one = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNull(pattern, "LAND with non-NOT second operand should not match")
    }

    @Test
    fun rejectsNonWhileOp() {
        // A non-WHILE op (here, an IF) shouldn't match the detector.
        val fn = DxirBuilder.function("ifNotWhile") {
            val x = param("x", f32s)
            val pred = op(OpKind.STEP, listOf(x), boolS)
            val ifResult = ifOp(
                cond = pred,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(x) },
            )
            listOf(ifResult)
        }
        val ifOpNode = fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.IF }
        val pattern = BreakBearingWhile.detect(ifOpNode)
        assertNull(pattern, "IF op should not match (only WHILE is in scope)")
    }
}
