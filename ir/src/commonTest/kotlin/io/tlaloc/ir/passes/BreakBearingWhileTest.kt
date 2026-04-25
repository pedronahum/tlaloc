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

    // --- D.3i Phase 3a (§0.4.126) breakCond classification tests ---

    @Test
    fun classifyConstantTrueBreakCond() {
        // breakCond is `const(true)` directly. Classifier returns Constant(true)
        // with breakIteration = 0 — the loop breaks on iteration 0.
        val fn = DxirBuilder.function("constTrueBreak") {
            val x = param("x", f32s)
            val n = const(7, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(const(true, boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        val klass = BreakBearingWhile.classifyBreakCond(pattern)
        val constant = klass as BreakBearingWhile.BreakCondClass.Constant
        assertEquals(true, constant.alwaysBreaks)
        assertEquals(0, constant.breakIteration)
    }

    @Test
    fun classifyConstantFalseBreakCond() {
        // breakCond is `const(false)` directly. Classifier returns Constant(false)
        // with breakIteration = null — the loop never breaks.
        val fn = DxirBuilder.function("constFalseBreak") {
            val x = param("x", f32s)
            val n = const(7, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(const(false, boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        val klass = BreakBearingWhile.classifyBreakCond(pattern)
        val constant = klass as BreakBearingWhile.BreakCondClass.Constant
        assertEquals(false, constant.alwaysBreaks)
        assertNull(constant.breakIteration)
    }

    @Test
    fun classifyLoopInvariantBreakCondViaParam() {
        // breakCond depends only on a function param (loop-invariant). Not a bare
        // const so doesn't fold to Constant — classifier returns LoopInvariant.
        val fn = DxirBuilder.function("loopInvBreak") {
            val x = param("x", f32s)
            val flag = param("flag", boolS)  // loop-invariant predicate
            val n = const(7, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(flag), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        val klass = BreakBearingWhile.classifyBreakCond(pattern)
        assertEquals(BreakBearingWhile.BreakCondClass.LoopInvariant, klass)
    }

    @Test
    fun classifyLoopInvariantBreakCondViaOpOnConst() {
        // breakCond is an op tree but reaches no cond block-arg — only constants.
        // Not a bare DxirConst at the top, so classified as LoopInvariant
        // (constant-folding the whole tree is out of Phase 3a's scope).
        val fn = DxirBuilder.function("constTreeBreak") {
            val x = param("x", f32s)
            val n = const(7, i32s)
            val zero = const(0, i32s)
            val threshold = const(5, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    // STEP(threshold) — depends only on a region-external const.
                    val brkInner = op(OpKind.STEP, listOf(threshold), boolS)
                    val notBrk = op(OpKind.NOT, listOf(brkInner), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        val klass = BreakBearingWhile.classifyBreakCond(pattern)
        assertEquals(BreakBearingWhile.BreakCondClass.LoopInvariant, klass)
    }

    @Test
    fun classifyCounterOnlyBreakCond() {
        // breakCond depends only on the counter block-arg (args[1]) plus a constant.
        // Phase 3b/Symja can solve for the break iteration symbolically.
        val fn = DxirBuilder.function("counterOnlyBreak") {
            val x = param("x", f32s)
            val n = const(10, i32s)
            val zero = const(0, i32s)
            val cap = const(3, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    // breakCond = STEP(SUB(args[1], cap)) — true when counter > cap.
                    val brkInner = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(args[1], cap), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(brkInner), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        val klass = BreakBearingWhile.classifyBreakCond(pattern)
        assertEquals(BreakBearingWhile.BreakCondClass.CounterOnly, klass)
    }

    @Test
    fun classifyCarriedDependentBreakCond() {
        // breakCond reaches args[0] (the f32 carried, NOT the counter at args[1]).
        // Closure must keep per-iteration evaluation of the predicate.
        val fn = DxirBuilder.function("carriedBreak") {
            val x = param("x", f32s)
            val n = const(7, i32s)
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
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        val klass = BreakBearingWhile.classifyBreakCond(pattern)
        assertEquals(BreakBearingWhile.BreakCondClass.CarriedDependent, klass)
    }

    @Test
    fun classifyMixedCounterAndCarriedAsCarriedDependent() {
        // breakCond reaches BOTH the counter (args[1]) and a non-counter carried
        // (args[0]). Any non-counter dependency forces CarriedDependent — the
        // classifier collapses the case rather than splitting "Mixed" out.
        val fn = DxirBuilder.function("mixedBreak") {
            val x = param("x", f32s)
            val n = const(7, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    // breakCond = LAND(STEP(args[0]), STEP(args[1])) — touches both.
                    val brkA = op(OpKind.STEP, listOf(args[0]), boolS)
                    val brkB = op(OpKind.STEP, listOf(args[1]), boolS)
                    val brkInner = op(OpKind.LAND, listOf(brkA, brkB), boolS)
                    val notBrk = op(OpKind.NOT, listOf(brkInner), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        val klass = BreakBearingWhile.classifyBreakCond(pattern)
        assertEquals(BreakBearingWhile.BreakCondClass.CarriedDependent, klass)
    }

    @Test
    fun classifyReturnsNullWhenCounterArgIdxIsNull() {
        // Foreign origCond shape leaves Pattern.counterArgIdx null. classifyBreakCond
        // requires Phase-1/1.5/2 invariants to be met — without a known counter index
        // it can't distinguish counter from generic carried args. Returns null.
        val fn = DxirBuilder.function("foreignCondForClassify") {
            val x = param("x", f32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(OpKind.STEP, listOf(args[0]), boolS)
                    val notBrk = op(OpKind.NOT, listOf(op(OpKind.STEP, listOf(args[1]), boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))!!
        assertNull(pattern.counterArgIdx, "precondition: foreign cond shape leaves counterArgIdx null")
        assertNull(BreakBearingWhile.classifyBreakCond(pattern))
    }

    // --- D.3i Phase 2 (§0.4.125) validation tests ---

    @Test
    fun rejectsNonZeroCounterInit() {
        // §0.4.125 — when the counter init isn't `const(0)`, the counter fields
        // get downgraded to null even though the structural LAND-NOT match holds.
        val fn = DxirBuilder.function("nonZeroInit") {
            val x = param("x", f32s)
            val n = const(7, i32s)
            val nonZeroStart = const(2, i32s)  // breaks the init invariant
            val w = whileOp(
                inits = listOf(x, nonZeroStart),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(op(OpKind.STEP, listOf(args[0]), boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNotNull(pattern, "structural LAND-NOT match still succeeds")
        assertNull(pattern.counterArgIdx, "non-zero init invalidates the counter extraction")
        assertNull(pattern.tripCountConst)
    }

    @Test
    fun rejectsNonStandardBackEdgeIncrement() {
        // §0.4.125 — when the body's counter back-edge isn't `ADD(args[counterIdx],
        // const(1))` (e.g., increment by 2 instead of 1), the counter fields downgrade.
        val fn = DxirBuilder.function("incrTwo") {
            val x = param("x", f32s)
            val n = const(7, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(op(OpKind.STEP, listOf(args[0]), boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(2, i32s)), i32s)  // +2 instead of +1
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNotNull(pattern, "structural LAND-NOT match still succeeds")
        assertNull(pattern.counterArgIdx, "non-+1 increment invalidates the counter extraction")
    }

    @Test
    fun rejectsBackEdgeReferencingDifferentArg() {
        // §0.4.125 — when the counter back-edge's first operand isn't the counter's
        // own block arg (e.g., `ADD(bodyArgs[0], const(1))` for counter at idx=1),
        // the counter fields downgrade. Catches mis-wired loop bodies.
        val fn = DxirBuilder.function("crossWiredBackEdge") {
            val x = param("x", f32s)
            val n = const(7, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(op(OpKind.STEP, listOf(args[0]), boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    // Wire the counter slot's back-edge to args[0] (the f32 carried)
                    // — semantically broken but structurally a valid ADD. Phase 2's
                    // validation must catch this.
                    val newI = op(OpKind.ADD, listOf(args[0], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNotNull(pattern, "structural LAND-NOT match still succeeds")
        assertNull(pattern.counterArgIdx, "back-edge referencing wrong arg invalidates the counter extraction")
    }

    @Test
    fun extractsConcreteTripCountAndCounterIndex() {
        // §0.4.124 — when origCond matches the canonical STEP(SUB(n, counter)) C5/C6
        // shape, the detector populates `counterArgIdx` + `tripCountConst`. Pin both.
        val fn = DxirBuilder.function("breakLoopWithCounter") {
            val x = param("x", f32s)
            val n = const(7, i32s)  // concrete trip-count bound
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    // origCond shape: STEP(SUB(n, args[1])) — counter arg at idx 1.
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(op(OpKind.STEP, listOf(args[0]), boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNotNull(pattern)
        assertEquals(1, pattern.counterArgIdx, "counter is the second cond-arg (idx=1)")
        assertEquals(7, pattern.tripCountConst, "trip-count bound = 7")
        assertNull(pattern.tripCountParam, "concrete bound shouldn't populate the symbolic field")
    }

    @Test
    fun extractsSymbolicTripCountParam() {
        // §0.4.124 — when n is a function param (loop-invariant scalar), the
        // detector populates `tripCountParam` instead of `tripCountConst`.
        val fn = DxirBuilder.function("symbolicTripBreak") {
            val x = param("x", f32s)
            val nParam = param("n", i32s)  // symbolic trip-count bound
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(nParam, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(op(OpKind.STEP, listOf(args[0]), boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNotNull(pattern)
        assertEquals(1, pattern.counterArgIdx)
        assertNull(pattern.tripCountConst, "symbolic bound shouldn't populate the concrete field")
        assertNotNull(pattern.tripCountParam, "symbolic bound should populate tripCountParam")
        assertEquals("n", pattern.tripCountParam!!.name)
    }

    @Test
    fun leavesCounterFieldsNullWhenOrigCondShapeIsForeign() {
        // origCond is `STEP(args[0])` — a STEP on a body arg directly, not the
        // SUB(n, counter) form. The counter fields should stay null.
        val fn = DxirBuilder.function("foreignCond") {
            val x = param("x", f32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    // Non-canonical origCond shape.
                    val origCond = op(OpKind.STEP, listOf(args[0]), boolS)
                    val notBrk = op(OpKind.NOT, listOf(op(OpKind.STEP, listOf(args[1]), boolS)), boolS)
                    yields(op(OpKind.LAND, listOf(origCond, notBrk), boolS))
                },
                body = { args ->
                    val newX = op(OpKind.MUL, listOf(args[0], const(2f, f32s)), f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], const(1, i32s)), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val pattern = BreakBearingWhile.detect(findWhile(fn))
        assertNotNull(pattern, "LAND-NOT structural match still works")
        assertNull(pattern.counterArgIdx, "non-canonical origCond should leave counterArgIdx null")
        assertNull(pattern.tripCountConst)
        assertNull(pattern.tripCountParam)
    }

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
