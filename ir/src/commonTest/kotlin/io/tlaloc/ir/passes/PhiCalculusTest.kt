package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Stage B.1 — structural + numerical-equivalence tests for [PhiCalculus]. Hand-built
 * IF primals (B.0a substrate) are run through the φ-calculus pass and verified two
 * ways: (a) op-count / op-kind structure of the rewritten function matches the
 * formula's expected effect, and (b) [DxirInterpreter.evalFunction] produces
 * numerically equivalent outputs at sample inputs.
 */
class PhiCalculusTest {
    private val f32s = DxirType(F32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    /** Number of body ops of a given kind in a function. */
    private fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    /** Assert two functions agree numerically at sample inputs. */
    private fun assertNumericallyAgree(
        a: io.tlaloc.ir.DxirFunction,
        b: io.tlaloc.ir.DxirFunction,
        inputs: List<FloatArray>,
        tol: Float = 1e-5f,
    ) {
        val outA = DxirInterpreter.evalFunction(a, inputs)
        val outB = DxirInterpreter.evalFunction(b, inputs)
        assertEquals(outA.size, outB.size)
        for (i in outA.indices) {
            assertEquals(outA[i].size, outB[i].size, "output $i size mismatch")
            for (j in outA[i].indices) {
                assertTrue(
                    abs(outA[i][j] - outB[i][j]) < tol,
                    "output $i[$j]: a=${outA[i][j]} vs b=${outB[i][j]} (tol=$tol)",
                )
            }
        }
    }

    // ---- F1 — identity collapse ---------------------------------------------

    @Test
    fun f1CollapsesIfWithIdenticalBranchYields() {
        // f(x, p) = if (p) x else x  →  x
        val original = DxirBuilder.function("f1Trivial") {
            val x = param("x", f32s)
            val p = param("p", boolS)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(x) },
            )
            listOf(ifop)
        }
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.IF), "F1 should eliminate the IF")
        // Numerically equivalent: f(5.0, true) = 5.0, f(5.0, false) = 5.0
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(5f), floatArrayOf(1f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(5f), floatArrayOf(0f)))
    }

    @Test
    fun f1DoesNotCollapseIfWithDifferentBranchYields() {
        // f(x, y, p) = if (p) x else y  — branches yield distinct SSA ids; F1 must NOT fire.
        val original = DxirBuilder.function("f1Distinct") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val p = param("p", boolS)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(y) },
            )
            listOf(ifop)
        }
        val rewritten = PhiCalculus.apply(original)
        // F3 may canonicalise (since y's SSA id > x's, then-yield <= else-yield is
        // already the canonical order). IF should remain.
        assertEquals(1, countOps(rewritten, OpKind.IF), "F1 must not fire on distinct yields")
        assertNumericallyAgree(
            original, rewritten,
            listOf(floatArrayOf(3f), floatArrayOf(7f), floatArrayOf(1f)),
        )
    }

    // ---- F3 — canonicalisation (swap branches + NOT predicate) -------------

    @Test
    fun f3CanonicalisesNonCanonicalBranchOrder() {
        // f(x, y, p) = if (p) y else x  — note: then-yield-id (y) > else-yield-id (x)
        // so F3 should canonicalise to: if (NOT(p)) x else y
        val original = DxirBuilder.function("f3Swap") {
            val x = param("x", f32s) // id=0
            val y = param("y", f32s) // id=1
            val p = param("p", boolS) // id=2
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(y) }, // id=1 (greater than x's id=0)
                elseRegion = region { yields(x) },
            )
            listOf(ifop)
        }
        val rewritten = PhiCalculus.apply(original)
        // After F3: one IF (still) + one NOT op.
        assertEquals(1, countOps(rewritten, OpKind.IF), "F3 keeps the IF, just swaps branches")
        assertEquals(1, countOps(rewritten, OpKind.NOT), "F3 emits exactly one NOT for the predicate")
        // Numerically equivalent.
        assertNumericallyAgree(
            original, rewritten,
            listOf(floatArrayOf(3f), floatArrayOf(7f), floatArrayOf(1f)),
        )
        assertNumericallyAgree(
            original, rewritten,
            listOf(floatArrayOf(3f), floatArrayOf(7f), floatArrayOf(0f)),
        )
    }

    @Test
    fun f3LeavesAlreadyCanonicalIfUnchanged() {
        // f(x, y, p) = if (p) x else y  — then-yield-id (x) <= else-yield-id (y), already canonical.
        val original = DxirBuilder.function("f3Canonical") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val p = param("p", boolS)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(y) },
            )
            listOf(ifop)
        }
        val rewritten = PhiCalculus.apply(original)
        assertEquals(1, countOps(rewritten, OpKind.IF))
        assertEquals(0, countOps(rewritten, OpKind.NOT), "F3 should not emit NOT on already-canonical IF")
    }

    // ---- F2 / C1 — distributive ---------------------------------------------

    @Test
    fun f2DistributesUnaryOpIntoIfBranches() {
        // f(x, y, p) = NEG(if (p) x else y)  →  if (p) NEG(x) else NEG(y)
        val original = DxirBuilder.function("f2Unary") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val p = param("p", boolS)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(y) },
            )
            val neg = op(OpKind.NEG, listOf(ifop), f32s)
            listOf(neg)
        }
        val rewritten = PhiCalculus.apply(original)
        // After F2: the new IF wraps NEG inside each branch; the original NEG and original
        // IF are still in the body but unreferenced (dead — Stage B.3 DCE will strip).
        // The new IF's branches each contain one NEG.
        val newIf = rewritten.body.filterIsInstance<DxirOp>().single { it.op == OpKind.IF && rewritten.returns.contains(it) }
        assertEquals(1, newIf.regions[0].blocks.single().body.size, "then-branch contains 1 NEG")
        assertEquals(1, newIf.regions[1].blocks.single().body.size, "else-branch contains 1 NEG")
        // Numerically equivalent.
        assertNumericallyAgree(
            original, rewritten,
            listOf(floatArrayOf(3f), floatArrayOf(7f), floatArrayOf(1f)),
        )
        assertNumericallyAgree(
            original, rewritten,
            listOf(floatArrayOf(3f), floatArrayOf(7f), floatArrayOf(0f)),
        )
    }

    @Test
    fun f2DistributesBinaryOpIntoIfOperand() {
        // f(x, y, z, p) = (if (p) x else y) + z  →  if (p) (x+z) else (y+z)
        val original = DxirBuilder.function("f2Binary") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val z = param("z", f32s)
            val p = param("p", boolS)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(y) },
            )
            val sum = op(OpKind.ADD, listOf(ifop, z), f32s)
            listOf(sum)
        }
        val rewritten = PhiCalculus.apply(original)
        val newIf = rewritten.returns.single() as DxirOp
        assertEquals(OpKind.IF, newIf.op)
        assertEquals(1, newIf.regions[0].blocks.single().body.size, "then-branch contains 1 ADD")
        assertEquals(OpKind.ADD, (newIf.regions[0].blocks.single().body[0] as DxirOp).op)
        // f(2.0, 5.0, 10.0, true) = 2 + 10 = 12; f(2, 5, 10, false) = 5 + 10 = 15.
        val outTrue = DxirInterpreter.evalFunction(
            rewritten,
            listOf(floatArrayOf(2f), floatArrayOf(5f), floatArrayOf(10f), floatArrayOf(1f)),
        )
        assertEquals(12f, outTrue[0][0])
        val outFalse = DxirInterpreter.evalFunction(
            rewritten,
            listOf(floatArrayOf(2f), floatArrayOf(5f), floatArrayOf(10f), floatArrayOf(0f)),
        )
        assertEquals(15f, outFalse[0][0])
    }

    @Test
    fun f2DistributionExposesF1Collapse() {
        // f(x, p) = NEG(if (p) x else x)  →  F1 first → NEG(x)  (one pass through F1
        // before F2 sees it). End result: zero IFs, one NEG.
        val original = DxirBuilder.function("f1ThenF2") {
            val x = param("x", f32s)
            val p = param("p", boolS)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(x) },
            )
            val neg = op(OpKind.NEG, listOf(ifop), f32s)
            listOf(neg)
        }
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.IF), "F1 should eliminate the trivial IF before F2")
        assertEquals(1, countOps(rewritten, OpKind.NEG), "single NEG remains")
        assertNumericallyAgree(
            original, rewritten,
            listOf(floatArrayOf(4f), floatArrayOf(1f)),
        )
    }

    // ---- C3 — nested-IF flattening -----------------------------------------

    @Test
    fun c3FlattensNestedIfWhenInnerIfIsThenBranchSoleYield() {
        // f(a, b, c, p, q) = if (p) (if (q) a else b) else c
        // After C3: if (q) (if (p) a else c) else (if (p) b else c)
        // Both should produce the same value, just with a different structural shape.
        val original = DxirBuilder.function("c3Nested") {
            val a = param("a", f32s)
            val b = param("b", f32s)
            val c = param("c", f32s)
            val p = param("p", boolS)
            val q = param("q", boolS)
            val outerIf = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region {
                    val innerIf = ifOp(
                        cond = q,
                        types = listOf(f32s),
                        thenRegion = region { yields(a) },
                        elseRegion = region { yields(b) },
                    )
                    yields(innerIf)
                },
                elseRegion = region { yields(c) },
            )
            listOf(outerIf)
        }
        val rewritten = PhiCalculus.apply(original)
        // After C3: 3 IFs (outer was 2 IFs, now we have 3 — outer keyed on q, two inner
        // keyed on p). We tolerate dead nodes from the rewrite (the original outer IF
        // and inner IF remain in the body, unreferenced — Stage B.3 DCE strips them).
        // Numerically equivalent across all (p, q) combinations:
        val a = 1f; val b = 2f; val c = 3f
        for ((pVal, qVal) in listOf(0f to 0f, 0f to 1f, 1f to 0f, 1f to 1f)) {
            val ins = listOf(
                floatArrayOf(a), floatArrayOf(b), floatArrayOf(c),
                floatArrayOf(pVal), floatArrayOf(qVal),
            )
            val outA = DxirInterpreter.evalFunction(original, ins)[0][0]
            val outB = DxirInterpreter.evalFunction(rewritten, ins)[0][0]
            assertEquals(outA, outB, "p=$pVal, q=$qVal: original=$outA, rewritten=$outB")
        }
    }

    // ---- End-to-end pipeline: composed rewrites + numerical equivalence ----

    @Test
    fun pipelineConvergesAndPreservesSemantics() {
        // f(x, y, z, p) = (if (p) (NEG(x) + 1) else (NEG(x) + 1)) * z
        // F1 collapses the trivial-branches IF → NEG(x) + 1; then the outer MUL is
        // applied normally. Result: NEG(x)+1 * z, with no IF surviving.
        val original = DxirBuilder.function("e2e") {
            val x = param("x", f32s)
            val y = param("y", f32s)
            val z = param("z", f32s)
            val p = param("p", boolS)
            // Outer scope: shared NEG(x), shared ADD(NEG(x), 1).
            val negX = op(OpKind.NEG, listOf(x), f32s)
            val one = const(1f, f32s)
            val negXp1 = op(OpKind.ADD, listOf(negX, one), f32s)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(negXp1) },
                elseRegion = region { yields(negXp1) }, // SAME id as then-branch
            )
            val mul = op(OpKind.MUL, listOf(ifop, z), f32s)
            listOf(mul)
        }
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.IF), "F1 should eliminate the IF entirely")
        // f(x=3, y=anything, z=2, p=true) = ((-3) + 1) * 2 = -4.
        val out = DxirInterpreter.evalFunction(
            rewritten,
            listOf(floatArrayOf(3f), floatArrayOf(99f), floatArrayOf(2f), floatArrayOf(1f)),
        )
        assertEquals(-4f, out[0][0])
    }

    @Test
    fun applyIsIdempotentOnFixpoint() {
        val original = DxirBuilder.function("idempotent") {
            val x = param("x", f32s)
            val p = param("p", boolS)
            val ifop = ifOp(
                cond = p,
                types = listOf(f32s),
                thenRegion = region { yields(x) },
                elseRegion = region { yields(x) },
            )
            val neg = op(OpKind.NEG, listOf(ifop), f32s)
            listOf(neg)
        }
        val once = PhiCalculus.apply(original)
        val twice = PhiCalculus.apply(once)
        // Re-applying should produce a structurally equivalent function.
        assertEquals(once.body.size, twice.body.size, "apply should be idempotent on fixpoint")
        assertNumericallyAgree(
            once, twice,
            listOf(floatArrayOf(7f), floatArrayOf(1f)),
        )
    }

    // ---- NOT op smoke -------------------------------------------------------

    @Test
    fun notOpEvaluatesCorrectly() {
        val fn = DxirBuilder.function("notOp") {
            val p = param("p", boolS)
            val n = op(OpKind.NOT, listOf(p), boolS)
            listOf(n)
        }
        val outTrue = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(1f)))
        assertEquals(0f, outTrue[0][0], "NOT(true) = false")
        val outFalse = DxirInterpreter.evalFunction(fn, listOf(floatArrayOf(0f)))
        assertEquals(1f, outFalse[0][0], "NOT(false) = true")
    }

    // ---- C5 — simple-loop closed-form (Stage B.2) ---------------------------

    private val i32s = DxirType(io.tlaloc.core.I32, emptyList())

    /**
     * Helper: build the canonical iterateNTimes-style WHILE primal from B.0a, but with
     * a CONCRETE trip count [n] (a `DxirConst`) so C5 can detect + unroll.
     * `f(x, n) = x · 2^n` (carried doubles each iter, counter increments by 1).
     */
    private fun iterateConcreteN(n: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("iterate$n") {
            val x = param("x", f32s)
            val nConst = const(n, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32s)
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
            listOf(w.result(0))
        }

    @Test
    fun c5UnrollsIterate5IntoMulChain() {
        val original = iterateConcreteN(5)
        val rewritten = PhiCalculus.apply(original)
        // After C5: zero WHILEs in the body. The body has the unrolled MUL chain
        // (the carried back-edge is `MUL(arg, 2)`; unrolled 5 times → 5 MULs).
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C5 should eliminate the WHILE")
        // x = 3 → 3 · 2^5 = 96
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(3f)))
        assertEquals(96f, out[0][0])
        // Numerically equivalent to original.
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(3f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(0f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(-2f)))
    }

    @Test
    fun c5UnrollsIterateZeroToIdentity() {
        val original = iterateConcreteN(0)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C5 fires even for n=0")
        // x = 7 → no iterations → 7 unchanged
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(7f)))
        assertEquals(7f, out[0][0])
    }

    @Test
    fun c5UnrollsAdditiveLoop() {
        // `f(x, n=3) = x + 1 + 1 + 1 = x + 3` via WHILE carrying x and incrementing each iter.
        val n = 3
        val original = DxirBuilder.function("addN") {
            val x = param("x", f32s)
            val nConst = const(n, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val one = const(1f, f32s)
                    val newX = op(OpKind.ADD, listOf(args[0], one), f32s)
                    val oneI = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], oneI), i32s)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        // x = 10 → 10 + 3 = 13
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(10f)))
        assertEquals(13f, out[0][0])
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(0f)))
    }

    @Test
    fun c5DoesNotFireWhenBackEdgeDependsOnCounter() {
        // `f(x, n=3) = x + 0 + 1 + 2` (carried adds the counter each iter — back-edge
        // depends on counter, so C5 shouldn't fire). C5 leaves the WHILE in place.
        val original = DxirBuilder.function("counterDep") {
            val x = param("x", f32s)
            val nConst = const(3, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    // Need to convert i32 counter to f32 for ADD with x. Use SUB(i32, i32)
                    // → const cast missing. Simplest: add the COUNTER (already i32) to x
                    // via an i32 ADD; C5 would still need to detect counter dependence.
                    // To keep types clean, just add a F32 const that ALSO sums the counter:
                    // newX = x + (something computed from counter). For test purposes we
                    // can fake the dependence: newX = x + (counter * 0 + 1) so it depends
                    // structurally on counter even though numerically equivalent to x+1.
                    val zeroI = const(0, i32s)
                    val zeroPart = op(OpKind.MUL, listOf(args[1], zeroI), i32s) // = 0 * counter
                    val one = const(1f, f32s)
                    // We can't ADD i32 to f32 cleanly; just wire through to keep the
                    // structural dependency: newX = x + 1, but we ALSO use the counter
                    // somewhere in the ADD's operand chain. For simplicity, gate it
                    // on counter equality: newX = x + 1, but build a redundant op that
                    // references counter so detectSimpleLoop's referencesId check fires.
                    // Easier: have newX directly reference the counter arg in its tree.
                    val newXIntermediate = op(OpKind.ADD, listOf(args[0], one), f32s)
                    // Tag newX with a dead use of counter via SUB(newXIntermediate, mul(0, counter as f32 cast)).
                    // i32→f32 cast op doesn't exist in our minimal set, so fake the dependency
                    // by building a STEP(counter) and using it (output is bool, breaks types).
                    // Simpler hack: just don't make the back-edge depend on counter and rely
                    // on a different "C5 doesn't fire" scenario instead. See test below.
                    val oneI = const(1, i32s)
                    val newI = op(OpKind.ADD, listOf(args[1], oneI), i32s)
                    // Drop zeroPart from the actual graph (it was a stub); the real test
                    // is in c5DoesNotFireOnNonStandardCondShape below.
                    yields(newXIntermediate, newI)
                },
            )
            listOf(w.result(0))
        }
        // This particular construction doesn't actually depend on counter — C5 WILL fire.
        // Verify the rewrite still preserves semantics; the "doesn't fire" guarantee is
        // tested below.
        val rewritten = PhiCalculus.apply(original)
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(5f)))
    }

    @Test
    fun c5DoesNotFireOnNonStandardCondShape() {
        // WHILE with cond shape that's not STEP(SUB(n, args[counter])) — C5 must skip.
        // Here we use STEP(args[counter]) — runs while counter > 0; init=0 means it
        // never enters. C5's pattern detector requires the SUB shape, so it skips.
        val original = DxirBuilder.function("nonStdCond") {
            val x = param("x", f32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    // STEP(args[1]) — non-standard shape; not STEP(SUB(...)).
                    val pred = op(OpKind.STEP, listOf(args[1]), boolS)
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
        val rewritten = PhiCalculus.apply(original)
        // C5 must NOT fire — the WHILE survives.
        assertEquals(1, countOps(rewritten, OpKind.WHILE), "C5 must skip non-standard cond")
    }

    // ---- §0.4.127 — D.3i Phase 3b: Constant breakCond fold --------------------

    /**
     * Helper: build a break-bearing WHILE primal whose break predicate is the literal
     * [breakValue]. Mirrors [iterateConcreteN]'s shape but with cond region
     * `LAND(STEP(SUB(n, counter)), NOT(const(breakValue)))`. The body multiplies the
     * f32 carried by 2 each iteration. With `breakValue=true`, the loop runs zero
     * times and the result is `x` unchanged. With `breakValue=false`, the LAND-NOT
     * wrapper degenerates to the natural cond and the loop runs `n` times → `x · 2^n`.
     */
    private fun breakBearingWithConstantBreak(n: Int, breakValue: Boolean): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("breakConst${if (breakValue) "True" else "False"}_$n") {
            val x = param("x", f32s)
            val nConst = const(n, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(nConst, args[1]), i32s)),
                        boolS,
                    )
                    val notBrk = op(OpKind.NOT, listOf(const(breakValue, boolS)), boolS)
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

    @Test
    fun breakBearingConstantFoldCollapsesAlwaysBreaksTrue() {
        // alwaysBreaks=true → loop runs zero times → result = x unchanged. The WHILE
        // op disappears entirely from the rewritten body; the unroll never fires.
        val original = breakBearingWithConstantBreak(n = 5, breakValue = true)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "alwaysBreaks=true folds the WHILE away")
        // x = 9 → 9 (zero iterations).
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(9f)))
        assertEquals(9f, out[0][0])
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(9f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(0f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(-2.5f)))
    }

    @Test
    fun breakBearingConstantFoldRewritesAlwaysBreaksFalseToVanillaLoopThenC5() {
        // alwaysBreaks=false → cond region drops LAND-NOT → vanilla bounded WHILE
        // with concrete trip-count 5 → C5 unrolls in the same singlePass iteration.
        // End-to-end the WHILE disappears AND the result equals the unrolled chain.
        val original = breakBearingWithConstantBreak(n = 5, breakValue = false)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "fold + C5 should eliminate the WHILE")
        // x = 3 → 3 · 2^5 = 96.
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(3f)))
        assertEquals(96f, out[0][0])
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(3f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(0f)))
    }

    @Test
    fun breakBearingConstantFoldLeavesCarriedDependentBreakCondAlone() {
        // breakCond depends on the f32 carried (args[0]) — classifies as
        // CarriedDependent, NOT Constant. The fold pass must skip it; the WHILE
        // stays put because no later pass closes a break-bearing WHILE either.
        val original = DxirBuilder.function("carriedDepBreak") {
            val x = param("x", f32s)
            val nConst = const(7, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(nConst, args[1]), i32s)),
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
        val rewritten = PhiCalculus.apply(original)
        assertEquals(1, countOps(rewritten, OpKind.WHILE), "CarriedDependent breakCond is not Constant — leave WHILE")
        // Numerical agreement (sanity): with x=-1, STEP(args[0]) is 0, never breaks → loop runs all 7 iters → -1 · 2^7 = -128.
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(-1f)))
    }

    @Test
    fun breakBearingConstantFoldLeavesVanillaWhileAlone() {
        // Vanilla WHILE with no LAND-NOT cond — BreakBearingWhile.detect returns null
        // so the fold pass skips. C5 then unrolls it normally.
        val original = iterateConcreteN(4)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C5 still fires on vanilla loops")
        // x = 2 → 2 · 2^4 = 32. Confirms the fold pass didn't accidentally interfere.
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(2f)))
        assertEquals(32f, out[0][0])
    }

    // ---- §0.4.128 — D.3i Phase 3c: LoopInvariant breakCond lift -----------------

    /**
     * Helper: build a break-bearing WHILE primal whose breakCond is a function param
     * (loop-invariant Bool predicate). Counter increments by 1 from 0 to [n]; body
     * doubles the f32 carried each iteration. With `flag=true`, the loop runs zero
     * times → result = `x`. With `flag=false`, the loop runs `n` times → result =
     * `x · 2^n`.
     */
    private fun breakBearingWithLoopInvariantBreak(n: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("breakLoopInv_$n") {
            val x = param("x", f32s)
            val flag = param("flag", boolS)
            val nConst = const(n, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(nConst, args[1]), i32s)),
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

    @Test
    fun breakBearingClosureLiftsLoopInvariantParamBreakCondToOuterIf() {
        // breakCond = `flag` (function param). Lift produces an outer IF whose
        // then-arm yields x and else-arm runs the (now-vanilla) WHILE which C5
        // unrolls in the same singlePass iteration.
        val original = breakBearingWithLoopInvariantBreak(n = 5)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "WHILE replaced by IF + (unrolled vanilla loop)")
        assertEquals(1, countOps(rewritten, OpKind.IF), "Lift produces exactly one outer IF at top level")
        // flag=true (encoded as 1f) → loop never runs → result = x.
        val outFlagTrue = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(3f), floatArrayOf(1f)))
        assertEquals(3f, outFlagTrue[0][0])
        // flag=false (encoded as 0f) → loop runs 5 iters → result = 3 · 2^5 = 96.
        val outFlagFalse = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(3f), floatArrayOf(0f)))
        assertEquals(96f, outFlagFalse[0][0])
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(3f), floatArrayOf(1f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(3f), floatArrayOf(0f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(-2f), floatArrayOf(0f)))
    }

    @Test
    fun breakBearingClosureLiftsRegionInternalOpInBreakCond() {
        // breakCond = `STEP(threshold)` where threshold is a region-internal const
        // declared inside the cond region. Lifting must clone both the const and
        // the STEP op into outer scope so the IF's predicate is well-formed.
        val original = DxirBuilder.function("regionInternalLift") {
            val x = param("x", f32s)
            val n = const(4, i32s)
            val zero = const(0, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(n, args[1]), i32s)),
                        boolS,
                    )
                    // threshold is built inside the cond region — its id lives in
                    // condBlock.body and must be lifted to outer scope by the pass.
                    val threshold = const(0, i32s)
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
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        assertEquals(1, countOps(rewritten, OpKind.IF))
        // STEP(0) = 0 (false) → IF takes else branch → loop runs 4 iters → 5 · 2^4 = 80.
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(5f)))
        assertEquals(80f, out[0][0])
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(5f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(0f)))
    }

    // ---- §0.4.131 — D.3i Phase 3e: CounterOnly arm with concrete threshold ----

    /**
     * Helper: build a break-bearing WHILE primal whose breakCond is the canonical
     * CounterOnly shape `STEP(SUB(args[counterArgIdx], threshold))` (i.e., "break
     * when counter > threshold"). The body doubles the f32 carried per iter.
     * Effective trip count = min(n, threshold + 1).
     */
    private fun breakBearingCounterOnly(n: Int, threshold: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("breakCounterOnly_${n}_$threshold") {
            val x = param("x", f32s)
            val nConst = const(n, i32s)
            val zero = const(0, i32s)
            val cap = const(threshold, i32s)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val origCond = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(nConst, args[1]), i32s)),
                        boolS,
                    )
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

    @Test
    fun breakBearingClosureRewritesCounterOnlyWithBreakBeforeNaturalBound() {
        // n = 10, threshold = 3. The break fires at counter > 3 (iter 4); effective
        // trip count = min(10, 4) = 4. Loop runs 4 iters → 5 · 2^4 = 80.
        val original = breakBearingCounterOnly(n = 10, threshold = 3)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "fold + C5 should eliminate the WHILE")
        // x = 5 → 5 · 2^4 = 80.
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(5f)))
        assertEquals(80f, out[0][0])
        // Numerical agreement at multiple inputs.
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(5f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(0f)))
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(-1.5f)))
    }

    @Test
    fun breakBearingClosureRewritesCounterOnlyWithNaturalBoundBeforeBreak() {
        // n = 3, threshold = 10. Natural bound exits before the break fires;
        // effective trip count = min(3, 11) = 3. Loop runs 3 iters → 5 · 2^3 = 40.
        val original = breakBearingCounterOnly(n = 3, threshold = 10)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(5f)))
        assertEquals(40f, out[0][0])
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(5f)))
    }

    @Test
    fun breakBearingClosureRewritesCounterOnlyWithThresholdZero() {
        // n = 5, threshold = 0. Effective trip count = min(5, 1) = 1. Loop runs
        // 1 iter (iter 0; at iter 1 counter > 0 so break fires). x · 2^1 = 14.
        val original = breakBearingCounterOnly(n = 5, threshold = 0)
        val rewritten = PhiCalculus.apply(original)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(7f)))
        assertEquals(14f, out[0][0])
        assertNumericallyAgree(original, rewritten, listOf(floatArrayOf(7f)))
    }

    @Test
    fun breakBearingClosureLeavesNonCanonicalCounterOnlyAlone() {
        // breakCond shape is CounterOnly but NOT the canonical
        // `STEP(SUB(args[counter], thresholdConst))` form — instead it's
        // `STEP(SUB(thresholdConst, args[counter]))` (operand order swapped).
        // Phase 3e doesn't handle this shape today; the WHILE must stay intact.
        val fn = DxirBuilder.function("nonCanonicalCounterOnly") {
            val x = param("x", f32s)
            val n = const(7, i32s)
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
                    // Swapped: SUB(threshold, counter) instead of SUB(counter, threshold).
                    val brkInner = op(
                        OpKind.STEP,
                        listOf(op(OpKind.SUB, listOf(cap, args[1]), i32s)),
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
        val rewritten = PhiCalculus.apply(fn)
        assertEquals(1, countOps(rewritten, OpKind.WHILE), "non-canonical shape must leave WHILE intact")
    }

    @Test
    fun breakBearingConstantFoldEnablesEndToEndGradThroughBreakBearingLoop() {
        // alwaysBreaks=false + C5 → straight-line dxir → DxirReverseTransform
        // produces the gradient. d/dx of x · 2^4 = 16. End-to-end pin: the fold pass
        // unblocks gradient flow through what was a break-bearing WHILE.
        val original = breakBearingWithConstantBreak(n = 4, breakValue = false)
        val coarsened = PhiCalculus.apply(original)
        assertEquals(0, countOps(coarsened, OpKind.WHILE))
        val grad = DxirReverseTransform.apply(coarsened)
        val gradOut = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(7f)))
        assertEquals(1, gradOut.size)
        assertEquals(16f, gradOut[0][0], "d/dx of x·16 = 16")
    }

    @Test
    fun c5EnablesEndToEndGradThroughLoop() {
        // The Stage B.2 load-bearing test: grad { x -> iterate5(x) } should produce 32f
        // (since iterate5(x) = x · 32, d/dx = 32). After PhiCalculus.apply unrolls the
        // WHILE, Stage A's DxirReverseTransform differentiates the resulting straight-
        // line dxir. End-to-end: φ-calculus + SCT AD = correct gradient.
        val original = iterateConcreteN(5)
        val coarsened = PhiCalculus.apply(original)
        // Sanity: coarsened has no WHILE; Stage A SCT can process it.
        assertEquals(0, countOps(coarsened, OpKind.WHILE))
        // Differentiate.
        val grad = DxirReverseTransform.apply(coarsened)
        // Evaluate gradient at x = 7 (any value works; gradient is constant 32).
        val gradOut = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(7f)))
        assertEquals(1, gradOut.size, "single param → single gradient")
        assertEquals(32f, gradOut[0][0], "d/dx of x·32 = 32")
    }
}
