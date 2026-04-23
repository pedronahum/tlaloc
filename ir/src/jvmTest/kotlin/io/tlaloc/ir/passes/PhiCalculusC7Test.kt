package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirRegionBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Stage B.3 — C7 (engine-backed indexed-affine-recurrence closed form) tests. Lives in
 * jvmTest because C7 requires a [SymbolicEngine] impl. C7 generalises C6 by allowing
 * the additive term to depend on the loop counter `i`, producing a closed form
 * `a^n · p + Σ_{i=0}^{n-1} a^i · b[n-1-i]`. This is the corollary BGDHyperOpt's inner
 * for-loop closes through (paper Fig. 6c lines 1-2).
 *
 * Counter is f32-typed in these primals (rather than the i32 used in C5/C6 tests)
 * because dxir doesn't yet have an i32→f32 CAST op, and C7's offset must add cleanly
 * to the f32 carried. The pattern matchers were widened in this session to accept any
 * scalar numeric counter type, so f32 counters are now first-class.
 */
class PhiCalculusC7Test {
    private val f32s = DxirType(F32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    private val engine = SymjaEngine()

    private fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    /**
     * Build a C7-pattern WHILE primal: `d ← a · d + b[i]` where `b[i]` is constructed
     * by the [bExpr] lambda from the f32-typed counter arg. Counter starts at 0,
     * increments by 1.0f.
     *
     * @param a multiplicative coefficient (concrete float)
     * @param nConcrete trip count: pass an Int for compile-time count, or null for a
     *   runtime f32-typed parameter (the C7 unique-value case)
     * @param pAsConst whether the carried init is `const(p_init)` or a runtime param
     * @param bExpr lambda to construct b[i] inside the body region; receives the
     *   f32-typed counter block-arg, returns the f32-typed offset DxirNode
     */
    private fun indexedAffineLoop(
        a: Float,
        nConcrete: Int?,
        pAsConst: Boolean,
        pInitConcrete: Float = 0f,
        bExpr: DxirRegionBuilder.(counterArg: io.tlaloc.ir.DxirBlockArg) -> DxirNode,
    ): io.tlaloc.ir.DxirFunction = DxirBuilder.function("indexedAffine") {
        val pInit = if (pAsConst) const(pInitConcrete, f32s) else param("p", f32s)
        val nBound = if (nConcrete != null) const(nConcrete.toFloat(), f32s) else param("n", f32s)
        val zero = const(0f, f32s)
        val w = whileOp(
            inits = listOf(pInit, zero),
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nBound, args[1]), f32s)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                val carriedArg = args[0]
                val counterArg = args[1]
                val offset = bExpr(counterArg)
                val newD = if (a == 1f) {
                    op(OpKind.ADD, listOf(carriedArg, offset), f32s)
                } else {
                    val aConst = const(a, f32s)
                    val aTimesD = op(OpKind.MUL, listOf(aConst, carriedArg), f32s)
                    op(OpKind.ADD, listOf(aTimesD, offset), f32s)
                }
                val one = const(1f, f32s)
                val newI = op(OpKind.ADD, listOf(counterArg, one), f32s)
                yields(newD, newI)
            },
        )
        listOf(w.result(0))
    }

    /** Reference: iterate the loop step-by-step. */
    private fun indexedAffineReference(
        a: Float,
        n: Int,
        p: Float,
        bAt: (Float) -> Float,
    ): Float {
        var d = p
        for (i in 0 until n) d = a * d + bAt(i.toFloat())
        return d
    }

    // ---- C7 with a=1 (additive accumulator, indexed offset) -----------------

    @Test
    fun c7AdditiveSumOfCounterClosesForConcreteN() {
        // f(p, n=5) = p + Σ_{i=0}^{4} i = p + 10. b[i] = i (the counter itself).
        val original = indexedAffineLoop(a = 1f, nConcrete = 5, pAsConst = false) { counter ->
            counter
        }
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C7 should eliminate the WHILE")
        // p = 0 → 0 + 10 = 10
        val out0 = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(0f)))
        assertTrue(abs(out0[0][0] - 10f) < 1e-3f, "expected 10.0 at p=0, got ${out0[0][0]}")
        // p = 7 → 17
        val out7 = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(7f)))
        assertTrue(abs(out7[0][0] - 17f) < 1e-3f, "expected 17.0 at p=7, got ${out7[0][0]}")
    }

    @Test
    fun c7AdditiveCounterPlusOneClosesForConcreteN() {
        // f(p, n=4) = p + Σ_{i=0}^{3} (i+1) = p + 10. b[i] = i + 1.
        val original = indexedAffineLoop(a = 1f, nConcrete = 4, pAsConst = false) { counter ->
            val one = const(1f, f32s)
            op(OpKind.ADD, listOf(counter, one), f32s)
        }
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        // p = 0 → 0 + (1+2+3+4) = 10
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(0f)))
        assertTrue(abs(out[0][0] - 10f) < 1e-3f, "expected 10.0, got ${out[0][0]}")
    }

    // ---- C7 with a=1 + symbolic n — the headline value-add ------------------

    @Test
    fun c7AdditiveSumOfCounterClosesForSymbolicN() {
        // f(p, n) = p + Σ_{i=0}^{n-1} i = p + n(n-1)/2
        val original = indexedAffineLoop(a = 1f, nConcrete = null, pAsConst = false) { counter ->
            counter
        }
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C7 closes loops with symbolic trip count")
        // (p, n) sweep with reference comparison.
        for (p in listOf(0f, 1f, 5f, -3f)) {
            for (n in listOf(0, 1, 3, 7, 10)) {
                val expected = indexedAffineReference(1f, n, p) { i -> i }
                val out = DxirInterpreter.evalFunction(
                    rewritten,
                    listOf(floatArrayOf(p), floatArrayOf(n.toFloat())),
                )
                assertTrue(
                    abs(out[0][0] - expected) < 1e-3f,
                    "at (p=$p, n=$n): expected=$expected got=${out[0][0]}",
                )
            }
        }
    }

    // ---- C7 with a≠1 (geometric weights × indexed offset) -------------------

    @Test
    fun c7WithMultiplicativeAClosesForConcreteN() {
        // a=2, b[i]=i, n=3, p=1.
        // Iter: d=1, d=2*1+0=2, d=2*2+1=5, d=2*5+2=12.
        // Closed form: 2^3 * 1 + Σ_{i=0}^{2} 2^i * (2-i) = 8 + (1*2 + 2*1 + 4*0) = 12.
        val original = indexedAffineLoop(a = 2f, nConcrete = 3, pAsConst = true, pInitConcrete = 1f) { counter ->
            counter
        }
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        val out = DxirInterpreter.evalFunction(rewritten, emptyList())
        assertTrue(abs(out[0][0] - 12f) < 1e-3f, "expected 12.0, got ${out[0][0]}")
    }

    @Test
    fun c7WithMultiplicativeAAndSymbolicN() {
        // a=2, b[i]=i, n symbolic, p symbolic. Closed form involves POW on n.
        val original = indexedAffineLoop(a = 2f, nConcrete = null, pAsConst = false) { counter ->
            counter
        }
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        // Sweep over (p, n) and compare to step-by-step reference.
        for (p in listOf(0f, 1f, 2.5f)) {
            for (n in listOf(0, 1, 2, 4, 6)) {
                val expected = indexedAffineReference(2f, n, p) { i -> i }
                val out = DxirInterpreter.evalFunction(
                    rewritten,
                    listOf(floatArrayOf(p), floatArrayOf(n.toFloat())),
                )
                assertTrue(
                    abs(out[0][0] - expected) < 1e-2f,
                    "at (p=$p, n=$n): expected=$expected got=${out[0][0]}",
                )
            }
        }
    }

    // ---- C7 with a polynomial-in-counter offset (BGDHyperOpt-shape) ---------

    @Test
    fun c7HandlesQuadraticOffsetInCounter() {
        // f(p, n=4) = p + Σ_{i=0}^{3} (2i² + 3i + 1) = p + (1 + 6 + 15 + 28) = p + 50.
        // Verifies the lift handles compound arithmetic in the offset subtree.
        val original = indexedAffineLoop(a = 1f, nConcrete = 4, pAsConst = false) { counter ->
            val c2 = const(2f, f32s)
            val c3 = const(3f, f32s)
            val c1 = const(1f, f32s)
            val iSq = op(OpKind.MUL, listOf(counter, counter), f32s)
            val twoISq = op(OpKind.MUL, listOf(c2, iSq), f32s)
            val threeI = op(OpKind.MUL, listOf(c3, counter), f32s)
            val ax = op(OpKind.ADD, listOf(twoISq, threeI), f32s)
            op(OpKind.ADD, listOf(ax, c1), f32s)
        }
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(0f)))
        assertTrue(abs(out[0][0] - 50f) < 1e-2f, "expected 50.0, got ${out[0][0]}")
        // Also at p=10: 60.
        val out10 = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(10f)))
        assertTrue(abs(out10[0][0] - 60f) < 1e-2f, "expected 60.0, got ${out10[0][0]}")
    }

    // ---- Negative cases ------------------------------------------------------

    @Test
    fun c7DoesNotFireWhenOffsetDoesNotDependOnCounter() {
        // b[i] = constant — that's a C6 pattern, not C7.
        val original = indexedAffineLoop(a = 1f, nConcrete = 4, pAsConst = false) { _ ->
            const(3f, f32s)
        }
        val rewritten = PhiCalculus.apply(original, engine)
        // C7 should skip; C6 should fire (a=1, b=3 → closed form `p + n*3 = p + 12`).
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C6 should pick this up")
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(0f)))
        assertTrue(abs(out[0][0] - 12f) < 1e-3f, "expected 12.0 (= 0 + 4*3), got ${out[0][0]}")
    }

    @Test
    fun c7SkipsWhenOffsetDependsOnCarriedAndC5UnrollsInstead() {
        // b[i] depends on the carried — violates C7's "f is affine in carried with
        // offset independent of carried" hypothesis. C7 correctly skips. C6 also skips
        // (MUL-coefficient is non-linear in carried). Pre-§0.4.40 C5 ALSO skipped
        // because the back-edge referenced the counter, leaving the WHILE in place;
        // §0.4.40 relaxed C5's counter-independence check, so C5 now unrolls this
        // shape into straight-line dxir via per-iter counter substitution. C7's
        // "vacuously correct" skip is preserved — there's no WHILE left for C7 to
        // attempt against — and the numerical answer is identical to what C7's
        // closed form would have produced on a matching shape.
        val original = DxirBuilder.function("badC7") {
            val p = param("p", f32s)
            val nBound = const(2f, f32s)
            val zero = const(0f, f32s)
            val w = whileOp(
                inits = listOf(p, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nBound, args[1]), f32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    // back-edge: ADD(args[carried], MUL(args[counter], args[carried]))
                    val offset = op(OpKind.MUL, listOf(args[1], args[0]), f32s)
                    val newD = op(OpKind.ADD, listOf(args[0], offset), f32s)
                    val one = const(1f, f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), f32s)
                    yields(newD, newI)
                },
            )
            listOf(w.result(0))
        }
        val rewritten = PhiCalculus.apply(original, engine)
        // C5 unrolls: no WHILE survives.
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C5 should unroll the WHILE")
        // Unrolled: iter 0: i=0, d = p + 0·p = p; iter 1: i=1, d = p + 1·p = 2p.
        // At p=3, result = 6.
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(3f)))
        assertTrue(abs(out[0][0] - 6f) < 1e-3f, "expected 6.0 (= 2·p), got ${out[0][0]}")
    }
}
