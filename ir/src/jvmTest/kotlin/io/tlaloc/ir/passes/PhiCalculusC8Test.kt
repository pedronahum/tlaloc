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
 * Stage B.3 — C8 (engine-backed variable-coefficient affine recurrence) tests. Lives
 * in jvmTest because C8 requires a [SymbolicEngine] impl. C8 generalises C7 by
 * allowing the multiplicative coefficient `a[i]` to also depend on the counter, with
 * the closed form `d_exit = p · ∏_{i=0}^{n-1} a[i] + Σ_{k=0}^{n-1} b[k] · ∏_{j=k+1}^{n-1} a[j]`
 * (derivation-verified against by-hand iteration in §0.4.18).
 */
class PhiCalculusC8Test {
    private val f32s = DxirType(F32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    private val engine = SymjaEngine()

    private fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    /**
     * Build a C8-pattern WHILE primal: `d ← a[i] · d + b[i]` where both [aExpr] and
     * [bExpr] are constructed from the f32-typed counter arg. Counter starts at 0,
     * increments by 1.0f.
     */
    private fun variableCoefficientLoop(
        nConcrete: Int?,
        pAsConst: Boolean,
        pInitConcrete: Float = 0f,
        aExpr: DxirRegionBuilder.(counterArg: io.tlaloc.ir.DxirBlockArg) -> DxirNode,
        bExpr: DxirRegionBuilder.(counterArg: io.tlaloc.ir.DxirBlockArg) -> DxirNode,
    ): io.tlaloc.ir.DxirFunction = DxirBuilder.function("varCoeff") {
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
                val a = aExpr(counterArg)
                val b = bExpr(counterArg)
                val aTimesD = op(OpKind.MUL, listOf(a, carriedArg), f32s)
                val newD = op(OpKind.ADD, listOf(aTimesD, b), f32s)
                val one = const(1f, f32s)
                val newI = op(OpKind.ADD, listOf(counterArg, one), f32s)
                yields(newD, newI)
            },
        )
        listOf(w.result(0))
    }

    /** Reference iteration: `d_{k+1} = a[k] · d_k + b[k]`. */
    private fun varCoeffReference(
        n: Int,
        p: Float,
        aAt: (Float) -> Float,
        bAt: (Float) -> Float,
    ): Float {
        var d = p
        for (i in 0 until n) d = aAt(i.toFloat()) * d + bAt(i.toFloat())
        return d
    }

    // ---- C8 with concrete n ------------------------------------------------

    @Test
    fun c8FactorialLikeRecurrenceClosesForConcreteN() {
        // a[i] = i+1, b[i] = 0 isn't valid (b must depend on counter for C8; if b is
        // const, C7 handles). Use a[i] = i+1, b[i] = i.
        // Iteration with p=1: d_0=1, d_1=1·1+0=1, d_2=2·1+1=3, d_3=3·3+2=11, d_4=4·11+3=47
        val original = variableCoefficientLoop(
            nConcrete = 4,
            pAsConst = true,
            pInitConcrete = 1f,
            aExpr = { counter ->
                val one = const(1f, f32s)
                op(OpKind.ADD, listOf(counter, one), f32s)
            },
            bExpr = { counter -> counter },
        )
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C8 should eliminate the WHILE")
        val out = DxirInterpreter.evalFunction(rewritten, emptyList())
        assertTrue(abs(out[0][0] - 47f) < 1e-3f, "expected 47.0, got ${out[0][0]}")
    }

    @Test
    fun c8WithSymbolicPClosesForConcreteN() {
        // Same shape as above but p is a runtime param.
        val original = variableCoefficientLoop(
            nConcrete = 4,
            pAsConst = false,
            aExpr = { counter ->
                val one = const(1f, f32s)
                op(OpKind.ADD, listOf(counter, one), f32s)
            },
            bExpr = { counter -> counter },
        )
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        // Reference: a[i]=i+1, b[i]=i, n=4
        for (p in listOf(0f, 1f, 2.5f, -1f)) {
            val expected = varCoeffReference(4, p, { i -> i + 1 }, { i -> i })
            val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(p)))
            assertTrue(
                abs(out[0][0] - expected) < 1e-3f,
                "at p=$p: expected=$expected got=${out[0][0]}",
            )
        }
    }

    @Test
    fun c8WithLinearAAndConstBClosesForConcreteN() {
        // a[i] = 2i + 1, b[i] = 5 (constant). Even though b is constant, C8 fires
        // because a depends on counter (C7 requires constant a).
        // Iteration with p=0, n=3: d_0=0, d_1=1·0+5=5, d_2=3·5+5=20, d_3=5·20+5=105
        val original = variableCoefficientLoop(
            nConcrete = 3,
            pAsConst = true,
            pInitConcrete = 0f,
            aExpr = { counter ->
                val two = const(2f, f32s)
                val one = const(1f, f32s)
                val twoI = op(OpKind.MUL, listOf(two, counter), f32s)
                op(OpKind.ADD, listOf(twoI, one), f32s)
            },
            bExpr = { _ -> const(5f, f32s) },
        )
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        val out = DxirInterpreter.evalFunction(rewritten, emptyList())
        assertTrue(abs(out[0][0] - 105f) < 1e-3f, "expected 105.0, got ${out[0][0]}")
    }

    @Test
    fun c8NumericalEquivalenceSweep() {
        // a[i] = i + 1, b[i] = 2i. Sweep p across multiple values for n=5.
        val original = variableCoefficientLoop(
            nConcrete = 5,
            pAsConst = false,
            aExpr = { counter ->
                val one = const(1f, f32s)
                op(OpKind.ADD, listOf(counter, one), f32s)
            },
            bExpr = { counter ->
                val two = const(2f, f32s)
                op(OpKind.MUL, listOf(two, counter), f32s)
            },
        )
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        for (p in listOf(0f, 1f, 0.5f, -2f)) {
            val expected = varCoeffReference(5, p, { i -> i + 1 }, { i -> 2 * i })
            val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(p)))
            assertTrue(
                abs(out[0][0] - expected) < 1e-2f,
                "at p=$p: expected=$expected got=${out[0][0]}",
            )
        }
    }

    // ---- Negative cases ----------------------------------------------------

    @Test
    fun c8DoesNotFireWhenAIsConstant() {
        // a is a const (= 2), b[i] = i. C7 fires (not C8), producing the C7 closed form.
        // C8's pattern matcher rejects (a doesn't depend on counter); C7's accepts.
        val original = variableCoefficientLoop(
            nConcrete = 3,
            pAsConst = true,
            pInitConcrete = 1f,
            aExpr = { _ -> const(2f, f32s) },
            bExpr = { counter -> counter },
        )
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C7 should fire (a is constant)")
        // Reference: d_0=1, d_1=2·1+0=2, d_2=2·2+1=5, d_3=2·5+2=12
        val out = DxirInterpreter.evalFunction(rewritten, emptyList())
        assertTrue(abs(out[0][0] - 12f) < 1e-3f, "expected 12.0, got ${out[0][0]}")
    }

    @Test
    fun c8SkipsWhenADependsOnCarriedAndC5UnrollsInstead() {
        // a depends on the carried — violates the affine hypothesis. C8/C7/C6 all
        // correctly skip. Pre-§0.4.40 C5 also skipped (counter-dependent back-edge);
        // §0.4.40 relaxed that check, so C5 unrolls this primal via per-iter counter
        // substitution. C8's vacuous skip is preserved — the WHILE is gone before C8
        // runs — and the unrolled primal evaluates correctly.
        val original = DxirBuilder.function("badC8") {
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
                    // a = args[carried] (depends on carried — invalid for affine)
                    val a = args[0]
                    val b = args[1]
                    val aTimesD = op(OpKind.MUL, listOf(a, args[0]), f32s)
                    val newD = op(OpKind.ADD, listOf(aTimesD, b), f32s)
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
        // Unrolled with p=2:
        //   iter 0: i=0, d = d*d + 0 = 4;     i = 1
        //   iter 1: i=1, d = d*d + 1 = 17;    i = 2
        // Result = 17.
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(2f)))
        assertTrue(abs(out[0][0] - 17f) < 1e-3f, "expected 17.0 (= p⁴ + 1 at p=2), got ${out[0][0]}")
    }
}
