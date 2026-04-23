package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Stage B.3 — C9 (engine-backed power-form recurrence) tests. Lives in jvmTest because
 * C9 requires a [SymbolicEngine] impl. C9 closes WHILE primals of the form
 * `d ← a · d^b` to the closed form `d_exit = a^(Σ b^i) · p^(b^n)` (derivation-verified;
 * the §0.4.11 paper transcription `a^{b+n-1} · p^{b^n}` is wrong — see §0.4.19).
 */
class PhiCalculusC9Test {
    private val f32s = DxirType(F32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    private val engine = SymjaEngine()

    private fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    /**
     * Build a C9-pattern WHILE primal: `d ← a · d^b` with concrete float coefficients.
     * Counter is f32-typed (per the §0.4.17 widening); pattern matchers accept this.
     *
     * @param a multiplicative coefficient (float const)
     * @param b power exponent (float const)
     * @param nConcrete trip count: Int for compile-time, null for runtime f32 param
     * @param pAsConst whether the carried init is a const or a runtime param
     * @param pInitConcrete value of the carried init when [pAsConst] is true
     */
    private fun powerFormLoop(
        a: Float,
        b: Float,
        nConcrete: Int?,
        pAsConst: Boolean,
        pInitConcrete: Float = 1f,
    ): io.tlaloc.ir.DxirFunction = DxirBuilder.function("powerForm") {
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
                val bConst = const(b, f32s)
                val powed = op(OpKind.POW, listOf(carriedArg, bConst), f32s)
                val newD = if (a == 1f) {
                    powed
                } else {
                    val aConst = const(a, f32s)
                    op(OpKind.MUL, listOf(aConst, powed), f32s)
                }
                val one = const(1f, f32s)
                val newI = op(OpKind.ADD, listOf(counterArg, one), f32s)
                yields(newD, newI)
            },
        )
        listOf(w.result(0))
    }

    /** Reference iteration: `d_{k+1} = a · d_k^b`. */
    private fun powerFormReference(a: Float, b: Float, n: Int, p: Float): Float {
        var d = p.toDouble()
        for (i in 0 until n) d = a.toDouble() * d.pow(b.toDouble())
        return d.toFloat()
    }

    // ---- C9 with concrete n ------------------------------------------------

    @Test
    fun c9SquaringRecurrenceClosesForConcreteN() {
        // a=1, b=2: d ← d^2 (squaring). Closed form: d_n = p^(2^n).
        // p=2, n=3 → 2^(2^3) = 2^8 = 256.
        val original = powerFormLoop(a = 1f, b = 2f, nConcrete = 3, pAsConst = true, pInitConcrete = 2f)
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C9 should eliminate the WHILE")
        val out = DxirInterpreter.evalFunction(rewritten, emptyList())
        assertTrue(abs(out[0][0] - 256f) < 1e-2f, "expected 256.0, got ${out[0][0]}")
    }

    @Test
    fun c9MultiplicativePowerRecurrenceClosesForConcreteN() {
        // a=2, b=2, p=1, n=3: d ← 2·d^2.
        // Iteration: d_0=1, d_1=2·1=2, d_2=2·4=8, d_3=2·64=128.
        // Closed form: a^(Σ b^i) · p^(b^n) = 2^(1+2+4) · 1^8 = 2^7 = 128.
        val original = powerFormLoop(a = 2f, b = 2f, nConcrete = 3, pAsConst = true, pInitConcrete = 1f)
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        val out = DxirInterpreter.evalFunction(rewritten, emptyList())
        assertTrue(abs(out[0][0] - 128f) < 1e-2f, "expected 128.0, got ${out[0][0]}")
    }

    @Test
    fun c9CubingRecurrenceWithSymbolicP() {
        // a=2, b=3, n=2, p=symbolic. Closed form: 2^(1+3) · p^(3^2) = 16·p^9.
        // At p=1 → 16. At p=2 → 16·512 = 8192. At p=0.5 → 16·(1/512) = 0.03125.
        val original = powerFormLoop(a = 2f, b = 3f, nConcrete = 2, pAsConst = false)
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        for (p in listOf(1f, 2f, 0.5f)) {
            val expected = powerFormReference(2f, 3f, 2, p)
            val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(p)))
            val tolerance = if (abs(expected) > 1f) abs(expected) * 1e-3f else 1e-3f
            assertTrue(
                abs(out[0][0] - expected) < tolerance,
                "at p=$p: expected=$expected got=${out[0][0]} (tol=$tolerance)",
            )
        }
    }

    // ---- C9 with symbolic n — the engine-only-can-do case -------------------

    @Test
    fun c9SquaringRecurrenceClosesForSymbolicN() {
        // a=1, b=2, n=symbolic. Closed form: d_exit = p^(2^n). Sweep verifies it works
        // across multiple n values.
        val original = powerFormLoop(a = 1f, b = 2f, nConcrete = null, pAsConst = false)
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C9 closes loops with symbolic n")
        for (p in listOf(1f, 2f, 1.5f)) {
            for (n in listOf(0, 1, 2, 3, 4)) {
                val expected = powerFormReference(1f, 2f, n, p)
                val out = DxirInterpreter.evalFunction(
                    rewritten,
                    listOf(floatArrayOf(p), floatArrayOf(n.toFloat())),
                )
                val tolerance = if (abs(expected) > 1f) abs(expected) * 1e-2f else 1e-3f
                assertTrue(
                    abs(out[0][0] - expected) < tolerance,
                    "at (p=$p, n=$n): expected=$expected got=${out[0][0]} (tol=$tolerance)",
                )
            }
        }
    }

    // ---- Negative cases ----------------------------------------------------

    @Test
    fun c9DoesNotFireWhenBackEdgeIsNotPowerForm() {
        // d ← d * 2 (linear, not power-form). C6 should fire instead, producing the
        // C6 closed form. C9 must skip.
        val original = DxirBuilder.function("notC9") {
            val p = param("p", f32s)
            val nBound = const(3f, f32s)
            val zero = const(0f, f32s)
            val w = whileOp(
                inits = listOf(p, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nBound, args[1]), f32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    // Linear: d ← 2·d. C6 will close this to d_exit = 2^n · p (with b=0).
                    val twoConst = const(2f, f32s)
                    val newD = op(OpKind.MUL, listOf(twoConst, args[0]), f32s)
                    val one = const(1f, f32s)
                    val newI = op(OpKind.ADD, listOf(args[1], one), f32s)
                    yields(newD, newI)
                },
            )
            listOf(w.result(0))
        }
        val rewritten = PhiCalculus.apply(original, engine)
        // C6 fires (its `MUL(const, args[carried])` shape — b=0 case). WHILE eliminated.
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C6 should pick this up")
        // p=2 → 2·8 = 16
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(2f)))
        assertTrue(abs(out[0][0] - 16f) < 1e-3f, "expected 16.0, got ${out[0][0]}")
    }

    @Test
    fun c9SkipsWhenExponentDependsOnCounterAndC5UnrollsInstead() {
        // d ← d^counter (exponent depends on counter, not constant). C9 correctly
        // skips (pattern requires concrete `b` exponent-const). Pre-§0.4.40 C5 also
        // skipped (counter-dependent back-edge); §0.4.40 relaxed that check, so C5
        // unrolls with per-iter counter substitution. C9's vacuous skip is preserved.
        val original = DxirBuilder.function("varExp") {
            val p = param("p", f32s)
            val nBound = const(3f, f32s)
            val zero = const(0f, f32s)
            val w = whileOp(
                inits = listOf(p, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nBound, args[1]), f32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val newD = op(OpKind.POW, listOf(args[0], args[1]), f32s) // d^counter
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
        // Unrolled with p=2, N=3:
        //   iter 0: i=0, d = 2^0 = 1.
        //   iter 1: i=1, d = 1^1 = 1.
        //   iter 2: i=2, d = 1^2 = 1.
        // Result regardless of p: 1.
        val out = DxirInterpreter.evalFunction(rewritten, listOf(floatArrayOf(2f)))
        assertTrue(abs(out[0][0] - 1f) < 1e-3f, "expected 1.0 (= 1 after first p^0), got ${out[0][0]}")
    }
}
