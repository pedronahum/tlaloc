package io.tlaloc.ir.passes

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Stage B.3 — **BGDHyperOpt Fig. 6 e2e port**. The big-payoff validation of the
 * C5/C6/C7/C8/C9 stack: hand-build the paper's batch-gradient-descent hyperparameter-
 * optimization kernel (paper Fig. 6a) and verify that `PhiCalculus.apply` produces a
 * closed-form primal numerically equivalent to the original loop.
 *
 * **Scope**: this test ports the OUTER while loop (paper Fig. 6a lines 3-11). The
 * inner for-loops (lines 6-7 and 13-14) are pre-collapsed into scalar sums `Sxy`,
 * `Sx2`, `Sy2` passed as runtime params — equivalent to the §0.4.13 bake-off's
 * approach. Once the dxir surface gains array-indexing ops (GATHER / EMBEDDING),
 * the inner-for-loops become C7-pattern WHILEs whose closure happens automatically
 * via `PhiCalculus.apply`.
 *
 * The outer while: `w_{k+1} = w_k - r · d_k / M` where `d_k = 2·(Sxy - Sx2·w_k)`
 * (i.e., `d_k = 2·Sxy - 2·Sx2·w_k`). Substituting:
 *   `w_{k+1} = w_k - r · (2·Sxy - 2·Sx2·w_k) / M`
 *           `= w_k · (1 + 2r·Sx2/M) - 2r·Sxy/M`
 * affine recurrence with:
 *   `a = 1 + 2r·Sx2/M`  (loop-invariant runtime-param subtree — the §0.4.20 widening)
 *   `b = -2r·Sxy/M`     (loop-invariant runtime-param subtree)
 *
 * C6 closes this to:
 *   `w_K = a^K · 0 + b · Σ_{i=0}^{K-1} a^i = b · (a^K − 1)/(a − 1)`
 *
 * Verified at concrete `r = 0.01`, `K = 3`, real-data Sxy/Sx2/M values from the
 * §0.4.13 bake-off harness.
 */
class PhiCalculusBgdHyperOptTest {
    private val f32s = DxirType(F32, emptyList())
    private val boolS = DxirType(Bool, emptyList())

    private val engine = SymjaEngine()

    private fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

    /** Reference Kotlin implementation of the BGDHyperOpt outer loop. */
    private fun bgdOuterLoop(r: Float, Sxy: Float, Sx2: Float, M: Float, K: Int): Float {
        val a = 1f + 2f * r * Sx2 / M
        val b = -2f * r * Sxy / M
        var w = 0f
        for (k in 0 until K) w = a * w + b
        return w
    }

    /**
     * Build the BGDHyperOpt outer-loop primal. The signature is
     * `(r, Sxy, Sx2, M) → w_final` with concrete `K` baked into the dxir.
     *
     * The construction precisely mirrors the paper's Fig. 6a structure (lines 3-11)
     * with the inner-for-loop d-computation pre-collapsed to scalar sums.
     */
    private fun bgdHyperOptPrimal(K: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("bgdOuterLoop") {
            val r = param("r", f32s)
            val Sxy = param("Sxy", f32s)
            val Sx2 = param("Sx2", f32s)
            val M = param("M", f32s)
            val wInit = const(0f, f32s)
            val kBound = const(K.toFloat(), f32s)
            val kZero = const(0f, f32s)
            val w = whileOp(
                inits = listOf(wInit, kZero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(kBound, args[1]), f32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val carriedW = args[0]
                    val counterArg = args[1]
                    // Build a = 1 + 2r·Sx2/M and b = -2r·Sxy/M as loop-invariant subtrees.
                    // These appear inside the body region but reference outer-scope
                    // params + consts (so the C6 widening accepts them).
                    val one = const(1f, f32s)
                    val two = const(2f, f32s)
                    val twoR = op(OpKind.MUL, listOf(two, r), f32s)
                    val twoRSx2 = op(OpKind.MUL, listOf(twoR, Sx2), f32s)
                    val twoRSx2OverM = op(OpKind.DIV, listOf(twoRSx2, M), f32s)
                    val aExpr = op(OpKind.ADD, listOf(one, twoRSx2OverM), f32s)
                    val twoRSxy = op(OpKind.MUL, listOf(twoR, Sxy), f32s)
                    val twoRSxyOverM = op(OpKind.DIV, listOf(twoRSxy, M), f32s)
                    val bExpr = op(OpKind.NEG, listOf(twoRSxyOverM), f32s)
                    // Back-edge: w' = a·w + b
                    val aw = op(OpKind.MUL, listOf(aExpr, carriedW), f32s)
                    val newW = op(OpKind.ADD, listOf(aw, bExpr), f32s)
                    // Counter increment.
                    val counterIncr = const(1f, f32s)
                    val newK = op(OpKind.ADD, listOf(counterArg, counterIncr), f32s)
                    yields(newW, newK)
                },
            )
            listOf(w.result(0))
        }

    /** Compute Sxy / Sx2 / Sy2 from concrete data arrays. */
    private fun moments(x: DoubleArray, y: DoubleArray): Triple<Float, Float, Float> {
        var sxy = 0.0; var sx2 = 0.0; var sy2 = 0.0
        for (i in x.indices) {
            sxy += x[i] * y[i]
            sx2 += x[i] * x[i]
            sy2 += y[i] * y[i]
        }
        return Triple(sxy.toFloat(), sx2.toFloat(), sy2.toFloat())
    }

    @Test
    fun bgdOuterLoopClosesViaC6AndMatchesReference() {
        // Real-data setup matching the §0.4.13 Symja bake-off harness.
        val xData = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val yData = doubleArrayOf(2.1, 3.9, 6.1, 8.0, 10.2)
        val (Sxy, Sx2, _) = moments(xData, yData)
        val M = xData.size.toFloat()

        // Build the primal with concrete K = 3.
        val K = 3
        val original = bgdHyperOptPrimal(K)

        // Coarsening pass — C6 should fire on the outer while.
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "C6 should close the outer while")

        // Numerical equivalence: compare to the Kotlin reference at multiple r values.
        for (r in listOf(0.001f, 0.01f, 0.05f, 0.1f)) {
            val expected = bgdOuterLoop(r, Sxy, Sx2, M, K)
            val out = DxirInterpreter.evalFunction(
                rewritten,
                listOf(floatArrayOf(r), floatArrayOf(Sxy), floatArrayOf(Sx2), floatArrayOf(M)),
            )
            val tolerance = abs(expected) * 1e-3f + 1e-5f
            assertTrue(
                abs(out[0][0] - expected) < tolerance,
                "at r=$r: expected=$expected got=${out[0][0]} (tol=$tolerance)",
            )
        }
    }

    @Test
    fun bgdOuterLoopOriginalAndRewrittenAgreeStepByStep() {
        // Stronger check: BOTH the original WHILE primal and the rewritten closed form
        // produce the same value at every sample input. Exercises both the dxir
        // interpreter's WHILE evaluation AND the C6 closed-form lowering.
        val xData = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val yData = doubleArrayOf(2.1, 3.9, 6.1, 8.0, 10.2)
        val (Sxy, Sx2, _) = moments(xData, yData)
        val M = xData.size.toFloat()
        val K = 3
        val original = bgdHyperOptPrimal(K)
        val rewritten = PhiCalculus.apply(original, engine)
        for (r in listOf(0.001f, 0.01f, 0.05f, 0.1f)) {
            val origOut = DxirInterpreter.evalFunction(
                original,
                listOf(floatArrayOf(r), floatArrayOf(Sxy), floatArrayOf(Sx2), floatArrayOf(M)),
            )[0][0]
            val rewrittenOut = DxirInterpreter.evalFunction(
                rewritten,
                listOf(floatArrayOf(r), floatArrayOf(Sxy), floatArrayOf(Sx2), floatArrayOf(M)),
            )[0][0]
            val tolerance = abs(origOut) * 1e-3f + 1e-5f
            assertTrue(
                abs(origOut - rewrittenOut) < tolerance,
                "at r=$r: original=$origOut, rewritten=$rewrittenOut (tol=$tolerance)",
            )
        }
    }

    @Test
    fun bgdOuterLoopClosesForLargerK() {
        // K = 10. C6's closed form is O(1) ops regardless of K; C5's direct unroll
        // would produce a 10-deep MUL+ADD chain. This validates that C6's value-add
        // (compact closed form) holds at non-trivial K.
        val xData = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val yData = doubleArrayOf(2.1, 3.9, 6.1, 8.0, 10.2)
        val (Sxy, Sx2, _) = moments(xData, yData)
        val M = xData.size.toFloat()
        val K = 10
        val original = bgdHyperOptPrimal(K)
        val rewritten = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE))
        for (r in listOf(0.001f, 0.01f)) {
            val expected = bgdOuterLoop(r, Sxy, Sx2, M, K)
            val out = DxirInterpreter.evalFunction(
                rewritten,
                listOf(floatArrayOf(r), floatArrayOf(Sxy), floatArrayOf(Sx2), floatArrayOf(M)),
            )
            val tolerance = abs(expected) * 1e-2f + 1e-4f
            assertTrue(
                abs(out[0][0] - expected) < tolerance,
                "at r=$r, K=$K: expected=$expected got=${out[0][0]} (tol=$tolerance)",
            )
        }
    }

    @Test
    fun bgdGradThroughCoarsenedOuterLoopMatchesFiniteDifferences() {
        // Integration test: after C6 closes the outer while, Stage A SCT differentiates
        // the coarsened primal. Compare the symbolic d/dr to a finite-differenced
        // reference on the Kotlin-side loop implementation.
        //
        // This exercises the FULL Stage A + Stage B pipeline — the "grad through
        // coarsened loops" capability the plan has been building toward.
        val xData = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val yData = doubleArrayOf(2.1, 3.9, 6.1, 8.0, 10.2)
        val (Sxy, Sx2, _) = moments(xData, yData)
        val M = xData.size.toFloat()
        val K = 3

        val original = bgdHyperOptPrimal(K)
        val coarsened = PhiCalculus.apply(original, engine)
        assertEquals(0, countOps(coarsened, OpKind.WHILE), "C6 should close the outer while")

        // Stage A SCT: returns 4 gradients (one per param: r, Sxy, Sx2, M). We care
        // about d/dr, which is returns[0].
        val grad = DxirReverseTransform.apply(coarsened)
        assertEquals(4, grad.returns.size, "4-param primal yields 4 gradients")

        // Finite-difference reference for d(w_final)/dr.
        val h = 1e-4f
        for (r in listOf(0.001f, 0.01f, 0.05f)) {
            val expected = (bgdOuterLoop(r + h, Sxy, Sx2, M, K) - bgdOuterLoop(r - h, Sxy, Sx2, M, K)) / (2f * h)
            val out = DxirInterpreter.evalFunction(
                grad,
                listOf(floatArrayOf(r), floatArrayOf(Sxy), floatArrayOf(Sx2), floatArrayOf(M)),
            )
            val symbolicDdr = out[0][0]
            val tolerance = kotlin.math.abs(expected) * 1e-2f + 1e-3f
            assertTrue(
                kotlin.math.abs(symbolicDdr - expected) < tolerance,
                "at r=$r: symbolic d/dr=$symbolicDdr, FD d/dr=$expected (tol=$tolerance)",
            )
        }
    }

    @Test
    fun bgdFullErrComputationIsStraightLineAfterCoarsening() {
        // The full BGDHyperOpt err computation: after the outer while closes via C6,
        // the post-loop `err = sqrt(Sy2 - 2·Sxy·w_4 + Sx2·w_4²) / M` is straight-line
        // dxir that the interpreter handles natively. This test verifies the full-err
        // primal coarsens cleanly + matches the reference.
        val xData = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val yData = doubleArrayOf(2.1, 3.9, 6.1, 8.0, 10.2)
        val (Sxy, Sx2, Sy2) = moments(xData, yData)
        val M = xData.size.toFloat()
        val K = 3

        val full = DxirBuilder.function("bgdFullErr") {
            val r = param("r", f32s)
            val SxyP = param("Sxy", f32s)
            val Sx2P = param("Sx2", f32s)
            val Sy2P = param("Sy2", f32s)
            val MP = param("M", f32s)
            val wInit = const(0f, f32s)
            val kBound = const(K.toFloat(), f32s)
            val kZero = const(0f, f32s)
            val w = whileOp(
                inits = listOf(wInit, kZero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(kBound, args[1]), f32s)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val carriedW = args[0]
                    val counterArg = args[1]
                    val one = const(1f, f32s)
                    val two = const(2f, f32s)
                    val twoR = op(OpKind.MUL, listOf(two, r), f32s)
                    val twoRSx2 = op(OpKind.MUL, listOf(twoR, Sx2P), f32s)
                    val twoRSx2OverM = op(OpKind.DIV, listOf(twoRSx2, MP), f32s)
                    val aExpr = op(OpKind.ADD, listOf(one, twoRSx2OverM), f32s)
                    val twoRSxy = op(OpKind.MUL, listOf(twoR, SxyP), f32s)
                    val twoRSxyOverM = op(OpKind.DIV, listOf(twoRSxy, MP), f32s)
                    val bExpr = op(OpKind.NEG, listOf(twoRSxyOverM), f32s)
                    val aw = op(OpKind.MUL, listOf(aExpr, carriedW), f32s)
                    val newW = op(OpKind.ADD, listOf(aw, bExpr), f32s)
                    val counterIncr = const(1f, f32s)
                    val newK = op(OpKind.ADD, listOf(counterArg, counterIncr), f32s)
                    yields(newW, newK)
                },
            )
            // Post-loop: err = sqrt(Sy2 - 2·Sxy·w + Sx2·w²) / M.
            // Use POW(x, 0.5) for sqrt since dxir has no SQRT-aware interpreter for
            // generic exponents — POW handles it via kotlin.math.pow.
            val wFinal = w.result(0)
            val two2 = const(2f, f32s)
            val twoSxyW = op(OpKind.MUL, listOf(op(OpKind.MUL, listOf(two2, SxyP), f32s), wFinal), f32s)
            val wSq = op(OpKind.MUL, listOf(wFinal, wFinal), f32s)
            val Sx2WSq = op(OpKind.MUL, listOf(Sx2P, wSq), f32s)
            val under = op(OpKind.ADD, listOf(op(OpKind.SUB, listOf(Sy2P, twoSxyW), f32s), Sx2WSq), f32s)
            val half = const(0.5f, f32s)
            val rootForm = op(OpKind.POW, listOf(under, half), f32s)
            val err = op(OpKind.DIV, listOf(rootForm, MP), f32s)
            listOf(err)
        }
        val rewritten = PhiCalculus.apply(full, engine)
        assertEquals(0, countOps(rewritten, OpKind.WHILE), "outer while should close")

        // Reference: run the full computation in Kotlin.
        fun referenceErr(r: Float): Float {
            val w = bgdOuterLoop(r, Sxy, Sx2, M, K)
            val under = Sy2 - 2f * Sxy * w + Sx2 * w * w
            return sqrt(under) / M
        }
        for (r in listOf(0.001f, 0.01f, 0.05f)) {
            val expected = referenceErr(r)
            val out = DxirInterpreter.evalFunction(
                rewritten,
                listOf(
                    floatArrayOf(r),
                    floatArrayOf(Sxy), floatArrayOf(Sx2), floatArrayOf(Sy2),
                    floatArrayOf(M),
                ),
            )
            val tolerance = abs(expected) * 1e-2f + 1e-4f
            assertTrue(
                abs(out[0][0] - expected) < tolerance,
                "at r=$r: expected=$expected got=${out[0][0]} (tol=$tolerance)",
            )
        }
    }
}
