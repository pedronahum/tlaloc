package io.tlaloc.ir.passes

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Stage B.0b — **Symja adequacy bake-off** per docs/STAGE_B_PLAN.md §3.2.3 and §7.1.b.
 *
 * Single hardest test in Stage B's runway: can Symja, the default symbolic engine,
 * close a nontrivial AD scenario (paper Fig. 6's BGDHyperOpt kernel) into a closed-form
 * expression and differentiate it within the spec'd time budget — and does the result
 * agree numerically with finite-differencing the original loop primal?
 *
 * If this passes, green light for B.1 (F1/F2/F3/C1/C3 implementation on hand-built IF
 * primals). If it fails, three options per plan §7.1.b: tune Symja, cut to custom Kotlin
 * CAS (~2-4 weeks scope), or descope to Stage B' (F1-F5 + C1-C4 only, ~40% paper
 * speedup).
 *
 * **Scope:** rather than re-implementing the paper's Fig. 6c walkthrough as a sequence
 * of `SymbolicEngine` calls (which is Stage B.1-B.3's actual deliverable), this test
 * constructs the closed form Symja should produce *given* a correct walkthrough — i.e.,
 * the form on Fig. 6c lines 18 + 26. The bake-off then exercises Symja's heavy lifting:
 * `simplify` on the assembled expression, `diff` on the simplified form, and FD-vs-symbolic
 * comparison at concrete numeric inputs. This isolates "can Symja handle the algebraic
 * complexity" from "have we ported the φ-calculus walkthrough correctly". Stage B.1+
 * incrementally builds the walkthrough; failures there are bugs in our pass code, not in
 * Symja.
 *
 * Time budgets per plan §3.2.3 (a) + (c):
 *  - Closed-form construction + simplify: < 60 seconds.
 *  - Differentiation + simplify: < 30 seconds.
 *  - Numerical agreement vs. FD on the loop primal: < 1e-3 relative error.
 */
class SymjaBakeoffTest {

    private val engine = SymjaEngine()

    /** Trip count for the `while` loop in BGDHyperOpt (Fig. 6a). Small but >= 2 to exercise C6. */
    private val K = 3

    /**
     * Reference implementation — Kotlin port of paper Fig. 6a's BGDHyperOpt kernel, with
     * the `if (d < 0.001) break` skipped (the closed form on Fig. 6c assumes K iterations
     * always run; the bake-off compares against this assumption-matching reference).
     *
     * Inputs: learning rate `r`, data `x[]` + `y[]` (must be same length M), trip count `K`.
     * Output: the final `err` value (Fig. 6a line 15: `err = (e0.5)/M`).
     */
    private fun bgdErrLoop(r: Double, x: DoubleArray, y: DoubleArray, K: Int): Double {
        require(x.size == y.size) { "x and y must have same length" }
        val M = x.size
        var w = 0.0
        for (k in 0 until K) {
            var d = 0.0
            for (i in 0 until M) d += 2.0 * x[i] * (y[i] - x[i] * w)
            d /= M.toDouble()
            // No break — closed form assumes always K iterations.
            w -= r * d
        }
        var e = 0.0
        for (j in 0 until M) {
            val r0 = y[j] - x[j] * w
            e += r0 * r0
        }
        return sqrt(e) / M.toDouble()
    }

    /**
     * Construct the closed-form `err` symbolically as a function of `(r, M, Sxy, Sx2, Sy2)`.
     * Per paper Fig. 6c lines 10 + 16 + 18 + 26:
     *
     *   w_3_exit = -2*r*Sxy/M * Sum_{k=0}^{K-1} (1 + 2*r*Sx2/M)^k
     *   w_4 = w_3_exit                                              (assuming no break)
     *   err = sqrt(Sy2 - 2*Sxy*w_4 + Sx2*w_4^2) / M
     *
     * The Sum is unrolled with concrete K (= [K]) since Symja closes geometric series
     * trivially when n is concrete; symbolic-K closure (paper's actual claim) is a Stage
     * B.1+ test against `engine.sum`.
     */
    private fun buildClosedFormErr(): Triple<SymExpr, Map<String, SymExpr>, SymExpr> {
        val r = engine.variable("r")
        val M = engine.variable("M")
        val Sxy = engine.variable("Sxy")
        val Sx2 = engine.variable("Sx2")
        val Sy2 = engine.variable("Sy2")
        val two = engine.rational(2)

        // a = 1 + 2*r*Sx2/M
        val a = engine.add(
            engine.rational(1),
            engine.div(engine.mul(engine.mul(two, r), Sx2), M),
        )
        // b = -2*r*Sxy/M
        val b = engine.neg(engine.div(engine.mul(engine.mul(two, r), Sxy), M))

        // Sum_{k=0}^{K-1} a^k — unroll for concrete K.
        var geomSum: SymExpr = engine.rational(0)
        for (k in 0 until K) {
            geomSum = engine.add(geomSum, engine.pow(a, engine.rational(k.toLong())))
        }
        // w_4 = b * geomSum
        val w4 = engine.mul(b, geomSum)

        // err = sqrt(Sy2 - 2*Sxy*w_4 + Sx2*w_4^2) / M
        val term1 = Sy2
        val term2 = engine.mul(engine.mul(two, Sxy), w4)
        val w4Sq = engine.mul(w4, w4)
        val term3 = engine.mul(Sx2, w4Sq)
        val under = engine.add(engine.sub(term1, term2), term3)
        val sqrt = engine.pow(under, engine.div(engine.rational(1), engine.rational(2)))
        val err = engine.div(sqrt, M)

        val symbols = mapOf("r" to r, "M" to M, "Sxy" to Sxy, "Sx2" to Sx2, "Sy2" to Sy2)
        return Triple(err, symbols, r)
    }

    /** Pre-compute `Sxy / Sx2 / Sy2` from concrete `x[] / y[]` arrays. */
    private fun precomputeMoments(x: DoubleArray, y: DoubleArray): Triple<Double, Double, Double> {
        var sxy = 0.0
        var sx2 = 0.0
        var sy2 = 0.0
        for (i in x.indices) {
            sxy += x[i] * y[i]
            sx2 += x[i] * x[i]
            sy2 += y[i] * y[i]
        }
        return Triple(sxy, sx2, sy2)
    }

    @Test
    fun bgdHyperOptClosedFormConstructsAndSimplifiesUnderBudget() {
        val constructionMs = measureTimeMillis {
            val (err, _, _) = buildClosedFormErr()
            // Construction itself doesn't trigger heavy work — exercise simplify.
            engine.simplify(err)
        }
        println("[bake-off] closed-form construction + simplify: ${constructionMs}ms")
        assertTrue(
            constructionMs < 60_000,
            "Symja closed-form construction + simplify exceeded 60s budget: ${constructionMs}ms",
        )
    }

    @Test
    fun bgdHyperOptGradientDifferentiatesUnderBudget() {
        val (err, _, r) = buildClosedFormErr()
        val errSimplified = engine.simplify(err)
        val gradMs = measureTimeMillis {
            val grad = engine.diff(errSimplified, r)
            engine.simplify(grad)
        }
        println("[bake-off] differentiation + simplify: ${gradMs}ms")
        assertTrue(
            gradMs < 30_000,
            "Symja diff + simplify exceeded 30s budget: ${gradMs}ms",
        )
    }

    @Test
    fun bgdHyperOptSymbolicGradientAgreesWithFiniteDifferences() {
        // Build the closed form + its derivative.
        val (err, syms, r) = buildClosedFormErr()
        val grad = engine.simplify(engine.diff(err, r))

        // Concrete inputs: small dataset, small learning rate (so the loop never breaks
        // even if break check were applied).
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.1, 8.0, 10.2)
        val rValue = 0.01
        val (sxy, sx2, sy2) = precomputeMoments(x, y)
        val M = x.size

        // Substitute concrete numeric values into the closed form's gradient.
        val subst = mapOf(
            syms["r"]!! to engine.realLiteral(rValue),
            syms["M"]!! to engine.realLiteral(M.toDouble()),
            syms["Sxy"]!! to engine.realLiteral(sxy),
            syms["Sx2"]!! to engine.realLiteral(sx2),
            syms["Sy2"]!! to engine.realLiteral(sy2),
        )
        val gradAtR = engine.substitute(grad, subst)
        val gradNumeric = engine.toDouble(gradAtR)

        // Finite-difference reference on the loop primal.
        val h = 1e-4
        val errPlus = bgdErrLoop(rValue + h, x, y, K)
        val errMinus = bgdErrLoop(rValue - h, x, y, K)
        val fdGrad = (errPlus - errMinus) / (2.0 * h)

        val relErr = abs(gradNumeric - fdGrad) / (abs(fdGrad) + 1e-12)
        println("[bake-off] symbolic d(err)/dr = $gradNumeric, FD d(err)/dr = $fdGrad, relErr = $relErr")
        assertTrue(
            relErr < 1e-3,
            "Symbolic gradient disagrees with FD: symbolic=$gradNumeric FD=$fdGrad relErr=$relErr",
        )
    }
}

