package io.tlaloc.benchmarks

import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.PhiCalculus
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.214 — QWOP Phase 2 second slice: multiplicative-coupling single-loop
 * coarsened test on [Qwop.sumFineStepsPrimal]. Contrasts with §0.4.213's
 * [QwopSumPositionsTest] (pure-ADD recurrence with constant gradient) by
 * pinning the **first input-dependent** gradient on a coarsened QWOP primal.
 *
 * **What this test pins**:
 *  - Structure: 1 WHILE + 0 IFs in the primal body.
 *  - Coarsening: `PhiCalculus.apply` unrolls the constant-trip-count WHILE
 *    via C5 → 0 top-level WHILEs in the coarsened form.
 *  - Forward eval at (shoulder=2, coarseDist=4): result = 24.0
 *    (= `nSteps × shoulder × coarseDist` = 3 × 2 × 4).
 *  - Input-dependent gradient through coarsened single-loop:
 *    `df/dshoulder = nSteps × coarseDist`, `df/dcoarseDist = nSteps × shoulder`.
 *
 * **The discriminator**: re-evaluating at a second input set produces a
 * different gradient (verifying input-dependence). This catches a class of
 * bugs that §0.4.213's constant-gradient test cannot — namely, any "chain
 * rule through MUL with shared operand" misrouting that would leave the
 * gradient input-independent.
 */
class QwopSumFineStepsTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.sumFineStepsPrimal()
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "sumFineStepsPrimal: 1 WHILE in primal body (the multiplicative accumulator)",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "sumFineStepsPrimal: 0 IFs (pure MUL+ADD recurrence)",
        )
    }

    @Test
    fun forwardEval() {
        val primal = Qwop.sumFineStepsPrimal()
        // After 3 iterations: acc = 3 * shoulder * coarseDist
        // At (2, 4): acc = 3 * 2 * 4 = 24
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(2.0f), floatArrayOf(4.0f)),
        )
        assertTrue(
            abs(out[0][0] - 24.0f) < 1e-3f,
            "expected forward result = 24.0 at (shoulder=2, coarseDist=4), got ${out[0][0]}",
        )
    }

    @Test
    fun phiCalculusUnrollsConstantTripCountWhile() {
        val primal = Qwop.sumFineStepsPrimal()
        val coarsened = PhiCalculus.apply(primal)
        // C5 unroll: nSteps=3 is a constant trip count → 0 top-level WHILEs.
        assertEquals(
            0,
            BenchmarkPrimals.countOps(coarsened, OpKind.WHILE),
            "After PhiCalculus.apply, the constant-trip-count WHILE should be unrolled",
        )
    }

    @Test
    fun gradientIsInputDependent() {
        // §0.4.214 — the headline pin. Closed-form:
        //   df/dshoulder = nSteps * coarseDist = 3 * 4 = 12
        //   df/dcoarseDist = nSteps * shoulder = 3 * 2 = 6
        val primal = Qwop.sumFineStepsPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(2.0f), floatArrayOf(4.0f)),
        )

        assertTrue(
            abs(out[0][0] - 12.0f) < 1e-3f,
            "expected df/dshoulder = 12.0 (= nSteps × coarseDist = 3 × 4), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0] - 6.0f) < 1e-3f,
            "expected df/dcoarseDist = 6.0 (= nSteps × shoulder = 3 × 2), got ${out[1][0]}",
        )
    }

    @Test
    fun gradientChangesWithInputs() {
        // Discriminator: re-evaluate at a wildly different input set and
        // verify the gradient also changes (proving input-dependence).
        // At (shoulder=5, coarseDist=7):
        //   df/dshoulder = nSteps * coarseDist = 3 * 7 = 21
        //   df/dcoarseDist = nSteps * shoulder = 3 * 5 = 15
        val primal = Qwop.sumFineStepsPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(5.0f), floatArrayOf(7.0f)),
        )
        assertTrue(
            abs(out[0][0] - 21.0f) < 1e-3f,
            "expected df/dshoulder = 21.0 at (5, 7), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0] - 15.0f) < 1e-3f,
            "expected df/dcoarseDist = 15.0 at (5, 7), got ${out[1][0]}",
        )
    }

    @Test
    fun gradientAtNegativeAndZeroInputs() {
        // Edge-case discriminator: signed and zero inputs. The closed-form
        // is symmetric (df/dshoulder = nSteps × coarseDist regardless of
        // shoulder's value) — but a gradient implementation that mistakenly
        // squared an input or applied an absolute-value would fail here.
        // At (shoulder=-3, coarseDist=2): df/dshoulder = 3 × 2 = 6,
        //                                 df/dcoarseDist = 3 × -3 = -9
        // At (shoulder=0, coarseDist=10): df/dshoulder = 3 × 10 = 30,
        //                                 df/dcoarseDist = 3 × 0 = 0
        val primal = Qwop.sumFineStepsPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val outNeg = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(-3.0f), floatArrayOf(2.0f)),
        )
        assertTrue(abs(outNeg[0][0] - 6.0f) < 1e-3f, "df/dshoulder=6 at (-3, 2)")
        assertTrue(abs(outNeg[1][0] - (-9.0f)) < 1e-3f, "df/dcoarseDist=-9 at (-3, 2)")

        val outZero = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(0.0f), floatArrayOf(10.0f)),
        )
        assertTrue(abs(outZero[0][0] - 30.0f) < 1e-3f, "df/dshoulder=30 at (0, 10)")
        assertTrue(abs(outZero[1][0] - 0.0f) < 1e-3f, "df/dcoarseDist=0 at (0, 10)")
    }
}
