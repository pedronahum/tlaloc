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
 * §0.4.215 — QWOP Phase 2 third slice: squared-input single-loop coarsened
 * test on [Qwop.frictionAccumPrimal]. Exercises the chain rule through MUL
 * with **self**-operand; contrasts with §0.4.214's [QwopSumFineStepsTest]
 * (chain rule through MUL with **cross**-operand).
 *
 * **What this test pins**:
 *  - Structure: 1 WHILE + 0 IFs in the primal body.
 *  - Coarsening: `PhiCalculus.apply` unrolls the constant-trip-count WHILE
 *    via C5 → 0 top-level WHILEs in the coarsened form.
 *  - Forward eval at coupling=3.0: result = 27.0 (= `nSteps × coupling²` = 3×9).
 *  - Squared-input gradient: `df/dcoupling = nSteps × 2 × coupling`. The
 *    factor of 2 is the discriminator: a "MUL with shared operand" bug that
 *    only counts one partial would yield `nSteps × coupling`, not
 *    `nSteps × 2 × coupling`.
 *  - Sign-preserving: at coupling=-2 the gradient is -12 (negative), proving
 *    the gradient flows linearly through `coupling` (not through `|coupling|`
 *    or `coupling²`).
 *
 * **Why this matters for QWOP coarsening**: a self-MUL recurrence triggers
 * VjpRegistry's MulRule with both operands pointing at the same primal node.
 * Reverse-mode AD must sum both partial contributions correctly — a common
 * source of bugs in hand-coded AD systems. Pinning this on a coarsened
 * single-loop primal gives the QWOP pipeline a regression bulwark for
 * shared-operand chain-rule routing.
 */
class QwopFrictionAccumTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.frictionAccumPrimal()
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "frictionAccumPrimal: 1 WHILE in primal body (the squared-input accumulator)",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "frictionAccumPrimal: 0 IFs (pure self-MUL+ADD recurrence)",
        )
    }

    @Test
    fun forwardEval() {
        val primal = Qwop.frictionAccumPrimal()
        // After 3 iterations: acc = 3 * coupling²
        // At coupling=3: acc = 3 * 9 = 27
        val out = DxirInterpreter.evalFunction(primal, listOf(floatArrayOf(3.0f)))
        assertTrue(
            abs(out[0][0] - 27.0f) < 1e-3f,
            "expected forward result = 27.0 at coupling=3, got ${out[0][0]}",
        )
    }

    @Test
    fun phiCalculusUnrollsConstantTripCountWhile() {
        val primal = Qwop.frictionAccumPrimal()
        val coarsened = PhiCalculus.apply(primal)
        assertEquals(
            0,
            BenchmarkPrimals.countOps(coarsened, OpKind.WHILE),
            "After PhiCalculus.apply, the constant-trip-count WHILE should be unrolled",
        )
    }

    @Test
    fun gradientHasFactorOfTwo() {
        // §0.4.215 — the headline pin. Closed-form: df/dcoupling = nSteps × 2 × coupling.
        // At coupling=3: df/dcoupling = 3 × 2 × 3 = 18.
        // A "shared-operand chain rule" bug counting only one partial would give 9.
        val primal = Qwop.frictionAccumPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3.0f)))
        assertTrue(
            abs(out[0][0] - 18.0f) < 1e-3f,
            "expected df/dcoupling = 18.0 (= nSteps × 2 × coupling = 3 × 2 × 3), " +
                "got ${out[0][0]} (a 9.0 result indicates a MUL shared-operand chain-rule bug)",
        )
    }

    @Test
    fun gradientScalesLinearly() {
        // Discriminator: re-evaluate at multiple coupling values and verify the
        // gradient scales linearly (= nSteps × 2 × coupling). A quadratic-grad
        // implementation would fail this check.
        val primal = Qwop.frictionAccumPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        // At coupling=5: df/dcoupling = 3 × 2 × 5 = 30
        val out5 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(5.0f)))
        assertTrue(abs(out5[0][0] - 30.0f) < 1e-3f, "df/dcoupling=30 at coupling=5")

        // At coupling=10: df/dcoupling = 3 × 2 × 10 = 60
        val out10 = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(10.0f)))
        assertTrue(abs(out10[0][0] - 60.0f) < 1e-3f, "df/dcoupling=60 at coupling=10")

        // Linear scaling: out10 / out5 should equal 2 (since 10/5 = 2).
        assertTrue(
            abs(out10[0][0] / out5[0][0] - 2.0f) < 1e-3f,
            "gradient should scale linearly: out(coupling=10)/out(coupling=5) = 2",
        )
    }

    @Test
    fun gradientAtNegativeAndZeroInputs() {
        // Sign-preservation discriminator: gradient flows linearly through coupling,
        // so it should be negative for negative inputs and zero at zero.
        // At coupling=-2: df/dcoupling = 3 × 2 × -2 = -12 (negative!).
        // At coupling=0:  df/dcoupling = 3 × 2 × 0 = 0.
        val primal = Qwop.frictionAccumPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val outNeg = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(-2.0f)))
        assertTrue(
            abs(outNeg[0][0] - (-12.0f)) < 1e-3f,
            "expected df/dcoupling = -12.0 at coupling=-2 (sign-preserving), got ${outNeg[0][0]}",
        )

        val outZero = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(0.0f)))
        assertTrue(
            abs(outZero[0][0] - 0.0f) < 1e-3f,
            "expected df/dcoupling = 0.0 at coupling=0, got ${outZero[0][0]}",
        )
    }
}
