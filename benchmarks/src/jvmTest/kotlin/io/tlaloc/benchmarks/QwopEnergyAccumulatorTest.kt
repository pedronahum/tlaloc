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
 * §0.4.218 — QWOP Phase 2 sixth slice: MUL-in-else-branch + upper-bound IF
 * coarsened test. Combines §0.4.214's cross-operand MUL chain rule with
 * §0.4.217's upper-bound IF gradient routing — the first Phase 2 slice
 * that exercises both axes simultaneously.
 *
 * **What this test pins**:
 *  - Structure: 1 WHILE + 0 top-level IFs.
 *  - Coarsening: `PhiCalculus.apply` unrolls the constant-trip-count WHILE.
 *  - Forward eval, three regimes: below clamp, always clamp, partial clamp.
 *  - Gradient: cross-operand MUL chain rule routed through the upper-bound IF.
 *    - Below clamp: `df/dtorque = nSteps × dist`, `df/ddist = nSteps × torque`.
 *    - Any clamping: both gradients = 0.
 *
 * **Discriminator power**: §0.4.217's `forwardKinematicsArmPrimal` had a
 * pure-ADD recurrence inside the else-branch. §0.4.218 puts a **MUL** there.
 * Catches a class of bugs where chain-rule routing through the IF's
 * else-branch fails specifically for MUL operands (e.g., a "BROADCAST const"
 * gets misrouted into the MUL's gradient path).
 *
 * **Why this matters**: validates that VjpRegistry's MulRule (cross-operand
 * gradient) composes correctly with the IF gradient routing layer.
 */
class QwopEnergyAccumulatorTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.energyAccumulatorPrimal()
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "energyAccumulatorPrimal: 1 WHILE in primal body (the energy accumulator)",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "energyAccumulatorPrimal: 0 top-level IFs (energy-threshold IF lives inside WHILE)",
        )
    }

    @Test
    fun forwardEvalNonClamping() {
        val primal = Qwop.energyAccumulatorPrimal()
        // (torque=2, dist=3) → product=6, raw values 6,12,18 — all < 100.
        // Final: nSteps × torque × dist = 3 × 2 × 3 = 18.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(2.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0] - 18.0f) < 1e-3f,
            "expected forward = 18.0 at (2, 3) below-clamp, got ${out[0][0]}",
        )
    }

    @Test
    fun forwardEvalAlwaysClamp() {
        val primal = Qwop.energyAccumulatorPrimal()
        // (torque=20, dist=20) → product=400, raw[0]=400 > 100, clamps from iter 0.
        // Final: 100.0. Discriminator: a "no IF" forward gives 1200.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(20.0f), floatArrayOf(20.0f)),
        )
        assertTrue(
            abs(out[0][0] - 100.0f) < 1e-3f,
            "expected forward = 100.0 at (20, 20) always-clamp, got ${out[0][0]}",
        )
    }

    @Test
    fun forwardEvalPartialClamp() {
        val primal = Qwop.energyAccumulatorPrimal()
        // (torque=5, dist=10) → product=50.
        // iter 0: raw=50, !>100, energy[1]=50.
        // iter 1: raw=100, !>100 (STEP(0)=0), energy[2]=100.
        // iter 2: raw=150, >100, energy[3]=100 (clamped).
        // Final: 100. Discriminator: a "no IF" forward gives 150.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(5.0f), floatArrayOf(10.0f)),
        )
        assertTrue(
            abs(out[0][0] - 100.0f) < 1e-3f,
            "expected forward = 100.0 at (5, 10) partial-clamp, got ${out[0][0]}",
        )
    }

    @Test
    fun phiCalculusUnrollsConstantTripCountWhile() {
        val primal = Qwop.energyAccumulatorPrimal()
        val coarsened = PhiCalculus.apply(primal)
        assertEquals(
            0,
            BenchmarkPrimals.countOps(coarsened, OpKind.WHILE),
            "After PhiCalculus.apply, the constant-trip-count WHILE should be unrolled",
        )
    }

    @Test
    fun gradientNonClamping() {
        // §0.4.218 — the headline pin. Cross-operand MUL chain rule WITHIN an
        // IF's else-branch. At (torque=2, dist=3):
        //   df/dtorque = nSteps × dist = 3 × 3 = 9
        //   df/ddist = nSteps × torque = 3 × 2 = 6
        val primal = Qwop.energyAccumulatorPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(2.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0] - 9.0f) < 1e-3f,
            "expected df/dtorque = 9.0 (= nSteps × dist = 3 × 3), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0] - 6.0f) < 1e-3f,
            "expected df/ddist = 6.0 (= nSteps × torque = 3 × 2), got ${out[1][0]}",
        )
    }

    @Test
    fun gradientWithClamping() {
        // Both clamping regimes (always-clamp and partial-clamp) zero the
        // gradient because the final iteration takes the then-branch yielding
        // const-100f. Pin both in one test to keep the file compact.
        val primal = Qwop.energyAccumulatorPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        // Always-clamp at (20, 20):
        val outAlways = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(20.0f), floatArrayOf(20.0f)),
        )
        assertTrue(
            abs(outAlways[0][0]) < 1e-3f,
            "df/dtorque = 0 always-clamp (20, 20), got ${outAlways[0][0]}",
        )
        assertTrue(
            abs(outAlways[1][0]) < 1e-3f,
            "df/ddist = 0 always-clamp (20, 20), got ${outAlways[1][0]}",
        )

        // Partial-clamp at (5, 10):
        val outPartial = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(5.0f), floatArrayOf(10.0f)),
        )
        assertTrue(
            abs(outPartial[0][0]) < 1e-3f,
            "df/dtorque = 0 partial-clamp (5, 10), got ${outPartial[0][0]}",
        )
        assertTrue(
            abs(outPartial[1][0]) < 1e-3f,
            "df/ddist = 0 partial-clamp (5, 10), got ${outPartial[1][0]}",
        )
    }

    @Test
    fun gradientWithOneZeroInput() {
        // §0.4.218 — MUL chain-rule asymmetry pin. At (torque=0, dist=3),
        // product = 0, raw values all zero, never clamps. Forward = 0.
        // Gradient is asymmetric:
        //   df/dtorque = nSteps × dist = 3 × 3 = 9 (still flows!)
        //   df/ddist = nSteps × torque = 3 × 0 = 0 (zero — torque is 0)
        // This is the discriminator: a buggy MUL chain rule that conflated
        // the operands could yield (0, 9) instead of (9, 0), or (0, 0).
        val primal = Qwop.energyAccumulatorPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(0.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0] - 9.0f) < 1e-3f,
            "expected df/dtorque = 9.0 at (0, 3) (= nSteps × dist), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0]) < 1e-3f,
            "expected df/ddist = 0 at (0, 3) (= nSteps × torque = 0), got ${out[1][0]}",
        )
    }

    @Test
    fun gradientNegativeProduct() {
        // §0.4.218 — negative-product pin. The IF predicate is `> 100f`, so
        // negative raw values never clamp. At (torque=-2, dist=3):
        //   product = -6, raw values -6, -12, -18, all < 100, no clamping.
        //   df/dtorque = nSteps × dist = 9
        //   df/ddist = nSteps × torque = -6 (negative — sign-preserving)
        val primal = Qwop.energyAccumulatorPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(-2.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0] - 9.0f) < 1e-3f,
            "expected df/dtorque = 9.0 at (-2, 3) (negative product, no clamp), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0] - (-6.0f)) < 1e-3f,
            "expected df/ddist = -6.0 at (-2, 3) (sign-preserving), got ${out[1][0]}",
        )
    }
}
