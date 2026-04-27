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
 * §0.4.217 — QWOP Phase 2 fifth slice: upper-bound IF in WHILE coarsened
 * test. Symmetric counterpart of §0.4.216's [QwopForwardKinematicsLegTest]
 * (which tested a lower-bound ground-contact IF).
 *
 * **What this test pins**:
 *  - Structure: 1 WHILE + 0 top-level IFs (the upper-clamp IF lives inside).
 *  - Coarsening: `PhiCalculus.apply` unrolls the constant-trip-count WHILE.
 *  - Forward eval, three regimes:
 *    1. Below clamp (sum ≤ 4/3): forward = nSegs × sum.
 *    2. Always clamp (sum > 4): forward = 4f (clamps from iter 0).
 *    3. Partial clamp (4/3 < sum ≤ 4): forward = 4f (clamps mid-loop).
 *  - Gradient, two regimes:
 *    1. Below clamp: ∂/∂shoulder = ∂/∂friction = nSegs (linear gradient).
 *    2. Any clamping (final iter takes then-branch): both gradients = 0.
 *
 * **Discriminator value**: a "no IF" forward at clamping inputs would
 * produce nSegs × sum (e.g., 18 at sum=6) instead of the correct 4. A "no
 * IF" gradient would produce (3, 3) regardless of clamping — failing the
 * partial-clamp discriminator.
 *
 * **Why this matters**: the IF predicate `acc > 4f` is in the **opposite
 * direction** of §0.4.216's `tip < 0f`. Validates that branch-aware AD
 * works symmetrically for upper-bound and lower-bound IFs.
 */
class QwopForwardKinematicsArmTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.forwardKinematicsArmPrimal()
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "forwardKinematicsArmPrimal: 1 WHILE in primal body (the swing integration loop)",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "forwardKinematicsArmPrimal: 0 top-level IFs (torque-limit IF lives inside WHILE)",
        )
    }

    @Test
    fun forwardEvalBelowClamp() {
        val primal = Qwop.forwardKinematicsArmPrimal()
        // (shoulder=0.5, friction=0.5) → sum=1, well below 4/3. No clamping.
        // swing[0]=0; acc[0]=1, swing[1]=1; acc[1]=2, swing[2]=2; acc[2]=3, swing[3]=3.
        // Final: nSegs × sum = 3 × 1 = 3.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(0.5f), floatArrayOf(0.5f)),
        )
        assertTrue(
            abs(out[0][0] - 3.0f) < 1e-3f,
            "expected forward = 3.0 at (0.5, 0.5) below-clamp, got ${out[0][0]}",
        )
    }

    @Test
    fun forwardEvalAlwaysClamp() {
        val primal = Qwop.forwardKinematicsArmPrimal()
        // (shoulder=3, friction=3) → sum=6 > 4. Clamps at iter 0 and stays.
        // acc[0]=6, swing[1]=4; acc[1]=10, swing[2]=4; acc[2]=10, swing[3]=4.
        // Final: 4 (clamped). Discriminator: a "no IF" forward gives 18.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(3.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0] - 4.0f) < 1e-3f,
            "expected forward = 4.0 at (3, 3) always-clamp, got ${out[0][0]}",
        )
    }

    @Test
    fun forwardEvalPartialClamp() {
        val primal = Qwop.forwardKinematicsArmPrimal()
        // (shoulder=1, friction=1) → sum=2, in partial-clamp range (4/3, 4].
        // acc[0]=2, !>4, swing[1]=2; acc[1]=4, !>4 (STEP(0)=0), swing[2]=4;
        // acc[2]=6, >4, swing[3]=4.
        // Final: 4 (clamped at iter 2). Discriminator: a "no IF" gives 6.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(1.0f), floatArrayOf(1.0f)),
        )
        assertTrue(
            abs(out[0][0] - 4.0f) < 1e-3f,
            "expected forward = 4.0 at (1, 1) partial-clamp, got ${out[0][0]}",
        )
    }

    @Test
    fun phiCalculusUnrollsConstantTripCountWhile() {
        val primal = Qwop.forwardKinematicsArmPrimal()
        val coarsened = PhiCalculus.apply(primal)
        assertEquals(
            0,
            BenchmarkPrimals.countOps(coarsened, OpKind.WHILE),
            "After PhiCalculus.apply, the constant-trip-count WHILE should be unrolled",
        )
    }

    @Test
    fun gradientBelowClamp() {
        // §0.4.217 — non-clamping branch pin. At (0.5, 0.5) sum=1, every iter
        // takes else. ∂swing[3]/∂shoulder = nSegs = 3.
        val primal = Qwop.forwardKinematicsArmPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(0.5f), floatArrayOf(0.5f)),
        )
        assertTrue(abs(out[0][0] - 3.0f) < 1e-3f, "df/dshoulder = 3.0 below-clamp")
        assertTrue(abs(out[1][0] - 3.0f) < 1e-3f, "df/dfriction = 3.0 below-clamp")
    }

    @Test
    fun gradientAlwaysClamp() {
        // §0.4.217 — clamping branch pin. At (3, 3), final iter takes then-branch
        // yielding const 4. Gradient through the constant is 0 — both inputs.
        val primal = Qwop.forwardKinematicsArmPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(3.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0]) < 1e-3f,
            "df/dshoulder = 0 always-clamp (then-branch yields const 4f), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0]) < 1e-3f,
            "df/dfriction = 0 always-clamp, got ${out[1][0]}",
        )
    }

    @Test
    fun gradientPartialClamp() {
        // §0.4.217 — discriminator pin. At (1, 1) sum=2, the iter-0 else-branch
        // executes with shoulder/friction dependence (swing[1]=2), but iter 2
        // takes then-branch, yielding const 4 as the final output. The
        // gradient through the const-4f kills all upstream contributions.
        // df/dshoulder = df/dfriction = 0.
        // Discriminator: a "no IF" grad implementation would give (3, 3).
        val primal = Qwop.forwardKinematicsArmPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(1.0f), floatArrayOf(1.0f)),
        )
        assertTrue(
            abs(out[0][0]) < 1e-3f,
            "df/dshoulder = 0 partial-clamp (final iter takes then-branch), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0]) < 1e-3f,
            "df/dfriction = 0 partial-clamp, got ${out[1][0]}",
        )
    }
}
