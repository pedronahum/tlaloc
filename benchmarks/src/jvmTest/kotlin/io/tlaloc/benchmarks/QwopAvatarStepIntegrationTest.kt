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
 * §0.4.220 — QWOP Phase 2 closure: full `Qwop.avatarStepPrimal()` integration
 * test. Exercises all 12 top-level WHILEs (13 logical loops, including the
 * inner WHILE nested inside `crossLimbCoupling`) and all 8 IFs simultaneously.
 *
 * **What this test pins**:
 *  - Structure: 12 top-level WHILEs, 0 top-level IFs (IFs all live inside
 *    WHILE region bodies).
 *  - Forward eval at small no-clamping inputs (`m* = 0.1` for all four
 *    muscles): result ≈ 0.7227. Hand-traced through all 6 helpers
 *    (integrateMuscle ×4 → sumPositions → sumFineSteps → crossLimbCoupling →
 *    frictionAccum → forwardKinematicsLeg → forwardKinematicsArm →
 *    energyTorquePerJoint → energyAccumulator → 4 ADDs).
 *  - Coarsening doesn't throw on the full primal.
 *  - Reverse-mode AD doesn't throw on the coarsened primal.
 *  - Gradient function has 4 outputs (one per muscle input).
 *  - **Finite-difference (FD) validation**: per-input gradient agrees with
 *    central-difference numeric gradient within 1% relative tolerance at
 *    the small-input point. Validates that all per-slice gradient pins
 *    (§0.4.211–§0.4.219) compose correctly when combined into the full primal.
 *
 * **Why FD validation, not closed form**: the full primal's analytical
 * gradient is intractable to hand-derive (4 muscle inputs flowing through
 * 13 nested helpers with branching). FD at a fixed safe point (no clamp
 * triggers anywhere) is the standard integration-test approach.
 *
 * **Safe-input choice**: `m* = 0.1` keeps every IF predicate's input well
 * below its clamp threshold. ε = 0.01 perturbations don't flip any branch.
 */
class QwopAvatarStepIntegrationTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.avatarStepPrimal()
        // Top-level WHILEs:
        //   4 integrateMuscle + 1 sumPositions + 1 sumFineSteps + 1 outer
        //   crossLimbCoupling + 1 frictionAccum + 1 forwardKinematicsLeg +
        //   1 forwardKinematicsArm + 1 energyTorquePerJoint + 1 energyAccumulator
        //   = 12. (The inner WHILE inside crossLimbCoupling is nested,
        //   not visible to top-level countOps.)
        assertEquals(
            12,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "avatarStepPrimal: 12 top-level WHILEs (13 logical loops; one is nested)",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "avatarStepPrimal: 0 top-level IFs (all 8 IFs live inside WHILE bodies)",
        )
    }

    @Test
    fun forwardEvalAtSmallInputs() {
        val primal = Qwop.avatarStepPrimal()
        // m* = 0.1 for all four muscles. Hand-trace:
        //   Phase A: hip = knee = ankle = shoulder = 0.04 (4 × 0.1 × 0.1)
        //   Phase B: coarseDist = 3 × 0.12 = 0.36
        //            fineDist = 3 × 0.04 × 0.36 = 0.0432
        //   Phase C: coupling = 9 × 0.04 × 0.08 = 0.0288
        //            friction = 3 × 0.0288² ≈ 0.002488
        //   Phase D: legChain = 3 × 0.12 = 0.36
        //            armChain = 3 × (0.04 + 0.002488) ≈ 0.1275
        //   Phase E: torque ≈ 3 × (legChain + armChain) ≈ 3 × 0.4875 = 1.4625
        //            energy ≈ 3 × torque × fineDist ≈ 3 × 1.4625 × 0.0432 ≈ 0.1895
        //   Final: combined4 = fineDist + friction + legChain + armChain + energy
        //                   ≈ 0.0432 + 0.0025 + 0.36 + 0.1275 + 0.1895 ≈ 0.7227
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(
                floatArrayOf(0.1f), floatArrayOf(0.1f),
                floatArrayOf(0.1f), floatArrayOf(0.1f),
            ),
        )
        // Use 0.01 tolerance to absorb f32 accumulation across the chain.
        assertTrue(
            abs(out[0][0] - 0.7227f) < 0.01f,
            "expected forward ≈ 0.7227 at m*=0.1 (all-no-clamp), got ${out[0][0]}",
        )
    }

    @Test
    fun coarseningDoesNotThrow() {
        // §0.4.220 — observability pin. PhiCalculus.apply on the full
        // 12-WHILE primal exercises §0.4.176 nested coarsening + C5 unroll
        // at scale. Verifies the pipeline doesn't blow up on the larger
        // shape. Forward eval of coarsened should match primal forward.
        val primal = Qwop.avatarStepPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val outPrimal = DxirInterpreter.evalFunction(
            primal,
            listOf(
                floatArrayOf(0.1f), floatArrayOf(0.1f),
                floatArrayOf(0.1f), floatArrayOf(0.1f),
            ),
        )
        val outCoarsened = DxirInterpreter.evalFunction(
            coarsened,
            listOf(
                floatArrayOf(0.1f), floatArrayOf(0.1f),
                floatArrayOf(0.1f), floatArrayOf(0.1f),
            ),
        )
        assertTrue(
            abs(outCoarsened[0][0] - outPrimal[0][0]) < 1e-2f,
            "coarsened forward ≈ primal forward at m*=0.1; got primal=${outPrimal[0][0]}, coarsened=${outCoarsened[0][0]}",
        )
    }

    @Test
    fun reverseTransformDoesNotThrow() {
        // §0.4.220 — observability pin. DxirReverseTransform.apply on the
        // full coarsened primal exercises the §0.4.212 lift pass + reverse-
        // mode AD at scale. Validates the pipeline is self-contained for
        // the full QWOP shape — the largest dxir function produced in the
        // benchmark suite.
        val primal = Qwop.avatarStepPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)
        // Gradient function should have 4 returns (one per muscle input).
        assertEquals(
            4,
            grad.returns.size,
            "gradient function should return 4 grads (one per muscle input)",
        )
    }

    @Test
    fun gradientFiniteDifferenceValidated() {
        // §0.4.220 — the headline pin. Per-input gradient agrees with
        // central-difference FD within 1% relative tolerance at m*=0.1.
        // ε = 0.01 — small enough to approximate the derivative, large
        // enough to dominate f32 noise; doesn't flip any IF branch since
        // every predicate has a wide safety margin at this input.
        val primal = Qwop.avatarStepPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val baseInputs = floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f)
        val analyticalGrad = DxirInterpreter.evalFunction(
            grad,
            baseInputs.map { floatArrayOf(it) },
        )

        val eps = 0.01f
        for (i in 0 until 4) {
            // Perturb input i by ±ε.
            val plus = baseInputs.copyOf().also { it[i] += eps }
            val minus = baseInputs.copyOf().also { it[i] -= eps }
            val fPlus = DxirInterpreter.evalFunction(
                primal, plus.map { floatArrayOf(it) },
            )[0][0]
            val fMinus = DxirInterpreter.evalFunction(
                primal, minus.map { floatArrayOf(it) },
            )[0][0]
            val fdGrad = (fPlus - fMinus) / (2f * eps)
            val analGrad = analyticalGrad[i][0]

            // 1% relative tolerance, plus a small absolute floor (for cases
            // where both grad values are near zero — relative tolerance
            // breaks down there).
            val absDiff = abs(fdGrad - analGrad)
            val relTol = 0.01f * abs(fdGrad)
            val absFloor = 1e-3f
            assertTrue(
                absDiff < maxOf(relTol, absFloor),
                "input $i: analytical grad ${analGrad} ≠ FD grad ${fdGrad} " +
                    "(absDiff=${absDiff}, relTol=${relTol})",
            )
        }
    }

    @Test
    fun gradientReturnsExpectedSignsAtSmallInputs() {
        // Sanity discriminator: at m*=0.1 (all positive), every Phase A
        // muscle directly contributes to the final output via positive-
        // dependent helpers. So all four gradients should be POSITIVE.
        // A sign error in any per-slice gradient routing would surface here.
        val primal = Qwop.avatarStepPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(
                floatArrayOf(0.1f), floatArrayOf(0.1f),
                floatArrayOf(0.1f), floatArrayOf(0.1f),
            ),
        )
        for (i in 0 until 4) {
            assertTrue(
                out[i][0] > 0f,
                "input $i gradient should be positive at m*=0.1 (all-positive inputs); got ${out[i][0]}",
            )
        }
    }
}
