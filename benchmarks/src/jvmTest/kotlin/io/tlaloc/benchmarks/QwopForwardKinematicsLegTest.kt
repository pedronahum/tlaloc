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
 * §0.4.216 — QWOP Phase 2 fourth slice: first IF-in-WHILE multi-input
 * coarsened test. Combines §0.4.211/§0.4.212's [QwopHipUpdateTest]'s
 * IF-in-WHILE shape with §0.4.213's [QwopSumPositionsTest]'s multi-input
 * gradient routing on a single primal.
 *
 * **What this test pins**:
 *  - Structure: 1 WHILE in the primal body; 0 top-level IFs (the
 *    ground-contact IF lives inside the WHILE region body).
 *  - Coarsening: `PhiCalculus.apply` unrolls the constant-trip-count WHILE
 *    via C5 → 0 top-level WHILEs in the coarsened form.
 *  - Forward eval at non-clamping inputs: `pos = nSegs × (hip + knee + ankle)`.
 *  - Forward eval at clamping inputs: `pos = 0`.
 *  - Gradient at non-clamping inputs: each ∂/∂param = `nSegs`.
 *  - Gradient at clamping inputs: each ∂/∂param = `0` (the constant-zero
 *    then-branch kills upstream gradient flow).
 *
 * **Discriminator value**: a "no IF" implementation would always yield the
 * non-clamping gradient (3,3,3) regardless of inputs, failing the clamping
 * gradient pin (0,0,0). Likewise a "no clamping" forward would yield
 * `nSegs × (hip + knee + ankle)` even when negative.
 *
 * **Why this matters**: validates §0.4.212's `liftIfRegionBodies` pre-pass
 * fix on a multi-input shape — the §0.4.173 KNOWN LEAK that surfaced on
 * QWOP's hipUpdate (§0.4.211) would have crashed on this primal too without
 * the fix.
 */
class QwopForwardKinematicsLegTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.forwardKinematicsLegPrimal()
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "forwardKinematicsLegPrimal: 1 WHILE in primal body (the segment integration loop)",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "forwardKinematicsLegPrimal: 0 top-level IFs (ground-contact IF lives inside WHILE)",
        )
    }

    @Test
    fun forwardEvalNonClamping() {
        val primal = Qwop.forwardKinematicsLegPrimal()
        // (hip=1, knee=2, ankle=3) → sum=6 > 0 → no clamping.
        // pos[0]=0; tip[0]=6, pos[1]=6; tip[1]=12, pos[2]=12; tip[2]=18, pos[3]=18.
        // Final: nSegs × sum = 3 × 6 = 18.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(1.0f), floatArrayOf(2.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0] - 18.0f) < 1e-3f,
            "expected forward result = 18.0 at (1,2,3) non-clamping, got ${out[0][0]}",
        )
    }

    @Test
    fun forwardEvalClamping() {
        val primal = Qwop.forwardKinematicsLegPrimal()
        // (hip=-1, knee=-1, ankle=-1) → sum=-3 < 0 → clamping every iteration.
        // pos stays at 0 throughout. Final: 0.
        // Discriminator: a "no IF" implementation would yield nSegs × sum = -9.
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(-1.0f), floatArrayOf(-1.0f), floatArrayOf(-1.0f)),
        )
        assertTrue(
            abs(out[0][0] - 0.0f) < 1e-3f,
            "expected forward result = 0.0 at (-1,-1,-1) clamping, got ${out[0][0]}",
        )
    }

    @Test
    fun phiCalculusUnrollsConstantTripCountWhile() {
        val primal = Qwop.forwardKinematicsLegPrimal()
        val coarsened = PhiCalculus.apply(primal)
        assertEquals(
            0,
            BenchmarkPrimals.countOps(coarsened, OpKind.WHILE),
            "After PhiCalculus.apply, the constant-trip-count WHILE should be unrolled",
        )
    }

    @Test
    fun gradientNonClamping() {
        // §0.4.216 — headline pin (non-clamping branch). At (1,2,3) sum=6>0,
        // every iteration takes the else-branch. ∂pos[3]/∂hip = 3 (= nSegs)
        // since each iter contributes 1 via tip = pos + hip + knee + ankle.
        val primal = Qwop.forwardKinematicsLegPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(1.0f), floatArrayOf(2.0f), floatArrayOf(3.0f)),
        )
        assertTrue(abs(out[0][0] - 3.0f) < 1e-3f, "df/dhip = 3.0 (non-clamping)")
        assertTrue(abs(out[1][0] - 3.0f) < 1e-3f, "df/dknee = 3.0 (non-clamping)")
        assertTrue(abs(out[2][0] - 3.0f) < 1e-3f, "df/dankle = 3.0 (non-clamping)")
    }

    @Test
    fun gradientClamping() {
        // §0.4.216 — discriminator pin (clamping branch). At (-1,-1,-1) sum=-3<0,
        // every iteration takes the then-branch (constant 0). The gradient
        // through the constant-0 yield is identically 0; pos becomes constant
        // w.r.t. all three muscle inputs. df/dhip = df/dknee = df/dankle = 0.
        // Discriminator: a "no IF" implementation would give (3,3,3) here too.
        val primal = Qwop.forwardKinematicsLegPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(-1.0f), floatArrayOf(-1.0f), floatArrayOf(-1.0f)),
        )
        assertTrue(
            abs(out[0][0]) < 1e-3f,
            "df/dhip = 0 at clamping inputs (then-branch yields const 0), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0]) < 1e-3f,
            "df/dknee = 0 at clamping inputs, got ${out[1][0]}",
        )
        assertTrue(
            abs(out[2][0]) < 1e-3f,
            "df/dankle = 0 at clamping inputs, got ${out[2][0]}",
        )
    }

    @Test
    fun gradientNonClampingAtZeroInputs() {
        // Boundary pin: at (0, 0, 0), sum=0. STEP(0)=0 by Tlaloc convention,
        // so the predicate is false and the else-branch runs every iteration.
        // pos stays at 0 in forward, but gradient flows through tip = pos +
        // hip + knee + ankle each iteration → df/dhip = nSegs = 3.
        // (A "STEP(0) = 1" convention would clamp here and yield gradient 0;
        //  this pins the STEP(0)=0 contract.)
        val primal = Qwop.forwardKinematicsLegPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(0.0f), floatArrayOf(0.0f), floatArrayOf(0.0f)),
        )
        assertTrue(abs(out[0][0] - 3.0f) < 1e-3f, "df/dhip = 3.0 at zero inputs")
        assertTrue(abs(out[1][0] - 3.0f) < 1e-3f, "df/dknee = 3.0 at zero inputs")
        assertTrue(abs(out[2][0] - 3.0f) < 1e-3f, "df/dankle = 3.0 at zero inputs")
    }
}
