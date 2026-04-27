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
 * §0.4.213 — QWOP Phase 2 first slice: pure-WHILE no-IF single-loop coarsened
 * test on [Qwop.sumPositionsPrimal]. Contrasts with §0.4.211/§0.4.212's
 * [QwopHipUpdateTest] (which exercised single-input WHILE+IF) by pinning the
 * **multi-input** coarsened-gradient surface on a pure ADD-chain WHILE.
 *
 * **What this test pins**:
 *  - Structure: 1 WHILE in the primal body; 0 IFs.
 *  - Coarsening: `PhiCalculus.apply` unrolls the constant-trip-count WHILE
 *    via C5 → 0 top-level WHILEs in the coarsened form.
 *  - Forward eval at (hip, knee, ankle) = (1.0, 2.0, 3.0): result = 18.0
 *    (= `nSteps × (hip + knee + ankle)` = 3 × 6).
 *  - Multi-input gradient through coarsened single-loop: each of the three
 *    closed-form gradients equals `nSteps = 3` (independent of input values).
 *
 * **The discriminator**: a "no coarsening" or "wrong gradient routing"
 * implementation would yield gradient values other than 3.0 for one or more
 * inputs. Pinning all three inputs simultaneously catches per-input grad
 * misrouting that a single-input test (like hipUpdate) cannot.
 */
class QwopSumPositionsTest {

    @Test
    fun primalStructure() {
        val primal = Qwop.sumPositionsPrimal()
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "sumPositionsPrimal: 1 WHILE in primal body (the accumulation loop)",
        )
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "sumPositionsPrimal: 0 IFs (pure-ADD recurrence, no branch)",
        )
    }

    @Test
    fun forwardEval() {
        val primal = Qwop.sumPositionsPrimal()
        // After 3 iterations: acc = 3 * (hip + knee + ankle)
        // At (1.0, 2.0, 3.0): acc = 3 * 6 = 18.0
        val out = DxirInterpreter.evalFunction(
            primal,
            listOf(floatArrayOf(1.0f), floatArrayOf(2.0f), floatArrayOf(3.0f)),
        )
        assertTrue(
            abs(out[0][0] - 18.0f) < 1e-3f,
            "expected forward result = 18.0 at (hip=1, knee=2, ankle=3), got ${out[0][0]}",
        )
    }

    @Test
    fun phiCalculusUnrollsConstantTripCountWhile() {
        val primal = Qwop.sumPositionsPrimal()
        val coarsened = PhiCalculus.apply(primal)
        // C5 unroll: nSteps=3 is a constant trip count → 0 top-level WHILEs.
        assertEquals(
            0,
            BenchmarkPrimals.countOps(coarsened, OpKind.WHILE),
            "After PhiCalculus.apply, the constant-trip-count WHILE should be unrolled",
        )
    }

    @Test
    fun gradientWrtAllThreeInputs() {
        // §0.4.213 — the headline pin for QWOP Phase 2's first slice.
        // Closed-form: out = nSteps * (hip + knee + ankle), so each partial
        // derivative is simply `nSteps = 3` (constant, independent of inputs).
        val primal = Qwop.sumPositionsPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        // Gradient function returns one grad per primal param, in param order.
        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(1.0f), floatArrayOf(2.0f), floatArrayOf(3.0f)),
        )

        assertTrue(
            abs(out[0][0] - 3.0f) < 1e-3f,
            "expected df/dhip = 3.0 (= nSteps), got ${out[0][0]}",
        )
        assertTrue(
            abs(out[1][0] - 3.0f) < 1e-3f,
            "expected df/dknee = 3.0 (= nSteps), got ${out[1][0]}",
        )
        assertTrue(
            abs(out[2][0] - 3.0f) < 1e-3f,
            "expected df/dankle = 3.0 (= nSteps), got ${out[2][0]}",
        )
    }

    @Test
    fun gradientIsInputIndependent() {
        // The pure-ADD recurrence makes the gradient constant w.r.t. input
        // values. Pin this by re-evaluating at a wildly different input set
        // and verifying the same constant-3.0 gradient comes back. A "wrong
        // gradient routing" implementation that leaks input values into the
        // grad expression would fail this discriminator.
        val primal = Qwop.sumPositionsPrimal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)

        val out = DxirInterpreter.evalFunction(
            grad,
            listOf(floatArrayOf(-7.5f), floatArrayOf(100.0f), floatArrayOf(0.001f)),
        )
        assertTrue(abs(out[0][0] - 3.0f) < 1e-3f, "df/dhip = 3.0 at hip=-7.5")
        assertTrue(abs(out[1][0] - 3.0f) < 1e-3f, "df/dknee = 3.0 at knee=100.0")
        assertTrue(abs(out[2][0] - 3.0f) < 1e-3f, "df/dankle = 3.0 at ankle=0.001")
    }
}
