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
 * §0.4.211 — QWOP Phase 1 first-slice forward-only test. First per-body-part
 * regression pin on the QWOP synthetic primal — **forward computation only**.
 * Mirrors HMC's Phase 1 dxir-level pattern (`MultiBranchIfBenchmark`) but
 * stops short of the AD step.
 *
 * **What this test pins** (forward + coarsening only):
 *  - Structure: [Qwop.hipUpdatePrimal] produces 1 WHILE at top-level
 *    (the IF lives inside the WHILE region body, not at top-level).
 *  - Forward result at mHip=2.0 (non-clamping; trace below): 0.8.
 *  - Forward result at mHip=8.0 (clamping triggers at iter 2 since
 *    state = 0 → 0.8 → 1.6 > 1.5 → clamped to 1.5 → +0.8 = 2.3).
 *  - PhiCalculus.apply succeeds on this primal (validates the C5 unroll
 *    + multi-result IF coarsening surface).
 *
 * **Why no gradient pin**: §0.4.211 surfaced a `DxirReverseTransform`
 * **CSE bug** when applied to the QWOP hipUpdate primal post-coarsening:
 * the produced grad function references an undefined node id (e.g.,
 * `references unknown node ids: [16]`). The CSE pass at
 * `DxirReverseTransform.kt:466` is eliminating a node that's still
 * referenced by a STEP operand. This is a real bug in the AD pipeline
 * specific to the QWOP shape (a coarsened WHILE producing multiple
 * unrolled IF copies whose CSE deduplication misses an operand reference).
 * Per the /loop's "checkpoint, don't barrel forward with a hack" rule,
 * this firing ships the forward-only test + flags the AD bug as the
 * QWOP Phase 1 next-firing target.
 *
 * **The two-input forward pin** is the discriminator: verifies both the
 * non-clamping path and the clamping path of the collision-response IF
 * actually execute as designed. A "no IF" implementation would produce
 * `4 * 0.1 * 8.0 = 3.2` at mHip=8.0 instead of the correct 2.3.
 */
class QwopHipUpdateTest {

    @Test
    fun forwardEvalAtNonClampingInput() {
        val primal = Qwop.hipUpdatePrimal()

        // Verify primal structure: 1 WHILE (the muscle-integration loop).
        assertEquals(
            1,
            BenchmarkPrimals.countOps(primal, OpKind.WHILE),
            "hipUpdatePrimal: 1 WHILE in primal body (the integration loop)",
        )
        // The IF is region-internal (inside the WHILE body), so top-level countOps for IF is 0.
        assertEquals(
            0,
            BenchmarkPrimals.countOps(primal, OpKind.IF),
            "hipUpdatePrimal: 0 top-level IFs (the collision IF lives inside the WHILE region)",
        )

        // Forward eval (no AD): at mHip=2.0, final state = 4 * 0.1 * 2.0 = 0.8 < 1.5.
        // Trace:
        //   iter 0: state=0, IF(state>1.5) false → state = 0 + 2*0.1 = 0.2
        //   iter 1: state=0.2, IF false → state = 0.2 + 0.2 = 0.4
        //   iter 2: state=0.4, IF false → state = 0.4 + 0.2 = 0.6
        //   iter 3: state=0.6, IF false → state = 0.6 + 0.2 = 0.8
        // Final: 0.8.
        val outNonClamping = DxirInterpreter.evalFunction(primal, listOf(floatArrayOf(2.0f)))
        assertTrue(
            abs(outNonClamping[0][0] - 0.8f) < 1e-3f,
            "expected forward result = 0.8 at mHip=2.0 (non-clamping), got ${outNonClamping[0][0]}",
        )
    }

    @Test
    fun forwardEvalAtClampingInput() {
        val primal = Qwop.hipUpdatePrimal()

        // Forward eval at mHip=8.0 (clamping). Trace:
        //   iter 0: state=0, IF(state>1.5) false → state = 0 + 8*0.1 = 0.8
        //   iter 1: state=0.8 < 1.5, IF false → state = 0.8 + 0.8 = 1.6
        //   iter 2: state=1.6 > 1.5, IF true → state = 1.5 (clamped to maxAngle)
        //   iter 3: state=1.5 NOT > 1.5 (STEP(0)=0), IF false → state = 1.5 + 0.8 = 2.3
        // Final: 2.3.
        // A "no clamping" implementation would yield 4 * 0.1 * 8.0 = 3.2 instead.
        val outClamping = DxirInterpreter.evalFunction(primal, listOf(floatArrayOf(8.0f)))
        assertTrue(
            abs(outClamping[0][0] - 2.3f) < 1e-3f,
            "expected forward result = 2.3 at mHip=8.0 (clamping at iter 2), got ${outClamping[0][0]}",
        )
    }

    @Test
    fun phiCalculusApplySucceedsOnHipUpdatePrimal() {
        val primal = Qwop.hipUpdatePrimal()
        // PhiCalculus.apply should run cleanly on the primal — coarsens the WHILE
        // via C5 (constant trip count) + threads the IF through the unroll.
        // Asserts that the pipeline doesn't throw for this primal shape.
        val coarsened = PhiCalculus.apply(primal)
        // Coarsened body should NOT contain the original WHILE — C5 unrolls it.
        assertEquals(
            0,
            BenchmarkPrimals.countOps(coarsened, OpKind.WHILE),
            "After PhiCalculus.apply, the constant-trip-count WHILE should be unrolled (no top-level WHILE)",
        )
    }
}
