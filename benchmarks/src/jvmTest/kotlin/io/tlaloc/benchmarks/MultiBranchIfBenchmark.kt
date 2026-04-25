package io.tlaloc.benchmarks

import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.PhiCalculus
import kotlin.math.abs
import kotlin.system.measureTimeMillis
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.156 — fourth `:benchmarks` inhabitant. Exercises the multi-live-index
 * MR IF AD path that landed in §0.4.155 (Phase 5b): a primal whose result
 * references BOTH `ifop.result(0)` AND `ifop.result(1)` downstream, so the
 * gradient flow has to seed each branch's reverse walk at distinct terminator
 * slots and merge the per-index contributions back through `handleIfAdjoint`.
 *
 * Companion to the §0.4.145 / §0.4.146 / §0.4.147 / §0.4.148 inhabitants:
 *  - §0.4.145 [EndToEndAdBenchmark] — pipeline-wide on `iterateNTimes` (WHILE).
 *  - §0.4.146 [CoarseningThroughputBenchmark] — φ-only on `iterateNTimes`.
 *  - §0.4.147 [SctThroughputBenchmark] — SCT-only on `iterateNTimes`.
 *  - §0.4.156 [MultiBranchIfBenchmark] — pipeline on a multi-live-index MR IF
 *    primal (different shape from `iterateNTimes`; exercises the multi-live-index
 *    AD path the iterateN throughput sweeps don't reach).
 *
 * Different primal shape than the `iterateNTimes` family — three of the existing
 * inhabitants share the same WHILE-based primal at different scales. §0.4.156
 * widens the benchmark surface to a different IR shape (multi-result IF without
 * any WHILE), so a regression specific to one path doesn't slip past all four.
 */
class MultiBranchIfBenchmark {

    @Test
    fun adPipelineOnMultiBranchIfPrimal() {
        val primal = BenchmarkPrimals.multiBranchIfPrimal()

        // Pre-rewrite structure: 1 IF (the multi-result one), zero WHILEs.
        assertEquals(1, BenchmarkPrimals.countOps(primal, OpKind.IF), "1 multi-result IF")
        assertEquals(0, BenchmarkPrimals.countOps(primal, OpKind.WHILE), "no WHILEs in primal")

        // Time PhiCalculus.apply (no WHILEs to coarsen, so this is mostly a fixpoint
        // pass with no productive rewrites — a proxy for "what's the per-pass cost
        // when nothing fires?").
        var coarsened: io.tlaloc.ir.DxirFunction? = null
        val phiMs = measureTimeMillis {
            coarsened = PhiCalculus.apply(primal)
        }

        // Time DxirReverseTransform.apply on the coarsened (or pass-through) function.
        var grad: io.tlaloc.ir.DxirFunction? = null
        val sctMs = measureTimeMillis {
            grad = DxirReverseTransform.apply(coarsened!!)
        }

        // Numerical pins at three sample points covering both branches and the
        // boundary. The closed-form derivative:
        //   x > 0: f(x) = x² + 2x;  df/dx = 2x + 2.
        //   x ≤ 0: f(x) =  -x + x;  df/dx =  -1 + 1 = 0.
        val cases = listOf(
            5f to 12f,    // x = 5 (then-branch); df/dx = 2*5 + 2 = 12
            7f to 16f,    // x = 7 (then-branch); df/dx = 2*7 + 2 = 16
            -3f to 0f,    // x = -3 (else-branch); df/dx = 0
            0f to 0f,     // x = 0 (boundary; STEP(0) = 0 → else); df/dx = 0
        )
        for ((x, expected) in cases) {
            val out = DxirInterpreter.evalFunction(grad!!, listOf(floatArrayOf(x)))
            assertTrue(
                abs(out[0][0] - expected) < 1e-3f,
                "expected df/dx = $expected at x=$x, got ${out[0][0]}",
            )
        }

        // Side-channel timings.
        println("[bench multiBranchIf] phaseMs(PhiCalculus.apply)=$phiMs phaseMs(DxirReverseTransform.apply)=$sctMs")
    }
}
