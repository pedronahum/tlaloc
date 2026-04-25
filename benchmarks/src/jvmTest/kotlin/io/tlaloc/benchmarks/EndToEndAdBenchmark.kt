package io.tlaloc.benchmarks

import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.PhiCalculus
import kotlin.system.measureTimeMillis
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.145 — first inhabitant of the `:benchmarks` Gradle module. Times the
 * complete AD pipeline (PhiCalculus φ-calculus rewrite → DxirReverseTransform
 * SCT → DxirInterpreter evaluation) on a hand-built 10-iteration WHILE primal
 * `f(x) = x · 2^10`. Numerical correctness is pinned (`d/dx = 1024` regardless
 * of `x`); per-phase timings print to stdout for visibility.
 *
 * §0.4.148 — `iterateNTimes` extracted to [BenchmarkPrimals]; this file
 * now just composes the pipeline phases around the shared primal.
 *
 * Goals of this module per §0.4.108's deferred register:
 *  - Provide a separate Gradle module so future perf work (HMC port, JAX/PyTorch
 *    bake-offs, latency regressions) has a home that's distinct from the regular
 *    test suite.
 *  - Mirror the §0.4.106 IR-size harness's structure: `assertEquals`-style
 *    pins for behaviour + side-channel `println` for human-readable timings.
 *  - Keep a low fixed-cost surface so adding new benchmarks is cheap.
 */
class EndToEndAdBenchmark {

    @Test
    fun adPipelineOnTenIterationLoop() {
        val n = 10
        val primal = BenchmarkPrimals.iterateNTimes(n)

        var coarsened: io.tlaloc.ir.DxirFunction? = null
        val phiMs = measureTimeMillis {
            coarsened = PhiCalculus.apply(primal)
        }
        val grad = DxirReverseTransform.apply(coarsened!!)
        val gradOut = DxirInterpreter.evalFunction(grad, listOf(floatArrayOf(3f)))

        // Numerical pin: d/dx of x · 2^n = 2^n.
        val expected = (1 shl n).toFloat()  // 2^10 = 1024
        assertEquals(expected, gradOut[0][0], "expected 2^$n = $expected, got ${gradOut[0][0]}")

        // Side-channel timings for visibility. Not asserted — perf budgets live
        // outside the test surface (per §0.4.106's IR-size harness convention).
        println("[bench iterateN AD] n=$n phaseMs(PhiCalculus.apply)=$phiMs grad=${gradOut[0][0]}")
    }
}
