package io.tlaloc.benchmarks

import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.PhiCalculus
import kotlin.system.measureTimeMillis
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.146 — second `:benchmarks` inhabitant. Times [PhiCalculus.apply] across
 * progressively larger `iterateNTimes` primals (n = 5, 10, 20) and pins
 * post-coarsening op counts so a future regression in the C5 unroll path
 * fires here before it surfaces in correctness tests.
 *
 * §0.4.148 — `iterateNTimes` extracted to [BenchmarkPrimals]; this file
 * now just times the φ-calculus pass around the shared primal.
 *
 * Mirrors §0.4.145's [EndToEndAdBenchmark] convention: structured
 * `assertEquals` pins for behaviour, side-channel `println` for ms timings.
 * Together the benchmarks let a /loop iteration spot-check the coarsening
 * throughput at three representative scales.
 */
class CoarseningThroughputBenchmark {

    @Test
    fun phiCalculusCoarseningSweepOverIterateN() {
        // C5 unrolls iterateNTimes(n)'s carried-back-edge `MUL(arg, two)` n
        // times, leaving n MULs in the body and zero WHILEs. The counter slot's
        // ADD(arg, one) chain is also unrolled n times, but the existing CSE
        // pipeline collapses dead counter contributions when the user only
        // references the carried result (w.result(0)). Pin the MUL count and
        // WHILE absence at three n values; print per-iter ms.
        val sweepNs = listOf(5, 10, 20)
        val results = mutableMapOf<Int, Pair<Long, Int>>()  // n → (ms, mulCount)
        for (n in sweepNs) {
            val primal = BenchmarkPrimals.iterateNTimes(n)
            var rewritten: io.tlaloc.ir.DxirFunction? = null
            val ms = measureTimeMillis {
                rewritten = PhiCalculus.apply(primal)
            }
            assertEquals(0, BenchmarkPrimals.countOps(rewritten!!, OpKind.WHILE), "n=$n: WHILE should be unrolled")
            val muls = BenchmarkPrimals.countOps(rewritten!!, OpKind.MUL)
            results[n] = ms to muls
        }
        // Pin: MUL count tracks n exactly (one per unrolled iteration).
        assertEquals(5, results[5]!!.second, "n=5 should produce 5 MULs")
        assertEquals(10, results[10]!!.second, "n=10 should produce 10 MULs")
        assertEquals(20, results[20]!!.second, "n=20 should produce 20 MULs")
        // Side-channel timings.
        for (n in sweepNs) {
            val (ms, muls) = results[n]!!
            println("[bench coarseningSweep] n=$n PhiCalculus.apply=${ms}ms muls=$muls")
        }
    }
}
