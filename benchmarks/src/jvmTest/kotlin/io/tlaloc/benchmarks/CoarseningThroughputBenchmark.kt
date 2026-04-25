package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
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
 * Mirrors §0.4.145's [EndToEndAdBenchmark] convention: structured
 * `assertEquals` pins for behaviour, side-channel `println` for ms timings.
 * Together the two benchmarks let a /loop iteration spot-check the
 * coarsening throughput at three representative scales.
 */
class CoarseningThroughputBenchmark {

    private val f32 = DxirType(F32, emptyList())
    private val i32 = DxirType(I32, emptyList())
    private val boolS = DxirType(io.tlaloc.core.Bool, emptyList())

    /**
     * Same shape as §0.4.145's helper but copied here to keep the benchmark
     * self-contained — the `:benchmarks` module's role is to host probes that
     * stay structurally close to the IR/AD substrate they exercise; sharing
     * helpers across benchmarks would couple them in ways that obscure each
     * probe's specific scope.
     */
    private fun iterateNTimes(n: Int): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("iterate$n") {
            val x = param("x", f32)
            val nConst = const(n, i32)
            val zero = const(0, i32)
            val w = whileOp(
                inits = listOf(x, zero),
                cond = { args ->
                    val diff = op(OpKind.SUB, listOf(nConst, args[1]), i32)
                    val pred = op(OpKind.STEP, listOf(diff), boolS)
                    yields(pred)
                },
                body = { args ->
                    val two = const(2f, f32)
                    val newX = op(OpKind.MUL, listOf(args[0], two), f32)
                    val one = const(1, i32)
                    val newI = op(OpKind.ADD, listOf(args[1], one), i32)
                    yields(newX, newI)
                },
            )
            listOf(w.result(0))
        }

    private fun countOps(fn: io.tlaloc.ir.DxirFunction, kind: OpKind): Int =
        fn.body.filterIsInstance<DxirOp>().count { it.op == kind }

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
            val primal = iterateNTimes(n)
            var rewritten: io.tlaloc.ir.DxirFunction? = null
            val ms = measureTimeMillis {
                rewritten = PhiCalculus.apply(primal)
            }
            assertEquals(0, countOps(rewritten!!, OpKind.WHILE), "n=$n: WHILE should be unrolled")
            val muls = countOps(rewritten!!, OpKind.MUL)
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
