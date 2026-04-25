package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
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
 * Goals of this module per §0.4.108's deferred register:
 *  - Provide a separate Gradle module so future perf work (HMC port, JAX/PyTorch
 *    bake-offs, latency regressions) has a home that's distinct from the regular
 *    test suite.
 *  - Mirror the §0.4.106 IR-size harness's structure: `assertEquals`-style
 *    pins for behaviour + side-channel `println` for human-readable timings.
 *  - Keep a low fixed-cost surface so adding new benchmarks is cheap.
 */
class EndToEndAdBenchmark {

    private val f32 = DxirType(F32, emptyList())
    private val i32 = DxirType(I32, emptyList())
    private val boolS = DxirType(io.tlaloc.core.Bool, emptyList())

    /**
     * The reference primal: `f(x) = x · 2^N` via an `N`-iteration WHILE loop
     * whose body doubles the carried `x` each iter and increments a counter.
     * After [PhiCalculus.apply] unrolls C5, the gradient is a chain of `N`
     * MULs whose closed-form derivative is `2^N`.
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

    @Test
    fun adPipelineOnTenIterationLoop() {
        val n = 10
        val primal = iterateNTimes(n)

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
