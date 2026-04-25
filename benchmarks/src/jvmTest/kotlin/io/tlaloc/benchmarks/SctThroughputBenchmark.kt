package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.PhiCalculus
import kotlin.system.measureTimeMillis
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * §0.4.147 — third `:benchmarks` inhabitant. Times only [DxirReverseTransform.apply]
 * (i.e., the SCT pass — Source-to-Code Transformation reverse-mode AD) on
 * pre-coarsened functions, isolating SCT cost from PhiCalculus's. The probe
 * runs `PhiCalculus.apply` once outside the timing block to produce a
 * straight-line dxir, then sweeps `DxirReverseTransform.apply` at three
 * scales and pins the gradient evaluates to the closed-form derivative.
 *
 * Companion to:
 *  - §0.4.145's [EndToEndAdBenchmark] — pipeline-wide (φ + SCT + interp).
 *  - §0.4.146's [CoarseningThroughputBenchmark] — φ-only.
 *
 * Together the three benchmarks isolate the three regression-prone phases
 * (coarsening, SCT, interpreter eval) so a perf regression surfaces at the
 * scope it actually originated.
 */
class SctThroughputBenchmark {

    private val f32 = DxirType(F32, emptyList())
    private val i32 = DxirType(I32, emptyList())
    private val boolS = DxirType(io.tlaloc.core.Bool, emptyList())

    /**
     * Same shape as §0.4.145 / §0.4.146's helpers — duplicated per the
     * substrate's "stay structurally close" principle (§0.4.145, §0.4.146).
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
    fun dxirReverseTransformSweepOverIterateN() {
        // Pre-coarsen each primal once outside the timing block so the sweep
        // measures only the SCT cost (DxirReverseTransform.apply). The
        // coarsened function is straight-line dxir — n MULs and no WHILEs —
        // and DxirReverseTransform produces a gradient body that evaluates to
        // 2^n at any x (closed form of d/dx of `x · 2^n`).
        val sweepNs = listOf(5, 10, 20)
        val results = mutableMapOf<Int, Triple<Long, Int, Float>>()  // n → (sctMs, gradWhiles, gradEval)
        for (n in sweepNs) {
            val coarsened = PhiCalculus.apply(iterateNTimes(n))
            assertEquals(0, countOps(coarsened, OpKind.WHILE), "n=$n: coarsening should remove WHILE before timing SCT")
            var grad: io.tlaloc.ir.DxirFunction? = null
            val sctMs = measureTimeMillis {
                grad = DxirReverseTransform.apply(coarsened)
            }
            val gradOut = DxirInterpreter.evalFunction(grad!!, listOf(floatArrayOf(3f)))
            results[n] = Triple(sctMs, countOps(grad!!, OpKind.WHILE), gradOut[0][0])
        }
        // Pin: gradient evaluates to 2^n at any x (derivative of x · 2^n is 2^n).
        assertEquals((1 shl 5).toFloat(), results[5]!!.third, "n=5: expected 2^5 = 32")
        assertEquals((1 shl 10).toFloat(), results[10]!!.third, "n=10: expected 2^10 = 1024")
        assertEquals((1 shl 20).toFloat(), results[20]!!.third, "n=20: expected 2^20 = 1048576")
        // Pin: gradient body has no WHILEs (DxirReverseTransform doesn't introduce loops).
        for (n in sweepNs) {
            assertEquals(0, results[n]!!.second, "n=$n: gradient body should have zero WHILEs")
        }
        // Side-channel timings.
        for (n in sweepNs) {
            val (sctMs, _, gradEval) = results[n]!!
            println("[bench sctSweep] n=$n DxirReverseTransform.apply=${sctMs}ms grad@x=3=$gradEval")
        }
    }
}
