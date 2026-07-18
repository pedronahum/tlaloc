package io.tlaloc.runtime.pjrt

import io.tlaloc.core.Bool
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.360 — the new/activated shape-plumbing lowerings certified
 * against real XLA: one program exercising COMPARE → WHERE, SLICE,
 * PAD, and CONCAT compiles through the emitter's new/dormant MLIR
 * spellings (`stablehlo.compare/select/slice/pad/concatenate`) and
 * executes on the GB10, agreeing with the interpreter.
 */
class PjrtShapePlumbingSmokeTest {

    @Test
    fun shapePlumbingProgramRunsOnGpuAndMatchesInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val t = DxirType(F32, listOf(2, 4))
        val bT = DxirType(Bool, listOf(2, 4))
        val fn = DxirBuilder.function("plumbing") {
            val a = param("a", t)
            val b = param("b", t)
            // Elementwise max via compare+where…
            val m = op(OpKind.COMPARE, listOf(a, b), bT, attrs = mapOf("direction" to "GT"))
            val mx = op(OpKind.WHERE, listOf(m, a, b), t)
            // …slice a window, pad it back out, concat with the max.
            val sl = op(
                OpKind.SLICE, listOf(mx), DxirType(F32, listOf(2, 2)),
                attrs = mapOf(
                    "start_indices" to listOf(0, 1),
                    "limit_indices" to listOf(2, 3),
                    "strides" to listOf(1, 1),
                ),
            )
            val pd = op(
                OpKind.PAD, listOf(sl), t,
                attrs = mapOf("low" to listOf(0, 1), "high" to listOf(0, 1)),
            )
            listOf(op(OpKind.CONCAT, listOf(mx, pd), DxirType(F32, listOf(2, 8)), attrs = mapOf("dimension" to 1)))
        }

        val rng = java.util.Random(7)
        val a = FloatArray(8) { rng.nextFloat() * 2f - 1f }
        val b = FloatArray(8) { rng.nextFloat() * 2f - 1f }
        val want = DxirInterpreter.evalFunction(fn, listOf(a, b)).single()

        val got: FloatArray
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            got = session.runOn(fn, listOf(a, b)).single()
        }

        var maxAbs = 0f
        for (i in got.indices) maxAbs = maxOf(maxAbs, abs(got[i] - want[i]))
        println("[pjrt-plumbing] compare/where/slice/pad/concat on GB10 vs interpreter: max|diff|=$maxAbs")
        assertTrue(maxAbs == 0f, "shape plumbing diverges from interpreter: $maxAbs")
    }
}
