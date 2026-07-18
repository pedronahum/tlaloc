package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.354 — F64 through the full PJRT-CUDA path (track 3 of the
 * Software-2.0 gap list: dtype breadth). An all-F64 DXIR program is
 * emitted (`f64` tensors in StableHLO), compiled, and dispatched on the
 * GB10 with double-precision marshalling both ways.
 *
 * The precision pin is the point: `x·x + x` at `x = 1 + 1e-9` differs
 * from its f32 rounding in the 9th decimal — asserted against Kotlin
 * Double math at 1e-15 relative, two orders below anything f32 could
 * deliver.
 */
class PjrtF64SmokeTest {

    @Test
    fun runsAllF64ProgramAtDoublePrecision() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")

        val vec = DxirType(F64, listOf(4))
        val fn = DxirBuilder.function("f64_probe") {
            val x = param("x", vec)
            val sq = op(OpKind.MUL, listOf(x, x), vec)
            listOf(op(OpKind.ADD, listOf(sq, x), vec))
        }

        val inputs = doubleArrayOf(1.0 + 1e-9, -2.5, 3.141592653589793, 1e-12)
        val expected = inputs.map { it * it + it }

        val out: List<DoubleArray>
        PjrtSession(target = PjrtTarget.Cuda).use { session ->
            out = session.runOnF64(fn, listOf(inputs))
        }

        assertEquals(1, out.size)
        var maxRel = 0.0
        for (i in expected.indices) {
            val rel = abs(out[0][i] - expected[i]) / maxOf(abs(expected[i]), 1e-300)
            if (rel > maxRel) maxRel = rel
        }
        println("[pjrt-f64] x*x + x on GB10 f64: max rel=$maxRel vs Kotlin Double")
        assertTrue(maxRel <= 1e-15, "f64 path lost precision: max rel $maxRel")

        // And the f32 impossibility check: the first element's fractional
        // part is invisible at f32.
        val f32Rounded = (inputs[0].toFloat() * inputs[0].toFloat() + inputs[0].toFloat()).toDouble()
        assertTrue(
            abs(out[0][0] - expected[0]) < abs(f32Rounded - expected[0]),
            "result is no better than f32 rounding — f64 path suspect",
        )
    }
}
