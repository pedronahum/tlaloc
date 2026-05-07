package io.tlaloc.runtime.pjrt

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
 * §0.4.305 — replaces the §0.4.302 PjrtRuntimeSmokeTest. Same coverage
 * (scalar affine + rank-1 elementwise, agreement vs DxirInterpreter at 1e-5)
 * but the underlying dispatch path is now FFM, not Python subprocess.
 *
 * Self-skips when the PJRT plugin isn't resolvable. CPU and CUDA share the
 * same bundled plugin (`xla_cuda_plugin.so` registers both backends); the
 * CUDA arms additionally require an NVIDIA GPU on the host.
 */
class RunOnPjrtSmokeTest {

    private val f32Scalar = DxirType(F32, emptyList())
    private val v4 = DxirType(F32, listOf(4))

    private fun assertCloseToInterpreter(
        fn: io.tlaloc.ir.DxirFunction,
        inputs: List<FloatArray>,
        actual: List<FloatArray>,
        tol: Float = 1e-5f,
    ) {
        val expected = DxirInterpreter.evalFunction(fn, inputs)
        assertTrue(expected.size == actual.size, "result count mismatch")
        for (i in expected.indices) {
            assertTrue(expected[i].size == actual[i].size, "result[$i] size mismatch")
            for (j in expected[i].indices) {
                val diff = abs(expected[i][j] - actual[i][j])
                assertTrue(
                    diff <= tol,
                    "result[$i][$j] disagreement: interpreter=${expected[i][j]} pjrt=${actual[i][j]} diff=$diff (tol=$tol)",
                )
            }
        }
    }

    private fun scalarAffine() = DxirBuilder.function("scalar_affine") {
        val x = param("x", f32Scalar)
        val two = const(2.0f, f32Scalar)
        val one = const(1.0f, f32Scalar)
        val mul = op(OpKind.MUL, listOf(x, two), f32Scalar)
        val add = op(OpKind.ADD, listOf(mul, one), f32Scalar)
        listOf(add)
    }

    private fun rank1Triple() = DxirBuilder.function("rank1_triple") {
        val v = param("v", v4)
        val twoV = op(OpKind.ADD, listOf(v, v), v4)
        val threeV = op(OpKind.ADD, listOf(twoV, v), v4)
        listOf(threeV)
    }

    @Test
    fun runOnPjrtCudaMatchesInterpreterOnScalarAffine() {
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = scalarAffine()
        val inputs = listOf(floatArrayOf(7.0f))  // 2*7 + 1 = 15
        val out = runOnPjrt(fn, inputs, target = PjrtTarget.Cuda)
        assertCloseToInterpreter(fn, inputs, out)
    }

    @Test
    fun runOnPjrtCudaMatchesInterpreterOnRank1Triple() {
        assumeTrue(PjrtBinaries.available, "PJRT plugin not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val fn = rank1Triple()
        val inputs = listOf(floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f))  // [1.5, 4.5, 7.5, 10.5]
        val out = runOnPjrt(fn, inputs, target = PjrtTarget.Cuda)
        assertCloseToInterpreter(fn, inputs, out)
    }
}
