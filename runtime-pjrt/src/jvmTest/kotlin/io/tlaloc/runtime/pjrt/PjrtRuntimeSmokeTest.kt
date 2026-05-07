package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.302 — first-slice smoke for PjrtRuntime / runOnPjrt. Verifies the full
 * stack:
 *
 *   DxirBuilder.function → toStablehlo → pjrt_dispatch.py
 *                       → PJRT-XLA-CUDA / -CPU backend
 *                       → npy outputs
 *                       → FloatArray decode.
 *
 * Self-skips when python+jax aren't on host or the dispatcher script is
 * missing. CPU smoke runs whenever python+jax is available; CUDA smoke needs
 * an NVIDIA device.
 */
class PjrtRuntimeSmokeTest {

    private val f32Scalar = DxirType(F32, emptyList())
    private val v4 = DxirType(F32, listOf(4))

    private fun assertCloseToInterpreter(
        fn: io.tlaloc.ir.DxirFunction,
        inputs: List<FloatArray>,
        actual: List<FloatArray>,
        tol: Float = 1e-5f,
    ) {
        val expected = DxirInterpreter.evalFunction(fn, inputs)
        assertEquals(expected.size, actual.size, "result count mismatch")
        for (i in expected.indices) {
            assertEquals(expected[i].size, actual[i].size, "result[$i] size mismatch")
            for (j in expected[i].indices) {
                val diff = abs(expected[i][j] - actual[i][j])
                assertTrue(
                    diff <= tol,
                    "result[$i][$j] disagreement: interpreter=${expected[i][j]} pjrt=${actual[i][j]} diff=$diff (tol=$tol)",
                )
            }
        }
    }

    private fun scalarAffine(): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("scalar_affine") {
            val x = param("x", f32Scalar)
            val two = const(2.0f, f32Scalar)
            val one = const(1.0f, f32Scalar)
            val mul = op(OpKind.MUL, listOf(x, two), f32Scalar)
            val add = op(OpKind.ADD, listOf(mul, one), f32Scalar)
            listOf(add)
        }

    private fun rank1Triple(): io.tlaloc.ir.DxirFunction =
        DxirBuilder.function("rank1_triple") {
            val v = param("v", v4)
            val twoV = op(OpKind.ADD, listOf(v, v), v4)
            val threeV = op(OpKind.ADD, listOf(twoV, v), v4)
            listOf(threeV)
        }

    @Test
    fun runOnPjrtCpuMatchesInterpreterOnScalarAffine() {
        assumeTrue(PjrtBinaries.available, "python+jax not resolved — skipping CPU smoke.")
        val fn = scalarAffine()
        val inputs = listOf(floatArrayOf(3.5f))  // expected: 2 * 3.5 + 1 = 8.0
        val out = runOnPjrt(fn, inputs, target = PjrtTarget.LlvmCpu)
        assertCloseToInterpreter(fn, inputs, out)
    }

    @Test
    fun runOnPjrtCpuMatchesInterpreterOnRank1Triple() {
        assumeTrue(PjrtBinaries.available, "python+jax not resolved — skipping CPU smoke.")
        val fn = rank1Triple()
        val inputs = listOf(floatArrayOf(1f, 2f, -3f, 4f))  // expected: [3, 6, -9, 12]
        val out = runOnPjrt(fn, inputs, target = PjrtTarget.LlvmCpu)
        assertCloseToInterpreter(fn, inputs, out)
    }

    @Test
    fun runOnPjrtCudaMatchesInterpreterOnScalarAffine() {
        assumeTrue(PjrtBinaries.available, "python+jax not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device visible to JAX — skipping CUDA smoke.")
        val fn = scalarAffine()
        val inputs = listOf(floatArrayOf(7.0f))  // expected: 2 * 7 + 1 = 15.0
        val out = runOnPjrt(fn, inputs, target = PjrtTarget.Cuda)
        assertCloseToInterpreter(fn, inputs, out)
    }

    @Test
    fun runOnPjrtCudaMatchesInterpreterOnRank1Triple() {
        assumeTrue(PjrtBinaries.available, "python+jax not resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device visible to JAX — skipping CUDA smoke.")
        val fn = rank1Triple()
        val inputs = listOf(floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f))  // expected: [1.5, 4.5, 7.5, 10.5]
        val out = runOnPjrt(fn, inputs, target = PjrtTarget.Cuda)
        assertCloseToInterpreter(fn, inputs, out)
    }

    @Test
    fun pjrtModuleRemembersTarget() {
        val module = PjrtRuntime.compile(
            // Smallest valid StableHLO so this test stays binary-free —
            // doesn't actually invoke, just checks the handle's target.
            "module @t { func.func @main() -> () { return } }",
            target = PjrtTarget.Cuda,
        )
        assertEquals(PjrtTarget.Cuda, module.target)
    }
}
