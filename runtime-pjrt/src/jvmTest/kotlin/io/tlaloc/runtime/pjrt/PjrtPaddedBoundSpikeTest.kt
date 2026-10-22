package io.tlaloc.runtime.pjrt

import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * TorchTPU's padded form of a bounded dimension, on XLA:GPU: the program takes a static buffer
 * at the bound and an i32 size, marks the dimension dynamic with `stablehlo.set_dimension_size`,
 * and lets XLA's dynamic padder treat the tail as absent. The bounded-dims spike found the PJRT
 * C API cannot create a `#stablehlo.bounds` buffer; this form needs none. Each case puts garbage
 * in the padding, so a result that reads it is wrong.
 *
 * What the cases show is recorded in docs/work-log/torch-tpu-plan.md (item 13).
 */
class PjrtPaddedBoundSpikeTest {

    private val bound = 16
    private val n = 11
    private val x = FloatArray(bound) { if (it < n) (it - 5) * 0.25f else 1e6f }

    private fun run(body: String, resultType: String): FloatArray? {
        val text = """
            func.func @main(%x: tensor<${bound}xf32>, %n: tensor<i32>) -> $resultType {
              %d = stablehlo.set_dimension_size %x, %n, dim = 0 : (tensor<${bound}xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<$bound>>
            $body
            }
        """.trimIndent()
        return PjrtSession(target = PjrtTarget.Cuda, diskCache = null).use { s ->
            val xb = s.bufferFromHostF32(x, listOf(bound))
            val nb = s.bufferFromHostI32(intArrayOf(n), emptyList())
            try {
                runCatching {
                    val outs = s.executeStablehlo(text, listOf(xb, nb))
                    val size = if (resultType.contains("x")) bound else 1
                    outs[0].toFloatArray(size).also { outs.forEach { it.close() } }
                }.onFailure { println("[padded-bound] failed: ${it.message?.take(300)}") }.getOrNull()
            } finally {
                xb.close(); nb.close()
            }
        }
    }

    private fun report(case: String, got: FloatArray?, want: Float) {
        val ok = got != null && abs(got[0] - want) <= 1e-4f * maxOf(1f, abs(want))
        println("[padded-bound] $case: got ${got?.get(0)}, want $want -> ${if (ok) "honours the size" else "WRONG"}")
    }

    @Test
    fun whichOpsHonourTheSizeOnXlaGpu() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu || TestBackend.isMetal || TestBackend.isCerebras, "CUDA lane")
        val real = x.copyOf(n)
        val dyn = "tensor<?xf32, #stablehlo.bounds<$bound>>"

        report("sum", run("""
              %z = stablehlo.constant dense<0.0> : tensor<f32>
              %s = stablehlo.reduce(%d init: %z) applies stablehlo.add across dimensions = [0] : ($dyn, tensor<f32>) -> tensor<f32>
              return %s : tensor<f32>
        """, "tensor<f32>"), real.sum())

        report("max", run("""
              %z = stablehlo.constant dense<-1.0e30> : tensor<f32>
              %s = stablehlo.reduce(%d init: %z) applies stablehlo.maximum across dimensions = [0] : ($dyn, tensor<f32>) -> tensor<f32>
              return %s : tensor<f32>
        """, "tensor<f32>"), real.max())

        report("sum of squares (elementwise, then reduce)", run("""
              %q = stablehlo.multiply %d, %d : $dyn
              %z = stablehlo.constant dense<0.0> : tensor<f32>
              %s = stablehlo.reduce(%q init: %z) applies stablehlo.add across dimensions = [0] : ($dyn, tensor<f32>) -> tensor<f32>
              return %s : tensor<f32>
        """, "tensor<f32>"), real.sumOf { (it * it).toDouble() }.toFloat())

        report("dot with itself", run("""
              %s = stablehlo.dot_general %d, %d, contracting_dims = [0] x [0], precision = [HIGHEST, HIGHEST] : ($dyn, $dyn) -> tensor<f32>
              return %s : tensor<f32>
        """, "tensor<f32>"), real.sumOf { (it * it).toDouble() }.toFloat())

        val sm = run("""
              %z = stablehlo.constant dense<-1.0e30> : tensor<f32>
              %m = stablehlo.reduce(%d init: %z) applies stablehlo.maximum across dimensions = [0] : ($dyn, tensor<f32>) -> tensor<f32>
              %mb = stablehlo.broadcast_in_dim %m, dims = [] : (tensor<f32>) -> $dyn
              %sh = stablehlo.subtract %d, %mb : $dyn
              %e = stablehlo.exponential %sh : $dyn
              %z0 = stablehlo.constant dense<0.0> : tensor<f32>
              %t = stablehlo.reduce(%e init: %z0) applies stablehlo.add across dimensions = [0] : ($dyn, tensor<f32>) -> tensor<f32>
              %first = stablehlo.slice %e [0:1] : ($dyn) -> tensor<1xf32>
              %f = stablehlo.reshape %first : (tensor<1xf32>) -> tensor<f32>
              %p = stablehlo.divide %f, %t : tensor<f32>
              return %p : tensor<f32>
        """, "tensor<f32>")
        val mx = real.max()
        report("softmax[0]", sm, (exp((real[0] - mx).toDouble()) / real.sumOf { exp((it - mx).toDouble()) }).toFloat())
        assertTrue(true)
    }
}
