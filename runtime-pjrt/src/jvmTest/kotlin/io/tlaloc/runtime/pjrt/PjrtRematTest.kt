package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirRemat
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.stablehlo.toStablehlo
import io.tlaloc.testing.Tolerances
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Rematerialization ([DxirRemat]) on the device: the gradient of a stack of MLP blocks, with and
 * without remat ([DxirRemat.SAVE_MARKED], block outputs kept), each in its own session. The
 * gradients agree, and the device's peak memory falls, because the wide activation inside each
 * block is recomputed during the backward pass instead of kept from the forward pass.
 */
class PjrtRematTest {

    private val batch = 4096
    private val width = 256
    private val hidden = 4 * width
    private val layers = 12

    /** [layers] blocks `h ← tanh(tanh(h·W_up)·W_down)`, each block's output marked as a checkpoint. */
    private fun mlp(): DxirFunction = DxirBuilder.function("deep_mlp") {
        val x = param("x", DxirType(F32, listOf(batch, width)))
        val ws = (1..layers).map { param("up$it", DxirType(F32, listOf(width, hidden))) to param("down$it", DxirType(F32, listOf(hidden, width))) }
        val t = DxirType(F32, listOf(batch, width))
        val wide = DxirType(F32, listOf(batch, hidden))
        var h: DxirNode = x
        for ((up, down) in ws) {
            val a = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(h, up), wide)), wide)
            h = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(a, down), t)), t, attrs = mapOf(DxirRemat.SAVE_ATTR to true))
        }
        listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(h, h), t)), DxirType(F32, emptyList())))
    }

    private fun peak(fn: DxirFunction, inputs: List<FloatArray>): Pair<List<FloatArray>, Long> =
        TestBackend.session().use { s ->
            val out = s.runOn(fn, inputs)
            out to (s.memoryStats().second ?: error("the plugin reports no peak"))
        }

    @Test
    fun rematLowersPeakMemoryAndKeepsTheGradient() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val r = Random(3)
        val inputs = listOf(FloatArray(batch * width) { r.nextFloat() - 0.5f }) +
            List(layers) { listOf(FloatArray(width * hidden) { (r.nextFloat() - 0.5f) * 0.1f }, FloatArray(hidden * width) { (r.nextFloat() - 0.5f) * 0.1f }) }.flatten()
        val grad = DxirReverseTransform.apply(mlp(), includeForward = true)
        val remat = DxirRemat.apply(grad, DxirRemat.SAVE_MARKED)
        assertTrue("stablehlo.optimization_barrier" in remat.toStablehlo())

        val (plain, plainPeak) = peak(grad, inputs)
        val (re, rematPeak) = peak(remat, inputs)
        for (i in plain.indices) {
            Tolerances.assertClose(
                DoubleArray(re[i].size) { re[i][it].toDouble() },
                DoubleArray(plain[i].size) { plain[i][it].toDouble() },
                1e-4,
                "result $i",
            )
        }
        val mib = 1024.0 * 1024.0
        println("PjrtRematTest: peak ${"%.0f".format(plainPeak / mib)} MiB plain, ${"%.0f".format(rematPeak / mib)} MiB with remat")
        // Without remat each block keeps its [batch, hidden] activation (16 MiB) for the backward
        // pass; with it only the [batch, width] block outputs (4 MiB) are kept.
        assertTrue(rematPeak < plainPeak * 0.75, "remat peak $rematPeak should be well below $plainPeak")
    }
}
