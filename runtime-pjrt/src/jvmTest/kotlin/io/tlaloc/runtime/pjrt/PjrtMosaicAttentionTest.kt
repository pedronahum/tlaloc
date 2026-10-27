package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.stablehlo.mosaic.MosaicAttention.causalFlashAttention
import io.tlaloc.testing.Tolerances
import org.junit.jupiter.api.Assumptions.assumeFalse
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test

/**
 * The gradient of a loss through [causalFlashAttention] off a TPU: the session replaces each
 * Mosaic kernel by its reference, and the device's results match the interpreter's.
 */
class PjrtMosaicAttentionTest {

    @Test
    fun theFlashAttentionGradientRunsOffTpu() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        assumeFalse(TestBackend.isTpu, "on a TPU the kernels run; PjrtTpuMosaicKernelTest's lane")
        val seq = 256
        val d = 128
        val f = DxirType(F32, listOf(seq, d))
        val b = DxirType(BF16, listOf(seq, d))
        val fn = DxirBuilder.function("attn_loss") {
            val (q, k, v) = listOf("q", "k", "v").map { op(OpKind.CAST, listOf(param(it, f)), b) }
            val w = param("w", f)
            val o = op(OpKind.CAST, listOf(causalFlashAttention(q, k, v)), f)
            listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(o, w), f)), DxirType(F32, emptyList())))
        }
        val grad = DxirReverseTransform.apply(fn, includeForward = true, inputOnlyTrailingParams = 1)
        val r = Random(4)
        val inputs = List(4) { FloatArray(seq * d) { r.nextFloat() - 0.5f } }
        val want = DxirInterpreter.evalFunction(grad, inputs)
        val got = TestBackend.session().use { it.runOn(grad, inputs) }
        for (i in want.indices) {
            Tolerances.assertClose(DoubleArray(got[i].size) { got[i][it].toDouble() }, DoubleArray(want[i].size) { want[i][it].toDouble() }, 2e-2, "result $i")
        }
    }
}
