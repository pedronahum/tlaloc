package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.recognizeRmsNorm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * KPTX v1.8 (§0.4.336) — [RmsNormKernel] template decision tests.
 * Mirrors the [RmsNormCoarsenerTest] fixture shapes; the template's
 * contract is "GB10 + eps-form + rank-2 x, else decompose".
 */
class RmsNormKernelTemplateTest {

    private val xType = DxirType(F32, listOf(8, 64))
    private val mType = DxirType(F32, listOf(8, 1))
    private val epsType = DxirType(F32, listOf(8, 1))   // matches mean's keepdims shape (the LlamaDecoder form)

    private fun coarsenedRmsNorm(withEps: Boolean): DxirOp {
        val fn = DxirBuilder.function("rms_norm") {
            val x = param("x", xType)
            val sq = op(OpKind.MUL, listOf(x, x), xType)
            val mean = op(OpKind.MEAN, listOf(sq), mType)
            val rsqrtIn = if (withEps) {
                val eps = param("eps", epsType)
                op(OpKind.ADD, listOf(mean, eps), mType)
            } else mean
            val r = op(OpKind.RSQRT, listOf(rsqrtIn), mType)
            val out = op(OpKind.MUL, listOf(x, r), xType)
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRmsNorm(fn))
        return coarsened.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
    }

    @Test
    fun claimsEpsFormOnGb10WithTypedFfiDescriptor() {
        val descriptor = RmsNormKernel.pickFor(coarsenedRmsNorm(withEps = true), KernelTarget.NVIDIA_GB10)
        assertNotNull(descriptor)
        assertEquals("kptx_rms_norm", descriptor.kernelName)
        assertEquals("tlaloc", descriptor.vendor)
        assertTrue(descriptor.typedFfi, "KPTX kernels dispatch via the typed-FFI convention (§0.4.332)")
    }

    @Test
    fun declinesNonGb10Targets() {
        val co = coarsenedRmsNorm(withEps = true)
        assertNull(RmsNormKernel.pickFor(co, KernelTarget.CPU_GENERIC))
        assertNull(RmsNormKernel.pickFor(co, KernelTarget.NVIDIA_H100))
        assertNull(RmsNormKernel.pickFor(co, KernelTarget.GOOGLE_TPU_V6E))
    }

    @Test
    fun declinesNoEpsForm() {
        // Kernel signature is (x_ptr, eps_ptr, out_ptr, n_cols) — the
        // no-eps COARSENED has one operand and must decompose.
        assertNull(RmsNormKernel.pickFor(coarsenedRmsNorm(withEps = false), KernelTarget.NVIDIA_GB10))
    }
}
