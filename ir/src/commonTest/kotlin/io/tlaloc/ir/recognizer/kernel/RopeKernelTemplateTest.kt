package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.recognizeRope
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.349 — [RopeKernel] template decisions: claims the SUB-form /
 * cos-first COARSENED (the LlamaDecoder shape the KPTX kernel
 * implements) on GB10; declines the ADD form and non-GB10 targets.
 */
class RopeKernelTemplateTest {

    private val t = DxirType(F32, listOf(8, 64))

    private fun coarsenedRope(subForm: Boolean): DxirOp {
        val fn = DxirBuilder.function("rope") {
            val q = param("q", t)
            val theta = param("theta", t)
            val cosT = op(OpKind.COS, listOf(theta), t)
            val sinT = op(OpKind.SIN, listOf(theta), t)
            val real = op(OpKind.MUL, listOf(q, cosT), t)
            val imag = op(OpKind.MUL, listOf(theta, sinT), t)
            val out = op(if (subForm) OpKind.SUB else OpKind.ADD, listOf(real, imag), t)
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(fn, recognizeRope(fn))
        return coarsened.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
    }

    @Test
    fun claimsSubFormOnGb10() {
        val descriptor = RopeKernel.pickFor(coarsenedRope(subForm = true), KernelTarget.NVIDIA_GB10)
        assertNotNull(descriptor)
        assertEquals("kptx_rope", descriptor.kernelName)
        assertTrue(descriptor.typedFfi)
    }

    @Test
    fun declinesAddForm() {
        // The kernel computes real·cos − imag·sin; the ADD recombination
        // must decompose rather than run the wrong math.
        assertNull(RopeKernel.pickFor(coarsenedRope(subForm = false), KernelTarget.NVIDIA_GB10))
    }

    @Test
    fun declinesNonGb10Targets() {
        val co = coarsenedRope(subForm = true)
        assertNull(RopeKernel.pickFor(co, KernelTarget.CPU_GENERIC))
        assertNull(RopeKernel.pickFor(co, KernelTarget.NVIDIA_H100))
    }
}
