package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.recognizeCrossEntropy
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.351 — [CrossEntropyKernel] template decisions: claims the
 * scalar-loss CE COARSENED on GB10 with the launch-chain scratch shape
 * derived from the logits row count; declines elsewhere.
 */
class CrossEntropyKernelTemplateTest {

    private val tLogits = DxirType(F32, listOf(8, 32))
    private val tScalar = DxirType(F32, emptyList())

    private fun coarsenedCe(): DxirOp {
        val fn = DxirBuilder.function("ce") {
            val logits = param("logits", tLogits)
            val labels = param("labels", tLogits)
            val probs = op(OpKind.SOFTMAX, listOf(logits), tLogits)
            val logp = op(OpKind.LOG, listOf(probs), tLogits)
            val pw = op(OpKind.MUL, listOf(labels, logp), tLogits)
            val loss = op(OpKind.SUM, listOf(pw), tScalar)
            listOf(loss)
        }
        val coarsened = coarsenRecognizedPatterns(fn, recognizeCrossEntropy(fn))
        return coarsened.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
    }

    @Test
    fun claimsScalarCeOnGb10WithRowScratch() {
        val descriptor = CrossEntropyKernel.pickFor(coarsenedCe(), KernelTarget.NVIDIA_GB10)
        assertNotNull(descriptor)
        assertEquals("kptx_cross_entropy", descriptor.kernelName)
        assertTrue(descriptor.typedFfi)
        // Scratch = row_loss[rows], rows from the logits shape.
        assertEquals(listOf(listOf(8)), descriptor.scratchResults)
    }

    @Test
    fun declinesNonGb10Targets() {
        val co = coarsenedCe()
        assertNull(CrossEntropyKernel.pickFor(co, KernelTarget.CPU_GENERIC))
        assertNull(CrossEntropyKernel.pickFor(co, KernelTarget.NVIDIA_H100))
    }
}
