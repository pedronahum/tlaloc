package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** §0.4.358 — [AttentionKernel] template decisions. */
class AttentionKernelTemplateTest {

    private fun coarsenedAttention(t: Int = 8, d: Int = 4): DxirOp {
        val qT = DxirType(F32, listOf(t, d))
        val ktT = DxirType(F32, listOf(d, t))
        val sT = DxirType(F32, listOf(t, t))
        val fn = DxirBuilder.function("attn") {
            val q = param("Q", qT)
            val k = param("K", ktT)
            val v = param("V", qT)
            val qk = op(OpKind.MATMUL, listOf(q, k), sT)
            val sm = op(OpKind.SOFTMAX, listOf(qk), sT)
            val out = op(OpKind.MATMUL, listOf(sm, v), qT)
            listOf(out)
        }
        val coarsened = coarsenRecognizedPatterns(fn, recognizeFlashAttention(fn))
        return coarsened.body.filterIsInstance<DxirOp>().single { it.op == OpKind.COARSENED }
    }

    @Test
    fun claimsFlashAttentionOnGb10WithScoreScratch() {
        val descriptor = AttentionKernel.pickFor(coarsenedAttention(), KernelTarget.NVIDIA_GB10)
        assertNotNull(descriptor)
        assertEquals("kptx_attention", descriptor.kernelName)
        assertTrue(descriptor.typedFfi)
        // Scratch = the T×T score matrix.
        assertEquals(listOf(listOf(8, 8)), descriptor.scratchResults)
    }

    @Test
    fun declinesNonGb10Targets() {
        val co = coarsenedAttention()
        assertNull(AttentionKernel.pickFor(co, KernelTarget.CPU_GENERIC))
        assertNull(AttentionKernel.pickFor(co, KernelTarget.NVIDIA_H100))
    }
}
