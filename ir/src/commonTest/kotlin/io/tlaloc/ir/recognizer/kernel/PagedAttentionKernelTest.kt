package io.tlaloc.ir.recognizer.kernel

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * §0.4.471 — Phase H4: the claiming half, offline.
 *
 * Two properties, and they are the whole contract of the inference lane:
 *
 * 1. The template claims exactly the shape the KPTX chain can run, and
 *    declines everything else by returning null rather than throwing —
 *    a decline is a normal outcome (it is what every non-GB10 machine
 *    gets), not an error.
 * 2. `lowerKernelChoice`'s inference lane ANNOTATES and never decomposes.
 *    An unclaimed `PAGED_ATTENTION` comes back structurally identical,
 *    because the op carries its own lowering; a claimed one comes back as
 *    the same op with one extra attr. This is the property that makes the
 *    KPTX swap local — H1a's whole reason for making paged attention a
 *    coarse kind.
 */
class PagedAttentionKernelTest {

    private val numSeqs = 2
    private val numHeads = 4
    private val numKvHeads = 2
    private val headDim = 3
    private val blockSize = 2
    private val numBlocks = 6
    private val maxBlocksPerSeq = 3

    private val qType = DxirType(F32, listOf(numSeqs, numHeads, headDim))
    private val cacheType = DxirType(F32, listOf(numBlocks, blockSize, numKvHeads, headDim))
    private val tableType = DxirType(I32, listOf(numSeqs, maxBlocksPerSeq))
    private val lensType = DxirType(I32, listOf(numSeqs))

    private fun pagedFn(
        qT: DxirType = qType,
        cacheT: DxirType = cacheType,
        attrs: Map<String, Any> = mapOf("scale" to 0.5),
    ): DxirFunction = DxirBuilder.function("paged") {
        val q = param("q", qT)
        val k = param("k", cacheT)
        val v = param("v", cacheT)
        val t = param("t", tableType)
        val l = param("l", lensType)
        listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qT, attrs))
    }

    private fun pagedOp(fn: DxirFunction): DxirOp =
        fn.body.filterIsInstance<DxirOp>().single { it.op == OpKind.PAGED_ATTENTION }

    @Test
    fun claimsTheDecodeShapeOnGb10WithTheScoreMatrixAsScratch() {
        val d = PagedAttentionKernel.pickFor(pagedOp(pagedFn()), KernelTarget.NVIDIA_GB10)
        assertNotNull(d, "the GB10 decode shape is exactly what the KPTX chain runs")
        assertEquals("kptx_paged_attention", d.kernelName)
        assertEquals("tlaloc", d.vendor)
        assertTrue(d.typedFfi, "KPTX dispatch is the typed-FFI convention (§0.4.332)")
        // The inter-stage score matrix: one row per (sequence, query head),
        // one column per PADDED context lane. Both are derived from operand
        // shapes — never from an attr.
        assertEquals(
            listOf(listOf(numSeqs * numHeads, maxBlocksPerSeq * blockSize)),
            d.scratchResults,
        )
        assertEquals(0.5, d.customCallAttrs["scale"])
    }

    @Test
    fun declinesEveryTargetButTheGb10() {
        val others = listOf(
            KernelTarget.NVIDIA_H100, KernelTarget.NVIDIA_A100, KernelTarget.NVIDIA_B200,
            KernelTarget.AMD_MI300X, KernelTarget.GOOGLE_TPU_V5E, KernelTarget.AWS_TRAINIUM2,
            KernelTarget.CPU_GENERIC,
        )
        for (t in others) {
            assertNull(
                PagedAttentionKernel.pickFor(pagedOp(pagedFn()), t),
                "KPTX is NVIDIA-GB10-only by construction; $t must fall back to the emission",
            )
        }
    }

    @Test
    fun declinesBf16PoolsRatherThanClaimingAnF32Kernel() {
        val bq = DxirType(BF16, listOf(numSeqs, numHeads, headDim))
        val bc = DxirType(BF16, listOf(numBlocks, blockSize, numKvHeads, headDim))
        assertNull(
            PagedAttentionKernel.pickFor(pagedOp(pagedFn(bq, bc)), KernelTarget.NVIDIA_GB10),
            "the v1 chain is f32 loops; bf16 pools are a named deferral, not a silent miscompute",
        )
    }

    @Test
    fun declinesRatherThanThrowsOnAnIllegalPagedAttention() {
        // No `scale` attr at all — PagedAttentionAttrs.parse refuses it.
        // A template must DECLINE on that, not propagate the failure: the
        // claiming pass runs over whole graphs and a malformed op is the
        // verifier's business, not the kernel picker's.
        val fn = DxirBuilder.function("bad") {
            val q = param("q", qType)
            val k = param("k", cacheType)
            val v = param("v", cacheType)
            val t = param("t", tableType)
            val l = param("l", lensType)
            listOf(op(OpKind.PAGED_ATTENTION, listOf(q, k, v, t, l), qType, emptyMap()))
        }
        assertNull(PagedAttentionKernel.pickFor(pagedOp(fn), KernelTarget.NVIDIA_GB10))
    }

    @Test
    fun declinesEveryOtherOpKind() {
        val fn = DxirBuilder.function("add") {
            val a = param("a", qType)
            listOf(op(OpKind.ADD, listOf(a, a), qType))
        }
        val addOp = fn.body.filterIsInstance<DxirOp>().single()
        assertNull(PagedAttentionKernel.pickFor(addOp, KernelTarget.NVIDIA_GB10))
    }

    @Test
    fun loweringAnnotatesTheClaimedOpInPlace() {
        val lowered = lowerKernelChoice(
            pagedFn(),
            KernelTarget.NVIDIA_GB10,
            inferenceRegistry = kptxInferenceKernelTemplates,
        )
        val op = pagedOp(lowered)
        // Same kind, same type, same operand count — ONLY an added attr.
        assertEquals(OpKind.PAGED_ATTENTION, op.op)
        assertEquals(qType, op.type)
        assertEquals(5, op.operands.size)
        assertEquals(0.5, op.attrs["scale"], "the op's own attrs survive annotation")
        val d = op.attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        assertNotNull(d)
        assertEquals("kptx_paged_attention", d.kernelName)
    }

    @Test
    fun theDefaultRegistryClaimsNothingAndLeavesTheGraphAlone() {
        // The default inference registry is empty on purpose: claiming emits
        // a custom_call naming a kernel some runtime must have registered.
        assertTrue(defaultInferenceKernelTemplates.isEmpty())
        val fn = pagedFn()
        val lowered = lowerKernelChoice(fn, KernelTarget.NVIDIA_GB10)
        assertTrue(
            KernelDescriptor.ATTR_KEY !in pagedOp(lowered).attrs,
            "an unclaimed PAGED_ATTENTION must emit its own gather-composed lowering",
        )
        // And the pass is a genuine no-op: same identity, not a rebuild.
        assertTrue(lowered === fn, "nothing claimed ⇒ nothing rebuilt")
    }

    @Test
    fun aNonGb10TargetLeavesTheGraphAloneEvenWithTheKptxRegistry() {
        val fn = pagedFn()
        val lowered = lowerKernelChoice(
            fn,
            KernelTarget.CPU_GENERIC,
            inferenceRegistry = kptxInferenceKernelTemplates,
        )
        assertTrue(lowered === fn)
    }
}
