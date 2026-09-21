package io.tlaloc.kptx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.471 — Phase H4: the offline pins on the paged-attention launch
 * chain. The numerical oracle lives on the GPU
 * (`KptxPagedAttentionKernelTest`, which self-skips without CUDA); what
 * always runs is the structure the registry marshals against, plus the
 * one lesson this kernel taught the hard way.
 */
class PagedAttentionModuleTest {

    private val module = KptxKernels.pagedAttentionModule(block = 256, scale = 0.125f)

    @Test
    fun theChainIsThreeStagesInRegistryOrder() {
        assertEquals(
            listOf("kptx_paged_scores", "kptx_paged_softmax", "kptx_paged_out"),
            module.kernels.map { it.name },
        )
    }

    @Test
    fun theStageSignaturesMatchWhatTheRegistryMarshals() {
        // inputs-then-outputs buffer pointers, then the trailing i32 shape
        // params — every one of which is READ FROM A BUFFER SHAPE at
        // dispatch, never baked (the sentinel-dims rule).
        val scores = module.kernels.single { it.name == "kptx_paged_scores" }
        assertEquals(
            listOf("q_ptr", "k_ptr", "tab_ptr", "len_ptr", "s_ptr", "n_h", "n_d", "n_bs", "n_kv", "n_mb"),
            scores.params.map { it.name },
        )
        val out = module.kernels.single { it.name == "kptx_paged_out" }
        assertEquals(
            listOf("s_ptr", "v_ptr", "tab_ptr", "len_ptr", "o_ptr", "n_h", "n_d", "n_bs", "n_kv", "n_mb"),
            out.params.map { it.name },
        )
        // The softmax stage is the DENSE chain's kernel, reused verbatim —
        // only its row width is named differently.
        val softmax = module.kernels.single { it.name == "kptx_paged_softmax" }
        assertEquals(listOf("s_ptr", "n_ctx"), softmax.params.map { it.name })
    }

    @Test
    fun theScaleAttrIsBakedAndTheModuleIsCachedPerScale() {
        // `scale` is a compile-time literal on the op, so specializing the
        // PTX on it is the house cache pattern, not a sentinel-dims
        // violation. 0.125f = 0x3E000000.
        assertTrue("0f3E000000" in module.emitPtx(), "the scale immediate must appear in the scores stage")
        assertTrue("0f3E000000" !in KptxKernels.pagedAttentionModule(256, 0.25f).emitPtx())
        assertTrue(module === KptxKernels.pagedAttentionModule(256, 0.125f), "cached per (block, scale)")
    }

    @Test
    fun theDenseChainIsUnchangedByTheSharedSoftmaxExtraction() {
        // §0.4.358's module now calls the same helper the paged one does.
        // Its kernel names and signatures are the contract three Stage
        // lambdas in KptxAttentionKernelTest marshal against.
        val dense = KptxKernels.attentionModule(block = 256)
        assertEquals(
            listOf("kptx_attn_scores", "kptx_attn_softmax", "kptx_attn_out"),
            dense.kernels.map { it.name },
        )
        assertEquals(
            listOf("s_ptr", "n_t"),
            dense.kernels.single { it.name == "kptx_attn_softmax" }.params.map { it.name },
        )
    }

    @Test
    fun theEmittedPtxIsPureAscii() {
        // ptxas rejects a non-ASCII byte anywhere in the file — including
        // inside a comment — with `Unexpected non-ASCII character`, and the
        // failure surfaces as CUDA_ERROR_INVALID_PTX from cuModuleLoadData
        // deep inside an XLA execution, which names nothing. A typographic
        // dot in a kernel comment cost exactly one debugging round trip
        // here; this pin is cheaper than the next one.
        for (m in listOf(module, KptxKernels.attentionModule(256))) {
            val ptx = m.emitPtx()
            val bad = ptx.indexOfFirst { it.code > 127 }
            assertTrue(
                bad < 0,
                "non-ASCII at offset $bad: ...${ptx.substring(maxOf(0, bad - 40), minOf(ptx.length, bad + 40))}...",
            )
        }
    }
}
