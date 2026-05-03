package io.tlaloc.benchmarks

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.271 — LlamaDecoderPrimal builds a valid DXIR function tests.
 *
 * Phase 2 step 1 of the dual-track Llama-decoder benchmark plan. v1
 * tests cover: tiny config compiles, scalar-loss return type, expected
 * param count, llama3_8b also compiles, bad config rejected. The
 * "all 5 patterns recognize+coarsen" structural test is §0.4.272's
 * scope; this phase just pins that the primal builds.
 */
class LlamaDecoderPrimalTest {

    @Test
    fun tinyConfigBuildsValidPrimal() {
        val fn = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        assertEquals("llama_decoder_layer_loss", fn.name)
        assertTrue(fn.body.isNotEmpty(), "primal body must contain ops")
    }

    @Test
    fun primalReturnsScalarLoss() {
        val fn = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        assertEquals(1, fn.returns.size, "primal returns a single value (the scalar loss)")
        assertEquals(DxirType(F32, listOf()), fn.returns.single().type, "loss is a scalar F32")
    }

    @Test
    fun tinyConfigParamCount() {
        // Per the LlamaDecoderPrimal docblock, the primal takes 13 params:
        // x_in, labels, theta, q_w, k_w, v_w, out_w, gate_w, up_w, down_w,
        // lm_head_w, eps_attn, eps_mlp.
        val fn = LlamaDecoderPrimal.build(LlamaDecoderConfig.tiny)
        assertEquals(13, fn.params.size)
        val expectedNames = listOf(
            "x_in", "labels", "theta",
            "q_w", "k_w", "v_w", "out_w",
            "gate_w", "up_w", "down_w",
            "lm_head_w",
            "eps_attn", "eps_mlp",
        )
        assertEquals(expectedNames, fn.params.map { it.name })
    }

    @Test
    fun llama3_8bConfigBuildsWithoutThrowing() {
        // Realistic shapes — won't be exercised by Phase 2 tests but the
        // builder must handle them. (Smoke that the dim arithmetic doesn't
        // overflow Int and that no shape validation fires spuriously at
        // production scale.)
        val fn = LlamaDecoderPrimal.build(LlamaDecoderConfig.llama3_8b)
        assertEquals("llama_decoder_layer_loss", fn.name)
        // d_ff = 4096 * 3.5 = 14336 — pin this since the docblock claims it.
        assertEquals(14336, LlamaDecoderConfig.llama3_8b.dFf)
        assertEquals(2048, LlamaDecoderConfig.llama3_8b.tokens)
    }

    @Test
    fun configValidationRejectsBadShapes() {
        assertFailsWith<IllegalArgumentException> {
            LlamaDecoderConfig(
                batch = 1, seq = 16,
                dModel = 65,        // not divisible by nHeads * headDim
                nHeads = 4, headDim = 16,
                ffnMult = 4.0, vocab = 256,
            )
        }
    }
}
