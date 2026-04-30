package io.tlaloc.ir.recognizer

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.250+ — FlashAttention recognizer tests.
 *
 * Per the L3 audit's recognizer-soundness requirement: each recognizer
 * needs a positive-match test plus *at least four adversarial cases*
 * that almost match but shouldn't. This file covers FlashAttention.
 * RMS norm / RoPE / cross-entropy follow the same pattern in L3.1.
 */
class FlashAttentionRecognizerTest {

    private val rank4 = DxirType(F32, listOf(2, 8, 64, 64))   // (Batch, Heads, SeqLen, Dim)
    private val scoreType = DxirType(F32, listOf(2, 8, 64, 64)) // raw QK^T
    private val outputType = DxirType(F32, listOf(2, 8, 64, 64))

    @Test
    fun positiveMatchOnMatmulSoftmaxMatmulChain() {
        // The canonical compound form a user-written attention forward
        // produces: QK^T → softmax → P·V.
        val fn = DxirBuilder.function("attn") {
            val q = param("q", rank4)
            val k = param("k", rank4)
            val v = param("v", rank4)
            val qk = op(OpKind.MATMUL, listOf(q, k), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, v), outputType)
            listOf(out)
        }
        val matches = recognizeFlashAttention(fn)
        assertEquals(1, matches.size, "expected exactly one FlashAttention match")
        val m = matches.single()
        assertEquals(OpKind.MATMUL, m.qkMatmul.op)
        assertEquals(OpKind.SOFTMAX, m.softmax.op)
        assertEquals(OpKind.MATMUL, m.pvMatmul.op)
        assertEquals(scoreType, m.scoreType)
        assertEquals(outputType, m.outputType)
        assertEquals(3, m.ops.size)
    }

    @Test
    fun positiveMatchSurvivesUnrelatedOpsBetween() {
        // The recognizer is structural — interleaving an unrelated op
        // (here a separate RELU on a different tensor) doesn't break the
        // chain.
        val fn = DxirBuilder.function("attn_with_unrelated_op") {
            val q = param("q", rank4)
            val k = param("k", rank4)
            val v = param("v", rank4)
            val unrelatedConst = param("scale", DxirType(F32, listOf()))  // scalar
            val unrelatedRelu = op(OpKind.RELU, listOf(unrelatedConst), DxirType(F32, listOf()))
            val qk = op(OpKind.MATMUL, listOf(q, k), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, v), outputType)
            listOf(out, unrelatedRelu)
        }
        val matches = recognizeFlashAttention(fn)
        assertEquals(1, matches.size)
    }

    @Test
    fun adversarial1WrongReduceKindRejected() {
        // Adversarial case 1: the "softmax" step is actually a SUM.
        // Recognizer's pre-filter fires only on OpKind.SOFTMAX, so this
        // doesn't even trigger near-miss diagnostics — it just returns
        // empty. (The SUM op's own pre-filter doesn't exist for attention.)
        val fn = DxirBuilder.function("matmul_sum_matmul") {
            val q = param("q", rank4)
            val k = param("k", rank4)
            val v = param("v", rank4)
            val qk = op(OpKind.MATMUL, listOf(q, k), scoreType)
            val s = op(OpKind.SUM, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(s, v), outputType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeFlashAttention(fn, diag)
        assertEquals(0, matches.size, "SUM shouldn't be recognized as softmax")
        assertTrue(diag.isEmpty(), "no SOFTMAX op present, so no diagnostic should fire")
    }

    @Test
    fun adversarial2SoftmaxOperandNotMatmulRejectedWithDiagnostic() {
        // Adversarial case 2: SOFTMAX consumes a RELU (not a MATMUL).
        // Pre-filter triggers, full match rejects, diagnostic fires.
        val fn = DxirBuilder.function("relu_softmax_matmul") {
            val x = param("x", rank4)
            val v = param("v", rank4)
            val activated = op(OpKind.RELU, listOf(x), rank4)
            val sm = op(OpKind.SOFTMAX, listOf(activated), rank4)
            val out = op(OpKind.MATMUL, listOf(sm, v), outputType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeFlashAttention(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size, "expected exactly one near-miss diagnostic")
        val d = diag.single()
        assertEquals("FlashAttention", d.pattern)
        assertTrue(
            "RELU" in d.reason && "MATMUL" in d.reason,
            "diagnostic should explain RELU vs MATMUL mismatch; got: ${d.reason}",
        )
    }

    @Test
    fun adversarial3SoftmaxNoMatmulConsumerRejectedWithDiagnostic() {
        // Adversarial case 3: SOFTMAX → SUM (final-loss term, not attention).
        // Pre-filter triggers; full match rejects because no MATMUL consumer.
        val fn = DxirBuilder.function("matmul_softmax_sum") {
            val q = param("q", rank4)
            val k = param("k", rank4)
            val qk = op(OpKind.MATMUL, listOf(q, k), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val loss = op(OpKind.SUM, listOf(sm), DxirType(F32, listOf()))
            listOf(loss)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeFlashAttention(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        val d = diag.single()
        assertTrue("MATMUL consumer" in d.reason, "diagnostic should mention missing MATMUL consumer; got: ${d.reason}")
    }

    @Test
    fun adversarial4SelfContractionRejectedWithDiagnostic() {
        // Adversarial case 4: Q ≡ K ≡ V (all the same tensor).
        // A·A^T·A is a stylized matmul chain, not attention.
        val fn = DxirBuilder.function("self_contraction") {
            val a = param("a", rank4)
            val qk = op(OpKind.MATMUL, listOf(a, a), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, a), outputType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeFlashAttention(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(1, diag.size)
        assertTrue("self-contraction" in diag.single().reason)
    }

    @Test
    fun multipleAttentionLayersAllMatched() {
        // Two attention blocks back-to-back (a transformer's attention
        // sub-layer). Recognizer should match both; resolveLargestMatch
        // doesn't prune them (no overlap in matched op ids).
        val fn = DxirBuilder.function("two_attentions") {
            val q1 = param("q1", rank4)
            val k1 = param("k1", rank4)
            val v1 = param("v1", rank4)
            val qk1 = op(OpKind.MATMUL, listOf(q1, k1), scoreType)
            val sm1 = op(OpKind.SOFTMAX, listOf(qk1), scoreType)
            val out1 = op(OpKind.MATMUL, listOf(sm1, v1), outputType)
            val q2 = param("q2", rank4)
            val k2 = param("k2", rank4)
            val v2 = param("v2", rank4)
            val qk2 = op(OpKind.MATMUL, listOf(q2, k2), scoreType)
            val sm2 = op(OpKind.SOFTMAX, listOf(qk2), scoreType)
            val out2 = op(OpKind.MATMUL, listOf(sm2, v2), outputType)
            listOf(out1, out2)
        }
        val matches = recognizeFlashAttention(fn)
        assertEquals(2, matches.size)
    }

    @Test
    fun recognizeAllAggregatorReturnsFlashAttentionMatch() {
        val fn = DxirBuilder.function("attn_via_aggregator") {
            val q = param("q", rank4)
            val k = param("k", rank4)
            val v = param("v", rank4)
            val qk = op(OpKind.MATMUL, listOf(q, k), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, v), outputType)
            listOf(out)
        }
        val matches = recognizeAll(fn)
        assertEquals(1, matches.size)
        val m = matches.single()
        assertNotNull(m as? RecognitionMatch.FlashAttention)
        assertEquals("FlashAttention", m.patternName)
    }

    @Test
    fun emptyFunctionMatchesNothing() {
        // Sanity guard: a function with no SOFTMAX returns an empty list
        // and emits no diagnostics.
        val fn = DxirBuilder.function("noop") {
            val a = param("a", rank4)
            val r = op(OpKind.RELU, listOf(a), rank4)
            listOf(r)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeFlashAttention(fn, diag)
        assertEquals(0, matches.size)
        assertEquals(0, diag.size)
    }
}
