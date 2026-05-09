package io.tlaloc.ir.recognizer

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * §0.4.320 — GroupedQueryAttention recognizer tests. Mirrors the
 * [FlashAttentionRecognizerTest] shape: positive matches (MQA + GQA
 * canonical forms), adversarial near-misses, plus the resolver scenario
 * that demonstrates GQA winning over the bare FlashAttention match.
 */
class GroupedQueryAttentionRecognizerTest {

    // Shape conventions for these tests: [H, T, D].
    private val hq = 4
    private val hKv1 = 1   // MQA: single shared KV head
    private val hKv2 = 2   // GQA: 2 KV heads, group_ratio = 2
    private val tlen = 8
    private val dim = 16

    private val qType = DxirType(F32, listOf(hq, tlen, dim))
    private val kvExpandedType = DxirType(F32, listOf(hq, tlen, dim))
    private val mqaRawType = DxirType(F32, listOf(hKv1, tlen, dim))
    private val gqaRawType = DxirType(F32, listOf(hKv2, tlen, dim))             // [2, T, D]
    private val gqaUnsqueezedType = DxirType(F32, listOf(hKv2, 1, tlen, dim))   // [2, 1, T, D] — size-1 inserted
    private val gqaBcastType = DxirType(F32, listOf(hKv2, 2, tlen, dim))        // [2, 2, T, D] — expanded size-1 → 2

    private val scoreType = DxirType(F32, listOf(hq, tlen, tlen))
    private val outType = DxirType(F32, listOf(hq, tlen, dim))

    @Test
    fun positiveMqaShape() {
        // K and V have a single head; one explicit BROADCAST per side
        // expands to H_q. group_ratio = 4.
        val fn = DxirBuilder.function("mqa") {
            val q = param("q", qType)
            val kRaw = param("k_raw", mqaRawType)
            val vRaw = param("v_raw", mqaRawType)
            val kExpanded = op(OpKind.BROADCAST, listOf(kRaw), kvExpandedType)
            val vExpanded = op(OpKind.BROADCAST, listOf(vRaw), kvExpandedType)
            val qk = op(OpKind.MATMUL, listOf(q, kExpanded), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, vExpanded), outType)
            listOf(out)
        }
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(1, matches.size, "expected exactly one MQA match")
        val m = matches.single()
        assertEquals(hq, m.groupRatio, "MQA group ratio = H_q / 1")
        assertEquals(OpKind.BROADCAST, m.kBroadcast.op)
        assertEquals(OpKind.BROADCAST, m.vBroadcast.op)
        assertSame(m.kRawInput, fn.params[1])
        assertSame(m.vRawInput, fn.params[2])
        assertEquals(5, m.ops.size, "qkMatmul + softmax + pvMatmul + kBcast + vBcast = 5")
    }

    @Test
    fun positiveGqaWithReshapeBroadcastReshape() {
        // The canonical PyTorch `repeat_kv` form: insert a size-1 axis
        // via RESHAPE, broadcast it, then RESHAPE to flatten.
        val fn = DxirBuilder.function("gqa_canonical") {
            val q = param("q", qType)
            val kRaw = param("k_raw", gqaRawType)
            val vRaw = param("v_raw", gqaRawType)
            // K expansion chain: [2, T, D] → [2, 1, T, D] → [2, 2, T, D] → [4, T, D]
            val kUnsqueezed = op(OpKind.RESHAPE, listOf(kRaw), gqaUnsqueezedType)
            val kBcast = op(OpKind.BROADCAST, listOf(kUnsqueezed), gqaBcastType)
            val kFlat = op(OpKind.RESHAPE, listOf(kBcast), kvExpandedType)
            val vUnsqueezed = op(OpKind.RESHAPE, listOf(vRaw), gqaUnsqueezedType)
            val vBcast = op(OpKind.BROADCAST, listOf(vUnsqueezed), gqaBcastType)
            val vFlat = op(OpKind.RESHAPE, listOf(vBcast), kvExpandedType)
            val qk = op(OpKind.MATMUL, listOf(q, kFlat), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, vFlat), outType)
            listOf(out)
        }
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(1, matches.size, "expected exactly one GQA match")
        val m = matches.single()
        assertEquals(2, m.groupRatio, "GQA group ratio = H_q / H_kv = 4 / 2")
        // ops absorbed: 3 attention + (RESHAPE, BROADCAST, RESHAPE) × 2
        assertEquals(9, m.ops.size, "3 attention + 6 expansion ops")
        assertSame(m.kRawInput, fn.params[1], "K raw input is the leaf, not the inner RESHAPE")
        assertSame(m.vRawInput, fn.params[2])
    }

    @Test
    fun standardMhaIsNotMatchedAsGqa() {
        // No BROADCAST in K/V chains → standard multi-head attention.
        // FlashAttention matches this; GroupedQueryAttention shouldn't.
        val fn = DxirBuilder.function("mha") {
            val q = param("q", qType)
            val k = param("k", qType)   // same head count as Q
            val v = param("v", qType)
            val qk = op(OpKind.MATMUL, listOf(q, k), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, v), outType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeGroupedQueryAttention(fn, diag)
        assertEquals(0, matches.size, "MHA shouldn't be matched as GQA")
        assertTrue(diag.any { "standard MHA" in it.reason })
    }

    @Test
    fun kBroadcastedButVNotIsRejected() {
        // Half-expanded shape: K goes through a BROADCAST but V doesn't.
        // This isn't a sound GQA — V's head dim wouldn't match the
        // attention output's expected H_q.
        val fn = DxirBuilder.function("k_only_bcast") {
            val q = param("q", qType)
            val kRaw = param("k_raw", mqaRawType)
            val v = param("v", qType)   // already at H_q, no broadcast
            val kExpanded = op(OpKind.BROADCAST, listOf(kRaw), kvExpandedType)
            val qk = op(OpKind.MATMUL, listOf(q, kExpanded), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, v), outType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeGroupedQueryAttention(fn, diag)
        assertEquals(0, matches.size)
        assertTrue(diag.any { "K is expanded but V isn't" in it.reason })
    }

    @Test
    fun mismatchedGroupRatiosRejected() {
        // K broadcasts to [4, T, D] (ratio 4) but V to [2, T, D] in a
        // different score type — group ratios disagree, so the head
        // structure is inconsistent.
        val midType = DxirType(F32, listOf(2, tlen, dim))
        val fn = DxirBuilder.function("mismatched_ratios") {
            val q = param("q", qType)
            val kRaw = param("k_raw", mqaRawType)   // H=1
            val vRaw = param("v_raw", mqaRawType)   // H=1
            val kExpanded = op(OpKind.BROADCAST, listOf(kRaw), kvExpandedType)   // H=1 → 4
            val vExpanded = op(OpKind.BROADCAST, listOf(vRaw), midType)          // H=1 → 2
            val qk = op(OpKind.MATMUL, listOf(q, kExpanded), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            // pv matmul shape doesn't matter here; the recognizer rejects on
            // ratio mismatch before checking output type compatibility.
            val out = op(OpKind.MATMUL, listOf(sm, vExpanded), outType)
            listOf(out)
        }
        val diag = mutableListOf<RecognitionDiagnostic>()
        val matches = recognizeGroupedQueryAttention(fn, diag)
        assertEquals(0, matches.size)
        assertTrue(diag.any { "inconsistent head structure" in it.reason })
    }

    @Test
    fun degenerateBroadcastWithRatio1IsNotGqa() {
        // BROADCAST that doesn't change shape — degenerate identity.
        // Reject rather than report group_ratio = 1.
        val fn = DxirBuilder.function("identity_bcast") {
            val q = param("q", qType)
            val k = param("k", qType)
            val v = param("v", qType)
            // Same input/output type — broadcast is a noop.
            val kBcast = op(OpKind.BROADCAST, listOf(k), qType)
            val vBcast = op(OpKind.BROADCAST, listOf(v), qType)
            val qk = op(OpKind.MATMUL, listOf(q, kBcast), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, vBcast), outType)
            listOf(out)
        }
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(0, matches.size, "ratio-1 broadcast isn't an expansion")
    }

    @Test
    fun gqaSupersedesFlashAttentionInRecognizeAll() {
        // The crucial v2-resolver scenario: both FlashAttention and
        // GroupedQueryAttention match the same softmax. GQA absorbs the
        // BROADCAST ops in its `ops` list, so it's larger; the resolver
        // picks GQA.
        val fn = DxirBuilder.function("mqa_full") {
            val q = param("q", qType)
            val kRaw = param("k_raw", mqaRawType)
            val vRaw = param("v_raw", mqaRawType)
            val kExpanded = op(OpKind.BROADCAST, listOf(kRaw), kvExpandedType)
            val vExpanded = op(OpKind.BROADCAST, listOf(vRaw), kvExpandedType)
            val qk = op(OpKind.MATMUL, listOf(q, kExpanded), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, vExpanded), outType)
            listOf(out)
        }
        val matches = recognizeAll(fn)
        assertEquals(1, matches.size, "exactly one match should survive the resolver")
        val match = matches.single()
        assertTrue(
            match is RecognitionMatch.GroupedQueryAttention,
            "expected GQA to win over FlashAttention by op count; got ${match.patternName}",
        )
        assertEquals(5, match.ops.size, "qk/sm/pv + 2 BROADCAST")
    }
}
