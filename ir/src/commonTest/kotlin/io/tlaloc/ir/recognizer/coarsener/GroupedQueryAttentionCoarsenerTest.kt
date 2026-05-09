package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeGroupedQueryAttention
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * §0.4.321 — GroupedQueryAttention analytical-backward coarsener tests.
 * Mirrors [FlashAttentionCoarsenerTest] / [RmsNormCoarsenerTest]:
 * structural-only verification (op count, body signatures, gradient
 * collapse). Numerical correctness of the analytical VJP is out of
 * v1 scope.
 */
class GroupedQueryAttentionCoarsenerTest {

    // [H, T, D] shapes for an MQA-canonical attention.
    private val hq = 4
    private val hKv1 = 1
    private val tlen = 8
    private val dim = 16

    private val qType = DxirType(F32, listOf(hq, tlen, dim))
    private val kRawType = DxirType(F32, listOf(hKv1, tlen, dim))
    private val vRawType = DxirType(F32, listOf(hKv1, tlen, dim))
    private val kExpandedType = DxirType(F32, listOf(hq, tlen, dim))
    private val vExpandedType = DxirType(F32, listOf(hq, tlen, dim))
    private val scoreType = DxirType(F32, listOf(hq, tlen, tlen))
    private val outType = DxirType(F32, listOf(hq, tlen, dim))

    private fun buildMqaFn(): DxirFunction = DxirBuilder.function("mqa") {
        val q = param("q", qType)
        val kRaw = param("k_raw", kRawType)
        val vRaw = param("v_raw", vRawType)
        val kExpanded = op(
            OpKind.BROADCAST, listOf(kRaw), kExpandedType,
            attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
        )
        val vExpanded = op(
            OpKind.BROADCAST, listOf(vRaw), vExpandedType,
            attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
        )
        val qk = op(OpKind.MATMUL, listOf(q, kExpanded), scoreType)
        val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
        val out = op(OpKind.MATMUL, listOf(sm, vExpanded), outType)
        listOf(out)
    }

    @Test
    fun coarsensCanonicalMqaIntoOneCoarsenedOp() {
        val fn = buildMqaFn()
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        // Body: only the COARSENED remains.
        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(3, co.operands.size, "COARSENED takes (Q, K_raw, V_raw)")
        assertEquals(outType, co.types.single())
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsenedOpHasGqaPrimalAndGradientBodies() {
        val fn = buildMqaFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeGroupedQueryAttention(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as DxirFunction
        assertEquals("gqa_primal", primal.name)
        assertEquals(3, primal.params.size, "primal takes (Q, K_raw, V_raw)")
        assertEquals(qType, primal.params[0].type)
        assertEquals(kRawType, primal.params[1].type)
        assertEquals(vRawType, primal.params[2].type)
        assertEquals(1, primal.returns.size)
        assertEquals(outType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as DxirFunction
        assertEquals("gqa_grad", grad.name)
        // (dO, Q, K_raw, V_raw) → (dQ, dK_raw, dV_raw)
        assertEquals(4, grad.params.size)
        assertEquals(outType, grad.params[0].type, "param[0] = upstream dO")
        assertEquals(qType, grad.params[1].type)
        assertEquals(kRawType, grad.params[2].type)
        assertEquals(vRawType, grad.params[3].type)
        assertEquals(3, grad.returns.size)
        assertEquals(qType, grad.returns[0].type, "dQ")
    }

    @Test
    fun gradientCollapsesToRawShapeNotExpanded() {
        // The crucial structural assertion: dK / dV come back at the RAW
        // shape (e.g. [1, T, D] for MQA), not the expanded shape
        // ([H_q, T, D]). The coarsener emits a SUM-keepdim over the
        // expansion axis to invert the BROADCAST.
        val fn = buildMqaFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeGroupedQueryAttention(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction

        assertEquals(kRawType, grad.returns[1].type, "dK_raw should be at raw shape, not expanded")
        assertEquals(vRawType, grad.returns[2].type, "dV_raw should be at raw shape, not expanded")

        // Pin the inverting SUMs: at least 2 SUMs over the K/V expansion
        // axis (axis 0 for our [1, T, D] shape). The other SUM in the
        // gradient body is for the softmax derivative on the score's
        // last axis (different reduction_dims).
        val gradOps = grad.body.filterIsInstance<DxirOp>()
        val sumsOnAxis0 = gradOps.count { it.op == OpKind.SUM && it.type == kRawType }
        assertTrue(
            sumsOnAxis0 >= 2,
            "expected ≥2 SUMs producing kRawType (dK and dV collapses); got $sumsOnAxis0",
        )
    }

    @Test
    fun readsPrimalIndicesIncludesAllThreeOperands() {
        val fn = buildMqaFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeGroupedQueryAttention(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        // The gradient body recomputes K_expanded and V_expanded (BROADCAST
        // the raws) and uses Q for the score recompute → all three primal
        // operands are referenced (idx 0=Q, 1=K_raw, 2=V_raw after
        // stripping the upstream dO at idx 0).
        assertTrue(0 in reads, "Q is referenced (param idx 0)")
        assertTrue(1 in reads, "K_raw is referenced (param idx 1)")
        assertTrue(2 in reads, "V_raw is referenced (param idx 2)")
    }

    @Test
    fun coarsenerDeclinesGqaCanonicalReshapeBroadcastReshape() {
        // GQA-canonical (the PyTorch repeat_kv form) has 3 ops per side
        // wrapping the BROADCAST. v1 coarsener returns null for those —
        // the recognizer's match is then untouched and downstream lowering
        // decomposes back to primitives. Keeps v1 honest about scope.
        val gqaRawType = DxirType(F32, listOf(2, tlen, dim))           // [H_kv=2, T, D]
        val gqaUnsqType = DxirType(F32, listOf(2, 1, tlen, dim))       // [2, 1, T, D]
        val gqaBcastType = DxirType(F32, listOf(2, 2, tlen, dim))      // [2, 2, T, D]

        val fn = DxirBuilder.function("gqa_canonical") {
            val q = param("q", qType)
            val kRaw = param("k_raw", gqaRawType)
            val vRaw = param("v_raw", gqaRawType)
            val kUnsq = op(OpKind.RESHAPE, listOf(kRaw), gqaUnsqType)
            val kBcast = op(
                OpKind.BROADCAST, listOf(kUnsq), gqaBcastType,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2, 3)),
            )
            val kFlat = op(OpKind.RESHAPE, listOf(kBcast), kExpandedType)
            val vUnsq = op(OpKind.RESHAPE, listOf(vRaw), gqaUnsqType)
            val vBcast = op(
                OpKind.BROADCAST, listOf(vUnsq), gqaBcastType,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2, 3)),
            )
            val vFlat = op(OpKind.RESHAPE, listOf(vBcast), vExpandedType)
            val qk = op(OpKind.MATMUL, listOf(q, kFlat), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, vFlat), outType)
            listOf(out)
        }
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(1, matches.size, "recognizer matches GQA-canonical (its v1 covers both forms)")

        // Coarsener should decline. Body should still contain all the
        // original ops (no rewrite happened) since the only matched
        // pattern declined.
        val coarsened = coarsenRecognizedPatterns(fn, matches)
        val coarsenedOps = coarsened.body.filterIsInstance<DxirOp>().count { it.op == OpKind.COARSENED }
        assertEquals(0, coarsenedOps, "no COARSENED should appear; coarsener declined GQA-canonical")
    }
}
