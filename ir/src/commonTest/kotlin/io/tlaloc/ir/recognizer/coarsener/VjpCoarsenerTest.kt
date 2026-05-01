package io.tlaloc.ir.recognizer.coarsener

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Layer 3 §0.4.252+ — VJP coarsener + FlashAttention analytical-backward
 * tests.
 *
 * Tests verify *structural* correctness only. Runtime evaluation of the
 * coarsened gradient body needs interpreter support for rank-(r-1) →
 * rank-r BROADCAST, which is deferred to L3.4. Today's contract: the
 * coarsened op validates against `validateCoarsenedShape`, the gradient
 * body has the right signature, and `DxirReverseTransform` doesn't
 * choke when fed a function containing the COARSENED.
 */
class VjpCoarsenerTest {

    // Rank-2 attention (M=8, K=4, N=8, D_v=4).
    private val mDim = 8
    private val kDim = 4
    private val nDim = 8
    private val dvDim = 4

    private val qType = DxirType(F32, listOf(mDim, kDim))   // [M, K]
    private val kType = DxirType(F32, listOf(kDim, nDim))   // [K, N]  (pre-transposed)
    private val vType = DxirType(F32, listOf(nDim, dvDim))  // [N, D_v]
    private val sType = DxirType(F32, listOf(mDim, nDim))   // [M, N]
    private val oType = DxirType(F32, listOf(mDim, dvDim))  // [M, D_v]

    private fun buildAttentionFn() = DxirBuilder.function("attn") {
        val q = param("Q", qType)
        val k = param("K", kType)
        val v = param("V", vType)
        val qk = op(OpKind.MATMUL, listOf(q, k), sType)
        val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
        val out = op(OpKind.MATMUL, listOf(sm, v), oType)
        listOf(out)
    }

    @Test
    fun coarsensCanonicalAttentionGraphIntoOneCoarsenedOp() {
        val fn = buildAttentionFn()
        val matches = recognizeFlashAttention(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        // Body: only the COARSENED remains; QK MATMUL + SOFTMAX + PV MATMUL absorbed.
        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(3, co.operands.size, "COARSENED takes (Q, K, V)")
        assertEquals(1, co.types.size, "single-result attention")
        assertEquals(oType, co.types.single())

        // Function returns the COARSENED (replaces the old pvMatmul).
        assertEquals(1, coarsened.returns.size)
        assertEquals(co.id, coarsened.returns.single().id)
    }

    @Test
    fun coarsenedOpHasFlashAttentionPrimalAndGradientBodies() {
        val fn = buildAttentionFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeFlashAttention(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        val primal = co.attrs["primal_body"]
        assertNotNull(primal)
        primal as io.tlaloc.ir.DxirFunction
        assertEquals("flash_attention_primal", primal.name)
        assertEquals(3, primal.params.size)
        assertEquals(qType, primal.params[0].type)
        assertEquals(kType, primal.params[1].type)
        assertEquals(vType, primal.params[2].type)
        assertEquals(1, primal.returns.size)
        assertEquals(oType, primal.returns.single().type)

        val grad = co.attrs["gradient_body"]
        assertNotNull(grad)
        grad as io.tlaloc.ir.DxirFunction
        assertEquals("flash_attention_grad", grad.name)
        // K=1 (single-result COARSENED) + N=3 (Q,K,V operands) = 4 params.
        assertEquals(4, grad.params.size)
        assertEquals(oType, grad.params[0].type, "param[0] is upstream dO")
        assertEquals(qType, grad.params[1].type, "param[1] is Q")
        assertEquals(kType, grad.params[2].type, "param[2] is K")
        assertEquals(vType, grad.params[3].type, "param[3] is V")
        assertEquals(3, grad.returns.size, "returns (dQ, dK, dV)")
        assertEquals(qType, grad.returns[0].type)
        assertEquals(kType, grad.returns[1].type)
        assertEquals(vType, grad.returns[2].type)
    }

    @Test
    fun readsPrimalIndicesIncludesAllThreeOperands() {
        val fn = buildAttentionFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeFlashAttention(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()

        @Suppress("UNCHECKED_CAST")
        val reads = co.attrs["reads_primal_indices"] as Set<Int>
        assertTrue(0 in reads, "gradient body references Q (param idx 0)")
        assertTrue(1 in reads, "gradient body references K (param idx 1)")
        assertTrue(2 in reads, "gradient body references V (param idx 2)")
    }

    @Test
    fun gradientBodyContainsAnalyticalShapeOps() {
        val fn = buildAttentionFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeFlashAttention(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as io.tlaloc.ir.DxirFunction

        val opKinds = grad.body.filterIsInstance<DxirOp>().map { it.op }
        // Sanity: the analytical backward emits MATMUL (×5 — recompute S,
        // dV, dP, dQ, dK), TRANSPOSE (×4 — P^T, V^T, K^T, Q^T), MUL (×2 —
        // P·dP and the final dS gating), SUM (×1 — last-axis reduce),
        // BROADCAST (×1 — bring the reduced axis back), SUB (×1 — dP -
        // sum), and SOFTMAX (×1 — recompute P).
        assertEquals(5, opKinds.count { it == OpKind.MATMUL })
        assertEquals(4, opKinds.count { it == OpKind.TRANSPOSE })
        assertEquals(2, opKinds.count { it == OpKind.MUL })
        assertEquals(1, opKinds.count { it == OpKind.SUM })
        assertEquals(1, opKinds.count { it == OpKind.BROADCAST })
        assertEquals(1, opKinds.count { it == OpKind.SUB })
        assertEquals(1, opKinds.count { it == OpKind.SOFTMAX })
    }

    @Test
    fun nonMatchingFunctionPassesThroughUnchanged() {
        // No SOFTMAX → no FlashAttention match → coarsener is a no-op.
        val fn = DxirBuilder.function("simple") {
            val x = param("x", qType)
            val y = op(OpKind.RELU, listOf(x), qType)
            listOf(y)
        }
        val out = coarsenRecognizedPatterns(fn, recognizeFlashAttention(fn))
        // Same body shape (one RELU op), no COARSENED introduced.
        val ops = out.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size)
        assertEquals(OpKind.RELU, ops.single().op)
        assertFalse(ops.any { it.op == OpKind.COARSENED })
    }

    @Test
    fun externalConsumerOnSoftmaxBlocksCoarsening() {
        // The softmax intermediate has a consumer outside the matched
        // sub-graph (a side-channel SUM on softmax probs). Coarsening
        // would lose that consumer's value, so the coarsener declines.
        val fn = DxirBuilder.function("attn_with_softmax_side_use") {
            val q = param("Q", qType)
            val k = param("K", kType)
            val v = param("V", vType)
            val qk = op(OpKind.MATMUL, listOf(q, k), sType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
            val attentionOut = op(OpKind.MATMUL, listOf(sm, v), oType)
            // Side channel: also reduce the softmax to a scalar.
            val sideChannel = op(
                OpKind.SUM,
                listOf(sm),
                DxirType(F32, listOf()),
            )
            listOf(attentionOut, sideChannel)
        }
        val matches = recognizeFlashAttention(fn)
        assertEquals(1, matches.size, "recognizer still finds the structural pattern")

        val out = coarsenRecognizedPatterns(fn, matches)
        // No COARSENED was emitted (external consumer blocked the rewrite).
        val ops = out.body.filterIsInstance<DxirOp>()
        assertFalse(
            ops.any { it.op == OpKind.COARSENED },
            "external consumer must block coarsening; got body ${ops.map { it.op }}",
        )
        // And the original ops are still present.
        assertTrue(ops.any { it.op == OpKind.MATMUL })
        assertTrue(ops.any { it.op == OpKind.SOFTMAX })
    }

    @Test
    fun unregisteredPatternIsIgnored() {
        // Empty registry — even if FlashAttention is recognized, no
        // coarsener is registered for it, so the function passes
        // through.
        val fn = buildAttentionFn()
        val matches = recognizeFlashAttention(fn)
        val out = coarsenRecognizedPatterns(fn, matches, coarseners = emptyMap())
        val ops = out.body.filterIsInstance<DxirOp>()
        assertFalse(ops.any { it.op == OpKind.COARSENED })
        assertEquals(3, ops.size, "all three original ops survive")
    }

    @Test
    fun coarsenedFunctionPassesValidation() {
        // The DxirFunction.init invariant runs `validateCoarsenedShape`
        // and throws on any operand-count / type-mismatch. Reaching this
        // point at all means the COARSENED op's primal_body, gradient_body,
        // and reads_primal_indices are mutually consistent.
        val fn = buildAttentionFn()
        val out = coarsenRecognizedPatterns(fn, recognizeFlashAttention(fn))
        // Re-running through DxirBuilder.function rebuilds + revalidates.
        val rebuilt = DxirBuilder.function(out.name) {
            val nodeMap = HashMap<Int, io.tlaloc.ir.DxirNode>()
            for (p in out.params) {
                nodeMap[p.id] = param(p.name, p.type, p.sharding)
            }
            for (node in out.body) {
                if (node !is DxirOp) continue
                val operands = node.operands.map { nodeMap[it.id]!! }
                if (node.op == OpKind.COARSENED) {
                    val primal = node.attrs["primal_body"] as io.tlaloc.ir.DxirFunction
                    val grad = node.attrs["gradient_body"] as io.tlaloc.ir.DxirFunction
                    @Suppress("UNCHECKED_CAST")
                    val reads = node.attrs["reads_primal_indices"] as Set<Int>
                    nodeMap[node.id] = coarsened(operands, primal, grad, reads)
                } else {
                    nodeMap[node.id] = op(node.op, operands, node.type, node.attrs, node.sharding)
                }
            }
            out.returns.map { nodeMap[it.id]!! }
        }
        // If we got here, validation passed.
        assertEquals(1, rebuilt.body.filterIsInstance<DxirOp>().size)
    }
}
