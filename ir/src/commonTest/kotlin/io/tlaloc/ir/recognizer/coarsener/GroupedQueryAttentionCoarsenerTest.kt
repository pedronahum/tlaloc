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

    // -------------------------------------------------------------------------
    // GQA-canonical (§0.4.322) — the PyTorch `repeat_kv` shape with
    // RESHAPE → BROADCAST → RESHAPE per side.
    // -------------------------------------------------------------------------

    private val hKv2 = 2
    private val gqaRawType = DxirType(F32, listOf(hKv2, tlen, dim))           // [2, T, D]
    private val gqaUnsqType = DxirType(F32, listOf(hKv2, 1, tlen, dim))       // [2, 1, T, D]
    private val gqaBcastType = DxirType(F32, listOf(hKv2, 2, tlen, dim))      // [2, 2, T, D]
    // Final post-flatten K shape == kExpandedType ([4, T, D]) — Q's rank.

    private fun buildGqaCanonicalFn(): DxirFunction = DxirBuilder.function("gqa_canonical") {
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

    @Test
    fun coarsensCanonicalGqaIntoOneCoarsenedOp() {
        val fn = buildGqaCanonicalFn()
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")
        assertEquals(9, matches.single().ops.size, "GQA absorbs 3 attn + 3 ops × 2 sides")

        val coarsened = coarsenRecognizedPatterns(fn, matches)

        val ops = coarsened.body.filterIsInstance<DxirOp>()
        assertEquals(1, ops.size, "expected exactly one body op (the COARSENED); got ${ops.map { it.op }}")
        val co = ops.single()
        assertEquals(OpKind.COARSENED, co.op)
        assertEquals(3, co.operands.size, "COARSENED takes (Q, K_raw, V_raw)")
        assertEquals(outType, co.types.single())
    }

    @Test
    fun gqaCanonicalGradientCollapsesToRawShapeViaReshapeChain() {
        // The crucial v2 assertion: dK_raw / dV_raw come back at the
        // RAW shape ([H_kv=2, T, D]), not the broadcasted-but-not-yet-
        // flattened shape ([H_kv, group, T, D]) or the post-flatten
        // shape ([H_q, T, D]). The coarsener emits an outer RESHAPE
        // inverse + SUM-keepdim + inner RESHAPE inverse.
        val fn = buildGqaCanonicalFn()
        val coarsened = coarsenRecognizedPatterns(fn, recognizeGroupedQueryAttention(fn))
        val co = coarsened.body.filterIsInstance<DxirOp>().single()
        val grad = co.attrs["gradient_body"] as DxirFunction

        assertEquals(4, grad.params.size, "(dO, Q, K_raw, V_raw)")
        assertEquals(gqaRawType, grad.params[2].type, "K_raw param at raw shape")
        assertEquals(gqaRawType, grad.params[3].type, "V_raw param at raw shape")
        assertEquals(3, grad.returns.size)
        assertEquals(qType, grad.returns[0].type, "dQ at Q shape")
        assertEquals(gqaRawType, grad.returns[1].type, "dK_raw at raw shape, not expanded")
        assertEquals(gqaRawType, grad.returns[2].type, "dV_raw at raw shape, not expanded")

        // Pin the inverse RESHAPEs and inverse SUMs structurally. Inverse
        // chain per side is: RESHAPE (outer-inverse) + SUM-keepdim +
        // RESHAPE (inner-inverse) = 3 ops × 2 sides = 6. Plus the
        // forward expansion's 3 ops × 2 sides recompute = 6. Total
        // RESHAPE+BROADCAST+SUM/etc count is bounded but exact numbers
        // vary; pin only the gradient-side raw-shape RESHAPEs.
        val gradOps = grad.body.filterIsInstance<DxirOp>()
        val reshapesToRawShape = gradOps.count { it.op == OpKind.RESHAPE && it.type == gqaRawType }
        assertTrue(
            reshapesToRawShape >= 2,
            "expected ≥2 RESHAPEs producing gqaRawType (inner-inverse for K and V); got $reshapesToRawShape",
        )
    }

    @Test
    fun coarsensWithTwoOuterReshapes() {
        // §0.4.324 widening: outer chains up to 4 ops of {RESHAPE,
        // TRANSPOSE} are now in scope. Two outer RESHAPEs in a row
        // (a degenerate but well-formed chain) coarsens.
        val intermediate = DxirType(F32, listOf(hq, dim, tlen))   // arbitrary mid shape
        val fn = DxirBuilder.function("two_outer_reshapes") {
            val q = param("q", qType)
            val kRaw = param("k_raw", kRawType)
            val vRaw = param("v_raw", vRawType)
            val kBcast = op(
                OpKind.BROADCAST, listOf(kRaw), kExpandedType,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
            )
            val kReshape1 = op(OpKind.RESHAPE, listOf(kBcast), intermediate)
            val kReshape2 = op(OpKind.RESHAPE, listOf(kReshape1), kExpandedType)
            val vBcast = op(
                OpKind.BROADCAST, listOf(vRaw), vExpandedType,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2)),
            )
            val qk = op(OpKind.MATMUL, listOf(q, kReshape2), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, vBcast), outType)
            listOf(out)
        }
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(1, matches.size, "recognizer accepts longer outer chains")

        val coarsened = coarsenRecognizedPatterns(fn, matches)
        val coarsenedOps = coarsened.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
        assertEquals(1, coarsenedOps.size, "coarsener now handles two outer RESHAPEs")
    }

    @Test
    fun coarsensGqaWithTransposeInOuterChain() {
        // §0.4.324 — the Llama-3 / Mistral attention layout: K's outer
        // chain has the `K^T` TRANSPOSE before the QK matmul. Asymmetric
        // K vs V (V has just the canonical RESHAPE chain). Both fire.
        val kFlatType = DxirType(F32, listOf(hq, tlen, dim))             // post-flatten K
        val kTransposedType = DxirType(F32, listOf(hq, dim, tlen))       // K^T for Q · K^T

        val fn = DxirBuilder.function("gqa_with_kt") {
            val q = param("q", qType)
            // K side: GQA-canonical chain plus a final TRANSPOSE.
            val kRawT = param("k_raw", gqaRawType)
            val vRawT = param("v_raw", gqaRawType)
            val kUnsq = op(OpKind.RESHAPE, listOf(kRawT), gqaUnsqType)
            val kBcast = op(
                OpKind.BROADCAST, listOf(kUnsq), gqaBcastType,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2, 3)),
            )
            val kFlat = op(OpKind.RESHAPE, listOf(kBcast), kFlatType)
            val kT = op(
                OpKind.TRANSPOSE, listOf(kFlat), kTransposedType,
                attrs = mapOf("permutation" to listOf(0, 2, 1)),
            )
            // V side: standard GQA-canonical (no transpose).
            val vUnsq = op(OpKind.RESHAPE, listOf(vRawT), gqaUnsqType)
            val vBcast = op(
                OpKind.BROADCAST, listOf(vUnsq), gqaBcastType,
                attrs = mapOf("broadcast_dimensions" to listOf(0, 1, 2, 3)),
            )
            val vFlat = op(OpKind.RESHAPE, listOf(vBcast), vExpandedType)
            // Attention with K^T contracted: Q [hq, T, D] × K^T [hq, D, T] → [hq, T, T].
            val qk = op(OpKind.MATMUL, listOf(q, kT), scoreType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), scoreType)
            val out = op(OpKind.MATMUL, listOf(sm, vFlat), outType)
            listOf(out)
        }
        val matches = recognizeGroupedQueryAttention(fn)
        assertEquals(1, matches.size, "recognizer prerequisite")

        val coarsened = coarsenRecognizedPatterns(fn, matches)
        val ops = coarsened.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
        assertEquals(1, ops.size, "v3 coarsens TRANSPOSE-in-outer-chain shape")
        val co = ops.single()
        val grad = co.attrs["gradient_body"] as DxirFunction

        // dK_raw should land at the raw K shape ([hKv2, T, D]) despite
        // the TRANSPOSE in the outer chain. Pin via the gradient
        // body's return type.
        assertEquals(gqaRawType, grad.returns[1].type, "dK_raw at raw shape")
        assertEquals(gqaRawType, grad.returns[2].type, "dV_raw at raw shape")

        // Gradient body must contain a TRANSPOSE inverse on the K
        // side. The forward TRANSPOSE has perm=[0, 2, 1]; the inverse
        // TRANSPOSE in the gradient should also have perm [0, 2, 1]
        // (self-inverse for swapping last two axes). Pin via attrs.
        val gradOps = grad.body.filterIsInstance<DxirOp>()
        // Two TRANSPOSEs are the FlashAttention chain rule's last-two
        // swaps + one is the K-chain inverse + maybe more for V. Filter
        // to the one that produces the K-flat shape (output of inverse).
        val kFlatTransposes = gradOps.filter { it.op == OpKind.TRANSPOSE && it.type == kFlatType }
        assertTrue(
            kFlatTransposes.isNotEmpty(),
            "expected at least one TRANSPOSE producing kFlatType (the chain inverse); got ${gradOps.map { it.op }}",
        )
    }
}
