package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §0.4.370 — Phase A3b (DiffKT parity): the `embedding` VjpRule. EMBEDDING was
 * wired below the surface (emitter as gather + cost model) but had no reverse
 * rule; this pins the adjoint at IR level. `d/dTable embedding(table, idx)` is a
 * scatter-ADD of the upstream rows back to the vocab slots the indices selected
 * (collisions sum), expressed as the fused [OpKind.EMBEDDING_GRAD] op. The idx
 * operand is non-differentiable.
 */
class DxirEmbeddingGradTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun embeddingTableGradientScatterAddsUpstreamRows() {
        // loss = Σ embedding(table, idx) ⊙ w, with a repeated index (slot 0 is
        // selected twice) so the collision-summing behaviour is exercised.
        // table: [V=3, D=2]; idx = [0, 2, 0, 1] → emb: [4, 2]; w: [4, 2].
        val idxData = floatArrayOf(0f, 2f, 0f, 1f)
        val fn = DxirBuilder.function("embed_loss") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(4, 2)))
            val idx = const(idxData, DxirType(I32, listOf(4)))
            val emb = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 2)))
            val p = op(OpKind.MUL, listOf(emb, w), DxirType(F32, listOf(4, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val grad = DxirReverseTransform.apply(fn)
        val table = FloatArray(6) { it.toFloat() }
        val w = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val out = DxirInterpreter.evalFunction(grad, listOf(table, w))

        // dTable[v, :] = Σ_{p : idx[p]==v} w[p, :]:
        //   v=0 ← positions 0,2 → w[0,:]+w[2,:] = [1+5, 2+6] = [6, 8]
        //   v=1 ← position 3    → w[3,:]         = [7, 8]
        //   v=2 ← position 1    → w[1,:]         = [3, 4]
        val wantDTable = floatArrayOf(6f, 8f, 7f, 8f, 3f, 4f)
        for (i in wantDTable.indices) {
            assertTrue(
                abs(out[0][i] - wantDTable[i]) < 1e-6f,
                "dTable[$i] = ${out[0][i]}, want ${wantDTable[i]}",
            )
        }

        // dw = embedding(table, idx): the gathered rows.
        // idx=[0,2,0,1], table rows: [0,1],[2,3],[4,5] → emb = [0,1, 4,5, 0,1, 2,3].
        val wantDw = floatArrayOf(0f, 1f, 4f, 5f, 0f, 1f, 2f, 3f)
        for (i in wantDw.indices) {
            assertTrue(
                abs(out[1][i] - wantDw[i]) < 1e-6f,
                "dw[$i] = ${out[1][i]}, want ${wantDw[i]}",
            )
        }
    }

    @Test
    fun paddedEmbeddingZeroesPaddedRowsBothWays() {
        // §0.4.409 — `padding_index = 1`: positions whose index is 1 produce
        // exact-zero output rows AND scatter nothing back, while the collision
        // on slot 0 still sums. idx = [0, 1, 0, 2] over vocab 3.
        val idxData = floatArrayOf(0f, 1f, 0f, 2f)
        val fn = DxirBuilder.function("embed_pad") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(4, 2)))
            val idx = const(idxData, DxirType(I32, listOf(4)))
            val emb = op(
                OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 2)),
                attrs = mapOf("padding_index" to 1),
            )
            val p = op(OpKind.MUL, listOf(emb, w), DxirType(F32, listOf(4, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        // Forward: position 1's row must be EXACT zeros; others gather.
        val fwd = DxirBuilder.function("embed_pad_fwd") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val idx = const(idxData, DxirType(I32, listOf(4)))
            listOf(
                op(
                    OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 2)),
                    attrs = mapOf("padding_index" to 1),
                ),
            )
        }
        val table = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val embOut = DxirInterpreter.evalFunction(fwd, listOf(table)).single()
        assertTrue(embOut[2] == 0f && embOut[3] == 0f, "padded position must be exact zero: ${embOut.toList()}")
        assertTrue(embOut[0] == 0.3f && embOut[1] == -1.2f, "non-padded rows must still gather")

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(table, w))
        // dTable[0,:] = w[0,:] + w[2,:] (collision), dTable[1,:] = EXACT zero
        // (padded), dTable[2,:] = w[3,:].
        val wantDTable = floatArrayOf(1f + 5f, 2f + 6f, 0f, 0f, 7f, 8f)
        for (i in wantDTable.indices) {
            assertTrue(
                abs(grads[0][i] - wantDTable[i]) < 1e-6f,
                "dTable[$i] = ${grads[0][i]}, want ${wantDTable[i]}",
            )
        }
        assertTrue(grads[0][2] == 0f && grads[0][3] == 0f, "padded vocab row must be EXACT zero")
        // dw rows for padded positions must be exact zero too (the tangent-side
        // mask: dw = embedding(table, idx, pad)).
        assertTrue(grads[1][2] == 0f && grads[1][3] == 0f, "padded position's dw must be EXACT zero")
    }

    @Test
    fun paddedEmbeddingJvpVjpCrossIdentity() {
        // §0.4.409 — the cross-identity under the padding mask: the forward
        // tangent must mask padded positions exactly as the reverse does (the
        // attr rides the tangent EMBEDDING), or ⟨∇f, v⟩ ≠ df.
        val idxData = floatArrayOf(2f, 1f, 0f, 2f, 1f)
        val fn = DxirBuilder.function("embed_pad_cross") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(5, 2)))
            val idx = const(idxData, DxirType(I32, listOf(5)))
            val emb = op(
                OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(5, 2)),
                attrs = mapOf("padding_index" to 1),
            )
            val p = op(OpKind.MUL, listOf(emb, w), DxirType(F32, listOf(5, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val table = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 1.1f, 0.9f, -0.3f, 0.5f, 0.7f, -1.1f)
        val vTable = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.53f, 0.67f)
        val vW = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f, 0.19f, -0.07f, 0.23f, -0.05f, 0.41f, -0.37f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(table, w))
        var dot = 0.0
        for (i in table.indices) dot += grads[0][i].toDouble() * vTable[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vW[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(table, w, vTable, vW),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken under padding: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
        // And the padded vocab row's gradient is exactly zero.
        assertTrue(grads[0][2] == 0f && grads[0][3] == 0f, "padded vocab row must be EXACT zero")
    }

    @Test
    fun rankTwoIndexBatchJvpVjpCrossIdentity() {
        // §0.4.409 — [B=2, N=3] index batch → [2, 3, D] embedding, with a
        // collision ACROSS batch rows (slot 1 appears in both rows).
        val idxData = floatArrayOf(0f, 1f, 2f, 1f, 0f, 0f)
        val fn = DxirBuilder.function("embed_batch_cross") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(2, 3, 2)))
            val idx = const(idxData, DxirType(I32, listOf(2, 3)))
            val emb = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(2, 3, 2)))
            val p = op(OpKind.MUL, listOf(emb, w), DxirType(F32, listOf(2, 3, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val table = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = FloatArray(12) { 0.1f * (it + 1) * if (it % 2 == 0) 1 else -1 }
        val vTable = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.53f, 0.67f)
        val vW = FloatArray(12) { 0.05f * (12 - it) * if (it % 3 == 0) -1 else 1 }

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(table, w))
        // Analytic pin: dTable[v,:] = Σ_{p: idx[p]==v} w[p,:] over the FLAT walk —
        // slot 0 ← flat positions 0, 4, 5 (a cross-batch collision).
        val idxFlat = intArrayOf(0, 1, 2, 1, 0, 0)
        val wantDTable = FloatArray(6)
        for (p in idxFlat.indices) {
            wantDTable[idxFlat[p] * 2] += w[p * 2]
            wantDTable[idxFlat[p] * 2 + 1] += w[p * 2 + 1]
        }
        for (i in wantDTable.indices) {
            assertTrue(
                abs(grads[0][i] - wantDTable[i]) < 1e-6f,
                "batched dTable[$i] = ${grads[0][i]}, want ${wantDTable[i]}",
            )
        }

        var dot = 0.0
        for (i in table.indices) dot += grads[0][i].toDouble() * vTable[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vW[i]
        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(table, w, vTable, vW),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through the batched embedding: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }

    @Test
    fun embeddingJvpVjpCrossIdentity() {
        // f(table, w) = Σ embedding(table, idx) ⊙ w; ⟨∇f, v⟩ must equal the
        // forward tangent — certifies EmbeddingRule and the EMBEDDING forward
        // tangent against each other.
        val idxData = floatArrayOf(2f, 0f, 1f, 2f, 0f)
        val fn = DxirBuilder.function("embed_cross") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val w = param("w", DxirType(F32, listOf(5, 2)))
            val idx = const(idxData, DxirType(I32, listOf(5)))
            val emb = op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(5, 2)))
            val p = op(OpKind.MUL, listOf(emb, w), DxirType(F32, listOf(5, 2)))
            listOf(op(OpKind.SUM, listOf(p), scalar))
        }
        val table = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val w = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 1.1f, 0.9f, -0.3f, 0.5f, 0.7f, -1.1f)
        val vTable = floatArrayOf(0.11f, -0.23f, 0.37f, -0.41f, 0.53f, 0.67f)
        val vW = floatArrayOf(-0.29f, 0.31f, 0.13f, -0.17f, 0.19f, -0.07f, 0.23f, -0.05f, 0.41f, -0.37f)

        val grads = DxirInterpreter.evalFunction(DxirReverseTransform.apply(fn), listOf(table, w))
        var dot = 0.0
        for (i in table.indices) dot += grads[0][i].toDouble() * vTable[i]
        for (i in w.indices) dot += grads[1][i].toDouble() * vW[i]

        val jvp = DxirInterpreter.evalFunction(
            DxirForwardTransform.apply(fn), listOf(table, w, vTable, vW),
        )
        val tangent = jvp[1].single().toDouble()
        assertTrue(
            abs(dot - tangent) < 1e-5,
            "JVP⇄VJP cross-identity broken through embedding: ⟨grad,v⟩=$dot vs tangent=$tangent",
        )
    }
}
