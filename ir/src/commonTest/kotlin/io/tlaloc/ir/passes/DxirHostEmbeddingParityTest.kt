package io.tlaloc.ir.passes

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Lit
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.hostI32
import io.tlaloc.core.ops.embedding
import io.tlaloc.core.ops.embeddingGrad
import io.tlaloc.core.ops.intZerosLike
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.400 — the `:core` host twins `embedding` / `embeddingGrad` are BIT-EXACT
 * against the dxir interpreter's EMBEDDING / EMBEDDING_GRAD arms — the
 * [DxirHostConvParityTest] discipline for the embedding front-end. The K2
 * plugin synthesises `grad {}` embedding gradients into calls on these twins,
 * and the interpreted dxir is the oracle those results are certified against,
 * so the parity must be exact (assertContentEquals on raw floats, no
 * tolerance).
 *
 * The gradient graph is produced by [DxirReverseTransform] on a real loss —
 * not hand-built — so the EMBEDDING_GRAD node under test carries exactly the
 * operand contract [VjpRegistry.EmbeddingRule] emits (indices, upstream,
 * tableTemplate — the §0.4.400 template-operand spelling). Collisions are in
 * every index vector on purpose: scatter-add summing is the semantics under
 * test.
 */
class DxirHostEmbeddingParityTest {

    private val scalar = DxirType(F32, emptyList())

    @Test
    fun hostEmbeddingMatchesInterpreterArmBitExact() {
        // idx = [0, 2, 0, 1] — slot 0 gathered twice.
        val idxData = floatArrayOf(0f, 2f, 0f, 1f)
        val fn = DxirBuilder.function("embed_fwd") {
            val table = param("table", DxirType(F32, listOf(3, 2)))
            val idx = const(idxData, DxirType(I32, listOf(4)))
            listOf(op(OpKind.EMBEDDING, listOf(table, idx), DxirType(F32, listOf(4, 2))))
        }
        val tableData = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val want = DxirInterpreter.evalFunction(fn, listOf(tableData)).single()

        val got = embedding(
            Tensors.f32Matrix<Sym, Sym>(3, 2, tableData),
            Tensors.i32Vector<Sym>(intArrayOf(0, 2, 0, 1)),
        )
        assertContentEquals(want, got.hostF32(), "host embedding diverges from the interpreter arm")
        assertContentEquals(listOf(4, 2), got.dims.toList())
    }

    @Test
    fun hostEmbeddingGradMatchesInterpreterArmBitExact() {
        // loss = Σ embedding(table, idx) ⊙ w with idx = [0, 2, 0, 1]: the reverse
        // transform emits EMBEDDING_GRAD(idx, w-clone, table-clone); replaying the
        // host twin with the same upstream must match the interpreted dTable
        // bit-for-bit, collisions (slot 0, twice) summing identically.
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
        // The rule's contract: exactly one EMBEDDING_GRAD, with the 3-operand
        // template spelling.
        val gradNode = grad.body.filterIsInstance<DxirOp>().single { it.op == OpKind.EMBEDDING_GRAD }
        assertEquals(3, gradNode.operands.size, "EmbeddingRule must emit the 3-operand template spelling")

        val tableData = floatArrayOf(0.3f, -1.2f, 2.1f, 0.7f, 1.6f, -0.4f)
        val wData = floatArrayOf(1.5f, -0.8f, 0.2f, -0.6f, 1.1f, 0.9f, -0.3f, 0.5f)
        val wantDTable = DxirInterpreter.evalFunction(grad, listOf(tableData, wData))[0]

        val got = embeddingGrad(
            upstream = DTensor<Rank2<Sym, Sym>, F32>(HostF32Storage(wData.copyOf()), intArrayOf(4, 2), F32),
            indices = Tensors.i32Vector<Sym>(intArrayOf(0, 2, 0, 1)),
            tableTemplate = Tensors.f32Matrix<Sym, Sym>(3, 2, tableData),
        )
        assertContentEquals(wantDTable, got.hostF32(), "host embeddingGrad diverges from the interpreter arm")

        // Sanity against the closed form dTable[v,:] = Σ_{p: idx[p]==v} w[p,:]:
        // v=0 ← positions 0 and 2 (the collision), v=1 ← position 3, v=2 ←
        // position 1.
        assertContentEquals(
            floatArrayOf(1.5f + 1.1f, -0.8f + 0.9f, -0.3f, 0.5f, 0.2f, -0.6f),
            got.hostF32(),
            "collision-summing dTable diverges from the analytic scatter-add",
        )
    }

    @Test
    fun hostEmbeddingGradZeroesUnselectedVocabRows() {
        // vocab = 5 but only slots {1, 3} are ever selected: rows 0, 2, 4 of the
        // gradient must be EXACT zeros (never touched by the scatter walk).
        val got = embeddingGrad(
            upstream = DTensor<Rank2<Sym, Sym>, F32>(
                HostF32Storage(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)),
                intArrayOf(3, 2),
                F32,
            ),
            indices = Tensors.i32Vector<Sym>(intArrayOf(3, 1, 3)),
            tableTemplate = Tensors.f32Zeros<Sym, Sym>(5, 2),
        )
        assertContentEquals(
            floatArrayOf(0f, 0f, 3f, 4f, 0f, 0f, 1f + 5f, 2f + 6f, 0f, 0f),
            got.hostF32(),
        )
    }

    @Test
    fun intZerosLikeShapesOffTheTemplate() {
        val z = intZerosLike(Tensors.i32Vector<Lit<Int>>(intArrayOf(7, 1, 3)))
        assertContentEquals(intArrayOf(0, 0, 0), z.hostI32())
        assertContentEquals(listOf(3), z.dims.toList())
        assertTrue(z.dtype == I32)
    }
}
