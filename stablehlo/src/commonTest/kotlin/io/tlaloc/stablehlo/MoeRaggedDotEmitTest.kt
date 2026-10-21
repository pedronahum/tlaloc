package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.core.I8
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.MoeExpertsAttrs
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.raggedDotMoe
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

class MoeRaggedDotEmitTest {

    private fun moe(rows: Int) = DxirBuilder.function("moe") {
        val x = param("x", DxirType(F32, listOf(rows, 64)))
        val l = param("logits", DxirType(F32, listOf(rows, 16)))
        val gu = param("gateUp", DxirType(F32, listOf(16, 64, 64)))
        val dn = param("down", DxirType(F32, listOf(16, 64, 32)))
        listOf(op(OpKind.MOE_EXPERTS, listOf(x, l, gu, dn), DxirType(F32, listOf(rows, 64)), mapOf("top_k" to 4)))
    }

    @Test
    fun theRaggedFormIsTwoRaggedDotsOverRowsPaddedTo256() {
        val text = raggedDotMoe(moe(300)).toStablehlo()
        assertEquals(2, Regex("\"chlo.ragged_dot\"").findAll(text).count(), text)
        // 300 rows x top 4 = 1200 pairs, padded to 1280.
        assertTrue("tensor<1280x64xf32>" in text, text)
        assertTrue("lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [2], lhs_ragged_dimensions = [0], rhs_group_dimensions = [0]" in text)
        assertTrue("#chlo<precision HIGHEST>" in text)
        assertTrue("stablehlo.while" !in text, "no tile loop in the ragged form")
    }

    @Test
    fun quantizedExpertsKeepTheirFormAndAnExplicitRaggedAttributeOnThemIsRefused() {
        val q = DxirBuilder.function("moe_q") {
            val x = param("x", DxirType(F32, listOf(4, 64)))
            val l = param("logits", DxirType(F32, listOf(4, 16)))
            val gu = param("gateUp", DxirType(I8, listOf(16, 64, 64)))
            val gus = param("gateUpScale", DxirType(F32, listOf(16, 64)))
            val dn = param("down", DxirType(I8, listOf(16, 64, 32)))
            val dns = param("downScale", DxirType(F32, listOf(16, 64)))
            listOf(op(OpKind.MOE_EXPERTS, listOf(x, l, gu, gus, dn, dns), DxirType(F32, listOf(4, 64)), mapOf("top_k" to 4)))
        }
        assertSame(q, raggedDotMoe(q))
        val op = q.body.filterIsInstance<DxirOp>().single { it.op == OpKind.MOE_EXPERTS }
        val forced = DxirOp(op.id, op.op, op.operands, op.attrs + (MoeExpertsAttrs.LOWERING to MoeExpertsAttrs.RAGGED_DOT), op.types)
        assertFailsWith<IllegalArgumentException> { MoeExpertsAttrs.parse(forced, "test") }
    }
}
